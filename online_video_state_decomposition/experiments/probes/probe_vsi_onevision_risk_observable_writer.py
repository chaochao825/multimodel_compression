from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_variable_video_tokens,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_query_group_fallback_transfer import (
    compressed_top1_margin,
    summarize_raw,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import (
    contiguous_group_means,
    hybrid_group_tokens,
)
from probe_vsi_onevision_tiny_group_risk_controller import (
    CONTROLLER_WIDTH,
    MAX_EPOCHS,
    SKETCH_SEED,
    SKETCH_WIDTH,
    capture_states,
    controller_scores,
    fixed_sign_projection,
    question_suffix_embedding,
    selector_summary,
    train_controller,
    write_csv,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


MODE_WIDTH = 8
KEY_WIDTH = 32
WRITER_SCALAR_COUNT = 10


class WriterBackbone(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mode_encoder = nn.Linear(hidden, MODE_WIDTH, bias=False)
        self.mode_mixer = nn.Linear(5 * MODE_WIDTH, KEY_WIDTH)
        self.query_encoder = nn.Linear(hidden, KEY_WIDTH, bias=False)

    def forward(
        self,
        modes: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode_codes = self.mode_encoder(modes).flatten(start_dim=-2)
        writer_keys = torch.nn.functional.normalize(
            self.mode_mixer(mode_codes),
            dim=-1,
        )
        query_keys = torch.nn.functional.normalize(
            self.query_encoder(queries),
            dim=-1,
        )
        return writer_keys, query_keys


class WriterDotScorer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.backbone = WriterBackbone(hidden)
        self.scalar_head = nn.Linear(WRITER_SCALAR_COUNT, 1)

    def forward(
        self,
        modes: torch.Tensor,
        queries: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        writer_keys, query_keys = self.backbone(modes, queries)
        dot = torch.einsum("sgk,sk->sg", writer_keys, query_keys) * math.sqrt(
            KEY_WIDTH
        )
        return dot + self.scalar_head(scalars).squeeze(-1)


class JointWriterController(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.backbone = WriterBackbone(hidden)
        self.controller = nn.Sequential(
            nn.Linear(3 * KEY_WIDTH + WRITER_SCALAR_COUNT, CONTROLLER_WIDTH),
            nn.GELU(),
            nn.Linear(CONTROLLER_WIDTH, 1),
        )

    def forward(
        self,
        modes: torch.Tensor,
        queries: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        writer_keys, query_keys = self.backbone(modes, queries)
        repeated_query = query_keys[:, None, :].expand_as(writer_keys)
        features = torch.cat(
            (
                writer_keys,
                repeated_query,
                writer_keys * repeated_query,
                scalars,
            ),
            dim=-1,
        )
        return self.controller(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--prospective-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--refined-group-count", type=int, default=98)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def hadamard_modes(
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_count, group_size, hidden = exact_groups.shape
    if group_size != 4:
        raise ValueError("risk-observable writer requires group_size=4")
    if approximate_means.shape != (group_count, hidden):
        raise ValueError("approximate group means have the wrong shape")
    transform = exact_groups.new_tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=torch.float32,
    ).div_(2.0)
    delta = exact_groups.float() - approximate_means[:, None, :].float()
    residual_modes = torch.einsum("mt,gth->gmh", transform, delta)
    modes = torch.cat((approximate_means[:, None, :].float(), residual_modes), dim=1)
    rms = torch.linalg.vector_norm(modes, dim=-1).div(math.sqrt(hidden))
    normalized = modes / rms.clamp_min(1e-6)[:, :, None]
    return normalized, rms


def writer_position_features(
    group_count: int,
    *,
    groups_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(group_count, device=device)
    frame = (indices // groups_per_frame).float()
    within = (indices % groups_per_frame).float()
    frame_denominator = max(group_count // groups_per_frame - 1, 1)
    within_denominator = max(groups_per_frame - 1, 1)
    phase = 2.0 * torch.pi * within / max(groups_per_frame, 1)
    return torch.stack(
        (
            frame / frame_denominator,
            within / within_denominator,
            torch.sin(phase),
            torch.cos(phase),
        ),
        dim=1,
    )


def attach_writer_inputs(
    states: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    processor: object,
    model: nn.Module,
    model_dtype: torch.dtype,
    moments: object,
    basis: torch.Tensor,
) -> None:
    device = next(model.parameters()).device
    groups_per_frame = 196 // args.group_size
    for position, state in enumerate(states, start=1):
        sample = state["sample"]
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0],
            args.frame_budget,
        )
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = payload["features"].index_select(0, positions).to(
            device=device,
            dtype=model_dtype,
        )
        selected_frame_indices = [
            payload["pool_indices"][index] for index in selected_positions
        ]
        frames, _, _ = decode_video_frames(sample.video_path, selected_frame_indices)
        prompt_batch = build_prompt_batch(
            processor,
            sample,
            np.stack(frames),
            device=device,
            dtype=model_dtype,
        )
        query = question_suffix_embedding(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
        )
        approximate = reconstruct(reference, mean=moments.mean, basis=basis).to(
            model_dtype
        )
        exact_groups, _ = contiguous_group_means(
            reference,
            group_size=args.group_size,
        )
        _, approximate_means = contiguous_group_means(
            approximate,
            group_size=args.group_size,
        )
        modes, mode_rms = hadamard_modes(exact_groups, approximate_means)
        query_score = (
            torch.nn.functional.normalize(approximate_means.float(), dim=1)
            @ query.float()
        )
        position_features = writer_position_features(
            exact_groups.shape[0],
            groups_per_frame=groups_per_frame,
            device=device,
        )
        scalars = torch.cat(
            (
                torch.log1p(mode_rms),
                query_score[:, None],
                position_features,
            ),
            dim=1,
        )
        if scalars.shape[1] != WRITER_SCALAR_COUNT:
            raise ValueError("writer scalar contract mismatch")
        state["writer_modes"] = modes.detach().to(dtype=torch.float16).cpu()
        state["writer_scalars"] = scalars.detach().cpu()
        state["query_vector"] = query.detach().cpu()
        print(
            json.dumps(
                {
                    "event": "risk_observable_writer_input_ok",
                    "role": state["role"],
                    "position": position,
                    "total": len(states),
                    "sample_id": state["sample_id"],
                }
            ),
            flush=True,
        )


def materialize_writer_batch(
    states: list[dict[str, object]],
    *,
    device: torch.device,
    scalar_mean: torch.Tensor | None = None,
    scalar_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modes = torch.stack([state["writer_modes"] for state in states]).to(
        device=device,
        dtype=torch.float32,
    )
    queries = torch.stack([state["query_vector"] for state in states]).to(
        device=device,
        dtype=torch.float32,
    )
    scalars = torch.stack([state["writer_scalars"] for state in states]).to(
        device=device,
        dtype=torch.float32,
    )
    labels = torch.stack([state["teacher_labels"] for state in states]).to(
        device=device,
        dtype=torch.float32,
    )
    if scalar_mean is not None and scalar_std is not None:
        scalars = (scalars - scalar_mean) / scalar_std
    return modes, queries, scalars, labels


def split_score_tensor(scores: torch.Tensor) -> list[torch.Tensor]:
    return [row.detach().cpu() for row in scores]


def score_writer_model(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[torch.Tensor]:
    modes, queries, scalars, _ = batch
    model.eval()
    with torch.inference_mode():
        return split_score_tensor(model(modes, queries, scalars))


def train_writer_model(
    model: nn.Module,
    *,
    train_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    validation_states: list[dict[str, object]],
    topk: int,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    train_modes, train_queries, train_scalars, train_labels = train_batch
    validation_modes, validation_queries, validation_scalars, _ = validation_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = train_labels.sum()
    negative = train_labels.numel() - positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=(negative / positive).detach())
    best_recall = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_modes, train_queries, train_scalars)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_scores = split_score_tensor(
                model(validation_modes, validation_queries, validation_scalars)
            )
        validation = selector_summary(
            validation_states,
            validation_scores,
            topk=topk,
        )
        recall = float(validation["mean_topk_recall"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "validation_topk_recall": recall,
                "validation_risk_mass_capture": float(
                    validation["mean_risk_mass_capture"]
                ),
            }
        )
        if recall > best_recall:
            best_recall = recall
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def evaluate_writer_reader(
    *,
    states: list[dict[str, object]],
    selector_scores: dict[str, list[torch.Tensor]],
    args: argparse.Namespace,
    processor: object,
    model: nn.Module,
    model_dtype: torch.dtype,
    moments: object,
    basis: torch.Tensor,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    rows = []
    for position, state in enumerate(states, start=1):
        sample = state["sample"]
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0],
            args.frame_budget,
        )
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = payload["features"].index_select(0, positions).to(
            device=device,
            dtype=model_dtype,
        )
        selected_frame_indices = [
            payload["pool_indices"][index] for index in selected_positions
        ]
        frames, _, _ = decode_video_frames(sample.video_path, selected_frame_indices)
        prompt_batch = build_prompt_batch(
            processor,
            sample,
            np.stack(frames),
            device=device,
            dtype=model_dtype,
        )
        approximate = reconstruct(reference, mean=moments.mean, basis=basis).to(
            model_dtype
        )
        exact_groups, _ = contiguous_group_means(
            reference,
            group_size=args.group_size,
        )
        _, approximate_means = contiguous_group_means(
            approximate,
            group_size=args.group_size,
        )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidates = state["reference_candidates"].to(device)
        with torch.inference_mode():
            for method, method_scores in selector_scores.items():
                selected = torch.topk(
                    method_scores[position - 1],
                    k=args.refined_group_count,
                ).indices.sort().values.to(device)
                logits = first_token_logits_from_variable_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=hybrid_group_tokens(
                        exact_groups,
                        approximate_means,
                        selected,
                    ),
                )
                candidates = logits.float().index_select(0, token_tensor)
                approximate_index = int(torch.argmax(candidates).item())
                rows.append(
                    {
                        "sample_id": state["sample_id"],
                        "role": state["role"],
                        "method": method,
                        "teacher_index": state["teacher_index"],
                        "answer_index": state["answer_index"],
                        "approximate_index": approximate_index,
                        "compressed_top1_margin": compressed_top1_margin(candidates),
                        "candidate_kl": candidate_kl(
                            reference_candidates,
                            candidates,
                        ),
                        "prediction_match": int(
                            approximate_index == state["teacher_index"]
                        ),
                        "baseline_correct": int(
                            state["teacher_index"] == state["answer_index"]
                        ),
                        "approximate_correct": int(
                            approximate_index == state["answer_index"]
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "event": "risk_observable_writer_reader_ok",
                    "position": position,
                    "total": len(states),
                    "sample_id": state["sample_id"],
                }
            ),
            flush=True,
        )
    return rows


def gate_conditions(
    *,
    selector: dict[str, dict[str, float | int]],
    reader: dict[str, dict[str, float | int]],
    scorer_macs: int,
) -> tuple[str, dict[str, bool]]:
    joint_recall = float(selector["joint_writer_controller"]["mean_topk_recall"])
    writer_recall = float(selector["writer_dot"]["mean_topk_recall"])
    fixed_recall = float(selector["fixed_controller"]["mean_topk_recall"])
    joint_reader = reader["joint_writer_controller"]
    writer_reader = reader["writer_dot"]
    joint_conditions = {
        "joint_recall_at_least_45pct": joint_recall >= 0.45,
        "joint_risk_mass_at_least_50pct": float(
            selector["joint_writer_controller"]["mean_risk_mass_capture"]
        )
        >= 0.50,
        "joint_recall_three_points_above_fixed": joint_recall
        >= fixed_recall + 0.03,
        "joint_recall_three_points_above_writer": joint_recall
        >= writer_recall + 0.03,
        "joint_agreement_at_least_22_of_24": float(joint_reader["agreement"])
        >= 22 / 24,
        "joint_harmful_at_most_one": int(joint_reader["harmful_count"]) <= 1,
        "joint_kl_at_most_005": float(joint_reader["candidate_kl_mean"]) <= 0.05,
        "joint_no_accuracy_loss": float(joint_reader["candidate_accuracy"])
        >= float(joint_reader["baseline_accuracy"]),
        "joint_scorer_below_two_million_macs": scorer_macs < 2_000_000,
    }
    writer_conditions = {
        "writer_recall_at_least_45pct": writer_recall >= 0.45,
        "writer_risk_mass_at_least_50pct": float(
            selector["writer_dot"]["mean_risk_mass_capture"]
        )
        >= 0.50,
        "writer_agreement_at_least_22_of_24": float(writer_reader["agreement"])
        >= 22 / 24,
        "writer_harmful_at_most_one": int(writer_reader["harmful_count"]) <= 1,
        "writer_kl_at_most_005": float(writer_reader["candidate_kl_mean"]) <= 0.05,
        "writer_no_accuracy_loss": float(writer_reader["candidate_accuracy"])
        >= float(writer_reader["baseline_accuracy"]),
    }
    conditions = {**joint_conditions, **writer_conditions}
    if all(joint_conditions.values()):
        return "JOINT_GO", conditions
    if all(writer_conditions.values()):
        return "WRITER_ONLY_GO", conditions
    return "NO_GO", conditions


def main() -> int:
    args = parse_args()
    if args.group_size != 4:
        raise ValueError("registered writer Gate requires group_size=4")
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    total_count = args.train_count + args.validation_count + args.prospective_count
    if total_count > 96:
        raise ValueError("writer Gate cannot read calibration positions after 96")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=total_count,
    )
    train_samples = samples[: args.train_count]
    validation_stop = args.train_count + args.validation_count
    validation_samples = samples[args.train_count : validation_stop]
    prospective_samples = samples[validation_stop:]
    expected_calibration_ids = {
        str(scene["sample_id"]) for scene in split["roles"]["calibration"]
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    moments = calibration_moments(
        args.feature_dir,
        expected_sample_ids=expected_calibration_ids,
        device=device,
    )
    _, basis = descending_eigenspace(moments.covariance, rank=args.rank)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    model_dtype = next(model.parameters()).dtype
    hidden = int(moments.mean.numel())
    hidden_projection = fixed_sign_projection(
        (hidden, SKETCH_WIDTH),
        seed=SKETCH_SEED,
    ).to(device)
    residual_projection = fixed_sign_projection(
        (args.group_size, hidden, SKETCH_WIDTH),
        seed=SKETCH_SEED + 1,
    ).to(device)
    started = time.perf_counter()

    train_states = capture_states(
        samples=train_samples,
        role="train",
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
        hidden_projection=hidden_projection,
        residual_projection=residual_projection,
    )
    validation_states = capture_states(
        samples=validation_samples,
        role="validation",
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
        hidden_projection=hidden_projection,
        residual_projection=residual_projection,
    )
    attach_writer_inputs(
        train_states,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )
    attach_writer_inputs(
        validation_states,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )

    fixed_controller, feature_mean, feature_std, fixed_history = train_controller(
        train_states,
        validation_states,
        topk=args.refined_group_count,
        device=device,
    )
    validation_fixed_scores = controller_scores(
        fixed_controller,
        validation_states,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )

    train_batch_unscaled = materialize_writer_batch(train_states, device=device)
    scalar_mean = train_batch_unscaled[2].mean(dim=(0, 1), keepdim=True)
    scalar_std = train_batch_unscaled[2].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    train_batch = (
        train_batch_unscaled[0],
        train_batch_unscaled[1],
        (train_batch_unscaled[2] - scalar_mean) / scalar_std,
        train_batch_unscaled[3],
    )
    validation_batch = materialize_writer_batch(
        validation_states,
        device=device,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
    )
    torch.manual_seed(SKETCH_SEED)
    writer_dot, writer_history = train_writer_model(
        WriterDotScorer(hidden).to(device),
        train_batch=train_batch,
        validation_batch=validation_batch,
        validation_states=validation_states,
        topk=args.refined_group_count,
    )
    torch.manual_seed(SKETCH_SEED)
    joint_writer, joint_history = train_writer_model(
        JointWriterController(hidden).to(device),
        train_batch=train_batch,
        validation_batch=validation_batch,
        validation_states=validation_states,
        topk=args.refined_group_count,
    )
    validation_writer_scores = score_writer_model(writer_dot, validation_batch)
    validation_joint_scores = score_writer_model(joint_writer, validation_batch)

    del train_batch_unscaled, train_batch, validation_batch
    torch.cuda.empty_cache()

    prospective_states = capture_states(
        samples=prospective_samples,
        role="prospective",
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
        hidden_projection=hidden_projection,
        residual_projection=residual_projection,
    )
    attach_writer_inputs(
        prospective_states,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )
    prospective_fixed_scores = controller_scores(
        fixed_controller,
        prospective_states,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )
    prospective_batch = materialize_writer_batch(
        prospective_states,
        device=device,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
    )
    prospective_writer_scores = score_writer_model(writer_dot, prospective_batch)
    prospective_joint_scores = score_writer_model(joint_writer, prospective_batch)

    validation_scores = {
        "residual_energy": [state["residual_scores"] for state in validation_states],
        "query_cosine": [state["query_scores"] for state in validation_states],
        "fixed_controller": validation_fixed_scores,
        "writer_dot": validation_writer_scores,
        "joint_writer_controller": validation_joint_scores,
        "target_gradient_risk": [
            state["teacher_risk"] for state in validation_states
        ],
    }
    prospective_scores = {
        "residual_energy": [state["residual_scores"] for state in prospective_states],
        "query_cosine": [state["query_scores"] for state in prospective_states],
        "fixed_controller": prospective_fixed_scores,
        "writer_dot": prospective_writer_scores,
        "joint_writer_controller": prospective_joint_scores,
        "target_gradient_risk": [
            state["teacher_risk"] for state in prospective_states
        ],
    }
    validation_selector = {
        method: selector_summary(
            validation_states,
            scores,
            topk=args.refined_group_count,
        )
        for method, scores in validation_scores.items()
    }
    prospective_selector = {
        method: selector_summary(
            prospective_states,
            scores,
            topk=args.refined_group_count,
        )
        for method, scores in prospective_scores.items()
    }
    reader_rows = evaluate_writer_reader(
        states=prospective_states,
        selector_scores=prospective_scores,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )
    methods = tuple(prospective_scores)
    reader_summary = {
        method: summarize_raw([row for row in reader_rows if row["method"] == method])
        for method in methods
    }

    group_count = args.frame_budget * 196 // args.group_size
    writer_macs = group_count * (
        5 * hidden * MODE_WIDTH + 5 * MODE_WIDTH * KEY_WIDTH
    )
    writer_key_bytes_fp16 = group_count * KEY_WIDTH * 2
    writer_dot_macs = hidden * KEY_WIDTH + group_count * (
        KEY_WIDTH + WRITER_SCALAR_COUNT
    )
    joint_scorer_macs = hidden * KEY_WIDTH + group_count * (
        (3 * KEY_WIDTH + WRITER_SCALAR_COUNT) * CONTROLLER_WIDTH
        + CONTROLLER_WIDTH
    )
    decision, conditions = gate_conditions(
        selector=prospective_selector,
        reader=reader_summary,
        scorer_macs=joint_scorer_macs,
    )
    history_rows = []
    for method, history in (
        ("fixed_controller", fixed_history),
        ("writer_dot", writer_history),
        ("joint_writer_controller", joint_history),
    ):
        history_rows.extend({"method": method, **row} for row in history)
    best_epochs = {
        method: int(max(history, key=lambda row: row["validation_topk_recall"])["epoch"])
        for method, history in (
            ("fixed_controller", fixed_history),
            ("writer_dot", writer_history),
            ("joint_writer_controller", joint_history),
        )
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "risk_observable_writer_prospective_calibration_gate",
        "counts": {
            "train": len(train_states),
            "validation": len(validation_states),
            "prospective": len(prospective_states),
            "reserved_calibration": len(split["roles"]["calibration"])
            - total_count,
        },
        "rank": args.rank,
        "group_size": args.group_size,
        "refined_group_count": args.refined_group_count,
        "writer": {
            "mode_width": MODE_WIDTH,
            "key_width": KEY_WIDTH,
            "stored_key_bytes_fp16": writer_key_bytes_fp16,
            "one_time_writer_macs": writer_macs,
            "writer_dot_macs_per_question": writer_dot_macs,
            "joint_scorer_macs_per_question": joint_scorer_macs,
        },
        "best_epochs": best_epochs,
        "validation_selector_diagnostics": validation_selector,
        "prospective_selector_diagnostics": prospective_selector,
        "prospective_reader": reader_summary,
        "prospective_gate": {
            "decision": decision,
            "conditions": conditions,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration prospective observability result only. Positions 97-120, "
            "selection, and formal remain unread. Writer bytes and MACs are proxies, "
            "not measured latency."
        ),
    }
    torch.save(
        {
            "fixed_controller": fixed_controller.state_dict(),
            "fixed_feature_mean": feature_mean.cpu(),
            "fixed_feature_std": feature_std.cpu(),
            "writer_dot": writer_dot.state_dict(),
            "joint_writer_controller": joint_writer.state_dict(),
            "writer_scalar_mean": scalar_mean.cpu(),
            "writer_scalar_std": scalar_std.cpu(),
        },
        args.out_dir / "writer_models.pt",
    )
    write_csv(args.out_dir / "training_history.csv", history_rows)
    write_csv(args.out_dir / "prospective_reader_metrics.csv", reader_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
