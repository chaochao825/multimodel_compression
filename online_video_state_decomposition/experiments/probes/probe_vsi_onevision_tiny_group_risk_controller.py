from __future__ import annotations

import argparse
import copy
import csv
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
    first_token_logits_from_features,
    first_token_logits_from_variable_video_tokens,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_query_group_fallback_transfer import (
    apply_fallback,
    calibrate_mismatch_threshold,
    compressed_top1_margin,
    summarize_raw,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    candidate_margins,
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import (
    contiguous_group_means,
    hybrid_group_tokens,
    normalized_adverse_group_risk,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


SKETCH_SEED = 20260830
SKETCH_WIDTH = 8
CONTROLLER_WIDTH = 32
MAX_EPOCHS = 100


class TinyGroupRiskController(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, CONTROLLER_WIDTH),
            nn.GELU(),
            nn.Linear(CONTROLLER_WIDTH, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--prospective-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--refined-group-count", type=int, default=98)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--max-fallback-rate", type=float, default=0.15)
    parser.add_argument("--max-token-retention", type=float, default=0.53)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def fixed_sign_projection(
    shape: tuple[int, ...],
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    return values.float().mul_(2.0).sub_(1.0).div_(math.sqrt(math.prod(shape[:-1])))


def question_suffix_embedding(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    video_positions = torch.nonzero(
        input_ids[0] == model.config.video_token_index,
        as_tuple=False,
    ).flatten()
    if video_positions.numel() == 0:
        raise ValueError("prompt contains no video placeholder")
    suffix_start = int(video_positions[-1].item()) + 1
    suffix_mask = attention_mask[0, suffix_start:].bool()
    suffix_ids = input_ids[0, suffix_start:][suffix_mask]
    if suffix_ids.numel() == 0:
        raise ValueError("prompt has no post-video question tokens")
    query = model.get_input_embeddings()(suffix_ids).float().mean(dim=0)
    return torch.nn.functional.normalize(query.detach(), dim=0)


def controller_group_features(
    *,
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
    query: torch.Tensor,
    hidden_projection: torch.Tensor,
    residual_projection: torch.Tensor,
    groups_per_frame: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    group_count, group_size, hidden = exact_groups.shape
    if approximate_means.shape != (group_count, hidden):
        raise ValueError("approximate group means have the wrong shape")
    if hidden_projection.shape != (hidden, SKETCH_WIDTH):
        raise ValueError("hidden projection has the wrong shape")
    if residual_projection.shape != (group_size, hidden, SKETCH_WIDTH):
        raise ValueError("residual projection has the wrong shape")
    delta = exact_groups.float() - approximate_means[:, None, :].float()
    residual_energy = torch.linalg.vector_norm(delta, dim=(1, 2)).div(
        math.sqrt(group_size * hidden)
    )
    quotient_rms = torch.linalg.vector_norm(approximate_means.float(), dim=1).div(
        math.sqrt(hidden)
    )
    normalized_groups = torch.nn.functional.normalize(
        approximate_means.float(),
        dim=1,
    )
    query_score = normalized_groups @ query.float()
    residual_sketch = torch.einsum(
        "gth,thk->gk",
        delta,
        residual_projection.float(),
    )
    quotient_sketch = approximate_means.float() @ hidden_projection.float()
    query_sketch = query.float() @ hidden_projection.float()
    repeated_query = query_sketch[None, :].expand(group_count, -1)
    interaction = quotient_sketch * repeated_query
    indices = torch.arange(group_count, device=exact_groups.device)
    frame = (indices // groups_per_frame).float()
    within = (indices % groups_per_frame).float()
    frame_denominator = max(group_count // groups_per_frame - 1, 1)
    within_denominator = max(groups_per_frame - 1, 1)
    phase = 2.0 * torch.pi * within / max(groups_per_frame, 1)
    position = torch.stack(
        (
            frame / frame_denominator,
            within / within_denominator,
            torch.sin(phase),
            torch.cos(phase),
        ),
        dim=1,
    )
    features = torch.cat(
        (
            query_score[:, None],
            torch.log1p(residual_energy)[:, None],
            torch.log1p(quotient_rms)[:, None],
            residual_sketch,
            quotient_sketch,
            repeated_query,
            interaction,
            position,
        ),
        dim=1,
    )
    diagnostics = {
        "residual_energy": residual_energy,
        "query_score": query_score,
    }
    return features, diagnostics


def teacher_topk_labels(risk: torch.Tensor, *, topk: int) -> torch.Tensor:
    labels = torch.zeros_like(risk)
    labels[torch.topk(risk, k=topk).indices] = 1.0
    return labels


def selector_summary(
    states: list[dict[str, object]],
    scores: list[torch.Tensor],
    *,
    topk: int,
) -> dict[str, float | int]:
    recalls = []
    risk_mass = []
    for state, sample_scores in zip(states, scores, strict=True):
        labels = state["teacher_labels"]
        risk = state["teacher_risk"]
        selected = torch.topk(sample_scores, k=topk).indices
        recalls.append(float(labels[selected].sum().item() / topk))
        risk_mass.append(
            float(
                (
                    risk[selected].sum()
                    / risk.sum().clamp_min(torch.finfo(torch.float32).eps)
                ).item()
            )
        )
    return {
        "sample_count": len(states),
        "mean_topk_recall": float(np.mean(recalls)),
        "minimum_topk_recall": float(np.min(recalls)),
        "mean_risk_mass_capture": float(np.mean(risk_mass)),
    }


def controller_scores(
    controller: TinyGroupRiskController,
    states: list[dict[str, object]],
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    controller.eval()
    values = []
    with torch.inference_mode():
        for state in states:
            features = state["controller_features"].to(device)
            standardized = (features - feature_mean) / feature_std
            values.append(controller(standardized).cpu())
    return values


def train_controller(
    train_states: list[dict[str, object]],
    validation_states: list[dict[str, object]],
    *,
    topk: int,
    device: torch.device,
) -> tuple[
    TinyGroupRiskController,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, float | int]],
]:
    train_features = torch.cat(
        [state["controller_features"] for state in train_states]
    ).to(device)
    train_labels = torch.cat(
        [state["teacher_labels"] for state in train_states]
    ).to(device)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)
    standardized_train = (train_features - feature_mean) / feature_std
    torch.manual_seed(SKETCH_SEED)
    controller = TinyGroupRiskController(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    positive = train_labels.sum()
    negative = train_labels.numel() - positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=(negative / positive).detach())
    history = []
    best_recall = -1.0
    best_state = copy.deepcopy(controller.state_dict())
    for epoch in range(1, MAX_EPOCHS + 1):
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        logits = controller(standardized_train)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        validation_scores = controller_scores(
            controller,
            validation_states,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
        )
        validation = selector_summary(
            validation_states,
            validation_scores,
            topk=topk,
        )
        validation_recall = float(validation["mean_topk_recall"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "validation_topk_recall": validation_recall,
                "validation_risk_mass_capture": float(
                    validation["mean_risk_mass_capture"]
                ),
            }
        )
        if validation_recall > best_recall:
            best_recall = validation_recall
            best_state = copy.deepcopy(controller.state_dict())
    controller.load_state_dict(best_state)
    return controller, feature_mean, feature_std, history


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_states(
    *,
    samples: list[object],
    role: str,
    args: argparse.Namespace,
    processor: object,
    model: nn.Module,
    model_dtype: torch.dtype,
    moments: object,
    basis: torch.Tensor,
    hidden_projection: torch.Tensor,
    residual_projection: torch.Tensor,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    states = []
    groups_per_frame = 196 // args.group_size
    for position, sample in enumerate(samples, start=1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
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
        probe = reference.detach().float().requires_grad_(True)
        reference_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=probe,
        )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidates = reference_logits.detach().float().index_select(
            0, token_tensor
        )
        teacher_index = int(torch.argmax(reference_candidates).item())
        competitor_indices = [
            index for index in range(len(token_ids)) if index != teacher_index
        ]
        margins = candidate_margins(
            reference_logits,
            token_ids,
            teacher_index=teacher_index,
            competitor_indices=competitor_indices,
        )
        gradients = []
        for competitor_position, margin in enumerate(margins):
            gradient = torch.autograd.grad(
                margin,
                probe,
                retain_graph=competitor_position + 1 < len(margins),
            )[0]
            gradients.append(gradient.detach().reshape(-1, gradient.shape[-1]))
        gradient_tensor = torch.stack(gradients).float()
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
        query = question_suffix_embedding(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
        )
        features, diagnostics = controller_group_features(
            exact_groups=exact_groups,
            approximate_means=approximate_means,
            query=query,
            hidden_projection=hidden_projection,
            residual_projection=residual_projection,
            groups_per_frame=groups_per_frame,
        )
        risk = normalized_adverse_group_risk(
            gradient_tensor,
            exact_groups,
            approximate_means,
            margins.detach(),
            margin_floor=args.margin_floor,
        )
        states.append(
            {
                "sample": sample,
                "sample_id": sample.sample_id,
                "role": role,
                "controller_features": features.detach().cpu(),
                "teacher_risk": risk.detach().cpu(),
                "teacher_labels": teacher_topk_labels(
                    risk.detach().cpu(),
                    topk=args.refined_group_count,
                ),
                "query_scores": diagnostics["query_score"].detach().cpu(),
                "residual_scores": diagnostics["residual_energy"].detach().cpu(),
                "reference_candidates": reference_candidates.cpu(),
                "teacher_index": teacher_index,
                "answer_index": sample.answer_index,
            }
        )
        print(
            json.dumps(
                {
                    "event": "tiny_group_risk_capture_ok",
                    "role": role,
                    "position": position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )
    return states


def evaluate_reader(
    *,
    states: list[dict[str, object]],
    controller_predictions: list[torch.Tensor],
    args: argparse.Namespace,
    processor: object,
    model: nn.Module,
    model_dtype: torch.dtype,
    moments: object,
    basis: torch.Tensor,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    rows = []
    for position, (state, learned_scores) in enumerate(
        zip(states, controller_predictions, strict=True),
        start=1,
    ):
        sample = state["sample"]
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
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
        selector_scores = {
            "residual_energy": state["residual_scores"],
            "query_cosine": state["query_scores"],
            "tiny_controller": learned_scores,
        }
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidates = state["reference_candidates"].to(device)
        with torch.inference_mode():
            for method, scores in selector_scores.items():
                selected = torch.topk(
                    scores,
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
                    "event": "tiny_group_risk_reader_ok",
                    "role": state["role"],
                    "position": position,
                    "total": len(states),
                    "sample_id": state["sample_id"],
                }
            ),
            flush=True,
        )
    return rows


def serializable_states(states: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": state["sample_id"],
            "role": state["role"],
            "controller_features": state["controller_features"],
            "teacher_risk": state["teacher_risk"],
            "teacher_labels": state["teacher_labels"],
            "query_scores": state["query_scores"],
            "residual_scores": state["residual_scores"],
            "reference_candidates": state["reference_candidates"],
            "teacher_index": state["teacher_index"],
            "answer_index": state["answer_index"],
        }
        for state in states
    ]


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    total_count = args.train_count + args.validation_count + args.prospective_count
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
    controller, feature_mean, feature_std, history = train_controller(
        train_states,
        validation_states,
        topk=args.refined_group_count,
        device=device,
    )
    validation_controller_scores = controller_scores(
        controller,
        validation_states,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )
    validation_reader_rows = evaluate_reader(
        states=validation_states,
        controller_predictions=validation_controller_scores,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )
    validation_controller_rows = [
        row for row in validation_reader_rows if row["method"] == "tiny_controller"
    ]
    fallback_threshold = calibrate_mismatch_threshold(validation_controller_rows)

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
    prospective_controller_scores = controller_scores(
        controller,
        prospective_states,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )
    prospective_reader_rows = evaluate_reader(
        states=prospective_states,
        controller_predictions=prospective_controller_scores,
        args=args,
        processor=processor,
        model=model,
        model_dtype=model_dtype,
        moments=moments,
        basis=basis,
    )
    prospective_controller_rows = [
        row for row in prospective_reader_rows if row["method"] == "tiny_controller"
    ]
    full_token_count = args.frame_budget * 196
    base_group_count = full_token_count // args.group_size
    hybrid_token_count = base_group_count + args.refined_group_count * (
        args.group_size - 1
    )
    delivered_rows, progressive = apply_fallback(
        prospective_controller_rows,
        threshold=fallback_threshold,
        hybrid_token_count=hybrid_token_count,
        full_token_count=full_token_count,
    )
    validation_selectors = {
        "residual_energy": selector_summary(
            validation_states,
            [state["residual_scores"] for state in validation_states],
            topk=args.refined_group_count,
        ),
        "query_cosine": selector_summary(
            validation_states,
            [state["query_scores"] for state in validation_states],
            topk=args.refined_group_count,
        ),
        "tiny_controller": selector_summary(
            validation_states,
            validation_controller_scores,
            topk=args.refined_group_count,
        ),
    }
    prospective_selectors = {
        "residual_energy": selector_summary(
            prospective_states,
            [state["residual_scores"] for state in prospective_states],
            topk=args.refined_group_count,
        ),
        "query_cosine": selector_summary(
            prospective_states,
            [state["query_scores"] for state in prospective_states],
            topk=args.refined_group_count,
        ),
        "tiny_controller": selector_summary(
            prospective_states,
            prospective_controller_scores,
            topk=args.refined_group_count,
        ),
    }
    input_dim = int(train_states[0]["controller_features"].shape[1])
    controller_macs = base_group_count * (
        input_dim * CONTROLLER_WIDTH + CONTROLLER_WIDTH
    )
    conditions = {
        "delivered_agreement_at_least_98pct": float(
            progressive["delivered_agreement"]
        )
        >= 0.98,
        "remaining_harmful_zero": int(progressive["remaining_harmful_count"]) == 0,
        "fallback_rate_within_budget": float(progressive["fallback_rate"])
        <= args.max_fallback_rate,
        "effective_token_retention_within_budget": float(
            progressive["effective_token_retention"]
        )
        <= args.max_token_retention,
        "task_accuracy_loss_at_most_one_point": float(
            progressive["delivered_accuracy"]
        )
        >= float(progressive["baseline_accuracy"]) - 0.01,
        "controller_macs_below_one_million": controller_macs < 1_000_000,
        "controller_recall_exceeds_residual": float(
            prospective_selectors["tiny_controller"]["mean_topk_recall"]
        )
        > float(prospective_selectors["residual_energy"]["mean_topk_recall"]),
        "controller_recall_exceeds_query": float(
            prospective_selectors["tiny_controller"]["mean_topk_recall"]
        )
        > float(prospective_selectors["query_cosine"]["mean_topk_recall"]),
    }
    best_record = max(history, key=lambda row: row["validation_topk_recall"])
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "tiny_group_risk_controller_prospective_calibration_gate",
        "counts": {
            "train": len(train_states),
            "validation": len(validation_states),
            "prospective": len(prospective_states),
        },
        "rank": args.rank,
        "group_size": args.group_size,
        "refined_group_count": args.refined_group_count,
        "input_dim": input_dim,
        "controller_width": CONTROLLER_WIDTH,
        "best_epoch": int(best_record["epoch"]),
        "fallback_threshold": fallback_threshold,
        "controller_macs": controller_macs,
        "residual_metadata_scalars_per_group": 1 + SKETCH_WIDTH,
        "validation_selector_diagnostics": validation_selectors,
        "prospective_selector_diagnostics": prospective_selectors,
        "validation_reader": {
            method: summarize_raw(
                [row for row in validation_reader_rows if row["method"] == method]
            )
            for method in ("residual_energy", "query_cosine", "tiny_controller")
        },
        "prospective_reader": {
            method: summarize_raw(
                [row for row in prospective_reader_rows if row["method"] == method]
            )
            for method in ("residual_energy", "query_cosine", "tiny_controller")
        },
        "prospective_progressive": progressive,
        "prospective_gate": {
            "decision": "GO" if all(conditions.values()) else "NO_GO",
            "conditions": conditions,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration teacher-distillation result only. The prospective role is "
            "the third 24-question calibration slice; selection and formal roles are "
            "unread. Controller MACs and token counts are proxies, not measured latency."
        ),
    }
    all_states = train_states + validation_states + prospective_states
    torch.save(
        {
            "states": serializable_states(all_states),
            "hidden_projection": hidden_projection.cpu(),
            "residual_projection": residual_projection.cpu(),
        },
        args.out_dir / "risk_dataset.pt",
    )
    torch.save(
        {
            "state_dict": controller.state_dict(),
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "input_dim": input_dim,
            "width": CONTROLLER_WIDTH,
            "fallback_threshold": fallback_threshold,
        },
        args.out_dir / "controller_artifact.pt",
    )
    write_csv(args.out_dir / "training_history.csv", history)
    write_csv(
        args.out_dir / "sample_metrics.csv",
        validation_reader_rows + prospective_reader_rows,
    )
    write_csv(args.out_dir / "delivered_sample_metrics.csv", delivered_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
