from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_features,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    logits_from_positioned_inputs,
    set_language_attention_eager,
)
from probe_vsi_onevision_true_2x2_geometry import (
    quotient_inputs,
    spatial_2x2_means_and_offsets,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


BASELINE = "representative_position"
CANDIDATE = "ppe_center_ranked_k4"
METHODS = (BASELINE, CANDIDATE)
ROLE = "exposed_true_2x2_ppe_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--m0-summary", type=Path, required=True)
    parser.add_argument("--m1-summary", type=Path, required=True)
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--geometry-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def center_ranked_member_offsets(
    features: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    if group_offsets.ndim != 2 or group_offsets.shape[1] != 4:
        raise ValueError("PPE requires four constituent offsets per group")
    flat = features.reshape(-1, features.shape[-1])
    members = flat.index_select(0, group_offsets.flatten()).reshape(
        group_offsets.shape[0], 4, features.shape[-1]
    )
    center = members.mean(dim=1, keepdim=True)
    squared_distance = (members.float() - center.float()).square().sum(dim=-1)
    order = torch.argsort(squared_distance, dim=1, stable=True)
    return torch.gather(group_offsets, dim=1, index=order)


def build_frequency_position_ids(
    *,
    base_position_ids: torch.Tensor,
    ordered_group_offsets: torch.Tensor,
    visual_start: int,
    rotary_pair_count: int,
) -> torch.Tensor:
    if base_position_ids.ndim != 2 or base_position_ids.shape[0] != 1:
        raise ValueError("PPE control requires one sequence of scalar position IDs")
    if ordered_group_offsets.ndim != 2:
        raise ValueError("ordered offsets must have shape [groups, capacity]")
    capacity = ordered_group_offsets.shape[1]
    if rotary_pair_count % capacity:
        raise ValueError("rotary frequency pairs must divide evenly across PPE slots")
    group_count = ordered_group_offsets.shape[0]
    if visual_start + group_count > base_position_ids.shape[1]:
        raise ValueError("PPE visual span exceeds the compact sequence")

    frequency_positions = (
        base_position_ids.unsqueeze(-1).expand(-1, -1, rotary_pair_count).clone()
    )
    constituent_positions = visual_start + ordered_group_offsets.to(
        device=base_position_ids.device,
        dtype=base_position_ids.dtype,
    )
    positions_per_slot = rotary_pair_count // capacity
    visual_frequency_positions = constituent_positions.repeat_interleave(
        positions_per_slot, dim=1
    )
    frequency_positions[0, visual_start : visual_start + group_count] = (
        visual_frequency_positions
    )
    return frequency_positions


def rotary_cos_sin_from_frequency_positions(
    *,
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    frequency_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frequency_position_ids.ndim != 3:
        raise ValueError("frequency position IDs must have shape [batch, seq, pairs]")
    if frequency_position_ids.shape[0] != x.shape[0]:
        raise ValueError("frequency position batch does not match hidden states")
    if frequency_position_ids.shape[1] != x.shape[1]:
        raise ValueError("frequency position sequence does not match hidden states")
    if frequency_position_ids.shape[2] != inv_freq.numel():
        raise ValueError("frequency position width does not match rotary pairs")

    device_type = (
        x.device.type
        if isinstance(x.device.type, str) and x.device.type != "mps"
        else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (
            frequency_position_ids.to(device=x.device, dtype=torch.float32)
            * inv_freq.to(device=x.device, dtype=torch.float32)[None, None, :]
        )
        embedding = torch.cat((freqs, freqs), dim=-1)
        cos = embedding.cos() * attention_scaling
        sin = embedding.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PPERotaryController:
    def __init__(self, rotary: torch.nn.Module) -> None:
        if rotary.rope_type != "default":
            raise ValueError("registered PPE control requires default Qwen2 RoPE")
        self.rotary = rotary
        self.original_forward = rotary.forward
        self.frequency_position_ids: torch.Tensor | None = None
        rotary.forward = self.forward

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frequency_position_ids is None:
            return self.original_forward(x, position_ids)
        return rotary_cos_sin_from_frequency_positions(
            x=x,
            inv_freq=self.rotary.inv_freq,
            attention_scaling=float(self.rotary.attention_scaling),
            frequency_position_ids=self.frequency_position_ids,
        )

    @contextmanager
    def use(self, frequency_position_ids: torch.Tensor) -> Iterator[None]:
        if self.frequency_position_ids is not None:
            raise RuntimeError("nested PPE rotary scopes are not allowed")
        self.frequency_position_ids = frequency_position_ids
        try:
            yield
        finally:
            self.frequency_position_ids = None


def read_geometry_baseline(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["geometry"] == "spatial_2x2" and row["mode"] == "positioned_group_mass"
    ]
    if len(selected) != 24:
        raise ValueError("geometry baseline row count changed")
    return {row["sample_id"]: row for row in selected}


def classify_ppe(comparison: dict[str, float | int]) -> str:
    if (
        int(comparison["mismatch_reduction"]) >= 2
        and int(comparison["harmful_delta"]) <= 0
        and float(comparison["mean_kl_ratio"]) <= 0.8
        and float(comparison["p95_kl_ratio"]) <= 0.8
    ):
        return "PPE_STRICT_HEADROOM"
    if (
        int(comparison["mismatch_reduction"]) >= 1
        and int(comparison["harmful_delta"]) <= 0
        and float(comparison["mean_kl_ratio"]) <= 1.0
        and float(comparison["p95_kl_ratio"]) <= 1.0
    ):
        return "PPE_DECISION_HEADROOM"
    return "NO_PPE_HEADROOM"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty PPE rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], method: str) -> dict[str, float | int]:
    selected = [row for row in rows if row["method"] == method]
    kl_values = np.asarray(
        [float(row["candidate_kl"]) for row in selected], dtype=np.float64
    )
    return {
        "sample_count": len(selected),
        "agreement": sum(int(row["prediction_match"]) for row in selected)
        / len(selected),
        "mismatch_count": sum(not int(row["prediction_match"]) for row in selected),
        "harmful_count": sum(int(row["harmful"]) for row in selected),
        "candidate_kl_mean": float(kl_values.mean()),
        "candidate_kl_p95": float(np.quantile(kl_values, 0.95)),
        "token_retention": float(selected[0]["token_retention"]),
    }


def main() -> int:
    args = parse_args()
    m0_summary = json.loads(args.m0_summary.read_text(encoding="utf-8"))
    if m0_summary["decision"] != "SAME_KERNEL_MASS_VALID":
        raise ValueError("M0 decision identity changed")
    m1_summary = json.loads(args.m1_summary.read_text(encoding="utf-8"))
    if m1_summary["decision"] != "NO_BATCHED_CURRENT_SUPPORT_PATH":
        raise ValueError("M1 decision identity changed")
    geometry_summary = json.loads(args.geometry_summary.read_text(encoding="utf-8"))
    if geometry_summary["decision"] != "TRUE_2X2_DECISION_HEADROOM":
        raise ValueError("true-2x2 decision identity changed")
    geometry_rows = read_geometry_baseline(args.geometry_rows)

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    selected = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )[args.sample_offset : args.sample_offset + args.sample_count]
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
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype
    rotary = model.model.language_model.rotary_emb
    controller = PPERotaryController(rotary)
    rotary_pair_count = int(rotary.inv_freq.numel())
    if rotary_pair_count != 64:
        raise ValueError("registered Qwen2 rotary pair count changed")

    rows: list[dict[str, object]] = []
    maximum_baseline_kl_repeat_error = 0.0
    baseline_prediction_repeat_mismatches = 0
    maximum_standard_rotary_logit_error = 0.0
    started = time.perf_counter()
    for sample_position, sample in enumerate(
        selected,
        start=args.sample_offset + 1,
    ):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        position_tensor = torch.tensor(selected_positions, dtype=torch.long)
        reference = (
            payload["features"]
            .index_select(0, position_tensor)
            .to(
                device=device,
                dtype=model_dtype,
            )
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
        quotient_means, group_offsets = spatial_2x2_means_and_offsets(approximate)
        ordered_offsets = center_ranked_member_offsets(approximate, group_offsets)
        full_video_token_count = reference.shape[0] * reference.shape[1]
        variable_embeds, explicit_mask, position_ids = quotient_inputs(
            mode="positioned_group_mass",
            model=model,
            prompt_input_ids=prompt_batch["input_ids"],
            prompt_attention_mask=prompt_batch["attention_mask"],
            quotient_means=quotient_means,
            group_offsets=group_offsets,
            full_video_token_count=full_video_token_count,
        )
        video_mask = prompt_batch["input_ids"][0] == model.config.video_token_index
        placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
        visual_start = int(placeholder_positions[0].item())
        ppe_position_ids = build_frequency_position_ids(
            base_position_ids=position_ids,
            ordered_group_offsets=ordered_offsets,
            visual_start=visual_start,
            rotary_pair_count=rotary_pair_count,
        )
        scalar_expansion = position_ids.unsqueeze(-1).expand(-1, -1, rotary_pair_count)

        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        with torch.inference_mode():
            dense_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            ).float()
            baseline_logits = logits_from_positioned_inputs(
                model=model,
                variable_embeds=variable_embeds,
                attention_mask=explicit_mask,
                position_ids=position_ids,
            )
            with controller.use(scalar_expansion):
                standard_repeat_logits = logits_from_positioned_inputs(
                    model=model,
                    variable_embeds=variable_embeds,
                    attention_mask=explicit_mask,
                    position_ids=position_ids,
                )
            with controller.use(ppe_position_ids):
                ppe_logits = logits_from_positioned_inputs(
                    model=model,
                    variable_embeds=variable_embeds,
                    attention_mask=explicit_mask,
                    position_ids=position_ids,
                )

        standard_rotary_error = float(
            (baseline_logits - standard_repeat_logits).abs().max().item()
        )
        maximum_standard_rotary_logit_error = max(
            maximum_standard_rotary_logit_error, standard_rotary_error
        )
        dense_candidates = dense_logits.index_select(0, token_tensor)
        teacher_index = int(torch.argmax(dense_candidates).item())
        baseline_correct = teacher_index == sample.answer_index
        for method, logits in (
            (BASELINE, baseline_logits),
            (CANDIDATE, ppe_logits),
        ):
            candidates = logits.index_select(0, token_tensor)
            approximate_index = int(torch.argmax(candidates).item())
            kl_value = candidate_kl(dense_candidates, candidates)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_position": sample_position,
                    "method": method,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "candidate_kl": kl_value,
                    "prediction_match": int(approximate_index == teacher_index),
                    "baseline_correct": int(baseline_correct),
                    "approximate_correct": int(
                        approximate_index == sample.answer_index
                    ),
                    "harmful": int(
                        baseline_correct and approximate_index != sample.answer_index
                    ),
                    "quotient_token_count": quotient_means.shape[0],
                    "token_retention": quotient_means.shape[0] / full_video_token_count,
                }
            )
        previous = geometry_rows[sample.sample_id]
        baseline_kl = float(rows[-2]["candidate_kl"])
        maximum_baseline_kl_repeat_error = max(
            maximum_baseline_kl_repeat_error,
            abs(baseline_kl - float(previous["candidate_kl"])),
        )
        baseline_prediction_repeat_mismatches += int(
            rows[-2]["approximate_index"]
        ) != int(previous["approximate_index"])

        print(
            json.dumps(
                {
                    "event": "true_2x2_ppe_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "standard_rotary_logit_error": standard_rotary_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if maximum_baseline_kl_repeat_error > 1e-6:
        raise RuntimeError("PPE baseline did not reproduce the geometry control")
    if baseline_prediction_repeat_mismatches:
        raise RuntimeError("PPE baseline predictions changed")
    if maximum_standard_rotary_logit_error > 1e-5:
        raise RuntimeError("frequency-wise scalar positions changed Qwen2 RoPE")

    summaries = {method: summarize(rows, method) for method in METHODS}
    incumbent = summaries[BASELINE]
    candidate = summaries[CANDIDATE]
    baseline_rows = {
        str(row["sample_id"]): row for row in rows if row["method"] == BASELINE
    }
    candidate_rows = [row for row in rows if row["method"] == CANDIDATE]
    comparison: dict[str, float | int] = {
        "mismatch_reduction": int(incumbent["mismatch_count"])
        - int(candidate["mismatch_count"]),
        "harmful_delta": int(candidate["harmful_count"])
        - int(incumbent["harmful_count"]),
        "mean_kl_ratio": float(candidate["candidate_kl_mean"])
        / float(incumbent["candidate_kl_mean"]),
        "p95_kl_ratio": float(candidate["candidate_kl_p95"])
        / float(incumbent["candidate_kl_p95"]),
        "paired_kl_wins": sum(
            float(row["candidate_kl"])
            < float(baseline_rows[str(row["sample_id"])]["candidate_kl"])
            for row in candidate_rows
        ),
        "prediction_match_wins": sum(
            int(row["prediction_match"])
            > int(baseline_rows[str(row["sample_id"])]["prediction_match"])
            for row in candidate_rows
        ),
        "prediction_match_losses": sum(
            int(row["prediction_match"])
            < int(baseline_rows[str(row["sample_id"])]["prediction_match"])
            for row in candidate_rows
        ),
    }
    decision = classify_ppe(comparison)
    write_csv(args.out_dir / "true_2x2_ppe_rows.csv", rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [73, 96],
        "sample_count": len(selected),
        "methods": list(METHODS),
        "attention_implementation": "eager",
        "ppe_capacity": 4,
        "rotary_pair_count": rotary_pair_count,
        "maximum_baseline_kl_repeat_error": maximum_baseline_kl_repeat_error,
        "baseline_prediction_repeat_mismatches": baseline_prediction_repeat_mismatches,
        "maximum_standard_rotary_logit_error": maximum_standard_rotary_logit_error,
        "summaries": summaries,
        "comparison": comparison,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Paper-faithful K=4 one-dimensional PPE control on the fixed true-2x2 "
            "group-mass quotient and exposed positions 73-96. It does not test "
            "support routing, progressive refinement, selection, formal, or latency."
        ),
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
