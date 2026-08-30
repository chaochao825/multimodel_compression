from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mvbench_onevision_utils import (
    build_prompt_batch,
    decode_video_frames,
    first_token_logits_from_features,
    load_onevision_model,
    uniform_frame_indices,
)
from probe_vsi_onevision_cmrq_stage_b import feature_path_for_sample
from probe_vsi_onevision_query_fixed_measure_remainder import SelectedLayerCapture
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (
    LAYERS,
    build_gaussian_components,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


TOPOLOGIES = ("flat_contiguous_4", "spatial_2x2")
METHODS = ("centroid", "key_secant", "joint_secant", "independent_secant")
QUANTIZATIONS = ("fp32", "symmetric_int8")
ROLE = "exposed_query_fixed_local_secant_measure"


@dataclass
class SecantState:
    mean_key: torch.Tensor
    mean_value: torch.Tensor
    key_direction: torch.Tensor
    value_direction: torch.Tensor
    key_coordinate: torch.Tensor
    value_coordinate: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prototype-summary", type=Path, required=True)
    parser.add_argument("--ppe-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def group_offsets(
    *, frame_count: int, token_count: int, topology: str, device: torch.device
) -> torch.Tensor:
    if frame_count != 8 or token_count != 196:
        raise ValueError("registered visual geometry must be 8 x 14 x 14")
    grid = torch.arange(
        frame_count * token_count, device=device, dtype=torch.long
    ).reshape(frame_count, 14, 14)
    if topology == "flat_contiguous_4":
        offsets = grid.reshape(-1, 4)
    elif topology == "spatial_2x2":
        offsets = (
            grid.reshape(frame_count, 7, 2, 7, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4)
        )
    else:
        raise ValueError(f"unregistered topology: {topology}")
    expected = torch.arange(frame_count * token_count, device=device)
    if not torch.equal(torch.sort(offsets.flatten()).values, expected):
        raise RuntimeError("local groups do not partition the visual tokens")
    return offsets


def _rank_one(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 4:
        raise ValueError("rank-one input must have [heads, groups, members, dim]")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    coordinate = left[..., 0] * singular[..., :1]
    direction = right[..., 0, :]
    return coordinate, direction


def build_secant_state(
    member_key: torch.Tensor, member_value: torch.Tensor, *, method: str
) -> SecantState:
    if member_key.ndim != 4 or member_value.shape != member_key.shape:
        raise ValueError("member K/V must share [heads, groups, members, dim]")
    mean_key = member_key.mean(dim=2)
    mean_value = member_value.mean(dim=2)
    centered_key = member_key - mean_key.unsqueeze(2)
    centered_value = member_value - mean_value.unsqueeze(2)
    zero_direction = torch.zeros_like(mean_key)
    zero_coordinate = torch.zeros_like(centered_key[..., 0, 0])

    if method == "centroid":
        return SecantState(
            mean_key=mean_key,
            mean_value=mean_value,
            key_direction=zero_direction,
            value_direction=zero_direction,
            key_coordinate=zero_coordinate.unsqueeze(-1).expand_as(
                centered_key[..., 0]
            ),
            value_coordinate=zero_coordinate.unsqueeze(-1).expand_as(
                centered_key[..., 0]
            ),
        )

    key_coordinate, key_direction = _rank_one(centered_key)
    if method == "key_secant":
        denominator = key_coordinate.square().sum(dim=2).clamp_min(1e-12)
        value_direction = torch.einsum(
            "hgm,hgmd->hgd", key_coordinate, centered_value
        ) / denominator.unsqueeze(-1)
        return SecantState(
            mean_key=mean_key,
            mean_value=mean_value,
            key_direction=key_direction,
            value_direction=value_direction,
            key_coordinate=key_coordinate,
            value_coordinate=key_coordinate,
        )
    if method == "independent_secant":
        value_coordinate, value_direction = _rank_one(centered_value)
        return SecantState(
            mean_key=mean_key,
            mean_value=mean_value,
            key_direction=key_direction,
            value_direction=value_direction,
            key_coordinate=key_coordinate,
            value_coordinate=value_coordinate,
        )
    if method == "joint_secant":
        key_scale = centered_key.square().mean(dim=(2, 3)).sqrt().clamp_min(1e-6)
        value_scale = centered_value.square().mean(dim=(2, 3)).sqrt().clamp_min(1e-6)
        balanced = torch.cat(
            (
                centered_key / key_scale[:, :, None, None],
                centered_value / value_scale[:, :, None, None],
            ),
            dim=-1,
        )
        coordinate, direction = _rank_one(balanced)
        head_dim = member_key.shape[-1]
        return SecantState(
            mean_key=mean_key,
            mean_value=mean_value,
            key_direction=direction[..., :head_dim] * key_scale.unsqueeze(-1),
            value_direction=direction[..., head_dim:] * value_scale.unsqueeze(-1),
            key_coordinate=coordinate,
            value_coordinate=coordinate,
        )
    raise ValueError(f"unregistered secant method: {method}")


def symmetric_int8_dequantize(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim < 1:
        raise ValueError("quantized vectors must have at least one dimension")
    scale = vectors.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(vectors / scale).clamp(-127, 127) * scale


def quantize_state(state: SecantState, *, quantization: str) -> SecantState:
    if quantization == "fp32":
        return state
    if quantization != "symmetric_int8":
        raise ValueError(f"unregistered quantization: {quantization}")
    return SecantState(
        mean_key=symmetric_int8_dequantize(state.mean_key),
        mean_value=symmetric_int8_dequantize(state.mean_value),
        key_direction=symmetric_int8_dequantize(state.key_direction),
        value_direction=symmetric_int8_dequantize(state.value_direction),
        key_coordinate=state.key_coordinate,
        value_coordinate=state.value_coordinate,
    )


def reconstructed_members(state: SecantState) -> tuple[torch.Tensor, torch.Tensor]:
    key = state.mean_key.unsqueeze(2) + torch.einsum(
        "hgm,hgd->hgmd", state.key_coordinate, state.key_direction
    )
    value = state.mean_value.unsqueeze(2) + torch.einsum(
        "hgm,hgd->hgmd", state.value_coordinate, state.value_direction
    )
    return key, value


def state_cost(
    *, method: str, quantization: str, group_count: int, group_size: int, head_dim: int
) -> dict[str, float | int]:
    dense_bytes = group_count * group_size * 2 * head_dim * 2
    vector_count = 2 if method == "centroid" else 4
    vector_bytes = 4 if quantization == "fp32" else 1
    coordinate_count = (
        0
        if method == "centroid"
        else group_size * (1 if method in {"key_secant", "joint_secant"} else 2)
    )
    scale_count = 0 if quantization == "fp32" else vector_count
    state_bytes = group_count * (
        vector_count * head_dim * vector_bytes + coordinate_count * 2 + scale_count * 2
    )
    arithmetic_ratio = 4.0 if method == "centroid" else 2.0
    return {
        "dense_state_bytes_per_head": dense_bytes,
        "state_bytes_per_head": state_bytes,
        "state_byte_ratio": dense_bytes / state_bytes,
        "attention_arithmetic_ratio": arithmetic_ratio,
    }


def evaluate_state(components, state: SecantState) -> dict[str, float]:
    approximate_key, approximate_value = reconstructed_members(state)
    approximate_scores = torch.einsum(
        "hd,hgmd->hgm", components.query_scaled, approximate_key
    )
    exact_maximum = components.maximum.squeeze(-1)
    stabilizer = torch.maximum(exact_maximum, approximate_scores.amax(dim=(1, 2)))
    approximate_exp = torch.exp(approximate_scores - stabilizer[:, None, None])
    visual_z = approximate_exp.sum(dim=(1, 2))
    visual_n = torch.einsum("hgm,hgmd->hd", approximate_exp, approximate_value)
    exact_rescale = torch.exp(exact_maximum - stabilizer)
    nonvisual_z = components.nonvisual_z * exact_rescale
    nonvisual_n = components.nonvisual_n * exact_rescale.unsqueeze(-1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(-1)
    head_error = torch.linalg.vector_norm(
        visual_output - components.exact_visual_output, dim=-1
    ) / torch.linalg.vector_norm(components.exact_visual_output, dim=-1).clamp_min(
        1e-12
    )
    visual_relative = torch.linalg.vector_norm(
        visual_output - components.exact_visual_output
    ) / torch.linalg.vector_norm(components.exact_visual_output).clamp_min(1e-12)
    full_relative = torch.linalg.vector_norm(
        full_output - components.exact_full_output
    ) / torch.linalg.vector_norm(components.exact_full_output).clamp_min(1e-12)
    member_key_error = torch.linalg.vector_norm(
        approximate_key - components.member_key
    ) / torch.linalg.vector_norm(components.member_key).clamp_min(1e-12)
    member_value_error = torch.linalg.vector_norm(
        approximate_value - components.member_value
    ) / torch.linalg.vector_norm(components.member_value).clamp_min(1e-12)
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
        "member_key_relative_l2": float(member_key_error.item()),
        "member_value_relative_l2": float(member_value_error.item()),
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (str(row["topology"]), str(row["method"]), str(row["quantization"]))
            for row in rows
        }
    )
    summaries: list[dict[str, object]] = []
    for topology, method, quantization in keys:
        selected = [
            row
            for row in rows
            if row["topology"] == topology
            and row["method"] == method
            and row["quantization"] == quantization
        ]
        visual = np.asarray([float(row["visual_relative_l2"]) for row in selected])
        full = np.asarray([float(row["full_relative_l2"]) for row in selected])
        summaries.append(
            {
                "topology": topology,
                "method": method,
                "quantization": quantization,
                "cell_count": len(selected),
                "visual_mean": float(visual.mean()),
                "visual_p95": float(np.quantile(visual, 0.95)),
                "visual_worst": float(visual.max()),
                "visual_worst_head": max(
                    float(row["visual_worst_head_relative_l2"]) for row in selected
                ),
                "full_mean": float(full.mean()),
                "full_p95": float(np.quantile(full, 0.95)),
                "full_worst": float(full.max()),
                "member_key_mean": float(
                    np.mean([float(row["member_key_relative_l2"]) for row in selected])
                ),
                "member_value_mean": float(
                    np.mean(
                        [float(row["member_value_relative_l2"]) for row in selected]
                    )
                ),
                "state_byte_ratio": float(selected[0]["state_byte_ratio"]),
                "attention_arithmetic_ratio": float(
                    selected[0]["attention_arithmetic_ratio"]
                ),
            }
        )
    return summaries


def deployable_pass(summary: dict[str, object]) -> bool:
    return (
        summary["method"] != "centroid"
        and summary["quantization"] == "symmetric_int8"
        and int(summary["cell_count"]) == 72
        and float(summary["visual_mean"]) <= 0.01
        and float(summary["visual_p95"]) <= 0.02
        and float(summary["visual_worst"]) <= 0.05
        and float(summary["full_mean"]) <= 0.005
        and float(summary["full_p95"]) <= 0.01
        and float(summary["state_byte_ratio"]) >= 3.5
        and float(summary["attention_arithmetic_ratio"]) >= 1.8
    )


def capacity_pass(summary: dict[str, object]) -> bool:
    return (
        summary["method"] != "centroid"
        and summary["quantization"] == "fp32"
        and int(summary["cell_count"]) == 72
        and float(summary["visual_mean"]) <= 0.005
        and float(summary["visual_p95"]) <= 0.01
        and float(summary["visual_worst"]) <= 0.02
        and float(summary["full_mean"]) <= 0.0025
        and float(summary["full_p95"]) <= 0.005
    )


def classify_outcome(
    summaries: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    deployable = [summary for summary in summaries if deployable_pass(summary)]
    capacity = [summary for summary in summaries if capacity_pass(summary)]
    secants = [summary for summary in summaries if summary["method"] != "centroid"]
    best = min(secants, key=lambda summary: float(summary["visual_mean"]))
    diagnostics = {
        "deployable_pass_count": len(deployable),
        "capacity_pass_count": len(capacity),
        "best_secant": best,
    }
    if deployable:
        return "LOCAL_SECANT_DEPLOYABLE_CAPACITY", diagnostics
    if capacity:
        return "LOCAL_SECANT_CAPACITY_ONLY", diagnostics
    return "NO_LOCAL_SECANT_PATH", diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty local-secant rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    prototype = json.loads(args.prototype_summary.read_text(encoding="utf-8"))
    ppe = json.loads(args.ppe_summary.read_text(encoding="utf-8"))
    if prototype["decision"] != "NO_PROTOTYPE_MIXTURE_PATH":
        raise ValueError("prototype-mixture prerequisite changed")
    if ppe["decision"] != "NO_PPE_HEADROOM":
        raise ValueError("PPE prerequisite changed")
    expected_count = 1 if args.smoke else 24
    if args.sample_offset != 72 or args.sample_count != expected_count:
        raise ValueError(
            "registered Gate requires position 73 for smoke or positions 73-96 formal"
        )
    if args.frame_budget != 8:
        raise ValueError("registered frame budget changed")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )[args.sample_offset : args.sample_offset + args.sample_count]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype
    capture = SelectedLayerCapture(model, LAYERS)
    rows: list[dict[str, object]] = []
    maximum_replay_error = 0.0
    started = time.perf_counter()

    for sample_position, sample in enumerate(samples, start=args.sample_offset + 1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        reference = (
            payload["features"]
            .index_select(0, torch.tensor(selected_positions, dtype=torch.long))
            .to(device=device, dtype=model_dtype)
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
        capture.clear()
        with torch.inference_mode():
            first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
        if set(capture.captures) != set(LAYERS):
            raise RuntimeError("not every registered attention layer was captured")

        video_mask = prompt_batch["input_ids"][0] == model.config.video_token_index
        placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
        visual_start = int(placeholder_positions[0].item())
        visual_token_count = reference.shape[0] * reference.shape[1]
        for topology in TOPOLOGIES:
            offsets = group_offsets(
                frame_count=reference.shape[0],
                token_count=reference.shape[1],
                topology=topology,
                device=device,
            )
            for layer_index in LAYERS:
                components = build_gaussian_components(
                    capture.captures[layer_index],
                    visual_start=visual_start,
                    visual_token_count=visual_token_count,
                    group_offsets=offsets,
                    max_rank=3,
                )
                maximum_replay_error = max(
                    maximum_replay_error, components.replay_error
                )
                for method in METHODS:
                    state = build_secant_state(
                        components.member_key, components.member_value, method=method
                    )
                    for quantization in QUANTIZATIONS:
                        candidate = quantize_state(state, quantization=quantization)
                        cost = state_cost(
                            method=method,
                            quantization=quantization,
                            group_count=offsets.shape[0],
                            group_size=offsets.shape[1],
                            head_dim=components.member_key.shape[-1],
                        )
                        rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "sample_position": sample_position,
                                "layer_index": layer_index,
                                "topology": topology,
                                "method": method,
                                "quantization": quantization,
                                **cost,
                                **evaluate_state(components, candidate),
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "local_secant_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_replay_error": maximum_replay_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if maximum_replay_error > 1e-4:
        raise RuntimeError("captured Q/K/V did not reconstruct attention output")
    summaries = summarize_rows(rows)
    decision, diagnostics = classify_outcome(summaries)
    write_csv(args.out_dir / "local_secant_rows.csv", rows)
    write_csv(args.out_dir / "local_secant_summary.csv", summaries)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": f"{ROLE}_smoke" if args.smoke else ROLE,
        "decision": decision,
        "sample_positions": [
            args.sample_offset + 1,
            args.sample_offset + args.sample_count,
        ],
        "sample_count": len(samples),
        "layers": list(LAYERS),
        "topologies": list(TOPOLOGIES),
        "methods": list(METHODS),
        "quantizations": list(QUANTIZATIONS),
        "maximum_replay_error": maximum_replay_error,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fixed-query single-layer exposed capacity diagnostic. Arithmetic and "
            "state-byte ratios are analytic proxies; no reader accuracy, reusable "
            "KV cache, selection, formal, TTFT, kernel, or wall-clock claim."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
