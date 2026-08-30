from __future__ import annotations

import argparse
import csv
import json
import math
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
from probe_vsi_onevision_query_fixed_headwise_support_ceiling import (
    evaluate_support,
)
from probe_vsi_onevision_query_fixed_measure_remainder import (
    BUDGETS,
    LAYERS,
    AttentionCapture,
    MeasureState,
    SelectedLayerCapture,
    grouped_measure_state,
    nonvisual_measure,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from probe_vsi_onevision_true_2x2_geometry import spatial_2x2_means_and_offsets
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


ORDERS = (0, 1, 2, 3)
METHODS = tuple(f"taylor_order{order}" for order in ORDERS)
ROLE = "exposed_query_fixed_taylor_cross_moment"


@dataclass
class TaylorState:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_group_z: torch.Tensor
    exact_group_n: torch.Tensor
    coarse_group_z: torch.Tensor
    coarse_group_n: torch.Tensor
    oracle_priority: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--headwise-summary", type=Path, required=True)
    parser.add_argument("--headwise-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def taylor_polynomial(values: torch.Tensor, order: int) -> torch.Tensor:
    if order not in ORDERS:
        raise ValueError("Taylor order is not registered")
    result = torch.ones_like(values)
    power = torch.ones_like(values)
    for degree in range(1, order + 1):
        power = power * values
        result = result + power / math.factorial(degree)
    return result


def build_taylor_state(
    capture: AttentionCapture,
    *,
    exact_state: MeasureState,
    visual_start: int,
    visual_token_count: int,
    group_offsets: torch.Tensor,
    order: int,
) -> tuple[TaylorState | None, int]:
    query = capture.query[:, -1].float()
    key = capture.key.float()
    value = capture.value.float()
    scores = torch.einsum("hd,hsd->hs", query, key) * float(capture.module.scaling)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask[:, -1].float()
    maximum = scores.max(dim=1, keepdim=True).values
    visual_stop = visual_start + visual_token_count
    visual_scores = scores[:, visual_start:visual_stop]
    visual_key = key[:, visual_start:visual_stop]
    visual_value = value[:, visual_start:visual_stop]
    head_count, _, head_dim = visual_value.shape
    offsets = group_offsets.to(device=key.device)
    group_count, group_size = offsets.shape
    member_scores = visual_scores.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size
    )
    member_value = visual_value.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size, head_dim
    )
    member_key = visual_key.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size, head_dim
    )
    key_center = member_key.mean(dim=2)
    score_center = torch.einsum("hd,hgd->hg", query, key_center) * float(
        capture.module.scaling
    )
    score_residual = member_scores - score_center.unsqueeze(-1)
    polynomial = taylor_polynomial(score_residual, order)
    common_scale = torch.exp(score_center - maximum).unsqueeze(-1)
    approximate_weights = common_scale * polynomial
    coarse_group_z = approximate_weights.sum(dim=2)
    coarse_group_n = torch.einsum("hgm,hgmd->hgd", approximate_weights, member_value)
    nonpositive_group_count = int((coarse_group_z <= 0).sum().item())
    if nonpositive_group_count:
        return None, nonpositive_group_count
    exact_group_z = exact_state.exact_group_z
    exact_group_n = exact_state.exact_group_n
    exact_visual_z = exact_group_z.sum(dim=1)
    exact_visual_output = exact_state.exact_visual_output
    local_output_defect = (exact_group_n - coarse_group_n) / exact_visual_z[
        :, None, None
    ] - exact_visual_output[:, None, :] * (exact_group_z - coarse_group_z)[
        :, :, None
    ] / exact_visual_z[:, None, None]
    oracle_priority = torch.linalg.vector_norm(local_output_defect, dim=-1)
    return (
        TaylorState(
            exact_visual_output=exact_visual_output,
            exact_full_output=exact_state.exact_full_output,
            exact_group_z=exact_group_z,
            exact_group_n=exact_group_n,
            coarse_group_z=coarse_group_z,
            coarse_group_n=coarse_group_n,
            oracle_priority=oracle_priority,
        ),
        0,
    )


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for method in METHODS:
        method_summary: dict[str, object] = {}
        for budget in BUDGETS:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["exact_group_count"] == budget
            ]
            valid = [row for row in selected if row["state_valid"] == 1]
            metrics: dict[str, object] = {
                "cell_count": len(selected),
                "valid_cell_count": len(valid),
                "invalid_cell_count": len(selected) - len(valid),
            }
            if valid:
                visual = np.asarray([row["visual_relative_l2"] for row in valid])
                full = np.asarray([row["full_relative_l2"] for row in valid])
                metrics.update(
                    {
                        "visual_mean": float(visual.mean()),
                        "visual_p95": float(np.quantile(visual, 0.95)),
                        "visual_worst": float(visual.max()),
                        "visual_worst_head": max(
                            float(row["visual_worst_head_relative_l2"]) for row in valid
                        ),
                        "full_mean": float(full.mean()),
                        "full_p95": float(np.quantile(full, 0.95)),
                        "full_worst": float(full.max()),
                    }
                )
            else:
                metrics.update(
                    {
                        "visual_mean": None,
                        "visual_p95": None,
                        "visual_worst": None,
                        "visual_worst_head": None,
                        "full_mean": None,
                        "full_p95": None,
                        "full_worst": None,
                    }
                )
            method_summary[str(budget)] = metrics
        summaries[method] = method_summary
    return summaries


def capacity_pass(metrics: dict[str, object]) -> bool:
    if metrics["valid_cell_count"] != metrics["cell_count"]:
        return False
    return (
        float(metrics["visual_mean"]) <= 0.01
        and float(metrics["visual_p95"]) <= 0.02
        and float(metrics["visual_worst"]) <= 0.05
        and float(metrics["full_mean"]) <= 0.005
        and float(metrics["full_p95"]) <= 0.01
    )


def classify_outcome(
    summaries: dict[str, dict[str, object]],
) -> tuple[str, dict[str, bool]]:
    passes = {
        f"order{order}_pass": capacity_pass(summaries[f"taylor_order{order}"]["196"])
        for order in ORDERS
    }
    if passes["order1_pass"]:
        return "TAYLOR_CROSS_MOMENT_ORDER1_PASS", passes
    if passes["order2_pass"]:
        return "TAYLOR_CROSS_MOMENT_ORDER2_PASS", passes
    if passes["order3_pass"]:
        return "TAYLOR_CROSS_MOMENT_ORDER3_ONLY", passes
    return "NO_TAYLOR_CROSS_MOMENT_CAPACITY", passes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_order0_control(
    rows: list[dict[str, object]], previous_rows_path: Path
) -> None:
    with previous_rows_path.open(newline="", encoding="utf-8") as handle:
        previous_rows = list(csv.DictReader(handle))
    expected = {
        (
            row["sample_id"],
            int(row["layer_index"]),
            int(row["exact_group_count"]),
        ): row
        for row in previous_rows
        if row["method"] == "headwise_exact_local"
    }
    order0_rows = [row for row in rows if row["method"] == "taylor_order0"]
    for row in order0_rows:
        key = (
            str(row["sample_id"]),
            int(row["layer_index"]),
            int(row["exact_group_count"]),
        )
        reference = expected[key]
        for metric in (
            "visual_relative_l2",
            "visual_worst_head_relative_l2",
            "full_relative_l2",
        ):
            if abs(float(row[metric]) - float(reference[metric])) > 1e-8:
                raise RuntimeError("Taylor order-0 control did not reproduce")


def main() -> int:
    args = parse_args()
    previous = json.loads(args.headwise_summary.read_text(encoding="utf-8"))
    if previous["decision"] != "HEADWISE_SUPPORT_PARTIAL":
        raise ValueError("headwise-support prerequisite decision changed")
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    selected_samples = select_calibration_questions(
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
    nonpositive_group_counts = {order: 0 for order in ORDERS}
    replay_maximum = 0.0
    local_bound_violations = 0
    started = time.perf_counter()
    for sample_position, sample in enumerate(
        selected_samples, start=args.sample_offset + 1
    ):
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
            processor, sample, np.stack(frames), device=device, dtype=model_dtype
        )
        capture.clear()
        with torch.inference_mode():
            first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
        video_mask = prompt_batch["input_ids"][0] == model.config.video_token_index
        placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
        visual_start = int(placeholder_positions[0].item())
        visual_token_count = reference.shape[0] * reference.shape[1]
        _, group_offsets = spatial_2x2_means_and_offsets(reference)
        for layer_index in LAYERS:
            layer_capture = capture.captures[layer_index]
            exact_state = grouped_measure_state(
                layer_capture,
                visual_start=visual_start,
                visual_token_count=visual_token_count,
                group_offsets=group_offsets,
            )
            nonvisual_z, nonvisual_n = nonvisual_measure(
                layer_capture,
                visual_start=visual_start,
                visual_token_count=visual_token_count,
            )
            replay_maximum = max(
                replay_maximum, exact_state.exact_projected_relative_error
            )
            local_bound_violations += (
                exact_state.local_z_bound_violations
                + exact_state.local_n_bound_violations
            )
            for order in ORDERS:
                state, nonpositive_group_count = build_taylor_state(
                    layer_capture,
                    exact_state=exact_state,
                    visual_start=visual_start,
                    visual_token_count=visual_token_count,
                    group_offsets=group_offsets,
                    order=order,
                )
                nonpositive_group_counts[order] += nonpositive_group_count
                if state is None:
                    for budget in BUDGETS:
                        rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "sample_position": sample_position,
                                "layer_index": layer_index,
                                "method": f"taylor_order{order}",
                                "taylor_order": order,
                                "exact_group_count": budget,
                                "visual_token_retention": (392 + 3 * budget) / 1568,
                                "state_valid": 0,
                                "nonpositive_group_count": nonpositive_group_count,
                                "visual_relative_l2": None,
                                "visual_worst_head_relative_l2": None,
                                "full_relative_l2": None,
                            }
                        )
                    continue
                support_order = torch.argsort(
                    state.oracle_priority, dim=1, descending=True, stable=True
                )
                for budget in BUDGETS:
                    metrics = evaluate_support(
                        state,
                        support_order[:, :budget],
                        nonvisual_z=nonvisual_z,
                        nonvisual_n=nonvisual_n,
                    )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "sample_position": sample_position,
                            "layer_index": layer_index,
                            "method": f"taylor_order{order}",
                            "taylor_order": order,
                            "exact_group_count": budget,
                            "visual_token_retention": (392 + 3 * budget) / 1568,
                            "state_valid": 1,
                            "nonpositive_group_count": 0,
                            **metrics,
                        }
                    )
        print(
            json.dumps(
                {
                    "event": "taylor_cross_moment_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if replay_maximum > 1e-4 or local_bound_violations:
        raise RuntimeError("Taylor capture or local-bound guard failed")
    validate_order0_control(rows, args.headwise_rows)
    summaries = summarize_rows(rows)
    decision, diagnostics = classify_outcome(summaries)
    write_csv(args.out_dir / "taylor_cross_moment_rows.csv", rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [
            args.sample_offset + 1,
            args.sample_offset + args.sample_count,
        ],
        "sample_count": len(selected_samples),
        "layers": list(LAYERS),
        "orders": list(ORDERS),
        "budgets": list(BUDGETS),
        "maximum_replay_error": replay_maximum,
        "local_bound_violations": local_bound_violations,
        "nonpositive_group_counts": nonpositive_group_counts,
        "order0_control_reproduced": True,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Target-visible per-head support with oracle leaf-derived Taylor moments "
            "on exposed positions 73-96. It is a capacity mechanism probe, not a "
            "compressed state, deployable method, reader endpoint, or latency claim."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
