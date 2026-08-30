from __future__ import annotations

import argparse
import csv
import json
import time
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
from probe_vsi_onevision_query_fixed_measure_remainder import (
    BUDGETS,
    LAYERS,
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


METHODS = (
    "shared_exact_local",
    "headwise_attention_mass",
    "headwise_exact_local",
    "headwise_exact_greedy",
)
HEADWISE_METHODS = METHODS[1:]
ROLE = "exposed_query_fixed_headwise_support_ceiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-summary", type=Path, required=True)
    parser.add_argument("--query-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def shared_exact_local_order(state: MeasureState) -> torch.Tensor:
    return torch.argsort(
        torch.linalg.vector_norm(state.oracle_priority, dim=0),
        descending=True,
        stable=True,
    )


def headwise_exact_greedy_order(state: MeasureState) -> torch.Tensor:
    head_count, group_count = state.exact_group_z.shape
    delta_z = state.exact_group_z - state.coarse_group_z
    delta_n = state.exact_group_n - state.coarse_group_n
    current_z = state.coarse_group_z.sum(dim=1)
    current_n = state.coarse_group_n.sum(dim=1)
    selected = torch.zeros(
        (head_count, group_count), device=current_z.device, dtype=torch.bool
    )
    order = torch.empty(
        (head_count, group_count), device=current_z.device, dtype=torch.long
    )
    head_indices = torch.arange(head_count, device=current_z.device)
    for step in range(group_count):
        candidate_z = current_z[:, None] + delta_z
        candidate_n = current_n[:, None, :] + delta_n
        candidate_output = candidate_n / candidate_z.unsqueeze(-1)
        candidate_error = torch.linalg.vector_norm(
            candidate_output - state.exact_visual_output[:, None, :], dim=-1
        )
        candidate_error[selected] = torch.inf
        next_group = torch.argmin(candidate_error, dim=1)
        selected[head_indices, next_group] = True
        current_z = current_z + delta_z[head_indices, next_group]
        current_n = current_n + delta_n[head_indices, next_group]
        order[:, step] = next_group
    if not torch.all(
        torch.sort(order, dim=1).values
        == torch.arange(group_count, device=order.device)
    ):
        raise RuntimeError("headwise exact greedy order is not a per-head permutation")
    return order


def selector_orders(state: MeasureState) -> dict[str, torch.Tensor]:
    return {
        "shared_exact_local": shared_exact_local_order(state),
        "headwise_attention_mass": torch.argsort(
            state.mass_priority, dim=1, descending=True, stable=True
        ),
        "headwise_exact_local": torch.argsort(
            state.oracle_priority, dim=1, descending=True, stable=True
        ),
        "headwise_exact_greedy": headwise_exact_greedy_order(state),
    }


def evaluate_support(
    state: MeasureState,
    selected_indices: torch.Tensor,
    *,
    nonvisual_z: torch.Tensor,
    nonvisual_n: torch.Tensor,
) -> dict[str, float]:
    head_count, group_count = state.exact_group_z.shape
    selected = torch.zeros(
        (head_count, group_count),
        device=state.exact_group_z.device,
        dtype=torch.bool,
    )
    if selected_indices.ndim == 1:
        selected[:, selected_indices] = True
    elif selected_indices.ndim == 2:
        selected.scatter_(1, selected_indices, True)
    else:
        raise ValueError("selected support must be shared or headwise")
    visual_z = torch.where(selected, state.exact_group_z, state.coarse_group_z).sum(
        dim=1
    )
    visual_n = torch.where(
        selected.unsqueeze(-1), state.exact_group_n, state.coarse_group_n
    ).sum(dim=1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(-1)
    head_error = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output, dim=-1
    ) / torch.linalg.vector_norm(state.exact_visual_output, dim=-1).clamp_min(1e-12)
    visual_relative = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output
    ) / torch.linalg.vector_norm(state.exact_visual_output).clamp_min(1e-12)
    full_relative = torch.linalg.vector_norm(
        full_output - state.exact_full_output
    ) / torch.linalg.vector_norm(state.exact_full_output).clamp_min(1e-12)
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
    }


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
            visual = np.asarray([row["visual_relative_l2"] for row in selected])
            full = np.asarray([row["full_relative_l2"] for row in selected])
            method_summary[str(budget)] = {
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
            }
        summaries[method] = method_summary
    return summaries


def build_headwise_envelope(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int]]]:
    envelope_rows: list[dict[str, object]] = []
    cells = sorted(
        {
            (
                str(row["sample_id"]),
                int(row["sample_position"]),
                int(row["layer_index"]),
            )
            for row in rows
        }
    )
    for sample_id, sample_position, layer_index in cells:
        for budget in BUDGETS:
            candidates = [
                row
                for row in rows
                if row["sample_id"] == sample_id
                and row["layer_index"] == layer_index
                and row["exact_group_count"] == budget
                and row["method"] in HEADWISE_METHODS
            ]
            best = min(candidates, key=lambda row: float(row["visual_relative_l2"]))
            envelope_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_position": sample_position,
                    "layer_index": layer_index,
                    "exact_group_count": budget,
                    "selected_method": best["method"],
                    "visual_relative_l2": best["visual_relative_l2"],
                    "visual_worst_head_relative_l2": best[
                        "visual_worst_head_relative_l2"
                    ],
                    "full_relative_l2": best["full_relative_l2"],
                }
            )
    summary: dict[str, dict[str, float | int]] = {}
    for budget in BUDGETS:
        selected = [row for row in envelope_rows if row["exact_group_count"] == budget]
        visual = np.asarray([row["visual_relative_l2"] for row in selected])
        full = np.asarray([row["full_relative_l2"] for row in selected])
        summary[str(budget)] = {
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
        }
    return envelope_rows, summary


def classify_outcome(
    summaries: dict[str, dict[str, object]],
    envelope_summary: dict[str, dict[str, float | int]],
) -> tuple[str, dict[str, float | bool]]:
    baseline = summaries["shared_exact_local"]["196"]
    envelope = envelope_summary["196"]
    capacity_pass = (
        envelope["visual_mean"] <= 0.01
        and envelope["visual_p95"] <= 0.02
        and envelope["visual_worst"] <= 0.05
        and envelope["full_mean"] <= 0.005
        and envelope["full_p95"] <= 0.01
    )
    relative_improvement = 1.0 - float(envelope["visual_mean"]) / float(
        baseline["visual_mean"]
    )
    diagnostics = {
        "capacity_pass": bool(capacity_pass),
        "visual_mean_relative_improvement": relative_improvement,
    }
    if capacity_pass:
        return "HEADWISE_SUPPORT_CAPACITY_PASS", diagnostics
    if relative_improvement >= 0.25:
        return "HEADWISE_SUPPORT_PARTIAL", diagnostics
    return "NO_HEADWISE_SUPPORT_CAPACITY", diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_shared_control(
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
        if row["method"] == "exact_local_score"
    }
    shared_rows = [row for row in rows if row["method"] == "shared_exact_local"]
    for row in shared_rows:
        key = (
            str(row["sample_id"]),
            int(row["layer_index"]),
            int(row["exact_group_count"]),
        )
        reference = expected[key]
        for current_key, expected_key in (
            ("visual_relative_l2", "visual_relative_l2"),
            ("visual_worst_head_relative_l2", "visual_worst_head_relative_l2"),
            ("full_relative_l2", "full_relative_l2"),
        ):
            if abs(float(row[current_key]) - float(reference[expected_key])) > 1e-8:
                raise RuntimeError("shared exact-local control did not reproduce")


def main() -> int:
    args = parse_args()
    previous = json.loads(args.query_summary.read_text(encoding="utf-8"))
    if previous["decision"] != "NO_REGISTERED_QUERY_FIXED_MEASURE_PATH":
        raise ValueError("query-fixed prerequisite decision changed")
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
            state = grouped_measure_state(
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
            replay_maximum = max(replay_maximum, state.exact_projected_relative_error)
            local_bound_violations += (
                state.local_z_bound_violations + state.local_n_bound_violations
            )
            orders = selector_orders(state)
            for method in METHODS:
                for budget in BUDGETS:
                    if orders[method].ndim == 1:
                        selected_indices = orders[method][:budget]
                    else:
                        selected_indices = orders[method][:, :budget]
                    metrics = evaluate_support(
                        state,
                        selected_indices,
                        nonvisual_z=nonvisual_z,
                        nonvisual_n=nonvisual_n,
                    )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "sample_position": sample_position,
                            "layer_index": layer_index,
                            "method": method,
                            "exact_group_count": budget,
                            "visual_token_retention": (392 + 3 * budget) / 1568,
                            **metrics,
                        }
                    )
        print(
            json.dumps(
                {
                    "event": "headwise_support_sample_ok",
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
        raise RuntimeError("headwise support capture or local-bound guard failed")
    summaries = summarize_rows(rows)
    validate_shared_control(rows, args.query_rows)
    envelope_rows, envelope_summary = build_headwise_envelope(rows)
    decision, diagnostics = classify_outcome(summaries, envelope_summary)
    write_csv(args.out_dir / "headwise_support_rows.csv", rows)
    write_csv(args.out_dir / "headwise_support_envelope_rows.csv", envelope_rows)
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
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "maximum_replay_error": replay_maximum,
        "local_bound_violations": local_bound_violations,
        "shared_control_reproduced": True,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "headwise_envelope": envelope_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Target-visible per-head support capacity on exposed positions 73-96. "
            "It is not a deployable router, global subset optimum, reader endpoint, "
            "latency result, or hidden-data claim."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
