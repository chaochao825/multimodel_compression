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
from probe_vsi_onevision_query_fixed_measure_remainder import SelectedLayerCapture
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (
    GaussianComponents,
    build_gaussian_components,
    hierarchical_group_offsets,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


LAYERS = (0, 13, 27)
TOPOLOGIES = ("spatial_7x7", "temporal2_spatial_7x7")
SELECTORS = ("centroid_score", "quest_box_bound", "exact_mass", "oracle_local")
EXACT_FRACTIONS = (0.125, 0.25, 0.5, 0.625, 0.75, 1.0)
DEPLOYABLE_SELECTORS = ("centroid_score", "quest_box_bound")
ORACLE_SELECTORS = ("exact_mass", "oracle_local")
ROLE = "exposed_query_fixed_progressive_exact_pages"


@dataclass
class PageState:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_group_z: torch.Tensor
    exact_group_n: torch.Tensor
    centroid_priority: torch.Tensor
    quest_priority: torch.Tensor
    exact_mass_priority: torch.Tensor
    oracle_priority: torch.Tensor
    upper_log_group_z: torch.Tensor
    value_norm_max: torch.Tensor
    maximum_bound_violation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--positive-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_page_state(components: GaussianComponents) -> PageState:
    query = components.query_scaled[:, None, :]
    box_corner = torch.where(
        query >= 0,
        components.key_max,
        components.key_min,
    )
    upper_score = (query * box_corner).sum(dim=-1)
    upper_log_group_z = (
        math.log(components.group_size) + upper_score - components.maximum
    )
    centroid_priority = (query * components.mean_key).sum(dim=-1)
    visual_z = components.exact_group_z.sum(dim=1)
    normalized_z = components.exact_group_z / visual_z.unsqueeze(-1)
    normalized_n = components.exact_group_n / visual_z[:, None, None]
    local_output = normalized_n - (
        components.exact_visual_output[:, None, :] * normalized_z.unsqueeze(-1)
    )
    tiny = torch.finfo(components.exact_group_z.dtype).tiny
    exact_log_group_z = torch.log(components.exact_group_z.clamp_min(tiny))
    maximum_bound_violation = float(
        (exact_log_group_z - upper_log_group_z).max().item()
    )
    if maximum_bound_violation > 2e-5:
        raise RuntimeError("Quest box score failed to upper-bound exact page mass")
    return PageState(
        exact_visual_output=components.exact_visual_output,
        exact_full_output=components.exact_full_output,
        exact_group_z=components.exact_group_z,
        exact_group_n=components.exact_group_n,
        centroid_priority=centroid_priority,
        quest_priority=upper_log_group_z,
        exact_mass_priority=components.exact_group_z,
        oracle_priority=torch.linalg.vector_norm(local_output, dim=-1),
        upper_log_group_z=upper_log_group_z,
        value_norm_max=components.visual_value_norm_max,
        maximum_bound_violation=maximum_bound_violation,
    )


def selector_orders(state: PageState) -> dict[str, torch.Tensor]:
    priorities = {
        "centroid_score": state.centroid_priority,
        "quest_box_bound": state.quest_priority,
        "exact_mass": state.exact_mass_priority,
        "oracle_local": state.oracle_priority,
    }
    return {
        name: torch.argsort(values, dim=1, descending=True, stable=True)
        for name, values in priorities.items()
    }


def evaluate_exact_pages(
    state: PageState,
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
    selected.scatter_(1, selected_indices, True)
    visual_z = torch.where(selected, state.exact_group_z, 0.0).sum(dim=1)
    visual_n = torch.where(selected.unsqueeze(-1), state.exact_group_n, 0.0).sum(dim=1)
    if bool((visual_z <= 0).any().item()):
        raise RuntimeError("progressive exact path selected no positive visual mass")
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(-1)

    exact_visual_z = state.exact_group_z.sum(dim=1)
    selected_mass_fraction = visual_z / exact_visual_z
    head_error = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output, dim=-1
    ) / torch.linalg.vector_norm(state.exact_visual_output, dim=-1).clamp_min(1e-12)
    visual_relative = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output
    ) / torch.linalg.vector_norm(state.exact_visual_output).clamp_min(1e-12)
    full_relative = torch.linalg.vector_norm(
        full_output - state.exact_full_output
    ) / torch.linalg.vector_norm(state.exact_full_output).clamp_min(1e-12)

    unselected_upper_log = torch.where(
        ~selected,
        state.upper_log_group_z,
        torch.full_like(state.upper_log_group_z, -torch.inf),
    )
    log_upper_tail = torch.logsumexp(unselected_upper_log, dim=1)
    log_selected_mass = torch.log(visual_z)
    upper_tail_fraction = torch.sigmoid(log_upper_tail - log_selected_mass)
    exact_output_norm = torch.linalg.vector_norm(
        state.exact_visual_output, dim=-1
    ).clamp_min(1e-12)
    relative_error_bound = (
        2.0 * state.value_norm_max * upper_tail_fraction / exact_output_norm
    )
    certificate_coverage = (head_error <= relative_error_bound + 1e-5).float()
    if float(certificate_coverage.min().item()) != 1.0:
        raise RuntimeError("Quest tail certificate failed to cover exact head error")

    exact_tail = torch.where(~selected, state.exact_group_z, 0.0).sum(dim=1)
    tiny = torch.finfo(exact_tail.dtype).tiny
    log10_bound_looseness = (
        log_upper_tail - torch.log(exact_tail.clamp_min(tiny))
    ) / math.log(10.0)
    log10_bound_looseness = torch.where(
        exact_tail > 0,
        log10_bound_looseness,
        torch.zeros_like(log10_bound_looseness),
    )
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
        "selected_visual_mass_mean": float(selected_mass_fraction.mean().item()),
        "selected_visual_mass_min": float(selected_mass_fraction.min().item()),
        "upper_tail_fraction_mean": float(upper_tail_fraction.mean().item()),
        "upper_tail_fraction_max": float(upper_tail_fraction.max().item()),
        "certificate_relative_bound_mean": float(relative_error_bound.mean().item()),
        "certificate_relative_bound_max": float(relative_error_bound.max().item()),
        "certificate_coverage": float(certificate_coverage.mean().item()),
        "tail_bound_log10_looseness_mean": float(log10_bound_looseness.mean().item()),
        "tail_bound_log10_looseness_max": float(log10_bound_looseness.max().item()),
    }


def read_costs(
    *,
    selector: str,
    group_count: int,
    group_size: int,
    head_dim: int,
    exact_group_count: int,
) -> tuple[int, int, int, float, float]:
    dense = group_count * group_size * 2 * head_dim
    exact = exact_group_count * group_size * 2 * head_dim
    if selector == "centroid_score":
        metadata = group_count * head_dim
    elif selector == "quest_box_bound":
        metadata = group_count * 2 * head_dim
    elif selector in ORACLE_SELECTORS:
        metadata = dense
    else:
        raise ValueError(f"unregistered selector: {selector}")
    active_ratio = dense / (metadata + exact)
    leaf_only_ratio = dense / exact
    return dense, metadata, exact, active_ratio, leaf_only_ratio


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["topology"]),
                str(row["selector"]),
                float(row["exact_fraction"]),
            )
            for row in rows
        }
    )
    summaries: list[dict[str, object]] = []
    for topology, selector, exact_fraction in keys:
        selected = [
            row
            for row in rows
            if row["topology"] == topology
            and row["selector"] == selector
            and row["exact_fraction"] == exact_fraction
        ]
        visual = np.asarray([float(row["visual_relative_l2"]) for row in selected])
        full = np.asarray([float(row["full_relative_l2"]) for row in selected])
        summaries.append(
            {
                "topology": topology,
                "selector": selector,
                "exact_fraction": exact_fraction,
                "exact_group_count": int(selected[0]["exact_group_count"]),
                "cell_count": len(selected),
                "active_read_ratio": float(selected[0]["active_read_ratio"]),
                "leaf_only_read_ratio": float(selected[0]["leaf_only_read_ratio"]),
                "visual_mean": float(visual.mean()),
                "visual_p95": float(np.quantile(visual, 0.95)),
                "visual_worst": float(visual.max()),
                "visual_worst_head": max(
                    float(row["visual_worst_head_relative_l2"]) for row in selected
                ),
                "full_mean": float(full.mean()),
                "full_p95": float(np.quantile(full, 0.95)),
                "full_worst": float(full.max()),
                "selected_visual_mass_mean": float(
                    np.mean(
                        [float(row["selected_visual_mass_mean"]) for row in selected]
                    )
                ),
                "upper_tail_fraction_mean": float(
                    np.mean(
                        [float(row["upper_tail_fraction_mean"]) for row in selected]
                    )
                ),
                "tail_bound_log10_looseness_mean": float(
                    np.mean(
                        [
                            float(row["tail_bound_log10_looseness_mean"])
                            for row in selected
                        ]
                    )
                ),
            }
        )
    return summaries


def quality_pass(summary: dict[str, object], *, oracle: bool) -> bool:
    if int(summary["cell_count"]) != 72:
        return False
    if oracle:
        return (
            float(summary["visual_mean"]) <= 0.005
            and float(summary["visual_p95"]) <= 0.01
            and float(summary["visual_worst"]) <= 0.02
            and float(summary["full_mean"]) <= 0.0025
            and float(summary["full_p95"]) <= 0.005
        )
    return (
        float(summary["visual_mean"]) <= 0.01
        and float(summary["visual_p95"]) <= 0.02
        and float(summary["visual_worst"]) <= 0.05
        and float(summary["full_mean"]) <= 0.005
        and float(summary["full_p95"]) <= 0.01
    )


def classify_outcome(
    summaries: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    deployable = [
        row
        for row in summaries
        if row["selector"] in DEPLOYABLE_SELECTORS
        and float(row["exact_fraction"]) <= 0.25
        and float(row["active_read_ratio"]) >= 2.0
        and quality_pass(row, oracle=False)
    ]
    oracle = [
        row
        for row in summaries
        if row["selector"] in ORACLE_SELECTORS
        and float(row["exact_fraction"]) <= 0.25
        and float(row["leaf_only_read_ratio"]) >= 2.0
        and quality_pass(row, oracle=True)
    ]
    eligible = [
        row
        for row in summaries
        if float(row["exact_fraction"]) <= 0.25
        and float(row["leaf_only_read_ratio"]) >= 2.0
    ]
    diagnostics = {
        "deployable_pass_count": len(deployable),
        "oracle_pass_count": len(oracle),
        "best_eligible": min(eligible, key=lambda row: float(row["visual_mean"])),
    }
    if deployable:
        return "PROGRESSIVE_EXACT_PAGE_PATH", diagnostics
    if oracle:
        return "PROGRESSIVE_EXACT_PAGE_CAPACITY_ONLY", diagnostics
    return "NO_PROGRESSIVE_EXACT_PAGE_PATH", diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty progressive exact page rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    previous = json.loads(args.positive_summary.read_text(encoding="utf-8"))
    if previous["decision"] != "NO_POSITIVE_GAUSSIAN_MEASURE_PATH":
        raise ValueError("positive-Gaussian prerequisite decision changed")
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered Gate is restricted to calibration positions 73-96")
    if args.frame_budget != 8:
        raise ValueError("registered frame budget changed")

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
    maximum_replay_error = 0.0
    maximum_bound_violation = -math.inf
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
            offsets = hierarchical_group_offsets(
                frame_count=reference.shape[0],
                token_count=reference.shape[1],
                topology=topology,
                device=device,
            )
            group_count, group_size = offsets.shape
            for layer_index in LAYERS:
                components = build_gaussian_components(
                    capture.captures[layer_index],
                    visual_start=visual_start,
                    visual_token_count=visual_token_count,
                    group_offsets=offsets,
                    max_rank=16,
                )
                state = build_page_state(components)
                orders = selector_orders(state)
                maximum_replay_error = max(
                    maximum_replay_error, components.replay_error
                )
                maximum_bound_violation = max(
                    maximum_bound_violation, state.maximum_bound_violation
                )
                head_dim = components.exact_group_n.shape[-1]
                for selector in SELECTORS:
                    for exact_fraction in EXACT_FRACTIONS:
                        exact_group_count = int(round(group_count * exact_fraction))
                        metrics = evaluate_exact_pages(
                            state,
                            orders[selector][:, :exact_group_count],
                            nonvisual_z=components.nonvisual_z,
                            nonvisual_n=components.nonvisual_n,
                        )
                        dense, metadata, exact, active_ratio, leaf_ratio = read_costs(
                            selector=selector,
                            group_count=group_count,
                            group_size=group_size,
                            head_dim=head_dim,
                            exact_group_count=exact_group_count,
                        )
                        rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "sample_position": sample_position,
                                "layer_index": layer_index,
                                "topology": topology,
                                "group_count": group_count,
                                "group_size": group_size,
                                "selector": selector,
                                "exact_fraction": exact_fraction,
                                "exact_group_count": exact_group_count,
                                "dense_read_floats_per_head": dense,
                                "metadata_read_floats_per_head": metadata,
                                "exact_read_floats_per_head": exact,
                                "active_read_ratio": active_ratio,
                                "leaf_only_read_ratio": leaf_ratio,
                                **metrics,
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "progressive_exact_page_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_replay_error": maximum_replay_error,
                    "maximum_bound_violation": maximum_bound_violation,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if maximum_replay_error > 1e-4:
        raise RuntimeError("captured Q/K/V did not reconstruct attention output")
    full_exact_rows = [row for row in rows if row["exact_fraction"] == 1.0]
    if (
        not full_exact_rows
        or max(
            max(
                float(row["visual_relative_l2"]),
                float(row["full_relative_l2"]),
            )
            for row in full_exact_rows
        )
        > 1e-7
    ):
        raise RuntimeError("full exact pages did not reproduce the dense measure")
    summaries = summarize_rows(rows)
    decision, diagnostics = classify_outcome(summaries)
    write_csv(args.out_dir / "progressive_exact_page_rows.csv", rows)
    write_csv(args.out_dir / "progressive_exact_page_summary.csv", summaries)
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
        "topologies": list(TOPOLOGIES),
        "selectors": list(SELECTORS),
        "exact_fractions": list(EXACT_FRACTIONS),
        "maximum_replay_error": maximum_replay_error,
        "maximum_bound_violation": maximum_bound_violation,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fixed-query, single-layer progressive exact-page diagnostic on "
            "exposed calibration positions 73-96. Oracle selectors and read "
            "ratios are capacity/arithmetic probes, not reader or latency claims."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
