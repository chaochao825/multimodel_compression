"""Screen residual-width-guided exact writes on LongLive Q/K/V captures.

The residual-width selector evaluates same-record dense AV defects and is an
explicitly non-deployable singleton-marginal oracle. Its purpose is to test
whether changing the write objective is worth a later Q/K/V-only predictor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import torch

from ar_video_residual_memory_core import (
    adaptive_rank_projection,
    arithmetic_reduction,
    attention_from_representatives,
    build_representatives,
    dense_attention,
    make_recency_plan,
    phase_align_keys_for_temporal_summaries,
    select_residual_event_tiles,
)
from ar_video_residual_width_core import (
    TileCandidate,
    enumerate_summary_tiles,
    event_mask_from_indices,
    indices_from_event_mask,
    jaccard_similarity,
    normalized_tail_objective,
    select_top_indices,
    selection_budget,
    selection_signature,
)
from probe_ar_video_residual_memory import (
    capture_identity,
    error_components,
    sha256_file,
    validate_manifests,
    validate_metadata,
)


SELECTOR_ACCESS = {
    "kv_deviation": "runtime_qkv_no_dense_target",
    "dense_attention_mass_oracle": "same_record_dense_attention",
    "value_leverage_oracle": "same_record_dense_attention_and_value",
    "residual_width_singleton_oracle": "same_record_dense_av_defect",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--max-captures", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def required_capture_keys(protocol: dict[str, Any]) -> set[tuple[str, int, int, int]]:
    scope = protocol["scope"]
    return {
        (
            str(prompt_id),
            int(cell["layer"]),
            int(cell["current_start_frame"]),
            int(cell["denoising_call_index"]),
        )
        for prompt_id in scope["prompt_ids"]
        for cell in scope["cells"]
    }


def protocol_capture_key(metadata: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(metadata["prompt_id"]),
        int(metadata["layer"]),
        int(metadata["current_start_frame"]),
        int(metadata["denoising_call_index"]),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dense_selector_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidates: list[TileCandidate],
    summary_groups: Iterable[Iterable[int]],
    spatial_tokens: int,
    query_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidate tiles by dense mass and attention-weighted V deviation."""

    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    frames = key.shape[0] // spatial_tokens
    key_view = key.reshape(frames, spatial_tokens, key.shape[1], key.shape[2])
    value_view = value.reshape(frames, spatial_tokens, value.shape[1], value.shape[2])
    value_deviation = torch.zeros(
        (frames, spatial_tokens, value.shape[1]), device=value.device, dtype=torch.float32
    )
    for group in summary_groups:
        indices = torch.tensor(tuple(group), device=value.device, dtype=torch.long)
        group_value = value_view.index_select(0, indices).float()
        mean_value = group_value.mean(dim=0, keepdim=True)
        deviation = (group_value - mean_value).square().sum(dim=-1).sqrt()
        value_deviation.index_copy_(0, indices, deviation)
    value_deviation_flat = value_deviation.reshape(-1, value.shape[1]).transpose(0, 1)

    mass = torch.zeros(len(candidates), device=query.device, dtype=torch.float64)
    leverage = torch.zeros_like(mass)
    key_float = key.float()
    scale = 1.0 / math.sqrt(query.shape[-1])
    for start in range(0, query.shape[0], query_chunk_size):
        query_float = query[start : start + query_chunk_size].float()
        logits = torch.einsum("qhd,khd->hqk", query_float, key_float) * scale
        weights = torch.softmax(logits, dim=-1)
        token_mass = weights.sum(dim=(0, 1)).double()
        token_leverage = (
            weights * value_deviation_flat[:, None, :]
        ).sum(dim=(0, 1)).double()
        for candidate in candidates:
            begin = candidate.frame * spatial_tokens + candidate.start
            end = candidate.frame * spatial_tokens + candidate.end
            mass[candidate.index] += token_mass[begin:end].sum()
            leverage[candidate.index] += token_leverage[begin:end].sum()
    return mass.float(), leverage.float()


def estimate_with_mask(
    query: torch.Tensor,
    key_view: torch.Tensor,
    value_view: torch.Tensor,
    plan,
    summary_key: torch.Tensor,
    event_mask: torch.Tensor,
    query_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    representatives = build_representatives(
        key_view,
        value_view,
        plan,
        event_mask=event_mask,
        summary_key=summary_key,
    )
    estimate = attention_from_representatives(
        query, representatives, query_chunk_size=query_chunk_size
    )
    return estimate, representatives.token_count


def singleton_width_scores(
    query: torch.Tensor,
    key_view: torch.Tensor,
    value_view: torch.Tensor,
    target: torch.Tensor,
    base_estimate: torch.Tensor,
    plan,
    summary_key: torch.Tensor,
    candidates: list[TileCandidate],
    rank: int,
    query_chunk_size: int,
) -> torch.Tensor:
    """Compute exact singleton marginal reductions of normalized tail energy."""

    base_objective = normalized_tail_objective(target - base_estimate, target, rank)
    scores = torch.empty(len(candidates), device=query.device, dtype=torch.float32)
    frames, spatial_tokens = key_view.shape[:2]
    for candidate in candidates:
        mask = event_mask_from_indices(
            candidates,
            [candidate.index],
            frames,
            spatial_tokens,
            query.device,
        )
        estimate, _ = estimate_with_mask(
            query,
            key_view,
            value_view,
            plan,
            summary_key,
            mask,
            query_chunk_size,
        )
        objective = normalized_tail_objective(target - estimate, target, rank)
        scores[candidate.index] = (base_objective - objective).float()
    return scores


def append_rows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    selector: str,
    fraction: float,
    selected_indices: list[int],
    target: torch.Tensor,
    estimate: torch.Tensor,
    rank: int,
    dense_tokens: int,
    representative_tokens: int,
    parity: torch.Tensor,
    selector_seconds: float,
) -> None:
    variants = [
        ("none", 0, estimate),
        (
            "adaptive_rank_oracle",
            rank,
            estimate + adaptive_rank_projection(target - estimate, rank),
        ),
    ]
    for correction, correction_rank, corrected in variants:
        error, numerator_sq, denominator_sq = error_components(target, corrected)
        for local_head, head_index in enumerate(metadata["head_indices"]):
            rows.append(
                {
                    "protocol_id": metadata["protocol_id"],
                    "candidate_protocol_id": "ar-video-residual-width-write-screen-v1",
                    "prompt_id": metadata["prompt_id"],
                    "split": metadata["prompt_split"],
                    "seed": int(metadata["seed"]),
                    "layer": int(metadata["layer"]),
                    "current_start_frame": int(metadata["current_start_frame"]),
                    "denoising_call_index": int(metadata["denoising_call_index"]),
                    "denoising_timestep": int(metadata["denoising_timestep"]),
                    "head_index": int(head_index),
                    "selector": selector,
                    "selector_access": SELECTOR_ACCESS.get(selector, "none"),
                    "event_fraction": fraction,
                    "selected_tile_count": len(selected_indices),
                    "selection_signature": selection_signature(selected_indices),
                    "correction": correction,
                    "rank": correction_rank,
                    "deployable": selector == "kv_deviation" and correction == "none",
                    "relative_av_l2": float(error[local_head]),
                    "numerator_sq": float(numerator_sq[local_head]),
                    "denominator_sq": float(denominator_sq[local_head]),
                    "dense_reference_parity": float(parity[local_head]),
                    "dense_key_tokens": dense_tokens,
                    "representative_tokens": representative_tokens,
                    "arithmetic_reduction": arithmetic_reduction(
                        dense_tokens, representative_tokens
                    ),
                    "selector_seconds_per_capture": selector_seconds,
                    "measured_kernel_speedup": "",
                }
            )


@torch.inference_mode()
def evaluate_capture(
    payload: dict[str, Any],
    source_protocol: dict[str, Any],
    protocol: dict[str, Any],
    device: torch.device,
    query_chunk_size: int,
    rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> None:
    metadata = payload["metadata"]
    validate_metadata(metadata, payload, source_protocol)
    query = payload["query"].to(device)
    key = payload["key"].to(device)
    value = payload["value"].to(device)
    target = payload["dense_output"].to(device)
    dense_math = dense_attention(query, key, value, query_chunk_size)
    parity, _, _ = error_components(target, dense_math)

    scope = protocol["scope"]
    frame_seq_len = int(source_protocol["capture"]["frame_seq_len"])
    frames = key.shape[0] // frame_seq_len
    key_view = key.reshape(frames, frame_seq_len, key.shape[1], key.shape[2])
    value_view = value.reshape(frames, frame_seq_len, value.shape[1], value.shape[2])
    plan = make_recency_plan(
        num_frames=frames,
        sink_frames=int(scope["exact_sink_frames"]),
        recent_frames=int(scope["exact_recent_frames"]),
        max_summary_groups=int(scope["summary_groups"]),
    )
    grid = metadata["grid_sizes"]
    if len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("capture requires exactly one [frames, height, width] grid")
    summary_key = phase_align_keys_for_temporal_summaries(
        key=key_view,
        plan=plan,
        absolute_frame_ids=metadata["key_frame_ids"],
        height=int(grid[0][1]),
        width=int(grid[0][2]),
        rope_freqs=payload["rope_freqs"].to(device),
    )
    candidates = enumerate_summary_tiles(
        plan.summary_groups,
        spatial_tokens=frame_seq_len,
        tile_size=int(scope["event_tile_size"]),
    )
    empty_mask = torch.zeros((frames, frame_seq_len), dtype=torch.bool, device=device)
    base_estimate, base_tokens = estimate_with_mask(
        query,
        key_view,
        value_view,
        plan,
        summary_key,
        empty_mask,
        query_chunk_size,
    )
    append_rows(
        rows,
        metadata,
        "no_event",
        0.0,
        [],
        target,
        base_estimate,
        int(scope["low_rank_residual_rank"]),
        int(key.shape[0]),
        base_tokens,
        parity,
        0.0,
    )

    _synchronize(device)
    started = time.perf_counter()
    mass_scores, leverage_scores = dense_selector_scores(
        query,
        key,
        value,
        candidates,
        plan.summary_groups,
        frame_seq_len,
        query_chunk_size,
    )
    _synchronize(device)
    dense_selector_seconds = time.perf_counter() - started

    _synchronize(device)
    started = time.perf_counter()
    width_scores = singleton_width_scores(
        query,
        key_view,
        value_view,
        target,
        base_estimate,
        plan,
        summary_key,
        candidates,
        int(scope["low_rank_residual_rank"]),
        query_chunk_size,
    )
    _synchronize(device)
    width_seconds = time.perf_counter() - started

    estimate_cache: dict[tuple[int, ...], tuple[torch.Tensor, int]] = {}
    score_map = {
        "dense_attention_mass_oracle": mass_scores,
        "value_leverage_oracle": leverage_scores,
        "residual_width_singleton_oracle": width_scores,
    }
    selected_by_fraction: dict[float, dict[str, list[int]]] = defaultdict(dict)
    for fraction in map(float, scope["event_tile_fractions"]):
        budget = selection_budget(len(candidates), fraction)
        kv_mask = select_residual_event_tiles(
            key=summary_key,
            value=value_view,
            plan=plan,
            tile_size=int(scope["event_tile_size"]),
            tile_fraction=fraction,
            query=query,
        )
        selected_by_fraction[fraction]["kv_deviation"] = indices_from_event_mask(
            candidates, kv_mask
        )
        for selector, scores in score_map.items():
            selected_by_fraction[fraction][selector] = select_top_indices(scores, budget)

        for selector in protocol["selectors"]:
            selected = selected_by_fraction[fraction][selector]
            cache_key = tuple(sorted(selected))
            if cache_key not in estimate_cache:
                mask = event_mask_from_indices(
                    candidates, selected, frames, frame_seq_len, device
                )
                estimate_cache[cache_key] = estimate_with_mask(
                    query,
                    key_view,
                    value_view,
                    plan,
                    summary_key,
                    mask,
                    query_chunk_size,
                )
            estimate, representative_tokens = estimate_cache[cache_key]
            seconds = (
                width_seconds
                if selector == "residual_width_singleton_oracle"
                else dense_selector_seconds
                if selector in {
                    "dense_attention_mass_oracle",
                    "value_leverage_oracle",
                }
                else 0.0
            )
            append_rows(
                rows,
                metadata,
                selector,
                fraction,
                selected,
                target,
                estimate,
                int(scope["low_rank_residual_rank"]),
                int(key.shape[0]),
                representative_tokens,
                parity,
                seconds,
            )

        for left_index, left in enumerate(protocol["selectors"]):
            for right in protocol["selectors"][left_index + 1 :]:
                selections.append(
                    {
                        "prompt_id": metadata["prompt_id"],
                        "split": metadata["prompt_split"],
                        "seed": int(metadata["seed"]),
                        "layer": int(metadata["layer"]),
                        "current_start_frame": int(metadata["current_start_frame"]),
                        "denoising_call_index": int(metadata["denoising_call_index"]),
                        "event_fraction": fraction,
                        "left_selector": left,
                        "right_selector": right,
                        "jaccard": jaccard_similarity(
                            selected_by_fraction[fraction][left],
                            selected_by_fraction[fraction][right],
                        ),
                    }
                )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes = {
        "all": {"calibration", "validation", "test"},
        "calibration": {"calibration"},
        "held_out": {"validation", "test"},
    }
    output = []
    for scope, splits in scopes.items():
        grouped: defaultdict[tuple[str, float, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["split"] in splits:
                grouped[
                    (
                        str(row["selector"]),
                        float(row["event_fraction"]),
                        str(row["correction"]),
                        int(row["rank"]),
                    )
                ].append(row)
        for (selector, fraction, correction, rank), group in sorted(grouped.items()):
            numerator = sum(float(row["numerator_sq"]) for row in group)
            denominator = sum(float(row["denominator_sq"]) for row in group)
            errors = [float(row["relative_av_l2"]) for row in group]
            reductions = [float(row["arithmetic_reduction"]) for row in group]
            output.append(
                {
                    "scope": scope,
                    "selector": selector,
                    "event_fraction": fraction,
                    "correction": correction,
                    "rank": rank,
                    "aggregate_relative_av_l2": math.sqrt(numerator / denominator),
                    "mean_head_relative_av_l2": statistics.fmean(errors),
                    "p95_head_relative_av_l2": percentile(errors, 0.95),
                    "worst_head_relative_av_l2": max(errors),
                    "minimum_arithmetic_reduction": min(reductions),
                    "mean_arithmetic_reduction": statistics.fmean(reductions),
                    "head_records": len(group),
                }
            )
    return output


def find_summary(
    summaries: list[dict[str, Any]],
    scope: str,
    selector: str,
    fraction: float,
    correction: str,
    rank: int,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in summaries
        if item["scope"] == scope
        and item["selector"] == selector
        and math.isclose(float(item["event_fraction"]), fraction)
        and item["correction"] == correction
        and int(item["rank"]) == rank
    ]
    if len(matches) > 1:
        raise AssertionError("summary key is not unique")
    return matches[0] if matches else None


def decide(
    protocol: dict[str, Any], summaries: list[dict[str, Any]], complete: bool
) -> dict[str, Any]:
    scope = protocol["scope"]
    evaluation = protocol["evaluation"]
    rank = int(scope["low_rank_residual_rank"])
    comparisons = []
    strict_pass = False
    mechanism_pass = False
    arithmetic_pass = False
    for fraction in map(float, scope["event_tile_fractions"]):
        width = find_summary(
            summaries,
            "held_out",
            "residual_width_singleton_oracle",
            fraction,
            "adaptive_rank_oracle",
            rank,
        )
        baselines = [
            find_summary(
                summaries,
                "held_out",
                selector,
                fraction,
                "adaptive_rank_oracle",
                rank,
            )
            for selector in (
                "kv_deviation",
                "dense_attention_mass_oracle",
                "value_leverage_oracle",
            )
        ]
        baselines = [item for item in baselines if item is not None]
        if width is None or not baselines:
            continue
        strongest = min(baselines, key=lambda item: item["aggregate_relative_av_l2"])
        improvement = 1.0 - width["aggregate_relative_av_l2"] / max(
            strongest["aggregate_relative_av_l2"], 1e-24
        )
        quality = (
            width["aggregate_relative_av_l2"]
            <= float(evaluation["aggregate_oracle_gate"])
            and width["worst_head_relative_av_l2"]
            <= float(evaluation["worst_oracle_gate"])
        )
        mechanism = improvement >= float(
            evaluation["minimum_relative_improvement_over_strongest_baseline"]
        )
        arithmetic = width["minimum_arithmetic_reduction"] >= float(
            evaluation["minimum_arithmetic_reduction"]
        )
        strict_pass = strict_pass or quality
        mechanism_pass = mechanism_pass or mechanism
        arithmetic_pass = arithmetic_pass or arithmetic
        comparisons.append(
            {
                "event_fraction": fraction,
                "width": width,
                "strongest_baseline": strongest,
                "relative_improvement": improvement,
                "strict_quality_pass": quality,
                "mechanism_improvement_pass": mechanism,
                "arithmetic_screen_pass": arithmetic,
            }
        )
    decision: dict[str, Any] = {
        "complete": complete,
        "comparisons": comparisons,
        "strict_quality_pass": strict_pass,
        "mechanism_improvement_pass": mechanism_pass,
        "arithmetic_screen_pass": arithmetic_pass,
        "deployable": False,
        "measured_speedup_claim_allowed": False,
        "note": (
            "Residual-width, dense-mass, and value-leverage selectors are same-record "
            "oracles. A pass opens predictor work only and cannot support a speed claim."
        ),
    }
    if not complete:
        decision.update(classification="incomplete", action="finish_frozen_screen")
    elif strict_pass and arithmetic_pass:
        decision.update(
            classification="representation_pass",
            action="fit_qkv_only_residual_width_predictor",
        )
    elif mechanism_pass:
        decision.update(
            classification="mechanism_signal_only",
            action="do_not_build_kernel_without_strict_quality_pass",
        )
    else:
        decision.update(
            classification="null",
            action="stop_residual_width_memory_direction",
        )
    return decision


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_captures is not None and args.max_captures <= 0:
        raise ValueError("max captures must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        args.output_dir / name
        for name in ("metrics.csv", "selection_overlap.csv", "summary.json", "decision.json")
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite result artifacts: {existing}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    if protocol["source_protocol_id"] != source_protocol["protocol_id"]:
        raise ValueError("candidate/source protocol IDs do not match")
    validate_manifests(
        args.capture_dir, source_protocol, sha256_file(args.source_protocol)
    )
    required = required_capture_keys(protocol)
    selected: dict[tuple[str, int, int, int], Path] = {}
    for path in sorted(args.capture_dir.glob("**/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "metadata" not in payload:
            raise ValueError(f"invalid capture payload: {path}")
        validate_metadata(payload["metadata"], payload, source_protocol)
        key = protocol_capture_key(payload["metadata"])
        if key in required:
            if key in selected:
                raise ValueError(f"duplicate required capture: {key}")
            selected[key] = path
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"missing {len(missing)} required captures; first={missing[0]}")
    selected_items = sorted(selected.items())
    if args.max_captures is not None:
        if not args.allow_partial:
            raise ValueError("max captures requires --allow-partial")
        selected_items = selected_items[: args.max_captures]

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested but CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)
    rows: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for index, (identity, path) in enumerate(selected_items, start=1):
        print(f"[evaluate {index}/{len(selected_items)}] {identity}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        evaluate_capture(
            payload,
            source_protocol,
            protocol,
            device,
            args.query_chunk_size,
            rows,
            overlaps,
        )
    summaries = summarize(rows)
    complete = len(selected_items) == len(required)
    decision = decide(protocol, summaries, complete)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "selection_overlap.csv", overlaps)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": sha256_file(args.protocol),
                "source_protocol_id": source_protocol["protocol_id"],
                "source_protocol_sha256": sha256_file(args.source_protocol),
                "required_capture_count": len(required),
                "evaluated_capture_count": len(selected_items),
                "complete": complete,
                "capture_paths": [str(path) for _, path in selected_items],
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
