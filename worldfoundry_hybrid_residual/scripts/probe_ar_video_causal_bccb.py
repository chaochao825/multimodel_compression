"""Evaluate causal spatial BCCB/Toeplitz representations on LongLive captures.

The record-generated variants may use current Q/K but never dense AV targets.
Adaptive low-rank corrections and frozen-basis oracle coefficients do use the
dense defect and are explicitly capacity diagnostics, not deployable methods.
The FFT cost model excludes dynamic-kernel construction and is therefore only
an optimistic arithmetic screen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from ar_video_causal_bccb_core import (
    build_spatial_kernel_bank,
    exact_sink_recent_frames,
    fft_arithmetic_reduction,
    make_captured_query_layout,
    pool_kernel_bank_by_relative_frame,
    structured_attention_from_kernel_bank,
)
from ar_video_residual_memory_core import (
    adaptive_rank_projection,
    dense_attention,
    fit_output_basis,
    make_recency_plan,
    project_onto_output_basis,
    select_residual_event_tiles,
)
from probe_ar_video_residual_memory import (
    capture_identity,
    error_components,
    expected_identities,
    sha256_file,
    summarize_rows,
    validate_manifests,
    validate_metadata,
)


@dataclass(frozen=True)
class Variant:
    name: str
    kernel_source: str
    periodic: bool
    query_groups: str
    temporal_sharing: str
    exact_policy: str
    event_fraction: float

    @property
    def structure(self) -> str:
        return "periodic_bccb" if self.periodic else "nonperiodic_toeplitz"


@dataclass
class CandidateResult:
    metadata: dict[str, Any]
    variant: Variant
    target: torch.Tensor
    estimate: torch.Tensor
    defect: torch.Tensor
    parity_by_head: torch.Tensor
    arithmetic_reduction: float

    @property
    def cell_key(self) -> tuple[int, int, int, str]:
        return (
            int(self.metadata["layer"]),
            int(self.metadata["current_start_frame"]),
            int(self.metadata["denoising_call_index"]),
            self.variant.name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--max-captures", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def make_variants(protocol: dict[str, Any]) -> list[Variant]:
    output = []
    names: set[str] = set()
    for item in protocol["methods"]["variants"]:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate variant name: {name}")
        names.add(name)
        structure = str(item["spatial_structure"])
        if structure not in {"periodic_bccb", "nonperiodic_toeplitz"}:
            raise ValueError(f"unsupported spatial structure: {structure}")
        query_groups = str(item["query_groups"])
        if query_groups not in {"global", "capture_tiles"}:
            raise ValueError(f"unsupported query grouping: {query_groups}")
        temporal_sharing = str(item["temporal_sharing"])
        if temporal_sharing not in {"frame_pair", "relative_frame_offset"}:
            raise ValueError(f"unsupported temporal sharing: {temporal_sharing}")
        exact_policy = str(item["exact_policy"])
        if exact_policy not in {"none", "sink_recent"}:
            raise ValueError(f"unsupported exact policy: {exact_policy}")
        kernel_source = str(item["kernel_source"])
        if kernel_source not in {"record_qk_projection", "calibration_mean"}:
            raise ValueError(f"unsupported kernel source: {kernel_source}")
        event_fraction = float(item["event_fraction"])
        if not 0.0 <= event_fraction <= 1.0:
            raise ValueError("event fraction must be in [0, 1]")
        output.append(
            Variant(
                name=name,
                kernel_source=kernel_source,
                periodic=structure == "periodic_bccb",
                query_groups=query_groups,
                temporal_sharing=temporal_sharing,
                exact_policy=exact_policy,
                event_fraction=event_fraction,
            )
        )
    return output


def select_variants(
    variants: list[Variant], selected_names: Iterable[str] | None
) -> list[Variant]:
    if not selected_names:
        return variants
    requested = list(selected_names)
    by_name = {variant.name: variant for variant in variants}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"unknown variants requested: {missing}")
    return [by_name[name] for name in requested]


def _grid(metadata: dict[str, Any]) -> tuple[int, int]:
    grids = metadata.get("grid_sizes")
    if not isinstance(grids, list) or len(grids) != 1 or len(grids[0]) != 3:
        raise ValueError("capture requires exactly one [frames, height, width] grid")
    return int(grids[0][1]), int(grids[0][2])


def _layout(
    payload: dict[str, Any], source_protocol: dict[str, Any], device: torch.device
):
    capture = source_protocol["capture"]
    return make_captured_query_layout(
        saved_query_tokens=int(payload["query"].shape[0]),
        frame_seq_len=int(capture["frame_seq_len"]),
        tile_size=int(capture["query_tile_size"]),
        tile_count=int(capture["query_tiles_per_record"]),
        device=device,
    )


def build_variant_bank(
    query: torch.Tensor,
    key: torch.Tensor,
    metadata: dict[str, Any],
    layout,
    variant: Variant,
    height: int,
    width: int,
) -> torch.Tensor:
    key_frames = int(metadata["key_frames"])
    bank = build_spatial_kernel_bank(
        query=query,
        key=key,
        layout=layout,
        key_frames=key_frames,
        height=height,
        width=width,
        periodic=variant.periodic,
        query_groups=variant.query_groups,
    )
    if variant.temporal_sharing == "relative_frame_offset":
        bank = pool_kernel_bank_by_relative_frame(
            bank,
            metadata["query_frame_ids"],
            metadata["key_frame_ids"],
        )
    return bank


def fit_calibration_banks(
    paths: list[Path],
    source_protocol: dict[str, Any],
    variants: list[Variant],
    device: torch.device,
) -> dict[tuple[int, int, int, str], torch.Tensor]:
    frozen = [variant for variant in variants if variant.kernel_source == "calibration_mean"]
    if not frozen:
        return {}
    sums: dict[tuple[int, int, int, str], torch.Tensor] = {}
    counts: defaultdict[tuple[int, int, int, str], int] = defaultdict(int)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("prompt_split") != "calibration":
            continue
        validate_metadata(metadata, payload, source_protocol)
        query = payload["query"].to(device)
        key = payload["key"].to(device)
        layout = _layout(payload, source_protocol, device)
        height, width = _grid(metadata)
        for variant in frozen:
            bank = build_variant_bank(
                query, key, metadata, layout, variant, height, width
            ).detach().cpu()
            cell = (
                int(metadata["layer"]),
                int(metadata["current_start_frame"]),
                int(metadata["denoising_call_index"]),
                variant.name,
            )
            sums[cell] = bank if cell not in sums else sums[cell] + bank
            counts[cell] += 1
    return {cell: value / counts[cell] for cell, value in sums.items()}


def append_metric_rows(
    rows: list[dict[str, Any]],
    result: CandidateResult,
    estimate: torch.Tensor,
    correction: str,
    rank: int,
    coefficient_scope: str,
) -> None:
    error, numerator_sq, denominator_sq = error_components(result.target, estimate)
    metadata = result.metadata
    for local_head, head_index in enumerate(metadata["head_indices"]):
        rows.append(
            {
                "protocol_id": metadata["protocol_id"],
                "candidate_protocol_id": "ar-video-causal-bccb-longlive-v1",
                "prompt_id": metadata["prompt_id"],
                "split": metadata["prompt_split"],
                "seed": int(metadata["seed"]),
                "layer": int(metadata["layer"]),
                "current_start_frame": int(metadata["current_start_frame"]),
                "denoising_call_index": int(metadata["denoising_call_index"]),
                "denoising_timestep": int(metadata["denoising_timestep"]),
                "head_index": int(head_index),
                "method": result.variant.name,
                "kernel_source": result.variant.kernel_source,
                "spatial_structure": result.variant.structure,
                "query_groups": result.variant.query_groups,
                "temporal_sharing": result.variant.temporal_sharing,
                "exact_policy": result.variant.exact_policy,
                "event_fraction": result.variant.event_fraction,
                "correction": correction,
                "rank": rank,
                "coefficient_scope": coefficient_scope,
                "deployable": False,
                "representation_only": True,
                "relative_av_l2": float(error[local_head].item()),
                "numerator_sq": float(numerator_sq[local_head].item()),
                "denominator_sq": float(denominator_sq[local_head].item()),
                "dense_reference_parity": float(result.parity_by_head[local_head].item()),
                "arithmetic_reduction": result.arithmetic_reduction,
                "kernel_construction_cost_accounted": False,
            }
        )


def evaluate_capture(
    payload: dict[str, Any],
    source_protocol: dict[str, Any],
    candidate_protocol: dict[str, Any],
    variants: list[Variant],
    calibration_banks: dict[tuple[int, int, int, str], torch.Tensor],
    device: torch.device,
    query_chunk_size: int,
    rows: list[dict[str, Any]],
) -> list[CandidateResult]:
    metadata = payload["metadata"]
    validate_metadata(metadata, payload, source_protocol)
    query = payload["query"].to(device)
    key = payload["key"].to(device)
    value = payload["value"].to(device)
    target = payload["dense_output"].to(device)
    dense_math = dense_attention(query, key, value, query_chunk_size)
    parity_by_head, _, _ = error_components(target, dense_math)
    layout = _layout(payload, source_protocol, device)
    height, width = _grid(metadata)
    key_frames = int(metadata["key_frames"])
    spatial = height * width
    if key.shape[0] != key_frames * spatial:
        raise ValueError("candidate grid does not cover the captured K/V tokens")

    methods = candidate_protocol["methods"]
    capture = candidate_protocol["capture"]
    plan = make_recency_plan(
        num_frames=key_frames,
        sink_frames=int(capture["exact_sink_frames"]),
        recent_frames=int(capture["exact_recent_frames"]),
        max_summary_groups=1,
    )
    key_view = key.reshape(key_frames, spatial, key.shape[1], key.shape[2])
    value_view = value.reshape(key_frames, spatial, value.shape[1], value.shape[2])
    event_cache: dict[float, torch.Tensor] = {0.0: torch.zeros(
        (key_frames, spatial), dtype=torch.bool, device=device
    )}
    bank_cache: dict[tuple[bool, str, str], torch.Tensor] = {}
    results = []
    ranks = [int(rank) for rank in methods["low_rank_residual_ranks"] if int(rank) > 0]

    for variant in variants:
        if variant.event_fraction not in event_cache:
            event_cache[variant.event_fraction] = select_residual_event_tiles(
                key=key_view,
                value=value_view,
                plan=plan,
                tile_size=int(capture["event_tile_size"]),
                tile_fraction=variant.event_fraction,
                query=query,
            )
        exact_frames = (
            exact_sink_recent_frames(
                key_frames,
                int(capture["exact_sink_frames"]),
                int(capture["exact_recent_frames"]),
            )
            if variant.exact_policy == "sink_recent"
            else ()
        )
        if variant.kernel_source == "record_qk_projection":
            bank_key = (variant.periodic, variant.query_groups, variant.temporal_sharing)
            if bank_key not in bank_cache:
                bank_cache[bank_key] = build_variant_bank(
                    query, key, metadata, layout, variant, height, width
                )
            bank = bank_cache[bank_key]
        else:
            cell = (
                int(metadata["layer"]),
                int(metadata["current_start_frame"]),
                int(metadata["denoising_call_index"]),
                variant.name,
            )
            if cell not in calibration_banks:
                raise ValueError(f"missing calibration kernel for {cell}")
            bank = calibration_banks[cell].to(device)
        estimate = structured_attention_from_kernel_bank(
            query=query,
            key=key,
            value=value,
            bank=bank,
            layout=layout,
            key_frames=key_frames,
            height=height,
            width=width,
            periodic=variant.periodic,
            exact_frame_indices=exact_frames,
            event_mask=event_cache[variant.event_fraction],
        )
        groups = 1 if variant.query_groups == "global" else layout.tile_count
        reduction = fft_arithmetic_reduction(
            query_frames=layout.query_frames,
            key_frames=key_frames,
            height=height,
            width=width,
            query_groups=groups,
            exact_frames=len(exact_frames),
            event_fraction_of_structured=variant.event_fraction,
            periodic=variant.periodic,
            fft_constant_factor=float(candidate_protocol["cost_model"]["fft_constant_factor"]),
        )
        defect = target - estimate
        result = CandidateResult(
            metadata=metadata,
            variant=variant,
            target=target.detach().cpu(),
            estimate=estimate.detach().cpu(),
            defect=defect.detach().cpu(),
            parity_by_head=parity_by_head.detach().cpu(),
            arithmetic_reduction=reduction,
        )
        results.append(result)
        append_metric_rows(rows, result, result.estimate, "none", 0, "runtime_qk_rule")
        for rank in ranks:
            projection = adaptive_rank_projection(defect, rank)
            append_metric_rows(
                rows,
                result,
                (estimate + projection).detach().cpu(),
                "adaptive_rank_oracle",
                rank,
                "same_record_dense_defect",
            )
    return results


def add_frozen_basis_metrics(
    candidates: list[CandidateResult],
    candidate_protocol: dict[str, Any],
    device: torch.device,
    rows: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    grouped: defaultdict[tuple[int, int, int, str], list[CandidateResult]] = defaultdict(list)
    for result in candidates:
        grouped[result.cell_key].append(result)
    ranks = [
        int(rank)
        for rank in candidate_protocol["methods"]["low_rank_residual_ranks"]
        if int(rank) > 0
    ]
    artifacts = {}
    for cell, items in sorted(grouped.items()):
        calibration = [item for item in items if item.metadata["prompt_split"] == "calibration"]
        if not calibration:
            continue
        for rank in ranks:
            basis = fit_output_basis([item.defect.to(device) for item in calibration], rank)
            artifact_key = f"l{cell[0]:02d}_f{cell[1]:03d}_c{cell[2]:02d}__{cell[3]}__r{rank}"
            artifacts[artifact_key] = basis.detach().cpu().float()
            for item in items:
                projection = project_onto_output_basis(item.defect.to(device), basis)
                scope = (
                    "calibration_in_sample_dense_defect"
                    if item.metadata["prompt_split"] == "calibration"
                    else "held_out_dense_defect_with_calibration_basis"
                )
                append_metric_rows(
                    rows,
                    item,
                    (item.estimate.to(device) + projection).detach().cpu(),
                    "frozen_calibration_basis_oracle_coefficients",
                    rank,
                    scope,
                )
    return artifacts


def find_summary(
    summaries: list[dict[str, Any]],
    scope: str,
    method: str,
    correction: str,
    rank: int,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in summaries
        if item["scope"] == scope
        and item["method"] == method
        and item["correction"] == correction
        and int(item["rank"]) == rank
    ]
    if len(matches) > 1:
        raise AssertionError("summary key is not unique")
    return matches[0] if matches else None


def decide_gate(
    protocol: dict[str, Any],
    summaries: list[dict[str, Any]],
    capture_validation: dict[str, Any],
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    primary = str(protocol["methods"]["primary_candidate"])
    rank = int(protocol["methods"]["primary_low_rank_residual_rank"])
    adaptive = find_summary(summaries, "held_out", primary, "adaptive_rank_oracle", rank)
    frozen = find_summary(
        summaries,
        "held_out",
        primary,
        "frozen_calibration_basis_oracle_coefficients",
        rank,
    )
    decision: dict[str, Any] = {
        "primary_method": primary,
        "primary_rank": rank,
        "adaptive_oracle": adaptive,
        "frozen_basis_transfer_oracle_coefficients": frozen,
        "measured_speedup_claim_allowed": False,
        "kernel_construction_cost_accounted": False,
        "note": (
            "The dynamic kernel reads runtime Q/K, while low-rank diagnostics read dense "
            "defects. The FFT proxy excludes kernel construction and cannot support a speed claim."
        ),
    }
    if not capture_validation["complete"]:
        decision.update(classification="incomplete", action="finish_full_frozen_protocol")
        return decision
    if capture_validation["worst_dense_reference_parity"] > float(
        evaluation["dense_reference_parity_gate"]
    ):
        decision.update(classification="invalid", action="repair_dense_reference_parity")
        return decision
    if adaptive is None or frozen is None:
        decision.update(classification="invalid", action="repair_missing_primary_metrics")
        return decision
    adaptive_pass = (
        adaptive["aggregate_relative_av_l2"] <= float(evaluation["aggregate_oracle_gate"])
        and adaptive["worst_head_relative_av_l2"] <= float(evaluation["worst_oracle_gate"])
    )
    transfer_pass = (
        frozen["aggregate_relative_av_l2"] <= float(evaluation["aggregate_transfer_gate"])
        and frozen["worst_head_relative_av_l2"] <= float(evaluation["worst_transfer_gate"])
    )
    arithmetic_pass = adaptive["minimum_arithmetic_reduction"] >= float(
        evaluation["minimum_primary_arithmetic_reduction"]
    )
    decision.update(
        adaptive_quality_pass=adaptive_pass,
        frozen_basis_transfer_pass=transfer_pass,
        arithmetic_screen_pass=arithmetic_pass,
    )
    if not adaptive_pass or not arithmetic_pass:
        decision.update(
            classification="null",
            action="stop_recurrent_bccb_and_do_not_build_kernel",
        )
    elif not transfer_pass:
        decision.update(
            classification="boundary",
            action="test_residual_width_guided_episodic_write_only",
        )
    else:
        decision.update(
            classification="representation_pass",
            action="open_bounded_residual_width_memory_gate",
        )
    return decision


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no metric rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_captures is not None and args.max_captures <= 0:
        raise ValueError("max captures must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        args.output_dir / "metrics.csv",
        args.output_dir / "summary.json",
        args.output_dir / "gate_decision.json",
        args.output_dir / "calibration_artifacts.pt",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing result artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    candidate_protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    if candidate_protocol["source_protocol_id"] != source_protocol["protocol_id"]:
        raise ValueError("candidate/source protocol IDs do not match")
    manifests, paths = validate_manifests(
        args.capture_dir, source_protocol, sha256_file(args.source_protocol)
    )
    expected = expected_identities(source_protocol)
    path_by_identity = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "metadata" not in payload:
            raise ValueError(f"invalid capture payload: {path}")
        validate_metadata(payload["metadata"], payload, source_protocol)
        identity = capture_identity(payload["metadata"])
        if identity in path_by_identity:
            raise ValueError(f"duplicate capture identity: {identity}")
        if identity not in expected:
            raise ValueError(f"capture identity lies outside source protocol: {identity}")
        path_by_identity[identity] = path
    selected_items = sorted(path_by_identity.items())
    if args.max_captures is not None:
        if not args.allow_partial:
            raise ValueError("max captures requires --allow-partial")
        selected_items = selected_items[: args.max_captures]
    selected_identities = {identity for identity, _ in selected_items}
    selected_paths = [path for _, path in selected_items]
    missing = sorted(expected - selected_identities)
    if missing and not args.allow_partial:
        raise ValueError(f"missing {len(missing)} frozen captures; first={missing[0]}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested but CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)
    variants = select_variants(make_variants(candidate_protocol), args.variants)
    calibration_banks = fit_calibration_banks(
        selected_paths, source_protocol, variants, device
    )
    rows: list[dict[str, Any]] = []
    candidates: list[CandidateResult] = []
    for index, path in enumerate(selected_paths, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[evaluate {index}/{len(selected_paths)}] {capture_identity(payload['metadata'])}")
        candidates.extend(
            evaluate_capture(
                payload,
                source_protocol,
                candidate_protocol,
                variants,
                calibration_banks,
                device,
                args.query_chunk_size,
                rows,
            )
        )
    output_bases = add_frozen_basis_metrics(
        candidates, candidate_protocol, device, rows
    )
    summaries = summarize_rows(rows)
    parity_values = [
        float(value)
        for result in candidates[:: max(1, len(variants))]
        for value in result.parity_by_head.tolist()
    ]
    capture_validation = {
        "expected_capture_count": len(expected),
        "found_capture_count": len(selected_identities),
        "complete": not missing,
        "missing_identities": missing,
        "manifest_count": len(manifests),
        "aggregate_dense_reference_parity": math.sqrt(
            sum(value * value for value in parity_values) / len(parity_values)
        ),
        "worst_dense_reference_parity": max(parity_values),
    }
    gate = decide_gate(candidate_protocol, summaries, capture_validation)
    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": candidate_protocol["protocol_id"],
                "protocol_sha256": sha256_file(args.protocol),
                "source_protocol_id": source_protocol["protocol_id"],
                "source_protocol_sha256": sha256_file(args.source_protocol),
                "capture_validation": capture_validation,
                "manifests": manifests,
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": 1,
            "protocol_id": candidate_protocol["protocol_id"],
            "calibration_kernel_means": calibration_banks,
            "calibration_output_bases": output_bases,
        },
        args.output_dir / "calibration_artifacts.pt",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
