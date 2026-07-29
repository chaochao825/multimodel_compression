#!/usr/bin/env python3
"""Probe whether hardware-shaped support makes Wan attention defects low-rank.

Every support and tail basis may inspect the held-out dense AV output.  This is
therefore a representation oracle, not a router-transfer or latency result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    object_sha256,
    require_fresh_output_dir,
)
from probe_dynamic_sparse_lowrank_oracle import aligned_query_tile_starts
from support_manifold_oracle_core import (
    GroupProblem,
    SearchResult,
    adaptive_tail,
    aggregate_statistics,
    assemble_defect,
    atom_statistics,
    contiguous_partition,
    motion_path_selection,
    optimize_support,
    selected_output,
    slice_statistics,
    support_cost,
    thw_partition,
)


@dataclass(frozen=True)
class Cell:
    name: str
    layer: int
    sampling_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-kind", choices=("smoke", "registered"), default="registered")
    parser.add_argument("--capture-hash-mode", choices=("sha256", "metadata"), default="sha256")
    parser.add_argument("--execution-resource-note", required=True)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError(f"unsupported protocol schema: {path}")
    scope = protocol["scope"]
    sample_ids = tuple(map(str, scope["sample_ids"]))
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("scope sample IDs must be non-empty and unique")
    cells = tuple(
        Cell(str(raw["name"]), int(raw["layer"]), int(raw["sampling_step"]))
        for raw in scope["cells"]
    )
    if not cells or len({cell.name for cell in cells}) != len(cells):
        raise ValueError("cells must be non-empty and uniquely named")
    families = tuple(map(str, protocol["support_search"]["families"]))
    supported = {
        "fixed64",
        "shifted64",
        "hierarchical32",
        "shifted32",
        "thw8x8",
        "motion_warp8x8",
    }
    if not families or set(families) - supported:
        raise ValueError(f"unsupported support families: {sorted(set(families) - supported)}")
    query_size = int(scope["query_tile_size"])
    rank = int(scope["rank"])
    if query_size != 64:
        raise ValueError("registered support families currently require 64-query tiles")
    if rank <= 0 or rank > query_size:
        raise ValueError("rank must be in [1, query_tile_size]")
    return protocol


def cells_from_protocol(protocol: dict[str, object]) -> tuple[Cell, ...]:
    return tuple(
        Cell(str(raw["name"]), int(raw["layer"]), int(raw["sampling_step"]))
        for raw in protocol["scope"]["cells"]
    )


def resolve_captures(
    index_path: Path,
    sample_ids: tuple[str, ...],
    cells: tuple[Cell, ...],
    branch: str,
) -> list[dict[str, object]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captures: list[dict[str, object]] = []
    for sample_id in sample_ids:
        for cell in cells:
            matches = [
                row
                for row in rows
                if row["sample_id"] == sample_id
                and row["branch"] == branch
                and int(row["layer"]) == cell.layer
                and int(row["sampling_step"]) == cell.sampling_step
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one capture for sample={sample_id}, cell={cell.name}; found {len(matches)}"
                )
            path = Path(matches[0]["path"])
            if not path.is_absolute():
                path = index_path.parent / path
            if not path.is_file():
                raise FileNotFoundError(path)
            captures.append({"sample_id": sample_id, "cell": cell, "path": path.resolve()})
    return captures


def capture_fingerprint(path: Path, mode: str) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {"path": str(path), "bytes": stat.st_size}
    if mode == "sha256":
        record["sha256"] = file_sha256(path)
    else:
        record["mtime_ns"] = stat.st_mtime_ns
    return record


def grid_shape(metadata: dict[str, object], tokens: int) -> tuple[int, int, int]:
    raw = metadata.get("grid_size")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("capture must contain a three-dimensional grid_size")
    shape = tuple(int(value) for value in raw)
    if math.prod(shape) != tokens:
        raise ValueError(f"grid {shape} does not match {tokens} tokens")
    if metadata.get("token_flatten_order") != "t,h,w":
        raise ValueError("support probe requires t,h,w token flattening")
    return shape


def pair_groups(atoms: int, shifted: bool) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    start = 0
    if shifted:
        groups.append((0,))
        start = 1
    while start < atoms:
        groups.append(tuple(range(start, min(start + 2, atoms))))
        start += 2
    return tuple(groups)


def make_group(reference: torch.Tensor, statistics, density: float) -> GroupProblem:
    logical_tokens = int(statistics.valid_counts.sum().item())
    budget = max(
        1,
        min(
            statistics.contributions.shape[0],
            int(round(density * logical_tokens / statistics.execution_width)),
        ),
    )
    return GroupProblem(reference=reference, statistics=statistics, budget=budget)


def family_groups(
    family: str,
    reference: torch.Tensor,
    density: float,
    contiguous32,
    shifted32,
    geometry64,
) -> tuple[GroupProblem, ...]:
    if family == "fixed64":
        stats = aggregate_statistics(contiguous32, pair_groups(contiguous32.contributions.shape[0], False), 64)
        return (make_group(reference, stats, density),)
    if family == "shifted64":
        stats = aggregate_statistics(contiguous32, pair_groups(contiguous32.contributions.shape[0], True), 64)
        return (make_group(reference, stats, density),)
    if family in {"hierarchical32", "shifted32"}:
        stats = contiguous32 if family == "hierarchical32" else shifted32
        return (
            make_group(reference[:32], slice_statistics(stats, 0, 32), density),
            make_group(reference[32:], slice_statistics(stats, 32, 64), density),
        )
    if family in {"thw8x8", "motion_warp8x8"}:
        return (make_group(reference, geometry64, density),)
    raise ValueError(f"unsupported family: {family}")


def direct_motion_result(
    groups: tuple[GroupProblem, ...],
    rank: int,
    shape: tuple[int, int, int],
    tile_height: int,
    tile_width: int,
) -> SearchResult:
    tile_rows = math.ceil(shape[1] / tile_height)
    tile_columns = math.ceil(shape[2] / tile_width)
    selection = motion_path_selection(groups[0], shape[0], tile_rows, tile_columns)
    selections = (selection,)
    defect = assemble_defect(groups, selections)
    basis, residual, _ = adaptive_tail(defect, rank)
    residual_sq = float(residual.square().sum())
    return SearchResult(
        selections=selections,
        defect=defect,
        basis=basis,
        residual_sq=residual_sq,
        initial_residual_sq=residual_sq,
        accepted_alternations=0,
        accepted_swaps=0,
        trace=({"stage": "motion_path", "residual_sq": residual_sq},),
    )


def rank_diagnostics(
    defect: torch.Tensor,
    reference_sq: float,
    gate: float,
) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(defect)
    energy = singular.square()
    total = float(energy.sum())
    cumulative = torch.cumsum(energy, dim=0)
    output: dict[str, float | int] = {}
    for rank in (4, 8, 16, 32):
        used = min(rank, energy.numel())
        captured = float(cumulative[used - 1] / max(total, 1e-30)) if used else 0.0
        output[f"defect_energy_rank{rank}"] = captured
    threshold_sq = gate * gate * reference_sq
    required = energy.numel()
    for rank in range(energy.numel() + 1):
        residual_sq = total if rank == 0 else max(0.0, total - float(cumulative[rank - 1]))
        if residual_sq <= threshold_sq:
            required = rank
            break
    output["rank_required_for_record_gate"] = required
    return output


def selection_mass_mean(groups: tuple[GroupProblem, ...], selections: tuple[torch.Tensor, ...]) -> float:
    total = 0.0
    queries = 0
    for problem, selected in zip(groups, selections):
        _, mass = selected_output(problem, selected)
        total += float(mass.sum())
        queries += mass.numel()
    return total / queries


def selection_signature(family: str, selections: tuple[torch.Tensor, ...]) -> str:
    text = family + "|" + "|".join(
        ",".join(map(str, selected.detach().cpu().tolist())) for selected in selections
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@torch.inference_mode()
def process_capture(
    capture: dict[str, object],
    protocol: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    scope = protocol["scope"]
    search = protocol["support_search"]
    gates = protocol["gates"]
    payload = torch.load(Path(capture["path"]), map_location="cpu", weights_only=False)
    q_all, k_all, v_all = payload["q"][0], payload["k"][0], payload["v"][0]
    if q_all.shape != k_all.shape or q_all.shape != v_all.shape:
        raise ValueError("capture Q/K/V shapes differ")
    tokens, heads, channels = q_all.shape
    shape = grid_shape(dict(payload.get("metadata", {})), tokens)
    tile_size = int(scope["query_tile_size"])
    starts = aligned_query_tile_starts(tokens, tile_size, int(scope["query_tiles"]))
    rank = int(scope["rank"])
    scale = float(payload.get("softmax_scale", channels**-0.5))
    tile_height = int(search["thw_tile_height"])
    tile_width = int(search["thw_tile_width"])
    partition32 = contiguous_partition(tokens, 32)
    partition32_shifted = contiguous_partition(tokens, 32, offset=16)
    partition_thw = thw_partition(shape, tile_height, tile_width)
    rows: list[dict[str, object]] = []

    for head in range(heads):
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        for tile_index, start in enumerate(starts):
            scores = q[start : start + tile_size] @ k.T * scale
            attention = torch.softmax(scores, dim=1)
            reference = attention @ v
            reference_sq = float(reference.square().sum())
            contiguous_stats = atom_statistics(attention, v, partition32)
            shifted_stats = atom_statistics(attention, v, partition32_shifted)
            geometry_stats = atom_statistics(attention, v, partition_thw)
            for density in map(float, scope["densities"]):
                family_rows: list[dict[str, object]] = []
                for family in map(str, search["families"]):
                    groups = family_groups(
                        family,
                        reference,
                        density,
                        contiguous_stats,
                        shifted_stats,
                        geometry_stats,
                    )
                    if family == "motion_warp8x8":
                        result = direct_motion_result(groups, rank, shape, tile_height, tile_width)
                    else:
                        result = optimize_support(
                            groups,
                            rank=rank,
                            alternations=int(search["alternations"]),
                            swap_steps=int(search["swap_steps"]),
                            shortlist=int(search["swap_shortlist"]),
                        )
                    critical_sq = float(result.defect.square().sum())
                    cost = support_cost(groups, result.selections, tokens)
                    row: dict[str, object] = {
                        "sample_id": capture["sample_id"],
                        "cell": capture["cell"].name,
                        "layer": capture["cell"].layer,
                        "sampling_step": capture["cell"].sampling_step,
                        "branch": scope["branch"],
                        "head": head,
                        "tile_index": tile_index,
                        "query_start": start,
                        "family": family,
                        "chosen_family": family,
                        "density_target": density,
                        "rank": rank,
                        "reference_sq": reference_sq,
                        "critical_residual_sq": critical_sq,
                        "adaptive_residual_sq": result.residual_sq,
                        "initial_adaptive_residual_sq": result.initial_residual_sq,
                        "critical_output_relative_l2": math.sqrt(critical_sq / max(reference_sq, 1e-30)),
                        "adaptive_output_relative_l2": math.sqrt(result.residual_sq / max(reference_sq, 1e-30)),
                        "initial_adaptive_output_relative_l2": math.sqrt(
                            result.initial_residual_sq / max(reference_sq, 1e-30)
                        ),
                        "rank_aware_relative_improvement": 1.0
                        - result.residual_sq / max(result.initial_residual_sq, 1e-30),
                        "selected_attention_mass_mean": selection_mass_mean(groups, result.selections),
                        "accepted_alternations": result.accepted_alternations,
                        "accepted_swaps": result.accepted_swaps,
                        "selection_signature": selection_signature(family, result.selections),
                        "oracle_access": "heldout_dense_AV_support_and_tail",
                        **cost,
                        **rank_diagnostics(
                            result.defect,
                            reference_sq,
                            float(gates["oracle_worst_record_output_relative_l2"]),
                        ),
                    }
                    family_rows.append(row)
                    rows.append(row)
                best = min(family_rows, key=lambda item: float(item["adaptive_residual_sq"]))
                oracle = dict(best)
                oracle["family"] = "support_family_oracle"
                oracle["chosen_family"] = best["family"]
                oracle["oracle_access"] = "heldout_dense_AV_family_support_and_tail"
                rows.append(oracle)
            del scores, attention, reference, contiguous_stats, shifted_stats, geometry_stats
        del q, k, v
    return rows


def aggregate_records(
    records: list[dict[str, object]], protocol: dict[str, object]
) -> list[dict[str, object]]:
    gates = protocol["gates"]
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["cell"]), str(row["family"]), float(row["density_target"]))].append(row)
    output: list[dict[str, object]] = []
    for (cell, family, density), rows in sorted(grouped.items()):
        reference_sq = sum(float(row["reference_sq"]) for row in rows)
        critical_sq = sum(float(row["critical_residual_sq"]) for row in rows)
        adaptive_sq = sum(float(row["adaptive_residual_sq"]) for row in rows)
        aggregate = math.sqrt(adaptive_sq / max(reference_sq, 1e-30))
        worst = max(float(row["adaptive_output_relative_l2"]) for row in rows)
        execution_density = sum(float(row["execution_density"]) for row in rows) / len(rows)
        quality_pass = aggregate <= float(gates["oracle_aggregate_output_relative_l2"]) and worst <= float(
            gates["oracle_worst_record_output_relative_l2"]
        )
        output.append(
            {
                "cell": cell,
                "family": family,
                "density_target": density,
                "records": len(rows),
                "critical_output_relative_l2": math.sqrt(critical_sq / max(reference_sq, 1e-30)),
                "adaptive_rank16_output_relative_l2": aggregate,
                "adaptive_rank16_worst_record_relative_l2": worst,
                "adaptive_rank16_p95_record_relative_l2": sorted(
                    float(row["adaptive_output_relative_l2"]) for row in rows
                )[max(0, math.ceil(0.95 * len(rows)) - 1)],
                "initial_adaptive_rank16_output_relative_l2": math.sqrt(
                    sum(float(row["initial_adaptive_residual_sq"]) for row in rows)
                    / max(reference_sq, 1e-30)
                ),
                "rank_aware_relative_improvement_mean": sum(
                    float(row["rank_aware_relative_improvement"]) for row in rows
                )
                / len(rows),
                "rank_required_for_record_gate_mean": sum(
                    int(row["rank_required_for_record_gate"]) for row in rows
                )
                / len(rows),
                "rank_required_for_record_gate_max": max(
                    int(row["rank_required_for_record_gate"]) for row in rows
                ),
                "selected_attention_mass_mean": sum(
                    float(row["selected_attention_mass_mean"]) for row in rows
                )
                / len(rows),
                "logical_density_mean": sum(float(row["logical_density"]) for row in rows) / len(rows),
                "execution_density_mean": execution_density,
                "kernel_tiles_mean": sum(int(row["kernel_tiles"]) for row in rows) / len(rows),
                "quality_gate": quality_pass,
                "execution_budget_gate": execution_density
                <= float(gates["max_execution_density"]),
                "support_pregate_pass": quality_pass
                and execution_density <= float(gates["max_execution_density"]),
            }
        )
    baseline = {
        (str(row["cell"]), float(row["density_target"])): float(row["kernel_tiles_mean"])
        for row in output
        if row["family"] == "fixed64"
    }
    for row in output:
        key = (str(row["cell"]), float(row["density_target"]))
        row["kernel_tile_multiplier_vs_fixed64"] = float(row["kernel_tiles_mean"]) / baseline[key]
        row["kernel_tile_budget_gate"] = float(row["kernel_tile_multiplier_vs_fixed64"]) <= float(
            gates["max_kernel_tile_multiplier"]
        )
        row["support_pregate_pass"] = bool(row["support_pregate_pass"]) and bool(
            row["kernel_tile_budget_gate"]
        )
    return output


def build_decision(
    summary: list[dict[str, object]], protocol: dict[str, object]
) -> dict[str, object]:
    layer14_cells = {
        str(raw["name"])
        for raw in protocol["scope"]["cells"]
        if int(raw["layer"]) == 14
    }
    candidates: list[dict[str, object]] = []
    keys = sorted({(str(row["family"]), float(row["density_target"])) for row in summary})
    for family, density in keys:
        rows = [
            row
            for row in summary
            if row["family"] == family
            and float(row["density_target"]) == density
            and row["cell"] in layer14_cells
        ]
        if {str(row["cell"]) for row in rows} != layer14_cells:
            continue
        candidates.append(
            {
                "family": family,
                "density_target": density,
                "cells": len(rows),
                "max_aggregate_output_relative_l2": max(
                    float(row["adaptive_rank16_output_relative_l2"]) for row in rows
                ),
                "max_worst_record_output_relative_l2": max(
                    float(row["adaptive_rank16_worst_record_relative_l2"]) for row in rows
                ),
                "max_execution_density": max(float(row["execution_density_mean"]) for row in rows),
                "max_kernel_tile_multiplier": max(
                    float(row["kernel_tile_multiplier_vs_fixed64"]) for row in rows
                ),
                "all_layer14_cells_pass": all(bool(row["support_pregate_pass"]) for row in rows),
            }
        )
    best = min(candidates, key=lambda row: float(row["max_worst_record_output_relative_l2"]))
    deployable = [row for row in candidates if row["family"] != "support_family_oracle"]
    passed = [row for row in deployable if bool(row["all_layer14_cells_pass"])]
    oracle_passed = [
        row
        for row in candidates
        if row["family"] == "support_family_oracle" and bool(row["all_layer14_cells_pass"])
    ]
    if passed:
        verdict = "GO_CALIBRATION_ONLY_GRASSMANN_CHART_PROBE"
    elif oracle_passed:
        verdict = "ONLY_POSTHOC_SUPPORT_FAMILY_ORACLE_PASSES"
    else:
        verdict = "STOP_SUPPORT_SHAPING_RANK16_PREGATE_FAILS"
    return {
        "verdict": verdict,
        "best_layer14_candidate": best,
        "passing_deployable_supports": passed,
        "passing_support_family_oracle": oracle_passed,
        "layer14_candidates": candidates,
        "claim_boundary": protocol["claim_boundary"],
        "next_stage_allowed": bool(passed),
    }


def main() -> None:
    args = parse_args()
    if args.sample_shard_count <= 0 or not 0 <= args.sample_shard_index < args.sample_shard_count:
        raise ValueError("sample shard index must be in [0, sample_shard_count)")
    started = time.time()
    protocol_path = args.protocol_config.resolve()
    capture_index = args.capture_index.resolve()
    protocol = load_protocol(protocol_path)
    require_fresh_output_dir(args.output_dir)
    run_id = uuid.uuid4().hex
    events = JsonlEventLog(args.output_dir / "events.jsonl", run_id)
    events.emit("run_started", run_kind=args.run_kind, device=args.device)
    sample_ids = tuple(map(str, protocol["scope"]["sample_ids"]))
    captures = resolve_captures(
        capture_index,
        sample_ids,
        cells_from_protocol(protocol),
        str(protocol["scope"]["branch"]),
    )
    sample_shards = {
        sample_id: index % args.sample_shard_count for index, sample_id in enumerate(sample_ids)
    }
    captures = [
        capture
        for capture in captures
        if sample_shards[str(capture["sample_id"])] == args.sample_shard_index
    ]
    if not captures:
        raise ValueError("sample shard contains no captures")
    fingerprints = [
        capture_fingerprint(Path(capture["path"]), args.capture_hash_mode) for capture in captures
    ]
    device = torch.device(args.device)
    records: list[dict[str, object]] = []
    for index, capture in enumerate(captures):
        capture_rows = process_capture(capture, protocol, device)
        records.extend(capture_rows)
        events.emit(
            "capture_processed",
            index=index,
            captures=len(captures),
            sample_id=capture["sample_id"],
            cell=capture["cell"].name,
            records=len(capture_rows),
        )
        print(
            f"[support] {index + 1}/{len(captures)} sample={capture['sample_id']} "
            f"cell={capture['cell'].name} rows={len(capture_rows)}",
            flush=True,
        )
    summary = aggregate_records(records, protocol)
    decision = build_decision(summary, protocol)
    atomic_write_csv(args.output_dir / "support_records.csv", records)
    atomic_write_csv(args.output_dir / "support_summary.csv", summary)
    atomic_write_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": args.run_kind,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "command": vars(args) | {"capture_index": str(args.capture_index), "protocol_config": str(args.protocol_config), "output_dir": str(args.output_dir)},
        "protocol": protocol,
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_object_sha256": object_sha256(protocol),
        "capture_index": str(capture_index),
        "capture_index_sha256": file_sha256(capture_index),
        "capture_fingerprints": fingerprints,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "execution_resource_note": args.execution_resource_note,
        },
        "semantics": {
            "normalization": "selected-key renormalized sparse AV; adaptive output-defect tail is post-hoc",
            "support_access": "all support selections inspect each held-out dense AV record",
            "tail_access": "rank-r basis and coefficients inspect each held-out dense AV defect",
            "cost": "logical and padded execution pairs; no measured H200 latency claim",
            "query_granularity": "layer x step x head x contiguous 64-query tile; hierarchical families split internally to 32 queries",
            "sample_shard": {
                "index": args.sample_shard_index,
                "count": args.sample_shard_count,
                "sample_ids": sorted({str(capture["sample_id"]) for capture in captures}),
            },
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    artifacts = {
        name: file_sha256(args.output_dir / name)
        for name in ("support_records.csv", "support_summary.csv", "decision.json", "manifest.json", "events.jsonl")
    }
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {"run_id": run_id, "verdict": decision["verdict"], "artifact_sha256": artifacts},
    )
    print(f"[support] verdict={decision['verdict']}")
    print(f"[support] elapsed={time.time() - started:.1f}s output={args.output_dir}")


if __name__ == "__main__":
    main()
