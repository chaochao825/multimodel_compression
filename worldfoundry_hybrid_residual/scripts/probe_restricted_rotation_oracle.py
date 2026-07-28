#!/usr/bin/env python3
"""Registered restricted-rotation oracle for sparse-attention output tails.

The critical mask and every restricted transform are allowed to inspect dense
held-out defects.  The experiment is therefore a post-hoc function-class
capacity test, not a deployable predictor.  Prompt/seed holdouts only test how
far a source basis must move; they do not turn oracle fitting into transfer.
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
import torch.nn.functional as F

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    object_sha256,
    require_fresh_output_dir,
)
from probe_dynamic_sparse_lowrank_oracle import (
    aligned_query_tile_starts,
    block_output_contributions,
    outputs_from_selected_blocks,
)
from restricted_rotation_oracle_core import (
    fit_parameterized_rotation,
    greedy_givens_prefixes,
    householder_prefixes,
    orthonormalize,
    residual_after_basis,
    right_singular_basis,
    rotation_cost,
    subspace_overlap,
)


@dataclass(frozen=True)
class Cell:
    name: str
    layer: int
    sampling_step: int


@dataclass
class TailRecord:
    sample_id: str
    cell: Cell
    method: str
    density: float
    head: int
    tile_index: int
    query_start: int
    defect: torch.Tensor
    target_basis: torch.Tensor
    reference_sq: float
    critical_residual_sq: float
    adaptive_residual_sq: float
    selected_mass_mean: float
    key_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-kind", choices=("smoke", "registered"), default="registered")
    parser.add_argument("--capture-hash-mode", choices=("sha256", "metadata"), default="sha256")
    parser.add_argument("--execution-resource-note", required=True)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported protocol schema: {path}")
    scope = payload["scope"]
    sample_ids = tuple(map(str, scope["sample_ids"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("scope sample IDs must be unique")
    cells = [Cell(str(raw["name"]), int(raw["layer"]), int(raw["sampling_step"])) for raw in scope["cells"]]
    if len({cell.name for cell in cells}) != len(cells):
        raise ValueError("cell names must be unique")
    for holdout in payload["holdouts"]:
        source = tuple(map(str, holdout["source"]))
        target = tuple(map(str, holdout["target"]))
        if not source or not target or set(source) & set(target):
            raise ValueError(f"holdout {holdout['name']} must have non-empty disjoint source/target")
        unknown = (set(source) | set(target)) - set(sample_ids)
        if unknown:
            raise ValueError(f"holdout {holdout['name']} references unknown samples: {sorted(unknown)}")
    if int(scope["rank"]) > int(scope["query_tile_size"]):
        raise ValueError("rank cannot exceed query tile size for the per-tile SVD oracle")
    return payload


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def resolve_capture_rows(
    capture_index: Path,
    sample_ids: tuple[str, ...],
    cells: tuple[Cell, ...],
    branch: str,
) -> list[dict[str, object]]:
    with capture_index.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    resolved: list[dict[str, object]] = []
    for sample_id in sample_ids:
        for cell in cells:
            matches = [
                row
                for row in raw_rows
                if row["sample_id"] == sample_id
                and row["branch"] == branch
                and int(row["layer"]) == cell.layer
                and int(row["sampling_step"]) == cell.sampling_step
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one capture for sample={sample_id}, cell={cell.name}; "
                    f"found {len(matches)}"
                )
            path = Path(matches[0]["path"])
            if not path.is_absolute():
                path = capture_index.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved.append({"sample_id": sample_id, "cell": cell, "path": path})
    return resolved


def capture_fingerprint(path: Path, mode: str) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "bytes": stat.st_size}
    if mode == "sha256":
        result["sha256"] = file_sha256(path)
    else:
        result["mtime_ns"] = stat.st_mtime_ns
    return result


def _selection_order(
    method: str,
    contributions: torch.Tensor,
    block_mass: torch.Tensor,
) -> torch.Tensor:
    if method == "mass_topk":
        return block_mass.sum(dim=1).argsort(descending=True)
    if method == "contribution_norm":
        return contributions.square().sum(dim=(1, 2)).argsort(descending=True)
    raise ValueError(f"unsupported critical method: {method}")


@torch.inference_mode()
def process_capture(
    capture: dict[str, object],
    protocol: dict[str, object],
    device: torch.device,
) -> list[TailRecord]:
    scope = protocol["scope"]
    path = Path(capture["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    q_all, k_all, v_all = payload["q"][0], payload["k"][0], payload["v"][0]
    if q_all.shape != k_all.shape or q_all.shape != v_all.shape:
        raise ValueError(f"Q/K/V shape mismatch in {path}")
    tokens, heads, channels = q_all.shape
    tile_size = int(scope["query_tile_size"])
    tile_starts = aligned_query_tile_starts(tokens, tile_size, int(scope["query_tiles"]))
    block_size = int(scope["key_block_size"])
    key_blocks = math.ceil(tokens / block_size)
    rank = int(scope["rank"])
    scale = float(payload.get("softmax_scale", channels**-0.5))
    records: list[TailRecord] = []

    for head in range(heads):
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        for tile_index, start in enumerate(tile_starts):
            scores = q[start : start + tile_size] @ k.T * scale
            attention = torch.softmax(scores, dim=1)
            reference = attention @ v
            reference_sq = float(reference.square().sum())
            contributions, block_mass, _ = block_output_contributions(attention, v, block_size)
            for method in map(str, scope["critical_methods"]):
                order = _selection_order(method, contributions, block_mass)
                for density in map(float, scope["densities"]):
                    budget = max(1, min(key_blocks, int(round(density * key_blocks))))
                    selected = order[:budget]
                    _, sparse_output, selected_mass = outputs_from_selected_blocks(
                        contributions, block_mass, selected
                    )
                    defect = reference - sparse_output
                    target_basis = right_singular_basis(defect, rank)
                    adaptive_residual = residual_after_basis(defect, target_basis)
                    records.append(
                        TailRecord(
                            sample_id=str(capture["sample_id"]),
                            cell=capture["cell"],
                            method=method,
                            density=density,
                            head=head,
                            tile_index=tile_index,
                            query_start=start,
                            defect=defect.cpu(),
                            target_basis=target_basis.cpu(),
                            reference_sq=reference_sq,
                            critical_residual_sq=float(defect.square().sum()),
                            adaptive_residual_sq=float(adaptive_residual.square().sum()),
                            selected_mass_mean=float(selected_mass.mean()),
                            key_tokens=tokens,
                        )
                    )
            del scores, attention, reference, contributions, block_mass
        del q, k, v
    return records


def aggregate_pregate(records: list[TailRecord], protocol: dict[str, object]) -> list[dict[str, object]]:
    gates = protocol["gates"]
    grouped: dict[tuple[str, str, float], list[TailRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.cell.name, record.method, record.density)].append(record)
    rows: list[dict[str, object]] = []
    for (cell, method, density), group in sorted(grouped.items()):
        adaptive_sq = sum(record.adaptive_residual_sq for record in group)
        critical_sq = sum(record.critical_residual_sq for record in group)
        reference_sq = sum(record.reference_sq for record in group)
        worst = max(math.sqrt(record.adaptive_residual_sq / record.reference_sq) for record in group)
        aggregate = math.sqrt(adaptive_sq / reference_sq)
        rows.append(
            {
                "cell": cell,
                "method": method,
                "density": density,
                "records": len(group),
                "critical_output_relative_l2": math.sqrt(critical_sq / reference_sq),
                "adaptive_rank_output_relative_l2": aggregate,
                "adaptive_rank_worst_record_relative_l2": worst,
                "selected_attention_mass_mean": sum(record.selected_mass_mean for record in group) / len(group),
                "aggregate_gate": float(aggregate <= float(gates["oracle_aggregate_output_relative_l2"])),
                "worst_gate": float(worst <= float(gates["oracle_worst_record_output_relative_l2"])),
                "adaptive_pregate_pass": aggregate <= float(gates["oracle_aggregate_output_relative_l2"])
                and worst <= float(gates["oracle_worst_record_output_relative_l2"]),
            }
        )
    return rows


def _record_lookup_key(record: TailRecord) -> tuple[object, ...]:
    return (
        record.sample_id,
        record.cell.name,
        record.method,
        record.density,
        record.head,
        record.tile_index,
    )


def source_basis(
    lookup: dict[tuple[object, ...], TailRecord],
    sample_ids: tuple[str, ...],
    template: TailRecord,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    defects = []
    for sample_id in sample_ids:
        key = (
            sample_id,
            template.cell.name,
            template.method,
            template.density,
            template.head,
            template.tile_index,
        )
        defects.append(lookup[key].defect)
    return right_singular_basis(torch.cat(defects).to(device), rank)


def candidate_row(
    record: TailRecord,
    split_name: str,
    family: str,
    generators: int,
    basis: torch.Tensor,
    protocol: dict[str, object],
) -> dict[str, object]:
    scope = protocol["scope"]
    gates = protocol["gates"]
    # Use one high-precision audit path for every family.  This does not alter
    # the float32 attention/mask/optimizer semantics; it prevents tiny QR or
    # eigensolver drift from dominating near-zero reference outputs.
    basis = orthonormalize(basis.to(dtype=torch.float64))
    defect = record.defect.to(device=basis.device, dtype=basis.dtype)
    target = orthonormalize(
        record.target_basis.to(device=basis.device, dtype=basis.dtype)
    )
    residual_sq = float(residual_after_basis(defect, basis).square().sum())
    output_error = math.sqrt(residual_sq / record.reference_sq)
    cost = rotation_cost(
        family,
        generators,
        query_tokens=int(scope["query_tile_size"]),
        key_tokens=record.key_tokens,
        channels=defect.shape[1],
        rank=int(scope["rank"]),
        block_size=int(protocol["families"]["orthogonal_bcm"]["block_size"]),
    )
    basis_payload = defect.shape[1] * int(scope["rank"])
    oracle_access = "source_only" if family == "frozen" else "heldout_defect_posthoc"
    return {
        "split": split_name,
        "sample_id": record.sample_id,
        "cell": record.cell.name,
        "layer": record.cell.layer,
        "sampling_step": record.cell.sampling_step,
        "method": record.method,
        "density": record.density,
        "head": record.head,
        "tile_index": record.tile_index,
        "query_start": record.query_start,
        "family": family,
        "generators": generators,
        "oracle_access": oracle_access,
        "residual_sq": residual_sq,
        "reference_sq": record.reference_sq,
        "output_relative_l2": output_error,
        "critical_output_relative_l2": math.sqrt(record.critical_residual_sq / record.reference_sq),
        "adaptive_output_relative_l2": math.sqrt(record.adaptive_residual_sq / record.reference_sq),
        "subspace_overlap": float(subspace_overlap(basis, target)),
        "selected_attention_mass_mean": record.selected_mass_mean,
        "dynamic_scalars": cost.dynamic_scalars,
        "dynamic_scalar_ratio_to_rank_basis": cost.dynamic_scalars / basis_payload,
        "rotation_macs": cost.rotation_macs,
        "tail_macs": cost.tail_macs,
        "dense_attention_macs": cost.dense_attention_macs,
        "extra_work_ratio": cost.work_ratio,
        "record_quality_gate": output_error <= float(gates["oracle_worst_record_output_relative_l2"]),
    }


def summarize_candidates(rows: list[dict[str, object]], protocol: dict[str, object]) -> list[dict[str, object]]:
    group_fields = ("split", "cell", "method", "density", "family", "generators")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    gates = protocol["gates"]
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        aggregate = math.sqrt(residual_sq / reference_sq)
        worst = max(float(row["output_relative_l2"]) for row in group)
        dynamic_scalars = max(int(row["dynamic_scalars"]) for row in group)
        work_ratio = max(float(row["extra_work_ratio"]) for row in group)
        summary = dict(zip(group_fields, key))
        summary.update(
            {
                "records": len(group),
                "aggregate_output_relative_l2": aggregate,
                "worst_record_output_relative_l2": worst,
                "subspace_overlap_mean": sum(float(row["subspace_overlap"]) for row in group) / len(group),
                "dynamic_scalars": dynamic_scalars,
                "dynamic_scalar_ratio_to_rank_basis": max(
                    float(row["dynamic_scalar_ratio_to_rank_basis"]) for row in group
                ),
                "extra_work_ratio": work_ratio,
                "aggregate_quality_gate": aggregate <= float(gates["oracle_aggregate_output_relative_l2"]),
                "worst_quality_gate": worst <= float(gates["oracle_worst_record_output_relative_l2"]),
                "work_gate": work_ratio <= float(gates["max_extra_work_ratio"]),
                "generator_gate": int(summary["generators"]) <= int(gates["max_generators"]),
                "payload_gate": dynamic_scalars <= int(gates["max_dynamic_scalars"]),
            }
        )
        summary["requested_gate_pass"] = (
            summary["aggregate_quality_gate"]
            and summary["worst_quality_gate"]
            and summary["work_gate"]
            and summary["generator_gate"]
        )
        summary["anti_tautology_gate_pass"] = summary["requested_gate_pass"] and summary["payload_gate"]
        summaries.append(summary)
    return summaries


def make_decision(
    pregate_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    protocol: dict[str, object],
) -> dict[str, object]:
    restricted = {"givens", "householder", "orthogonal_bcm", "dcd", "butterfly"}
    holdouts = tuple(str(raw["name"]) for raw in protocol["holdouts"])
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        if row["family"] in restricted:
            grouped[(row["cell"], row["method"], row["density"], row["family"], row["generators"])].append(row)

    candidates = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        seen = {str(row["split"]) for row in group}
        all_holdouts = seen == set(holdouts)
        requested = all_holdouts and all(bool(row["requested_gate_pass"]) for row in group)
        deployable = all_holdouts and all(bool(row["anti_tautology_gate_pass"]) for row in group)
        candidates.append(
            {
                "cell": key[0],
                "method": key[1],
                "density": key[2],
                "family": key[3],
                "generators": key[4],
                "holdouts_seen": sorted(seen),
                "all_holdouts_present": all_holdouts,
                "requested_gate_all_holdouts": requested,
                "anti_tautology_gate_all_holdouts": deployable,
                "worst_output_relative_l2": max(float(row["worst_record_output_relative_l2"]) for row in group),
                "worst_aggregate_output_relative_l2": max(float(row["aggregate_output_relative_l2"]) for row in group),
                "dynamic_scalars": max(int(row["dynamic_scalars"]) for row in group),
                "extra_work_ratio": max(float(row["extra_work_ratio"]) for row in group),
            }
        )
    deployable = [row for row in candidates if row["anti_tautology_gate_all_holdouts"]]
    requested_only = [row for row in candidates if row["requested_gate_all_holdouts"]]
    if deployable:
        verdict = "GO_TRAIN_QKV_CONDITIONED_ROTATION_GATE"
        best = min(deployable, key=lambda row: (row["worst_output_relative_l2"], row["dynamic_scalars"]))
    elif requested_only:
        verdict = "STOP_LOW_DIMENSION_ROTATION_DYNAMIC_PAYLOAD_DEGENERATES_TO_BASIS_GENERATION"
        best = min(requested_only, key=lambda row: (row["worst_output_relative_l2"], row["dynamic_scalars"]))
    else:
        verdict = "STOP_RESTRICTED_ROTATION_ORACLE_FAILS"
        best = None
    failed_cells = [row for row in pregate_rows if not bool(row["adaptive_pregate_pass"])]
    return {
        "verdict": verdict,
        "best_candidate": best,
        "candidate_gates": candidates,
        "adaptive_pregate_failures": failed_cells,
        "gate_definitions": protocol["gates"],
        "interpretation": {
            "full_procrustes": "tautological control that exactly exposes the adaptive rank-r ceiling",
            "restricted_families": "fit post-hoc to held-out AV defects; no Q/K/V parameter generator exists yet",
            "payload_guard": "prevents M dense Householder/DCD generators from being mislabeled as a low-dimensional gate",
            "speed_warning": "arithmetic work is not measured H200 latency and excludes parameter-generation overhead",
            "stop_rule": "no Q/K/V-conditioned gate is trained unless both prompt and seed holdouts pass all gates",
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    protocol_path = args.protocol_config.resolve()
    capture_index = args.capture_index.resolve()
    protocol = load_protocol(protocol_path)
    scope = protocol["scope"]
    cells = tuple(Cell(str(raw["name"]), int(raw["layer"]), int(raw["sampling_step"])) for raw in scope["cells"])
    sample_ids = tuple(map(str, scope["sample_ids"]))
    captures = resolve_capture_rows(capture_index, sample_ids, cells, str(scope["branch"]))
    require_fresh_output_dir(args.output_dir)
    run_id = f"restricted-rotation-{uuid.uuid4().hex[:12]}"
    event_log = JsonlEventLog(args.output_dir / "events.jsonl", run_id)
    event_log.emit("run_started", run_kind=args.run_kind, config_sha256=file_sha256(protocol_path))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    fingerprints = [capture_fingerprint(Path(item["path"]), args.capture_hash_mode) for item in captures]
    event_log.emit("capture_fingerprints_complete", captures=len(fingerprints), mode=args.capture_hash_mode)

    records: list[TailRecord] = []
    for index, capture in enumerate(captures):
        capture_records = process_capture(capture, protocol, device)
        records.extend(capture_records)
        event_log.emit(
            "capture_processed",
            index=index,
            captures=len(captures),
            sample_id=capture["sample_id"],
            cell=capture["cell"].name,
            records=len(capture_records),
        )
        print(
            f"[rotation] capture {index + 1}/{len(captures)} "
            f"sample={capture['sample_id']} cell={capture['cell'].name}",
            flush=True,
        )

    pregate_rows = aggregate_pregate(records, protocol)
    eligible = {
        (row["cell"], row["method"], float(row["density"]))
        for row in pregate_rows
        if bool(row["adaptive_pregate_pass"])
    }
    lookup = {_record_lookup_key(record): record for record in records}
    if len(lookup) != len(records):
        raise RuntimeError("duplicate tail record keys detected")
    rank = int(scope["rank"])
    family_config = protocol["families"]
    candidate_rows: list[dict[str, object]] = []
    source_cache: dict[tuple[object, ...], torch.Tensor] = {}

    for holdout in protocol["holdouts"]:
        split_name = str(holdout["name"])
        source_ids = tuple(map(str, holdout["source"]))
        target_ids = set(map(str, holdout["target"]))
        grouped_targets: dict[tuple[str, str, float], list[TailRecord]] = defaultdict(list)
        for record in records:
            if record.sample_id in target_ids:
                grouped_targets[(record.cell.name, record.method, record.density)].append(record)

        for config_key, target_records in sorted(grouped_targets.items()):
            sources: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            defects: list[torch.Tensor] = []
            for record in target_records:
                cache_key = (
                    split_name,
                    record.cell.name,
                    record.method,
                    record.density,
                    record.head,
                    record.tile_index,
                )
                if cache_key not in source_cache:
                    source_cache[cache_key] = source_basis(
                        lookup, source_ids, record, rank, device
                    ).cpu()
                source = source_cache[cache_key].to(device)
                target = record.target_basis.to(device)
                sources.append(source)
                targets.append(target)
                defects.append(record.defect.to(device))
                for family, generators, basis in (
                    ("frozen", 0, source),
                    ("adaptive", 0, target),
                    ("full_procrustes", target.shape[0], target),
                ):
                    candidate_rows.append(
                        candidate_row(record, split_name, family, generators, basis, protocol)
                    )

            if config_key not in eligible:
                event_log.emit("restricted_families_skipped", split=split_name, config=list(config_key), reason="adaptive_pregate_fail")
                continue

            generator_counts = tuple(map(int, family_config["generator_counts"]))
            for record, source, target, defect in zip(target_records, sources, targets, defects):
                givens, _ = greedy_givens_prefixes(
                    source,
                    defect,
                    generator_counts,
                    grid_points=int(family_config["givens"]["grid_points"]),
                )
                householder, _ = householder_prefixes(source, target, generator_counts)
                for generators in generator_counts:
                    candidate_rows.append(
                        candidate_row(record, split_name, "givens", generators, givens[generators], protocol)
                    )
                    candidate_rows.append(
                        candidate_row(
                            record,
                            split_name,
                            "householder",
                            generators,
                            householder[generators],
                            protocol,
                        )
                    )

            source_batch = torch.stack(sources)
            defect_batch = torch.stack(defects)
            optimized_families = tuple(
                map(
                    str,
                    family_config.get(
                        "optimized_families",
                        ("orthogonal_bcm", "dcd", "butterfly"),
                    ),
                )
            )
            unsupported = set(optimized_families) - {
                "orthogonal_bcm",
                "dcd",
                "butterfly",
            }
            if unsupported:
                raise ValueError(f"unsupported optimized families: {sorted(unsupported)}")
            for family in optimized_families:
                for generators in generator_counts:
                    fitted = fit_parameterized_rotation(
                        family,
                        source_batch,
                        defect_batch,
                        generators,
                        steps=int(family_config["optimization_steps"]),
                        learning_rate=float(family_config["learning_rate"]),
                        restarts=int(family_config["restarts"]),
                        block_size=int(family_config["orthogonal_bcm"]["block_size"]),
                        max_log_scale=float(family_config["dcd"]["max_log_scale"]),
                        seed=stable_seed(run_id if args.run_kind == "smoke" else object_sha256(protocol), split_name, config_key, family, generators),
                    )
                    for record, basis in zip(target_records, fitted):
                        candidate_rows.append(
                            candidate_row(record, split_name, family, generators, basis, protocol)
                        )
                    event_log.emit(
                        "structured_family_complete",
                        split=split_name,
                        config=list(config_key),
                        family=family,
                        generators=generators,
                    )
                    print(
                        f"[rotation] split={split_name} config={config_key} "
                        f"family={family} M={generators}",
                        flush=True,
                    )

    summaries = summarize_candidates(candidate_rows, protocol)
    decision = make_decision(pregate_rows, summaries, protocol)
    atomic_write_csv(args.output_dir / "adaptive_pregate.csv", pregate_rows)
    atomic_write_csv(args.output_dir / "rotation_records.csv", candidate_rows)
    atomic_write_csv(args.output_dir / "rotation_summary.csv", summaries)
    atomic_write_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": args.run_kind,
        "claim_boundary": protocol["claim_boundary"],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "protocol": protocol,
        "protocol_sha256": file_sha256(protocol_path),
        "capture_index": str(capture_index),
        "capture_index_sha256": file_sha256(capture_index),
        "capture_fingerprints": fingerprints,
        "records": len(records),
        "candidate_records": len(candidate_rows),
        "eligible_capacity_cells": [list(key) for key in sorted(eligible)],
        "methodology": {
            "granularity": "layer x step bucket x head x contiguous 64-query tile",
            "critical_path": "per-record block mask selected with dense attention; renormalized sparse output",
            "tail_objective": "direct pre-output-projection AV residual relative L2",
            "coefficients": "oracle projection coefficients for every family",
            "rotation_access": "restricted transforms are fit directly to held-out target defects",
            "adaptive_pregate": "restricted families run only where adaptive rank-r meets the registered 0.5%/1% gates",
            "anti_tautology": "dynamic scalar payload is checked separately from generator count and arithmetic work",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
            "execution_resource_note": args.execution_resource_note,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    artifact_hashes = {
        name: file_sha256(args.output_dir / name)
        for name in (
            "adaptive_pregate.csv",
            "rotation_records.csv",
            "rotation_summary.csv",
            "decision.json",
            "manifest.json",
        )
    }
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {
            "run_id": run_id,
            "completed_unix": time.time(),
            "artifact_sha256": artifact_hashes,
            "verdict": decision["verdict"],
        },
    )
    event_log.emit("run_completed", verdict=decision["verdict"])
    print(f"[rotation] verdict={decision['verdict']}", flush=True)
    print(f"[rotation] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
