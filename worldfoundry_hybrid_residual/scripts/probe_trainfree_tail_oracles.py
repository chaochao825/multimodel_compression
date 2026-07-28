#!/usr/bin/env python3
"""Run bounded value-aware, polynomial, and covariance Attention tail oracles."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    object_sha256,
    require_fresh_output_dir,
)
from probe_block_moment_marginal import (
    parse_strings,
    read_capture_rows,
    read_head_roles,
)
from probe_dynamic_sparse_lowrank_oracle import aligned_query_tile_starts
from probe_nystrom_sparse_tail import (
    assert_capture_metadata_unchanged,
    capture_provenance,
    validate_capture_payload,
)
from trainfree_tail_oracle_core import (
    coreset_tail_output,
    covariance_query_work_ratio,
    covariance_tail_output,
    finite_diagnostics,
    oracle_mass_block_selection,
    polynomial_tail_output,
    prepare_group_moments,
    stable_seed,
    token_coordinates,
)


FAMILIES = (
    "value_aware_coreset",
    "residual_tail_polynomial",
    "lowrank_covariance",
)


def parse_optional_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--head-stats-index", type=Path)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--run-kind",
        choices=("registered", "smoke"),
        default="registered",
    )
    parser.add_argument(
        "--families",
        type=parse_strings,
        default=FAMILIES,
        help="family subset; registered runs must use every protocol family",
    )
    parser.add_argument(
        "--heads",
        type=parse_optional_ints,
        default=(),
        help="head subset allowed only for smoke runs",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="capture limit allowed only for smoke runs",
    )
    parser.add_argument(
        "--query-tiles-override",
        type=int,
        default=0,
        help="smoke-only override; registered runs use the protocol value",
    )
    parser.add_argument(
        "--capture-hash-mode",
        choices=("metadata", "sha256"),
        default="sha256",
    )
    parser.add_argument(
        "--execution-resource-note",
        default="unspecified numerical capacity run; no latency claim",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported train-free tail protocol schema")
    required = {
        "scope",
        "value_aware_coreset",
        "residual_tail_polynomial",
        "lowrank_covariance",
        "gates",
        "claim_boundary",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing fields: {sorted(missing)}")
    scope = protocol["scope"]
    if not isinstance(scope, dict):
        raise TypeError("protocol scope must be an object")
    if any(not 0.0 < float(value) < 1.0 for value in scope["densities"]):
        raise ValueError("all protocol densities must lie in (0, 1)")
    return protocol


def selected_rows(
    args: argparse.Namespace,
    protocol: dict[str, object],
) -> list[dict[str, object]]:
    scope = protocol["scope"]
    reader_args = SimpleNamespace(
        capture_index=args.capture_index,
        layers=tuple(int(value) for value in scope["layers"]),
        steps=tuple(int(value) for value in scope["sampling_steps"]),
        branches=tuple(str(value) for value in scope["branches"]),
    )
    rows = read_capture_rows(reader_args)
    expected_samples = set(map(str, scope["sample_ids"]))
    observed_samples = {str(row["sample_id"]) for row in rows}
    if args.run_kind == "registered":
        if observed_samples != expected_samples:
            raise ValueError(
                "registered capture set differs from protocol: "
                f"observed={sorted(observed_samples)}, expected={sorted(expected_samples)}"
            )
        if args.heads or args.sample_limit or args.query_tiles_override:
            raise ValueError("registered runs cannot use smoke-only scope overrides")
        if tuple(args.families) != FAMILIES:
            raise ValueError("registered runs must evaluate every protocol family")
    else:
        rows = rows[: args.sample_limit or len(rows)]
    keys = [
        (row["sample_id"], row["sampling_step"], row["branch"], row["layer"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("capture selection contains duplicate cell keys")
    return rows


def grid_from_payload(payload: dict[str, object]) -> tuple[int, int, int]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("capture metadata must be a dictionary")
    grid = metadata.get("grid_size")
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        raise ValueError("capture metadata does not contain a T/H/W grid")
    return tuple(int(value) for value in grid)


def base_row(
    row: dict[str, object],
    head: int,
    role: str,
    tile_start: int,
    density: float,
    selected_keys: torch.Tensor,
    selected_mass: float,
    reference_sq: float,
) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "prompt_index": row["prompt_index"],
        "seed": row["seed"],
        "sampling_step": row["sampling_step"],
        "timestep": row["timestep"],
        "branch": row["branch"],
        "layer": row["layer"],
        "head": head,
        "head_role_diagnostic_only": role,
        "query_tile_start": tile_start,
        "density": density,
        "selected_key_fraction": float(selected_keys.float().mean()),
        "selected_attention_mass": selected_mass,
        "reference_sq": reference_sq,
        "diagnostic_oracle": True,
        "claim_boundary": "posthoc_function_class_capacity_only",
    }


def evaluated_row(
    base: dict[str, object],
    approximation: torch.Tensor,
    reference: torch.Tensor,
    diagnostics: dict[str, float],
    *,
    family: str,
    variant: str,
    landmarks: int = 0,
    order: int = 0,
    rank: int = 0,
    components: int = 0,
    restart: int = 0,
    projected_work_ratio: float = float("nan"),
    omitted_online_cost: str = "",
) -> dict[str, object]:
    if not bool(torch.isfinite(approximation).all()):
        raise ValueError(f"non-finite approximation for {family}/{variant}")
    finite_diagnostics(diagnostics)
    residual_sq = float((reference - approximation).square().sum())
    reference_sq = float(base["reference_sq"])
    return {
        **base,
        "family": family,
        "variant": variant,
        "landmarks": landmarks,
        "order": order,
        "rank": rank,
        "components": components,
        "restart": restart,
        "residual_sq": residual_sq,
        "output_relative_l2": math.sqrt(residual_sq / max(reference_sq, 1e-30)),
        "projected_query_work_ratio": projected_work_ratio,
        "arithmetic_speedup_upper_bound": (
            1.0 / projected_work_ratio
            if math.isfinite(projected_work_ratio) and projected_work_ratio > 0
            else float("nan")
        ),
        "omitted_online_cost": omitted_online_cost,
        **diagnostics,
    }


@torch.inference_mode()
def process_capture(
    row: dict[str, object],
    args: argparse.Namespace,
    protocol: dict[str, object],
    roles: dict[tuple[str, int, str, int, int], str],
    device: torch.device,
) -> list[dict[str, object]]:
    path = Path(row["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_capture_payload(payload, path, row)
    q_all = payload["q"][0]
    k_all = payload["k"][0]
    v_all = payload["v"][0]
    tokens, heads, dimension = q_all.shape
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    scope = protocol["scope"]
    tile_count = args.query_tiles_override or int(scope["query_tiles"])
    starts = aligned_query_tile_starts(tokens, int(scope["query_tile_size"]), tile_count)
    block_size = int(scope["block_size"])
    coordinates = token_coordinates(
        grid_from_payload(payload), device=device, dtype=torch.float32
    )
    if coordinates.shape[0] != tokens:
        raise ValueError("T/H/W grid product does not match token count")
    head_indices = tuple(args.heads) if args.heads else tuple(range(heads))
    if len(set(head_indices)) != len(head_indices) or any(
        head < 0 or head >= heads for head in head_indices
    ):
        raise ValueError(f"invalid requested head set: {head_indices}")

    covariance_config = protocol["lowrank_covariance"]
    max_rank = max(int(value) for value in covariance_config["ranks"])
    output: list[dict[str, object]] = []
    for head in head_indices:
        all_queries = q_all[:, head].to(device=device, dtype=torch.float32)
        keys = k_all[:, head].to(device=device, dtype=torch.float32)
        values = v_all[:, head].to(device=device, dtype=torch.float32)
        if not all(
            bool(torch.isfinite(tensor).all()) for tensor in (all_queries, keys, values)
        ):
            raise ValueError(f"non-finite Q/K/V in {path}, head={head}")
        covariance_moments = {}
        if "lowrank_covariance" in args.families:
            for components in covariance_config["components_per_block"]:
                covariance_moments[int(components)] = prepare_group_moments(
                    keys,
                    values,
                    block_size,
                    int(components),
                    max_rank,
                )
        role = roles.get(
            (
                str(row["sample_id"]),
                int(row["sampling_step"]),
                str(row["branch"]),
                int(row["layer"]),
                head,
            ),
            "unknown",
        )
        for tile_start in starts:
            queries = all_queries[tile_start : tile_start + int(scope["query_tile_size"])]
            scores = queries @ keys.T * scale
            probabilities = torch.softmax(scores, dim=1)
            reference = probabilities @ values
            reference_sq = float(reference.square().sum())
            for density in map(float, scope["densities"]):
                selected_blocks, selected_keys = oracle_mass_block_selection(
                    probabilities, block_size, density
                )
                selected_mass = float(probabilities[:, selected_keys].sum(1).mean())
                base = base_row(
                    row,
                    head,
                    role,
                    tile_start,
                    density,
                    selected_keys,
                    selected_mass,
                    reference_sq,
                )

                if "residual_tail_polynomial" in args.families:
                    polynomial = protocol["residual_tail_polynomial"]
                    for center in polynomial["centers"]:
                        for order in polynomial["orders"]:
                            approximation, diagnostics = polynomial_tail_output(
                                scores,
                                values,
                                selected_keys,
                                int(order),
                                str(center),
                            )
                            output.append(
                                evaluated_row(
                                    base,
                                    approximation,
                                    reference,
                                    diagnostics,
                                    family="residual_tail_polynomial",
                                    variant=str(center),
                                    order=int(order),
                                    omitted_online_cost=(
                                        "TensorSketch/random-Maclaurin feature realization is not implemented"
                                    ),
                                )
                            )

                if "value_aware_coreset" in args.families:
                    coreset = protocol["value_aware_coreset"]
                    for variant in coreset["variants"]:
                        for landmarks in coreset["landmarks"]:
                            for restart in range(int(coreset["restarts"])):
                                seed = stable_seed(
                                    row["sample_id"],
                                    row["sampling_step"],
                                    row["layer"],
                                    head,
                                    tile_start,
                                    density,
                                    variant,
                                    landmarks,
                                    restart,
                                )
                                approximation, diagnostics = coreset_tail_output(
                                    queries,
                                    keys,
                                    values,
                                    coordinates,
                                    scores,
                                    probabilities,
                                    reference,
                                    selected_keys,
                                    clusters=int(landmarks),
                                    variant=str(variant),
                                    scale=scale,
                                    iterations=int(coreset["kmeans_iterations"]),
                                    fit_tokens=int(coreset["fit_tokens"]),
                                    seed=seed,
                                )
                                active_groups = int(diagnostics["active_tail_groups"])
                                work_ratio = (
                                    int(selected_keys.sum()) + active_groups
                                ) / tokens
                                output.append(
                                    evaluated_row(
                                        base,
                                        approximation,
                                        reference,
                                        diagnostics,
                                        family="value_aware_coreset",
                                        variant=str(variant),
                                        landmarks=int(landmarks),
                                        restart=restart,
                                        projected_work_ratio=work_ratio,
                                        omitted_online_cost=(
                                            "dense-oracle leverage and online weighted k-means/assignment"
                                        ),
                                    )
                                )

                if "lowrank_covariance" in args.families:
                    for components, moments in covariance_moments.items():
                        variants: list[tuple[str, int]] = []
                        if covariance_config["include_centroid"]:
                            variants.append(("centroid", 0))
                        if covariance_config["include_diagonal_gaussian"]:
                            variants.append(("diag_gaussian", 0))
                        variants.extend(
                            ("lowrank_gaussian", int(rank))
                            for rank in covariance_config["ranks"]
                        )
                        for variant, rank in variants:
                            approximation, diagnostics = covariance_tail_output(
                                queries,
                                scores,
                                values,
                                selected_blocks,
                                selected_keys,
                                moments,
                                variant=variant,
                                rank=rank,
                                scale=scale,
                            )
                            work_ratio = covariance_query_work_ratio(
                                int(selected_keys.sum()),
                                tokens,
                                int(diagnostics["active_tail_groups"]),
                                variant,
                                rank,
                            )
                            output.append(
                                evaluated_row(
                                    base,
                                    approximation,
                                    reference,
                                    diagnostics,
                                    family="lowrank_covariance",
                                    variant=variant,
                                    rank=rank,
                                    components=components,
                                    projected_work_ratio=work_ratio,
                                    omitted_online_cost=(
                                        "online group moment formation and batched K SVD"
                                    ),
                                )
                            )
        del all_queries, keys, values, covariance_moments
    del payload, q_all, k_all, v_all, coordinates
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


CONFIG_FIELDS = (
    "family",
    "variant",
    "density",
    "landmarks",
    "order",
    "rank",
    "components",
    "restart",
)
RECORD_FIELDS = ("sample_id", "head")


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else float("nan")


def aggregate_rows(
    rows: list[dict[str, object]],
    group_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        errors = [
            float(
                row.get(
                    "output_relative_l2",
                    row.get("aggregate_output_relative_l2", float("nan")),
                )
            )
            for row in group
        ]
        if any(not math.isfinite(error) for error in errors):
            raise ValueError(f"non-finite grouped error for key={key}")
        mean_work = mean_finite(
            [
                float(
                    row.get(
                        "projected_query_work_ratio",
                        row.get("projected_query_work_ratio_mean", float("nan")),
                    )
                )
                for row in group
            ]
        )
        output.append(
            {
                **dict(zip(group_fields, key)),
                "records": len(group),
                "residual_sq": residual_sq,
                "reference_sq": reference_sq,
                "aggregate_output_relative_l2": math.sqrt(
                    residual_sq / max(reference_sq, 1e-30)
                ),
                "record_error_max": max(errors),
                "projected_query_work_ratio_mean": mean_work,
                "arithmetic_speedup_upper_bound": (
                    1.0 / mean_work
                    if math.isfinite(mean_work) and mean_work > 0
                    else float("nan")
                ),
                "selected_attention_mass_mean": statistics.mean(
                    float(
                        row.get(
                            "selected_attention_mass",
                            row.get("selected_attention_mass_mean", float("nan")),
                        )
                    )
                    for row in group
                ),
                "negative_tail_weight_fraction_mean": mean_finite(
                    [
                        float(
                            row.get(
                                "negative_tail_weight_fraction",
                                row.get(
                                    "negative_tail_weight_fraction_mean", float("nan")
                                ),
                            )
                        )
                        for row in group
                    ]
                ),
                "tail_score_range_mean": mean_finite(
                    [
                        float(
                            row.get(
                                "tail_score_range_mean",
                                row.get("tail_score_range_mean_mean", float("nan")),
                            )
                        )
                        for row in group
                    ]
                ),
            }
        )
    return output


def build_oracle_envelope(
    head_rows: list[dict[str, object]],
    gates: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in head_rows:
        groups[(str(row["family"]), str(row["sample_id"]), int(row["head"]))].append(row)
    selected = []
    for (family, sample_id, head), candidates in sorted(groups.items()):
        best = min(
            candidates,
            key=lambda row: float(row["residual_sq"]) / max(float(row["reference_sq"]), 1e-30),
        )
        selected.append(
            {
                **best,
                "oracle_selection": "posthoc_per_record_minimum_output_error",
                "envelope_family": family,
                "envelope_sample_id": sample_id,
                "envelope_head": head,
            }
        )
    summaries = aggregate_rows(selected, ("family",))
    aggregate_target = float(gates["oracle_aggregate_output_relative_l2"])
    worst_target = float(gates["oracle_worst_record_output_relative_l2"])
    for summary in summaries:
        summary["oracle_aggregate_target"] = aggregate_target
        summary["oracle_worst_target"] = worst_target
        summary["oracle_quality_gate"] = (
            "PASS"
            if float(summary["aggregate_output_relative_l2"]) <= aggregate_target
            and float(summary["record_error_max"]) <= worst_target
            else "FAIL"
        )
        summary["claim_boundary"] = "posthoc_per_record_oracle_capacity_only"
    return selected, summaries


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: list(value)
        if isinstance(value, tuple)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    args.capture_index = args.capture_index.resolve()
    args.protocol_config = args.protocol_config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.head_stats_index is not None:
        args.head_stats_index = args.head_stats_index.resolve()
    unknown_families = set(args.families) - set(FAMILIES)
    if unknown_families:
        raise ValueError(f"unknown families: {sorted(unknown_families)}")
    if args.sample_limit < 0 or args.query_tiles_override < 0:
        raise ValueError("smoke limits cannot be negative")
    protocol = load_protocol(args.protocol_config)
    rows = selected_rows(args, protocol)
    require_fresh_output_dir(args.output_dir)

    config = {
        "arguments": serializable_args(args),
        "protocol": protocol,
        "protocol_sha256": file_sha256(args.protocol_config),
    }
    config_hash = object_sha256(config)
    run_id = f"tail-oracle-{config_hash[:12]}-{int(time.time())}"
    log = JsonlEventLog(args.output_dir / "progress.jsonl", run_id)
    atomic_write_json(
        args.output_dir / "run_state.json",
        {"status": "RUNNING", "run_id": run_id, "config_sha256": config_hash},
    )
    log.emit("input_hashing_started", captures=len(rows), mode=args.capture_hash_mode)
    provenance = capture_provenance(rows, args.capture_hash_mode)
    atomic_write_json(
        args.output_dir / "input_manifest.json",
        {
            "schema_version": 1,
            "capture_index": str(args.capture_index),
            "capture_index_sha256": file_sha256(args.capture_index),
            "capture_hash_mode": args.capture_hash_mode,
            "captures": provenance,
        },
    )
    log.emit("input_hashing_completed", captures=len(rows))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    roles = read_head_roles(args.head_stats_index)
    detail: list[dict[str, object]] = []
    started = time.time()
    for index, row in enumerate(rows):
        capture_started = time.time()
        capture_rows = process_capture(row, args, protocol, roles, device)
        detail.extend(capture_rows)
        elapsed = time.time() - capture_started
        log.emit(
            "capture_completed",
            capture_index=index,
            sample_id=row["sample_id"],
            rows=len(capture_rows),
            elapsed_seconds=elapsed,
        )
        print(
            f"[tail-oracle] {index + 1}/{len(rows)} sample={row['sample_id']} "
            f"rows={len(capture_rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    assert_capture_metadata_unchanged(provenance)

    head_rows = aggregate_rows(detail, RECORD_FIELDS + CONFIG_FIELDS)
    config_summary = aggregate_rows(head_rows, CONFIG_FIELDS)
    envelope_heads, envelope_summary = build_oracle_envelope(
        head_rows, protocol["gates"]
    )
    any_pass = any(row["oracle_quality_gate"] == "PASS" for row in envelope_summary)
    decision = {
        "schema_version": 1,
        "run_kind": args.run_kind,
        "status": "GO_TO_DEPLOYABLE_GENERATOR_ONLY" if any_pass else "STOP_TRAINFREE_TAIL",
        "oracle_gate_passed_families": [
            row["family"]
            for row in envelope_summary
            if row["oracle_quality_gate"] == "PASS"
        ],
        "claim_boundary": protocol["claim_boundary"],
        "kernel_authorized": False,
        "latency_claim_authorized": False,
    }
    if args.run_kind == "smoke":
        decision["status"] = "NOT_EVALUATED_SMOKE"
    atomic_write_csv(args.output_dir / "trainfree_tail_oracle_tiles.csv", detail)
    atomic_write_csv(args.output_dir / "trainfree_tail_oracle_heads.csv", head_rows)
    atomic_write_csv(args.output_dir / "trainfree_tail_oracle_summary.csv", config_summary)
    atomic_write_csv(args.output_dir / "trainfree_tail_oracle_envelope_heads.csv", envelope_heads)
    atomic_write_csv(args.output_dir / "trainfree_tail_oracle_envelope_summary.csv", envelope_summary)
    atomic_write_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": args.run_kind,
        "scope": "stronger train-free sparse-critical residual-tail function-class oracle",
        "arguments": serializable_args(args),
        "protocol": protocol,
        "protocol_sha256": file_sha256(args.protocol_config),
        "config_sha256": config_hash,
        "capture_index_sha256": file_sha256(args.capture_index),
        "input_manifest_sha256": file_sha256(args.output_dir / "input_manifest.json"),
        "captures": len(rows),
        "families": list(args.families),
        "claim_boundary": protocol["claim_boundary"],
        "normalization": "exact and approximate numerator/denominator terms share one normalization",
        "critical_router": "tile-shared dense attention mass oracle; not deployable",
        "oracle_envelope": "per sample/head post-hoc minimum output error; not a frozen test estimate",
        "cost_boundary": (
            "query arithmetic is reported where defined; online clustering, moment/SVD formation, "
            "kernel launch, gather, and fusion costs are omitted"
        ),
        "execution_resource_note": args.execution_resource_note,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "elapsed_seconds": time.time() - started,
        "decision": decision,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    success = {
        "status": "SUCCESS",
        "run_id": run_id,
        "run_kind": args.run_kind,
        "config_sha256": config_hash,
        "detail_rows": len(detail),
        "head_rows": len(head_rows),
        "elapsed_seconds": time.time() - started,
        "decision": decision["status"],
    }
    atomic_write_json(args.output_dir / "run_state.json", success)
    atomic_write_json(args.output_dir / "SUCCESS.json", success)
    log.emit("run_completed", **success)
    print(json.dumps(success, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
