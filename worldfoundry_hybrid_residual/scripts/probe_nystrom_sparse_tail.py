#!/usr/bin/env python3
"""Probe current-Q/K/V Nystrom tails with an exact sparse-critical branch.

The deployable candidates never read dense probabilities when constructing an
output. Dense attention is computed only as the evaluation reference.  Every
deployable low-rank path is evaluated in an associatively executable form;
full-matrix clamping and dense-reference-mass rows are diagnostic only.
"""

from __future__ import annotations

import argparse
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_split_protocols,
    object_sha256,
    require_fresh_output_dir,
)
from probe_block_moment_marginal import (
    choose_blocks,
    moment_logits,
    padded_groups,
    parse_floats,
    parse_ints,
    parse_strings,
    read_capture_rows,
    read_head_roles,
)
from probe_dynamic_sparse_lowrank_oracle import aligned_query_tile_starts


DEPLOYABLE_METHODS = frozenset(
    {
        "nystrom_signed",
        "landmark_linear",
        "proxy_mass_nystrom_mixture",
        "proxy_mass_landmark_partition",
    }
)
DIAGNOSTIC_METHODS = frozenset(
    {
        "nystrom_nonnegative_clamped",
        "dense_mass_nystrom_mixture_diagnostic",
        "dense_mass_landmark_partition_diagnostic",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--head-stats-index", type=Path)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_ints, default=(0, 14, 29))
    parser.add_argument("--steps", type=parse_ints, default=(0, 9, 19))
    parser.add_argument("--branches", type=parse_strings, default=("cond",))
    parser.add_argument(
        "--heads",
        type=parse_ints,
        default=(),
        help="optional head subset for smoke/debug; empty evaluates every head",
    )
    parser.add_argument("--query-tile-size", type=int, default=64)
    parser.add_argument("--query-tiles", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--landmarks", type=parse_ints, default=(32, 64, 128))
    parser.add_argument("--landmark-modes", type=parse_strings, default=("segment",))
    parser.add_argument("--pinv-rtols", type=parse_floats, default=(1e-4,))
    parser.add_argument("--densities", type=parse_floats, default=(0.125, 0.25))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution-resource-note",
        default="unspecified",
        help="records co-tenancy/dedication; numerical metrics never imply latency",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--capture-hash-mode",
        choices=("metadata", "sha256"),
        default="sha256",
        help="strong hashes cost extra sequential I/O but make payload provenance exact",
    )
    parser.add_argument("--aggregate-target", type=float, default=0.01)
    parser.add_argument("--record-target", type=float, default=0.02)
    parser.add_argument("--speed-target", type=float, default=1.5)
    parser.add_argument("--max-work-ratio", type=float, default=0.50)
    return parser.parse_args()


def segment_landmarks(tensor: torch.Tensor, count: int) -> torch.Tensor:
    tokens, dimension = tensor.shape
    if not 0 < count <= tokens:
        raise ValueError(f"landmark count must lie in [1, {tokens}], got {count}")
    group_ids = torch.div(
        torch.arange(tokens, device=tensor.device) * count,
        tokens,
        rounding_mode="floor",
    )
    sums = tensor.new_zeros((count, dimension))
    sums.index_add_(0, group_ids, tensor)
    counts = torch.bincount(group_ids, minlength=count).to(tensor.dtype)
    return sums / counts[:, None]


def make_landmarks(tensor: torch.Tensor, count: int, mode: str) -> torch.Tensor:
    if mode == "segment":
        return segment_landmarks(tensor, count)
    if mode == "uniform":
        indices = torch.linspace(
            0,
            tensor.shape[0] - 1,
            count,
            dtype=torch.float64,
            device=tensor.device,
        ).round().long()
        return tensor.index_select(0, indices)
    raise ValueError(f"unsupported landmark mode: {mode}")


def nystrom_factors(
    all_queries: torch.Tensor,
    keys: torch.Tensor,
    landmark_count: int,
    landmark_mode: str,
    scale: float,
    pinv_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    query_landmarks = make_landmarks(all_queries, landmark_count, landmark_mode)
    key_landmarks = make_landmarks(keys, landmark_count, landmark_mode)
    middle = torch.softmax(query_landmarks @ key_landmarks.T * scale, dim=1)
    right = torch.softmax(query_landmarks @ keys.T * scale, dim=1)
    singular_values = torch.linalg.svdvals(middle)
    threshold = pinv_rtol * float(singular_values.max())
    effective_rank = int((singular_values > threshold).sum())
    smallest = float(singular_values.min())
    condition = float(singular_values.max()) / max(smallest, 1e-30)
    inverse = torch.linalg.pinv(middle, rtol=pinv_rtol)
    return key_landmarks, inverse, right, {
        "middle_condition_number": condition,
        "middle_effective_rank": effective_rank,
        "middle_smallest_singular": smallest,
    }


def nystrom_components(
    queries: torch.Tensor,
    key_landmarks: torch.Tensor,
    inverse: torch.Tensor,
    right: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    left = torch.softmax(queries @ key_landmarks.T * scale, dim=1)
    signed = left @ inverse @ right
    negative = signed.clamp_max(0).abs()
    absolute_mass = signed.abs().sum().clamp_min(1e-30)
    nonnegative = signed.clamp_min(0)
    nonnegative = nonnegative / nonnegative.sum(dim=1, keepdim=True).clamp_min(1e-30)
    diagnostics = {
        "negative_entry_fraction": float((signed < 0).float().mean()),
        "negative_absolute_mass_ratio": float(negative.sum() / absolute_mass),
        "signed_row_sum_error": float((signed.sum(dim=1) - 1).abs().mean()),
    }
    return left, signed, nonnegative, diagnostics


def key_mask_from_blocks(
    selected_blocks: torch.Tensor, tokens: int, block_size: int
) -> torch.Tensor:
    block_ids = torch.arange(tokens, device=selected_blocks.device) // block_size
    return selected_blocks.index_select(0, block_ids)


def exact_selected_output(
    exact_scores: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
) -> torch.Tensor:
    if not bool(selected_keys.any()):
        return torch.zeros(
            (exact_scores.shape[0], values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
    selected_scores = exact_scores[:, selected_keys]
    selected_values = values[selected_keys]
    exact_conditional = torch.softmax(selected_scores, dim=1)
    return exact_conditional @ selected_values


def mass_mixture_output(
    exact_scores: torch.Tensor,
    tail_output: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_mass: torch.Tensor,
) -> torch.Tensor:
    selected_output = exact_selected_output(exact_scores, values, selected_keys)
    if tail_output.shape != selected_output.shape:
        raise ValueError(
            "tail output shape must match exact selected output: "
            f"tail={tuple(tail_output.shape)}, exact={tuple(selected_output.shape)}"
        )
    if selected_mass.shape != (exact_scores.shape[0],):
        raise ValueError(
            f"selected mass must have shape {(exact_scores.shape[0],)}, "
            f"got {tuple(selected_mass.shape)}"
        )
    selected_mass = selected_mass.clamp(0, 1)
    return selected_mass[:, None] * selected_output + (
        1 - selected_mass
    )[:, None] * tail_output


def landmark_partition_output(
    exact_scores: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    right_values: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_mass: torch.Tensor,
) -> torch.Tensor:
    """Mix exact critical output with a positive low-rank marginal tail."""

    selected_output = exact_selected_output(exact_scores, values, selected_keys)
    if bool(selected_keys.any()):
        selected_right = right[:, selected_keys]
        selected_values = values[selected_keys]
        approximate_selected_mass = left @ selected_right.sum(dim=1)
        selected_numerator = left @ (selected_right @ selected_values)
        all_numerator = left @ right_values
        tail_numerator = all_numerator - selected_numerator
        tail_output = tail_numerator / (
            1 - approximate_selected_mass
        ).clamp_min(1e-6)[:, None]
    else:
        tail_output = left @ right_values

    selected_mass = selected_mass.clamp(0, 1)
    return selected_mass[:, None] * selected_output + (
        1 - selected_mass
    )[:, None] * tail_output


def validate_capture_payload(
    payload: dict[str, object], path: Path, row: dict[str, object]
) -> None:
    tensors = []
    for name in ("q", "k", "v"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path} does not contain tensor {name!r}")
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError(f"{path}:{name} expected [1,N,H,D], got {tuple(value.shape)}")
        tensors.append(value)
    if not (tensors[0].shape == tensors[1].shape == tensors[2].shape):
        raise ValueError(f"{path} has mismatched Q/K/V shapes")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{path} does not contain a metadata dictionary")
    expected = {
        "sample_id": str(row["sample_id"]),
        "prompt_index": int(row["prompt_index"]),
        "seed": int(row["seed"]),
        "sampling_step": int(row["sampling_step"]),
        "branch": str(row["branch"]),
        "layer": int(row["layer"]),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"capture/index metadata mismatch for {path}:{field}: "
                f"payload={metadata.get(field)!r}, index={value!r}"
            )
    if metadata.get("attention_kind") != "self":
        raise ValueError(f"expected self attention capture: {path}")
    if not math.isclose(
        float(metadata.get("timestep")),
        float(row["timestep"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"capture/index timestep mismatch: {path}")
    for name, tensor in zip(("q", "k", "v"), tensors):
        if metadata.get(f"{name}_shape") != list(tensor.shape):
            raise ValueError(f"metadata {name}_shape mismatch: {path}")
    tokens = tensors[0].shape[1]
    if int(metadata.get("token_count", -1)) != tokens:
        raise ValueError(f"metadata token_count mismatch: {path}")
    grid_size = metadata.get("grid_size")
    if not isinstance(grid_size, list) or math.prod(map(int, grid_size)) != tokens:
        raise ValueError(f"metadata grid_size mismatch: {path}")
    if metadata.get("dtype") != str(tensors[0].dtype):
        raise ValueError(f"metadata dtype mismatch: {path}")
    scale = float(payload.get("softmax_scale", float("nan")))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid softmax scale in {path}: {scale}")


def capture_provenance(
    rows: list[dict[str, object]], hash_mode: str
) -> list[dict[str, object]]:
    provenance = []
    for row in rows:
        path = Path(row["path"]).resolve()
        stat = path.stat()
        indexed_bytes = int(row.get("bytes", stat.st_size))
        if indexed_bytes != stat.st_size:
            raise ValueError(
                f"capture size differs from index: {path}, "
                f"index={indexed_bytes}, actual={stat.st_size}"
            )
        record: dict[str, object] = {
            "sample_id": row["sample_id"],
            "sampling_step": row["sampling_step"],
            "branch": row["branch"],
            "layer": row["layer"],
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "hash_mode": hash_mode,
        }
        if hash_mode == "sha256":
            record["sha256"] = file_sha256(path)
        provenance.append(record)
    return provenance


def assert_capture_metadata_unchanged(
    provenance: list[dict[str, object]],
) -> None:
    for record in provenance:
        path = Path(str(record["path"]))
        stat = path.stat()
        if stat.st_size != int(record["bytes"]) or stat.st_mtime_ns != int(
            record["mtime_ns"]
        ):
            raise ValueError(f"capture changed during probe: {path}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def arithmetic_work_ratio(
    method: str,
    landmark_count: int,
    tokens: int,
    selected_key_fraction: float,
    block_size: int,
    query_tile_size: int,
    head_dimension: int,
) -> tuple[float, dict[str, float | bool | str]]:
    """Return a full-attention arithmetic upper-bound proxy, not wall time."""

    nystrom = 2.0 * landmark_count / tokens
    router = 0.0
    critical = 0.0
    partition = 0.0
    materialization = 0.0
    if "mixture" in method or "partition" in method:
        router = 1.0 / (2.0 * block_size)
        critical = selected_key_fraction
    if "landmark_partition" in method:
        partition = (
            landmark_count * selected_key_fraction / (2.0 * query_tile_size)
            + landmark_count / (2.0 * tokens)
        )
    deployable = method in DEPLOYABLE_METHODS
    if method == "nystrom_nonnegative_clamped":
        # Full P materialization plus P@V cannot use the low-rank association.
        materialization = 0.5 + landmark_count / (2.0 * head_dimension)
    total = nystrom + router + critical + partition + materialization
    return total, {
        "nystrom_work_ratio": nystrom,
        "router_work_ratio": router,
        "critical_work_ratio": critical,
        "partition_work_ratio": partition,
        "materialization_work_ratio": materialization,
        "work_proxy_scope": "projected_full_attention_arithmetic_upper_bound",
        "work_proxy_valid_for_deployment": deployable,
    }


@torch.inference_mode()
def process_capture(
    row: dict[str, object],
    args: argparse.Namespace,
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
    starts = aligned_query_tile_starts(tokens, args.query_tile_size, args.query_tiles)
    results: list[dict[str, object]] = []

    head_indices = tuple(args.heads) if args.heads else tuple(range(heads))
    invalid_heads = [head for head in head_indices if not 0 <= head < heads]
    if invalid_heads:
        raise ValueError(
            f"requested heads outside [0, {heads - 1}] for {path}: {invalid_heads}"
        )
    if len(head_indices) != len(set(head_indices)):
        raise ValueError(f"duplicate requested heads: {head_indices}")

    for head in head_indices:
        all_queries = q_all[:, head].to(device=device, dtype=torch.float32)
        keys = k_all[:, head].to(device=device, dtype=torch.float32)
        values = v_all[:, head].to(device=device, dtype=torch.float32)
        if not (
            bool(torch.isfinite(all_queries).all())
            and bool(torch.isfinite(keys).all())
            and bool(torch.isfinite(values).all())
        ):
            raise ValueError(f"non-finite Q/K/V in {path}, head {head}")

        router_moments = padded_groups(keys, values, args.block_size)
        factor_cache = {}
        for landmark_mode in args.landmark_modes:
            for landmark_count in args.landmarks:
                for pinv_rtol in args.pinv_rtols:
                    key_landmarks, inverse, right, diagnostics = nystrom_factors(
                        all_queries,
                        keys,
                        landmark_count,
                        landmark_mode,
                        scale,
                        pinv_rtol,
                    )
                    right_values = right @ values
                    factor_cache[(landmark_mode, landmark_count, pinv_rtol)] = (
                        key_landmarks,
                        inverse,
                        right,
                        right_values,
                        diagnostics,
                    )

        accumulators: dict[tuple[object, ...], dict[str, float]] = defaultdict(
            lambda: {
                "residual_sq": 0.0,
                "reference_sq": 0.0,
                "tiles": 0.0,
                "selected_attention_mass": 0.0,
                "selected_key_fraction": 0.0,
                "proxy_selected_mass": 0.0,
                "mass_absolute_error": 0.0,
                "negative_entry_fraction": 0.0,
                "negative_absolute_mass_ratio": 0.0,
                "signed_row_sum_error": 0.0,
            }
        )

        for start in starts:
            queries = all_queries[start : start + args.query_tile_size]
            exact_scores = queries @ keys.T * scale
            dense_probabilities = torch.softmax(exact_scores, dim=1)
            reference = dense_probabilities @ values
            reference_sq = float(reference.square().sum())

            router_logits = moment_logits(queries, router_moments, scale, "centroid")
            router_probabilities = torch.softmax(router_logits, dim=1)
            density_cache = {}
            for density in args.densities:
                selected_blocks = choose_blocks(router_logits, density)
                selected_keys = key_mask_from_blocks(
                    selected_blocks, tokens, args.block_size
                )
                proxy_mass = router_probabilities[:, selected_blocks].sum(dim=1)
                actual_mass = dense_probabilities[:, selected_keys].sum(dim=1)
                density_cache[density] = (
                    selected_keys,
                    float(selected_keys.float().mean()),
                    proxy_mass,
                    actual_mass,
                )

            for (landmark_mode, landmark_count, pinv_rtol), (
                key_landmarks,
                inverse,
                right,
                right_values,
                factor_diagnostics,
            ) in factor_cache.items():
                left, signed, nonnegative, probability_diagnostics = nystrom_components(
                    queries, key_landmarks, inverse, right, scale
                )
                nystrom_output = left @ (inverse @ right_values)
                landmark_output = left @ right_values
                pure_outputs = {
                    "nystrom_signed": nystrom_output,
                    "landmark_linear": landmark_output,
                    "nystrom_nonnegative_clamped": nonnegative @ values,
                }
                for method, approximation in pure_outputs.items():
                    residual_sq = float((reference - approximation).square().sum())
                    key = (method, landmark_mode, landmark_count, pinv_rtol, 0.0)
                    record = accumulators[key]
                    record["residual_sq"] += residual_sq
                    record["reference_sq"] += reference_sq
                    record["tiles"] += 1
                    for field, value in probability_diagnostics.items():
                        record[field] += value

                for density, (
                    selected_keys,
                    selected_key_fraction,
                    proxy_mass,
                    actual_mass,
                ) in density_cache.items():
                    outputs = {
                        "proxy_mass_nystrom_mixture": mass_mixture_output(
                            exact_scores,
                            nystrom_output,
                            values,
                            selected_keys,
                            proxy_mass,
                        ),
                        "dense_mass_nystrom_mixture_diagnostic": mass_mixture_output(
                            exact_scores,
                            nystrom_output,
                            values,
                            selected_keys,
                            actual_mass,
                        ),
                        "proxy_mass_landmark_partition": landmark_partition_output(
                            exact_scores,
                            left,
                            right,
                            right_values,
                            values,
                            selected_keys,
                            proxy_mass,
                        ),
                        "dense_mass_landmark_partition_diagnostic": landmark_partition_output(
                            exact_scores,
                            left,
                            right,
                            right_values,
                            values,
                            selected_keys,
                            actual_mass,
                        ),
                    }
                    for method, approximation in outputs.items():
                        residual_sq = float((reference - approximation).square().sum())
                        key = (method, landmark_mode, landmark_count, pinv_rtol, density)
                        record = accumulators[key]
                        record["residual_sq"] += residual_sq
                        record["reference_sq"] += reference_sq
                        record["tiles"] += 1
                        record["selected_key_fraction"] += selected_key_fraction
                        record["selected_attention_mass"] += float(actual_mass.mean())
                        record["proxy_selected_mass"] += float(proxy_mass.mean())
                        record["mass_absolute_error"] += float(
                            (actual_mass - proxy_mass).abs().mean()
                        )
                        for field, value in probability_diagnostics.items():
                            record[field] += value

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
        for (
            method,
            landmark_mode,
            landmark_count,
            pinv_rtol,
            density,
        ), record in accumulators.items():
            tiles = record["tiles"]
            selected_key_fraction = record["selected_key_fraction"] / tiles
            work_ratio, work_parts = arithmetic_work_ratio(
                str(method),
                int(landmark_count),
                tokens,
                selected_key_fraction,
                args.block_size,
                args.query_tile_size,
                dimension,
            )
            factor_diagnostics = factor_cache[
                (landmark_mode, landmark_count, pinv_rtol)
            ][4]
            results.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_index": row["prompt_index"],
                    "seed": row["seed"],
                    "sampling_step": row["sampling_step"],
                    "timestep": row["timestep"],
                    "branch": row["branch"],
                    "layer": row["layer"],
                    "head": head,
                    "head_role_diagnostic_only": role,
                    "method": method,
                    "deployable_candidate": method in DEPLOYABLE_METHODS,
                    "landmark_mode": landmark_mode,
                    "landmarks": landmark_count,
                    "pinv_rtol": pinv_rtol,
                    "density": density,
                    "selected_key_fraction": selected_key_fraction,
                    "tokens": tokens,
                    "query_tiles": len(starts),
                    "query_tile_size": args.query_tile_size,
                    "residual_sq": record["residual_sq"],
                    "reference_sq": record["reference_sq"],
                    "output_relative_l2": math.sqrt(
                        record["residual_sq"] / max(record["reference_sq"], 1e-30)
                    ),
                    "selected_attention_mass": record["selected_attention_mass"] / tiles,
                    "proxy_selected_mass": record["proxy_selected_mass"] / tiles,
                    "mass_absolute_error": record["mass_absolute_error"] / tiles,
                    "projected_attention_work_ratio": work_ratio,
                    "arithmetic_speedup_upper_bound": 1.0 / work_ratio,
                    **work_parts,
                    **factor_diagnostics,
                    "negative_entry_fraction": record["negative_entry_fraction"] / tiles,
                    "negative_absolute_mass_ratio": record[
                        "negative_absolute_mass_ratio"
                    ]
                    / tiles,
                    "signed_row_sum_error": record["signed_row_sum_error"] / tiles,
                }
            )
        del all_queries, keys, values, router_moments, factor_cache
    return results


def aggregate_rows(
    rows: list[dict[str, object]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        errors = [float(row["output_relative_l2"]) for row in group]
        output.append(
            {
                **dict(zip(group_fields, key)),
                "records": len(group),
                "aggregate_output_relative_l2": math.sqrt(
                    residual_sq / max(reference_sq, 1e-30)
                ),
                "record_error_p95": quantile(errors, 0.95),
                "record_error_max": max(errors),
                "projected_attention_work_ratio_mean": sum(
                    float(row["projected_attention_work_ratio"]) for row in group
                )
                / len(group),
                "arithmetic_speedup_upper_bound": 1.0
                / (
                    sum(
                        float(row["projected_attention_work_ratio"])
                        for row in group
                    )
                    / len(group)
                ),
                "selected_attention_mass_mean": sum(
                    float(row["selected_attention_mass"]) for row in group
                )
                / len(group),
                "mass_absolute_error_mean": sum(
                    float(row["mass_absolute_error"]) for row in group
                )
                / len(group),
                "negative_absolute_mass_ratio_mean": sum(
                    float(row["negative_absolute_mass_ratio"]) for row in group
                )
                / len(group),
            }
        )
    return output


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
    if args.query_tile_size <= 0 or args.query_tiles <= 0 or args.block_size <= 0:
        raise ValueError("query and block sizes must be positive")
    if any(count <= 0 for count in args.landmarks):
        raise ValueError("landmark counts must be positive")
    if not set(args.landmark_modes) <= {"segment", "uniform"}:
        raise ValueError("unsupported landmark mode")
    if any(rtol <= 0 for rtol in args.pinv_rtols):
        raise ValueError("pinv rtol must be positive")
    if any(not 0 < density < 1 for density in args.densities):
        raise ValueError("hybrid densities must lie strictly between zero and one")

    args.capture_index = args.capture_index.resolve()
    args.split_config = args.split_config.resolve()
    args.output_dir = args.output_dir.resolve()
    require_fresh_output_dir(args.output_dir)
    protocols = load_split_protocols(args.split_config)
    rows = read_capture_rows(args)
    observed_samples = {str(row["sample_id"]) for row in rows}
    for protocol in protocols:
        protocol.assert_exact_coverage(observed_samples)
    capture_keys = [
        (
            row["sample_id"],
            row["sampling_step"],
            row["branch"],
            row["layer"],
        )
        for row in rows
    ]
    if len(capture_keys) != len(set(capture_keys)):
        raise ValueError("capture selection contains duplicate sample/step/branch/layer keys")

    config = serializable_args(args)
    config_hash = object_sha256(config)
    run_id = f"nystrom-{config_hash[:12]}-{int(time.time())}"
    event_log = JsonlEventLog(args.output_dir / "progress.jsonl", run_id)
    atomic_write_json(
        args.output_dir / "run_state.json",
        {"status": "RUNNING", "run_id": run_id, "config_sha256": config_hash},
    )
    event_log.emit(
        "input_hashing_started",
        captures=len(rows),
        capture_hash_mode=args.capture_hash_mode,
    )
    provenance = capture_provenance(rows, args.capture_hash_mode)
    input_manifest = {
        "schema_version": 1,
        "capture_index": str(args.capture_index),
        "capture_index_sha256": file_sha256(args.capture_index),
        "capture_hash_mode": args.capture_hash_mode,
        "captures": provenance,
    }
    atomic_write_json(args.output_dir / "input_manifest.json", input_manifest)
    event_log.emit("input_hashing_completed", captures=len(rows))
    event_log.emit("run_started", captures=len(rows), config_sha256=config_hash)

    roles = read_head_roles(args.head_stats_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    detail: list[dict[str, object]] = []
    for capture_index, row in enumerate(rows):
        capture_started = time.time()
        capture_rows = process_capture(row, args, roles, device)
        detail.extend(capture_rows)
        event_log.emit(
            "capture_completed",
            capture_index=capture_index,
            sample_id=row["sample_id"],
            sampling_step=row["sampling_step"],
            branch=row["branch"],
            layer=row["layer"],
            result_rows=len(capture_rows),
            elapsed_seconds=time.time() - capture_started,
        )
        print(
            f"[nystrom] {capture_index + 1}/{len(rows)} sample={row['sample_id']} "
            f"step={row['sampling_step']} branch={row['branch']} layer={row['layer']}",
            flush=True,
        )
    assert_capture_metadata_unchanged(provenance)

    summary = aggregate_rows(
        detail,
        (
            "method",
            "deployable_candidate",
            "landmark_mode",
            "landmarks",
            "pinv_rtol",
            "density",
        ),
    )
    cells = aggregate_rows(
        detail,
        (
            "sample_id",
            "sampling_step",
            "branch",
            "layer",
            "method",
            "landmark_mode",
            "landmarks",
            "pinv_rtol",
            "density",
        ),
    )
    atomic_write_csv(args.output_dir / "nystrom_sparse_tail_heads.csv", detail)
    atomic_write_csv(args.output_dir / "nystrom_sparse_tail_summary.csv", summary)
    atomic_write_csv(args.output_dir / "nystrom_sparse_tail_cells.csv", cells)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "current-Q/K/V Nystrom tail and sparse-critical hybrid numerical probe",
        "arguments": config,
        "config_sha256": config_hash,
        "capture_index_sha256": file_sha256(args.capture_index),
        "input_manifest_sha256": file_sha256(
            args.output_dir / "input_manifest.json"
        ),
        "split_config_sha256": file_sha256(args.split_config),
        "split_protocols": [protocol.as_dict() for protocol in protocols],
        "captures": len(rows),
        "sample_ids": sorted(observed_samples),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "deployable_methods": sorted(DEPLOYABLE_METHODS),
        "diagnostic_only_methods": sorted(DIAGNOSTIC_METHODS),
        "dense_mass_warning": (
            "dense-reference selected mass diagnoses router error but is not a "
            "quality oracle because the approximate tail is not the exact marginal"
        ),
        "fairness": "all methods share captures, query tiles, references, and method-specific arithmetic accounting",
        "speed_claim_boundary": (
            "work ratios are projected full-attention arithmetic upper bounds, "
            "not measured H200 latency"
        ),
        "execution_resource_note": args.execution_resource_note,
        "known_cost_omissions": [
            "landmark pooling",
            "small m-by-m pseudoinverse",
            "mask construction",
            "gather and launch overhead",
            "materialization strategy and kernel fusion",
        ],
        "head_role_warning": "head roles use dense statistics for analysis only and never route outputs",
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    success = {
        "status": "SUCCESS",
        "run_id": run_id,
        "config_sha256": config_hash,
        "detail_rows": len(detail),
        "summary_rows": len(summary),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output_dir / "run_state.json", success)
    atomic_write_json(args.output_dir / "SUCCESS.json", success)
    event_log.emit("run_completed", **success)
    print(f"[nystrom] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
