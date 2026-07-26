#!/usr/bin/env python3
"""Fit increasingly local BCM attention models on captured Wan QKV replays.

The probe deliberately separates periodic modulo deltas from physical THW
deltas. Tables are fitted on calibration replays only and then frozen for
held-out attention-output evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from probe_geometry_sparse_attention import (
    grid_from_metadata,
    stratified_query_indices,
    token_coordinates,
)


Triplet = tuple[int, int, int]
CellKey = tuple[int, int, str]


@dataclass(frozen=True)
class ModelSpec:
    method: str
    coarse_scale: Triplet
    query_grid: Triplet = (1, 1, 1)
    fine_scale: Triplet = (1, 1, 1)
    local_radius: Triplet = (0, 0, 0)

    @property
    def name(self) -> str:
        coarse = "x".join(map(str, self.coarse_scale))
        if self.method == "global_coarse_bccb":
            return f"global_bccb_s{coarse}"
        grid = "x".join(map(str, self.query_grid))
        if self.method == "query_block_multi_bcm":
            return f"multi_bcm_s{coarse}_g{grid}"
        fine = "x".join(map(str, self.fine_scale))
        radius = "x".join(map(str, self.local_radius))
        return f"hier_bcm_s{coarse}_g{grid}_f{fine}_r{radius}"


@dataclass
class FittedBCM:
    spec: ModelSpec
    shape: Triplet
    query_indices: torch.Tensor
    global_table: torch.Tensor
    tile_table: torch.Tensor | None = None
    fine_table: torch.Tensor | None = None

    def predict(self, device: torch.device | str | None = None) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.global_table.device
        queries = self.query_indices.to(target)
        periodic = self.spec.method == "global_coarse_bccb"
        coarse_index, _ = delta_bucket_indices(
            queries, self.shape, self.spec.coarse_scale, periodic=periodic
        )
        values = self.global_table.to(target).take(coarse_index)
        if self.spec.method == "query_block_multi_bcm":
            if self.tile_table is None:
                raise RuntimeError("query-block model is missing its conditioned table")
            groups = query_group_indices(queries, self.shape, self.spec.query_grid)
            coarse_count = bucket_count(self.shape, self.spec.coarse_scale, False)
            combined = groups[:, None] * coarse_count + coarse_index
            return normalize_nonnegative(self.tile_table.to(target).take(combined))
        if self.tile_table is not None:
            groups = query_group_indices(queries, self.shape, self.spec.query_grid)
            coarse_count = bucket_count(self.shape, self.spec.coarse_scale, False)
            combined = groups[:, None] * coarse_count + coarse_index
            values = normalize_nonnegative(values)
            values = values + self.tile_table.to(target).take(combined)
        if self.fine_table is not None:
            values = normalize_nonnegative(values)
            groups = query_group_indices(queries, self.shape, self.spec.query_grid)
            fine_index, local = local_bucket_indices(
                queries,
                self.shape,
                self.spec.fine_scale,
                self.spec.local_radius,
            )
            fine_count = local_bucket_count(self.spec.fine_scale, self.spec.local_radius)
            combined = groups[:, None] * fine_count + fine_index.clamp_min(0)
            fine = self.fine_table.to(target).take(combined)
            values = values + torch.where(local, fine, torch.zeros_like(fine))
        return normalize_nonnegative(values)

    @property
    def parameters_per_head(self) -> int:
        return parameter_count(self.spec, self.shape, heads=1)


def parse_triplet(text: str) -> Triplet:
    normalized = text.lower().replace(",", "x")
    try:
        values = tuple(int(item) for item in normalized.split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive TxHxW triplet") from error
    if len(values) != 3 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a positive TxHxW triplet")
    return values  # type: ignore[return-value]


def parse_triplet_list(text: str) -> tuple[Triplet, ...]:
    values = tuple(parse_triplet(item.strip()) for item in text.split(";") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected semicolon-separated TxHxW triplets")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("head indices must be non-negative")
    return values


def flatten_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(item.strip() for value in values for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-sample-id", action="append", required=True)
    parser.add_argument("--heldout-sample-id", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--heads", type=parse_int_list, default=())
    parser.add_argument(
        "--bucket-scales",
        type=parse_triplet_list,
        default=parse_triplet_list("1x4x4;1x8x8;2x8x8"),
    )
    parser.add_argument(
        "--query-group-grids",
        type=parse_triplet_list,
        default=parse_triplet_list("1x2x2;2x4x4"),
    )
    parser.add_argument("--fine-scale", type=parse_triplet, default=parse_triplet("1x2x2"))
    parser.add_argument("--local-radius", type=parse_triplet, default=parse_triplet("1x4x4"))
    parser.add_argument("--strict-error-target", type=float, default=0.02)
    parser.add_argument("--relaxed-error-target", type=float, default=0.05)
    return parser.parse_args()


def validate_split_ids(
    calibration: Iterable[str], heldout: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    splits = {"calibration": tuple(calibration), "heldout": tuple(heldout)}
    for name, values in splits.items():
        if not values:
            raise ValueError(f"{name} split cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} split contains duplicate sample IDs")
    overlap = sorted(set(splits["calibration"]) & set(splits["heldout"]))
    if overlap:
        raise ValueError(f"cross-replay split leakage: {overlap[:3]}")
    return splits


def read_capture_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "layer", "sampling_step", "branch", "path"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"capture index is empty or missing columns: {sorted(missing)}")
    return rows


def index_capture_rows(
    rows: list[dict[str, str]],
    splits: dict[str, tuple[str, ...]],
    index_path: Path,
) -> tuple[dict[str, dict[CellKey, Path]], tuple[CellKey, ...]]:
    owner = {sample: split for split, samples in splits.items() for sample in samples}
    indexed: dict[str, dict[CellKey, Path]] = {sample: {} for sample in owner}
    path_owners: dict[Path, tuple[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in owner:
            continue
        replay_path = Path(row["path"])
        if not replay_path.is_absolute():
            replay_path = index_path.parent / replay_path
        replay_path = replay_path.resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"missing replay: {replay_path}")
        previous = path_owners.get(replay_path)
        if previous is not None and previous[1] != owner[sample_id]:
            raise ValueError(
                "cross-replay split leakage through a shared replay path: "
                f"{previous[0]} and {sample_id}"
            )
        path_owners[replay_path] = (sample_id, owner[sample_id])
        cell = (int(row["layer"]), int(row["sampling_step"]), row["branch"])
        if cell in indexed[sample_id]:
            raise ValueError(f"duplicate replay cell for {sample_id}: {cell}")
        indexed[sample_id][cell] = replay_path
    missing_samples = [sample for sample, cells in indexed.items() if not cells]
    if missing_samples:
        raise ValueError(f"capture index has no replay for: {missing_samples}")
    reference = set(next(iter(indexed.values())))
    for sample_id, cells in indexed.items():
        if set(cells) != reference:
            raise ValueError(f"replay cell coverage mismatch for {sample_id}")
    return indexed, tuple(sorted(reference))


def coordinate_matrix(indices: torch.Tensor, shape: Triplet) -> torch.Tensor:
    return torch.stack(token_coordinates(indices, shape), dim=-1)


def bucket_dimensions(shape: Triplet, scale: Triplet, periodic: bool) -> Triplet:
    spans = shape if periodic else tuple(2 * value - 1 for value in shape)
    return tuple(math.ceil(span / step) for span, step in zip(spans, scale))  # type: ignore[return-value]


def bucket_count(shape: Triplet, scale: Triplet, periodic: bool) -> int:
    return math.prod(bucket_dimensions(shape, scale, periodic))


def delta_bucket_indices(
    query_indices: torch.Tensor,
    shape: Triplet,
    scale: Triplet,
    periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened delta buckets and long-distance modulo alias mask.

    Physical deltas retain signed boundary information. Periodic deltas use
    (key-query) mod axis size; wrap_alias marks physically long pairs that the
    torus maps onto a short displacement represented by the same generator.
    """
    device = query_indices.device
    keys = torch.arange(math.prod(shape), device=device)
    query_coords = coordinate_matrix(query_indices, shape)
    key_coords = coordinate_matrix(keys, shape)
    physical = key_coords[None, :, :] - query_coords[:, None, :]
    shape_tensor = torch.tensor(shape, device=device)
    scale_tensor = torch.tensor(scale, device=device)
    if periodic:
        values = torch.remainder(physical, shape_tensor)
        wrap_alias = (physical.abs() > (shape_tensor // 2)).any(dim=-1)
    else:
        values = physical + (shape_tensor - 1)
        wrap_alias = torch.zeros(physical.shape[:2], dtype=torch.bool, device=device)
    buckets = torch.div(values, scale_tensor, rounding_mode="floor")
    dimensions = bucket_dimensions(shape, scale, periodic)
    flattened = (buckets[..., 0] * dimensions[1] + buckets[..., 1]) * dimensions[2]
    flattened = flattened + buckets[..., 2]
    return flattened.long(), wrap_alias


def query_group_indices(
    query_indices: torch.Tensor, shape: Triplet, query_grid: Triplet
) -> torch.Tensor:
    coords = coordinate_matrix(query_indices, shape)
    shape_tensor = torch.tensor(shape, device=query_indices.device)
    grid_tensor = torch.tensor(query_grid, device=query_indices.device)
    groups = torch.div(coords * grid_tensor, shape_tensor, rounding_mode="floor")
    groups = torch.minimum(groups, grid_tensor - 1)
    return ((groups[:, 0] * query_grid[1] + groups[:, 1]) * query_grid[2] + groups[:, 2]).long()


def local_bucket_count(fine_scale: Triplet, radius: Triplet) -> int:
    return math.prod(math.ceil((2 * limit + 1) / step) for step, limit in zip(fine_scale, radius))


def local_bucket_indices(
    query_indices: torch.Tensor,
    shape: Triplet,
    fine_scale: Triplet,
    radius: Triplet,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = query_indices.device
    keys = torch.arange(math.prod(shape), device=device)
    physical = coordinate_matrix(keys, shape)[None] - coordinate_matrix(query_indices, shape)[:, None]
    radius_tensor = torch.tensor(radius, device=device)
    scale_tensor = torch.tensor(fine_scale, device=device)
    local = (physical.abs() <= radius_tensor).all(dim=-1)
    buckets = torch.div(physical + radius_tensor, scale_tensor, rounding_mode="floor")
    dimensions = tuple(
        math.ceil((2 * limit + 1) / step) for step, limit in zip(fine_scale, radius)
    )
    flattened = (buckets[..., 0] * dimensions[1] + buckets[..., 1]) * dimensions[2]
    flattened = flattened + buckets[..., 2]
    return torch.where(local, flattened, torch.full_like(flattened, -1)).long(), local


def normalize_nonnegative(values: torch.Tensor) -> torch.Tensor:
    nonnegative = values.clamp_min(0.0)
    row_sum = nonnegative.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(nonnegative, 1.0 / nonnegative.shape[-1])
    return torch.where(row_sum > 1e-30, nonnegative / row_sum.clamp_min(1e-30), uniform)


def mean_table(
    targets: Iterable[torch.Tensor],
    indices: torch.Tensor,
    table_size: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = torch.ones_like(indices, dtype=torch.bool) if mask is None else mask
    if selected.shape != indices.shape:
        raise ValueError("bucket mask must match index shape")
    flat_index = indices[selected]
    sums = torch.zeros(table_size, dtype=torch.float64, device=indices.device)
    counts = torch.zeros_like(sums)
    ones = torch.ones(flat_index.numel(), dtype=torch.float64, device=indices.device)
    for target in targets:
        values = target.to(device=indices.device, dtype=torch.float64)[selected]
        sums.scatter_add_(0, flat_index, values)
        counts.scatter_add_(0, flat_index, ones)
    return (sums / counts.clamp_min(1.0)).float()


def fit_attention_model(
    calibration_attention: list[torch.Tensor],
    query_indices: torch.Tensor,
    shape: Triplet,
    spec: ModelSpec,
) -> FittedBCM:
    if not calibration_attention:
        raise ValueError("calibration attention cannot be empty")
    expected = (query_indices.numel(), math.prod(shape))
    if any(tuple(attention.shape) != expected for attention in calibration_attention):
        raise ValueError(f"calibration attention must have shape {expected}")
    device = calibration_attention[0].device
    queries = query_indices.to(device)
    periodic = spec.method == "global_coarse_bccb"
    coarse_index, _ = delta_bucket_indices(queries, shape, spec.coarse_scale, periodic)
    global_table = mean_table(
        calibration_attention,
        coarse_index,
        bucket_count(shape, spec.coarse_scale, periodic),
    )
    if spec.method == "global_coarse_bccb":
        return FittedBCM(spec, shape, query_indices.cpu(), global_table.cpu())

    groups = query_group_indices(queries, shape, spec.query_grid)
    group_count = math.prod(spec.query_grid)
    coarse_count = bucket_count(shape, spec.coarse_scale, False)
    combined = groups[:, None] * coarse_count + coarse_index
    if spec.method == "query_block_multi_bcm":
        table = mean_table(calibration_attention, combined, group_count * coarse_count)
        return FittedBCM(
            spec,
            shape,
            query_indices.cpu(),
            torch.zeros_like(global_table).cpu(),
            table.cpu(),
        )

    if spec.method != "coarse_tile_local_residual":
        raise ValueError(f"unsupported method: {spec.method}")
    base_model = FittedBCM(spec, shape, query_indices.cpu(), global_table.cpu())
    base = base_model.predict(device)
    tile_targets = [attention - base for attention in calibration_attention]
    tile_table = mean_table(tile_targets, combined, group_count * coarse_count)
    tiled_model = FittedBCM(
        spec, shape, query_indices.cpu(), global_table.cpu(), tile_table.cpu()
    )
    tiled = tiled_model.predict(device)
    fine_index, local = local_bucket_indices(
        queries, shape, spec.fine_scale, spec.local_radius
    )
    fine_count = local_bucket_count(spec.fine_scale, spec.local_radius)
    fine_combined = groups[:, None] * fine_count + fine_index.clamp_min(0)
    fine_targets = [torch.where(local, attention - tiled, torch.zeros_like(attention)) for attention in calibration_attention]
    fine_table = mean_table(
        fine_targets, fine_combined, group_count * fine_count, mask=local
    )
    return FittedBCM(
        spec,
        shape,
        query_indices.cpu(),
        global_table.cpu(),
        tile_table.cpu(),
        fine_table.cpu(),
    )


def parameter_count(spec: ModelSpec, shape: Triplet, heads: int) -> int:
    if heads <= 0:
        raise ValueError("heads must be positive")
    periodic = spec.method == "global_coarse_bccb"
    coarse = bucket_count(shape, spec.coarse_scale, periodic)
    if spec.method == "global_coarse_bccb":
        per_head = coarse
    elif spec.method == "query_block_multi_bcm":
        per_head = math.prod(spec.query_grid) * coarse
    elif spec.method == "coarse_tile_local_residual":
        groups = math.prod(spec.query_grid)
        per_head = coarse + groups * coarse
        per_head += groups * local_bucket_count(spec.fine_scale, spec.local_radius)
    else:
        raise ValueError(f"unsupported method: {spec.method}")
    return per_head * heads


def model_specs(
    scales: Iterable[Triplet],
    grids: Iterable[Triplet],
    fine_scale: Triplet,
    local_radius: Triplet,
) -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    for scale in scales:
        specs.append(ModelSpec("global_coarse_bccb", scale))
        for grid in grids:
            specs.append(ModelSpec("query_block_multi_bcm", scale, grid))
            specs.append(
                ModelSpec(
                    "coarse_tile_local_residual",
                    scale,
                    grid,
                    fine_scale,
                    local_radius,
                )
            )
    return tuple(specs)


def attention_from_payload(
    payload: dict[str, object], query_indices: torch.Tensor, head: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_all = payload["q"]
    k_all = payload["k"]
    v_all = payload["v"]
    assert isinstance(q_all, torch.Tensor) and isinstance(k_all, torch.Tensor)
    assert isinstance(v_all, torch.Tensor)
    q = q_all[0, :, head].to(device=device, dtype=torch.float32)
    k = k_all[0, :, head].to(device=device, dtype=torch.float32)
    v = v_all[0, :, head].to(device=device, dtype=torch.float32)
    scale = float(payload.get("softmax_scale", q.shape[-1] ** -0.5))
    attention = torch.softmax(q.index_select(0, query_indices) @ k.T * scale, dim=-1)
    return attention, attention @ v, v


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-30))


def load_payload(path: Path, expected_sample: str, cell: CellKey) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    actual = (
        int(metadata.get("layer", -1)),
        int(metadata.get("sampling_step", -1)),
        str(metadata.get("branch", "")),
    )
    if str(metadata.get("sample_id", "")) != expected_sample or actual != cell:
        raise ValueError(f"replay metadata mismatch: {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.strict_error_target <= 0 or args.relaxed_error_target <= args.strict_error_target:
        raise ValueError("error targets must satisfy 0 < strict < relaxed")
    splits = validate_split_ids(
        flatten_ids(args.calibration_sample_id), flatten_ids(args.heldout_sample_id)
    )
    rows = read_capture_index(args.capture_index)
    replay_index, cells = index_capture_rows(rows, splits, args.capture_index.resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(
        args.bucket_scales, args.query_group_grids, args.fine_scale, args.local_radius
    )
    result_rows: list[dict[str, object]] = []
    started = time.time()

    for cell in cells:
        first_sample = splits["calibration"][0]
        first_payload = load_payload(replay_index[first_sample][cell], first_sample, cell)
        first_metadata = dict(first_payload.get("metadata", {}))
        q_tensor = first_payload["q"]
        assert isinstance(q_tensor, torch.Tensor)
        tokens, available_heads, _ = q_tensor.shape[1:]
        shape = grid_from_metadata(first_metadata, tokens, 30, 52)
        heads = args.heads or tuple(range(available_heads))
        if any(head >= available_heads for head in heads):
            raise ValueError(f"requested head outside [0, {available_heads})")
        query_indices = stratified_query_indices(shape, args.query_samples, 1, 1, device)
        del first_payload

        for head in heads:
            calibration_attention: list[torch.Tensor] = []
            for sample_id in splits["calibration"]:
                payload = load_payload(replay_index[sample_id][cell], sample_id, cell)
                attention, _, _ = attention_from_payload(payload, query_indices, head, device)
                calibration_attention.append(attention)
                del payload

            fitted_models = [
                (
                    spec,
                    fit_attention_model(
                        calibration_attention, query_indices, shape, spec
                    ),
                )
                for spec in specs
            ]
            for sample_id in splits["heldout"]:
                payload = load_payload(replay_index[sample_id][cell], sample_id, cell)
                metadata = dict(payload.get("metadata", {}))
                exact_attention, exact_output, v = attention_from_payload(
                    payload, query_indices, head, device
                )
                for spec, fitted in fitted_models:
                    prediction = fitted.predict(device)
                    periodic = spec.method == "global_coarse_bccb"
                    _, wrap_mask = delta_bucket_indices(
                        query_indices, shape, spec.coarse_scale, periodic=True
                    )
                    estimate = prediction @ v
                    error = relative_l2(exact_output, estimate)
                    exact_alias_mass = float(
                        (exact_attention * wrap_mask).sum(dim=1).mean()
                    )
                    predicted_alias_mass = float(
                        (prediction * wrap_mask).sum(dim=1).mean()
                    )
                    result_rows.append(
                        {
                            "method": spec.method,
                            "model": spec.name,
                            "sample_id": sample_id,
                            "prompt_index": metadata.get("prompt_index", ""),
                            "seed": metadata.get("seed", ""),
                            "layer": cell[0],
                            "sampling_step": cell[1],
                            "branch": cell[2],
                            "head": head,
                            "tokens": tokens,
                            "query_samples": query_indices.numel(),
                            "grid_t": shape[0],
                            "grid_h": shape[1],
                            "grid_w": shape[2],
                            "coarse_scale": "x".join(map(str, spec.coarse_scale)),
                            "query_grid": "x".join(map(str, spec.query_grid)),
                            "fine_scale": "x".join(map(str, spec.fine_scale)),
                            "local_radius": "x".join(map(str, spec.local_radius)),
                            "periodic_delta": periodic,
                            "parameters_per_head": fitted.parameters_per_head,
                            "parameters_all_evaluated_heads": parameter_count(
                                spec, shape, len(heads)
                            ),
                            "attention_relative_l2": relative_l2(exact_attention, prediction),
                            "attention_output_relative_l2": error,
                            "strict_2pct_pass": error <= args.strict_error_target,
                            "relaxed_5pct_pass": error <= args.relaxed_error_target,
                            "modulo_alias_pair_fraction": float(wrap_mask.float().mean()),
                            "exact_modulo_alias_region_mass": exact_alias_mass,
                            "predicted_modulo_alias_region_mass": predicted_alias_mass,
                            "wrap_leakage_applicable": periodic,
                            "wrap_leakage_mass_delta": (
                                predicted_alias_mass - exact_alias_mass if periodic else ""
                            ),
                            "calibration_sample_ids": ",".join(splits["calibration"]),
                            "heldout_only": True,
                        }
                    )
                    del prediction, estimate
                del payload, exact_attention, exact_output, v
            del calibration_attention, fitted_models
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "multiblock_bcm_attention_heldout.csv", result_rows)
    summaries: list[dict[str, object]] = []
    for spec in specs:
        selected = [row for row in result_rows if row["model"] == spec.name]
        errors = [float(row["attention_output_relative_l2"]) for row in selected]
        summaries.append(
            {
                "method": spec.method,
                "model": spec.name,
                "rows": len(selected),
                "parameters_per_head": selected[0]["parameters_per_head"] if selected else 0,
                "mean_attention_output_relative_l2": sum(errors) / len(errors) if errors else None,
                "max_attention_output_relative_l2": max(errors) if errors else None,
                "strict_2pct_all_pass": bool(errors) and all(error <= args.strict_error_target for error in errors),
                "relaxed_5pct_all_pass": bool(errors) and all(error <= args.relaxed_error_target for error in errors),
                "mean_wrap_leakage_mass_delta": (
                    sum(float(row["wrap_leakage_mass_delta"]) for row in selected)
                    / len(selected)
                    if selected and spec.method == "global_coarse_bccb"
                    else None
                ),
            }
        )
    write_csv(args.output_dir / "multiblock_bcm_attention_summary.csv", summaries)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "splits": splits,
        "cells": [list(cell) for cell in cells],
        "models": [asdict(spec) | {"name": spec.name} for spec in specs],
        "summary": summaries,
        "methodology": {
            "fit_target": "calibration attention probabilities; held-out replays never update tables",
            "global_coarse_bccb": "shared per-head modulo THW displacement generator",
            "query_block_multi_bcm": "non-periodic physical-delta generator conditioned on a coarse query THW group",
            "coarse_tile_local_residual": "physical global coarse table plus query-group coarse residual plus local fine residual",
            "normalization": "clamp fitted additive weights to non-negative values and normalize each query row",
            "wrap_leakage": "mass on physically longer-than-half-axis pairs aliased by toroidal modulo coordinates",
            "warning": "local pre-O-projection replay probe; no kernel or end-to-end speed claim",
        },
        "strict_error_target": args.strict_error_target,
        "relaxed_error_target": args.relaxed_error_target,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "result_rows": len(result_rows),
    }
    (args.output_dir / "multiblock_bcm_attention_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
