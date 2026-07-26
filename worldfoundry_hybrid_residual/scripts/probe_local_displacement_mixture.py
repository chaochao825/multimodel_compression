#!/usr/bin/env python3
"""Probe a boundary-aware local displacement mixture on Wan attention.

Each attention row is aligned into a fixed non-periodic THW offset stencil.
Calibration rows fit a mean displacement kernel plus M linear experts.  The
held-out probe reports both oracle expert coefficients and a low-cost ridge
gate from the current query vector.  This is a stronger and more precise test
than a fixed global BCCB table, while retaining an explicit displacement basis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch

from probe_block_moment_marginal import (
    parse_ints,
    parse_strings,
    read_capture_rows,
    read_head_roles,
    split_name,
    write_csv,
)
from probe_geometry_sparse_attention import grid_from_metadata


def parse_triplet(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().replace("x", ",").split(","))
    if len(values) != 3 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected a nonnegative TxHxW triplet")
    return values  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--head-stats-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_ints, default=(0, 14, 29))
    parser.add_argument("--steps", type=parse_ints, default=(0, 9, 19))
    parser.add_argument("--branches", type=parse_strings, default=("cond", "uncond"))
    parser.add_argument("--calibration-samples", type=parse_strings, default=(
        "s00_p00_seed20260740", "s01_p01_seed20260740"
    ))
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--radius", type=parse_triplet, default=(2, 4, 4))
    parser.add_argument("--ranks", type=parse_ints, default=(0, 2, 4, 8, 16))
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--localized-mass-target", type=float, default=0.50)
    parser.add_argument("--full-output-target", type=float, default=0.02)
    parser.add_argument("--max-rank-for-go", type=int, default=8)
    return parser.parse_args()


def displacement_offsets(
    radius: tuple[int, int, int], device: torch.device | str = "cpu"
) -> torch.Tensor:
    temporal = torch.arange(-radius[0], radius[0] + 1, device=device)
    height = torch.arange(-radius[1], radius[1] + 1, device=device)
    width = torch.arange(-radius[2], radius[2] + 1, device=device)
    return torch.cartesian_prod(temporal, height, width)


def interior_query_indices(
    shape: tuple[int, int, int],
    radius: tuple[int, int, int],
    samples: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if samples <= 0:
        raise ValueError("query samples must be positive")
    axes = []
    for size, margin in zip(shape, radius):
        if size <= 2 * margin:
            raise ValueError(f"radius {radius} leaves no interior in shape {shape}")
        axes.append(torch.arange(margin, size - margin, device=device))
    coordinates = torch.cartesian_prod(*axes)
    if samples > coordinates.shape[0]:
        raise ValueError("query sample count exceeds interior token count")
    positions = torch.linspace(
        0, coordinates.shape[0] - 1, samples, dtype=torch.float64, device=device
    ).round().long()
    chosen = coordinates.index_select(0, positions)
    return (chosen[:, 0] * shape[1] + chosen[:, 1]) * shape[2] + chosen[:, 2]


def local_key_indices(
    query_indices: torch.Tensor,
    shape: tuple[int, int, int],
    radius: tuple[int, int, int],
) -> torch.Tensor:
    spatial = shape[1] * shape[2]
    query_t = torch.div(query_indices, spatial, rounding_mode="floor")
    remainder = torch.remainder(query_indices, spatial)
    query_h = torch.div(remainder, shape[2], rounding_mode="floor")
    query_w = torch.remainder(remainder, shape[2])
    coordinates = torch.stack([query_t, query_h, query_w], dim=1)
    local = coordinates[:, None] + displacement_offsets(radius, query_indices.device)[None]
    return (local[..., 0] * shape[1] + local[..., 1]) * shape[2] + local[..., 2]


def fit_basis(rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = rows.mean(dim=0)
    centered = rows - mean
    used = min(rank, centered.shape[0], centered.shape[1])
    if used == 0:
        return mean, centered.new_zeros((centered.shape[1], 0)), 0.0
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    basis = eigenvectors[:, order[:used]].contiguous()
    total = eigenvalues.clamp_min(0).sum()
    explained = float(eigenvalues[order[:used]].clamp_min(0).sum() / total) if total > 0 else 1.0
    return mean, basis, explained


class RidgeGate:
    def __init__(
        self,
        weight: torch.Tensor,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        self.weight = weight
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y_mean = y_mean

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.x_mean) / self.x_scale
        augmented = torch.cat([normalized, normalized.new_ones((normalized.shape[0], 1))], dim=1)
        return augmented @ self.weight + self.y_mean


def fit_ridge_gate(
    features: torch.Tensor, targets: torch.Tensor, regularization: float
) -> RidgeGate:
    x_mean = features.mean(dim=0)
    x_scale = features.std(dim=0, unbiased=False).clamp_min(1e-5)
    normalized = (features - x_mean) / x_scale
    augmented = torch.cat([normalized, normalized.new_ones((normalized.shape[0], 1))], dim=1)
    y_mean = targets.mean(dim=0)
    centered_targets = targets - y_mean
    gram = augmented.T @ augmented
    penalty = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(
        gram + regularization * penalty,
        augmented.T @ centered_targets,
    )
    return RidgeGate(weight, x_mean, x_scale, y_mean)


def project_rows(
    rows: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    if basis.shape[1] == 0:
        return mean.expand_as(rows)
    return mean + ((rows - mean) @ basis) @ basis.T


@torch.inference_mode()
def reduce_capture(
    row: dict[str, object],
    radius: tuple[int, int, int],
    query_samples: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(row["path"], map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all = payload["q"][0]
    k_all = payload["k"][0]
    v_all = payload["v"][0]
    tokens, heads, dimension = q_all.shape
    shape = grid_from_metadata(metadata, tokens, fallback_height=30, fallback_width=52)
    query_indices = interior_query_indices(shape, radius, query_samples, device)
    key_indices = local_key_indices(query_indices, shape, radius)
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    query_cpu = query_indices.cpu()
    key_cpu = key_indices.cpu()
    queries = []
    probabilities = []
    dense_outputs = []
    local_outputs = []
    local_masses = []
    for head in range(heads):
        q = q_all[:, head].index_select(0, query_cpu).to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        dense_probabilities = torch.softmax(q @ k.T * scale, dim=1)
        local_probabilities = dense_probabilities.gather(1, key_indices)
        local_values = v.index_select(0, key_indices.reshape(-1)).reshape(
            query_samples, key_indices.shape[1], dimension
        )
        dense_output = dense_probabilities @ v
        local_output = torch.einsum("qo,qod->qd", local_probabilities, local_values)
        queries.append(q.cpu())
        probabilities.append(local_probabilities.cpu())
        dense_outputs.append(dense_output.cpu())
        local_outputs.append(local_output.cpu())
        local_masses.append(local_probabilities.sum(dim=1).cpu())
    return {
        "row": row,
        "shape": shape,
        "query_indices": query_cpu,
        "local_key_indices": key_cpu,
        "q": torch.stack(queries, dim=1),
        "local_probabilities": torch.stack(probabilities, dim=1),
        "dense_output": torch.stack(dense_outputs, dim=1),
        "local_output": torch.stack(local_outputs, dim=1),
        "local_mass": torch.stack(local_masses, dim=1),
    }


@torch.inference_mode()
def evaluate_capture(
    reduced: dict[str, object],
    fitted: dict[tuple[int, int], dict[str, object]],
    roles: dict[tuple[str, int, str, int, int], str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    row = reduced["row"]
    assert isinstance(row, dict)
    q = reduced["q"]
    probabilities = reduced["local_probabilities"]
    dense = reduced["dense_output"]
    local = reduced["local_output"]
    local_mass = reduced["local_mass"]
    key_indices = reduced["local_key_indices"]
    assert all(
        isinstance(value, torch.Tensor)
        for value in (q, probabilities, dense, local, local_mass, key_indices)
    )
    payload = torch.load(row["path"], map_location="cpu", weights_only=False)
    v_all = payload["v"][0]
    output = []
    heads = probabilities.shape[1]
    for head in range(heads):
        local_values = v_all[:, head].index_select(0, key_indices.reshape(-1)).reshape(
            key_indices.shape[0], key_indices.shape[1], v_all.shape[-1]
        ).to(device=device, dtype=torch.float32)
        actual_probabilities = probabilities[:, head].to(device)
        dense_output = dense[:, head].to(device)
        actual_local_output = local[:, head].to(device)
        features = q[:, head].to(device)
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
        for rank in args.ranks:
            model = fitted[(head, rank)]
            mean = model["mean"]
            basis = model["basis"]
            gate = model["gate"]
            assert isinstance(mean, torch.Tensor) and isinstance(basis, torch.Tensor)
            assert isinstance(gate, RidgeGate)
            oracle_linear = project_rows(actual_probabilities, mean, basis)
            ridge_coefficients = gate.predict(features)
            ridge_linear = mean + ridge_coefficients @ basis.T
            predictions = {
                "oracle_linear": oracle_linear,
                "oracle_nonnegative": oracle_linear.clamp_min(0),
                "ridge_linear": ridge_linear,
                "ridge_nonnegative": ridge_linear.clamp_min(0),
            }
            for prediction, predicted_probabilities in predictions.items():
                predicted_local = torch.einsum(
                    "qo,qod->qd", predicted_probabilities, local_values
                )
                residual_sq = float((predicted_local - actual_local_output).square().sum())
                dense_sq = float(dense_output.square().sum())
                local_sq = float(actual_local_output.square().sum())
                probability_residual_sq = float(
                    (predicted_probabilities - actual_probabilities).square().sum()
                )
                probability_reference_sq = float(actual_probabilities.square().sum())
                output.append(
                    {
                        "split": split_name(str(row["sample_id"])),
                        "sample_id": row["sample_id"],
                        "prompt_index": row["prompt_index"],
                        "seed": row["seed"],
                        "sampling_step": row["sampling_step"],
                        "branch": row["branch"],
                        "layer": row["layer"],
                        "head": head,
                        "head_role": role,
                        "radius": "x".join(map(str, args.radius)),
                        "offsets": actual_probabilities.shape[1],
                        "rank": rank,
                        "prediction": prediction,
                        "basis_energy": model["basis_energy"],
                        "local_mass_mean": float(local_mass[:, head].mean()),
                        "residual_sq": residual_sq,
                        "dense_reference_sq": dense_sq,
                        "local_reference_sq": local_sq,
                        "full_output_relative_l2": math.sqrt(
                            residual_sq / max(dense_sq, 1e-30)
                        ),
                        "local_output_relative_l2": math.sqrt(
                            residual_sq / max(local_sq, 1e-30)
                        ),
                        "probability_relative_l2": math.sqrt(
                            probability_residual_sq
                            / max(probability_reference_sq, 1e-30)
                        ),
                        "basis_parameters_per_head_cell": (rank + 1)
                        * actual_probabilities.shape[1],
                        "gate_parameters_per_head_cell": (
                            (features.shape[1] + 1) * rank
                            + 2 * features.shape[1]
                            + rank
                        ),
                        "stored_parameters_per_head_cell": (
                            (rank + 1) * actual_probabilities.shape[1]
                            + (features.shape[1] + 1) * rank
                            + 2 * features.shape[1]
                            + rank
                        ),
                    }
                )
    return output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def aggregate(
    rows: list[dict[str, object]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        dense_sq = sum(float(row["dense_reference_sq"]) for row in group)
        local_sq = sum(float(row["local_reference_sq"]) for row in group)
        errors = [float(row["full_output_relative_l2"]) for row in group]
        output.append(
            {
                **dict(zip(group_fields, key)),
                "records": len(group),
                "local_mass_mean": sum(float(row["local_mass_mean"]) for row in group)
                / len(group),
                "basis_energy_mean": sum(float(row["basis_energy"]) for row in group)
                / len(group),
                "aggregate_full_output_relative_l2": math.sqrt(
                    residual_sq / max(dense_sq, 1e-30)
                ),
                "aggregate_local_output_relative_l2": math.sqrt(
                    residual_sq / max(local_sq, 1e-30)
                ),
                "record_error_p95": quantile(errors, 0.95),
                "record_error_max": max(errors),
            }
        )
    return output


def make_decision(
    role_summary: list[dict[str, object]], args: argparse.Namespace
) -> dict[str, object]:
    candidates = [
        row
        for row in role_summary
        if row["split"] == "test"
        and row["head_role"] == "localized"
        and row["prediction"] == "oracle_nonnegative"
        and int(row["rank"]) <= args.max_rank_for_go
        and float(row["local_mass_mean"]) >= args.localized_mass_target
    ]
    best = min(
        candidates,
        key=lambda row: (
            float(row["aggregate_full_output_relative_l2"]),
            float(row["record_error_max"]),
        ),
    ) if candidates else None
    oracle_go = bool(best) and float(best["record_error_max"]) <= args.full_output_target
    ridge_candidates = [
        row
        for row in role_summary
        if row["split"] == "test"
        and row["head_role"] == "localized"
        and row["prediction"] == "ridge_nonnegative"
        and int(row["rank"]) <= args.max_rank_for_go
        and float(row["local_mass_mean"]) >= args.localized_mass_target
    ]
    ridge_best = min(
        ridge_candidates,
        key=lambda row: (
            float(row["aggregate_full_output_relative_l2"]),
            float(row["record_error_max"]),
        ),
    ) if ridge_candidates else None
    deployable_go = bool(ridge_best) and float(ridge_best["record_error_max"]) <= args.full_output_target
    if deployable_go:
        verdict = "GO_LOCAL_DISPLACEMENT_EXPERT_CANDIDATE"
    elif oracle_go:
        verdict = "CONDITIONAL_GO_GATE_PREDICTION_REQUIRED"
    else:
        verdict = "NO_GO_LOCAL_DISPLACEMENT_MIXTURE"
    return {
        "verdict": verdict,
        "best_localized_oracle": best,
        "best_localized_ridge": ridge_best,
        "gates": {
            "localized_local_mass_mean": args.localized_mass_target,
            "every_record_full_output_relative_l2": args.full_output_target,
            "maximum_expert_rank": args.max_rank_for_go,
        },
        "rank_boundary": (
            "mean plus M signed displacement experts is linear before clipping; "
            "the nonnegative clamp invalidates a simple rank<=M+1 statement"
        ),
        "scope_warning": (
            "the exact outside-local contribution is retained when measuring branch error; "
            "this is a local branch replacement probe, not full sparse attention and not a "
            "kernel-speed claim"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.query_samples <= 0 or args.ridge_lambda < 0:
        raise ValueError("invalid query count or ridge regularization")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_rows = read_capture_rows(args)
    roles = read_head_roles(args.head_stats_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cells: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in capture_rows:
        cells[(int(row["layer"]), int(row["sampling_step"]), str(row["branch"]))].append(row)
    started = time.time()
    detail = []
    for cell, rows in sorted(cells.items()):
        calibration_rows = [
            row for row in rows if row["sample_id"] in args.calibration_samples
        ]
        if not calibration_rows:
            raise ValueError(f"cell {cell} has no calibration captures")
        calibration = [
            reduce_capture(row, args.radius, args.query_samples, device)
            for row in calibration_rows
        ]
        heads = calibration[0]["local_probabilities"].shape[1]
        fitted: dict[tuple[int, int], dict[str, object]] = {}
        for head in range(heads):
            probabilities = torch.cat(
                [record["local_probabilities"][:, head] for record in calibration]
            ).to(device)
            features = torch.cat([record["q"][:, head] for record in calibration]).to(device)
            for rank in args.ranks:
                mean, basis, energy = fit_basis(probabilities, rank)
                coefficients = (probabilities - mean) @ basis
                gate = fit_ridge_gate(features, coefficients, args.ridge_lambda)
                fitted[(head, rank)] = {
                    "mean": mean,
                    "basis": basis,
                    "basis_energy": energy,
                    "gate": gate,
                }
        for row in rows:
            reduced = reduce_capture(row, args.radius, args.query_samples, device)
            detail.extend(evaluate_capture(reduced, fitted, roles, args, device))
            print(
                f"[displacement] sample={row['sample_id']} step={row['sampling_step']} "
                f"branch={row['branch']} layer={row['layer']}",
                flush=True,
            )
        del calibration, fitted
    summary = aggregate(
        detail,
        ("split", "prediction", "rank"),
    )
    role_summary = aggregate(
        detail,
        ("split", "head_role", "prediction", "rank"),
    )
    cell_summary = aggregate(
        detail,
        ("split", "sampling_step", "branch", "layer", "prediction", "rank"),
    )
    decision = make_decision(role_summary, args)
    write_csv(args.output_dir / "local_displacement_mixture_heads.csv", detail)
    write_csv(args.output_dir / "local_displacement_mixture_summary.csv", summary)
    write_csv(args.output_dir / "local_displacement_mixture_roles.csv", role_summary)
    write_csv(args.output_dir / "local_displacement_mixture_cells.csv", cell_summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "scope": "boundary-aware local displacement mixture transfer probe",
        "captures": len(capture_rows),
        "cells": len(cells),
        "calibration_samples": list(args.calibration_samples),
        "arguments": {
            key: list(value) if isinstance(value, tuple) else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "methodology": (
            "fit frozen mean plus M non-periodic THW displacement experts on calibration; "
            "evaluate held-out oracle and query-ridge coefficients on AV"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[displacement] verdict={decision['verdict']}", flush=True)
    print(f"[displacement] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
