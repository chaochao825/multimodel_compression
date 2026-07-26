#!/usr/bin/env python3
"""Measure whether geometry-attention defect bases transfer across samples.

The frozen basis is learned only from calibration samples. Validation selects
the optional ridge feature/regularization pair, and test samples are evaluation
only. Projection with held-out oracle coefficients is reported separately from
the deployable ridge predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from probe_geometry_sparse_attention import (
    DEFAULT_SPECS,
    GeometrySpec,
    geometry_mask,
    grid_from_metadata,
    relative_l2,
    stratified_query_indices,
)


DEFAULT_MASKS = "s3_temporal_pm2,s3_tfull,s3_tfull_anchor12"
DEFAULT_RANKS = "8,16"
DEFAULT_RIDGE_FEATURES = "q,sparse,concat"
DEFAULT_RIDGE_LAMBDAS = "1e-5,1e-4,1e-3,1e-2,1e-1,1"
CellKey = tuple[int, int, str]


@dataclass
class RidgeModel:
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    weight: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.x_mean) / self.x_scale
        return (normalized @ self.weight) + self.y_mean


def parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_positive_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in parse_csv_strings(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return values


def parse_positive_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in parse_csv_strings(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("ridge lambdas must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-sample-id", action="append", required=True,
        help="Calibration IDs; repeat the option or use a comma-separated list.",
    )
    parser.add_argument(
        "--validation-sample-id", action="append", required=True,
        help="Validation IDs; repeat the option or use a comma-separated list.",
    )
    parser.add_argument(
        "--test-sample-id", action="append", required=True,
        help="Test IDs; repeat the option or use a comma-separated list.",
    )
    parser.add_argument("--masks", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_MASKS))
    parser.add_argument("--ranks", type=parse_positive_ints, default=parse_positive_ints(DEFAULT_RANKS))
    parser.add_argument("--query-samples", type=int, default=128)
    parser.add_argument("--tile-height", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=8)
    parser.add_argument("--expected-heads", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ridge-features",
        type=parse_csv_strings,
        default=parse_csv_strings(DEFAULT_RIDGE_FEATURES),
    )
    parser.add_argument(
        "--ridge-lambdas",
        type=parse_positive_floats,
        default=parse_positive_floats(DEFAULT_RIDGE_LAMBDAS),
    )
    parser.add_argument("--disable-ridge", action="store_true")
    parser.add_argument("--strict-error-target", type=float, default=0.02)
    parser.add_argument("--relaxed-error-target", type=float, default=0.05)
    parser.add_argument("--basis-energy-target", type=float, default=0.80)
    parser.add_argument("--subspace-overlap-target", type=float, default=0.50)
    return parser.parse_args()


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


def validate_split_ids(
    calibration: Iterable[str], validation: Iterable[str], test: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    splits = {
        "calibration": tuple(calibration),
        "validation": tuple(validation),
        "test": tuple(test),
    }
    for name, values in splits.items():
        if not values:
            raise ValueError(f"{name} split cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} split contains duplicate sample IDs")
    names = tuple(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(splits[left]) & set(splits[right]))
            if overlap:
                raise ValueError(
                    f"split leakage between {left} and {right}: {overlap[:3]}"
                )
    return splits


def flatten_sample_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sample_id
        for value in values
        for sample_id in parse_csv_strings(value)
    )


def read_capture_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "layer", "sampling_step", "branch", "path"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"capture index is empty or missing columns: {sorted(missing)}")
    return rows


def row_cell_key(row: dict[str, str]) -> CellKey:
    return int(row["layer"]), int(row["sampling_step"]), row["branch"]


def index_capture_rows(
    rows: list[dict[str, str]],
    split_ids: dict[str, tuple[str, ...]],
    index_path: Path,
) -> tuple[dict[str, dict[CellKey, Path]], tuple[CellKey, ...]]:
    requested = {sample for values in split_ids.values() for sample in values}
    indexed: dict[str, dict[CellKey, Path]] = {sample: {} for sample in requested}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in requested:
            continue
        key = row_cell_key(row)
        if key in indexed[sample_id]:
            raise RuntimeError(f"duplicate replay for sample={sample_id}, cell={key}")
        replay_path = Path(row["path"])
        if not replay_path.is_absolute():
            replay_path = index_path.parent / replay_path
        replay_path = replay_path.resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"missing replay for sample={sample_id}: {replay_path}")
        indexed[sample_id][key] = replay_path

    missing_samples = [sample for sample, cells in indexed.items() if not cells]
    if missing_samples:
        raise RuntimeError(f"capture index has no rows for samples: {missing_samples}")
    reference_sample = next(iter(sorted(indexed)))
    expected_cells = set(indexed[reference_sample])
    for sample_id, cells in sorted(indexed.items()):
        actual = set(cells)
        if actual != expected_cells:
            raise RuntimeError(
                f"cell coverage mismatch for {sample_id}: "
                f"missing={sorted(expected_cells - actual)[:3]} "
                f"extra={sorted(actual - expected_cells)[:3]}"
            )
    return indexed, tuple(sorted(expected_cells))


def right_singular_basis(defect: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    if defect.ndim != 2:
        raise ValueError("defect must be a [observations, channels] matrix")
    if rank <= 0:
        raise ValueError("rank must be positive")
    _, singular, vh = torch.linalg.svd(defect.float(), full_matrices=False)
    used = min(rank, vh.shape[0])
    total = float(singular.square().sum())
    energy = 1.0 if total <= 1e-30 else float(singular[:used].square().sum() / total)
    return vh[:used].T.contiguous(), energy


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    used = min(left.shape[1], right.shape[1])
    if used == 0:
        return 1.0
    return float((left.T @ right).square().sum() / used)


def basis_transfer_metrics(
    calibration_defect: torch.Tensor,
    heldout_defect: torch.Tensor,
    rank: int,
    calibration_basis: torch.Tensor | None = None,
    calibration_oracle_energy: float | None = None,
) -> dict[str, object]:
    if calibration_basis is None or calibration_oracle_energy is None:
        calibration_basis, calibration_oracle_energy = right_singular_basis(
            calibration_defect, rank
        )
    heldout_basis, heldout_oracle_energy = right_singular_basis(heldout_defect, rank)
    heldout_energy = float(heldout_defect.square().sum())
    projected = (heldout_defect @ calibration_basis) @ calibration_basis.T
    projected_energy = float(projected.square().sum())
    frozen_energy = 1.0 if heldout_energy <= 1e-30 else projected_energy / heldout_energy
    return {
        "calibration_basis": calibration_basis,
        "calibration_self_oracle_rank_energy": calibration_oracle_energy,
        "heldout_self_oracle_rank_energy": heldout_oracle_energy,
        "heldout_basis": heldout_basis,
        "frozen_calibration_basis_energy": frozen_energy,
        "subspace_overlap": subspace_overlap(calibration_basis, heldout_basis),
        "coefficient_oracle_projection": projected,
    }


def fit_ridge(
    features: torch.Tensor, targets: torch.Tensor, regularization: float
) -> RidgeModel:
    if regularization <= 0.0:
        raise ValueError("ridge regularization must be positive")
    x = features.float()
    y = targets.float()
    x_mean = x.mean(dim=0, keepdim=True)
    x_scale = (x - x_mean).square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    y_mean = y.mean(dim=0, keepdim=True)
    x_normalized = (x - x_mean) / x_scale
    y_centered = y - y_mean
    observations, feature_count = x_normalized.shape
    if observations <= feature_count:
        gram = x_normalized @ x_normalized.T
        gram.diagonal().add_(observations * regularization)
        dual = torch.linalg.solve(gram, y_centered)
        weight = x_normalized.T @ dual
    else:
        gram = x_normalized.T @ x_normalized / observations
        gram.diagonal().add_(regularization)
        weight = torch.linalg.solve(gram, x_normalized.T @ y_centered / observations)
    return RidgeModel(x_mean, x_scale, y_mean, weight)


def feature_matrix(record: dict[str, torch.Tensor], feature_name: str) -> torch.Tensor:
    if feature_name == "q":
        return record["q"]
    if feature_name == "sparse":
        return record["sparse"]
    if feature_name == "concat":
        return torch.cat((record["q"], record["sparse"]), dim=1)
    raise ValueError(f"unsupported ridge feature: {feature_name}")


def select_ridge_candidate(
    candidate_rows: list[dict[str, object]],
) -> dict[tuple[CellKey, str, int, int], dict[str, object]]:
    """Select only from validation rows; accepting test rows is a hard leak error."""
    if any(row.get("split") != "validation" for row in candidate_rows):
        raise ValueError("ridge selection may consume validation rows only")
    grouped: dict[tuple[CellKey, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        key = (
            (int(row["layer"]), int(row["sampling_step"]), str(row["branch"])),
            str(row["mask"]),
            int(row["head"]),
            int(row["rank"]),
        )
        grouped[key].append(row)
    selected: dict[tuple[CellKey, str, int, int], dict[str, object]] = {}
    for key, rows in grouped.items():
        by_candidate: dict[tuple[str, float], list[float]] = defaultdict(list)
        for row in rows:
            by_candidate[(str(row["ridge_feature"]), float(row["ridge_lambda"]))].append(
                float(row["ridge_corrected_output_relative_l2"])
            )
        best = min(
            by_candidate,
            key=lambda candidate: (
                sum(by_candidate[candidate]) / len(by_candidate[candidate]),
                candidate[1],
                candidate[0],
            ),
        )
        selected[key] = {
            "ridge_feature": best[0],
            "ridge_lambda": best[1],
            "validation_mean_relative_l2": sum(by_candidate[best]) / len(by_candidate[best]),
            "validation_samples": len(by_candidate[best]),
        }
    return selected


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_stop_go(
    rows: list[dict[str, object]],
    strict_target: float,
    relaxed_target: float,
    energy_target: float,
    overlap_target: float,
    ridge_enabled: bool,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["mask"]), int(row["rank"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (split, mask, rank), selected in sorted(grouped.items()):
        coefficient_errors = [
            float(row["coefficient_oracle_corrected_output_relative_l2"])
            for row in selected
        ]
        ridge_errors = [
            float(row["ridge_corrected_output_relative_l2"])
            for row in selected
            if row.get("ridge_corrected_output_relative_l2", "") != ""
        ]
        energies = [float(row["frozen_calibration_basis_energy"]) for row in selected]
        overlaps = [float(row["subspace_overlap"]) for row in selected]
        summaries.append(
            {
                "split": split,
                "mask": mask,
                "rank": rank,
                "evaluations": len(selected),
                "coefficient_oracle_error_mean": sum(coefficient_errors) / len(coefficient_errors),
                "coefficient_oracle_error_p95": quantile(coefficient_errors, 0.95),
                "coefficient_oracle_error_max": max(coefficient_errors),
                "coefficient_oracle_strict_2pct_go": max(coefficient_errors) <= strict_target,
                "coefficient_oracle_relaxed_5pct_go": max(coefficient_errors) <= relaxed_target,
                "frozen_basis_energy_mean": sum(energies) / len(energies),
                "frozen_basis_energy_p05": quantile(energies, 0.05),
                "subspace_overlap_mean": sum(overlaps) / len(overlaps),
                "subspace_overlap_p05": quantile(overlaps, 0.05),
                "basis_energy_overlap_go": (
                    quantile(energies, 0.05) >= energy_target
                    and quantile(overlaps, 0.05) >= overlap_target
                ),
                "ridge_enabled": ridge_enabled,
                "ridge_error_mean": (
                    sum(ridge_errors) / len(ridge_errors) if ridge_errors else ""
                ),
                "ridge_error_p95": quantile(ridge_errors, 0.95) if ridge_errors else "",
                "ridge_error_max": max(ridge_errors) if ridge_errors else "",
                "ridge_strict_2pct_go": (
                    max(ridge_errors) <= strict_target if ridge_errors else False
                ),
                "ridge_relaxed_5pct_go": (
                    max(ridge_errors) <= relaxed_target if ridge_errors else False
                ),
            }
        )
    return summaries


def resolve_specs(mask_names: tuple[str, ...]) -> tuple[GeometrySpec, ...]:
    by_name = {spec.name: spec for spec in DEFAULT_SPECS}
    missing = sorted(set(mask_names) - set(by_name))
    if missing:
        raise ValueError(f"unknown geometry masks: {missing}")
    if len(mask_names) != len(set(mask_names)):
        raise ValueError("mask list contains duplicates")
    return tuple(by_name[name] for name in mask_names)


@torch.inference_mode()
def reduce_replay(
    replay_path: Path,
    expected_sample_id: str,
    expected_cell: CellKey,
    specs: tuple[GeometrySpec, ...],
    query_samples: int,
    tile_height: int,
    tile_width: int,
    expected_heads: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(replay_path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    actual_cell = (
        int(metadata.get("layer", -1)),
        int(metadata.get("sampling_step", -1)),
        str(metadata.get("branch", "")),
    )
    if str(metadata.get("sample_id", "")) != expected_sample_id or actual_cell != expected_cell:
        raise RuntimeError(
            f"replay metadata mismatch for {replay_path}: "
            f"sample={metadata.get('sample_id')} cell={actual_cell}"
        )
    q_all = payload["q"][0]
    k_all = payload["k"][0]
    v_all = payload["v"][0]
    if q_all.shape != k_all.shape or q_all.shape != v_all.shape:
        raise RuntimeError(f"Q/K/V shape mismatch in {replay_path}")
    tokens, heads, dimension = q_all.shape
    if heads != expected_heads:
        raise RuntimeError(
            f"head coverage mismatch in {replay_path}: expected {expected_heads}, got {heads}"
        )
    shape = grid_from_metadata(metadata, tokens, fallback_height=30, fallback_width=52)
    query_indices = stratified_query_indices(
        shape, query_samples, tile_height, tile_width, device
    )
    masks = {
        spec.name: geometry_mask(
            query_indices,
            shape,
            spec,
            tile_height,
            tile_width,
            anchor_phase=int(metadata["layer"]),
        )[0]
        for spec in specs
    }
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    reduced_q: list[torch.Tensor] = []
    dense_outputs: list[torch.Tensor] = []
    sparse_outputs: dict[str, list[torch.Tensor]] = {spec.name: [] for spec in specs}
    cpu_queries = query_indices.cpu()
    for head in range(heads):
        sampled_q = q_all[:, head].index_select(0, cpu_queries).to(device, torch.float32)
        keys = k_all[:, head].to(device, torch.float32)
        values = v_all[:, head].to(device, torch.float32)
        scores = sampled_q @ keys.T * scale
        dense_output = torch.softmax(scores, dim=-1) @ values
        reduced_q.append(sampled_q.cpu())
        dense_outputs.append(dense_output.cpu())
        for spec in specs:
            sparse_scores = scores.masked_fill(~masks[spec.name], float("-inf"))
            sparse_outputs[spec.name].append(
                (torch.softmax(sparse_scores, dim=-1) @ values).cpu()
            )
        del sampled_q, keys, values, scores, dense_output
    return {
        "sample_id": expected_sample_id,
        "cell": expected_cell,
        "grid_size": shape,
        "query_indices": cpu_queries,
        "q": torch.stack(reduced_q, dim=1),
        "dense": torch.stack(dense_outputs, dim=1),
        "sparse": {
            name: torch.stack(outputs, dim=1) for name, outputs in sparse_outputs.items()
        },
    }


def make_head_record(
    reduced: dict[str, object], mask: str, head: int, device: torch.device
) -> dict[str, torch.Tensor]:
    q = reduced["q"]
    dense = reduced["dense"]
    sparse_by_mask = reduced["sparse"]
    assert isinstance(q, torch.Tensor) and isinstance(dense, torch.Tensor)
    assert isinstance(sparse_by_mask, dict)
    sparse = sparse_by_mask[mask]
    assert isinstance(sparse, torch.Tensor)
    return {
        "q": q[:, head].to(device=device, dtype=torch.float32),
        "dense": dense[:, head].to(device=device, dtype=torch.float32),
        "sparse": sparse[:, head].to(device=device, dtype=torch.float32),
        "defect": (dense[:, head] - sparse[:, head]).to(
            device=device, dtype=torch.float32
        ),
    }


def main() -> None:
    args = parse_args()
    if args.query_samples <= 0 or args.expected_heads <= 0:
        raise ValueError("query samples and expected heads must be positive")
    if not (0.0 < args.strict_error_target <= args.relaxed_error_target):
        raise ValueError("error targets must satisfy 0 < strict <= relaxed")
    if not set(args.ridge_features) <= {"q", "sparse", "concat"}:
        raise ValueError("ridge features must be selected from q,sparse,concat")
    splits = validate_split_ids(
        flatten_sample_ids(args.calibration_sample_id),
        flatten_sample_ids(args.validation_sample_id),
        flatten_sample_ids(args.test_sample_id),
    )
    specs = resolve_specs(args.masks)
    args.capture_index = args.capture_index.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_rows = read_capture_index(args.capture_index)
    replay_index, cells = index_capture_rows(capture_rows, splits, args.capture_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    transfer_rows: list[dict[str, object]] = []
    ridge_validation_rows: list[dict[str, object]] = []
    ridge_selection_rows: list[dict[str, object]] = []
    started = time.time()

    for cell in cells:
        reduced_by_sample = {
            sample_id: reduce_replay(
                replay_index[sample_id][cell],
                sample_id,
                cell,
                specs,
                args.query_samples,
                args.tile_height,
                args.tile_width,
                args.expected_heads,
                device,
            )
            for sample_id in sorted(replay_index)
        }
        reference_queries = next(iter(reduced_by_sample.values()))["query_indices"]
        reference_grid = next(iter(reduced_by_sample.values()))["grid_size"]
        for sample_id, reduced in reduced_by_sample.items():
            if not torch.equal(reference_queries, reduced["query_indices"]):
                raise RuntimeError(f"query-index coverage mismatch for sample {sample_id}, cell {cell}")
            if reference_grid != reduced["grid_size"]:
                raise RuntimeError(f"grid mismatch for sample {sample_id}, cell {cell}")

        for spec in specs:
            for head in range(args.expected_heads):
                records = {
                    sample_id: make_head_record(reduced, spec.name, head, device)
                    for sample_id, reduced in reduced_by_sample.items()
                }
                calibration_defect = torch.cat(
                    [records[sample_id]["defect"] for sample_id in splits["calibration"]]
                )
                calibration_records = [records[sample_id] for sample_id in splits["calibration"]]
                for rank in args.ranks:
                    calibration_basis, calibration_energy = right_singular_basis(
                        calibration_defect, rank
                    )
                    ridge_models: dict[tuple[str, float], RidgeModel] = {}
                    if not args.disable_ridge:
                        target = calibration_defect @ calibration_basis
                        for feature_name in args.ridge_features:
                            features = torch.cat(
                                [feature_matrix(record, feature_name) for record in calibration_records]
                            )
                            for ridge_lambda in args.ridge_lambdas:
                                ridge_models[(feature_name, ridge_lambda)] = fit_ridge(
                                    features, target, ridge_lambda
                                )

                    heldout_cache: dict[str, dict[str, object]] = {}
                    local_validation_rows: list[dict[str, object]] = []
                    for split in ("validation", "test"):
                        for sample_id in splits[split]:
                            record = records[sample_id]
                            metrics = basis_transfer_metrics(
                                calibration_defect,
                                record["defect"],
                                rank,
                                calibration_basis,
                                calibration_energy,
                            )
                            projected = metrics["coefficient_oracle_projection"]
                            assert isinstance(projected, torch.Tensor)
                            self_basis = metrics["heldout_basis"]
                            assert isinstance(self_basis, torch.Tensor)
                            self_projection = (
                                record["defect"] @ self_basis
                            ) @ self_basis.T
                            base = {
                                "split": split,
                                "sample_id": sample_id,
                                "layer": cell[0],
                                "sampling_step": cell[1],
                                "branch": cell[2],
                                "mask": spec.name,
                                "head": head,
                                "rank": rank,
                                "query_samples": args.query_samples,
                                "baseline_sparse_output_relative_l2": relative_l2(
                                    record["dense"], record["sparse"]
                                ),
                                "calibration_self_oracle_rank_energy": calibration_energy,
                                "heldout_self_oracle_rank_energy": metrics[
                                    "heldout_self_oracle_rank_energy"
                                ],
                                "frozen_calibration_basis_energy": metrics[
                                    "frozen_calibration_basis_energy"
                                ],
                                "subspace_overlap": metrics["subspace_overlap"],
                                "heldout_self_oracle_corrected_output_relative_l2": relative_l2(
                                    record["dense"], record["sparse"] + self_projection
                                ),
                                "coefficient_oracle_corrected_output_relative_l2": relative_l2(
                                    record["dense"], record["sparse"] + projected
                                ),
                                "coefficient_oracle": True,
                                "ridge_feature": "",
                                "ridge_lambda": "",
                                "ridge_corrected_output_relative_l2": "",
                            }
                            heldout_cache[sample_id] = {"base": base, "record": record}
                            if split == "validation" and ridge_models:
                                for (feature_name, ridge_lambda), model in ridge_models.items():
                                    coefficients = model.predict(
                                        feature_matrix(record, feature_name)
                                    )
                                    correction = coefficients @ calibration_basis.T
                                    candidate_row = {
                                            **{
                                                key: base[key]
                                                for key in (
                                                    "split", "sample_id", "layer",
                                                    "sampling_step", "branch", "mask",
                                                    "head", "rank",
                                                )
                                            },
                                            "ridge_feature": feature_name,
                                            "ridge_lambda": ridge_lambda,
                                            "ridge_corrected_output_relative_l2": relative_l2(
                                                record["dense"],
                                                record["sparse"] + correction,
                                            ),
                                        }
                                    ridge_validation_rows.append(candidate_row)
                                    local_validation_rows.append(candidate_row)

                    selection_key = (cell, spec.name, head, rank)
                    if ridge_models:
                        selection = select_ridge_candidate(local_validation_rows)[selection_key]
                        selected_feature = str(selection["ridge_feature"])
                        selected_lambda = float(selection["ridge_lambda"])
                        selected_model = ridge_models[(selected_feature, selected_lambda)]
                        ridge_selection_rows.append(
                            {
                                "layer": cell[0],
                                "sampling_step": cell[1],
                                "branch": cell[2],
                                "mask": spec.name,
                                "head": head,
                                "rank": rank,
                                **selection,
                                "selection_split": "validation",
                                "test_used_for_selection": False,
                            }
                        )
                        for sample_id, cached in heldout_cache.items():
                            record = cached["record"]
                            base = cached["base"]
                            assert isinstance(record, dict) and isinstance(base, dict)
                            coefficients = selected_model.predict(
                                feature_matrix(record, selected_feature)
                            )
                            correction = coefficients @ calibration_basis.T
                            base["ridge_feature"] = selected_feature
                            base["ridge_lambda"] = selected_lambda
                            base["ridge_corrected_output_relative_l2"] = relative_l2(
                                record["dense"], record["sparse"] + correction
                            )
                    transfer_rows.extend(
                        cached["base"] for cached in heldout_cache.values()
                    )
        del reduced_by_sample

    stop_go_rows = summarize_stop_go(
        transfer_rows,
        args.strict_error_target,
        args.relaxed_error_target,
        args.basis_energy_target,
        args.subspace_overlap_target,
        not args.disable_ridge,
    )
    write_csv(args.output_dir / "geometry_basis_transfer_metrics.csv", transfer_rows)
    write_csv(
        args.output_dir / "geometry_basis_ridge_validation_candidates.csv",
        ridge_validation_rows,
    )
    write_csv(args.output_dir / "geometry_basis_ridge_selection.csv", ridge_selection_rows)
    write_csv(args.output_dir / "geometry_basis_transfer_stop_go.csv", stop_go_rows)
    test_rows = [row for row in stop_go_rows if row["split"] == "test"]
    summary = {
        "splits": splits,
        "capture_index": str(args.capture_index),
        "cells": [
            {"layer": cell[0], "sampling_step": cell[1], "branch": cell[2]}
            for cell in cells
        ],
        "masks": list(args.masks),
        "ranks": list(args.ranks),
        "query_samples": args.query_samples,
        "strict_coverage": {
            "samples": len(replay_index),
            "cells_per_sample": len(cells),
            "heads_per_cell": args.expected_heads,
            "query_indices_identical_across_samples": True,
        },
        "thresholds": {
            "strict_output_relative_l2": args.strict_error_target,
            "relaxed_output_relative_l2": args.relaxed_error_target,
            "frozen_basis_energy_p05": args.basis_energy_target,
            "subspace_overlap_p05": args.subspace_overlap_target,
        },
        "test_stop_go": test_rows,
        "all_test_coefficient_oracle_strict_go": bool(test_rows)
        and all(bool(row["coefficient_oracle_strict_2pct_go"]) for row in test_rows),
        "all_test_coefficient_oracle_relaxed_go": bool(test_rows)
        and all(bool(row["coefficient_oracle_relaxed_5pct_go"]) for row in test_rows),
        "all_test_basis_energy_overlap_go": bool(test_rows)
        and all(bool(row["basis_energy_overlap_go"]) for row in test_rows),
        "all_test_ridge_strict_go": bool(test_rows)
        and not args.disable_ridge
        and all(bool(row["ridge_strict_2pct_go"]) for row in test_rows),
        "all_test_ridge_relaxed_go": bool(test_rows)
        and not args.disable_ridge
        and all(bool(row["ridge_relaxed_5pct_go"]) for row in test_rows),
        "selection_contract": (
            "basis and ridge weights use calibration only; ridge feature/lambda use "
            "validation only; test samples are evaluated exactly once and never select"
        ),
        "coefficient_oracle_warning": (
            "D_heldout @ V_cal uses held-out dense defects to obtain coefficients and is "
            "an oracle upper bound, not a deployable correction"
        ),
        "scope_warning": (
            "sampled pre-output-projection attention replay is not end-to-end video "
            "quality or fused-kernel latency evidence"
        ),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "cpu"
        ),
    }
    (args.output_dir / "geometry_basis_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
