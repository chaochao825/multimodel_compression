#!/usr/bin/env python3
"""Factorial and orthogonality audit for the oracle hybrid decomposition.

The existing hybrid probe decomposes a dense attention matrix into four
non-negative components:

    sink + global_svd + local_cyclic + sparse_routing

This script answers a different question from ordinary leave-one-out ablation:
are those components actually orthogonal, additive, and order independent?
It uses the saved representative attention matrices and the balanced hybrid
configuration.  No model forward pass is performed, so all conclusions remain
matrix-level diagnostics rather than task-level evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from hybrid_attention_decomposition import (
    clipped_svd_global_component,
    decompose,
    local_cyclic_projection,
    topk_sparse_component,
)


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
HYBRID_DECOMPOSITION_SCRIPT = ROOT / "scripts" / "hybrid_attention_decomposition.py"
COMPONENT_NAMES = ("sink", "global_svd", "local_cyclic", "sparse_routing")
CURRENT_ORDER = ("sink", "local_cyclic", "global_svd", "sparse_routing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom <= 1e-15:
        return 0.0
    return float(np.linalg.norm(target - approx) / denom)


def row_normalize_nonnegative(matrix: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(matrix, dtype=np.float64), 0.0, None)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def frobenius_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denom <= 1e-15:
        return 0.0
    return float(np.sum(left64 * right64) / denom)


def support_mask(matrix: np.ndarray, relative_tolerance: float) -> np.ndarray:
    values = np.abs(np.asarray(matrix, dtype=np.float64))
    scale = float(values.max(initial=0.0))
    if scale <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    return values > scale * float(relative_tolerance)


def support_jaccard(left: np.ndarray, right: np.ndarray, relative_tolerance: float) -> float:
    left_mask = support_mask(left, relative_tolerance)
    right_mask = support_mask(right, relative_tolerance)
    union = int(np.logical_or(left_mask, right_mask).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(left_mask, right_mask).sum() / union)


def support_overlap_coefficient(left: np.ndarray, right: np.ndarray, relative_tolerance: float) -> float:
    left_mask = support_mask(left, relative_tolerance)
    right_mask = support_mask(right, relative_tolerance)
    denom = min(int(left_mask.sum()), int(right_mask.sum()))
    if denom == 0:
        return 0.0
    return float(np.logical_and(left_mask, right_mask).sum() / denom)


def mass_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_pos = np.clip(np.asarray(left, dtype=np.float64), 0.0, None)
    right_pos = np.clip(np.asarray(right, dtype=np.float64), 0.0, None)
    denom = min(float(left_pos.sum()), float(right_pos.sum()))
    if denom <= 1e-15:
        return 0.0
    return float(np.minimum(left_pos, right_pos).sum() / denom)


def normalized_gram(components: Mapping[str, np.ndarray]) -> np.ndarray:
    vectors = []
    for name in COMPONENT_NAMES:
        vector = np.asarray(components[name], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        vectors.append(vector / norm if norm > 1e-15 else np.zeros_like(vector))
    stacked = np.stack(vectors, axis=1)
    return stacked.T @ stacked


def load_balanced_config(item: Mapping[str, object]) -> Mapping[str, object]:
    for config in item["hybrid_configs"]:  # type: ignore[index]
        if config["name"] == "hybrid_balanced":
            return config
    raise KeyError("missing hybrid_balanced config for {}".format(item.get("key")))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subset_label(names: Iterable[str]) -> str:
    selected = list(names)
    return "+".join(selected) if selected else "empty"


def factorial_ablation(
    attention: np.ndarray,
    components: Mapping[str, np.ndarray],
) -> Tuple[List[dict], Dict[int, float], Dict[str, float], List[dict]]:
    """Evaluate all component subsets and exact Shapley contributions."""

    rows: List[dict] = []
    utilities: Dict[int, float] = {}
    count = len(COMPONENT_NAMES)
    for mask in range(1 << count):
        selected = [name for idx, name in enumerate(COMPONENT_NAMES) if mask & (1 << idx)]
        if selected:
            raw = np.sum([components[name] for name in selected], axis=0)
            approx = row_normalize_nonnegative(raw)
        else:
            approx = np.zeros_like(attention, dtype=np.float64)
        error = relative_error(attention, approx)
        utility = 1.0 - error
        utilities[mask] = utility
        rows.append(
            {
                "subset_mask": mask,
                "subset": subset_label(selected),
                "component_count": len(selected),
                "relative_fro_error": error,
                "utility_one_minus_error": utility,
            }
        )

    factorial_n = math.factorial(count)
    shapley: Dict[str, float] = {}
    for idx, name in enumerate(COMPONENT_NAMES):
        bit = 1 << idx
        contribution = 0.0
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = math.factorial(size) * math.factorial(count - size - 1) / factorial_n
            contribution += weight * (utilities[mask | bit] - utilities[mask])
        shapley[name] = float(contribution)

    pair_rows: List[dict] = []
    for left_idx, right_idx in itertools.combinations(range(count), 2):
        left_bit = 1 << left_idx
        right_bit = 1 << right_idx
        pair_mask = left_bit | right_bit
        remaining = [idx for idx in range(count) if idx not in (left_idx, right_idx)]
        for background_bits in range(1 << len(remaining)):
            background_mask = 0
            background_names = []
            for local_idx, component_idx in enumerate(remaining):
                if background_bits & (1 << local_idx):
                    background_mask |= 1 << component_idx
                    background_names.append(COMPONENT_NAMES[component_idx])
            interaction = (
                utilities[background_mask | pair_mask]
                - utilities[background_mask | left_bit]
                - utilities[background_mask | right_bit]
                + utilities[background_mask]
            )
            pair_rows.append(
                {
                    "left": COMPONENT_NAMES[left_idx],
                    "right": COMPONENT_NAMES[right_idx],
                    "background_mask": background_mask,
                    "background": subset_label(background_names),
                    "background_size": len(background_names),
                    "interaction": float(interaction),
                    "is_empty_context": bool(background_mask == 0),
                    "is_full_context": bool(len(background_names) == count - 2),
                }
            )
    return rows, utilities, shapley, pair_rows


def sequential_fit(
    attention: np.ndarray,
    grid_shape: Iterable[int],
    sink_cols: np.ndarray,
    rank: int,
    radius: int,
    sparse_k: int,
    order: Sequence[str],
) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
    """Fit the same component operators in an arbitrary order.

    Sink columns are fixed from the original attention matrix so the order test
    measures residual-allocation sensitivity rather than support reselection.
    """

    residual = np.asarray(attention, dtype=np.float64).copy()
    components: Dict[str, np.ndarray] = {}
    for name in order:
        if name == "sink":
            component = np.zeros_like(residual)
            component[:, sink_cols] = residual[:, sink_cols]
        elif name == "local_cyclic":
            component, _ = local_cyclic_projection(residual, grid_shape, int(radius))
        elif name == "global_svd":
            component = clipped_svd_global_component(residual, int(rank), sink_cols)
        elif name == "sparse_routing":
            component = topk_sparse_component(residual, int(sparse_k))
        else:
            raise ValueError("unknown component: {}".format(name))
        components[name] = np.asarray(component, dtype=np.float64)
        residual = np.clip(residual - component, 0.0, None)

    approx = row_normalize_nonnegative(np.sum([components[name] for name in order], axis=0))
    return relative_error(attention, approx), components, approx


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else float("nan")


def max_value(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.max(vals)) if vals else float("nan")


def run_audit(
    input_json: Path,
    input_npz: Path,
    support_tolerance: float,
    exact_cosine_tolerance: float,
    near_cosine_threshold: float,
    order_range_threshold: float,
    interaction_threshold: float,
) -> dict:
    metadata = json.loads(input_json.read_text(encoding="utf-8"))
    arrays = np.load(input_npz)
    example_rows: List[dict] = []
    pairwise_rows: List[dict] = []
    subset_rows: List[dict] = []
    interaction_rows: List[dict] = []
    order_rows: List[dict] = []
    gram_matrices: List[np.ndarray] = []

    for item in metadata["items"]:
        key = str(item["key"])
        attention = arrays[f"{key}_attention"].astype(np.float64)
        balanced = load_balanced_config(item)
        grid_shape = tuple(int(value) for value in item["grid_shape"])
        result = decompose(
            attention,
            grid_shape,
            int(balanced["sink_k"]),
            int(balanced["rank"]),
            int(balanced["radius"]),
            int(balanced["sparse_k"]),
        )
        components = {
            name: np.asarray(result[name], dtype=np.float64)
            for name in COMPONENT_NAMES
        }
        gram = normalized_gram(components)
        gram_matrices.append(gram)
        off_diag = np.abs(gram[np.triu_indices(len(COMPONENT_NAMES), k=1)])
        component_energy = sum(float(np.sum(value * value)) for value in components.values())
        sum_component = np.sum(list(components.values()), axis=0)
        additive_energy_ratio = float(np.sum(sum_component * sum_component) / max(component_energy, 1e-15))

        for left_idx, right_idx in itertools.combinations(range(len(COMPONENT_NAMES)), 2):
            left_name = COMPONENT_NAMES[left_idx]
            right_name = COMPONENT_NAMES[right_idx]
            left = components[left_name]
            right = components[right_name]
            pairwise_rows.append(
                {
                    "key": key,
                    "label": item["label"],
                    "map_id": item["map_id"],
                    "left": left_name,
                    "right": right_name,
                    "frobenius_cosine": frobenius_cosine(left, right),
                    "support_jaccard": support_jaccard(left, right, support_tolerance),
                    "support_overlap_coefficient": support_overlap_coefficient(left, right, support_tolerance),
                    "mass_overlap": mass_overlap(left, right),
                }
            )

        current_order_error, _, current_approx = sequential_fit(
            attention,
            grid_shape,
            np.asarray(result["sink_cols"], dtype=np.int64),
            int(balanced["rank"]),
            int(balanced["radius"]),
            int(balanced["sparse_k"]),
            CURRENT_ORDER,
        )
        saved_hybrid = arrays[f"{key}_hybrid"].astype(np.float64)
        saved_reproduction_max_abs = float(np.max(np.abs(current_approx - saved_hybrid)))
        saved_component_reproduction_max_abs = max(
            float(
                np.max(
                    np.abs(
                        (components["sink"] + components["global_svd"])
                        - arrays[f"{key}_sink_global_svd"].astype(np.float64)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        components["local_cyclic"]
                        - arrays[f"{key}_local_cyclic"].astype(np.float64)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        components["sparse_routing"]
                        - arrays[f"{key}_sparse_routing"].astype(np.float64)
                    )
                )
            ),
        )

        local_order_rows: List[dict] = []
        for order in itertools.permutations(COMPONENT_NAMES):
            error, _components, _approx = sequential_fit(
                attention,
                grid_shape,
                np.asarray(result["sink_cols"], dtype=np.int64),
                int(balanced["rank"]),
                int(balanced["radius"]),
                int(balanced["sparse_k"]),
                order,
            )
            row = {
                "key": key,
                "label": item["label"],
                "map_id": item["map_id"],
                "order": ">".join(order),
                "is_current_order": bool(tuple(order) == CURRENT_ORDER),
                "relative_fro_error": error,
            }
            order_rows.append(row)
            local_order_rows.append(row)

        local_subset_rows, utilities, shapley, local_interactions = factorial_ablation(attention, components)
        for row in local_subset_rows:
            subset_rows.append(
                {
                    "key": key,
                    "label": item["label"],
                    "map_id": item["map_id"],
                    **row,
                }
            )
        for row in local_interactions:
            interaction_rows.append(
                {
                    "key": key,
                    "label": item["label"],
                    "map_id": item["map_id"],
                    **row,
                }
            )

        order_errors = [float(row["relative_fro_error"]) for row in local_order_rows]
        interaction_abs = [abs(float(row["interaction"])) for row in local_interactions]
        exact_orthogonal = bool(float(off_diag.max(initial=0.0)) <= exact_cosine_tolerance)
        near_orthogonal = bool(
            float(off_diag.max(initial=0.0)) <= near_cosine_threshold
            and (max(order_errors) - min(order_errors)) <= order_range_threshold
            and max(interaction_abs, default=0.0) <= interaction_threshold
        )
        baseline = item["baseline_relative_fro_error"]
        full_mask = (1 << len(COMPONENT_NAMES)) - 1
        example_rows.append(
            {
                "key": key,
                "label": item["label"],
                "map_id": item["map_id"],
                "n": int(attention.shape[0]),
                "full_hybrid_error": float(result["relative_fro_error"]),
                "grid_bccb_error": float(baseline["grid_cyclic_bccb"]),
                "monarch_proxy_error": float(baseline["monarch_like_mask_proxy"]),
                "relative_improvement_vs_grid_bccb": float(
                    (float(baseline["grid_cyclic_bccb"]) - float(result["relative_fro_error"]))
                    / max(float(baseline["grid_cyclic_bccb"]), 1e-15)
                ),
                "relative_improvement_vs_monarch_proxy": float(
                    (float(baseline["monarch_like_mask_proxy"]) - float(result["relative_fro_error"]))
                    / max(float(baseline["monarch_like_mask_proxy"]), 1e-15)
                ),
                "max_pairwise_frobenius_cosine": float(off_diag.max(initial=0.0)),
                "mean_pairwise_frobenius_cosine": float(off_diag.mean()),
                "additive_energy_ratio": additive_energy_ratio,
                "normalized_gram_condition_number": float(np.linalg.cond(gram)),
                "current_order_error": current_order_error,
                "best_order_error": min(order_errors),
                "worst_order_error": max(order_errors),
                "order_error_range": max(order_errors) - min(order_errors),
                "order_error_std": float(np.std(order_errors)),
                "best_order": min(local_order_rows, key=lambda row: float(row["relative_fro_error"]))["order"],
                "worst_order": max(local_order_rows, key=lambda row: float(row["relative_fro_error"]))["order"],
                "max_abs_factorial_interaction": max(interaction_abs, default=0.0),
                "mean_abs_factorial_interaction": mean(interaction_abs),
                "factorial_full_utility": float(utilities[full_mask]),
                "shapley_sink": shapley["sink"],
                "shapley_global_svd": shapley["global_svd"],
                "shapley_local_cyclic": shapley["local_cyclic"],
                "shapley_sparse_routing": shapley["sparse_routing"],
                "shapley_efficiency_residual": float(
                    sum(shapley.values()) - (utilities[full_mask] - utilities[0])
                ),
                "saved_hybrid_reproduction_max_abs": saved_reproduction_max_abs,
                "saved_component_reproduction_max_abs": saved_component_reproduction_max_abs,
                "saved_component_split_note": (
                    "The input NPZ stores sink+global_svd jointly, so the individual sink/global split is recomputed "
                    "from the hashed decomposition implementation and is not independently identifiable from the NPZ."
                ),
                "exact_frobenius_orthogonal": exact_orthogonal,
                "near_orthogonal_under_declared_thresholds": near_orthogonal,
            }
        )

    average_gram = np.mean(np.stack(gram_matrices, axis=0), axis=0)
    pair_cosines = [float(row["frobenius_cosine"]) for row in pairwise_rows]
    pair_jaccards = [float(row["support_jaccard"]) for row in pairwise_rows]
    pair_mass_overlap = [float(row["mass_overlap"]) for row in pairwise_rows]
    order_ranges = [float(row["order_error_range"]) for row in example_rows]
    interaction_abs_all = [abs(float(row["interaction"])) for row in interaction_rows]
    exact_count = sum(bool(row["exact_frobenius_orthogonal"]) for row in example_rows)
    near_count = sum(bool(row["near_orthogonal_under_declared_thresholds"]) for row in example_rows)

    aggregate = {
        "examples": len(example_rows),
        "mean_full_hybrid_error": mean(row["full_hybrid_error"] for row in example_rows),
        "mean_grid_bccb_error": mean(row["grid_bccb_error"] for row in example_rows),
        "mean_monarch_proxy_error": mean(row["monarch_proxy_error"] for row in example_rows),
        "mean_relative_improvement_vs_grid_bccb": mean(
            row["relative_improvement_vs_grid_bccb"] for row in example_rows
        ),
        "mean_relative_improvement_vs_monarch_proxy": mean(
            row["relative_improvement_vs_monarch_proxy"] for row in example_rows
        ),
        "mean_pairwise_frobenius_cosine": mean(pair_cosines),
        "max_pairwise_frobenius_cosine": max_value(pair_cosines),
        "mean_support_jaccard": mean(pair_jaccards),
        "max_support_jaccard": max_value(pair_jaccards),
        "mean_mass_overlap": mean(pair_mass_overlap),
        "max_mass_overlap": max_value(pair_mass_overlap),
        "mean_order_error_range": mean(order_ranges),
        "max_order_error_range": max_value(order_ranges),
        "mean_abs_factorial_interaction": mean(interaction_abs_all),
        "max_abs_factorial_interaction": max_value(interaction_abs_all),
        "exact_orthogonal_examples": exact_count,
        "near_orthogonal_examples": near_count,
        "average_normalized_gram": average_gram.tolist(),
        "orthogonality_conclusion": (
            "not_orthogonal" if exact_count < len(example_rows) else "numerically_orthogonal"
        ),
        "near_orthogonality_conclusion": (
            "not_near_orthogonal_under_declared_thresholds"
            if near_count < len(example_rows)
            else "near_orthogonal_under_declared_thresholds"
        ),
        "evidence_grade": "oracle_matrix_level_representative_examples_only",
    }
    return {
        "component_names": list(COMPONENT_NAMES),
        "current_fit_order": list(CURRENT_ORDER),
        "thresholds": {
            "support_relative_tolerance": support_tolerance,
            "exact_pairwise_cosine_tolerance": exact_cosine_tolerance,
            "near_pairwise_cosine_threshold": near_cosine_threshold,
            "near_order_error_range_threshold": order_range_threshold,
            "near_factorial_interaction_threshold": interaction_threshold,
            "threshold_note": (
                "Near-orthogonality thresholds are declared engineering diagnostics, not universal statistical laws."
            ),
        },
        "method_notes": {
            "nonnegative_orthogonality": (
                "For non-negative components, exact Frobenius orthogonality requires disjoint positive support."
            ),
            "order_test": (
                "If the residual extractors were mutually orthogonal commuting projections, fitting order would not change the result."
            ),
            "factorial_test": (
                "All 2^4 subsets are row-normalized before matrix error; pair interactions expose non-additivity from overlap and normalization."
            ),
            "scope": (
                "Saved oracle components are selected from dense target attention. This cannot establish deployability, speedup, or task quality."
            ),
            "component_provenance": (
                "The saved NPZ contains sink+global_svd jointly. Their individual split is replayed with the separately hashed "
                "hybrid_attention_decomposition.py implementation."
            ),
        },
        "examples": example_rows,
        "pairwise": pairwise_rows,
        "factorial_subsets": subset_rows,
        "factorial_interactions": interaction_rows,
        "fit_orders": order_rows,
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=LOG_DIR / "hybrid_attention_decomposition_20260704.json",
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=LOG_DIR / "hybrid_attention_decomposition_20260704.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=LOG_DIR / "component_orthogonality_ablation_20260711.json",
    )
    parser.add_argument(
        "--output-pairwise-csv",
        type=Path,
        default=LOG_DIR / "component_orthogonality_pairwise_20260711.csv",
    )
    parser.add_argument(
        "--output-factorial-csv",
        type=Path,
        default=LOG_DIR / "component_orthogonality_factorial_20260711.csv",
    )
    parser.add_argument(
        "--output-interaction-csv",
        type=Path,
        default=LOG_DIR / "component_orthogonality_interactions_20260711.csv",
    )
    parser.add_argument(
        "--output-order-csv",
        type=Path,
        default=LOG_DIR / "component_orthogonality_orders_20260711.csv",
    )
    parser.add_argument("--support-relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--exact-cosine-tolerance", type=float, default=1e-8)
    parser.add_argument("--near-cosine-threshold", type=float, default=0.05)
    parser.add_argument("--order-error-range-threshold", type=float, default=0.05)
    parser.add_argument("--interaction-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    payload = run_audit(
        args.input_json,
        args.input_npz,
        args.support_relative_tolerance,
        args.exact_cosine_tolerance,
        args.near_cosine_threshold,
        args.order_error_range_threshold,
        args.interaction_threshold,
    )
    payload.update(
        {
            "created_unix": time.time(),
            "elapsed_sec": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": sha256_file(Path(__file__)),
            "hybrid_decomposition_script": repo_relative(HYBRID_DECOMPOSITION_SCRIPT),
            "hybrid_decomposition_script_sha256": sha256_file(HYBRID_DECOMPOSITION_SCRIPT),
            "input_json": repo_relative(args.input_json),
            "input_json_sha256": sha256_file(args.input_json),
            "input_npz": repo_relative(args.input_npz),
            "input_npz_sha256": sha256_file(args.input_npz),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_pairwise_csv, payload["pairwise"])
    write_csv(args.output_factorial_csv, payload["factorial_subsets"])
    write_csv(args.output_interaction_csv, payload["factorial_interactions"])
    write_csv(args.output_order_csv, payload["fit_orders"])
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
