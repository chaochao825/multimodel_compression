#!/usr/bin/env python3
"""Oracle hybrid decomposition diagnostics for attention matrices.

This script is intentionally value-free: it reads the representative attention
matrices exported earlier and tests whether a sink/global + local-cyclic +
sparse-routing decomposition explains A better than single-family structured
projections. The routing and sink choices are selected from the observed A, so
the result is a diagnostic upper bound, not a deployable replacement kernel.
The SVD-derived global component is clipped/capped for nonnegativity, so its
nominal budget is not a valid compressed low-rank representation cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(target - approx) / denom)


def row_normalize_nonnegative(matrix: np.ndarray) -> np.ndarray:
    out = np.clip(matrix.astype(np.float64), 0.0, None)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def offset_ids_for_grid(grid_shape: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(v) for v in grid_shape)
    coords = np.array(list(np.ndindex(shape)), dtype=np.int64)
    grid = np.array(shape, dtype=np.int64)
    offsets = (coords[None, :, :] - coords[:, None, :]) % grid
    multipliers = np.cumprod((1,) + shape[::-1])[:-1][::-1]
    offset_ids = (offsets * multipliers).sum(axis=-1)
    return offset_ids, coords


def local_cyclic_projection(matrix: np.ndarray, grid_shape: Iterable[int], radius: int) -> Tuple[np.ndarray, int]:
    shape = tuple(int(v) for v in grid_shape)
    n = int(np.prod(shape))
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match grid {shape}")
    offset_ids, offset_coords = offset_ids_for_grid(shape)
    flat_ids = offset_ids.reshape(-1)
    flat_values = matrix.astype(np.float64).reshape(-1)
    sums = np.bincount(flat_ids, weights=flat_values, minlength=n)
    counts = np.bincount(flat_ids, minlength=n)
    kernel = sums / np.maximum(counts, 1)

    shape_np = np.array(shape, dtype=np.int64)
    wrapped_dist = np.minimum(offset_coords, shape_np - offset_coords)
    keep = wrapped_dist.max(axis=1) <= int(radius)
    kernel = kernel * keep
    return kernel[offset_ids], int(keep.sum())


def clipped_svd_global_component(matrix: np.ndarray, rank: int, forbidden_cols: np.ndarray) -> np.ndarray:
    if rank <= 0:
        return np.zeros_like(matrix, dtype=np.float64)
    u, s, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    used = min(int(rank), s.shape[0])
    global_component = (u[:, :used] * s[:used]) @ vt[:used, :]
    global_component = np.clip(global_component, 0.0, None)
    if forbidden_cols.size:
        global_component[:, forbidden_cols] = 0.0
    return np.minimum(global_component, matrix)


def topk_sparse_component(matrix: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float64)
    if k <= 0:
        return out
    k = min(int(k), matrix.shape[1])
    idx = np.argpartition(matrix, -k, axis=1)[:, -k:]
    rows = np.arange(matrix.shape[0])[:, None]
    out[rows, idx] = matrix[rows, idx]
    return out


def default_configs(n: int) -> List[Dict[str, int | str]]:
    return [
        {
            "name": "hybrid_tiny",
            "sink_k": max(1, n // 96),
            "rank": 1,
            "radius": 1,
            "sparse_k": max(1, n // 96),
        },
        {
            "name": "hybrid_small",
            "sink_k": max(2, n // 64),
            "rank": 2 if n <= 64 else 3,
            "radius": 1,
            "sparse_k": max(2, n // 64),
        },
        {
            "name": "hybrid_balanced",
            "sink_k": max(2, n // 48),
            "rank": 2 if n <= 64 else 4,
            "radius": 1,
            "sparse_k": max(2, n // 48),
        },
        {
            "name": "hybrid_plus",
            "sink_k": max(3, n // 32),
            "rank": 4 if n <= 64 else 6,
            "radius": 2,
            "sparse_k": max(3, n // 32),
        },
    ]


def decompose(
    attention: np.ndarray,
    grid_shape: Iterable[int],
    sink_k: int,
    rank: int,
    radius: int,
    sparse_k: int,
) -> Dict[str, np.ndarray | float | int | List[int]]:
    attn = attention.astype(np.float64)
    n = attn.shape[0]
    sink_cols = np.argsort(-attn.sum(axis=0))[: int(sink_k)]

    sink = np.zeros_like(attn)
    sink[:, sink_cols] = attn[:, sink_cols]
    residual = np.clip(attn - sink, 0.0, None)

    local, local_params = local_cyclic_projection(residual, grid_shape, int(radius))
    residual = np.clip(residual - local, 0.0, None)

    global_svd = clipped_svd_global_component(residual, int(rank), sink_cols)
    residual = np.clip(residual - global_svd, 0.0, None)

    sparse = topk_sparse_component(residual, int(sparse_k))
    raw_sum = sink + global_svd + local + sparse
    approx = row_normalize_nonnegative(raw_sum)

    nominal_budget_params = (
        n * int(sink_k)
        + (2 * n * int(rank) + int(rank) if int(rank) > 0 else 0)
        + int(local_params)
        + 2 * n * int(sparse_k)
    )
    numeric_rank = int(np.linalg.matrix_rank(global_svd, tol=1e-8))
    component_mass = {
        "sink_mass": float(sink.sum() / max(attn.sum(), 1e-12)),
        "local_mass": float(local.sum() / max(attn.sum(), 1e-12)),
        "global_svd_mass": float(global_svd.sum() / max(attn.sum(), 1e-12)),
        "sparse_mass": float(sparse.sum() / max(attn.sum(), 1e-12)),
    }
    return {
        "attention": attn,
        "sink": sink,
        "global_svd": global_svd,
        "sink_global_svd": sink + global_svd,
        "local_cyclic": local,
        "sparse_routing": sparse,
        "hybrid": approx,
        "residual": np.abs(attn - approx),
        "relative_fro_error": relative_error(attn, approx),
        "nominal_budget_params": int(nominal_budget_params),
        "dense_params": int(attn.size),
        "nominal_budget_ratio": float(attn.size / max(nominal_budget_params, 1)),
        "global_svd_numeric_rank_after_clipping": numeric_rank,
        "local_params": int(local_params),
        "sink_cols": [int(v) for v in sink_cols.tolist()],
        "budget_note": (
            "Nominal budget only. The clipped/capped SVD-derived global component "
            "is not a strict rank-constrained matrix representation."
        ),
        **component_mass,
    }


def load_examples() -> List[Tuple[dict, Dict[str, np.ndarray]]]:
    sources = [
        (
            json.loads((LOG_DIR / "structured_attention_visual_vit_examples_20260703.json").read_text(encoding="utf-8")),
            np.load(LOG_DIR / "structured_attention_visual_vit_examples_20260703.npz"),
        ),
        (
            json.loads((LOG_DIR / "structured_attention_visual_qwen_examples_20260703.json").read_text(encoding="utf-8")),
            np.load(LOG_DIR / "structured_attention_visual_qwen_examples_20260703.npz"),
        ),
    ]
    examples: List[Tuple[dict, Dict[str, np.ndarray]]] = []
    for meta, arrays in sources:
        for item in meta["items"]:
            examples.append((item, arrays))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(LOG_DIR / "hybrid_attention_decomposition_20260704.json"))
    parser.add_argument("--out-npz", default=str(LOG_DIR / "hybrid_attention_decomposition_20260704.npz"))
    args = parser.parse_args()

    out_arrays: Dict[str, np.ndarray] = {}
    out_items: List[dict] = []
    summary_rows: List[dict] = []

    for item, arrays in load_examples():
        key = f"ex{len(out_items)}"
        source_key = str(item["key"])
        attn = arrays[f"{source_key}_attention"].astype(np.float64)
        n = attn.shape[0]
        grid_shape = tuple(int(v) for v in item["grid_shape"])

        baseline_metrics = {
            name: float(item["metrics"][name]["relative_fro_error"])
            for name in [
                "grid_cyclic_bccb",
                "flat_block_circulant",
                "permuted_flat_block_circulant",
                "monarch_like_mask_proxy",
            ]
        }

        configs = []
        for config in default_configs(n):
            result = decompose(
                attn,
                grid_shape,
                int(config["sink_k"]),
                int(config["rank"]),
                int(config["radius"]),
                int(config["sparse_k"]),
            )
            metric_row = {
                "name": config["name"],
                "sink_k": int(config["sink_k"]),
                "rank": int(config["rank"]),
                "radius": int(config["radius"]),
                "sparse_k": int(config["sparse_k"]),
                "relative_fro_error": float(result["relative_fro_error"]),
                "nominal_budget_params": int(result["nominal_budget_params"]),
                "dense_params": int(result["dense_params"]),
                "nominal_budget_ratio": float(result["nominal_budget_ratio"]),
                "global_svd_numeric_rank_after_clipping": int(result["global_svd_numeric_rank_after_clipping"]),
                "local_params": int(result["local_params"]),
                "sink_cols": result["sink_cols"],
                "sink_mass": float(result["sink_mass"]),
                "local_mass": float(result["local_mass"]),
                "global_svd_mass": float(result["global_svd_mass"]),
                "sparse_mass": float(result["sparse_mass"]),
                "budget_note": str(result["budget_note"]),
            }
            configs.append(metric_row)
            summary_rows.append(
                {
                    "label": item["label"],
                    "map_id": item["map_id"],
                    **metric_row,
                }
            )

            if config["name"] == "hybrid_balanced":
                for array_name in [
                    "attention",
                    "sink_global_svd",
                    "local_cyclic",
                    "sparse_routing",
                    "hybrid",
                    "residual",
                ]:
                    out_arrays[f"{key}_{array_name}"] = np.asarray(result[array_name], dtype=np.float32)

        best = min(configs, key=lambda row: row["relative_fro_error"])
        out_items.append(
            {
                "key": key,
                "source_key": source_key,
                "label": item["label"],
                "map_id": item["map_id"],
                "grid_shape": list(grid_shape),
                "baseline_relative_fro_error": baseline_metrics,
                "hybrid_configs": configs,
                "best_hybrid": best,
                "diagnostic_note": (
                    "Oracle diagnostic: sink columns and sparse routing are selected from observed A; "
                    "this tests structural decomposability, not a deployable replacement."
                ),
            }
        )

    out_json = Path(args.out_json)
    out_npz = Path(args.out_npz)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "method": "sink/low-rank + local-cyclic + sparse-routing oracle decomposition",
                "items": out_items,
                "summary_rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(out_npz, **out_arrays)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
