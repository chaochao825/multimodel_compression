#!/usr/bin/env python3
"""Matrix-level component intervention probe for attention patterns.

This probe uses the saved oracle hybrid decomposition:

    sink/global-SVD + local-cyclic + sparse-routing

For each representative attention matrix, it ablates or keeps one component at
a time and measures the relative Frobenius error to the original dense
attention matrix. This is a causal-like diagnostic of which structural family
is needed to explain A. It is not a task-level causal intervention because it
does not run the model forward or measure loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List

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


def component_density(matrix: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs(matrix) > eps))


def component_mass(matrix: np.ndarray, total_mass: float) -> float:
    return float(np.clip(matrix, 0.0, None).sum() / max(total_mass, 1e-12))


def mean_float(rows: Iterable[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def load_balanced_config(item: dict) -> dict:
    for config in item["hybrid_configs"]:
        if config["name"] == "hybrid_balanced":
            return config
    raise KeyError(f"missing hybrid_balanced config for {item.get('key')}")


def explain_dominant_failure(row: dict) -> str:
    deltas = {
        "sink_global": row["delta_no_sink_global"],
        "local_cyclic": row["delta_no_local_cyclic"],
        "sparse_routing": row["delta_no_sparse_routing"],
    }
    name, value = max(deltas.items(), key=lambda kv: kv[1])
    if value < 0.03:
        return "no_single_component_dominates"
    if name == "sink_global":
        return "sink_or_low_rank_global_needed"
    if name == "local_cyclic":
        return "local_cyclic_needed"
    return "sparse_dynamic_routing_needed"


def probe(input_json: Path, input_npz: Path) -> tuple[List[dict], List[dict]]:
    meta = json.loads(input_json.read_text(encoding="utf-8"))
    arrays = np.load(input_npz)
    rows: List[dict] = []
    for item in meta["items"]:
        key = str(item["key"])
        attention = arrays[f"{key}_attention"].astype(np.float64)
        sink_global = arrays[f"{key}_sink_global_svd"].astype(np.float64)
        local_cyclic = arrays[f"{key}_local_cyclic"].astype(np.float64)
        sparse_routing = arrays[f"{key}_sparse_routing"].astype(np.float64)
        full_hybrid = arrays[f"{key}_hybrid"].astype(np.float64)
        total_mass = float(attention.sum())

        variants: Dict[str, np.ndarray] = {
            "full_hybrid": full_hybrid,
            "only_sink_global": row_normalize_nonnegative(sink_global),
            "only_local_cyclic": row_normalize_nonnegative(local_cyclic),
            "only_sparse_routing": row_normalize_nonnegative(sparse_routing),
            "no_sink_global": row_normalize_nonnegative(local_cyclic + sparse_routing),
            "no_local_cyclic": row_normalize_nonnegative(sink_global + sparse_routing),
            "no_sparse_routing": row_normalize_nonnegative(sink_global + local_cyclic),
        }
        errors = {f"{name}_error": relative_error(attention, approx) for name, approx in variants.items()}
        balanced = load_balanced_config(item)
        baseline = item["baseline_relative_fro_error"]
        row = {
            "key": key,
            "label": item["label"],
            "map_id": item["map_id"],
            "grid_shape": "x".join(str(v) for v in item["grid_shape"]),
            "n": int(attention.shape[0]),
            "grid_bccb_error": float(baseline["grid_cyclic_bccb"]),
            "flat_bcm_error": float(baseline["flat_block_circulant"]),
            "permuted_bcm_error": float(baseline["permuted_flat_block_circulant"]),
            "monarch_proxy_error": float(baseline["monarch_like_mask_proxy"]),
            "sink_global_mass": component_mass(sink_global, total_mass),
            "local_cyclic_mass": component_mass(local_cyclic, total_mass),
            "sparse_routing_mass": component_mass(sparse_routing, total_mass),
            "sink_global_density": component_density(sink_global),
            "local_cyclic_density": component_density(local_cyclic),
            "sparse_routing_density": component_density(sparse_routing),
            "nominal_budget_ratio": float(balanced["nominal_budget_ratio"]),
            **errors,
        }
        row.update(
            {
                "delta_no_sink_global": row["no_sink_global_error"] - row["full_hybrid_error"],
                "delta_no_local_cyclic": row["no_local_cyclic_error"] - row["full_hybrid_error"],
                "delta_no_sparse_routing": row["no_sparse_routing_error"] - row["full_hybrid_error"],
            }
        )
        row["dominant_failure_mode"] = explain_dominant_failure(row)
        rows.append(row)

    summary = [
        {
            "scope": "representative_attention_component_intervention",
            "examples": len(rows),
            "mean_grid_bccb_error": mean_float(rows, "grid_bccb_error"),
            "mean_monarch_proxy_error": mean_float(rows, "monarch_proxy_error"),
            "mean_full_hybrid_error": mean_float(rows, "full_hybrid_error"),
            "mean_no_sink_global_error": mean_float(rows, "no_sink_global_error"),
            "mean_no_local_cyclic_error": mean_float(rows, "no_local_cyclic_error"),
            "mean_no_sparse_routing_error": mean_float(rows, "no_sparse_routing_error"),
            "mean_only_sink_global_error": mean_float(rows, "only_sink_global_error"),
            "mean_only_local_cyclic_error": mean_float(rows, "only_local_cyclic_error"),
            "mean_only_sparse_routing_error": mean_float(rows, "only_sparse_routing_error"),
            "mean_delta_no_sink_global": mean_float(rows, "delta_no_sink_global"),
            "mean_delta_no_local_cyclic": mean_float(rows, "delta_no_local_cyclic"),
            "mean_delta_no_sparse_routing": mean_float(rows, "delta_no_sparse_routing"),
            "method_note": (
                "Matrix-level intervention over saved oracle decomposition. "
                "Components are chosen from observed dense A; this does not prove task-level causality."
            ),
        }
    ]
    return rows, summary


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default=str(LOG_DIR / "hybrid_attention_decomposition_20260704.json"))
    parser.add_argument("--input-npz", default=str(LOG_DIR / "hybrid_attention_decomposition_20260704.npz"))
    parser.add_argument("--output-json", default=str(LOG_DIR / "attention_component_intervention_20260707.json"))
    parser.add_argument("--output-csv", default=str(LOG_DIR / "attention_component_intervention_20260707.csv"))
    args = parser.parse_args()

    t0 = time.time()
    rows, summary = probe(Path(args.input_json), Path(args.input_npz))
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "python": platform.python_version(),
        "input_json": args.input_json,
        "input_npz": args.input_npz,
        "rows": rows,
        "summary": summary,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, rows)
    print(json.dumps({"rows": len(rows), "summary": len(summary), "output_json": str(output_json), "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
