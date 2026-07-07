#!/usr/bin/env python3
"""Transfer diagnostics for oracle hybrid attention components.

The previous hybrid probe shows that an oracle

    sink/global + local-cyclic + sparse-routing

decomposition can explain representative attention matrices. This script asks a
different question: do the non-local supports/templates transfer across
examples, or are they strongly head/layer/content dependent?

It uses only saved attention matrices and saved hybrid components. No model
forward pass is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(target - approx) / denom)


def row_normalize_masked(attention: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, attention.astype(np.float64), 0.0)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def local_mask(grid_shape: Iterable[int], radius: int) -> np.ndarray:
    shape = tuple(int(v) for v in grid_shape)
    coords = np.array(list(np.ndindex(shape)), dtype=np.int64)
    grid = np.array(shape, dtype=np.int64)
    diff = (coords[None, :, :] - coords[:, None, :]) % grid
    wrapped = np.minimum(diff, grid - diff)
    return wrapped.max(axis=-1) <= int(radius)


def sink_mask(n: int, sink_cols: Iterable[int]) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    cols = [int(c) for c in sink_cols if 0 <= int(c) < n]
    if cols:
        mask[:, cols] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)


def mean_float(rows: list[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def family_from_map_id(map_id: str) -> str:
    if map_id.startswith("vit:"):
        return "vit"
    if map_id.startswith("qwen:"):
        return "qwen3vl_visual"
    return "unknown"


def balanced_config(item: dict) -> dict:
    for config in item["hybrid_configs"]:
        if config["name"] == "hybrid_balanced":
            return config
    raise KeyError(f"missing hybrid_balanced config for {item.get('key')}")


def load_examples(meta_path: Path, npz_path: Path) -> list[dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    examples = []
    for item in meta["items"]:
        key = str(item["key"])
        cfg = balanced_config(item)
        attention = arrays[f"{key}_attention"].astype(np.float64)
        sparse = arrays[f"{key}_sparse_routing"].astype(np.float64)
        hybrid = arrays[f"{key}_hybrid"].astype(np.float64)
        n = int(attention.shape[0])
        grid_shape = tuple(int(v) for v in item["grid_shape"])
        radius = int(cfg["radius"])
        support = (
            sink_mask(n, cfg["sink_cols"])
            | local_mask(grid_shape, radius)
            | (sparse > 1e-12)
        )
        examples.append(
            {
                "key": key,
                "label": item["label"],
                "map_id": item["map_id"],
                "family": family_from_map_id(str(item["map_id"])),
                "grid_shape": grid_shape,
                "n": n,
                "attention": attention,
                "hybrid": hybrid,
                "sparse_mask": sparse > 1e-12,
                "support": support,
                "sink_cols": [int(v) for v in cfg["sink_cols"]],
                "radius": radius,
                "grid_bccb_error": float(item["baseline_relative_fro_error"]["grid_cyclic_bccb"]),
                "monarch_proxy_error": float(item["baseline_relative_fro_error"]["monarch_like_mask_proxy"]),
                "target_oracle_hybrid_error": float(cfg["relative_fro_error"]),
            }
        )
    return examples


def probe(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    rows: List[dict] = []
    skipped = 0
    for target in examples:
        target_attention = target["attention"]
        target_support_approx = row_normalize_masked(target_attention, target["support"])
        for source in examples:
            if source["key"] == target["key"]:
                continue
            if source["n"] != target["n"] or source["grid_shape"] != target["grid_shape"]:
                skipped += 1
                continue
            transfer_support_approx = row_normalize_masked(target_attention, source["support"])
            source_hybrid_template = source["hybrid"]
            source_attention_template = row_normalize_masked(source["attention"], np.ones_like(source["attention"], dtype=bool))
            source_sink = sink_mask(source["n"], source["sink_cols"])
            target_sink = sink_mask(target["n"], target["sink_cols"])
            rows.append(
                {
                    "target_key": target["key"],
                    "target_label": target["label"],
                    "target_map_id": target["map_id"],
                    "target_family": target["family"],
                    "source_key": source["key"],
                    "source_label": source["label"],
                    "source_map_id": source["map_id"],
                    "source_family": source["family"],
                    "same_family": source["family"] == target["family"],
                    "grid_shape": "x".join(str(v) for v in target["grid_shape"]),
                    "n": target["n"],
                    "target_grid_bccb_error": target["grid_bccb_error"],
                    "target_monarch_proxy_error": target["monarch_proxy_error"],
                    "target_oracle_hybrid_error": target["target_oracle_hybrid_error"],
                    "target_support_mask_error": relative_error(target_attention, target_support_approx),
                    "source_support_transfer_error": relative_error(target_attention, transfer_support_approx),
                    "source_hybrid_template_error": relative_error(target_attention, source_hybrid_template),
                    "source_attention_template_error": relative_error(target_attention, source_attention_template),
                    "support_jaccard": jaccard(source["support"], target["support"]),
                    "sink_jaccard": jaccard(source_sink, target_sink),
                    "sparse_route_jaccard": jaccard(source["sparse_mask"], target["sparse_mask"]),
                    "target_support_density": float(target["support"].mean()),
                    "source_support_density": float(source["support"].mean()),
                }
            )

    summary_rows = []
    groups = {
        "all_same_grid_pairs": rows,
        "same_family_pairs": [row for row in rows if row["same_family"]],
        "cross_family_pairs": [row for row in rows if not row["same_family"]],
    }
    for name, items in groups.items():
        if not items:
            continue
        summary_rows.append(
            {
                "scope": name,
                "pairs": len(items),
                "skipped_incompatible_pairs": skipped,
                "mean_target_grid_bccb_error": mean_float(items, "target_grid_bccb_error"),
                "mean_target_monarch_proxy_error": mean_float(items, "target_monarch_proxy_error"),
                "mean_target_oracle_hybrid_error": mean_float(items, "target_oracle_hybrid_error"),
                "mean_target_support_mask_error": mean_float(items, "target_support_mask_error"),
                "mean_source_support_transfer_error": mean_float(items, "source_support_transfer_error"),
                "mean_source_hybrid_template_error": mean_float(items, "source_hybrid_template_error"),
                "mean_source_attention_template_error": mean_float(items, "source_attention_template_error"),
                "mean_support_jaccard": mean_float(items, "support_jaccard"),
                "mean_sink_jaccard": mean_float(items, "sink_jaccard"),
                "mean_sparse_route_jaccard": mean_float(items, "sparse_route_jaccard"),
                "method_note": (
                    "Transfer probe over saved representative matrices. It tests whether "
                    "hybrid supports/templates selected on a source map explain a target map; "
                    "target_support_mask_error is a semi-oracle support-only baseline."
                ),
            }
        )
    return rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
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
    parser.add_argument("--output-json", default=str(LOG_DIR / "hybrid_transfer_probe_20260708.json"))
    parser.add_argument("--output-csv", default=str(LOG_DIR / "hybrid_transfer_probe_20260708.csv"))
    args = parser.parse_args()

    t0 = time.time()
    examples = load_examples(Path(args.input_json), Path(args.input_npz))
    rows, summary = probe(examples)
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "python": platform.python_version(),
        "input_json": args.input_json,
        "input_npz": args.input_npz,
        "examples": [
            {
                "key": item["key"],
                "label": item["label"],
                "map_id": item["map_id"],
                "family": item["family"],
                "grid_shape": item["grid_shape"],
                "n": item["n"],
            }
            for item in examples
        ],
        "rows": rows,
        "summary": summary,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, rows)
    print(
        json.dumps(
            {
                "examples": len(examples),
                "rows": len(rows),
                "summary": len(summary),
                "output_json": str(output_json),
                "output_csv": str(output_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
