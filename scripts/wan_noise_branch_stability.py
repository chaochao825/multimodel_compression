#!/usr/bin/env python3
"""Wan high/low noise branch stability diagnostic.

This script reuses saved Wan2.2 Q/K activation probes. It compares overlapping
layer/head records between high-noise and low-noise branches to test whether the
measured 3D cyclic component is stable across noise/timestep regimes or only a
single-run accident.
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


DEFAULT_HIGH = LOG_DIR / "wan_bccb_high_noise_delta_perturb_smallgrid_layers0_8_20_39_heads0_10_20_30.json"
DEFAULT_LOW = LOG_DIR / "wan_bccb_low_noise_delta_perturb_smallgrid_layers0_8_heads0_10_20_30.json"


def mean_float(rows: Iterable[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(x_arr.std()) == 0.0 or float(y_arr.std()) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    out = np.empty_like(arr)
    out[order] = np.arange(arr.size, dtype=np.float64)
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(ranks(x).tolist(), ranks(y).tolist())


def by_layer_head(data: dict) -> dict[tuple[int, int], dict]:
    out = {}
    for rec in data["records"]:
        out[(int(rec["layer"]), int(rec["head"]))] = rec
    return out


def metric(rec: dict, path: tuple[str, ...]) -> float:
    cur = rec
    for key in path:
        cur = cur[key]
    return float(cur)


def perturb_r2(rec: dict, name: str) -> float:
    return metric(rec, ("delta_perturbations", name, "attention", "cyclic_r2"))


def make_rows(high: dict, low: dict) -> list[dict]:
    high_map = by_layer_head(high)
    low_map = by_layer_head(low)
    rows: List[dict] = []
    for layer, head in sorted(set(high_map) & set(low_map)):
        h = high_map[(layer, head)]
        l = low_map[(layer, head)]
        h_r2 = metric(h, ("attention", "cyclic_r2"))
        l_r2 = metric(l, ("attention", "cyclic_r2"))
        h_err = metric(h, ("attention", "relative_fro_error"))
        l_err = metric(l, ("attention", "relative_fro_error"))
        h_random = perturb_r2(h, "random_coord")
        l_random = perturb_r2(l, "random_coord")
        h_axis = float(np.mean([perturb_r2(h, name) for name in ("axis_hfw", "axis_fwh", "axis_whf")]))
        l_axis = float(np.mean([perturb_r2(l, name) for name in ("axis_hfw", "axis_fwh", "axis_whf")]))
        rows.append(
            {
                "layer": layer,
                "head": head,
                "high_attention_r2": h_r2,
                "low_attention_r2": l_r2,
                "attention_r2_delta_low_minus_high": l_r2 - h_r2,
                "high_attention_error": h_err,
                "low_attention_error": l_err,
                "attention_error_delta_low_minus_high": l_err - h_err,
                "high_random_coord_r2": h_random,
                "low_random_coord_r2": l_random,
                "high_random_coord_r2_drop": h_r2 - h_random,
                "low_random_coord_r2_drop": l_r2 - l_random,
                "high_axis_mean_r2": h_axis,
                "low_axis_mean_r2": l_axis,
                "high_axis_mean_r2_drop": h_r2 - h_axis,
                "low_axis_mean_r2_drop": l_r2 - l_axis,
                "high_logit_rowmax_mean": float(h["logit_rowmax_mean"]),
                "low_logit_rowmax_mean": float(l["logit_rowmax_mean"]),
                "high_logit_rowmax_std": float(h["logit_rowmax_std"]),
                "low_logit_rowmax_std": float(l["logit_rowmax_std"]),
            }
        )
    return rows


def summarize(rows: list[dict], high: dict, low: dict) -> list[dict]:
    high_r2 = [float(row["high_attention_r2"]) for row in rows]
    low_r2 = [float(row["low_attention_r2"]) for row in rows]
    high_drop = [float(row["high_random_coord_r2_drop"]) for row in rows]
    low_drop = [float(row["low_random_coord_r2_drop"]) for row in rows]
    high_axis_drop = [float(row["high_axis_mean_r2_drop"]) for row in rows]
    low_axis_drop = [float(row["low_axis_mean_r2_drop"]) for row in rows]
    return [
        {
            "scope": "wan_high_low_overlap_layer_head",
            "overlap_records": len(rows),
            "patch_grid": "x".join(str(v) for v in high["patch_grid"]),
            "high_branch_layers": ",".join(str(v) for v in high["layers"]),
            "low_branch_layers": ",".join(str(v) for v in low["layers"]),
            "heads": ",".join(str(v) for v in low["heads"]),
            "mean_high_attention_r2": mean_float(rows, "high_attention_r2"),
            "mean_low_attention_r2": mean_float(rows, "low_attention_r2"),
            "mean_attention_r2_delta_low_minus_high": mean_float(rows, "attention_r2_delta_low_minus_high"),
            "pearson_high_low_attention_r2": pearson(high_r2, low_r2),
            "spearman_high_low_attention_r2": spearman(high_r2, low_r2),
            "mean_high_random_coord_r2": mean_float(rows, "high_random_coord_r2"),
            "mean_low_random_coord_r2": mean_float(rows, "low_random_coord_r2"),
            "mean_high_random_coord_r2_drop": mean_float(rows, "high_random_coord_r2_drop"),
            "mean_low_random_coord_r2_drop": mean_float(rows, "low_random_coord_r2_drop"),
            "pearson_high_low_random_coord_drop": pearson(high_drop, low_drop),
            "mean_high_axis_r2_drop": float(np.mean(high_axis_drop)) if high_axis_drop else None,
            "mean_low_axis_r2_drop": float(np.mean(low_axis_drop)) if low_axis_drop else None,
            "method_note": (
                "No new model forward pass. This compares overlapping layer/head "
                "records from saved high-noise and low-noise Wan2.2 small-grid Q/K probes."
            ),
        }
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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
    parser.add_argument("--high-json", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--low-json", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--output-json", type=Path, default=LOG_DIR / "wan_noise_branch_stability_20260708.json")
    parser.add_argument("--output-csv", type=Path, default=LOG_DIR / "wan_noise_branch_stability_20260708.csv")
    args = parser.parse_args()

    t0 = time.time()
    high = json.loads(args.high_json.read_text(encoding="utf-8"))
    low = json.loads(args.low_json.read_text(encoding="utf-8"))
    rows = make_rows(high, low)
    summary = summarize(rows, high, low)
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "python": platform.python_version(),
        "high_json": str(args.high_json),
        "low_json": str(args.low_json),
        "rows": rows,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary": len(summary),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
