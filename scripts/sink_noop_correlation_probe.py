#!/usr/bin/env python3
"""Correlation diagnostics for sink/no-op/register attention mechanisms.

This probe reuses the saved head-output intervention logs. It asks whether
attention sinks behave like meaningful functional routes or like incidental
noise by correlating sink strength with entropy, effective rank, keep/drop
intervention errors, and value-subspace stress.
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


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    out = np.empty_like(values, dtype=np.float64)
    out[order] = np.arange(values.size, dtype=np.float64)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(ranks(x), ranks(y))


def mean_float(rows: Iterable[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data["rows"])
    return rows


def as_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def correlation_rows(rows: list[dict]) -> list[dict]:
    pairs = [
        ("top2_col_mass_fraction", "entropy_mean", "sink_vs_entropy"),
        ("top2_col_mass_fraction", "effective_rank_fraction", "sink_vs_effective_rank"),
        ("top2_col_mass_fraction", "row_argmax_unique_fraction", "sink_vs_argmax_diversity"),
        ("top2_col_mass_fraction", "drop_sink2_output_error", "sink_vs_drop_sink_error"),
        ("top2_col_mass_fraction", "sink2_raw_component_norm_ratio", "sink_vs_sink_component_norm"),
        ("top2_col_mass_fraction", "union_sink_local_top4_output_error", "sink_vs_keep_union_error"),
        ("top2_col_mass_fraction", "union_random_v_output_error", "sink_vs_random_v_union_error"),
        ("union_sink_local_top4_output_error", "union_random_v_output_error", "true_v_vs_random_v_union_error"),
        ("drop_union_sink_local_top4_output_error", "union_sink_local_top4_output_error", "drop_union_vs_keep_union"),
    ]
    out: list[dict] = []
    for family in sorted({str(row["family"]) for row in rows}):
        items = [row for row in rows if str(row["family"]) == family]
        for x_key, y_key, name in pairs:
            x = as_array(items, x_key)
            y = as_array(items, y_key)
            out.append(
                {
                    "family": family,
                    "pair": name,
                    "x": x_key,
                    "y": y_key,
                    "n": len(items),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                    "x_mean": float(x.mean()),
                    "y_mean": float(y.mean()),
                }
            )
    return out


def quartile_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    metric_keys = [
        "top2_col_mass_fraction",
        "entropy_mean",
        "effective_rank_fraction",
        "row_argmax_unique_fraction",
        "base_output_norm",
        "value_norm",
        "drop_sink2_output_error",
        "sink2_raw_component_norm_ratio",
        "union_sink_local_top4_output_error",
        "drop_union_sink_local_top4_output_error",
        "union_random_v_output_error",
    ]
    for family in sorted({str(row["family"]) for row in rows}):
        items = [row for row in rows if str(row["family"]) == family]
        sink = as_array(items, "top2_col_mass_fraction")
        q25, q75 = np.quantile(sink, [0.25, 0.75])
        for bucket, mask in [("low_sink_q1", sink <= q25), ("high_sink_q4", sink >= q75)]:
            subset = [row for row, keep in zip(items, mask) if bool(keep)]
            out_row = {
                "family": family,
                "bucket": bucket,
                "n": len(subset),
                "sink_threshold_q25": float(q25),
                "sink_threshold_q75": float(q75),
            }
            for key in metric_keys:
                out_row[f"mean_{key}"] = mean_float(subset, key)
            base = np.asarray([float(row["base_output_norm"]) for row in subset], dtype=np.float64)
            value = np.asarray([float(row["value_norm"]) for row in subset], dtype=np.float64)
            out_row["mean_base_output_norm_over_value_norm"] = float(np.mean(base / np.maximum(value, 1e-12)))
            out.append(out_row)
    return out


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
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            LOG_DIR / "attention_head_intervention_vit_20260707.json",
            LOG_DIR / "attention_head_intervention_qwen_20260707.json",
        ],
    )
    parser.add_argument("--output-json", type=Path, default=LOG_DIR / "sink_noop_correlation_20260708.json")
    parser.add_argument("--output-corr-csv", type=Path, default=LOG_DIR / "sink_noop_correlation_20260708.csv")
    parser.add_argument("--output-quartile-csv", type=Path, default=LOG_DIR / "sink_noop_quartiles_20260708.csv")
    args = parser.parse_args()

    t0 = time.time()
    rows = load_rows(args.inputs)
    corr = correlation_rows(rows)
    quartiles = quartile_rows(rows)
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "python": platform.python_version(),
        "inputs": [str(path) for path in args.inputs],
        "correlations": corr,
        "quartiles": quartiles,
        "summary": {
            "families": sorted({str(row["family"]) for row in rows}),
            "rows": len(rows),
            "method_note": (
                "Correlation over saved head-output intervention logs. This is not a "
                "task-loss causal test; it checks whether sink strength aligns with "
                "entropy, rank, and keep/drop output interventions."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_corr_csv, corr)
    write_csv(args.output_quartile_csv, quartiles)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "correlations": len(corr),
                "quartiles": len(quartiles),
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
