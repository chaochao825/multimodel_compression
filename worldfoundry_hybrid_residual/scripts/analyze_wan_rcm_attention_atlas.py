#!/usr/bin/env python3
"""Summarize and plot the frozen EXP-054 rCM attention atlas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


METRICS = ("aggregate", "worst_head", "worst_query_tile")
THRESHOLDS = {
    "calibration": {
        "aggregate": 0.008,
        "worst_head": 0.016,
        "worst_query_tile": 0.016,
    },
    "evaluation": {
        "aggregate": 0.010,
        "worst_head": 0.020,
        "worst_query_tile": 0.020,
    },
}
METRIC_LABELS = {
    "aggregate": "Aggregate",
    "worst_head": "Worst head",
    "worst_query_tile": "Worst 64-query tile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            record: dict[str, object] = {
                "identity": raw["identity"],
                "split": raw["split"],
                "step": int(raw["step"]),
                "layer": int(raw["layer"]),
            }
            for metric in METRICS:
                value = float(raw[metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {metric} in {record}")
                record[metric] = value
            records.append(record)
    if len(records) != 960:
        raise ValueError(f"expected 960 EXP-054 records, got {len(records)}")
    return records


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_cells(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["step"]), int(record["layer"]), str(record["split"]))].append(record)

    rows: list[dict[str, object]] = []
    for step in range(4):
        for layer in range(30):
            row: dict[str, object] = {"step": step, "layer": layer}
            for split in ("calibration", "evaluation"):
                split_rows = grouped[(step, layer, split)]
                if len(split_rows) != 4:
                    raise ValueError(
                        f"expected four {split} identities for step={step}, layer={layer}"
                    )
                ratios: list[float] = []
                for metric in METRICS:
                    values = [float(item[metric]) for item in split_rows]
                    row[f"{split}_{metric}_mean"] = sum(values) / len(values)
                    row[f"{split}_{metric}_max"] = max(values)
                    ratios.append(max(values) / THRESHOLDS[split][metric])
                row[f"{split}_threshold_ratio"] = max(ratios)
                row[f"{split}_passes"] = max(ratios) <= 1.0
            rows.append(row)
    return rows


def summarize_run(
    records: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "record_count": len(records),
        "cell_count": len(cell_rows),
        "splits": {},
    }
    split_summary: dict[str, object] = {}
    for split in ("calibration", "evaluation"):
        metrics: dict[str, object] = {}
        split_records = [record for record in records if record["split"] == split]
        for metric in METRICS:
            values = [float(record[metric]) for record in split_records]
            metrics[metric] = {
                "mean": sum(values) / len(values),
                "median": _quantile(values, 0.5),
                "p95": _quantile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
        passing_cells = [row for row in cell_rows if bool(row[f"{split}_passes"])]
        split_summary[split] = {
            "metrics": metrics,
            "metric_passing_cell_count": {
                metric: sum(
                    float(row[f"{split}_{metric}_max"])
                    <= THRESHOLDS[split][metric]
                    for row in cell_rows
                )
                for metric in METRICS
            },
            "passing_cell_count": len(passing_cells),
            "passing_cell_fraction": len(passing_cells) / len(cell_rows),
        }
    summary["splits"] = split_summary

    calibration_scores = np.asarray(
        [float(row["calibration_threshold_ratio"]) for row in cell_rows]
    )
    evaluation_scores = np.asarray(
        [float(row["evaluation_threshold_ratio"]) for row in cell_rows]
    )
    summary["cell_score_pearson"] = float(
        np.corrcoef(calibration_scores, evaluation_scores)[0, 1]
    )
    summary["best_calibration_cells"] = [
        {
            "step": int(row["step"]),
            "layer": int(row["layer"]),
            "threshold_ratio": float(row["calibration_threshold_ratio"]),
            "aggregate_max": float(row["calibration_aggregate_max"]),
            "worst_head_max": float(row["calibration_worst_head_max"]),
            "worst_query_tile_max": float(row["calibration_worst_query_tile_max"]),
        }
        for row in sorted(
            cell_rows, key=lambda item: float(item["calibration_threshold_ratio"])
        )[:10]
    ]
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_layers(cell_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in range(30):
        layer_cells = [row for row in cell_rows if int(row["layer"]) == layer]
        row: dict[str, object] = {"layer": layer}
        for split in ("calibration", "evaluation"):
            scores = [float(item[f"{split}_threshold_ratio"]) for item in layer_cells]
            row[f"{split}_threshold_ratio_mean"] = sum(scores) / len(scores)
            row[f"{split}_threshold_ratio_max"] = max(scores)
            row[f"{split}_passing_steps"] = sum(
                bool(item[f"{split}_passes"]) for item in layer_cells
            )
        rows.append(row)
    return rows


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_heatmaps(cell_rows: list[dict[str, object]], output_dir: Path) -> None:
    matrices: dict[tuple[str, str], np.ndarray] = {}
    all_ratios: list[float] = []
    for split in ("calibration", "evaluation"):
        for metric in METRICS:
            matrix = np.zeros((4, 30), dtype=np.float64)
            for row in cell_rows:
                step = int(row["step"])
                layer = int(row["layer"])
                value = float(row[f"{split}_{metric}_max"])
                matrix[step, layer] = value / THRESHOLDS[split][metric]
            matrices[(split, metric)] = matrix
            all_ratios.extend(matrix.ravel().tolist())

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 5.8), constrained_layout=True)
    upper = max(2.0, _quantile(all_ratios, 0.98))
    image = None
    for row_index, split in enumerate(("calibration", "evaluation")):
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                matrices[(split, metric)],
                aspect="auto",
                cmap="viridis",
                norm=LogNorm(vmin=0.5, vmax=upper),
                interpolation="nearest",
            )
            axis.contour(
                matrices[(split, metric)],
                levels=[1.0],
                colors=["#e66101"],
                linewidths=1.2,
            )
            axis.set_xlabel("Wan layer")
            axis.set_ylabel(f"{split.capitalize()} rCM step")
            axis.set_yticks(range(4))
            axis.text(
                0.02,
                0.94,
                METRIC_LABELS[metric],
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
            )
    if image is None:
        raise RuntimeError("heatmap image was not created")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.92, pad=0.015)
    colorbar.set_label("Maximum error / registered threshold (1 = pass boundary)")
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cell_threshold_ratio_heatmap.{suffix}", dpi=300)
    plt.close(figure)


def plot_distributions(records: list[dict[str, object]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    colors = {"calibration": "#0072B2", "evaluation": "#D55E00"}
    for axis, metric in zip(axes, METRICS, strict=True):
        for split in ("calibration", "evaluation"):
            values = np.sort(
                np.asarray(
                    [
                        100.0 * float(record[metric])
                        for record in records
                        if record["split"] == split
                    ],
                    dtype=np.float64,
                )
            )
            cumulative = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values, cumulative, color=colors[split], label=split.capitalize())
            axis.axvline(
                100.0 * THRESHOLDS[split][metric],
                color=colors[split],
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )
        axis.set_xscale("log")
        axis.set_xlabel(f"{METRIC_LABELS[metric]} relative L2 (%)")
        axis.set_ylabel("Empirical cumulative probability")
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
        _style_axis(axis)
    axes[0].legend(frameon=False, loc="lower right")
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"error_distributions.{suffix}", dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    records = load_records(args.records.resolve())
    cell_rows = summarize_cells(records)
    summary = summarize_run(records, cell_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cell_summary.csv", cell_rows)
    write_csv(output_dir / "layer_summary.csv", summarize_layers(cell_rows))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_heatmaps(cell_rows, output_dir)
    plot_distributions(records, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
