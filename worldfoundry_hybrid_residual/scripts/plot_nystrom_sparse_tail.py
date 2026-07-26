#!/usr/bin/env python3
"""Plot leakage-safe validation Pareto and frozen-config split metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    require_fresh_output_dir,
)


COLORS = {
    "nystrom_signed": "#0072B2",
    "landmark_linear": "#56B4E9",
    "proxy_mass_nystrom_mixture": "#D55E00",
    "proxy_mass_landmark_partition": "#E69F00",
    "nystrom_nonnegative_clamped": "#6B7280",
    "dense_mass_nystrom_mixture_diagnostic": "#009E73",
    "dense_mass_landmark_partition_diagnostic": "#2A9D8F",
}
SPLIT_COLORS = {
    "calibration": "#7A8B99",
    "validation": "#E69F00",
    "test": "#0072B2",
}
PROTOCOL_MARKERS = {
    "seed_holdout": "o",
    "prompt_holdout": "s",
    "combination_holdout": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty input CSV: {path}")
    return rows


def percent(value: str) -> float:
    parsed = float(value) * 100.0
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite percentage: {value!r}")
    return parsed


def add_threshold(ax: plt.Axes, y: float, label: str) -> None:
    ax.axhline(y, color="#B91C1C", linewidth=1.0, linestyle="--", zorder=0)
    ax.text(
        0.99,
        y,
        label,
        color="#991B1B",
        fontsize=8,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )


def main() -> None:
    args = parse_args()
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    success_path = selection_dir / "SUCCESS.json"
    manifest_path = selection_dir / "selection_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"incomplete selection artifacts in {selection_dir}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if success.get("status") != "SUCCESS":
        raise ValueError(f"selector artifact status is not SUCCESS: {success}")

    all_path = selection_dir / "all_config_split_metrics.csv"
    selected_path = selection_dir / "selected_protocol_metrics.csv"
    all_rows = read_csv(all_path)
    selected_rows = read_csv(selected_path)
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    selected_by_protocol: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_protocol[row["protocol"]].append(row)

    require_fresh_output_dir(output_dir)
    atomic_write_csv(output_dir / "validation_pareto_source.csv", validation_rows)
    atomic_write_csv(output_dir / "frozen_config_source.csv", selected_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    ax_pareto, ax_aggregate, ax_worst, ax_speed = axes.flat

    for row in validation_rows:
        method = row["method"]
        candidate_class = row["candidate_class"]
        ax_pareto.scatter(
            float(row["projected_attention_work_ratio_mean"]),
            percent(row["aggregate_output_relative_l2"]),
            s=38 if candidate_class == "deployable" else 25,
            marker=PROTOCOL_MARKERS.get(row["protocol"], "o"),
            facecolor=COLORS.get(method, "#374151")
            if candidate_class == "deployable"
            else "none",
            edgecolor=COLORS.get(method, "#374151"),
            linewidth=0.9,
            alpha=0.82,
        )
    add_threshold(ax_pareto, 1.0, "aggregate gate")
    ax_pareto.axvline(1 / 1.5, color="#B91C1C", linewidth=1.0, linestyle=":")
    ax_pareto.text(
        1 / 1.5,
        0.04,
        "1.5x work gate",
        color="#991B1B",
        fontsize=7.5,
        rotation=90,
        ha="left",
        va="bottom",
        transform=ax_pareto.get_xaxis_transform(),
    )
    ax_pareto.set_xlabel("Attention arithmetic work / dense")
    ax_pareto.set_ylabel("Validation aggregate output error (%)")
    ax_pareto.set_yscale("log")
    ax_pareto.grid(alpha=0.18, linewidth=0.6)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS[method], label=label)
        for method, label in (
            ("nystrom_signed", "Nystrom signed"),
            ("landmark_linear", "Landmark linear"),
            ("proxy_mass_nystrom_mixture", "Sparse + Nystrom"),
            ("proxy_mass_landmark_partition", "Sparse + landmark"),
        )
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#4B5563",
                label="Diagnostic (open)",
            ),
            Line2D([], [], marker="o", linestyle="none", color="#111827", label="Seed holdout"),
            Line2D([], [], marker="s", linestyle="none", color="#111827", label="Prompt holdout"),
            Line2D([], [], marker="D", linestyle="none", color="#111827", label="Pair holdout"),
        ]
    )
    ax_pareto.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        ncol=2,
        loc="lower left",
        fontsize=6.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    protocols = sorted(selected_by_protocol)
    splits = ("calibration", "validation", "test")
    width = 0.24
    x_positions = list(range(len(protocols)))
    for split_index, split in enumerate(splits):
        aggregate_values = []
        worst_values = []
        speed_values = []
        for protocol in protocols:
            matching = [
                row
                for row in selected_by_protocol[protocol]
                if row["split"] == split
            ]
            if len(matching) != 1:
                raise ValueError(f"expected one {protocol}/{split} selected row")
            aggregate_values.append(percent(matching[0]["aggregate_output_relative_l2"]))
            worst_values.append(percent(matching[0]["record_error_max"]))
            speed_values.append(
                float(matching[0]["arithmetic_speedup_upper_bound"])
            )
        positions = [x + (split_index - 1) * width for x in x_positions]
        common = {
            "width": width,
            "color": SPLIT_COLORS[split],
            "label": split,
            "edgecolor": "white",
            "linewidth": 0.5,
        }
        ax_aggregate.bar(positions, aggregate_values, **common)
        ax_worst.bar(positions, worst_values, **common)
        ax_speed.bar(positions, speed_values, **common)

    for ax in (ax_aggregate, ax_worst, ax_speed):
        ax.set_xticks(x_positions, [name.replace("_", "\n") for name in protocols])
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    add_threshold(ax_aggregate, 1.0, "aggregate gate")
    add_threshold(ax_worst, 2.0, "worst-record gate")
    add_threshold(ax_speed, 1.5, "speed gate")
    ax_aggregate.set_ylabel("Frozen-config aggregate error (%)")
    ax_worst.set_ylabel("Frozen-config worst record error (%)")
    ax_speed.set_ylabel("Arithmetic speedup upper bound (x)")
    ax_aggregate.legend(frameon=False, ncol=3, loc="upper left")

    for label, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.12,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )

    figure_path = output_dir / "nystrom_sparse_tail_pilot.png"
    pdf_path = output_dir / "nystrom_sparse_tail_pilot.pdf"
    fig.savefig(figure_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if figure_path.stat().st_size == 0 or pdf_path.stat().st_size == 0:
        raise ValueError("matplotlib produced an empty figure")

    atomic_write_json(
        output_dir / "plot_manifest.json",
        {
            "schema_version": 1,
            "selection_run_id": manifest["run_id"],
            "selection_scientific_gate": success["scientific_gate"],
            "selection_manifest_sha256": file_sha256(manifest_path),
            "all_config_metrics_sha256": file_sha256(all_path),
            "selected_metrics_sha256": file_sha256(selected_path),
            "figure_png_sha256": file_sha256(figure_path),
            "figure_pdf_sha256": file_sha256(pdf_path),
            "test_visualization_rule": (
                "all-candidate Pareto uses validation only; test appears only for the "
                "validation-frozen configuration"
            ),
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
        },
    )
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "scientific_gate": success["scientific_gate"],
            "figure": figure_path.name,
        },
    )
    print(
        f"[plot] wrote {figure_path} scientific_gate={success['scientific_gate']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
