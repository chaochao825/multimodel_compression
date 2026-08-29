from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#7F7F7F",
    "no_go": "#4C78A8",
    "capacity_only": "#59A14F",
    "harm": "#E15759",
    "correct": "#F28E2B",
}

LABELS = {
    "source_r456": "Source",
    "target_mean_source_r456": "Mean",
    "residual_swap_r16": "Swap-16",
    "residual_swap_r32": "Swap-32",
    "residual_swap_r64": "Swap-64",
    "residual_swap_r96": "Swap-96",
    "residual_swap_r128": "Swap-128",
    "target_pca_r456": "Target PCA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--fit-summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_color(decision: str) -> str:
    if decision == "BASELINE":
        return COLORS["baseline"]
    if decision == "CAPACITY_ONLY":
        return COLORS["capacity_only"]
    return COLORS["no_go"]


def main() -> int:
    args = parse_args()
    rows = read_csv(args.analysis_dir / "candidate_summary.csv")
    fit = json.loads(args.fit_summary.read_text(encoding="utf-8"))
    fit_candidates = fit["candidates"]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_rows = []
    for row in rows:
        candidate = row["candidate"]
        fit_row = fit_candidates[candidate]
        plot_rows.append(
            {
                "candidate": candidate,
                "label": LABELS[candidate],
                "decision": row["decision"],
                "adaptation_rank": int(row["adaptation_rank"]),
                "kl_ratio": float(row["kl_ratio_to_source"]),
                "p95_ratio": float(row["p95_ratio_to_source"]),
                "l2_ratio": float(row["feature_l2_ratio_to_source"]),
                "prediction_mismatches": int(row["prediction_mismatches"]),
                "harmful_flips": int(row["harmful_flips"]),
                "candidate_correct": int(row["candidate_correct"]),
                "source_overlap": min(
                    1.0, float(fit_row["source_subspace_overlap"])
                ),
                "calibration_l2": float(
                    fit_row["calibration_feature_relative_l2"]
                ),
                "selection_l2": float(row["feature_relative_l2_mean"]),
            }
        )

    data_path = args.output_prefix.with_name(f"{args.output_prefix.name}_data.csv")
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    positions = np.arange(len(plot_rows))
    labels = [row["label"] for row in plot_rows]
    colors = [candidate_color(row["decision"]) for row in plot_rows]

    axis = axes[0, 0]
    axis.plot(
        positions,
        [row["kl_ratio"] / 0.70 for row in plot_rows],
        marker="o",
        linewidth=1.7,
        label="Mean KL / 0.70",
        color="#4C78A8",
    )
    axis.plot(
        positions,
        [row["p95_ratio"] / 0.80 for row in plot_rows],
        marker="s",
        linewidth=1.7,
        label="P95 KL / 0.80",
        color="#E15759",
    )
    axis.plot(
        positions,
        [row["l2_ratio"] / 0.90 for row in plot_rows],
        marker="^",
        linewidth=1.7,
        label="Feature L2 / 0.90",
        color="#59A14F",
    )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.fill_between(
        [-0.4, len(plot_rows) - 0.6],
        [0.0, 0.0],
        [1.0, 1.0],
        color="#DDEBD8",
        alpha=0.35,
        zorder=-1,
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Metric / frozen capacity threshold")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.text(-0.11, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[0, 1]
    width = 0.25
    source_correct = plot_rows[0]["candidate_correct"]
    axis.bar(
        positions - width,
        [row["prediction_mismatches"] for row in plot_rows],
        width,
        color="#4C78A8",
        label="Mismatch",
    )
    axis.bar(
        positions,
        [row["harmful_flips"] for row in plot_rows],
        width,
        color=COLORS["harm"],
        label="Harmful",
    )
    axis.bar(
        positions + width,
        [row["candidate_correct"] - source_correct for row in plot_rows],
        width,
        color=COLORS["correct"],
        label="Correct delta",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.axhline(4.0, color="#4C78A8", linestyle="--", linewidth=0.9)
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Selection outcomes (count / 180)")
    axis.legend(frameon=False, fontsize=8, ncol=3)
    axis.text(-0.11, 1.04, "b", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 0]
    for row, color in zip(plot_rows, colors, strict=True):
        axis.scatter(
            row["kl_ratio"],
            row["prediction_mismatches"],
            s=65 + 45 * row["harmful_flips"],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        if row["candidate"] in {
            "source_r456",
            "residual_swap_r16",
            "residual_swap_r128",
            "target_pca_r456",
        }:
            axis.annotate(
                row["label"],
                (row["kl_ratio"], row["prediction_mismatches"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axvline(0.70, color="#333333", linestyle="--", linewidth=0.9)
    axis.axhline(4.0, color="#333333", linestyle="--", linewidth=0.9)
    axis.fill_betweenx([0.0, 4.0], 0.45, 0.70, color="#DDEBD8", alpha=0.35)
    axis.set_xlim(0.45, 1.08)
    axis.set_ylim(-0.3, 8.8)
    axis.set_xlabel("Mean KL ratio to source codec")
    axis.set_ylabel("Prediction mismatches (count / 180)")
    axis.text(-0.11, 1.04, "c", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 1]
    for row, color in zip(plot_rows, colors, strict=True):
        axis.scatter(
            row["source_overlap"],
            row["kl_ratio"],
            s=65,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        if row["candidate"] in {
            "source_r456",
            "residual_swap_r16",
            "residual_swap_r128",
            "target_pca_r456",
        }:
            axis.annotate(
                row["label"],
                (row["source_overlap"], row["kl_ratio"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axhline(0.70, color="#333333", linestyle="--", linewidth=0.9)
    axis.set_xlim(0.77, 1.015)
    axis.set_xlabel("Overlap with source PCA subspace")
    axis.set_ylabel("Mean KL ratio to source codec")
    axis.text(-0.11, 1.04, "d", transform=axis.transAxes, fontweight="bold", fontsize=12)

    for axis in axes.flat:
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.7)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    svg_path = args.output_prefix.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
