#!/usr/bin/env python3
"""Plot source-bound spiked-covariance diagnostics for Wan runtime defects."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ("#16697a", "#d08c32", "#4f772d", "#b33a3a", "#64727d")


def short_series(operator: str, normalization: str) -> str:
    operator_label = {
        "C": "C",
        "C_FORECAST_0.5": "F0.50",
        "C_FORECAST_0.75": "F0.75",
        "C_FORECAST_1": "F1.00",
        "Q": "Q",
    }.get(operator, operator)
    normalization_label = "std" if normalization == "channel_standardized" else "raw"
    return f"{operator_label} {normalization_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--eigenvalues", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-eigenvalues", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_eigenvalues <= 0:
        raise ValueError("--top-eigenvalues must be positive")

    summary = pd.read_csv(args.summary)
    eigenvalues = pd.read_csv(args.eigenvalues)
    summary = summary[summary["group"] == "all_blocks"].copy()
    eigenvalues = eigenvalues[eigenvalues["group"] == "all_blocks"].copy()
    if summary.empty or eigenvalues.empty:
        raise ValueError("RMT inputs contain no all_blocks rows")

    summary["series"] = [
        short_series(operator, normalization)
        for operator, normalization in zip(summary["operator"], summary["normalization"])
    ]
    operator_colors = {
        operator: COLORS[index % len(COLORS)]
        for index, operator in enumerate(sorted(summary["operator"].unique()))
    }
    eigenvalues = eigenvalues.merge(
        summary[["operator", "normalization", "series", "spike_threshold"]],
        on=["operator", "normalization"],
        how="inner",
        validate="many_to_one",
    )
    eigenvalues["eigenvalue_over_threshold"] = (
        eigenvalues["eigenvalue"] / eigenvalues["spike_threshold"]
    )
    eigenvalues = eigenvalues[eigenvalues["index"] <= args.top_eigenvalues]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "defect_rmt_plot_summary.csv", index=False)
    eigenvalues.to_csv(out_dir / "defect_rmt_plot_eigenvalues.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8), constrained_layout=True)

    for index, (series, frame) in enumerate(eigenvalues.groupby("series", sort=True)):
        operator = str(frame["operator"].iloc[0])
        normalization = str(frame["normalization"].iloc[0])
        axes[0, 0].plot(
            frame["index"],
            frame["eigenvalue_over_threshold"],
            color=operator_colors[operator],
            linestyle="--" if normalization == "channel_standardized" else "-",
            linewidth=1.8,
            label=series,
        )
    axes[0, 0].axhline(1.0, color="#17222b", linestyle="--", linewidth=1.1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Eigenvalue rank")
    axes[0, 0].set_ylabel("Eigenvalue / spike threshold")
    axes[0, 0].set_title("A  Runtime-defect eigenspectrum")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    positions = np.arange(len(summary))
    labels = summary["series"]
    axes[0, 1].bar(
        positions,
        summary["spike_energy_ratio"],
        color=[operator_colors[operator] for operator in summary["operator"]],
    )
    axes[0, 1].set_xticks(positions, labels, rotation=24, ha="right")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("Energy above conservative edge")
    axes[0, 1].set_title("B  Candidate energy above MP/null edge")

    width = 0.24
    for offset, rank in enumerate((8, 16, 32)):
        axes[1, 0].bar(
            positions + (offset - 1) * width,
            summary[f"energy_rank_{rank}"],
            width=width,
            label=f"rank {rank}",
            color=COLORS[offset],
        )
    axes[1, 0].set_xticks(positions, labels, rotation=24, ha="right")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Cumulative covariance energy")
    axes[1, 0].set_title("C  Low-rank energy is necessary, not sufficient")
    axes[1, 0].legend(frameon=False)

    marker_sizes = 35.0 + 1.8 * summary["spike_count"].clip(lower=0)
    for index, (_, row) in enumerate(summary.iterrows()):
        axes[1, 1].scatter(
            row["spike_energy_ratio"],
            row["subspace_overlap_mean"],
            s=marker_sizes.iloc[index],
            color=operator_colors[str(row["operator"])],
            marker="s" if row["normalization"] == "channel_standardized" else "o",
            edgecolor="white",
            linewidth=0.8,
            label=row["series"],
        )
    axes[1, 1].axhline(0.8, color="#17222b", linestyle="--", linewidth=1.1)
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_xlabel("Spike energy ratio")
    axes[1, 1].set_ylabel("Cross-run top-r subspace overlap")
    axes[1, 1].set_title("D  Stability gate for fused correction")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    fig.suptitle(
        "Wan activation-defect RMT probe: significance, compressibility, stability",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_dir / "defect_rmt_dashboard.png", dpi=240)
    fig.savefig(out_dir / "defect_rmt_dashboard.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
