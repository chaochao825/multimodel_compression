from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "euclidean": "#4C78A8",
    "fisher": "#E45756",
    "mixed": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def family(variant: str) -> str:
    return variant.split("_", maxsplit=1)[0]


def main() -> int:
    args = parse_args()
    variants = pd.read_csv(args.analysis_dir / "variant_summary.csv")
    tasks = pd.read_csv(args.analysis_dir / "task_summary.csv")
    variants["family"] = variants["variant"].map(family)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for name in ("euclidean", "fisher", "mixed"):
        frame = variants[variants["family"] == name].sort_values("rank")
        ax.plot(
            frame["rank"],
            frame["absolute_kl_ratio_to_r384_s4_euclidean"],
            marker="o",
            linewidth=1.8,
            color=COLORS[name],
            label=name.capitalize(),
        )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("PCA rank (support decreases from 4 to 0)")
    ax.set_ylabel("Aggregate candidate KL / r384+s4 Euclidean")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[0, 1]
    candidates = variants[variants["family"].isin(("fisher", "mixed"))]
    for name in ("fisher", "mixed"):
        frame = candidates[candidates["family"] == name].sort_values("rank")
        y = 100.0 * frame["paired_candidate_kl_reduction"].to_numpy()
        lower = 100.0 * (
            frame["paired_candidate_kl_reduction"]
            - frame["reduction_ci95_low"]
        ).to_numpy()
        upper = 100.0 * (
            frame["reduction_ci95_high"]
            - frame["paired_candidate_kl_reduction"]
        ).to_numpy()
        ax.errorbar(
            frame["rank"],
            y,
            yerr=np.vstack((lower, upper)),
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=COLORS[name],
            label=name.capitalize(),
        )
    ax.axhline(25.0, color="#333333", linestyle="--", linewidth=1, label="GO threshold")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("Paired candidate-KL reduction (%)")
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1, 0]
    offsets = {"fisher": -2.0, "mixed": 2.0}
    for name in ("fisher", "mixed"):
        frame = candidates[candidates["family"] == name].sort_values("rank")
        x = frame["rank"].to_numpy() + offsets[name]
        ax.scatter(
            x,
            frame["candidate_kl_p95_ratio"],
            s=45 + 16 * frame["positive_task_count"],
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            label=name.capitalize(),
        )
        for _, row in frame.iterrows():
            ax.annotate(
                f"{int(row['positive_task_count'])}/5",
                (row["rank"] + offsets[name], row["candidate_kl_p95_ratio"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("P95 KL ratio (bubble label: positive tasks)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1, 1]
    fisher_tasks = tasks[tasks["variant"].str.startswith("fisher_")].copy()
    allocation_order = ["r384_s4", "r402_s3", "r420_s2", "r438_s1"]
    task_order = sorted(fisher_tasks["task"].unique())
    matrix = fisher_tasks.pivot(
        index="allocation_id",
        columns="task",
        values="paired_reduction",
    ).reindex(index=allocation_order, columns=task_order)
    shown = np.clip(100.0 * matrix.to_numpy(), -100.0, 100.0)
    image = ax.imshow(shown, cmap="RdYlGn", vmin=-100.0, vmax=100.0, aspect="auto")
    for row_index in range(len(allocation_order)):
        for column_index in range(len(task_order)):
            value = 100.0 * matrix.iloc[row_index, column_index]
            label = f"{value:+.0f}%" if abs(value) < 1000 else f"{value / 100:+.1f}x"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )
    ax.set_xticks(range(len(task_order)), [value.replace("_", "\n") for value in task_order])
    ax.set_yticks(range(len(allocation_order)), allocation_order)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Fisher allocation")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Paired task reduction, clipped (%)")
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=12)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
