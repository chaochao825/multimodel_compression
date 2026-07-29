#!/usr/bin/env python3
"""Generate publication-style plots for the content-tail diagnostic."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "capacity": "#0072B2",
    "frozen": "#D55E00",
    "proxy": "#009E73",
    "baseline": "#6F6F6F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def plot_rank_frontier(directory: Path) -> None:
    data = rows(directory / "rank_frontier.csv")
    rank = np.array([int(row["rank"]) for row in data])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharex=True)
    series = (
        (
            "Transductive capacity",
            "transductive_per_tile_rank16_error",
            "transductive_worst_tile",
            COLORS["capacity"],
            "o",
        ),
        (
            "Frozen tail + support oracle",
            "frozen_oracle_per_tile_rank16_error",
            "frozen_oracle_worst_tile",
            COLORS["frozen"],
            "s",
        ),
        (
            "Frozen train-free proxy",
            "frozen_proxy_per_tile_rank16_error",
            "frozen_proxy_worst_tile",
            COLORS["proxy"],
            "^",
        ),
    )
    for label, aggregate, worst, color, marker in series:
        axes[0].plot(
            rank,
            100 * np.array([float(row[aggregate]) for row in data]),
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
        )
        axes[1].plot(
            rank,
            100 * np.array([float(row[worst]) for row in data]),
            color=color,
            marker=marker,
            linewidth=1.8,
        )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Oracle gate")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Per-tile aggregate rel-L2 (%)")
    axes[1].set_ylabel("Worst-tile rel-L2 (%)")
    for axis in axes:
        axis.set_xlabel("Linear feature rank")
        axis.set_xticks(rank)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, directory / "rank_sweep_quality")


def plot_router(directory: Path) -> None:
    data = rows(directory / "router_comparison_rank64.csv")
    label_map = {
        "oracle_trajectory_width_family": "Oracle family",
        "oracle_trajectory_width_fixed64": "Oracle fixed",
        "oracle_trajectory_width_semantic64": "Oracle semantic",
        "oracle_trajectory_width_value64": "Oracle value-aware",
        "proxy_fixed64_qk": "Proxy fixed",
        "proxy_svg2_style_semantic64": "Proxy semantic",
        "proxy_value_aware_semantic64": "Proxy value-aware",
    }
    labels = [label_map[row["route"]] for row in data]
    aggregate = 100 * np.array([float(row["per_tile_rank16_error"]) for row in data])
    worst = 100 * np.array([float(row["worst_tile"]) for row in data])
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.bar(x - 0.18, aggregate, width=0.36, color="#56B4E9", label="Aggregate")
    ax.bar(x + 0.18, worst, width=0.36, color="#E69F00", label="Worst tile")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="Aggregate gate")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Worst gate")
    ax.set_ylabel("Post-rank-16 rel-L2 (%)")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    save(fig, directory / "router_comparison_rank64")


def plot_heads(directory: Path) -> None:
    data = rows(directory / "head_errors_rank64.csv")
    routes = sorted({row["route"] for row in data})
    heads = sorted({int(row["head"]) for row in data})
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    width = 0.36
    for offset, route, color in zip((-0.18, 0.18), routes, (COLORS["frozen"], COLORS["proxy"])):
        by_head = {int(row["head"]): row for row in data if row["route"] == route}
        values = [100 * float(by_head[head]["per_tile_rank16_error"]) for head in heads]
        label = "Support-family oracle" if route.startswith("oracle_") else "Selected train-free proxy"
        ax.bar(np.arange(len(heads)) + offset, values, width=width, color=color, label=label)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="Deployment worst gate")
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Per-tile aggregate rel-L2 (%)")
    ax.set_xticks(np.arange(len(heads)), heads)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    save(fig, directory / "head_error_rank64")


def plot_baseline(directory: Path) -> None:
    data = rows(directory / "baseline_comparison.csv")
    labels = [
        "Old support\noracle",
        "Rank-64\ncapacity",
        "Frozen tail +\nsupport oracle",
        "Frozen tail +\nproxy",
    ]
    aggregate = 100 * np.array([float(row["per_tile_rank16_error"]) for row in data])
    worst = 100 * np.array([float(row["worst_tile"]) for row in data])
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(x - 0.18, aggregate, width=0.36, color="#56B4E9", label="Aggregate")
    ax.bar(x + 0.18, worst, width=0.36, color="#CC79A7", label="Worst tile")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="Aggregate gate")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Worst gate")
    ax.set_ylabel("Granularity-matched rel-L2 (%)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, directory / "baseline_comparison")


def plot_same_test_decomposition(directory: Path) -> None:
    data = rows(directory / "same_test_error_decomposition_rank64.csv")
    if len(data) != 3:
        raise ValueError("expected three same-test error stages")
    labels = ["Capacity\n(all-record fit)", "Frozen\nfeature map", "QKV-only\nproxy"]
    errors = 100 * np.array(
        [float(row["per_tile_rank16_output_relative_l2"]) for row in data]
    )
    shares = 100 * np.array(
        [float(row["incremental_squared_error_fraction_of_proxy"]) for row in data]
    )
    colors = [COLORS["capacity"], COLORS["frozen"], COLORS["proxy"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    bars = axes[0].bar(np.arange(3), errors, color=colors)
    axes[0].bar_label(bars, labels=[f"{value:.3f}%" for value in errors], padding=3)
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Same-test per-tile rel-L2 (%)")
    for value, label in ((0.5, "Oracle gate"), (1.0, "Deployment gate")):
        axes[0].text(
            0.98,
            value + 0.015,
            label,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )

    left = 0.0
    for label, value, color in zip(("Capacity floor", "Transfer", "Routing"), shares, colors):
        axes[1].barh([0], [value], left=left, color=color, label=label)
        if value >= 4:
            axes[1].text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", color="white")
        else:
            axes[1].annotate(
                f"{value:.1f}%",
                xy=(left + value / 2, 0),
                xytext=(left + value / 2, -0.5),
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
            )
        left += value
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-0.65, 0.65)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Fraction of final proxy squared error (%)")
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    fig.tight_layout()
    save(fig, directory / "same_test_error_decomposition_rank64")


def main() -> None:
    args = parse_args()
    style()
    plot_rank_frontier(args.analysis_dir)
    plot_router(args.analysis_dir)
    plot_heads(args.analysis_dir)
    plot_baseline(args.analysis_dir)
    plot_same_test_decomposition(args.analysis_dir)
    print(f"[content-tail-plots] wrote {args.analysis_dir}")


if __name__ == "__main__":
    main()
