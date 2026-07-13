#!/usr/bin/env python3
"""Plot the component orthogonality and factorial-ablation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "remote_logs" / "component_orthogonality_ablation_20260711.json"
COMPONENT_LABELS = ["Sink", "Global", "Local", "Sparse"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 10.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", va="top")


def plot_gram(ax: plt.Axes, payload: dict) -> None:
    gram = np.asarray(payload["aggregate"]["average_normalized_gram"], dtype=float)
    image = ax.imshow(gram, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(4), COMPONENT_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(4), COMPONENT_LABELS)
    ax.set_xlabel("Component")
    ax.set_ylabel("Component")
    for row in range(4):
        for col in range(4):
            color = "white" if gram[row, col] > 0.55 else "black"
            ax.text(col, row, f"{gram[row, col]:.2f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Frobenius cosine")
    panel_label(ax, "(a)")


def plot_order_sensitivity(ax: plt.Axes, payload: dict) -> None:
    examples = payload["examples"]
    x = np.arange(len(examples))
    best = np.asarray([item["best_order_error"] for item in examples], dtype=float)
    current = np.asarray([item["current_order_error"] for item in examples], dtype=float)
    worst = np.asarray([item["worst_order_error"] for item in examples], dtype=float)
    ax.vlines(x, best, worst, color="#999999", linewidth=2.0, zorder=1)
    ax.scatter(x, best, marker="v", color=COLORS[2], label="Best of 24", zorder=3)
    ax.scatter(x, current, marker="o", color=COLORS[0], label="Current order", zorder=3)
    ax.scatter(x, worst, marker="^", color=COLORS[1], label="Worst of 24", zorder=3)
    ax.set_xticks(x, [item["label"].replace(" ", "\n", 1) for item in examples])
    ax.set_ylabel("Relative Frobenius error")
    ax.set_xlabel("Representative attention map")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.7,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    panel_label(ax, "(b)")


def plot_method_error(ax: plt.Axes, payload: dict) -> None:
    examples = payload["examples"]
    names = ["Grid BCCB\n(nearest)", "Mask proxy\n(A-aware)", "Hybrid\n(A-aware)", "Best of 24\n(per-map)"]
    keys = ["grid_bccb_error", "monarch_proxy_error", "full_hybrid_error", "best_order_error"]
    values = np.asarray([[item[key] for item in examples] for key in keys], dtype=float)
    x = np.arange(len(names))
    means = values.mean(axis=1)
    bars = ax.bar(x, means, color=COLORS, width=0.68, alpha=0.78)
    offsets = np.linspace(-0.10, 0.10, values.shape[1])
    for idx in range(values.shape[0]):
        ax.scatter(
            np.full(values.shape[1], x[idx]) + offsets,
            values[idx],
            s=18,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Relative Frobenius error")
    ax.set_xlabel("Matrix-level approximation")
    ax.set_ylim(0.0, max(1.02, float(values.max()) * 1.12))
    ax.text(
        0.98,
        0.97,
        "Oracle diagnostics; unequal representation budgets",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    panel_label(ax, "(c)")


def plot_interactions(ax: plt.Axes, payload: dict) -> None:
    pair_order = [
        ("sink", "global_svd"),
        ("sink", "local_cyclic"),
        ("sink", "sparse_routing"),
        ("global_svd", "local_cyclic"),
        ("global_svd", "sparse_routing"),
        ("local_cyclic", "sparse_routing"),
    ]
    labels = ["S-G", "S-L", "S-R", "G-L", "G-R", "L-R"]
    rows = payload["factorial_interactions"]
    mean_values = []
    max_values = []
    for pair in pair_order:
        selected = [row for row in rows if (row["left"], row["right"]) == pair]
        by_example = {}
        for row in selected:
            by_example.setdefault(row["key"], []).append(abs(row["interaction"]))
        per_example_mean = [np.mean(values) for values in by_example.values()]
        per_example_max = [np.max(values) for values in by_example.values()]
        mean_values.append(np.mean(per_example_mean))
        max_values.append(np.mean(per_example_max))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, mean_values, width, color=COLORS[0], label="Mean over all contexts")
    ax.bar(x + width / 2, max_values, width, color=COLORS[1], label="Max over contexts")
    threshold = float(payload["thresholds"]["near_factorial_interaction_threshold"])
    ax.axhline(threshold, color="black", linewidth=1.0, linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean |factorial interaction|")
    ax.set_xlabel("Component pair (S/G/L/R)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.7,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    panel_label(ax, "(d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--name", default="fig21_component_orthogonality_ablation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    plot_gram(axes[0, 0], payload)
    plot_order_sensitivity(axes[0, 1], payload)
    plot_method_error(axes[1, 0], payload)
    plot_interactions(axes[1, 1], payload)
    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.output_dir / f"{args.name}.{suffix}"
        fig.savefig(path, format=suffix, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
