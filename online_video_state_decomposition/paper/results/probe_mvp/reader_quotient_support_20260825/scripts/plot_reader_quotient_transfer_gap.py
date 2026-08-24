from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-analysis", type=Path, required=True)
    parser.add_argument("--static-analysis", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {row["variant"]: row for row in payload["variant_summary"]}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    oracle = load_summary(args.oracle_analysis)
    static = load_summary(args.static_analysis)
    methods = (
        ("Oracle Fisher", oracle["fisher_s4"], "#0072B2"),
        ("Oracle mixed", oracle["mixed_s4"], "#009E73"),
        ("Static channel", static["channel_s4"], "#CC79A7"),
        ("Static Fisher", static["static_fisher_s4"], "#D55E00"),
        ("Static mixed", static["mixed_static_s4"], "#E69F00"),
    )
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"}
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))

    x = np.arange(len(methods))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [100.0 * float(row["aggregate_candidate_kl_reduction"]) for _, row, _ in methods],
        width,
        color=[color for _, _, color in methods],
        label="Candidate KL",
    )
    axes[0, 0].bar(
        x + width / 2,
        [100.0 * float(row["aggregate_vocabulary_kl_reduction"]) for _, row, _ in methods],
        width,
        facecolor="none",
        edgecolor=[color for _, _, color in methods],
        linewidth=1.8,
        label="Vocabulary KL",
    )
    axes[0, 0].axhline(0, color="#555555", linewidth=0.9)
    axes[0, 0].axhline(25, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xticks(x, [name for name, _, _ in methods], rotation=18, ha="right")
    axes[0, 0].set_ylabel("Aggregate KL reduction (%)")
    axes[0, 0].set_title("A  Oracle gain does not transfer statically")
    axes[0, 0].legend(frameon=False)

    tail_values = [float(row["candidate_kl_p95_ratio"]) for _, row, _ in methods]
    axes[0, 1].bar(x, tail_values, color=[color for _, _, color in methods])
    axes[0, 1].axhline(1, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xticks(x, [name for name, _, _ in methods], rotation=18, ha="right")
    axes[0, 1].set_ylabel("P95 KL / Euclidean P95")
    axes[0, 1].set_title("B  Tail-risk gate")

    oracle_tasks = load_csv(args.oracle_analysis / "task_summary.csv")
    static_tasks = load_csv(args.static_analysis / "task_summary.csv")
    tasks = sorted(
        {row["task"] for row in oracle_tasks if row["variant"] == "mixed_s4"}
    )
    oracle_by_task = {
        row["task"]: 100.0 * float(row["aggregate_reduction"])
        for row in oracle_tasks
        if row["variant"] == "mixed_s4"
    }
    static_by_task = {
        row["task"]: 100.0 * float(row["aggregate_reduction"])
        for row in static_tasks
        if row["variant"] == "mixed_static_s4"
    }
    x_task = np.arange(len(tasks))
    axes[1, 0].bar(
        x_task - width / 2,
        [oracle_by_task[task] for task in tasks],
        width,
        color="#009E73",
        label="Transductive mixed",
    )
    axes[1, 0].bar(
        x_task + width / 2,
        [static_by_task[task] for task in tasks],
        width,
        color="#E69F00",
        label="Static-prior mixed",
    )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.9)
    axes[1, 0].axhline(25, color="#555555", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xticks(
        x_task,
        [task.replace("_", " ") for task in tasks],
        rotation=20,
        ha="right",
    )
    axes[1, 0].set_ylabel("Candidate KL reduction (%)")
    axes[1, 0].set_title("C  Five transfer tasks")
    axes[1, 0].legend(frameon=False)

    epsilon = 1e-8
    distributions = []
    labels = []
    colors = []
    for analysis, variant, label, color in (
        (args.oracle_analysis, "mixed_s4", "Oracle mixed", "#009E73"),
        (args.static_analysis, "mixed_static_s4", "Static mixed", "#E69F00"),
    ):
        rows = [
            row
            for row in load_csv(analysis / "paired_samples.csv")
            if row["variant"] == variant
        ]
        distributions.append(
            np.log10(
                (
                    np.asarray([float(row["euclidean_candidate_kl"]) for row in rows])
                    + epsilon
                )
                / (
                    np.asarray([float(row["candidate_kl"]) for row in rows])
                    + epsilon
                )
            )
        )
        labels.append(label)
        colors.append(color)
    box = axes[1, 1].boxplot(
        distributions,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1, 1].axhline(0, color="#555555", linewidth=0.9)
    axes[1, 1].set_ylabel("log10(Euclidean KL / method KL)")
    axes[1, 1].set_title("D  Per-sample gain distribution")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.7)
    figure.tight_layout(pad=1.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
