from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    variants = {
        row["variant"]: row for row in summary["variant_summary"]
    }
    task_rows = read_csv(args.analysis_dir / "task_summary.csv")
    paired_rows = read_csv(args.analysis_dir / "paired_samples.csv")
    colors = {"fisher_s4": "#0072B2", "mixed_s4": "#009E73"}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.3))

    methods = ["fisher_s4", "mixed_s4"]
    x = np.arange(len(methods))
    width = 0.34
    candidate = [
        100.0 * float(variants[method]["aggregate_candidate_kl_reduction"])
        for method in methods
    ]
    vocabulary = [
        100.0 * float(variants[method]["aggregate_vocabulary_kl_reduction"])
        for method in methods
    ]
    axes[0, 0].bar(x - width / 2, candidate, width, color="#0072B2", label="Candidate KL")
    axes[0, 0].bar(x + width / 2, vocabulary, width, color="#E69F00", label="Vocabulary KL")
    axes[0, 0].axhline(25, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xticks(x, ["Fisher s4", "Mixed s4"])
    axes[0, 0].set_ylabel("Aggregate KL reduction (%)")
    axes[0, 0].set_title("A  Equal-byte readout preservation")
    axes[0, 0].legend(frameon=False)

    epsilon = 1e-8
    for method in methods:
        rows = [row for row in paired_rows if row["variant"] == method]
        baseline = np.maximum(
            np.asarray([float(row["euclidean_candidate_kl"]) for row in rows]),
            epsilon,
        )
        values = np.maximum(
            np.asarray([float(row["candidate_kl"]) for row in rows]),
            epsilon,
        )
        axes[0, 1].scatter(
            baseline,
            values,
            s=28,
            alpha=0.72,
            color=colors[method],
            label=method.replace("_s4", ""),
        )
    bounds = [epsilon, 1e-1]
    axes[0, 1].plot(bounds, bounds, color="#555555", linewidth=1.0, linestyle="--")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlim(bounds)
    axes[0, 1].set_ylim(bounds)
    axes[0, 1].set_xlabel("Euclidean s4 candidate KL")
    axes[0, 1].set_ylabel("Reader-aware candidate KL")
    axes[0, 1].set_title("B  Paired samples (below diagonal is better)")
    axes[0, 1].legend(frameon=False)

    tasks = sorted({row["task"] for row in task_rows})
    x = np.arange(len(tasks))
    for offset, method in zip((-width / 2, width / 2), methods, strict=True):
        by_task = {
            row["task"]: 100.0 * float(row["aggregate_reduction"])
            for row in task_rows
            if row["variant"] == method
        }
        axes[1, 0].bar(
            x + offset,
            [by_task[task] for task in tasks],
            width,
            color=colors[method],
            label=method.replace("_s4", ""),
        )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.9)
    axes[1, 0].axhline(25, color="#555555", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xticks(x, [task.replace("_", " ") for task in tasks], rotation=20, ha="right")
    axes[1, 0].set_ylabel("Candidate KL reduction (%)")
    axes[1, 0].set_title("C  Transfer by task")
    axes[1, 0].legend(frameon=False)

    for method in methods:
        rows = [row for row in paired_rows if row["variant"] == method]
        overlap = np.asarray(
            [float(row["support_overlap_with_euclidean"]) for row in rows]
        )
        log_gain = np.log10(
            (
                np.asarray([float(row["euclidean_candidate_kl"]) for row in rows])
                + epsilon
            )
            / (
                np.asarray([float(row["candidate_kl"]) for row in rows])
                + epsilon
            )
        )
        axes[1, 1].scatter(
            overlap,
            np.minimum(log_gain, 2.5),
            s=30,
            alpha=0.72,
            color=colors[method],
            label=method.replace("_s4", ""),
        )
    axes[1, 1].axhline(0, color="#555555", linewidth=0.9)
    axes[1, 1].set_ylim(-1.2, 2.65)
    axes[1, 1].set_xlabel("Support overlap with Euclidean")
    axes[1, 1].set_ylabel("log10(Euclidean KL / method KL)")
    axes[1, 1].set_title("D  Different support, functional gain")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].text(
        0.02,
        0.97,
        "one Fisher outlier clipped at 2.5",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8.5,
        color="#555555",
    )

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
