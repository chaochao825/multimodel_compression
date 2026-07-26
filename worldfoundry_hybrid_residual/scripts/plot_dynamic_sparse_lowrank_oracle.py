#!/usr/bin/env python3
"""Plot the dynamic sparse plus low-rank oracle and transfer gaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    return parser.parse_args()


def label(method: str) -> str:
    return {
        "mass_topk": "Attention mass",
        "contribution_norm": "Output norm",
        "dense_output_greedy": "Dense-output greedy",
        "renorm_output_greedy": "Renorm-output greedy",
    }.get(method, method)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit("matplotlib is required; install it in the experiment environment") from error

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["rank"]) == args.rank]
    if not selected:
        raise RuntimeError(f"no rank-{args.rank} rows in {args.summary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(row["method"] for row in selected))
    colors = dict(zip(methods, ("#0B6E75", "#D97941", "#4F6D2F", "#9A3E25")))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)

    for method in methods:
        renorm = sorted(
            (row for row in selected if row["method"] == method and row["normalization"] == "renormalized"),
            key=lambda row: float(row["density"]),
        )
        density = [100 * float(row["density"]) for row in renorm]
        axes[0, 0].plot(
            density,
            [100 * float(row["test_oracle_self_relative_l2"]) for row in renorm],
            marker="o",
            color=colors[method],
            label=label(method),
        )
        axes[0, 1].plot(
            density,
            [100 * float(row["test_frozen_calibration_basis_relative_l2"]) for row in renorm],
            marker="s",
            color=colors[method],
            label=label(method),
        )
        axes[1, 0].plot(
            density,
            [float(row["mask_jaccard_mean"]) for row in renorm],
            marker="D",
            color=colors[method],
            label=label(method),
        )

    normalization_rows = [
        row for row in selected if row["method"] == "renorm_output_greedy"
    ]
    for normalization, style, color in (
        ("dense_probability", "--", "#557A95"),
        ("renormalized", "-", "#B23A48"),
    ):
        group = sorted(
            (row for row in normalization_rows if row["normalization"] == normalization),
            key=lambda row: float(row["density"]),
        )
        axes[1, 1].plot(
            [100 * float(row["density"]) for row in group],
            [100 * float(row["test_oracle_self_relative_l2"]) for row in group],
            marker="o",
            linestyle=style,
            color=color,
            label=normalization.replace("_", " "),
        )

    axes[0, 0].set_title("Per-sample oracle: dynamic mask + adaptive tail")
    axes[0, 1].set_title("Static transfer upper bound: frozen mask + basis")
    axes[1, 0].set_title("Cross-seed dynamic-mask overlap")
    axes[1, 1].set_title("Normalization tax (best renorm-oriented router)")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.axhline(2.0, color="#333333", linestyle=":", linewidth=1.2, label="2% target")
        axis.set_ylabel("Output relative L2 (%)")
        axis.set_xlabel("Selected key-block density (%)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
    axes[1, 0].set_ylabel("Jaccard")
    axes[1, 0].set_xlabel("Selected key-block density (%)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.22)
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        axis.legend(dedup.values(), dedup.keys(), fontsize=8, frameon=False)
    fig.suptitle(f"Wan F81 output-aware sparse-critical + rank-{args.rank} tail", fontsize=15, fontweight="bold")
    fig.savefig(args.output_dir / "dynamic_sparse_lowrank_oracle.png", dpi=180)
    fig.savefig(args.output_dir / "dynamic_sparse_lowrank_oracle.pdf")
    print(f"[plot] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
