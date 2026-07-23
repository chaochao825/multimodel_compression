#!/usr/bin/env python3
"""Create publication-oriented plots from decomposition and H200 CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "main_only": "#667085",
    "robuq": "#D1495B",
    "qer_lowrank": "#00798C",
    "qer_bcm": "#EDAE49",
    "qer_bcm_lowrank": "#30638E",
    "qer_lowrank_bcm": "#6A994E",
    "qer_lowrank_sparse": "#9C6644",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def plot_quality(metrics: list[dict], output: Path) -> None:
    filtered = [
        row
        for row in metrics
        if row.get("main_kind") == "ternary"
        and "expanded" not in row.get("method", "")
        and "no_hadamard" not in row.get("method", "")
    ]
    grouped = defaultdict(list)
    for row in filtered:
        grouped[(row["family"], row["method"])].append(row)
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    points = []
    for (family, method), rows in grouped.items():
        x = sum(float(row["budget_ratio_vs_lr16"]) for row in rows) / len(rows)
        y = sum(float(row["activation_relative_l2"]) for row in rows) / len(rows)
        ax.scatter(x, y, s=58, color=COLORS.get(family, "#444444"), alpha=0.9)
        points.append((family, method, x, y))
    labels = []
    labels.extend(point for point in points if point[0] == "robuq")
    labels.extend(point for point in points if point[1] == "ternary_hadamard")
    labels.extend(point for point in points if point[1] == "ternary_qer_svd_r16")
    for family in ("qer_bcm_lowrank", "qer_lowrank_bcm", "qer_lowrank_sparse"):
        candidates = [point for point in points if point[0] == family and 0.9 <= point[2] <= 1.05]
        if candidates:
            labels.append(min(candidates, key=lambda point: point[3]))
    offsets = {
        "qer_bcm_lowrank": (8, 12),
        "qer_lowrank_bcm": (8, -14),
        "qer_lowrank_sparse": (-108, 8),
        "qer_lowrank": (8, -14),
    }
    seen = set()
    for family, method, x, y in labels:
        if (family, method) in seen:
            continue
        seen.add((family, method))
        if family == "robuq":
            label = method.replace("robuq_weight_svd_", "RobuQ ")
        elif method == "ternary_hadamard":
            label = "ternary + Hadamard"
        elif method == "ternary_qer_svd_r16":
            label = "fixed-error rank-16"
        else:
            label = family.replace("qer_", "").replace("_", " + ")
        ax.annotate(
            label,
            (x, y),
            xytext=offsets.get(family, (6, 5)),
            textcoords="offset points",
            fontsize=7.7,
        )
    ax.set_xlabel("High-precision residual parameters / rank-16 budget")
    ax.set_ylabel("Mean real-activation relative L2 error")
    ax.set_title("Wan DiT: residual quality under a matched high-precision budget")
    handles = []
    labels = []
    for family in sorted({row["family"] for row in filtered}):
        handles.append(ax.scatter([], [], color=COLORS.get(family, "#444444"), s=50))
        labels.append(family.replace("qer_", "QER ").replace("_", " "))
    ax.legend(handles, labels, frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_residual_budget.png")
    plt.close(fig)


def plot_spectrum(rows: list[dict], output: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append((int(row["rank"]), float(row["energy_ratio"])))
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    preferred = ["hadamard_weight", "ternary_error", "fp8_error"]
    for source in preferred:
        values = grouped.get(source)
        if not values:
            continue
        by_rank = defaultdict(list)
        for rank, energy in values:
            by_rank[rank].append(energy)
        ranks = sorted(by_rank)
        means = [sum(by_rank[rank]) / len(by_rank[rank]) for rank in ranks]
        ax.plot(ranks, means, marker="o", linewidth=2.2, label=source.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rank")
    ax.set_ylabel("Explained energy")
    ax.set_ylim(bottom=0)
    ax.set_title("Low-rank concentration before and after low-bit quantization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "residual_spectrum.png")
    plt.close(fig)


def plot_structure(rows: list[dict], output: Path) -> None:
    bcm = [row for row in rows if row.get("bcm_energy_ratio")]
    by_block = defaultdict(list)
    after_lr = defaultdict(list)
    for row in bcm:
        block = int(row["block_size"])
        by_block[block].append(float(row["bcm_energy_ratio"]))
        after_lr[block].append(float(row["bcm_after_lowrank_energy_ratio"]))
    blocks = sorted(by_block)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    width = 0.36
    positions = range(len(blocks))
    ax.bar(
        [p - width / 2 for p in positions],
        [sum(by_block[b]) / len(by_block[b]) for b in blocks],
        width,
        color="#EDAE49",
        label="BCM capture of quantization error",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [sum(after_lr[b]) / len(after_lr[b]) for b in blocks],
        width,
        color="#6A994E",
        label="BCM incremental capture after low rank",
    )
    ax.set_xticks(list(positions), [str(block) for block in blocks])
    ax.set_xlabel("Circulant block size")
    ax.set_ylabel("Energy ratio")
    ax.set_title("How much non-redundant structure remains for BCM?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "bcm_redundancy.png")
    plt.close(fig)


def plot_benchmark(rows: list[dict], output: Path) -> None:
    valid = [row for row in rows if row.get("status") == "ok"]
    row_counts = sorted({int(row["rows"]) for row in valid})
    selected_methods = [
        method
        for method in dict.fromkeys(row["method"] for row in valid)
        if "static_input" not in method
    ]
    fig, axes = plt.subplots(1, len(row_counts), figsize=(5.2 * len(row_counts), 5.4), squeeze=False)
    for axis, row_count in zip(axes[0], row_counts):
        subset = {row["method"]: row for row in valid if int(row["rows"]) == row_count}
        methods = [method for method in selected_methods if method in subset]
        speeds = [float(subset[method]["speedup_vs_bf16"]) for method in methods]
        colors = ["#667085" if method == "bf16_dense" else "#00798C" for method in methods]
        axis.barh(range(len(methods)), speeds, color=colors)
        axis.set_yticks(range(len(methods)), [method.replace("fp8_dynamic_", "") for method in methods], fontsize=8)
        axis.axvline(1.0, color="#D1495B", linestyle="--", linewidth=1.3)
        axis.set_xlabel("Measured speedup vs BF16")
        axis.set_title(f"M={row_count}")
    fig.suptitle("H200 eager latency: dynamic FP8 plus residual branches", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "h200_dynamic_speedup.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    largest = max(row_counts)
    for row in valid:
        if int(row["rows"]) != largest:
            continue
        marker = "s" if row["measurement_scope"] == "lower_bound" else "o"
        ax.scatter(
            float(row["latency_ms_median"]),
            float(row["output_relative_l2"]),
            marker=marker,
            s=68,
            color="#30638E" if "bcm" in row["method"] else "#00798C",
        )
        ax.annotate(
            row["method"].replace("fp8_", ""),
            (float(row["latency_ms_median"]), float(row["output_relative_l2"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.4,
        )
    ax.set_xlabel("Latency (ms), lower is better")
    ax.set_ylabel("Output relative L2, lower is better")
    ax.set_title(f"H200 quality-latency frontier at M={largest}")
    fig.tight_layout()
    fig.savefig(output / "h200_quality_latency_frontier.png")
    plt.close(fig)


def plot_alternating(rows: list[dict], output: Path) -> None:
    canonical = {
        "alternating_svd_r16": "rank-16",
        "alternating_bcm": "BCM + low rank",
        "rowsparse": "rank-8 + row sparse",
        "tilesparse": "rank-8 + tile sparse",
    }
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        if method == "alternating_svd_r16":
            label = canonical[method]
        elif "alternating_bcm" in method:
            label = canonical["alternating_bcm"]
        elif "rowsparse" in method:
            label = canonical["rowsparse"]
        elif "tilesparse" in method:
            label = canonical["tilesparse"]
        else:
            continue
        grouped[label][int(row["iteration"])].append(float(row["activation_relative_l2"]))
    colors = {
        "rank-16": "#D1495B",
        "BCM + low rank": "#30638E",
        "rank-8 + row sparse": "#6A994E",
        "rank-8 + tile sparse": "#9C6644",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for label, by_iteration in grouped.items():
        iterations = sorted(by_iteration)
        means = [sum(by_iteration[i]) / len(by_iteration[i]) for i in iterations]
        ax.plot(iterations, means, marker="o", linewidth=2.2, color=colors[label], label=label)
    ax.set_xticks(sorted({i for values in grouped.values() for i in values}))
    ax.set_xlabel("Training-free alternating projection iteration")
    ax.set_ylabel("Mean real-activation relative L2")
    ax.set_title("Quantization-aware refinement converges, but structure does not overtake rank-16")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "alternating_convergence.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition-dir", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--alternating-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    decomposition = Path(args.decomposition_dir)
    benchmark = Path(args.benchmark_dir)
    plot_quality(read_csv(decomposition / "decomposition_metrics.csv"), output)
    plot_spectrum(read_csv(decomposition / "residual_spectrum.csv"), output)
    plot_structure(read_csv(decomposition / "structure_stats.csv"), output)
    plot_benchmark(read_csv(benchmark / "h200_benchmark.csv"), output)
    if args.alternating_dir:
        plot_alternating(
            read_csv(Path(args.alternating_dir) / "alternating_metrics.csv"), output
        )
    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()
