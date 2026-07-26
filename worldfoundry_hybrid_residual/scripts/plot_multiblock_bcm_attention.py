#!/usr/bin/env python3
"""Visualize held-out multi-block BCM attention expressivity."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_COLORS = {
    "global_coarse_bccb": "#999999",
    "query_block_multi_bcm": "#4477AA",
    "coarse_tile_local_residual": "#EE7733",
}
METHOD_LABELS = {
    "global_coarse_bccb": "global periodic BCCB",
    "query_block_multi_bcm": "query-block multi-BCM",
    "coarse_tile_local_residual": "hierarchical BCM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_heads(
    heldout: list[dict[str, str]], baseline_model: str, selected_model: str
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in heldout:
        if row["model"] in (baseline_model, selected_model):
            grouped[(row["model"], int(row["head"]))].append(row)

    heads = sorted({head for _, head in grouped})
    output: list[dict[str, object]] = []
    for head in heads:
        baseline = grouped[(baseline_model, head)]
        selected = grouped[(selected_model, head)]
        baseline_error = mean(
            [float(row["attention_output_relative_l2"]) for row in baseline]
        )
        selected_error = mean(
            [float(row["attention_output_relative_l2"]) for row in selected]
        )
        output.append(
            {
                "head": head,
                "baseline_model": baseline_model,
                "selected_model": selected_model,
                "baseline_output_relative_l2": baseline_error,
                "selected_output_relative_l2": selected_error,
                "relative_improvement": (baseline_error - selected_error)
                / max(baseline_error, 1e-30),
                "selected_attention_relative_l2": mean(
                    [float(row["attention_relative_l2"]) for row in selected]
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    summary: list[dict[str, str]],
    head_rows: list[dict[str, object]],
    selected_model: str,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))

    for method in METHOD_COLORS:
        selected = [row for row in summary if row["method"] == method]
        axes[0, 0].scatter(
            [float(row["parameters_per_head"]) for row in selected],
            [100.0 * float(row["mean_attention_output_relative_l2"]) for row in selected],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            s=58,
            edgecolor="#222222",
            linewidth=0.45,
        )
    axes[0, 0].axhline(5.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[0, 0].axhline(2.0, color="#B23A48", linestyle=":", linewidth=1.2)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Fitted parameters per head (log scale)")
    axes[0, 0].set_ylabel("Mean held-out output relative L2 (%)")
    axes[0, 0].set_title("(a) Expressivity versus parameter count")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    heads = [int(row["head"]) for row in head_rows]
    width = 0.38
    axes[0, 1].bar(
        [head - width / 2 for head in heads],
        [100.0 * float(row["baseline_output_relative_l2"]) for row in head_rows],
        width=width,
        color="#999999",
        label="global BCCB",
    )
    axes[0, 1].bar(
        [head + width / 2 for head in heads],
        [100.0 * float(row["selected_output_relative_l2"]) for row in head_rows],
        width=width,
        color="#EE7733",
        label="best global hierarchical config",
    )
    axes[0, 1].axhline(5.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xticks(heads)
    axes[0, 1].set_xlabel("Attention head")
    axes[0, 1].set_ylabel("Held-out output relative L2 (%)")
    axes[0, 1].set_title("(b) Strong head heterogeneity")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.2)

    improvements = [100.0 * float(row["relative_improvement"]) for row in head_rows]
    axes[1, 0].bar(
        heads,
        improvements,
        color=["#228833" if value >= 0.0 else "#CC3311" for value in improvements],
    )
    axes[1, 0].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1, 0].set_xticks(heads)
    axes[1, 0].set_xlabel("Attention head")
    axes[1, 0].set_ylabel("Relative output-error reduction (%)")
    axes[1, 0].set_title("(c) Local tables help only selected heads")
    axes[1, 0].grid(axis="y", alpha=0.2)

    global_rows = [row for row in summary if row["method"] == "global_coarse_bccb"]
    for row in global_rows:
        scale = row["model"].split("_s", maxsplit=1)[-1]
        x_value = 100.0 * float(row["mean_wrap_leakage_mass_delta"])
        y_value = 100.0 * float(row["mean_attention_output_relative_l2"])
        axes[1, 1].scatter(
            x_value,
            y_value,
            color="#4477AA",
            s=72,
            edgecolor="#222222",
            linewidth=0.45,
        )
        axes[1, 1].annotate(scale, (x_value, y_value), xytext=(5, 4), textcoords="offset points")
    axes[1, 1].set_xlabel("Excess mass in modulo-alias region (percentage points)")
    axes[1, 1].set_ylabel("Mean held-out output relative L2 (%)")
    axes[1, 1].set_title("(d) Coarser periodic kernels amplify wrap leakage")
    axes[1, 1].grid(alpha=0.2)

    figure.text(
        0.5,
        0.005,
        f"Frozen selected model: {selected_model}. Pilot has two independent QKV content groups, not multi-prompt evidence.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"multiblock_bcm_attention.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    heldout = read_csv(args.heldout)
    summary = read_csv(args.summary)
    global_rows = [row for row in summary if row["method"] == "global_coarse_bccb"]
    hierarchical = [
        row for row in summary if row["method"] == "coarse_tile_local_residual"
    ]
    baseline = min(
        global_rows, key=lambda row: float(row["mean_attention_output_relative_l2"])
    )
    selected = min(
        hierarchical, key=lambda row: float(row["mean_attention_output_relative_l2"])
    )
    head_rows = aggregate_heads(heldout, baseline["model"], selected["model"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "multiblock_bcm_attention_head_plot_data.csv", head_rows)
    plot(summary, head_rows, selected["model"], args.output_dir)


if __name__ == "__main__":
    main()
