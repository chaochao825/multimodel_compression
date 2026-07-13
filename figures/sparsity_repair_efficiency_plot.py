#!/usr/bin/env python3
"""Plot parameter-efficient sparsity/pruning repair diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "remote_logs" / "sparsity_repair_probe_20260712.json"
DEFAULT_NAME = "fig24_sparsity_repair_parameter_efficiency"

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#777777",
    "black": "#222222",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, letter: str, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        f"({letter}) {label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def select(
    rows: Sequence[Mapping[str, object]],
    method: str,
    **filters: object,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and all(row.get(key) == value for key, value in filters.items())
    ]


def aggregate_point(
    rows: Sequence[Mapping[str, object]],
    method: str,
    metric: str = "normalized_mse",
    **filters: object,
) -> tuple[float, float]:
    selected = select(rows, method, **filters)
    return (
        100.0 * mean(float(row["parameter_fraction_of_dense_fp16"]) for row in selected),
        mean(float(row[metric]) for row in selected),
    )


def draw(payload: Mapping[str, object], output_dir: Path, name: str) -> None:
    rows: list[Mapping[str, object]] = payload["rows"]  # type: ignore[assignment]
    summary: Mapping[str, object] = payload["summary"]  # type: ignore[assignment]
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 13.6), constrained_layout=True)

    # (a) Exact row-top-k pruning and tiny tail repairs.
    ax = axes[0, 0]
    row_specs = (
        ("row_topk_renormalized", "Top-k + renorm", COLORS["red"], "o", "--"),
        ("row_topk_plus_global_uniform_gate", "+ 1 global gate", COLORS["blue"], "s", "-"),
        (
            "row_topk_plus_query_block_uniform_gate",
            "+ query-block gates",
            COLORS["orange"],
            "^",
            "-",
        ),
        (
            "row_topk_plus_mass_conserving_uniform_tail",
            "+ derived mass / uniform tail",
            COLORS["green"],
            "D",
            "-",
        ),
        (
            "row_topk_plus_column_prior_tail",
            "+ one column prior",
            COLORS["purple"],
            "P",
            "-",
        ),
        (
            "row_topk_plus_equal_prior_bits_extra_sparse",
            "+ equal-bit extra nnz",
            COLORS["gray"],
            "x",
            ":",
        ),
        (
            "row_topk_retained_column_group_scale",
            "Scale retained support only",
            COLORS["black"],
            "v",
            ":",
        ),
    )
    for method, label, color, marker, linestyle in row_specs:
        points = [
            aggregate_point(rows, method, keep_ratio=ratio)
            for ratio in (0.05, 0.10, 0.20, 0.40)
        ]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Nominal payload / dense FP16 (%)")
    ax.set_ylabel("Mean attention NRMSE (log)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right", ncol=1)
    panel_label(ax, "a", "Row-top-k: protect missing probability mass")

    # (b) Low-bit row-top-k: amplitude scale versus stochastic-tail repair.
    ax = axes[0, 1]
    bits_values = (2, 3, 4)
    quant_specs = (
        ("shape_only", "Sparse shape only", "normalized_mse", COLORS["red"], "o", "--"),
        (
            "mass_scale_only",
            "Row-mass protected (raw path)",
            "raw_normalized_mse",
            COLORS["blue"],
            "s",
            "-",
        ),
        (
            "mass_scale_only",
            "Row mass then renorm (no-op)",
            "normalized_mse",
            COLORS["gray"],
            "x",
            ":",
        ),
        (
            "mass_uniform_tail",
            "Mass + uniform tail",
            "normalized_mse",
            COLORS["green"],
            "D",
            "-",
        ),
        (
            "mass_prior_tail",
            "Mass + column-prior tail",
            "normalized_mse",
            COLORS["purple"],
            "P",
            "-",
        ),
    )
    for suffix, label, metric, color, marker, linestyle in quant_specs:
        y = []
        for bits in bits_values:
            selected = select(
                rows,
                f"row_topk_q{bits}_{suffix}",
                keep_ratio=0.10,
            )
            y.append(mean(float(row[metric]) for row in selected))
        ax.plot(
            bits_values,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(bits_values)
    ax.set_xlabel("Sparse value code bits (10% row-top-k)")
    ax.set_ylabel("Mean matrix NRMSE (log)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")
    panel_label(ax, "b", "Mass scale helps raw amplitude; tail repair survives renorm")

    # (c) Structured block pruning.
    ax = axes[1, 0]
    block_specs = (
        ("block_topk_renormalized", "Kept blocks only", COLORS["red"], "o", "--"),
        ("block_topk_plus_block_mass", "+ one mass / block", COLORS["blue"], "s", "-"),
        ("block_topk_plus_row_mass", "+ b row masses / block", COLORS["green"], "D", "-"),
        (
            "block_topk_plus_equal_block_mass_bits_extra_blocks",
            "+ equal-bit full blocks",
            COLORS["gray"],
            "x",
            ":",
        ),
    )
    for method, label, color, marker, linestyle in block_specs:
        points = [
            aggregate_point(rows, method, keep_ratio=ratio)
            for ratio in (0.05, 0.10, 0.25, 0.50)
        ]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Nominal payload / dense FP16 (%)")
    ax.set_ylabel("Mean attention NRMSE (log)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")
    panel_label(ax, "c", "Block pruning: mass summaries beat equal-bit extra blocks")

    # (d) Same 25% cap: fit order and sparse codec.
    ax = axes[1, 1]
    methods = (
        ("sparse_residual_fp16_exact", "FP16 exact"),
        ("sparse_residual_fp16_loss_aware_global_gain", "FP16 + folded gain"),
        ("sparse_residual_q4_global_loss_aware", "q4 global"),
        ("sparse_residual_q4_block_row_loss_aware", "q4 query-block"),
        ("sparse_residual_q4_per_row_loss_aware", "q4 row"),
        ("sparse_error_feedback_q4_block_row_t4", "q4 block / T4"),
    )
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.34
    for offset, order, label, color in (
        (-width / 2, "backbone_first", "Backbone → sparse", COLORS["sky"]),
        (width / 2, "component_first", "Sparse → backbone", COLORS["orange"]),
    ):
        y = [
            mean(
                float(row["normalized_mse"])
                for row in select(rows, method, cap_fraction=0.25, order=order)
            )
            for method, _ in methods
        ]
        ax.bar(x + offset, y, width, color=color, label=label)
        for xpos, value in zip(x + offset, y):
            if np.isfinite(value):
                ax.text(
                    xpos,
                    value + 0.003,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.0,
                )
    ax.set_xticks(x, [label for _, label in methods], rotation=18, ha="right")
    ax.set_ylabel("Mean attention NRMSE")
    ax.set_ylim(0.0, 0.18)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    panel_label(ax, "d", "Same 25% cap: outlier-first and fine scales win")

    # (e) Max scale versus row-normalized-loss-aware folded scale.
    ax = axes[2, 0]
    schemes = ("global", "block_row", "per_row")
    x = np.arange(len(schemes), dtype=np.float64)
    styles = (
        ("backbone_first", "max_scale", "Backbone-first / max", COLORS["sky"], "o", "--"),
        (
            "backbone_first",
            "loss_aware",
            "Backbone-first / loss-aware",
            COLORS["blue"],
            "s",
            "-",
        ),
        (
            "component_first",
            "max_scale",
            "Component-first / max",
            COLORS["orange"],
            "^",
            "--",
        ),
        (
            "component_first",
            "loss_aware",
            "Component-first / loss-aware",
            COLORS["green"],
            "D",
            "-",
        ),
    )
    for order, objective, label, color, marker, linestyle in styles:
        y = []
        for scheme in schemes:
            method = f"sparse_residual_q4_{scheme}_{objective}"
            selected = select(rows, method, cap_fraction=0.25, order=order)
            y.append(mean(float(row["normalized_mse"]) for row in selected))
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, ["Global", "Query-block", "Row"])
    ax.set_xlabel("Sparse scale granularity")
    ax.set_ylabel("Mean attention NRMSE (log)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")
    panel_label(ax, "e", "Same bits: optimize scale for the reported loss")

    # (f) Enumerated budget envelope and one-scalar calibration diagnostic.
    ax = axes[2, 1]
    envelope = [
        row
        for row in summary["budget_envelope"]  # type: ignore[index]
        if float(row["parameter_fraction_budget"]) <= 0.50
    ]
    budgets = [100.0 * float(row["parameter_fraction_budget"]) for row in envelope]
    ax.plot(
        budgets,
        [float(row["mean_best_normalized_mse"]) for row in envelope],
        color=COLORS["blue"],
        marker="o",
        label="Map-weighted target-fit envelope",
    )
    ax.plot(
        budgets,
        [float(row["source_balanced_mean_best_normalized_mse"]) for row in envelope],
        color=COLORS["orange"],
        marker="s",
        label="4-source-balanced envelope",
    )
    calibrated = summary["calibrated_global_gain"]  # type: ignore[index]
    ax.scatter(
        [25.0],
        [float(calibrated["mean_normalized_mse"])],
        color=COLORS["purple"],
        marker="*",
        s=130,
        label="LOSO one-gain diagnostic",
        zorder=5,
    )
    ax.annotate(
        f"gain={float(calibrated['mean_calibrated_gain']):.2f}",
        (25.0, float(calibrated["mean_normalized_mse"])),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8.2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Nominal payload budget / dense FP16 (%)")
    ax.set_ylabel("Mean best attention NRMSE (log)")
    ax.grid(True, which="both", alpha=0.22)
    ax.set_xlim(-1.0, 52.0)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.04,
        "Envelope selects support/config on each target;\nLOSO calibrates gain only, not sparse support.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )
    panel_label(ax, "f", "Parameter envelope remains target-fitted")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_style()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    draw(payload, args.output_dir, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
