from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


METHODS = (
    "shared_exact_local",
    "headwise_attention_mass",
    "headwise_exact_local",
    "headwise_exact_greedy",
)
METHOD_LABELS = {
    "shared_exact_local": "Shared exact-local",
    "headwise_attention_mass": "Headwise mass",
    "headwise_exact_local": "Headwise exact-local",
    "headwise_exact_greedy": "Headwise greedy",
}
METHOD_COLORS = {
    "shared_exact_local": "#7A7A7A",
    "headwise_attention_mass": "#E69F00",
    "headwise_exact_local": "#009E73",
    "headwise_exact_greedy": "#CC79A7",
}
EXPECTED_ROWS = 24 * 3 * len(METHODS) * 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(analysis_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(analysis_dir / "headwise_support_rows.csv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if tuple(rows["method"].drop_duplicates()) != METHODS:
        raise ValueError("registered headwise method order changed")
    if sorted(rows["sample_position"].unique().tolist()) != list(range(73, 97)):
        raise ValueError("exposed sample positions changed")
    return rows


def build_tables(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curve = (
        rows.groupby(["method", "exact_group_count"], sort=False)
        .agg(
            visual_token_retention=("visual_token_retention", "first"),
            visual_mean=("visual_relative_l2", "mean"),
            visual_p95=("visual_relative_l2", lambda values: values.quantile(0.95)),
            full_mean=("full_relative_l2", "mean"),
            full_p95=("full_relative_l2", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    budget = rows[rows["exact_group_count"].eq(196)]
    layer = (
        budget.groupby(["method", "layer_index"], sort=False)
        .agg(
            visual_mean=("visual_relative_l2", "mean"),
            visual_p95=("visual_relative_l2", lambda values: values.quantile(0.95)),
            visual_worst=("visual_relative_l2", "max"),
            full_mean=("full_relative_l2", "mean"),
            full_p95=("full_relative_l2", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    shared = layer[layer["method"].eq("shared_exact_local")].set_index("layer_index")
    headwise = layer[layer["method"].eq("headwise_exact_local")].set_index(
        "layer_index"
    )
    improvement = pd.DataFrame(
        {
            "layer_index": shared.index,
            "shared_visual_mean": shared["visual_mean"],
            "headwise_visual_mean": headwise["visual_mean"],
            "relative_improvement": 1 - headwise["visual_mean"] / shared["visual_mean"],
        }
    ).reset_index(drop=True)
    return {"curve": curve, "layer": layer, "improvement": improvement}


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["curve"].to_csv(
        output_dir / "query_fixed_headwise_curve_summary.csv", index=False
    )
    tables["layer"].to_csv(
        output_dir / "query_fixed_headwise_layer_summary.csv", index=False
    )
    tables["improvement"].to_csv(
        output_dir / "query_fixed_headwise_improvement_summary.csv", index=False
    )


def _plot_curves(
    axis: plt.Axes,
    curve: pd.DataFrame,
    *,
    mean_column: str,
    p95_column: str,
    threshold: float,
    ylabel: str,
) -> None:
    for method in METHODS:
        selected = curve[
            curve["method"].eq(method) & curve["exact_group_count"].lt(392)
        ].sort_values("visual_token_retention")
        x = selected["visual_token_retention"].to_numpy()
        mean = selected[mean_column].to_numpy()
        p95 = selected[p95_column].to_numpy()
        axis.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            label=METHOD_LABELS[method],
        )
        axis.fill_between(x, mean, p95, color=METHOD_COLORS[method], alpha=0.12)
    axis.axhline(threshold, color="#B2182B", linestyle="--", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_xlabel("Retained visual tokens")
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def render(
    rows: pd.DataFrame, tables: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    _plot_curves(
        axes[0, 0],
        tables["curve"],
        mean_column="visual_mean",
        p95_column="visual_p95",
        threshold=0.01,
        ylabel="Visual-output relative L2",
    )
    axes[0, 0].legend(loc="upper right", frameon=False)
    axes[0, 0].text(0.01, 0.98, "(a)", transform=axes[0, 0].transAxes, va="top")

    _plot_curves(
        axes[0, 1],
        tables["curve"],
        mean_column="full_mean",
        p95_column="full_p95",
        threshold=0.005,
        ylabel="Full-attention relative L2",
    )
    axes[0, 1].text(0.01, 0.98, "(b)", transform=axes[0, 1].transAxes, va="top")

    local = rows[
        rows["method"].eq("headwise_exact_local") & rows["exact_group_count"].eq(196)
    ]
    layer_values = [
        100 * local[local["layer_index"].eq(layer)]["visual_relative_l2"].to_numpy()
        for layer in (0, 13, 27)
    ]
    box = axes[1, 0].boxplot(
        layer_values,
        tick_labels=["0", "13", "27"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], ("#56B4E9", "#009E73", "#D55E00")):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(20260830)
    for index, values in enumerate(layer_values, start=1):
        axes[1, 0].scatter(
            index + rng.uniform(-0.11, 0.11, size=len(values)),
            values,
            color="#333333",
            alpha=0.42,
            s=13,
            linewidth=0,
        )
    axes[1, 0].axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xlabel("Qwen2 layer")
    axes[1, 0].set_ylabel("Headwise exact-local visual L2 (%)")
    axes[1, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 0].text(0.01, 0.98, "(c)", transform=axes[1, 0].transAxes, va="top")

    improvement = tables["improvement"]
    bars = axes[1, 1].bar(
        improvement["layer_index"].astype(str),
        improvement["relative_improvement"],
        color=("#56B4E9", "#009E73", "#D55E00"),
        width=0.58,
    )
    axes[1, 1].axhline(0.25, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 1].set_ylim(0, 0.76)
    axes[1, 1].set_xlabel("Qwen2 layer")
    axes[1, 1].set_ylabel("Improvement over shared support")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 1].text(0.01, 0.98, "(d)", transform=axes[1, 1].transAxes, va="top")
    for bar, value in zip(bars, improvement["relative_improvement"]):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    output_stem = output_dir / "query_fixed_headwise_support_ceiling"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.analysis_dir)
    tables = build_tables(rows)
    write_tables(tables, args.output_dir)
    render(rows, tables, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
