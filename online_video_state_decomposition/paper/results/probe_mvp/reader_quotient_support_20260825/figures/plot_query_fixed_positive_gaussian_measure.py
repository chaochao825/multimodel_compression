from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


RANKS = (0, 2, 4, 8, 16)
RANK_COLORS = {
    0: "#0072B2",
    2: "#009E73",
    4: "#E69F00",
    8: "#D55E00",
    16: "#CC79A7",
}
EXPECTED_ROWS = 24 * 3 * 2 * len(RANKS) * 2 * 5
EXPECTED_SUMMARIES = 2 * len(RANKS) * 2 * 5


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
            "legend.fontsize": 8.7,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(analysis_dir / "positive_gaussian_measure_rows.csv")
    summaries = pd.read_csv(analysis_dir / "positive_gaussian_measure_summary.csv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if len(summaries) != EXPECTED_SUMMARIES:
        raise ValueError(
            f"expected {EXPECTED_SUMMARIES} summaries, found {len(summaries)}"
        )
    if sorted(rows["sample_position"].unique().tolist()) != list(range(73, 97)):
        raise ValueError("exposed sample positions changed")
    full_exact = rows[rows["exact_fraction"].eq(1.0)]
    if full_exact[["visual_relative_l2", "full_relative_l2"]].to_numpy().max() != 0:
        raise ValueError("full-exact identity control changed")
    return rows, summaries


def build_tables(
    rows: pd.DataFrame, summaries: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    eligible = summaries[
        summaries["exact_fraction"].le(0.25) & summaries["active_read_ratio"].ge(2.0)
    ].sort_values("visual_mean")
    curves = summaries[
        summaries["topology"].eq("spatial_7x7")
        & summaries["selector"].eq("oracle_local")
        & summaries["exact_fraction"].lt(1.0)
    ].sort_values(["rank", "exact_fraction"])
    rank = summaries[
        summaries["topology"].eq("spatial_7x7")
        & summaries["selector"].eq("oracle_local")
        & summaries["exact_fraction"].eq(0.25)
    ].sort_values("rank")
    key = rows[
        rows["topology"].eq("spatial_7x7")
        & rows["rank"].eq(0)
        & rows["selector"].eq("oracle_local")
        & rows["exact_fraction"].eq(0.25)
    ]
    layer = (
        key.groupby("layer_index")
        .agg(
            visual_mean=("visual_relative_l2", "mean"),
            visual_p95=("visual_relative_l2", lambda values: values.quantile(0.95)),
            visual_worst=("visual_relative_l2", "max"),
            full_mean=("full_relative_l2", "mean"),
            full_p95=("full_relative_l2", lambda values: values.quantile(0.95)),
            worst_head=("visual_worst_head_relative_l2", "max"),
        )
        .reset_index()
    )
    return {
        "eligible": eligible,
        "curves": curves,
        "rank": rank,
        "key": key,
        "layer": layer,
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["eligible"].to_csv(
        output_dir / "positive_gaussian_eligible_frontier.csv", index=False
    )
    tables["rank"].to_csv(
        output_dir / "positive_gaussian_rank25_summary.csv", index=False
    )
    tables["layer"].to_csv(
        output_dir / "positive_gaussian_best_layer_summary.csv", index=False
    )


def render(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    curves = tables["curves"]
    for rank in RANKS:
        selected = curves[curves["rank"].eq(rank)]
        x = selected["exact_fraction"].to_numpy()
        mean = selected["visual_mean"].to_numpy()
        p95 = selected["visual_p95"].to_numpy()
        axes[0, 0].plot(
            x,
            mean,
            color=RANK_COLORS[rank],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=f"rank {rank}",
        )
        axes[0, 0].fill_between(x, mean, p95, color=RANK_COLORS[rank], alpha=0.10)
    axes[0, 0].axhline(0.01, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Exact visual groups")
    axes[0, 0].set_ylabel("Visual-output relative L2")
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0, 0].legend(loc="lower left", frameon=False, ncol=2)
    axes[0, 0].text(0.01, 0.98, "(a)", transform=axes[0, 0].transAxes, va="top")

    eligible = tables["eligible"]
    topology_colors = {
        "spatial_7x7": "#0072B2",
        "temporal2_spatial_7x7": "#D55E00",
    }
    selector_markers = {"compact_mass": "o", "oracle_local": "s"}
    for topology, color in topology_colors.items():
        for selector, marker in selector_markers.items():
            selected = eligible[
                eligible["topology"].eq(topology) & eligible["selector"].eq(selector)
            ]
            axes[0, 1].scatter(
                selected["active_read_ratio"],
                selected["visual_mean"],
                color=color,
                marker=marker,
                alpha=0.72,
                s=38,
                label=f"{topology.replace('_', ' ')} / {selector.replace('_', ' ')}",
            )
    best = eligible.iloc[0]
    axes[0, 1].scatter(
        [best["active_read_ratio"]],
        [best["visual_mean"]],
        color="#F0E442",
        edgecolor="#111111",
        marker="*",
        s=160,
        linewidth=0.8,
        zorder=5,
    )
    axes[0, 1].axhline(0.01, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(2.0, color="#555555", linestyle=":", linewidth=1.1)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Active-read arithmetic ratio")
    axes[0, 1].set_ylabel("Visual-output mean relative L2")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(alpha=0.22, linewidth=0.7)
    axes[0, 1].legend(loc="lower right", frameon=False, fontsize=7.8)
    axes[0, 1].text(0.01, 0.98, "(b)", transform=axes[0, 1].transAxes, va="top")

    key = tables["key"]
    layer_values = [
        100 * key[key["layer_index"].eq(layer)]["visual_relative_l2"].to_numpy()
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
    axes[1, 0].set_ylabel("Best eligible visual L2 (%)")
    axes[1, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 0].text(0.01, 0.98, "(c)", transform=axes[1, 0].transAxes, va="top")

    rank = tables["rank"]
    axes[1, 1].plot(
        rank["rank"],
        rank["visual_mean"],
        color="#0072B2",
        marker="o",
        linewidth=1.9,
        label="mean",
    )
    axes[1, 1].plot(
        rank["rank"],
        rank["visual_p95"],
        color="#D55E00",
        marker="s",
        linewidth=1.7,
        label="P95",
    )
    axes[1, 1].plot(
        rank["rank"],
        rank["visual_worst"],
        color="#CC79A7",
        marker="^",
        linewidth=1.5,
        label="worst",
    )
    axes[1, 1].axhline(0.01, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Gaussian covariance rank")
    axes[1, 1].set_ylabel("Visual-output relative L2")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 1].legend(loc="lower right", frameon=False)
    axes[1, 1].text(0.01, 0.98, "(d)", transform=axes[1, 1].transAxes, va="top")

    output_stem = output_dir / "query_fixed_positive_gaussian_measure"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    svg_path = output_stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows, summaries = load_data(args.analysis_dir)
    tables = build_tables(rows, summaries)
    write_tables(tables, args.output_dir)
    render(tables, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
