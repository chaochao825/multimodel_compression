from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


SELECTORS = ("centroid_score", "quest_box_bound", "exact_mass", "oracle_local")
SELECTOR_LABELS = {
    "centroid_score": "Centroid score",
    "quest_box_bound": "Quest box bound",
    "exact_mass": "Exact-mass oracle",
    "oracle_local": "Local-output oracle",
}
SELECTOR_COLORS = {
    "centroid_score": "#0072B2",
    "quest_box_bound": "#D55E00",
    "exact_mass": "#009E73",
    "oracle_local": "#CC79A7",
}
EXPECTED_ROWS = 24 * 3 * 2 * len(SELECTORS) * 6
EXPECTED_SUMMARIES = 2 * len(SELECTORS) * 6


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
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(analysis_dir / "progressive_exact_page_rows.csv")
    summaries = pd.read_csv(analysis_dir / "progressive_exact_page_summary.csv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if len(summaries) != EXPECTED_SUMMARIES:
        raise ValueError(
            f"expected {EXPECTED_SUMMARIES} summaries, found {len(summaries)}"
        )
    if sorted(rows["sample_position"].unique().tolist()) != list(range(73, 97)):
        raise ValueError("exposed sample positions changed")
    if rows["certificate_coverage"].min() != 1.0:
        raise ValueError("registered certificate coverage changed")
    full_exact = rows[rows["exact_fraction"].eq(1.0)]
    if full_exact[["visual_relative_l2", "full_relative_l2"]].to_numpy().max() != 0:
        raise ValueError("full-exact identity control changed")
    return rows, summaries


def build_tables(
    rows: pd.DataFrame, summaries: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    spatial = summaries[summaries["topology"].eq("spatial_7x7")].copy()
    budget = rows[rows["topology"].eq("spatial_7x7") & rows["exact_fraction"].eq(0.25)]
    layer = (
        budget.groupby(["selector", "layer_index"], sort=False)
        .agg(
            selected_mass=("selected_visual_mass_mean", "mean"),
            visual_mean=("visual_relative_l2", "mean"),
            visual_p95=("visual_relative_l2", lambda values: values.quantile(0.95)),
            visual_worst=("visual_relative_l2", "max"),
            full_mean=("full_relative_l2", "mean"),
            bound_log10_looseness=("tail_bound_log10_looseness_mean", "mean"),
        )
        .reset_index()
    )
    exact_mass = spatial[spatial["selector"].eq("exact_mass")].copy()
    return {"spatial": spatial, "layer": layer, "exact_mass": exact_mass}


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["spatial"].to_csv(
        output_dir / "progressive_exact_spatial_frontier.csv", index=False
    )
    tables["layer"].to_csv(
        output_dir / "progressive_exact_layer25_summary.csv", index=False
    )
    tables["exact_mass"].to_csv(
        output_dir / "progressive_exact_mass_curve.csv", index=False
    )


def render(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    spatial = tables["spatial"]

    for selector in SELECTORS:
        selected = spatial[
            spatial["selector"].eq(selector) & spatial["exact_fraction"].lt(1.0)
        ].sort_values("exact_fraction")
        x = selected["exact_fraction"].to_numpy()
        mean = selected["visual_mean"].to_numpy()
        p95 = selected["visual_p95"].to_numpy()
        axes[0, 0].plot(
            x,
            mean,
            color=SELECTOR_COLORS[selector],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=SELECTOR_LABELS[selector],
        )
        axes[0, 0].fill_between(
            x, mean, p95, color=SELECTOR_COLORS[selector], alpha=0.10
        )
    axes[0, 0].axhline(0.01, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Exact visual pages")
    axes[0, 0].set_ylabel("Visual-output relative L2")
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), frameon=False)
    axes[0, 0].text(0.01, 0.98, "(a)", transform=axes[0, 0].transAxes, va="top")

    for selector in SELECTORS:
        selected = spatial[spatial["selector"].eq(selector)].sort_values(
            "exact_fraction"
        )
        axes[0, 1].plot(
            selected["exact_fraction"],
            selected["selected_visual_mass_mean"],
            color=SELECTOR_COLORS[selector],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=SELECTOR_LABELS[selector],
        )
    axes[0, 1].axhline(0.95, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xlabel("Exact visual pages")
    axes[0, 1].set_ylabel("Selected visual attention mass")
    axes[0, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].set_ylim(0.25, 1.02)
    axes[0, 1].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0, 1].text(0.01, 0.98, "(b)", transform=axes[0, 1].transAxes, va="top")

    layer = tables["layer"]
    x = np.arange(3)
    width = 0.19
    for index, selector in enumerate(SELECTORS):
        selected = layer[layer["selector"].eq(selector)].set_index("layer_index")
        axes[1, 0].bar(
            x + (index - 1.5) * width,
            100 * selected.loc[[0, 13, 27], "visual_mean"].to_numpy(),
            width=width,
            color=SELECTOR_COLORS[selector],
            label=SELECTOR_LABELS[selector],
        )
    axes[1, 0].axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xticks(x, ["0", "13", "27"])
    axes[1, 0].set_xlabel("Qwen2 layer")
    axes[1, 0].set_ylabel("Visual L2 at 25% exact pages (%)")
    axes[1, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 0].text(0.01, 0.98, "(c)", transform=axes[1, 0].transAxes, va="top")

    for selector in ("quest_box_bound", "exact_mass"):
        selected = layer[layer["selector"].eq(selector)].set_index("layer_index")
        axes[1, 1].plot(
            [0, 13, 27],
            selected.loc[[0, 13, 27], "bound_log10_looseness"],
            color=SELECTOR_COLORS[selector],
            marker="o",
            linewidth=1.9,
            label=SELECTOR_LABELS[selector],
        )
    axes[1, 1].set_xlabel("Qwen2 layer")
    axes[1, 1].set_ylabel("Tail-bound looseness (log10 ratio)")
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 1].legend(loc="lower right", frameon=False)
    axes[1, 1].text(0.01, 0.98, "(d)", transform=axes[1, 1].transAxes, va="top")

    output_stem = output_dir / "query_fixed_progressive_exact_pages"
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
