from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


CLUSTER_FAMILIES = ("key", "key_value")
PROTOTYPE_COUNTS = (32, 64, 128)
SELECTORS = ("oracle_local", "prototype_mass")
FAMILY_LABELS = {"key": "K-only", "key_value": "K+V"}
FAMILY_COLORS = {"key": "#0072B2", "key_value": "#D55E00"}
SELECTOR_LABELS = {
    "oracle_local": "Target-visible support oracle",
    "prototype_mass": "Prototype-mass selector",
}
SELECTOR_MARKERS = {"oracle_local": "o", "prototype_mass": "s"}
SELECTOR_STYLES = {"oracle_local": "-", "prototype_mass": "--"}
EXPECTED_ROWS = 24 * 3 * len(CLUSTER_FAMILIES) * len(PROTOTYPE_COUNTS) * len(SELECTORS)
EXPECTED_SUMMARIES = len(CLUSTER_FAMILIES) * len(PROTOTYPE_COUNTS) * len(SELECTORS)
DIAGNOSTIC_RUNS = (
    "query_fixed_prototype_residual_greedy_diagnostic_v1",
    "query_fixed_prototype_reverse_greedy_diagnostic_v1",
)


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
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(
    analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(analysis_dir / "prototype_measure_rows.csv")
    summaries = pd.read_csv(analysis_dir / "prototype_measure_summary.csv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if len(summaries) != EXPECTED_SUMMARIES:
        raise ValueError(
            f"expected {EXPECTED_SUMMARIES} summaries, found {len(summaries)}"
        )
    if sorted(rows["sample_position"].unique().tolist()) != list(range(73, 97)):
        raise ValueError("exposed sample positions changed")
    if rows["active_token_count_max"].max() > 392:
        raise ValueError("registered active-token budget was exceeded")
    if rows["active_read_ratio"].min() < 3.8:
        raise ValueError("registered active-read eligibility changed")
    diagnostics = []
    for run_name in DIAGNOSTIC_RUNS:
        run_dir = analysis_dir.parent / run_name
        run_rows = pd.read_csv(run_dir / "prototype_measure_rows.csv")
        run_summary = pd.read_csv(run_dir / "prototype_measure_summary.csv")
        if len(run_rows) != 72 or len(run_summary) != 1:
            raise ValueError(f"unexpected support diagnostic shape: {run_name}")
        if sorted(run_rows["sample_position"].unique().tolist()) != list(range(73, 97)):
            raise ValueError(f"support diagnostic positions changed: {run_name}")
        diagnostics.append(run_summary)
    return rows, summaries, pd.concat(diagnostics, ignore_index=True)


def build_tables(
    rows: pd.DataFrame,
    summaries: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frontier = summaries.sort_values("visual_mean").reset_index(drop=True)
    best = frontier.iloc[0]
    best_rows = rows[
        rows["cluster_family"].eq(best["cluster_family"])
        & rows["prototype_count"].eq(best["prototype_count"])
        & rows["selector"].eq(best["selector"])
    ].copy()
    layer = (
        best_rows.groupby("layer_index")
        .agg(
            visual_mean=("visual_relative_l2", "mean"),
            visual_p95=("visual_relative_l2", lambda values: values.quantile(0.95)),
            visual_worst=("visual_relative_l2", "max"),
            full_mean=("full_relative_l2", "mean"),
            worst_head=("visual_worst_head_relative_l2", "max"),
            selected_mass=("selected_visual_mass_mean", "mean"),
        )
        .reset_index()
    )
    selector_gap = summaries.pivot(
        index=["cluster_family", "prototype_count"],
        columns="selector",
        values=["visual_mean", "visual_p95", "visual_worst"],
    )
    selector_gap.columns = [f"{metric}_{selector}" for metric, selector in selector_gap]
    selector_gap = selector_gap.reset_index()
    selector_gap["visual_mean_penalty_ratio"] = (
        selector_gap["visual_mean_prototype_mass"]
        / selector_gap["visual_mean_oracle_local"]
    )
    support_sensitivity = pd.concat(
        (
            summaries[
                summaries["cluster_family"].eq("key")
                & summaries["prototype_count"].eq(128)
                & summaries["selector"].isin(SELECTORS)
            ],
            diagnostics,
        ),
        ignore_index=True,
    )
    selector_order = {
        "prototype_mass": 0,
        "oracle_local": 1,
        "oracle_residual_greedy": 2,
        "oracle_reverse_greedy": 3,
    }
    support_sensitivity["selector_order"] = support_sensitivity["selector"].map(
        selector_order
    )
    support_sensitivity = support_sensitivity.sort_values("selector_order")

    correlations = []
    for family in CLUSTER_FAMILIES:
        for selector in SELECTORS:
            selected = rows[
                rows["cluster_family"].eq(family) & rows["selector"].eq(selector)
            ]
            correlations.append(
                {
                    "cluster_family": family,
                    "selector": selector,
                    "spearman_mass_error": selected[
                        ["selected_visual_mass_mean", "visual_relative_l2"]
                    ]
                    .corr(method="spearman")
                    .iloc[0, 1],
                }
            )
    return {
        "frontier": frontier,
        "best_rows": best_rows,
        "layer": layer,
        "selector_gap": selector_gap,
        "support_sensitivity": support_sensitivity,
        "correlations": pd.DataFrame(correlations),
        "rows": rows,
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["frontier"].to_csv(
        output_dir / "prototype_mixture_frontier.csv", index=False
    )
    tables["layer"].to_csv(
        output_dir / "prototype_mixture_best_layer_summary.csv", index=False
    )
    tables["selector_gap"].to_csv(
        output_dir / "prototype_mixture_selector_gap.csv", index=False
    )
    tables["support_sensitivity"].to_csv(
        output_dir / "prototype_mixture_support_sensitivity.csv", index=False
    )
    tables["correlations"].to_csv(
        output_dir / "prototype_mixture_mass_error_correlation.csv", index=False
    )


def render(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    frontier = tables["frontier"]

    for family in CLUSTER_FAMILIES:
        for selector in SELECTORS:
            selected = frontier[
                frontier["cluster_family"].eq(family)
                & frontier["selector"].eq(selector)
            ].sort_values("prototype_count")
            label = f"{FAMILY_LABELS[family]} / {SELECTOR_LABELS[selector]}"
            axes[0, 0].plot(
                selected["prototype_count"],
                selected["visual_mean"],
                color=FAMILY_COLORS[family],
                marker=SELECTOR_MARKERS[selector],
                linestyle=SELECTOR_STYLES[selector],
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
            axes[0, 0].fill_between(
                selected["prototype_count"],
                selected["visual_mean"],
                selected["visual_p95"],
                color=FAMILY_COLORS[family],
                alpha=0.08,
            )
    axes[0, 0].axhline(0.005, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(PROTOTYPE_COUNTS)
    axes[0, 0].set_xlabel("Positive prototypes per head")
    axes[0, 0].set_ylabel("Visual-output relative L2")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7.6)
    axes[0, 0].text(0.01, 0.98, "(a)", transform=axes[0, 0].transAxes, va="top")

    x = np.arange(len(PROTOTYPE_COUNTS))
    width = 0.19
    offset = 0
    for family in CLUSTER_FAMILIES:
        for selector in SELECTORS:
            selected = frontier[
                frontier["cluster_family"].eq(family)
                & frontier["selector"].eq(selector)
            ].set_index("prototype_count")
            axes[0, 1].bar(
                x + (offset - 1.5) * width,
                100 * selected.loc[list(PROTOTYPE_COUNTS), "visual_mean"].to_numpy(),
                width=width,
                color=FAMILY_COLORS[family],
                alpha=1.0 if selector == "oracle_local" else 0.48,
                hatch="" if selector == "oracle_local" else "//",
                label=f"{FAMILY_LABELS[family]} / {SELECTOR_LABELS[selector]}",
            )
            offset += 1
    axes[0, 1].axhline(0.5, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xticks(x, [str(value) for value in PROTOTYPE_COUNTS])
    axes[0, 1].set_xlabel("Positive prototypes per head")
    axes[0, 1].set_ylabel("Visual-output mean relative L2 (%)")
    axes[0, 1].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0, 1].text(0.01, 0.98, "(b)", transform=axes[0, 1].transAxes, va="top")

    sensitivity = tables["support_sensitivity"]
    layer_x = np.arange(len(sensitivity))
    axes[1, 0].bar(
        layer_x,
        100 * sensitivity["visual_mean"].to_numpy(),
        color=("#8C8C8C", "#0072B2", "#E69F00", "#009E73"),
        alpha=0.75,
        label="mean",
    )
    axes[1, 0].scatter(
        layer_x,
        100 * sensitivity["visual_p95"].to_numpy(),
        color="#111111",
        marker="o",
        s=35,
        label="P95",
        zorder=4,
    )
    axes[1, 0].scatter(
        layer_x,
        100 * sensitivity["visual_worst"].to_numpy(),
        color="#CC79A7",
        marker="^",
        s=42,
        label="worst",
        zorder=4,
    )
    axes[1, 0].axhline(0.5, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xticks(layer_x, ("Mass", "Local", "Forward", "Reverse"))
    axes[1, 0].set_xlabel("K-only / 128 support optimizer")
    axes[1, 0].set_ylabel("Visual-output relative L2 (%)")
    axes[1, 0].grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[1, 0].legend(loc="upper left", frameon=False)
    axes[1, 0].text(0.01, 0.98, "(c)", transform=axes[1, 0].transAxes, va="top")

    rows = tables["rows"]
    for family in CLUSTER_FAMILIES:
        for selector in SELECTORS:
            selected = rows[
                rows["cluster_family"].eq(family) & rows["selector"].eq(selector)
            ]
            axes[1, 1].scatter(
                selected["selected_visual_mass_mean"],
                selected["visual_relative_l2"],
                color=FAMILY_COLORS[family],
                marker=SELECTOR_MARKERS[selector],
                alpha=0.24,
                s=18,
                linewidth=0,
                label=f"{FAMILY_LABELS[family]} / {selector.replace('_', ' ')}",
            )
    axes[1, 1].axhline(0.005, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Attention mass replaced by exact clusters")
    axes[1, 1].set_ylabel("Visual-output relative L2")
    axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(alpha=0.22, linewidth=0.7)
    axes[1, 1].text(0.01, 0.98, "(d)", transform=axes[1, 1].transAxes, va="top")

    output_stem = output_dir / "query_fixed_prototype_mixture"
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
    rows, summaries, diagnostics = load_data(args.analysis_dir)
    tables = build_tables(rows, summaries, diagnostics)
    write_tables(tables, args.output_dir)
    render(tables, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
