from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent
RESULT_ROOT = FIGURE_DIR.parent
ANALYSIS_ROOT = RESULT_ROOT / "analysis" / "onevision_reader_quotient_stage_a_20260830"
ADDITIVE_ROOT = ANALYSIS_ROOT / "additive_nz_feature_state_dev_v1"
HYBRID_ROOT = ANALYSIS_ROOT / "exact_boundary_additive_tail_dev_v1"
GEOMETRY_ROOT = ANALYSIS_ROOT / "exact_boundary_tail_geometry_dev_v2_fp64"
COLORS = {
    "all": "#0072B2",
    "exact": "#999999",
    "mass": "#D55E00",
    "effect": "#009E73",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, ...]:
    additive_history = pd.read_csv(ADDITIVE_ROOT / "training_history.csv")
    additive_rows = pd.read_csv(ADDITIVE_ROOT / "learned_development_rows.csv")
    hybrid_history = pd.read_csv(HYBRID_ROOT / "training_history.csv")
    exact_rows = pd.read_csv(HYBRID_ROOT / "exact_only_development_rows.csv")
    hybrid_rows = pd.read_csv(HYBRID_ROOT / "learned_tail_development_rows.csv")
    geometry_rows = pd.read_csv(GEOMETRY_ROOT / "tail_geometry_head_rows.csv")
    return (
        additive_history,
        additive_rows,
        hybrid_history,
        exact_rows,
        hybrid_rows,
        geometry_rows,
    )


def write_derived_data(
    additive_history: pd.DataFrame,
    additive_rows: pd.DataFrame,
    hybrid_history: pd.DataFrame,
    exact_rows: pd.DataFrame,
    hybrid_rows: pd.DataFrame,
    geometry_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_curve = (
        additive_history.groupby("step", as_index=False)["development_visual_mean"]
        .mean()
        .assign(method="Whole visual additive state")
    )
    hybrid_curve = (
        hybrid_history.groupby(["selector", "step"], as_index=False)[
            "development_visual_mean"
        ]
        .mean()
        .assign(
            method=lambda frame: frame["selector"].map(
                {
                    "mass_topk": "Exact mass boundary + tail",
                    "effect_topk": "Exact effect boundary + tail",
                }
            )
        )
    )
    training = pd.concat(
        [
            all_curve[["step", "development_visual_mean", "method"]],
            hybrid_curve[["step", "development_visual_mean", "method"]],
        ],
        ignore_index=True,
    )

    layer_frames = []
    for label, frame in (
        ("Whole visual additive state", additive_rows),
        (
            "Exact mass only",
            exact_rows[exact_rows["selector"] == "mass_topk"],
        ),
        (
            "Exact mass boundary + tail",
            hybrid_rows[hybrid_rows["selector"] == "mass_topk"],
        ),
        (
            "Exact effect boundary + tail",
            hybrid_rows[hybrid_rows["selector"] == "effect_topk"],
        ),
    ):
        metrics = (
            frame.groupby("layer_index", as_index=False)[
                ["visual_relative_l2", "full_relative_l2"]
            ]
            .mean()
            .assign(method=label)
        )
        layer_frames.append(metrics)
    layer_metrics = pd.concat(layer_frames, ignore_index=True)
    geometry = geometry_rows.groupby("layer_index", as_index=False)[
        [
            "mass_retained",
            "tail_ess_fraction",
            "tail_normalized_entropy",
            "support_jaccard",
        ]
    ].mean()

    training.to_csv(
        FIGURE_DIR / "conditional_redundancy_training_curves.csv", index=False
    )
    layer_metrics.to_csv(
        FIGURE_DIR / "conditional_redundancy_layer_metrics.csv", index=False
    )
    geometry.to_csv(
        FIGURE_DIR / "conditional_redundancy_tail_geometry.csv", index=False
    )
    return training, layer_metrics, geometry


def method_color(method: str) -> str:
    if method == "Whole visual additive state":
        return COLORS["all"]
    if method == "Exact mass only":
        return COLORS["exact"]
    if "effect" in method.lower():
        return COLORS["effect"]
    return COLORS["mass"]


def draw() -> None:
    configure_style()
    data = load_data()
    training, layer_metrics, geometry = write_derived_data(*data)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), constrained_layout=True)

    ax = axes[0, 0]
    for method, frame in training.groupby("method", sort=False):
        ax.plot(
            frame["step"],
            100 * frame["development_visual_mean"],
            label=method,
            color=method_color(method),
            linewidth=2,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1% signal")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean visual output error (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[0, 1]
    methods = [
        "Whole visual additive state",
        "Exact mass only",
        "Exact mass boundary + tail",
        "Exact effect boundary + tail",
    ]
    layers = [0, 13, 27]
    width = 0.19
    x = np.arange(len(layers))
    for index, method in enumerate(methods):
        frame = layer_metrics[layer_metrics["method"] == method].set_index(
            "layer_index"
        )
        ax.bar(
            x + (index - 1.5) * width,
            [100 * frame.loc[layer, "visual_relative_l2"] for layer in layers],
            width=width,
            label=method,
            color=method_color(method),
            edgecolor="black",
            linewidth=0.35,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, [f"L{layer}" for layer in layers])
    ax.set_ylabel("Mean visual output error (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1, 0]
    geometry_metrics = [
        ("mass_retained", "Top-25% mass"),
        ("tail_ess_fraction", "Tail ESS fraction"),
        ("tail_normalized_entropy", "Tail entropy"),
        ("support_jaccard", "Mass/effect Jaccard"),
    ]
    geometry_colors = ["#0072B2", "#E69F00", "#56B4E9", "#009E73"]
    width = 0.19
    for index, ((column, label), color) in enumerate(
        zip(geometry_metrics, geometry_colors, strict=True)
    ):
        values = geometry.set_index("layer_index")[column]
        ax.bar(
            x + (index - 1.5) * width,
            [100 * values.loc[layer] for layer in layers],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xticks(x, [f"L{layer}" for layer in layers])
    ax.set_ylabel("Mean geometry statistic (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1, 1]
    full_methods = [
        "Whole visual additive state",
        "Exact mass boundary + tail",
        "Exact effect boundary + tail",
    ]
    width = 0.24
    for index, method in enumerate(full_methods):
        frame = layer_metrics[layer_metrics["method"] == method].set_index(
            "layer_index"
        )
        ax.bar(
            x + (index - 1) * width,
            [100 * frame.loc[layer, "full_relative_l2"] for layer in layers],
            width=width,
            label=method,
            color=method_color(method),
            edgecolor="black",
            linewidth=0.35,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5% signal")
    ax.axhline(0.25, color="black", linestyle=":", linewidth=1, label="0.25% oracle")
    ax.set_xticks(x, [f"L{layer}" for layer in layers])
    ax.set_ylabel("Mean full output error (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=12)

    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            FIGURE_DIR / f"conditional_redundancy_state_gate.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    draw()
