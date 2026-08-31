from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "online_video_state_decomposition"
    / "paper"
    / "results"
    / "probe_mvp"
    / "reader_quotient_support_20260825"
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
)
EXP050 = RESULT_ROOT / "control_variate_support_state_capacity_dev_v5_valid"
EXP051 = RESULT_ROOT / "joint_control_variate_support_state_capacity_dev_v1"
OUTPUT_STEM = ROOT / "figures" / "support_state_factorial_20260901"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_plot_data(rows: list[dict[str, object]]) -> None:
    with OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    exp050 = read_json(EXP050 / "summary.json")
    exp051 = read_json(EXP051 / "summary.json")
    methods = exp051["method_summaries"]

    method_order = [
        "exact_only_mass",
        "state_only",
        "independent_mass_correction",
        "independent_residual_correction",
        "joint_mass_correction",
        "joint_residual_correction",
    ]
    labels = [
        "Exact-only\nmass",
        "State-only",
        "Mass-trained\n+ mass",
        "Mass-trained\n+ residual",
        "Residual-trained\n+ mass",
        "Residual-trained\n+ residual",
    ]
    colors = ["#6B7280", "#9CA3AF", "#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    visual_mean = np.asarray(
        [float(methods[name]["visual_mean"]) * 100.0 for name in method_order]
    )

    frozen = exp050["method_summaries"]
    risk_matrix = np.asarray(
        [
            [
                float(frozen["independent_mass_correction"]["visual_risk_mean"]),
                float(frozen["joint_residual_oracle"]["visual_risk_mean"]),
            ],
            [
                float(methods["independent_mass_correction"]["visual_risk_mean"]),
                float(methods["independent_residual_correction"]["visual_risk_mean"]),
            ],
            [
                float(methods["joint_mass_correction"]["visual_risk_mean"]),
                float(methods["joint_residual_correction"]["visual_risk_mean"]),
            ],
        ]
    )

    tidy_rows: list[dict[str, object]] = []
    for name in method_order:
        summary = methods[name]
        tidy_rows.append(
            {
                "experiment": "EXP-051",
                "method": name,
                "visual_mean_percent": float(summary["visual_mean"]) * 100.0,
                "visual_p95_percent": float(summary["visual_p95"]) * 100.0,
                "visual_worst_percent": float(summary["visual_worst"]) * 100.0,
                "visual_risk_mean": float(summary["visual_risk_mean"]),
            }
        )
    write_plot_data(tidy_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.2), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(method_order))
    bars = ax.barh(y, visual_mean, color=colors, edgecolor="black", linewidth=0.45)
    ax.set_xscale("log")
    ax.set_xlabel("Mean visual relative L2 (%)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for threshold, style, text in ((0.5, "--", "mean gate"), (2.0, ":", "worst gate")):
        ax.axvline(threshold, color="#333333", linestyle=style, linewidth=1.0)
        ax.text(threshold * 1.06, 5.38, text, ha="left", va="bottom", fontsize=8)
    for bar, value in zip(bars, visual_mean, strict=True):
        ax.text(
            value * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7.5,
        )
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1]
    image = ax.imshow(risk_matrix, cmap="YlOrRd", norm="log", aspect="auto")
    ax.set_xticks([0, 1], ["Mass support", "Residual support"], rotation=18)
    ax.set_yticks(
        [0, 1, 2], ["Frozen whole-state", "Mass-support trained", "Residual-support trained"]
    )
    ax.set_xlabel("Exact-page selection rule")
    ax.set_ylabel("State training objective")
    for row in range(risk_matrix.shape[0]):
        for col in range(risk_matrix.shape[1]):
            value = risk_matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value > 0.02 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean visual risk")
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[2]
    development = read_csv(EXP051 / "development_rows.csv")
    layers = ("0", "13", "27")
    baseline_mean = []
    joint_mean = []
    risk_change = []
    for layer in layers:
        baseline_rows = [
            row
            for row in development
            if row["layer_index"] == layer
            and row["method"] == "independent_mass_correction"
        ]
        joint_rows = [
            row
            for row in development
            if row["layer_index"] == layer
            and row["method"] == "joint_residual_correction"
        ]
        baseline_mean.append(
            np.mean([float(row["visual_relative_l2"]) for row in baseline_rows]) * 100.0
        )
        joint_mean.append(
            np.mean([float(row["visual_relative_l2"]) for row in joint_rows]) * 100.0
        )
        baseline_risk = np.mean([float(row["visual_risk"]) for row in baseline_rows])
        joint_risk = np.mean([float(row["visual_risk"]) for row in joint_rows])
        risk_change.append(1.0 - joint_risk / baseline_risk)
    x = np.arange(len(layers))
    width = 0.36
    ax.bar(
        x - width / 2,
        baseline_mean,
        width,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.45,
        label="Mass-trained + mass",
    )
    ax.bar(
        x + width / 2,
        joint_mean,
        width,
        color="#E69F00",
        edgecolor="black",
        linewidth=0.45,
        label="Residual-trained + residual",
    )
    ax.set_xticks(x, [f"Layer {layer}" for layer in layers])
    ax.set_ylabel("Mean visual relative L2 (%)")
    ax.legend(frameon=False, fontsize=7.5)
    for index, change in enumerate(risk_change):
        ax.text(
            index,
            max(baseline_mean[index], joint_mean[index]) + 0.25,
            f"risk {change:+.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(max(baseline_mean), max(joint_mean)) * 1.22)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[3]
    history = read_csv(EXP051 / "training_history.csv")
    layer_colors = {"0": "#0072B2", "13": "#009E73", "27": "#D55E00"}
    for layer in ("0", "13", "27"):
        for mode, linestyle in (("mass", "-"), ("residual", "--")):
            selected = [
                row
                for row in history
                if row["layer_index"] == layer and row["support_mode"] == mode
            ]
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row["train_visual_error"]) * 100.0 for row in selected],
                color=layer_colors[layer],
                linestyle=linestyle,
                linewidth=1.6,
                marker="o",
                markersize=2.7,
                label=f"L{layer}, {mode}",
            )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training visual relative L2 (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=12)

    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
