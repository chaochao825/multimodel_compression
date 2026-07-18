from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "paper" / "results" / "probe_mvp"
OUT_DIR = RESULT_ROOT / "competitiveness_loss_redesign_20260718"
FORMAL_PREDICTIONS = (
    RESULT_ROOT
    / "mvbench_compressed_feature_confirmation_rank256_20260718_v1"
    / "aggregate"
    / "predictions.csv"
)
TARGETED_PREDICTIONS = (
    RESULT_ROOT
    / "mvbench_spatial_residual_targeted_8_20260718_v2"
    / "aggregate"
    / "predictions.csv"
)
BUDGET_PREDICTIONS = (
    RESULT_ROOT
    / "mvbench_adaptive_residual_budget_sweep_state_change_0157_20260718_v1"
    / "aggregate"
    / "predictions.csv"
)
SPATIAL_PREDICTIONS = (
    RESULT_ROOT
    / "mvbench_spatial_residual_state_change_0157_20260718_v1"
    / "aggregate"
    / "predictions.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = [
        ("Recent / full", "exact_recent", "full"),
        ("Query / full", "learned_recent_query_topk", "full"),
        ("Query / r256+s4", "learned_recent_query_topk", "pca_r256_s4"),
    ]
    output = []
    for label, selector, variant in specs:
        selected = [
            row
            for row in rows
            if row["selection_policy"] == selector
            and row["memory_variant"] == variant
        ]
        output.append(
            {
                "label": label,
                "selector": selector,
                "variant": variant,
                "samples": len(selected),
                "correct": sum(int(row["correct"]) for row in selected),
                "accuracy": sum(int(row["correct"]) for row in selected)
                / len(selected),
                "state_mib": sum(
                    int(row["selection_state_proxy_bytes"])
                    for row in selected
                )
                / len(selected)
                / 2**20,
            }
        )
    return output


def targeted_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    labels = {
        "full": "Full",
        "pca_r256_s4": "Fixed s4",
        "pca_r256_global_k64": "Global k64",
        "pca_r256_temporal_k64": "Causal temporal",
        "pca_r256_mean1_s3": "Mean + sparse",
        "pca_r256_grid2x2": "Spatial 2x2",
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["selection_policy"] == "learned_recent_query_topk":
            grouped[row["memory_variant"]].append(row)
    output = []
    for variant, label in labels.items():
        selected = grouped[variant]
        output.append(
            {
                "label": label,
                "variant": variant,
                "samples": len(selected),
                "accuracy": sum(int(row["correct"]) for row in selected)
                / len(selected),
                "mean_selected_error": sum(
                    float(row["selected_reconstruction_relative_error"])
                    for row in selected
                )
                / len(selected),
                "state_mib": sum(
                    int(row["selection_state_proxy_bytes"])
                    for row in selected
                )
                / len(selected)
                / 2**20,
            }
        )
    return output


def budget_rows(
    historical_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    merged = {row["memory_variant"]: row for row in historical_rows}
    merged.update({row["memory_variant"]: row for row in current_rows})
    output = []
    for row in merged.values():
        variant = row["memory_variant"]
        if variant == "full":
            family = "Full"
            short = "full"
        elif "_grid2x2" in variant:
            family = "Spatial 2x2"
            short = "grid2"
        elif "_mean1_s3" in variant:
            family = "Mean + sparse"
            short = "mean+s3"
        elif "_global_" in variant:
            family = "Global energy"
            short = "g" + variant.rsplit("k", 1)[-1]
        elif "_temporal_" in variant:
            family = (
                "Causal temporal"
                if variant.endswith("k64")
                else "Temporal novelty*"
            )
            short = "t" + variant.rsplit("k", 1)[-1]
        else:
            family = "Fixed quota"
            short = variant.rsplit("s", 1)[-1]
            short = "s" + short
        output.append(
            {
                "family": family,
                "variant": variant,
                "short_label": short,
                "correct": int(row["correct"]),
                "state_mib": int(row["selection_state_proxy_bytes"]) / 2**20,
                "selected_error": float(
                    row["selected_reconstruction_relative_error"]
                ),
            }
        )
    return output


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    stages: list[dict[str, object]],
    targeted: list[dict[str, object]],
    budget: list[dict[str, object]],
) -> None:
    configure()
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#B279A2",
        "#72B7B2",
        "#E45756",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))

    stage_accuracy = [100.0 * float(row["accuracy"]) for row in stages]
    axes[0].bar(range(len(stages)), stage_accuracy, color=colors[:3])
    axes[0].set_xticks(
        range(len(stages)),
        [str(row["label"]).replace(" / ", "\n") for row in stages],
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(40, 55)
    axes[0].grid(axis="y", alpha=0.22)
    for index, value in enumerate(stage_accuracy):
        axes[0].text(index, value + 0.35, f"{value:.1f}", ha="center")
    axes[0].text(-0.12, 1.03, "(a)", transform=axes[0].transAxes, weight="bold")

    target_accuracy = [100.0 * float(row["accuracy"]) for row in targeted]
    target_error = [
        100.0 * float(row["mean_selected_error"]) for row in targeted
    ]
    positions = range(len(targeted))
    axes[1].bar(positions, target_accuracy, color=colors)
    axes[1].set_xticks(
        positions,
        [str(row["label"]).replace(" ", "\n") for row in targeted],
    )
    axes[1].set_ylabel("Accuracy on 8 targeted samples (%)")
    axes[1].set_ylim(75, 104)
    axes[1].grid(axis="y", alpha=0.22)
    error_axis = axes[1].twinx()
    error_axis.plot(positions, target_error, color="#222222", marker="o")
    error_axis.set_ylabel("Mean selected error (%)")
    error_axis.set_ylim(0, 8)
    axes[1].text(-0.12, 1.03, "(b)", transform=axes[1].transAxes, weight="bold")

    family_colors = {
        "Fixed quota": "#4C78A8",
        "Global energy": "#54A24B",
        "Temporal novelty*": "#F58518",
        "Causal temporal": "#B279A2",
        "Mean + sparse": "#72B7B2",
        "Spatial 2x2": "#E45756",
        "Full": "#777777",
    }
    for family in family_colors:
        selected = sorted(
            (row for row in budget if row["family"] == family),
            key=lambda row: float(row["state_mib"]),
        )
        if not selected:
            continue
        axes[2].plot(
            [float(row["state_mib"]) for row in selected],
            [100.0 * float(row["selected_error"]) for row in selected],
            color=family_colors[family],
            linewidth=1.4,
            alpha=0.75,
            label=family,
        )
        for row in selected:
            correct = int(row["correct"])
            axes[2].scatter(
                float(row["state_mib"]),
                100.0 * float(row["selected_error"]),
                marker="o" if correct else "X",
                s=62,
                color="#2A9D8F" if correct else "#C44E52",
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            axes[2].annotate(
                str(row["short_label"]),
                (
                    float(row["state_mib"]),
                    100.0 * float(row["selected_error"]),
                ),
                xytext={
                    "s4": (-9, -13),
                    "s8": (-9, -13),
                    "s16": (-9, -13),
                    "g64": (4, 5),
                    "g128": (4, 5),
                    "g256": (4, 5),
                    "t64": (4, 9),
                    "t128": (4, -14),
                    "t256": (4, 9),
                    "mean+s3": (4, -14),
                    "grid2": (4, 8),
                    "full": (4, 5),
                }.get(str(row["short_label"]), (4, 4)),
                textcoords="offset points",
                fontsize=8,
            )
    axes[2].set_xlabel("Persistent state (MiB)")
    axes[2].set_ylabel("Selected reconstruction error (%)")
    axes[2].grid(alpha=0.22)
    handles, labels = axes[2].get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2A9D8F",
                markeredgecolor="white",
                markersize=7,
                label="Correct",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#C44E52",
                markeredgecolor="white",
                markersize=7,
                label="Wrong",
            ),
        ]
    )
    labels.extend(["Correct", "Wrong"])
    axes[2].legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper right",
        ncol=2,
    )
    axes[2].text(-0.12, 1.03, "(c)", transform=axes[2].transAxes, weight="bold")

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT_DIR / f"competitiveness_loss_redesign.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = stage_rows(read_csv(FORMAL_PREDICTIONS))
    targeted = targeted_rows(read_csv(TARGETED_PREDICTIONS))
    budget = budget_rows(
        read_csv(BUDGET_PREDICTIONS),
        read_csv(SPATIAL_PREDICTIONS),
    )
    write_csv(OUT_DIR / "stage_decomposition.csv", stages)
    write_csv(OUT_DIR / "targeted_equal_budget.csv", targeted)
    write_csv(OUT_DIR / "hard_case_budget_probe.csv", budget)
    plot(stages, targeted, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
