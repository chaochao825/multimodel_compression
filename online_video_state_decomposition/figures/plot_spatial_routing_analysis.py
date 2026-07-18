from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "paper"
    / "results"
    / "probe_mvp"
    / "spatial_routing_analysis_20260718"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    methods = read_csv(data_dir / "method_summary.csv")
    tasks = read_csv(data_dir / "routing_by_task.csv")
    samples = read_csv(data_dir / "sample_outcomes.csv")
    configure()

    colors = {
        "Full state": "#264653",
        "Low-rank only": "#9A8C73",
        "Fixed sparse s4": "#E76F51",
        "Spatial grid 2x2": "#2A9D8F",
        "Routed grid/sparse": "#E9C46A",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))

    labels = [row["label"] for row in methods]
    accuracy = [100.0 * float(row["accuracy"]) for row in methods]
    positions = np.arange(len(methods))
    axes[0, 0].bar(
        positions,
        accuracy,
        color=[colors[label] for label in labels],
        edgecolor="white",
        linewidth=0.7,
    )
    axes[0, 0].set_xticks(
        positions,
        [label.replace(" ", "\n", 1) for label in labels],
    )
    axes[0, 0].set_ylabel("Accuracy on reused 200 samples (%)")
    axes[0, 0].set_ylim(max(0.0, min(accuracy) - 4.0), max(accuracy) + 3.0)
    axes[0, 0].grid(axis="y", alpha=0.22)
    for position, row, value in zip(positions, methods, accuracy, strict=True):
        axes[0, 0].text(position, value + 0.35, f"{value:.1f}", ha="center")
        axes[0, 0].text(
            position,
            axes[0, 0].get_ylim()[0] + 0.35,
            f"+{row['better_samples']} / -{row['worse_samples']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0, 0].set_title("(a) Accuracy and paired changes vs full", loc="left")

    width = 0.36
    steady = [float(row["steady_state_mib"]) for row in methods]
    cold = [float(row["cold_start_mib"]) for row in methods]
    axes[0, 1].barh(
        positions + width / 2,
        steady,
        height=width,
        color="#2A9D8F",
        label="Steady per-stream",
    )
    axes[0, 1].barh(
        positions - width / 2,
        cold,
        height=width,
        color="#E9C46A",
        label="Cold start + shared codec",
    )
    axes[0, 1].set_yticks(positions, labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Accounted state (MiB)")
    axes[0, 1].grid(axis="x", alpha=0.22)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title("(b) Amortized and cold-start storage", loc="left")

    task_labels = [row["task"].replace("_", " ") for row in tasks]
    task_positions = np.arange(len(tasks))
    grid_rate = [100.0 * float(row["grid_frame_rate"]) for row in tasks]
    sparse_rate = [100.0 * float(row["sparse_frame_rate"]) for row in tasks]
    axes[1, 0].barh(
        task_positions,
        grid_rate,
        color="#2A9D8F",
        label="2x2 grid",
    )
    axes[1, 0].barh(
        task_positions,
        sparse_rate,
        left=grid_rate,
        color="#E76F51",
        label="top-4 sparse",
    )
    axes[1, 0].set_yticks(task_positions, task_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_xlabel("Routed pool frames (%)")
    axes[1, 0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    axes[1, 0].grid(axis="x", alpha=0.22)
    axes[1, 0].set_title("(c) Causal branch use by task", loc="left", y=1.08)

    fixed_error = np.array(
        [100.0 * float(row["fixed_selected_error"]) for row in samples]
    )
    grid_error = np.array(
        [100.0 * float(row["grid_selected_error"]) for row in samples]
    )
    routed_grid_rate = np.array(
        [float(row["routed_grid_frames"]) / 16.0 for row in samples]
    )
    correctness_disagreement = np.array(
        [int(row["fixed_correct"]) != int(row["grid_correct"]) for row in samples]
    )
    scatter = axes[1, 1].scatter(
        fixed_error[~correctness_disagreement],
        grid_error[~correctness_disagreement],
        c=routed_grid_rate[~correctness_disagreement],
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        s=24,
        alpha=0.62,
        linewidth=0,
    )
    axes[1, 1].scatter(
        fixed_error[correctness_disagreement],
        grid_error[correctness_disagreement],
        c=routed_grid_rate[correctness_disagreement],
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        marker="D",
        s=58,
        edgecolor="#D1495B",
        linewidth=1.1,
        label="Fixed/grid correctness differs",
    )
    low = min(float(fixed_error.min()), float(grid_error.min()))
    high = max(float(fixed_error.max()), float(grid_error.max()))
    axes[1, 1].plot([low, high], [low, high], linestyle="--", color="#555555")
    axes[1, 1].set_xlabel("Fixed-s4 selected error (%)")
    axes[1, 1].set_ylabel("Grid selected error (%)")
    axes[1, 1].grid(alpha=0.18)
    axes[1, 1].legend(frameon=False, fontsize=8)
    colorbar = figure.colorbar(scatter, ax=axes[1, 1], pad=0.02)
    colorbar.set_label("Grid-mode fraction")
    axes[1, 1].set_title("(d) Error does not determine task outcome", loc="left")

    figure.suptitle(
        "Equal-budget spatial/sparse routing (exploratory, reused set)",
        fontsize=13,
        weight="bold",
        y=1.01,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            data_dir / f"spatial_routing_analysis.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
