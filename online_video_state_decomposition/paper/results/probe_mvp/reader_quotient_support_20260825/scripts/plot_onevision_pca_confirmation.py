from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "full": "#4C78A8",
    "pca": "#54A24B",
    "harm": "#E45756",
    "benefit": "#72B7B2",
    "wrong_to_wrong": "#F2CF5B",
    "match": "#9D9DA0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    paired = read_csv(args.analysis_dir / "paired_samples.csv")
    tasks = read_csv(args.analysis_dir / "task_summary.csv")
    summary = json.loads(
        (args.analysis_dir / "summary.json").read_text(encoding="utf-8")
    )["metrics"]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    outcome_rows = {
        "Agreement": [row for row in paired if int(row["prediction_match"]) == 1],
        "Harmful": [row for row in paired if int(row["harmful_flip"]) == 1],
        "Beneficial": [row for row in paired if int(row["beneficial_flip"]) == 1],
        "Wrong-to-wrong": [
            row
            for row in paired
            if int(row["prediction_match"]) == 0
            and int(row["harmful_flip"]) == 0
            and int(row["beneficial_flip"]) == 0
        ],
    }
    plot_rows = []
    for row in tasks:
        plot_rows.append(
            {
                "panel": "task_accuracy",
                "label": row["task"],
                "full": float(row["reference_accuracy"]),
                "pca": float(row["candidate_accuracy"]),
            }
        )
    for label, rows in outcome_rows.items():
        plot_rows.append(
            {
                "panel": "prediction_outcome",
                "label": label,
                "count": len(rows),
                "candidate_kl_mean": float(
                    np.mean([float(row["candidate_kl"]) for row in rows])
                ),
                "feature_relative_l2_mean": float(
                    np.mean([float(row["feature_relative_l2"]) for row in rows])
                ),
            }
        )
    with args.output_prefix.with_name(
        f"{args.output_prefix.name}_data.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in plot_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    axis = axes[0, 0]
    task_labels = [row["task"].replace("_", "\n") for row in tasks]
    positions = np.arange(len(tasks))
    width = 0.37
    axis.bar(
        positions - width / 2,
        [100.0 * float(row["reference_accuracy"]) for row in tasks],
        width,
        color=COLORS["full"],
        label="Full",
    )
    axis.bar(
        positions + width / 2,
        [100.0 * float(row["candidate_accuracy"]) for row in tasks],
        width,
        color=COLORS["pca"],
        label="PCA-r456",
    )
    axis.set_xticks(positions, task_labels)
    axis.set_ylabel("Candidate accuracy (%)")
    axis.legend(frameon=False, ncol=2)
    axis.text(-0.12, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[0, 1]
    flip_labels = ["Harmful", "Beneficial", "Wrong-to-wrong"]
    flip_colors = [COLORS["harm"], COLORS["benefit"], COLORS["wrong_to_wrong"]]
    flip_counts = [len(outcome_rows[label]) for label in flip_labels]
    axis.bar(flip_labels, flip_counts, color=flip_colors)
    for index, count in enumerate(flip_counts):
        axis.text(index, count + 0.2, str(count), ha="center", va="bottom")
    axis.set_ylabel("Changed predictions (count / 500)")
    axis.text(
        0.98,
        0.95,
        f"Agreement: {100.0 * float(summary['prediction_agreement']):.1f}%\n"
        f"Gate: 98.0%",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.text(-0.12, 1.04, "b", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 0]
    distributions = [
        np.asarray([float(row["candidate_kl"]) for row in outcome_rows[label]])
        + 1e-8
        for label in outcome_rows
    ]
    box = axis.boxplot(
        distributions,
        tick_labels=list(outcome_rows),
        patch_artist=True,
        showfliers=True,
    )
    box_colors = [
        COLORS["match"],
        COLORS["harm"],
        COLORS["benefit"],
        COLORS["wrong_to_wrong"],
    ]
    for patch, color in zip(box["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    axis.set_yscale("log")
    axis.set_ylabel("Candidate KL (log scale)")
    axis.tick_params(axis="x", rotation=15)
    axis.text(-0.12, 1.04, "c", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 1]
    dense_bytes = max(int(row["dense_native_feature_bytes"]) for row in paired)
    compressed_bytes = max(int(row["native_feature_state_bytes"]) for row in paired)
    bars = axis.bar(
        ["Full native\nstate", "PCA-r456\nstate"],
        [dense_bytes / 2**20, compressed_bytes / 2**20],
        color=[COLORS["full"], COLORS["pca"]],
    )
    for bar, value in zip(bars, (dense_bytes, compressed_bytes), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value / 2**20:.2f} MiB",
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Persistent tensor payload (MiB)")
    axis.text(
        0.98,
        0.92,
        f"{dense_bytes / compressed_bytes:.2f}x smaller\n"
        f"Accuracy: {100.0 * float(summary['reference_accuracy']):.1f}% -> "
        f"{100.0 * float(summary['candidate_accuracy']):.1f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.text(-0.12, 1.04, "d", transform=axis.transAxes, fontweight="bold", fontsize=12)

    for axis in axes.flat:
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.7)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    svg_path = args.output_prefix.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
