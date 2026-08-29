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
    "positive": "#59A14F",
    "negative": "#E15759",
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
    durations = read_csv(args.analysis_dir / "duration_summary.csv")
    domains = read_csv(args.analysis_dir / "domain_summary.csv")
    summary = json.loads(
        (args.analysis_dir / "summary.json").read_text(encoding="utf-8")
    )["metrics"]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    harmful = [row for row in paired if int(row["harmful_flip"]) == 1]
    beneficial = [row for row in paired if int(row["beneficial_flip"]) == 1]
    wrong_to_wrong = [
        row
        for row in paired
        if int(row["prediction_match"]) == 0
        and int(row["harmful_flip"]) == 0
        and int(row["beneficial_flip"]) == 0
    ]
    plot_rows = []
    for row in durations:
        plot_rows.append(
            {
                "panel": "duration_accuracy",
                "label": row["duration"],
                "samples": int(row["samples"]),
                "full": float(row["reference_accuracy"]),
                "pca": float(row["candidate_accuracy"]),
            }
        )
    for row in domains:
        plot_rows.append(
            {
                "panel": "domain_delta",
                "label": row["domain"],
                "samples": int(row["samples"]),
                "accuracy_delta": float(row["accuracy_delta"]),
            }
        )
    for label, rows in (
        ("Harmful", harmful),
        ("Beneficial", beneficial),
        ("Wrong-to-wrong", wrong_to_wrong),
    ):
        plot_rows.append(
            {
                "panel": "prediction_outcome",
                "label": label,
                "samples": len(rows),
                "candidate_kl_mean": float(
                    np.mean([float(row["candidate_kl"]) for row in rows])
                )
                if rows
                else 0.0,
            }
        )
    data_path = args.output_prefix.with_name(f"{args.output_prefix.name}_data.csv")
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in plot_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    axis = axes[0, 0]
    positions = np.arange(len(durations))
    width = 0.37
    axis.bar(
        positions - width / 2,
        [100.0 * float(row["reference_accuracy"]) for row in durations],
        width,
        color=COLORS["full"],
        label="Full",
    )
    axis.bar(
        positions + width / 2,
        [100.0 * float(row["candidate_accuracy"]) for row in durations],
        width,
        color=COLORS["pca"],
        label="PCA-r456",
    )
    axis.set_xticks(positions, [row["duration"].title() for row in durations])
    axis.set_ylabel("Candidate accuracy (%)")
    axis.legend(frameon=False, ncol=2)
    axis.text(-0.12, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[0, 1]
    flip_labels = ["Harmful", "Beneficial", "Wrong-to-wrong"]
    flip_counts = [len(harmful), len(beneficial), len(wrong_to_wrong)]
    axis.bar(
        flip_labels,
        flip_counts,
        color=[COLORS["harm"], COLORS["benefit"], COLORS["wrong_to_wrong"]],
    )
    for index, count in enumerate(flip_counts):
        axis.text(index, count + 0.2, str(count), ha="center", va="bottom")
    axis.set_ylabel("Changed predictions (count / 600)")
    axis.text(
        0.03,
        0.95,
        f"Agreement: {100.0 * float(summary['prediction_agreement']):.1f}%\n"
        "Gate: 98.0%",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.text(-0.12, 1.04, "b", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 0]
    ordered_domains = sorted(
        domains,
        key=lambda row: float(row["accuracy_delta"]),
    )
    domain_labels = [row["domain"] for row in ordered_domains]
    domain_deltas = [100.0 * float(row["accuracy_delta"]) for row in ordered_domains]
    axis.barh(
        domain_labels,
        domain_deltas,
        color=[
            COLORS["positive"] if value >= 0.0 else COLORS["negative"]
            for value in domain_deltas
        ],
    )
    count_x = max(0.0, max(domain_deltas)) + 0.12
    for index, row in enumerate(ordered_domains):
        axis.text(
            count_x,
            index,
            f"n={int(row['samples'])}",
            ha="left",
            va="center",
            fontsize=8,
        )
    axis.axvline(0.0, color="#555555", linewidth=0.8)
    axis.set_xlabel("PCA minus Full accuracy (percentage points)")
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
