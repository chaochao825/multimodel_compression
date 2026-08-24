from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEAK_DIR = ROOT / "results" / "understanding_reader_quotient_support_oracle" / "analysis_v2"
STRONG_DIR = ROOT / "results" / "understanding_onevision_reader_quotient_replication" / "analysis_v2"
OUTPUT_STEM = Path(__file__).with_name("onevision_reader_quotient_replication")
COLORS = {"Fisher": "#0072B2", "Mixed": "#D55E00"}


def variant_map(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["variant"]: row for row in payload["variant_summary"]}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    weak = variant_map(WEAK_DIR / "summary.json")
    strong = variant_map(STRONG_DIR / "summary.json")
    tasks = read_csv(STRONG_DIR / "task_summary.csv")
    paired = read_csv(STRONG_DIR / "paired_samples.csv")

    figure_rows = []
    for reader, summaries in (("LLaVA-v1.5", weak), ("LLaVA-OneVision", strong)):
        for label, variant in (("Fisher", "fisher_s4"), ("Mixed", "mixed_s4")):
            row = summaries[variant]
            figure_rows.append(
                {
                    "reader": reader,
                    "method": label,
                    "aggregate_kl_reduction": row["aggregate_candidate_kl_reduction"],
                    "ci95_low": row["reduction_ci95_low"],
                    "ci95_high": row["reduction_ci95_high"],
                    "p95_ratio": row["candidate_kl_p95_ratio"],
                    "top1_delta": row["candidate_top1_match_delta"],
                    "positive_tasks": row["positive_task_count"],
                }
            )
    with OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_rows[0]))
        writer.writeheader()
        writer.writerows(figure_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    readers = ("LLaVA-v1.5", "LLaVA-OneVision")
    x = np.arange(len(readers))
    width = 0.34
    for offset, label in ((-width / 2, "Fisher"), (width / 2, "Mixed")):
        selected = [
            next(row for row in figure_rows if row["reader"] == reader and row["method"] == label)
            for reader in readers
        ]
        values = np.asarray([float(row["aggregate_kl_reduction"]) for row in selected])
        lower = values - np.asarray([float(row["ci95_low"]) for row in selected])
        upper = np.asarray([float(row["ci95_high"]) for row in selected]) - values
        ax.bar(x + offset, values * 100, width, color=COLORS[label], label=label)
        ax.errorbar(
            x + offset,
            values * 100,
            yerr=np.vstack((lower, upper)) * 100,
            fmt="none",
            ecolor="#222222",
            elinewidth=1,
            capsize=3,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(25, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, readers)
    ax.set_ylabel("Aggregate candidate-KL reduction (%)")
    ax.set_ylim(-105, 105)
    ax.legend(frameon=False, ncols=2, loc="lower left")
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = axes[0, 1]
    for offset, label in ((-width / 2, "Fisher"), (width / 2, "Mixed")):
        selected = [
            next(row for row in figure_rows if row["reader"] == reader and row["method"] == label)
            for reader in readers
        ]
        values = np.asarray([float(row["p95_ratio"]) for row in selected])
        bars = ax.bar(x + offset, values, width, color=COLORS[label], label=label)
        for bar, row in zip(bars, selected, strict=True):
            annotation = f"{int(row['positive_tasks'])}/5 tasks\n{float(row['top1_delta']):+.0%} top-1"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                annotation,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(1, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, readers)
    ax.set_ylabel("P95 candidate-KL ratio (lower is better)")
    ax.set_ylim(0, 1.15)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = axes[1, 0]
    task_names = sorted({row["task"] for row in tasks})
    y = np.arange(len(task_names))
    height = 0.34
    for offset, label, variant in (
        (-height / 2, "Fisher", "fisher_s4"),
        (height / 2, "Mixed", "mixed_s4"),
    ):
        values = [
            100
            * float(
                next(
                    row
                    for row in tasks
                    if row["task"] == task and row["variant"] == variant
                )["aggregate_reduction"]
            )
            for task in task_names
        ]
        ax.barh(y + offset, values, height, color=COLORS[label], label=label)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, [name.replace("_", " ") for name in task_names])
    ax.set_xlabel("Task-level candidate-KL reduction (%)")
    ax.set_xlim(-260, 100)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=13)

    ax = axes[1, 1]
    mixed = [row for row in paired if row["variant"] == "mixed_s4"]
    epsilon = 1e-7
    baseline = np.asarray([max(float(row["euclidean_candidate_kl"]), epsilon) for row in mixed])
    candidate = np.asarray([max(float(row["candidate_kl"]), epsilon) for row in mixed])
    ax.scatter(baseline, candidate, color=COLORS["Mixed"], edgecolor="white", linewidth=0.5)
    limits = (epsilon, max(baseline.max(), candidate.max()) * 1.6)
    ax.plot(limits, limits, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Euclidean s4 candidate KL")
    ax.set_ylabel("Mixed s4 candidate KL")
    for sample_id in ("fine_grained_pose_0061", "object_interaction_0088"):
        row = next(item for item in mixed if item["sample_id"] == sample_id)
        bx = max(float(row["euclidean_candidate_kl"]), epsilon)
        cy = max(float(row["candidate_kl"]), epsilon)
        label = "dominant gain" if sample_id.endswith("0061") else "top-1 flip"
        ax.annotate(
            label,
            (bx, cy),
            xytext=(7, -12 if label == "dominant gain" else 8),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.7},
        )
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=13)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
