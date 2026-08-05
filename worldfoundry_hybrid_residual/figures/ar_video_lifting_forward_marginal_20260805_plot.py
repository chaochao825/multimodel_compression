from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "lifting_forward_marginal_l14_heldout_20260805a"
OUTPUT = ROOT / "figures" / "ar_video_lifting_forward_marginal_20260805"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]]) -> tuple[float, float]:
    numerator = sum(float(row["numerator_sq"]) for row in rows)
    denominator = sum(float(row["denominator_sq"]) for row in rows)
    pooled = 100.0 * math.sqrt(numerator / max(denominator, 1e-24))
    worst = 100.0 * max(float(row["relative_av_l2"]) for row in rows)
    return pooled, worst


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    debug = json.loads((RESULT / "debug.json").read_text(encoding="utf-8"))
    metrics = read_csv(RESULT / "metrics.csv")
    correction = "adaptive_rank16_bf16_coeff"
    methods = [
        ("kv_detail_energy", "K/V energy"),
        ("adaptive_tail_singleton_selector", "Singleton AV"),
        ("adaptive_tail_forward_marginal_greedy", "Forward marginal"),
    ]

    trajectory_rows: list[dict[str, object]] = []
    for capture in debug:
        prompt_id = str(capture["capture"][0])
        for row in capture["forward_marginal_trajectory"]:
            trajectory_rows.append(
                {
                    "prompt_id": prompt_id,
                    "retained_blocks": int(row["step"]),
                    "aggregate_av_l2_percent": 100.0
                    * float(row["aggregate_relative_av_l2"]),
                    "selected_index": int(row["selected_index"]),
                    "evaluated_candidates": int(row["evaluated_candidates"]),
                }
            )
    write_csv(OUTPUT / "forward_marginal_trajectory.csv", trajectory_rows)

    final_rows: list[dict[str, object]] = []
    for method, label in methods:
        selected = [
            row
            for row in metrics
            if row["method"] == method and row["correction"] == correction
        ]
        pooled, worst = aggregate(selected)
        final_rows.append(
            {
                "method": method,
                "label": label,
                "aggregate_av_l2_percent": pooled,
                "worst_head_av_l2_percent": worst,
                "logical_bf16_cache_compression_ratio": min(
                    float(row["logical_bf16_cache_compression_ratio"])
                    for row in selected
                ),
            }
        )
    write_csv(OUTPUT / "final_comparison.csv", final_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    colors = {
        "test_nonrigid_s3": "#2F6B8A",
        "valid_multi_object_s2": "#D1495B",
    }
    labels = {
        "test_nonrigid_s3": "Non-rigid",
        "valid_multi_object_s2": "Multi-object",
    }
    axis = axes[0]
    for prompt_id in sorted({str(row["prompt_id"]) for row in trajectory_rows}):
        selected = [
            row for row in trajectory_rows if row["prompt_id"] == prompt_id
        ]
        axis.plot(
            [int(row["retained_blocks"]) for row in selected],
            [float(row["aggregate_av_l2_percent"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[prompt_id],
            label=labels[prompt_id],
        )
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1.2, label="0.5% gate")
    axis.set_xlabel("Retained padded 64-token detail blocks")
    axis.set_ylabel("Adaptive rank-16 AV error (%)")
    axis.set_yscale("log")
    axis.set_xlim(1, 24)
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(frameon=False)
    axis.text(-0.12, 1.03, "A", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1]
    positions = list(range(len(final_rows)))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        [float(row["aggregate_av_l2_percent"]) for row in final_rows],
        width,
        color="#2F6B8A",
        label="Aggregate",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [float(row["worst_head_av_l2_percent"]) for row in final_rows],
        width,
        color="#D1495B",
        label="Worst head",
    )
    axis.axhline(0.5, color="#2F6B8A", linestyle=":", linewidth=1)
    axis.axhline(1.0, color="#D1495B", linestyle=":", linewidth=1)
    axis.set_xticks(positions, [str(row["label"]) for row in final_rows])
    axis.set_ylabel("BF16-coefficient AV error (%)")
    axis.set_yscale("log")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    axis.text(-0.12, 1.03, "B", transform=axis.transAxes, fontweight="bold", fontsize=12)

    figure.suptitle("Layer-14 forward-marginal detail search", fontsize=12)
    figure.savefig(OUTPUT / "ar_video_lifting_forward_marginal_20260805.png", dpi=240)
    figure.savefig(OUTPUT / "ar_video_lifting_forward_marginal_20260805.pdf")


if __name__ == "__main__":
    main()
