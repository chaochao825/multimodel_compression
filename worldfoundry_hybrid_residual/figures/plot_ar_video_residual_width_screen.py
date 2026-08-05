"""Plot the causal BCCB and residual-width episodic-write screens."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "kv_deviation": "K/V deviation",
    "dense_attention_mass_oracle": "Dense mass oracle",
    "value_leverage_oracle": "Value leverage oracle",
    "residual_width_singleton_oracle": "Residual-width oracle",
}

COLORS = {
    "kv_deviation": "#7A7A73",
    "dense_attention_mass_oracle": "#D18C35",
    "value_leverage_oracle": "#2A7F76",
    "residual_width_singleton_oracle": "#275D8C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bccb-result",
        type=Path,
        default=Path("results/causal_bccb_primary_full_20260805a"),
    )
    parser.add_argument(
        "--width-result",
        type=Path,
        default=Path("results/residual_width_full_20260805a"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/ar_video_residual_width_screen_20260805"),
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bccb_rows(result: Path) -> list[dict[str, object]]:
    summaries = load_json(result / "summary.json")["summaries"]
    wanted = [
        ("none", 0, "No correction"),
        ("adaptive_rank_oracle", 8, "Adaptive rank 8"),
        ("adaptive_rank_oracle", 16, "Adaptive rank 16"),
        (
            "frozen_calibration_basis_oracle_coefficients",
            16,
            "Frozen-basis rank 16",
        ),
    ]
    output = []
    for correction, rank, label in wanted:
        matches = [
            item
            for item in summaries
            if item["scope"] == "held_out"
            and item["correction"] == correction
            and int(item["rank"]) == rank
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one BCCB summary for {(correction, rank)}")
        item = matches[0]
        output.append(
            {
                "panel": "bccb_gate",
                "method": label,
                "budget": "5% event",
                "layer": "all",
                "metric": "aggregate",
                "value_percent": 100 * float(item["aggregate_relative_av_l2"]),
            }
        )
        output.append(
            {
                "panel": "bccb_gate",
                "method": label,
                "budget": "5% event",
                "layer": "all",
                "metric": "worst_head",
                "value_percent": 100 * float(item["worst_head_relative_av_l2"]),
            }
        )
    return output


def width_summary_rows(result: Path) -> list[dict[str, object]]:
    summaries = load_json(result / "summary.json")["summaries"]
    output = []
    for item in summaries:
        if (
            item["scope"] == "held_out"
            and item["correction"] == "adaptive_rank_oracle"
            and item["selector"] in LABELS
        ):
            for metric, field in (
                ("aggregate", "aggregate_relative_av_l2"),
                ("worst_head", "worst_head_relative_av_l2"),
            ):
                output.append(
                    {
                        "panel": "width_budget",
                        "method": LABELS[item["selector"]],
                        "selector": item["selector"],
                        "budget": f"{100 * float(item['event_fraction']):.0f}%",
                        "layer": "all",
                        "metric": metric,
                        "value_percent": 100 * float(item[field]),
                    }
                )
    return output


def width_layer_rows(result: Path) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with (result / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] in {"validation", "test"}
                and row["correction"] == "adaptive_rank_oracle"
                and math.isclose(float(row["event_fraction"]), 0.1)
                and row["selector"] in LABELS
            ):
                grouped[(row["selector"], int(row["layer"]))].append(row)
    output = []
    for (selector, layer), rows in sorted(grouped.items()):
        numerator = sum(float(row["numerator_sq"]) for row in rows)
        denominator = sum(float(row["denominator_sq"]) for row in rows)
        output.append(
            {
                "panel": "width_layer",
                "method": LABELS[selector],
                "selector": selector,
                "budget": "10%",
                "layer": layer,
                "metric": "aggregate",
                "value_percent": 100 * math.sqrt(numerator / denominator),
            }
        )
    return output


def overlap_rows(result: Path) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    with (result / "selection_overlap.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if math.isclose(float(row["event_fraction"]), 0.1):
                grouped[(row["left_selector"], row["right_selector"])].append(
                    float(row["jaccard"])
                )
    output = []
    selectors = list(LABELS)
    for left in selectors:
        for right in selectors:
            if left == right:
                value = 1.0
            else:
                key = (left, right) if (left, right) in grouped else (right, left)
                values = grouped[key]
                value = sum(values) / len(values)
            output.append(
                {
                    "panel": "selector_overlap",
                    "method": LABELS[left],
                    "budget": "10%",
                    "layer": LABELS[right],
                    "metric": "jaccard",
                    "value_percent": 100 * value,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["panel", "method", "budget", "layer", "metric", "value_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        bccb_rows(args.bccb_result)
        + width_summary_rows(args.width_result)
        + width_layer_rows(args.width_result)
        + overlap_rows(args.width_result)
    )
    write_csv(args.output_dir / "ar_video_residual_width_screen_data.csv", rows)
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    bccb = [row for row in rows if row["panel"] == "bccb_gate"]
    labels = list(dict.fromkeys(str(row["method"]) for row in bccb))
    x = np.arange(len(labels))
    aggregate = [
        next(
            float(row["value_percent"])
            for row in bccb
            if row["method"] == label and row["metric"] == "aggregate"
        )
        for label in labels
    ]
    worst = [
        next(
            float(row["value_percent"])
            for row in bccb
            if row["method"] == label and row["metric"] == "worst_head"
        )
        for label in labels
    ]
    width = 0.36
    ax.bar(x - width / 2, aggregate, width, label="Aggregate", color="#2A7F76")
    ax.bar(x + width / 2, worst, width, label="Worst head", color="#D18C35")
    ax.axhline(1.0, color="#B23A48", linestyle="--", linewidth=1, label="1% gate")
    ax.set_ylabel("Held-out AV relative L2 (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=12)
    ax.legend(ncol=3, loc="upper left")

    ax = axes[0, 1]
    budget_rows = [
        row
        for row in rows
        if row["panel"] == "width_budget" and row["metric"] == "aggregate"
    ]
    for selector in LABELS:
        values = [
            next(
                float(row["value_percent"])
                for row in budget_rows
                if row.get("selector") == selector and row["budget"] == budget
            )
            for budget in ("5%", "10%")
        ]
        ax.plot(
            [5, 10],
            values,
            marker="o",
            linewidth=2,
            label=LABELS[selector],
            color=COLORS[selector],
        )
    ax.axhline(0.5, color="#B23A48", linestyle="--", linewidth=1, label="0.5% gate")
    ax.set_xlabel("Exact middle-history tile budget (%)")
    ax.set_ylabel("Held-out aggregate AV error (%)")
    ax.set_xticks([5, 10])
    ax.text(-0.12, 1.03, "b", transform=ax.transAxes, fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    layer_rows = [row for row in rows if row["panel"] == "width_layer"]
    layers = [14, 29]
    bar_width = 0.19
    for index, selector in enumerate(LABELS):
        values = [
            next(
                float(row["value_percent"])
                for row in layer_rows
                if row.get("selector") == selector and row["layer"] == layer
            )
            for layer in layers
        ]
        ax.bar(
            np.arange(len(layers)) + (index - 1.5) * bar_width,
            values,
            bar_width,
            label=LABELS[selector],
            color=COLORS[selector],
        )
    ax.axhline(0.5, color="#B23A48", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(layers)), [f"Layer {layer}" for layer in layers])
    ax.set_ylabel("10% budget aggregate AV error (%)")
    ax.text(-0.12, 1.03, "c", transform=ax.transAxes, fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    overlap = [row for row in rows if row["panel"] == "selector_overlap"]
    selectors = list(LABELS)
    matrix = np.array(
        [
            [
                next(
                    float(row["value_percent"]) / 100.0
                    for row in overlap
                    if row["method"] == LABELS[left] and row["layer"] == LABELS[right]
                )
                for right in selectors
            ]
            for left in selectors
        ]
    )
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    short = ["K/V", "Mass", "Value", "Width"]
    ax.set_xticks(range(len(short)), short)
    ax.set_yticks(range(len(short)), short)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > 0.62 else "black"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    ax.set_xlabel("Selector")
    ax.set_ylabel("Selector")
    ax.text(-0.12, 1.03, "d", transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap")

    output = args.output_dir / "ar_video_residual_width_screen_20260805"
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
