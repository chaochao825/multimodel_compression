"""Plot leakage-safe layer/head certification for the LongLive memory probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


CORRECTIONS = (
    "adaptive_rank_oracle",
    "frozen_calibration_basis_oracle_coefficients",
)
CORRECTION_LABELS = {
    "adaptive_rank_oracle": "Adaptive rank-16 oracle",
    "frozen_calibration_basis_oracle_coefficients": "Frozen basis, oracle coeffs",
}
HELD_OUT_SPLITS = {"validation", "test"}
AGGREGATE_GATE = 0.005
WORST_GATE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_error(rows: list[dict[str, float]]) -> float:
    numerator = sum(row["numerator_sq"] for row in rows)
    denominator = sum(row["denominator_sq"] for row in rows)
    if denominator <= 0:
        raise ValueError("non-positive AV reference energy")
    return math.sqrt(numerator / denominator)


def select_candidates(
    summaries: list[dict[str, Any]], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    calibration = [
        row
        for row in summaries
        if row["scope"] == "calibration"
        and row["correction"] == "adaptive_rank_oracle"
        and int(row["rank"]) == 16
    ]
    speed_pool = [
        row
        for row in calibration
        if float(row["minimum_arithmetic_reduction"]) >= 1.5
    ]
    if not speed_pool or not calibration:
        raise ValueError("calibration summaries are incomplete")

    selected = [
        {
            "role": "Preregistered primary",
            "method": gate["primary_method"],
            "selection_scope": "preregistered",
        },
        {
            "role": "Calibration-selected >=1.5x",
            "method": min(
                speed_pool, key=lambda row: row["aggregate_relative_av_l2"]
            )["method"],
            "selection_scope": "calibration",
        },
        {
            "role": "Calibration-selected quality",
            "method": min(
                calibration, key=lambda row: row["aggregate_relative_av_l2"]
            )["method"],
            "selection_scope": "calibration",
        },
    ]
    unique = []
    seen = set()
    for row in selected:
        if row["method"] not in seen:
            unique.append(row)
            seen.add(row["method"])
    return unique


def summarize_heads(
    metrics_path: Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_by_method = {row["method"]: row for row in candidates}
    groups: dict[tuple[str, str, int, int], list[dict[str, float]]] = defaultdict(list)
    reductions: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if (
                raw["split"] not in HELD_OUT_SPLITS
                or raw["method"] not in candidate_by_method
                or raw["correction"] not in CORRECTIONS
                or int(raw["rank"]) != 16
            ):
                continue
            key = (
                raw["method"],
                raw["correction"],
                int(raw["layer"]),
                int(raw["head_index"]),
            )
            groups[key].append(
                {
                    "relative_av_l2": float(raw["relative_av_l2"]),
                    "numerator_sq": float(raw["numerator_sq"]),
                    "denominator_sq": float(raw["denominator_sq"]),
                }
            )
            reductions[key].append(float(raw["arithmetic_reduction"]))

    rows = []
    for key, values in sorted(groups.items()):
        method, correction, layer, head = key
        aggregate = aggregate_error(values)
        worst = max(row["relative_av_l2"] for row in values)
        candidate = candidate_by_method[method]
        rows.append(
            {
                "candidate_role": candidate["role"],
                "selection_scope": candidate["selection_scope"],
                "evaluation_scope": "validation+test",
                "method": method,
                "correction": correction,
                "rank": 16,
                "layer": layer,
                "head": head,
                "aggregate_error_percent": 100 * aggregate,
                "worst_error_percent": 100 * worst,
                "passes_0p5_1p0_gate": aggregate <= AGGREGATE_GATE
                and worst <= WORST_GATE,
                "minimum_arithmetic_reduction": min(reductions[key]),
                "records": len(values),
            }
        )
    expected = len(candidates) * len(CORRECTIONS) * 3 * 12
    if len(rows) != expected:
        raise ValueError(f"expected {expected} head summaries, found {len(rows)}")
    return rows


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def display_method(method: str) -> str:
    return (
        method.replace("phasealigned_recency_", "phase-aligned ")
        .replace("postrope_recency_", "post-RoPE ")
        .replace("_event_", " / event=")
        .replace("0p", "0.")
    )


def plot_heatmaps(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]], output_dir: Path
) -> None:
    layers = [0, 14, 29]
    heads = list(range(12))
    vmax = max(10.0, max(float(row["worst_error_percent"]) for row in rows))
    norm = LogNorm(vmin=0.1, vmax=vmax)
    fig, axes = plt.subplots(
        len(candidates), 2, figsize=(15.2, 3.05 * len(candidates)), squeeze=False
    )
    image = None
    for row_index, candidate in enumerate(candidates):
        for column_index, correction in enumerate(CORRECTIONS):
            ax = axes[row_index, column_index]
            subset = [
                row
                for row in rows
                if row["method"] == candidate["method"]
                and row["correction"] == correction
            ]
            lookup = {(row["layer"], row["head"]): row for row in subset}
            matrix = np.array(
                [
                    [lookup[(layer, head)]["worst_error_percent"] for head in heads]
                    for layer in layers
                ]
            )
            image = ax.imshow(matrix, cmap="YlOrRd", norm=norm, aspect="auto")
            for y, layer in enumerate(layers):
                for x, head in enumerate(heads):
                    cell = lookup[(layer, head)]
                    value = float(cell["worst_error_percent"])
                    color = "white" if value > math.sqrt(0.1 * vmax) else "#222222"
                    ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7)
                    if cell["passes_0p5_1p0_gate"]:
                        ax.add_patch(
                            plt.Rectangle(
                                (x - 0.48, y - 0.48),
                                0.96,
                                0.96,
                                fill=False,
                                edgecolor="#007F5F",
                                linewidth=2.0,
                            )
                        )
            passed = sum(bool(row["passes_0p5_1p0_gate"]) for row in subset)
            reduction = min(float(row["minimum_arithmetic_reduction"]) for row in subset)
            ax.set_title(
                f"{candidate['role']} | {CORRECTION_LABELS[correction]}\n"
                f"{display_method(candidate['method'])} | pass {passed}/36 | >= {reduction:.2f}x arithmetic",
                fontsize=9,
            )
            ax.set_xticks(heads)
            ax.set_xticklabels(heads)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"Layer {layer}" for layer in layers])
            ax.set_xlabel("Attention head")
    if image is None:
        raise ValueError("no heatmap data")
    fig.subplots_adjust(hspace=0.42, wspace=0.13, right=0.9)
    colorbar_axis = fig.add_axes((0.92, 0.16, 0.014, 0.7))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Worst held-out AV error (%), logarithmic color scale")
    fig.suptitle(
        "LongLive residual-memory head certification\n"
        "Green outline: aggregate <=0.5% and worst <=1%; selection never uses held-out data",
        fontsize=13,
        y=1.01,
    )
    save_figure(fig, output_dir, "head_certification_heatmap")


def plot_counts(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]], output_dir: Path
) -> None:
    colors = {
        "adaptive_rank_oracle": "#009E73",
        "frozen_calibration_basis_oracle_coefficients": "#CC79A7",
    }
    labels = [candidate["role"] for candidate in candidates]
    x = np.arange(len(candidates))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for offset_index, correction in enumerate(CORRECTIONS):
        values = []
        details = []
        for candidate in candidates:
            subset = [
                row
                for row in rows
                if row["method"] == candidate["method"]
                and row["correction"] == correction
            ]
            counts = {
                layer: sum(
                    bool(row["passes_0p5_1p0_gate"])
                    for row in subset
                    if row["layer"] == layer
                )
                for layer in (0, 14, 29)
            }
            values.append(sum(counts.values()))
            details.append(counts)
        positions = x + (offset_index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            color=colors[correction],
            label=CORRECTION_LABELS[correction],
        )
        for bar, total, counts in zip(bars, values, details):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{total}/36\nL0:{counts[0]}  L14:{counts[14]}  L29:{counts[29]}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 24)
    ax.set_ylabel("Certified layer-head pairs")
    ax.set_title("Certification collapses under frozen-basis transfer, especially at Layer 14")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "head_certification_counts")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    gate = json.loads((args.result_dir / "gate_decision.json").read_text(encoding="utf-8"))
    candidates = select_candidates(summary["summaries"], gate)
    rows = summarize_heads(args.result_dir / "metrics.csv", candidates)
    write_csv(args.output_dir / "head_certification.csv", rows)
    write_csv(args.output_dir / "head_certification_candidates.csv", candidates)
    plot_heatmaps(rows, candidates, args.output_dir)
    plot_counts(rows, candidates, args.output_dir)
    print(f"[figures] wrote head certification artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
