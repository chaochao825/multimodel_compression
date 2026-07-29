#!/usr/bin/env python3
"""Analyze registered sparse-support manifold capacity artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    require_fresh_output_dir,
)


COLORS = {
    "fixed64": "#4D4D4D",
    "shifted64": "#377EB8",
    "hierarchical32": "#E41A1C",
    "shifted32": "#FF7F00",
    "thw8x8": "#4DAF4A",
    "motion_warp8x8": "#984EA3",
    "support_family_oracle": "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def verify_success(path: Path) -> dict[str, object]:
    success_path = path / "SUCCESS.json"
    if not success_path.is_file():
        raise FileNotFoundError(f"missing completed-run marker: {success_path}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    for name, expected in success["artifact_sha256"].items():
        actual = file_sha256(path / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch for {name}: {actual} != {expected}")
    return success


def layer14_frontier(summary: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        if row["cell"].startswith("layer14_"):
            grouped[(row["family"], float(row["density_target"]))].append(row)
    output = []
    for (family, density), rows in sorted(grouped.items()):
        output.append(
            {
                "family": family,
                "density_target": density,
                "cells": len(rows),
                "max_aggregate_output_relative_l2": max(
                    float(row["adaptive_rank16_output_relative_l2"]) for row in rows
                ),
                "max_worst_record_output_relative_l2": max(
                    float(row["adaptive_rank16_worst_record_relative_l2"]) for row in rows
                ),
                "max_p95_record_output_relative_l2": max(
                    float(row["adaptive_rank16_p95_record_relative_l2"]) for row in rows
                ),
                "mean_rank_aware_relative_improvement": sum(
                    float(row["rank_aware_relative_improvement_mean"]) for row in rows
                )
                / len(rows),
                "max_rank_required_for_record_gate": max(
                    int(float(row["rank_required_for_record_gate_max"])) for row in rows
                ),
                "max_execution_density": max(float(row["execution_density_mean"]) for row in rows),
                "max_kernel_tile_multiplier": max(
                    float(row["kernel_tile_multiplier_vs_fixed64"]) for row in rows
                ),
                "all_cells_pass": all(as_bool(row["support_pregate_pass"]) for row in rows),
            }
        )
    return output


def family_choices(records: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], Counter[str]] = defaultdict(Counter)
    for row in records:
        if row["family"] == "support_family_oracle":
            grouped[(row["cell"], float(row["density_target"]))][row["chosen_family"]] += 1
    output = []
    for (cell, density), counts in sorted(grouped.items()):
        total = sum(counts.values())
        for family, count in sorted(counts.items()):
            output.append(
                {
                    "cell": cell,
                    "density_target": density,
                    "chosen_family": family,
                    "records": count,
                    "fraction": count / total,
                }
            )
    return output


def rank_budget_frontier(records: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        for rank in (0, 4, 8, 16, 32):
            grouped[(row["cell"], row["family"], float(row["density_target"]), rank)].append(row)
    output = []
    for (cell, family, density, rank), rows in sorted(grouped.items()):
        reference_sq = sum(float(row["reference_sq"]) for row in rows)
        residuals = []
        for row in rows:
            if rank == 0:
                residual_sq = float(row["critical_residual_sq"])
            else:
                captured = float(row[f"defect_energy_rank{rank}"])
                residual_sq = float(row["critical_residual_sq"]) * max(0.0, 1.0 - captured)
            residuals.append((residual_sq, float(row["reference_sq"])))
        output.append(
            {
                "cell": cell,
                "family": family,
                "density_target": density,
                "rank": rank,
                "aggregate_output_relative_l2": math.sqrt(
                    sum(value for value, _ in residuals) / max(reference_sq, 1e-30)
                ),
                "worst_record_output_relative_l2": max(
                    math.sqrt(value / max(reference, 1e-30)) for value, reference in residuals
                ),
            }
        )
    return output


def head_fallback_frontier(
    records: list[dict[str, str]], manifest: dict[str, object]
) -> list[dict[str, object]]:
    gates = manifest["protocol"]["gates"]
    aggregate_gate = float(gates["oracle_aggregate_output_relative_l2"])
    worst_gate = float(gates["oracle_worst_record_output_relative_l2"])
    rank = int(manifest["protocol"]["scope"]["rank"])
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        if row["cell"].startswith("layer14_"):
            grouped[(row["family"], float(row["density_target"]))].append(row)
    output = []
    for (family, density), rows in sorted(grouped.items()):
        failing_heads = sorted(
            {
                int(row["head"])
                for row in rows
                if float(row["adaptive_output_relative_l2"]) > worst_gate
            }
        )
        kept = [row for row in rows if int(row["head"]) not in failing_heads]
        total_heads = len({int(row["head"]) for row in rows})
        if kept:
            reference_sq = sum(float(row["reference_sq"]) for row in kept)
            residual_sq = sum(float(row["adaptive_residual_sq"]) for row in kept)
            aggregate = math.sqrt(residual_sq / max(reference_sq, 1e-30))
            worst = max(float(row["adaptive_output_relative_l2"]) for row in kept)
            execution_density = sum(float(row["execution_density"]) for row in kept) / len(kept)
            key_tokens = sum(int(row["key_tokens"]) for row in kept) / len(kept)
        else:
            aggregate = worst = 0.0
            execution_density = density
            key_tokens = 32760.0
        dense_fraction = len(failing_heads) / total_heads
        tail_ratio = rank / key_tokens
        effective_work = dense_fraction + (1.0 - dense_fraction) * (execution_density + tail_ratio)
        output.append(
            {
                "family": family,
                "density_target": density,
                "fallback_heads": ",".join(map(str, failing_heads)),
                "fallback_head_count": len(failing_heads),
                "total_heads": total_heads,
                "kept_aggregate_output_relative_l2": aggregate,
                "kept_worst_record_output_relative_l2": worst,
                "quality_gate": aggregate <= aggregate_gate and worst <= worst_gate,
                "ideal_attention_arithmetic_speedup_upper_bound": 1.0 / effective_work,
                "claim_warning": "fallback heads are selected post-hoc from all evaluated dense layer-14 defects; routing and launch overhead are ignored",
            }
        )
    return output


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(summary: list[dict[str, str]], output_dir: Path) -> None:
    families = list(dict.fromkeys(row["family"] for row in summary))
    columns = list(dict.fromkeys((row["cell"], float(row["density_target"])) for row in summary))
    lookup = {
        (row["family"], row["cell"], float(row["density_target"])): 100
        * float(row["adaptive_rank16_worst_record_relative_l2"])
        for row in summary
    }
    matrix = np.array([[lookup[(family, cell, density)] for cell, density in columns] for family in families])
    figure, axis = plt.subplots(figsize=(13.2, 5.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(5.0, float(matrix.max())))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7)
    axis.set_yticks(range(len(families)), [family.replace("_", " ") for family in families])
    axis.set_xticks(
        range(len(columns)),
        [f"{cell.replace('layer', 'L').replace('_', ' ')}\n{100*density:.1f}%" for cell, density in columns],
        rotation=42,
        ha="right",
    )
    axis.set_title("Adaptive rank-16 worst record output error (%)")
    figure.colorbar(image, ax=axis, label="Worst relative L2 (%)")
    save_figure(figure, output_dir, "support_rank16_worst_heatmap")


def plot_frontier(frontier: list[dict[str, object]], output_dir: Path) -> None:
    densities = sorted({float(row["density_target"]) for row in frontier})
    figure, axes = plt.subplots(
        1, len(densities), figsize=(11.8, 4.8), sharex=True, sharey=True, constrained_layout=True
    )
    if len(densities) == 1:
        axes = [axes]
    for axis, density in zip(axes, densities):
        subset = [row for row in frontier if float(row["density_target"]) == density]
        for row in subset:
            family = str(row["family"])
            axis.scatter(
                float(row["max_kernel_tile_multiplier"]),
                100 * float(row["max_worst_record_output_relative_l2"]),
                s=72,
                color=COLORS[family],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.7,
                label=family.replace("_", " "),
            )
        axis.axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.1)
        axis.set_title(f"Executed density: {100*density:.1f}%")
        axis.set_xlabel("Kernel tile count multiplier vs fixed 64x64")
        axis.set_xlim(0.82, 4.18)
        style_axis(axis)
    axes[0].set_ylabel("Max Layer-14 worst output error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.06),
    )
    save_figure(figure, output_dir, "support_quality_kernel_tile_pareto")


def plot_rank_requirement(frontier: list[dict[str, object]], output_dir: Path) -> None:
    rows = sorted(
        (row for row in frontier if float(row["density_target"]) == 0.25),
        key=lambda row: int(row["max_rank_required_for_record_gate"]),
    )
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    labels = [str(row["family"]).replace("_", " ") for row in rows]
    values = [int(row["max_rank_required_for_record_gate"]) for row in rows]
    axis.bar(range(len(rows)), values, color=[COLORS[str(row["family"])] for row in rows])
    axis.axhline(16, color="#B2182B", linestyle="--", linewidth=1.1, label="rank-16 budget")
    axis.set_xticks(range(len(rows)), labels, rotation=28, ha="right")
    axis.set_ylabel("Worst record rank required for <=1% output error")
    style_axis(axis)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "support_intrinsic_rank_requirement")


def plot_rank_budget(rows: list[dict[str, object]], output_dir: Path) -> None:
    families = ("fixed64", "hierarchical32", "support_family_oracle")
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True, constrained_layout=True)
    cells = ("layer14_step00_early", "layer14_step09_middle", "layer14_step19_late")
    for axis, cell in zip(axes, cells):
        for family in families:
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["cell"] == cell
                    and row["family"] == family
                    and float(row["density_target"]) == 0.25
                ),
                key=lambda row: int(row["rank"]),
            )
            axis.plot(
                [int(row["rank"]) for row in subset],
                [100 * float(row["worst_record_output_relative_l2"]) for row in subset],
                marker="o",
                linewidth=1.8,
                color=COLORS[family],
                label=family.replace("_", " "),
            )
        axis.axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.0)
        axis.set_title(cell.replace("layer14_", "L14 ").replace("_", " "))
        axis.set_xlabel("Adaptive output-tail rank")
        style_axis(axis)
    axes[0].set_ylabel("Worst record output error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    save_figure(figure, output_dir, "support_rank_budget_curve")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    success = verify_success(input_dir)
    require_fresh_output_dir(args.output_dir)
    summary = read_csv(input_dir / "support_summary.csv")
    records = read_csv(input_dir / "support_records.csv")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((input_dir / "decision.json").read_text(encoding="utf-8"))
    frontier = layer14_frontier(summary)
    choices = family_choices(records)
    rank_frontier = rank_budget_frontier(records)
    fallback = head_fallback_frontier(records, manifest)
    atomic_write_csv(args.output_dir / "layer14_support_frontier.csv", frontier)
    atomic_write_csv(args.output_dir / "support_family_choices.csv", choices)
    atomic_write_csv(args.output_dir / "support_rank_budget_frontier.csv", rank_frontier)
    atomic_write_csv(args.output_dir / "layer14_head_fallback_frontier.csv", fallback)
    atomic_write_csv(args.output_dir / "support_plot_source.csv", summary)
    plot_heatmap(summary, args.output_dir)
    plot_frontier(frontier, args.output_dir)
    plot_rank_requirement(frontier, args.output_dir)
    plot_rank_budget(rank_frontier, args.output_dir)
    analysis = {
        "input_dir": str(input_dir),
        "input_success_sha256": file_sha256(input_dir / "SUCCESS.json"),
        "input_verdict": success["verdict"],
        "decision": decision,
        "best_layer14_frontier": min(
            frontier, key=lambda row: float(row["max_worst_record_output_relative_l2"])
        ),
        "head_fallback_quality_candidates": [row for row in fallback if bool(row["quality_gate"])],
        "claim_warning": "All masks, family choices, tails, and fallback heads are held-out post-hoc oracles. Speed values are arithmetic upper bounds, not H200 measurements.",
    }
    atomic_write_json(args.output_dir / "analysis_summary.json", analysis)
    artifact_names = (
        "layer14_support_frontier.csv",
        "support_family_choices.csv",
        "support_rank_budget_frontier.csv",
        "layer14_head_fallback_frontier.csv",
        "support_plot_source.csv",
        "support_rank16_worst_heatmap.png",
        "support_rank16_worst_heatmap.pdf",
        "support_quality_kernel_tile_pareto.png",
        "support_quality_kernel_tile_pareto.pdf",
        "support_intrinsic_rank_requirement.png",
        "support_intrinsic_rank_requirement.pdf",
        "support_rank_budget_curve.png",
        "support_rank_budget_curve.pdf",
        "analysis_summary.json",
    )
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {
            "verdict": decision["verdict"],
            "artifact_sha256": {name: file_sha256(args.output_dir / name) for name in artifact_names},
        },
    )
    print(f"[support-analysis] verdict={decision['verdict']}")
    print(f"[support-analysis] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
