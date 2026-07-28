#!/usr/bin/env python3
"""Summarize and plot a completed restricted-rotation oracle run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    "frozen": "#7A7A7A",
    "adaptive": "#000000",
    "full_procrustes": "#D55E00",
    "givens": "#0072B2",
    "householder": "#CC79A7",
    "orthogonal_bcm": "#009E73",
    "dcd": "#E69F00",
    "butterfly": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_success(input_dir: Path) -> dict[str, object]:
    success_path = input_dir / "SUCCESS.json"
    if not success_path.is_file():
        raise RuntimeError(f"input run has no SUCCESS marker: {input_dir}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    for name, expected in success["artifact_sha256"].items():
        actual = file_sha256(input_dir / name)
        if actual != expected:
            raise RuntimeError(f"artifact hash mismatch for {name}: {actual} != {expected}")
    return success


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_pregate(rows: list[dict[str, str]], output_dir: Path) -> None:
    cells = list(dict.fromkeys(row["cell"] for row in rows))
    densities = sorted({float(row["density"]) for row in rows})
    matrices = []
    for field in ("adaptive_rank_output_relative_l2", "adaptive_rank_worst_record_relative_l2"):
        matrix = np.full((len(cells), len(densities)), np.nan)
        for row in rows:
            matrix[cells.index(row["cell"]), densities.index(float(row["density"]))] = 100 * float(row[field])
        matrices.append(matrix)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), constrained_layout=True)
    labels = ("Aggregate output error (%)", "Worst record output error (%)")
    limits = (0.5, 1.0)
    for axis, matrix, label, limit in zip(axes, matrices, labels, limits):
        image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                marker = "PASS" if value <= limit else "FAIL"
                axis.text(column_index, row_index, f"{value:.2f}\n{marker}", ha="center", va="center", fontsize=8)
        axis.set_xticks(range(len(densities)), [f"{100*density:.1f}%" for density in densities])
        axis.set_yticks(range(len(cells)), [cell.replace("_", "\n") for cell in cells], fontsize=8)
        axis.set_xlabel("Critical-key density")
        axis.set_ylabel("Registered layer-step cell")
        axis.text(0.02, 1.02, label, transform=axis.transAxes, fontsize=10, fontweight="bold")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure(figure, output_dir, "adaptive_rank16_pregate")


def best_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["split"], row["cell"], row["method"], row["density"], row["family"])].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        best = min(
            group,
            key=lambda row: (
                float(row["worst_record_output_relative_l2"]),
                float(row["aggregate_output_relative_l2"]),
                int(row["dynamic_scalars"]),
            ),
        )
        output.append(
            {
                "split": key[0],
                "cell": key[1],
                "method": key[2],
                "density": float(key[3]),
                "family": key[4],
                "generators": int(best["generators"]),
                "aggregate_output_relative_l2": float(best["aggregate_output_relative_l2"]),
                "worst_record_output_relative_l2": float(best["worst_record_output_relative_l2"]),
                "subspace_overlap_mean": float(best["subspace_overlap_mean"]),
                "dynamic_scalars": int(best["dynamic_scalars"]),
                "extra_work_ratio": float(best["extra_work_ratio"]),
                "requested_gate_pass": best["requested_gate_pass"] == "True",
                "anti_tautology_gate_pass": best["anti_tautology_gate_pass"] == "True",
            }
        )
    return output


def cross_holdout_frontier(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    restricted = {"givens", "householder", "orthogonal_bcm", "dcd", "butterfly"}
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if row["family"] in restricted:
            grouped[
                (
                    row["cell"],
                    row["method"],
                    row["density"],
                    row["family"],
                    row["generators"],
                )
            ].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        output.append(
            {
                "cell": key[0],
                "method": key[1],
                "density": float(key[2]),
                "family": key[3],
                "generators": int(key[4]),
                "holdouts": ";".join(sorted(row["split"] for row in group)),
                "holdout_count": len(group),
                "max_aggregate_output_relative_l2": max(
                    float(row["aggregate_output_relative_l2"]) for row in group
                ),
                "max_worst_record_output_relative_l2": max(
                    float(row["worst_record_output_relative_l2"]) for row in group
                ),
                "min_subspace_overlap_mean": min(
                    float(row["subspace_overlap_mean"]) for row in group
                ),
                "dynamic_scalars": max(int(row["dynamic_scalars"]) for row in group),
                "extra_work_ratio": max(float(row["extra_work_ratio"]) for row in group),
                "requested_gate_all_holdouts": all(
                    row["requested_gate_pass"] == "True" for row in group
                ),
                "anti_tautology_gate_all_holdouts": all(
                    row["anti_tautology_gate_pass"] == "True" for row in group
                ),
            }
        )
    return output


def head_fallback_frontier(
    record_rows: list[dict[str, str]],
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    restricted = {"givens", "householder", "orthogonal_bcm", "dcd", "butterfly"}
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in record_rows:
        if row["family"] in restricted:
            grouped[
                (
                    row["cell"],
                    row["method"],
                    row["density"],
                    row["family"],
                    row["generators"],
                )
            ].append(row)
    gates = manifest["protocol"]["gates"]
    worst_gate = float(gates["oracle_worst_record_output_relative_l2"])
    aggregate_gate = float(gates["oracle_aggregate_output_relative_l2"])
    scalar_gate = int(gates["max_dynamic_scalars"])
    speed_gate = float(gates["whole_attention_measured_speedup_for_future_kernel"])
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        heads = sorted({int(row["head"]) for row in group})
        fallback = sorted(
            head
            for head in heads
            if max(
                float(row["output_relative_l2"])
                for row in group
                if int(row["head"]) == head
            )
            > worst_gate
        )
        split_metrics = []
        for split in sorted({row["split"] for row in group}):
            kept = [
                row
                for row in group
                if row["split"] == split and int(row["head"]) not in fallback
            ]
            if kept:
                aggregate = np.sqrt(
                    sum(float(row["residual_sq"]) for row in kept)
                    / sum(float(row["reference_sq"]) for row in kept)
                )
                worst = max(float(row["output_relative_l2"]) for row in kept)
            else:
                aggregate = 0.0
                worst = 0.0
            split_metrics.append((split, float(aggregate), worst))
        fallback_fraction = len(fallback) / len(heads)
        extra_work = max(float(row["extra_work_ratio"]) for row in group)
        sparse_work = min(1.0, float(key[2]) + extra_work)
        effective_work = fallback_fraction + (1.0 - fallback_fraction) * sparse_work
        ideal_speedup = 1.0 / effective_work
        max_aggregate = max(metric[1] for metric in split_metrics)
        max_worst = max(metric[2] for metric in split_metrics)
        dynamic_scalars = max(int(row["dynamic_scalars"]) for row in group)
        quality_pass = max_aggregate <= aggregate_gate and max_worst <= worst_gate
        output.append(
            {
                "cell": key[0],
                "method": key[1],
                "density": float(key[2]),
                "family": key[3],
                "generators": int(key[4]),
                "fallback_heads": ";".join(map(str, fallback)),
                "fallback_head_count": len(fallback),
                "total_heads": len(heads),
                "max_kept_aggregate_output_relative_l2": max_aggregate,
                "max_kept_worst_record_output_relative_l2": max_worst,
                "dynamic_scalars": dynamic_scalars,
                "extra_work_ratio": extra_work,
                "effective_attention_work_ratio_upper_bound": effective_work,
                "ideal_attention_speedup_upper_bound": ideal_speedup,
                "quality_gate_after_oracle_head_fallback": quality_pass,
                "payload_gate": dynamic_scalars <= scalar_gate,
                "speed_upper_bound_gate": ideal_speedup >= speed_gate,
                "oracle_hybrid_gate": quality_pass
                and dynamic_scalars <= scalar_gate
                and ideal_speedup >= speed_gate,
                "claim_warning": "fallback heads and transform parameters are selected post-hoc from held-out dense defects",
            }
        )
    return output


def plot_rotation_curves(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    restricted = {"givens", "householder", "orthogonal_bcm", "dcd", "butterfly"}
    restricted_rows = [row for row in summary_rows if row["family"] in restricted]
    if not restricted_rows:
        raise RuntimeError("completed run contains no restricted-family summaries")
    plotted_cell = next(
        (
            row["cell"]
            for row in restricted_rows
            if row["cell"] == "layer00_step00_capacity_control"
        ),
        restricted_rows[0]["cell"],
    )
    rows = [row for row in restricted_rows if row["cell"] == plotted_cell]
    splits = list(dict.fromkeys(row["split"] for row in rows))
    densities = sorted({float(row["density"]) for row in rows})
    figure, axes = plt.subplots(
        len(splits), len(densities), figsize=(9.2, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    axes_array = np.asarray(axes, dtype=object).reshape(len(splits), len(densities))
    for row_index, split in enumerate(splits):
        for column_index, density in enumerate(densities):
            axis = axes_array[row_index, column_index]
            subset = [row for row in rows if row["split"] == split and float(row["density"]) == density]
            for family in sorted(restricted):
                family_rows = sorted(
                    (row for row in subset if row["family"] == family),
                    key=lambda row: int(row["generators"]),
                )
                if not family_rows:
                    continue
                axis.plot(
                    [int(row["generators"]) for row in family_rows],
                    [100 * float(row["worst_record_output_relative_l2"]) for row in family_rows],
                    marker="o",
                    linewidth=1.7,
                    markersize=4,
                    color=COLORS[family],
                    label=family.replace("_", " "),
                )
            axis.axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.0)
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            axis.set_xlabel("Structured generators M")
            axis.set_ylabel("Worst output error (%)")
            axis.text(
                0.03,
                0.96,
                f"{split.replace('_', ' ')} | density {100*density:.1f}%",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
            )
            style_axis(axis)
    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    save_figure(figure, output_dir, "restricted_rotation_error_by_M")


def plot_payload_pareto(best: list[dict[str, object]], output_dir: Path) -> None:
    restricted = {"givens", "householder", "orthogonal_bcm", "dcd", "butterfly"}
    restricted_rows = [row for row in best if row["family"] in restricted]
    if not restricted_rows:
        raise RuntimeError("completed run contains no restricted-family best candidates")
    plotted_cell = next(
        (
            str(row["cell"])
            for row in restricted_rows
            if row["cell"] == "layer00_step00_capacity_control"
        ),
        str(restricted_rows[0]["cell"]),
    )
    rows = [row for row in restricted_rows if row["cell"] == plotted_cell]
    figure, axis = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    for family in sorted(restricted):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        axis.scatter(
            [max(1, int(row["dynamic_scalars"])) for row in family_rows],
            [100 * float(row["worst_record_output_relative_l2"]) for row in family_rows],
            s=56,
            alpha=0.85,
            color=COLORS[family],
            edgecolor="white",
            linewidth=0.6,
            label=family.replace("_", " "),
        )
    axis.axhline(1.0, color="#B2182B", linestyle="--", linewidth=1.1, label="1% worst-error gate")
    axis.axvline(512, color="#4D4D4D", linestyle=":", linewidth=1.1, label="512-scalar guard")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Dynamic transform scalars per head-tile")
    axis.set_ylabel("Worst output error (%)")
    style_axis(axis)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(figure, output_dir, "rotation_quality_payload_pareto")


def plot_head_fallback(frontier: list[dict[str, object]], output_dir: Path) -> None:
    candidates = [row for row in frontier if bool(row["oracle_hybrid_gate"])]
    if not candidates:
        candidates = sorted(
            frontier,
            key=lambda row: (
                float(row["max_kept_worst_record_output_relative_l2"]),
                -float(row["ideal_attention_speedup_upper_bound"]),
            ),
        )[:5]
    best_by_family: dict[str, dict[str, object]] = {}
    for row in candidates:
        family = str(row["family"])
        current = best_by_family.get(family)
        if current is None or float(row["ideal_attention_speedup_upper_bound"]) > float(
            current["ideal_attention_speedup_upper_bound"]
        ):
            best_by_family[family] = row
    selected = sorted(best_by_family.values(), key=lambda row: str(row["family"]))
    labels = [f"{str(row['family']).replace('_', ' ')}\nM={row['generators']}" for row in selected]
    speeds = [float(row["ideal_attention_speedup_upper_bound"]) for row in selected]
    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    bars = axis.bar(
        range(len(selected)),
        speeds,
        color=[COLORS[str(row["family"])] for row in selected],
        edgecolor="white",
    )
    for bar, row in zip(bars, selected):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{row['fallback_head_count']}/{row['total_heads']} dense",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.axhline(1.5, color="#B2182B", linestyle="--", linewidth=1.1)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Ideal attention speedup upper bound (x)")
    axis.set_xlabel("Post-hoc structured tail with universal head fallback")
    style_axis(axis)
    save_figure(figure, output_dir, "hybrid_head_fallback_upper_bound")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    verify_success(input_dir)
    require_fresh_output_dir(args.output_dir)
    pregate_rows = read_csv(input_dir / "adaptive_pregate.csv")
    summary_rows = read_csv(input_dir / "rotation_summary.csv")
    record_rows = read_csv(input_dir / "rotation_records.csv")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((input_dir / "decision.json").read_text(encoding="utf-8"))
    best = best_rows(summary_rows)
    cross_holdout = cross_holdout_frontier(summary_rows)
    fallback = head_fallback_frontier(record_rows, manifest)
    atomic_write_csv(args.output_dir / "best_candidates.csv", best)
    atomic_write_csv(args.output_dir / "cross_holdout_frontier.csv", cross_holdout)
    atomic_write_csv(args.output_dir / "head_fallback_frontier.csv", fallback)
    atomic_write_csv(args.output_dir / "rotation_plot_source.csv", summary_rows)
    plot_pregate(pregate_rows, args.output_dir)
    plot_rotation_curves(summary_rows, args.output_dir)
    plot_payload_pareto(best, args.output_dir)
    plot_head_fallback(fallback, args.output_dir)
    analysis = {
        "input_dir": str(input_dir),
        "input_success_sha256": file_sha256(input_dir / "SUCCESS.json"),
        "verdict": decision["verdict"],
        "best_candidate": decision["best_candidate"],
        "best_low_payload_cross_holdout": min(
            (
                row
                for row in cross_holdout
                if int(row["dynamic_scalars"])
                <= int(manifest["protocol"]["gates"]["max_dynamic_scalars"])
            ),
            key=lambda row: float(row["max_worst_record_output_relative_l2"]),
            default=None,
        ),
        "oracle_hybrid_candidates": [row for row in fallback if bool(row["oracle_hybrid_gate"])],
        "pregate_passes": [row for row in pregate_rows if row["adaptive_pregate_pass"] == "True"],
        "pregate_failures": [row for row in pregate_rows if row["adaptive_pregate_pass"] != "True"],
        "figures": [
            "adaptive_rank16_pregate.png",
            "restricted_rotation_error_by_M.png",
            "rotation_quality_payload_pareto.png",
            "hybrid_head_fallback_upper_bound.png",
        ],
    }
    atomic_write_json(args.output_dir / "analysis_summary.json", analysis)
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {
            "verdict": decision["verdict"],
            "artifact_sha256": {
                name: file_sha256(args.output_dir / name)
                for name in (
                    "best_candidates.csv",
                    "cross_holdout_frontier.csv",
                    "head_fallback_frontier.csv",
                    "rotation_plot_source.csv",
                    "adaptive_rank16_pregate.png",
                    "adaptive_rank16_pregate.pdf",
                    "restricted_rotation_error_by_M.png",
                    "restricted_rotation_error_by_M.pdf",
                    "rotation_quality_payload_pareto.png",
                    "rotation_quality_payload_pareto.pdf",
                    "hybrid_head_fallback_upper_bound.png",
                    "hybrid_head_fallback_upper_bound.pdf",
                    "analysis_summary.json",
                )
            },
        },
    )
    print(f"[rotation-analysis] verdict={decision['verdict']}")
    print(f"[rotation-analysis] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
