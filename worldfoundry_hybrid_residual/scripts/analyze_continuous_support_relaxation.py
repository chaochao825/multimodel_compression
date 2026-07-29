#!/usr/bin/env python3
"""Analyze continuous support weights and their executable top-k projections."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    require_fresh_output_dir,
)
from probe_continuous_support_relaxation import TOPK_MULTIPLIERS, multiplier_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def build_tradeoff_rows(records: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for record in records:
        base_density = float(record["density"])
        base_tile_budget = int(round(float(record["weight_sum"])))
        candidate_tile_count = int(record["weight_dynamic_scalars"])
        modes = [
            (
                "binary support",
                base_density,
                float(record["discrete_recomputed_output_relative_l2"]),
                0,
            )
        ]
        for multiplier in TOPK_MULTIPLIERS:
            density = min(1.0, base_density * multiplier)
            dynamic_scalars = min(
                candidate_tile_count, int(round(base_tile_budget * multiplier))
            )
            modes.extend(
                (
                    (
                        f"raw weighted top-k {multiplier:g}x",
                        density,
                        float(
                            record[
                                f"weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
                            ]
                        ),
                        dynamic_scalars,
                    ),
                    (
                        f"refit weighted top-k {multiplier:g}x",
                        density,
                        float(
                            record[
                                f"refit_weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
                            ]
                        ),
                        dynamic_scalars,
                    ),
                )
            )
        modes.append(
            (
                "continuous fractional",
                1.0,
                float(record["fractional_output_relative_l2"]),
                int(record["weight_dynamic_scalars"]),
            )
        )
        for mode, density, error, dynamic_scalars in modes:
            output.append(
                {
                    "sample_id": record["sample_id"],
                    "cell": record["cell"],
                    "head": int(record["head"]),
                    "tile_index": int(record["tile_index"]),
                    "mode": mode,
                    "tile_density": density,
                    "output_relative_l2": error,
                    "dynamic_tile_scalars": dynamic_scalars,
                    "ideal_tile_arithmetic_speedup_upper_bound": 1.0 / density,
                    "claim_warning": "post-hoc dense-AV oracle; speed ignores rank tail, routing, normalization, and kernel overhead",
                }
            )
    return output


def summarize_tradeoff(rows: list[dict[str, object]]) -> dict[str, object]:
    speed_target = 1.5
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    modes = []
    for mode, group in grouped.items():
        density = max(float(row["tile_density"]) for row in group)
        maximum = max(float(row["output_relative_l2"]) for row in group)
        modes.append(
            {
                "mode": mode,
                "tile_density": density,
                "max_output_relative_l2": maximum,
                "all_records_pass_1pct": maximum <= 0.01,
                "max_dynamic_tile_scalars": max(
                    int(row["dynamic_tile_scalars"]) for row in group
                ),
                "ideal_tile_arithmetic_speedup_upper_bound": 1.0 / density,
            }
        )
    modes.sort(key=lambda row: (float(row["tile_density"]), str(row["mode"])))
    executable = [
        row
        for row in modes
        if str(row["mode"]).startswith("refit weighted top-k")
        and bool(row["all_records_pass_1pct"])
    ]
    first_pass = min(executable, key=lambda row: float(row["tile_density"])) if executable else None
    quality_speed_joint_gate = bool(
        first_pass
        and float(first_pass["ideal_tile_arithmetic_speedup_upper_bound"]) >= speed_target
    )
    return {
        "modes": modes,
        "minimum_refit_weighted_topk_density_passing_1pct": (
            float(first_pass["tile_density"]) if first_pass else None
        ),
        "minimum_weighted_topk_density_passing_1pct": (
            float(first_pass["tile_density"]) if first_pass else None
        ),
        "corresponding_ideal_tile_arithmetic_speedup_upper_bound": (
            float(first_pass["ideal_tile_arithmetic_speedup_upper_bound"])
            if first_pass
            else None
        ),
        "ideal_tile_arithmetic_speedup_target": speed_target,
        "quality_speed_joint_gate_pass": quality_speed_joint_gate,
        "posthoc_arithmetic_quality_gate_pass": quality_speed_joint_gate,
        "verdict": (
            "STOP_POSTHOC_WEIGHTED_SUPPORT_NO_ARITHMETIC_INTERSECTION"
            if first_pass and not quality_speed_joint_gate
            else "POSTHOC_WEIGHTED_SUPPORT_CAPACITY_GATE_PASSES"
            if quality_speed_joint_gate
            else "STOP_POSTHOC_WEIGHTED_SUPPORT_QUALITY_GATE_FAILS"
        ),
    }


def plot_tradeoff(rows: list[dict[str, object]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)
    weighted = [row for row in rows if str(row["mode"]).startswith("refit weighted top-k")]
    cells = list(dict.fromkeys(str(row["cell"]) for row in weighted))
    colors = ("#2166AC", "#D6604D", "#4D9221")
    for cell, color in zip(cells, colors):
        subset = sorted(
            (row for row in weighted if row["cell"] == cell),
            key=lambda row: float(row["tile_density"]),
        )
        axes[0].plot(
            [100 * float(row["tile_density"]) for row in subset],
            [100 * float(row["output_relative_l2"]) for row in subset],
            marker="o",
            linewidth=2.0,
            color=color,
            label=cell.replace("layer14_", "L14 ").replace("_", " "),
        )
    axes[0].axhline(1.0, color="#111111", linestyle="--", linewidth=1.1, label="1% gate")
    axes[0].set_xlabel("Executed key-tile density (%)")
    axes[0].set_ylabel("Worst-record output error (%)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)

    summary = summarize_tradeoff(rows)
    modes = [
        row
        for row in summary["modes"]
        if str(row["mode"]).startswith("refit weighted top-k")
    ]
    color_map = plt.get_cmap("RdYlGn")
    axes[1].bar(
        [f"{100*float(row['tile_density']):.0f}%" for row in modes],
        [100 * float(row["max_output_relative_l2"]) for row in modes],
        color=[color_map(index / max(len(modes) - 1, 1)) for index in range(len(modes))],
    )
    axes[1].axhline(1.0, color="#111111", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Executed key-tile density")
    axes[1].set_ylabel("Maximum output error across cells (%)")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", labelrotation=55)
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    threshold = summary["minimum_refit_weighted_topk_density_passing_1pct"]
    threshold_text = f"{100*float(threshold):.2f}%" if threshold is not None else "no tested density"
    figure.suptitle(
        f"Post-hoc tile-weight refit: first 1% pass at {threshold_text} density"
    )
    figure.savefig(output_dir / "continuous_support_density_tradeoff.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "continuous_support_density_tradeoff.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    success = verify_success(input_dir)
    require_fresh_output_dir(args.output_dir)
    records = read_csv(input_dir / "continuous_support_relaxation.csv")
    rows = build_tradeoff_rows(records)
    summary = summarize_tradeoff(rows)
    summary.update(
        {
            "input_dir": str(input_dir),
            "input_success_sha256": file_sha256(input_dir / "SUCCESS.json"),
            "input_verdict": success["verdict"],
            "fractional_effective_atoms": [
                float(record["weight_effective_atoms"]) for record in records
            ],
            "fractional_near_binary_fraction": [
                float(record["weight_near_binary_fraction"]) for record in records
            ],
            "fractional_candidate_weight_count": max(
                int(record["weight_dynamic_scalars"]) for record in records
            ),
            "claim_boundary": "Worst-record post-hoc oracle only; no deployable router or H200 sparse kernel is evaluated.",
        }
    )
    atomic_write_csv(args.output_dir / "continuous_support_density_tradeoff.csv", rows)
    atomic_write_json(args.output_dir / "analysis_summary.json", summary)
    plot_tradeoff(rows, args.output_dir)
    artifacts = (
        "continuous_support_density_tradeoff.csv",
        "analysis_summary.json",
        "continuous_support_density_tradeoff.png",
        "continuous_support_density_tradeoff.pdf",
    )
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {
            "verdict": summary["verdict"],
            "artifact_sha256": {
                name: file_sha256(args.output_dir / name) for name in artifacts
            },
        },
    )
    print(f"[continuous-support-analysis] verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
