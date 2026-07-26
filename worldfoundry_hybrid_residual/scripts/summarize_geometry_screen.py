#!/usr/bin/env python3
"""Summarize and visualize a geometry sparse-attention screening run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict-error", type=float, default=0.02)
    parser.add_argument("--relaxed-error", type=float, default=0.05)
    parser.add_argument("--max-execution-density", type=float, default=0.125)
    parser.add_argument("--max-dense-heads", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, int]:
    return row["mask"], int(row["tail_rank"])


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.strict_error < args.relaxed_error:
        raise ValueError("error targets must satisfy 0 < strict < relaxed")
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    heads = read_rows(input_dir / "geometry_attention_heads.csv")
    tails = read_rows(input_dir / "geometry_attention_oracle_tail.csv")
    curves = read_rows(input_dir / "geometry_attention_fallback_curve.csv")
    gates = read_rows(input_dir / "geometry_attention_gates.csv")
    if not heads or not tails or not curves or not gates:
        raise RuntimeError("geometry screening CSV files are incomplete")

    head_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    tail_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    curve_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    gate_groups: dict[tuple[str, int, float], dict[str, str]] = {}
    for row in heads:
        head_groups[row["mask"]].append(row)
    for row in tails:
        tail_groups[key(row)].append(row)
    for row in curves:
        curve_groups[key(row)].append(row)
    for row in gates:
        gate_groups[(row["mask"], int(row["tail_rank"]), float(row["error_target"]))] = row

    summary_rows: list[dict[str, object]] = []
    for mask, rank in sorted(tail_groups):
        mask_heads = head_groups[mask]
        tail_rows = tail_groups[(mask, rank)]
        no_fallback = next(
            row
            for row in curve_groups[(mask, rank)]
            if int(row["dense_fallback_heads"]) == 0
        )
        row: dict[str, object] = {
            "mask": mask,
            "tail_rank": rank,
            "oracle_only": rank > 0,
            "execution_density": float(no_fallback["effective_attention_density"]),
            "no_fallback_output_relative_l2": float(no_fallback["output_relative_l2"]),
            "no_fallback_output_cosine": float(no_fallback["output_cosine"]),
            "mean_exact_attention_mass": mean(
                [float(item["exact_attention_mass_mean"]) for item in mask_heads]
            ),
            "static_sparse_heads": sum(
                item["static_sparse_head_gate"].lower() == "true"
                for item in mask_heads
            ),
            "mean_tail_defect_energy_explained": mean(
                [float(item["tail_defect_energy_explained"]) for item in tail_rows]
            ),
        }
        for label, target in (
            ("strict", args.strict_error),
            ("relaxed", args.relaxed_error),
        ):
            gate = gate_groups[(mask, rank, target)]
            density = float(gate["effective_attention_density"])
            dense_heads = int(gate["dense_fallback_heads"])
            row[f"{label}_target"] = target
            row[f"{label}_dense_fallback_heads"] = dense_heads
            row[f"{label}_effective_density"] = density
            row[f"{label}_output_relative_l2"] = float(gate["output_relative_l2"])
            row[f"{label}_budget_gate"] = (
                dense_heads <= args.max_dense_heads
                and density <= args.max_execution_density
            )
        summary_rows.append(row)

    strict_feasible = [row for row in summary_rows if bool(row["strict_budget_gate"])]
    relaxed_feasible = [row for row in summary_rows if bool(row["relaxed_budget_gate"])]
    strict_best = min(
        summary_rows, key=lambda row: float(row["strict_effective_density"])
    )
    relaxed_best = min(
        summary_rows, key=lambda row: float(row["relaxed_effective_density"])
    )
    decision = {
        "scope": "single-replay sampled pre-output-projection geometry screen",
        "strict_error_target": args.strict_error,
        "relaxed_error_target": args.relaxed_error,
        "max_execution_density": args.max_execution_density,
        "max_dense_heads": args.max_dense_heads,
        "strict_go": bool(strict_feasible),
        "strict_feasible_candidates": strict_feasible,
        "strict_lowest_density_candidate": strict_best,
        "relaxed_oracle_go": bool(relaxed_feasible),
        "relaxed_feasible_candidates": relaxed_feasible,
        "relaxed_lowest_density_candidate": relaxed_best,
        "rank0_static_heads_passing": sum(
            int(row["static_sparse_heads"])
            for row in summary_rows
            if int(row["tail_rank"]) == 0
        ),
        "warning": (
            "rank > 0 uses a current-replay activation-defect SVD oracle; it is "
            "not a deployable cross-prompt low-rank correction"
        ),
    }
    write_rows(output_dir / "geometry_screen_summary.csv", summary_rows)
    (output_dir / "geometry_screen_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    masks = sorted(
        head_groups,
        key=lambda mask: float(
            next(
                row["execution_density"]
                for row in summary_rows
                if row["mask"] == mask and int(row["tail_rank"]) == 0
            )
        ),
    )
    ranks = sorted({int(row["tail_rank"]) for row in summary_rows})
    colors = {0: "#7A271A", 8: "#B54708", 16: "#027A48"}
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    figure.patch.set_facecolor("#F7F4ED")
    for axis in axes:
        axis.set_facecolor("#FFFCF5")

    axis = axes[0]
    for rank in ranks:
        selected = [
            next(
                row
                for row in summary_rows
                if row["mask"] == mask and int(row["tail_rank"]) == rank
            )
            for mask in masks
        ]
        axis.plot(
            [float(row["execution_density"]) for row in selected],
            [float(row["no_fallback_output_relative_l2"]) for row in selected],
            marker="o",
            linewidth=1.8,
            color=colors.get(rank),
            label=f"rank {rank}" + (" oracle" if rank else ""),
        )
    axis.axhline(args.strict_error, color="#B42318", linestyle="--", linewidth=1.2)
    axis.axhline(args.relaxed_error, color="#B54708", linestyle=":", linewidth=1.2)
    axis.set_xlabel("64x64 execution density")
    axis.set_ylabel("No-fallback relative L2")
    axis.set_yscale("log")
    axis.legend(fontsize=8)

    axis = axes[1]
    positions = range(len(masks))
    width = 0.24
    for rank_index, rank in enumerate(ranks):
        selected = [
            next(
                row
                for row in summary_rows
                if row["mask"] == mask and int(row["tail_rank"]) == rank
            )
            for mask in masks
        ]
        offsets = [
            position + (rank_index - (len(ranks) - 1) / 2) * width
            for position in positions
        ]
        axis.bar(
            offsets,
            [float(row["strict_effective_density"]) for row in selected],
            width=width,
            color=colors.get(rank),
            alpha=0.82,
            label=f"rank {rank}" + (" oracle" if rank else ""),
        )
    axis.axhline(args.max_execution_density, color="#B42318", linestyle="--", linewidth=1.2)
    axis.set_xticks(list(positions), masks, rotation=28, ha="right", fontsize=8)
    axis.set_ylabel("Density needed for 2% error")
    axis.set_ylim(0.0, 1.04)
    axis.legend(fontsize=8)

    axis = axes[2]
    rank16 = [
        next(
            row
            for row in summary_rows
            if row["mask"] == mask and int(row["tail_rank"]) == 16
        )
        for mask in masks
    ]
    axis.bar(
        list(positions),
        [float(row["mean_tail_defect_energy_explained"]) for row in rank16],
        color="#1570EF",
        alpha=0.84,
    )
    axis.set_xticks(list(positions), masks, rotation=28, ha="right", fontsize=8)
    axis.set_ylabel("Rank-16 local defect energy explained")
    axis.set_ylim(0.75, 1.0)
    axis.axhline(0.95, color="#027A48", linestyle="--", linewidth=1.2)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_dir / "geometry_screen_decision.png", dpi=220)
    figure.savefig(output_dir / "geometry_screen_decision.pdf")
    plt.close(figure)

    report = f"""# F81 Geometry Sparse Attention Screen

## Decision

- Strict 2% gate: `{'GO' if strict_feasible else 'NO-GO'}`.
- Lowest density at 2%: `{strict_best['mask']}`, rank `{strict_best['tail_rank']}`, density `{float(strict_best['strict_effective_density']):.4f}`, dense fallback `{strict_best['strict_dense_fallback_heads']}/12`.
- Relaxed 5% oracle gate: `{'GO' if relaxed_feasible else 'NO-GO'}`.
- Lowest density at 5%: `{relaxed_best['mask']}`, rank `{relaxed_best['tail_rank']}`, density `{float(relaxed_best['relaxed_effective_density']):.4f}`, dense fallback `{relaxed_best['relaxed_dense_fallback_heads']}/12`.
- Rank-0 heads passing the static local gate across all masks: `{decision['rank0_static_heads_passing']}`.

## Interpretation

The fixed geometry masks do not satisfy the strict local error budget. Rank-16 can explain a large fraction of the current replay's activation defect and reaches the relaxed target for one candidate, but this is an input-specific SVD oracle. It does not establish a reusable low-rank basis, fused-kernel speedup, or end-to-end video quality.
"""
    (output_dir / "geometry_screen_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
