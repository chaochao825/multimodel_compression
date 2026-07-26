#!/usr/bin/env python3
"""Analyze why the registered Nystrom/landmark sparse-tail pilot failed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    require_fresh_output_dir,
)


DEPLOYABLE = frozenset(
    {
        "nystrom_signed",
        "landmark_linear",
        "proxy_mass_nystrom_mixture",
        "proxy_mass_landmark_partition",
    }
)
NUMERIC_FIELDS = (
    "residual_sq",
    "reference_sq",
    "output_relative_l2",
    "projected_attention_work_ratio",
    "arithmetic_speedup_upper_bound",
    "selected_attention_mass",
    "proxy_selected_mass",
    "mass_absolute_error",
    "negative_absolute_mass_ratio",
    "middle_condition_number",
    "middle_effective_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--transitional-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"empty CSV: {path}")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        for field in NUMERIC_FIELDS:
            row[field] = float(raw[field])
        for field in ("landmarks", "head", "sampling_step", "layer"):
            row[field] = int(raw[field])
        row["density"] = float(raw["density"])
        row["deployable_candidate"] = raw["deployable_candidate"].lower() == "true"
        rows.append(row)
    return rows


def config_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["method"],
        row["landmark_mode"],
        row["landmarks"],
        row["pinv_rtol"],
        row["density"],
    )


def aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    residual = sum(float(row["residual_sq"]) for row in rows)
    reference = sum(float(row["reference_sq"]) for row in rows)
    work = statistics.mean(
        float(row["projected_attention_work_ratio"]) for row in rows
    )
    return {
        "records": len(rows),
        "aggregate_error": math.sqrt(residual / max(reference, 1e-30)),
        "worst_error": max(float(row["output_relative_l2"]) for row in rows),
        "work_ratio": work,
        "speedup_upper_bound": 1.0 / work,
        "mass_absolute_error_mean": statistics.mean(
            float(row["mass_absolute_error"]) for row in rows
        ),
    }


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires paired values")
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0 else float("nan")


def validate_inputs(
    probe_dir: Path, selection_dir: Path, transitional_selection_dir: Path
) -> tuple[Path, dict[str, object], dict[str, object]]:
    detail_path = probe_dir / "nystrom_sparse_tail_heads.csv"
    probe_success = json.loads((probe_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    selection_manifest = json.loads(
        (selection_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    transitional_manifest = json.loads(
        (transitional_selection_dir / "selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if probe_success.get("status") != "SUCCESS":
        raise ValueError("probe is not complete")
    detail_hash = file_sha256(detail_path)
    for label, manifest in (
        ("all-head", selection_manifest),
        ("transitional", transitional_manifest),
    ):
        if manifest.get("probe_detail_sha256") != detail_hash:
            raise ValueError(f"{label} selector does not reference this probe detail")
    return detail_path, selection_manifest, transitional_manifest


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    transitional_dir = args.transitional_selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    detail_path, selection_manifest, transitional_manifest = validate_inputs(
        probe_dir, selection_dir, transitional_dir
    )
    rows = read_rows(detail_path)
    require_fresh_output_dir(output_dir)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[config_key(row)].append(row)
    config_metrics = []
    for key, group in grouped.items():
        metric = aggregate(group)
        config_metrics.append(
            {
                "method": key[0],
                "landmark_mode": key[1],
                "landmarks": key[2],
                "pinv_rtol": key[3],
                "density": key[4],
                "deployable": key[0] in DEPLOYABLE,
                **metric,
            }
        )
    eligible = [
        row
        for row in config_metrics
        if row["deployable"]
        and float(row["work_ratio"]) <= 0.5
        and float(row["speedup_upper_bound"]) >= 1.5
    ]
    if not eligible:
        raise ValueError("no deployable candidate satisfies the arithmetic cost gate")
    best = min(
        eligible,
        key=lambda row: (
            float(row["aggregate_error"]),
            float(row["worst_error"]),
            float(row["work_ratio"]),
        ),
    )
    best_key = (
        best["method"],
        best["landmark_mode"],
        best["landmarks"],
        str(best["pinv_rtol"]),
        best["density"],
    )
    best_rows = []
    for key, group in grouped.items():
        normalized_key = (key[0], key[1], key[2], str(key[3]), key[4])
        if normalized_key == best_key:
            best_rows = group
            break
    if not best_rows:
        raise ValueError(f"could not recover rows for best config: {best_key}")

    capacity_rows = [
        row
        for row in config_metrics
        if str(row["method"]).startswith("proxy_mass_")
    ]
    role_rows = []
    by_role: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        by_role[str(row["head_role_diagnostic_only"])].append(row)
    for role, group in sorted(by_role.items()):
        role_rows.append({"role": role, **aggregate(group)})

    mass_rows = [
        {
            "sample_id": row["sample_id"],
            "head": row["head"],
            "role": row["head_role_diagnostic_only"],
            "output_error": row["output_relative_l2"],
            "mass_absolute_error": row["mass_absolute_error"],
            "selected_attention_mass": row["selected_attention_mass"],
            "proxy_selected_mass": row["proxy_selected_mass"],
        }
        for row in best_rows
    ]
    mass_correlation = pearson(
        [float(row["mass_absolute_error"]) for row in mass_rows],
        [float(row["output_error"]) for row in mass_rows],
    )

    condition_rows = []
    for landmarks in sorted({int(row["landmarks"]) for row in rows}):
        subset = [
            row
            for row in rows
            if row["method"] == "nystrom_signed"
            and int(row["landmarks"]) == landmarks
        ]
        condition_rows.append(
            {
                "landmarks": landmarks,
                "effective_rank_mean": statistics.mean(
                    float(row["middle_effective_rank"]) for row in subset
                ),
                "condition_number_median": statistics.median(
                    float(row["middle_condition_number"]) for row in subset
                ),
                "condition_number_max": max(
                    float(row["middle_condition_number"]) for row in subset
                ),
                "negative_mass_ratio_mean": statistics.mean(
                    float(row["negative_absolute_mass_ratio"]) for row in subset
                ),
            }
        )

    samples = sorted({str(row["sample_id"]) for row in best_rows})
    heads = sorted({int(row["head"]) for row in best_rows})
    lookup = {
        (str(row["sample_id"]), int(row["head"])): float(
            row["output_relative_l2"]
        )
        for row in best_rows
    }
    heatmap_rows = [
        {
            "sample_id": sample,
            "head": head,
            "output_error": lookup[(sample, head)],
        }
        for sample in samples
        for head in heads
    ]

    routing_path = transitional_dir / "frozen_head_routing.csv"
    routing = list(csv.DictReader(routing_path.open(encoding="utf-8", newline="")))
    selected_routing = [
        row
        for row in routing
        if row["selected_by_calibration"].lower() == "true"
    ]
    transfer_rows = []
    for split in ("calibration", "validation", "test"):
        subset = [row for row in selected_routing if row["split"] == split]
        transfer_rows.append(
            {
                "split": split,
                "records": len(subset),
                "target_role_agreement": statistics.mean(
                    row["target_role_match_diagnostic_only"].lower() == "true"
                    for row in subset
                ),
            }
        )
    with (transitional_dir / "selected_protocol_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        transitional_selected = list(csv.DictReader(handle))
    transitional_errors = [
        float(row["aggregate_output_relative_l2"])
        for row in transitional_selected
        if row["split"] in {"validation", "test"}
    ]

    atomic_write_csv(output_dir / "config_metrics.csv", config_metrics)
    atomic_write_csv(output_dir / "capacity_grid.csv", capacity_rows)
    atomic_write_csv(output_dir / "role_metrics.csv", role_rows)
    atomic_write_csv(output_dir / "mass_error_scatter.csv", mass_rows)
    atomic_write_csv(output_dir / "conditioning.csv", condition_rows)
    atomic_write_csv(output_dir / "head_error_heatmap.csv", heatmap_rows)
    atomic_write_csv(output_dir / "frozen_role_transfer.csv", transfer_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1), constrained_layout=True)
    ax_capacity, ax_pareto, ax_role, ax_mass, ax_heatmap, ax_condition = axes.flat

    colors = {
        "proxy_mass_nystrom_mixture": "#D55E00",
        "proxy_mass_landmark_partition": "#0072B2",
    }
    for method in sorted({str(row["method"]) for row in capacity_rows}):
        for density in sorted(
            {float(row["density"]) for row in capacity_rows if row["method"] == method}
        ):
            subset = sorted(
                [
                    row
                    for row in capacity_rows
                    if row["method"] == method
                    and float(row["density"]) == density
                ],
                key=lambda row: int(row["landmarks"]),
            )
            ax_capacity.plot(
                [int(row["landmarks"]) for row in subset],
                [100 * float(row["aggregate_error"]) for row in subset],
                marker="o",
                color=colors[method],
                linestyle="-" if density == 0.25 else "--",
                label=f"{method.replace('proxy_mass_', '').replace('_', ' ')}; {density:.1%}",
            )
    ax_capacity.axhline(1.0, color="#B91C1C", linestyle=":", linewidth=1)
    ax_capacity.set_xlabel("Landmarks")
    ax_capacity.set_ylabel("Aggregate output error (%)")
    ax_capacity.legend(frameon=False, fontsize=6.4)
    ax_capacity.grid(alpha=0.18)

    for row in config_metrics:
        ax_pareto.scatter(
            float(row["work_ratio"]),
            100 * float(row["aggregate_error"]),
            s=32,
            facecolor="#D55E00" if row["deployable"] else "none",
            edgecolor="#D55E00" if row["deployable"] else "#6B7280",
            alpha=0.8,
        )
    ax_pareto.scatter(
        float(best["work_ratio"]),
        100 * float(best["aggregate_error"]),
        s=90,
        marker="*",
        color="#111827",
        label="Post-hoc diagnostic best",
    )
    ax_pareto.axhline(1.0, color="#B91C1C", linestyle=":", linewidth=1)
    ax_pareto.axvline(0.5, color="#B91C1C", linestyle=":", linewidth=1)
    ax_pareto.set_xlabel("Projected Attention work / dense")
    ax_pareto.set_ylabel("Aggregate output error (%)")
    ax_pareto.set_yscale("log")
    ax_pareto.legend(frameon=False)
    ax_pareto.grid(alpha=0.18)

    role_names = [str(row["role"]) for row in role_rows]
    positions = np.arange(len(role_names))
    ax_role.bar(
        positions - 0.18,
        [100 * float(row["aggregate_error"]) for row in role_rows],
        width=0.36,
        color="#0072B2",
        label="Aggregate",
    )
    ax_role.bar(
        positions + 0.18,
        [100 * float(row["worst_error"]) for row in role_rows],
        width=0.36,
        color="#E69F00",
        label="Worst",
    )
    ax_role.axhline(2.0, color="#B91C1C", linestyle=":", linewidth=1)
    ax_role.set_xticks(positions, role_names)
    ax_role.set_ylabel("Best-config output error (%)")
    ax_role.legend(frameon=False)
    ax_role.grid(axis="y", alpha=0.18)

    role_colors = {"localized": "#009E73", "transitional": "#E69F00", "diffuse": "#D55E00"}
    for role in sorted({str(row["role"]) for row in mass_rows}):
        subset = [row for row in mass_rows if row["role"] == role]
        ax_mass.scatter(
            [100 * float(row["mass_absolute_error"]) for row in subset],
            [100 * float(row["output_error"]) for row in subset],
            color=role_colors.get(role, "#6B7280"),
            label=role,
            alpha=0.82,
        )
    ax_mass.set_xlabel("Router selected-mass absolute error (%)")
    ax_mass.set_ylabel("Output error (%)")
    ax_mass.text(
        0.98,
        0.04,
        f"Pearson r = {mass_correlation:.2f}",
        transform=ax_mass.transAxes,
        ha="right",
        va="bottom",
    )
    ax_mass.legend(frameon=False)
    ax_mass.grid(alpha=0.18)

    matrix = np.array(
        [[100 * lookup[(sample, head)] for head in heads] for sample in samples]
    )
    image = ax_heatmap.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax_heatmap.set_xticks(range(len(heads)), heads)
    ax_heatmap.set_yticks(
        range(len(samples)), [sample.split("_")[0] for sample in samples]
    )
    ax_heatmap.set_xlabel("Head")
    ax_heatmap.set_ylabel("Sample")
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03)
    colorbar.set_label("Output error (%)")

    ax_condition.plot(
        [int(row["landmarks"]) for row in condition_rows],
        [float(row["condition_number_median"]) for row in condition_rows],
        marker="o",
        color="#0072B2",
        label="Median condition number",
    )
    ax_condition.set_yscale("log")
    ax_condition.set_xlabel("Landmarks")
    ax_condition.set_ylabel("Middle-matrix condition number")
    ax_condition.grid(alpha=0.18)
    negative_axis = ax_condition.twinx()
    negative_axis.plot(
        [int(row["landmarks"]) for row in condition_rows],
        [100 * float(row["negative_mass_ratio_mean"]) for row in condition_rows],
        marker="s",
        linestyle="--",
        color="#D55E00",
        label="Negative mass",
    )
    negative_axis.set_ylabel("Signed approximation negative mass (%)")
    handles_left, labels_left = ax_condition.get_legend_handles_labels()
    handles_right, labels_right = negative_axis.get_legend_handles_labels()
    ax_condition.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=7,
    )

    for label, axis in zip("ABCDEF", axes.flat):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    png_path = output_dir / "nystrom_sparse_tail_failure_analysis.png"
    pdf_path = output_dir / "nystrom_sparse_tail_failure_analysis.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    transfer_test = next(
        row for row in transfer_rows if row["split"] == "test"
    )
    findings = {
        "schema_version": 2,
        "gate": "FAIL",
        "bounded_decision": (
            "stop expanding this train-free Nystrom/landmark family; learned "
            "content-conditioned tails remain untested"
        ),
        "posthoc_full_pilot_best_deployable_diagnostic": best,
        "mass_error_output_error_pearson": mass_correlation,
        "calibration_frozen_transitional_role_test_agreement": transfer_test[
            "target_role_agreement"
        ],
        "calibration_frozen_transitional_aggregate_error_min": min(
            transitional_errors
        ),
        "calibration_frozen_transitional_aggregate_error_max": max(
            transitional_errors
        ),
        "role_metrics": role_rows,
        "conditioning": condition_rows,
        "interpretation": [
            "Increasing landmarks and density helps but remains far above the 1%/2% gates.",
            "Diffuse heads are worst, but calibration-frozen transitional heads also fail by a wide margin.",
            "Conditioning and signed negative mass worsen as landmark count increases.",
            "Selected-mass error is only one contributor; dense-reference mass is not a quality oracle.",
        ],
    }
    atomic_write_json(output_dir / "findings.json", findings)
    markdown = f"""# Nystrom / Landmark Sparse-Tail Failure Analysis

The registered pilot fails. A post-hoc search over all four captures
(diagnostic only, not a validation-frozen test estimate) finds
`{best['method']}` with `m={best['landmarks']}` and density
`{float(best['density']):.1%}`. Its aggregate error is
`{100 * float(best['aggregate_error']):.3f}%`, worst-record error is
`{100 * float(best['worst_error']):.3f}%`, and arithmetic speedup upper bound
is `{float(best['speedup_upper_bound']):.2f}x`.

The calibration-frozen transitional-head selection also fails at
`{100 * min(transitional_errors):.3f}-{100 * max(transitional_errors):.3f}%`
aggregate error. Therefore, the all-head failure is not explained
only by applying the tail to diffuse heads.

The selected-mass/output-error Pearson correlation is `{mass_correlation:.3f}`.
Increasing landmark count raises middle-matrix condition number and signed
negative mass while delivering only modest quality improvement. These results
support stopping this train-free Nystrom/landmark family at the registered
capacity. They do not test a learned content-conditioned tail.

All speed values are arithmetic upper bounds. No H200 latency claim is made.
"""
    atomic_write_text(output_dir / "report.md", markdown)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "probe_run_id": selection_manifest["probe_run_id"],
            "all_head_selection_run_id": selection_manifest["run_id"],
            "transitional_selection_run_id": transitional_manifest["run_id"],
            "probe_detail_sha256": file_sha256(detail_path),
            "figure_png_sha256": file_sha256(png_path),
            "figure_pdf_sha256": file_sha256(pdf_path),
            "analysis_scope": "registered layer14/step9/cond pilot only",
        },
    )
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "SUCCESS", "gate": "FAIL", "figure": png_path.name},
    )
    print(f"[failure-analysis] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
