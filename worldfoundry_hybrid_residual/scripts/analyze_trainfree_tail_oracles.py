#!/usr/bin/env python3
"""Create source-bound diagnostics for the stronger train-free tail oracles."""

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
from matplotlib.colors import LogNorm

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    require_fresh_output_dir,
)


FAMILY_ORDER = [
    "value_aware_coreset",
    "residual_tail_polynomial",
    "lowrank_covariance",
]
FAMILY_LABELS = {
    "value_aware_coreset": "value-aware coreset",
    "residual_tail_polynomial": "tail polynomial",
    "lowrank_covariance": "covariance moments",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: number(row, "aggregate_output_relative_l2"))


def config_label(row: dict[str, str]) -> str:
    family = row["family"]
    if family == "value_aware_coreset":
        return (
            f"{row['variant']}, m={int(float(row['landmarks']))}, "
            f"density={100 * number(row, 'density'):.1f}%"
        )
    if family == "residual_tail_polynomial":
        return (
            f"{row['variant']}, order={int(float(row['order']))}, "
            f"density={100 * number(row, 'density'):.1f}%"
        )
    return (
        f"{row['variant']}, rank={int(float(row['rank']))}, "
        f"components={int(float(row['components']))}, "
        f"density={100 * number(row, 'density'):.1f}%"
    )


def grouped(
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        output.setdefault(tuple(row[field] for field in fields), []).append(row)
    return output


def aggregate_envelope_records(
    rows: list[dict[str, str]],
    group_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """Aggregate already selected sample/head envelope records without reselection."""

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        residual_sq = sum(number(row, "residual_sq") for row in records)
        reference_sq = sum(number(row, "reference_sq") for row in records)
        output.append(
            {
                **dict(zip(group_fields, key)),
                "records": len(records),
                "residual_sq": residual_sq,
                "reference_sq": reference_sq,
                "aggregate_output_relative_l2": math.sqrt(
                    residual_sq / max(reference_sq, 1e-30)
                ),
                "record_error_max": max(
                    number(row, "aggregate_output_relative_l2") for row in records
                ),
                "selected_attention_mass_mean": float(
                    np.mean(
                        [number(row, "selected_attention_mass_mean") for row in records]
                    )
                ),
            }
        )
    return output


def candidate_counts(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter(
        (row["family"], config_label(row)) for row in rows
    )
    return [
        {"family": family, "configuration": configuration, "records": records}
        for (family, configuration), records in sorted(
            counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        )
    ]


def _annotated_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="YlOrRd",
        norm=LogNorm(vmin=max(0.5, float(values.min())), vmax=float(values.max())),
    )
    axis.set_yticks(np.arange(len(FAMILY_ORDER)), [FAMILY_LABELS[name] for name in FAMILY_ORDER])
    axis.set_xticks(np.arange(values.shape[1]), [str(index) for index in range(values.shape[1])])
    axis.set_xlabel("Attention head")
    axis.set_title(title, loc="left", fontweight="bold")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= np.sqrt(values.min() * values.max()) else "#111827",
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)


def plot_head_diagnostics(
    envelope_heads: list[dict[str, str]],
    by_head: list[dict[str, object]],
    by_sample: list[dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    head_lookup = {
        (str(row["family"]), int(str(row["head"]))): row for row in by_head
    }
    aggregate = np.array(
        [
            [
                100 * float(head_lookup[(family, head)]["aggregate_output_relative_l2"])
                for head in range(12)
            ]
            for family in FAMILY_ORDER
        ]
    )
    worst = np.array(
        [
            [
                100 * float(head_lookup[(family, head)]["record_error_max"])
                for head in range(12)
            ]
            for family in FAMILY_ORDER
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    _annotated_heatmap(
        axes[0, 0],
        aggregate,
        title="A  Aggregate error by head",
        colorbar_label="Relative L2 (%)",
    )
    _annotated_heatmap(
        axes[0, 1],
        worst,
        title="B  Worst sample error by head",
        colorbar_label="Relative L2 (%)",
    )

    sample_ids = sorted({str(row["sample_id"]) for row in by_sample})
    sample_lookup = {
        (str(row["family"]), str(row["sample_id"])): row for row in by_sample
    }
    x = np.arange(len(sample_ids))
    width = 0.24
    colors = ["#D55E00", "#009E73", "#0072B2"]
    for index, family in enumerate(FAMILY_ORDER):
        axes[1, 0].bar(
            x + (index - 1) * width,
            [
                100
                * float(
                    sample_lookup[(family, sample_id)][
                        "aggregate_output_relative_l2"
                    ]
                )
                for sample_id in sample_ids
            ],
            width,
            color=colors[index],
            label=FAMILY_LABELS[family],
        )
    axes[1, 0].axhline(0.5, color="#B91C1C", linestyle=":", label="0.5% oracle gate")
    axes[1, 0].set_xticks(x, [sample_id.split("_")[0] for sample_id in sample_ids])
    axes[1, 0].set_ylabel("Aggregate output error (%)")
    axes[1, 0].set_title("C  Failure persists across all samples", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.18)

    mass_by_head = []
    best_error_by_head = []
    for head in range(12):
        records = [row for row in envelope_heads if int(float(row["head"])) == head]
        mass_by_head.append(
            100 * float(np.mean([number(row, "selected_attention_mass_mean") for row in records]))
        )
        best_error_by_head.append(float(aggregate[:, head].min()))
    mass_axis = axes[1, 1]
    error_axis = mass_axis.twinx()
    mass_axis.bar(np.arange(12), mass_by_head, color="#9CA3AF", alpha=0.72, label="critical mass")
    error_axis.plot(
        np.arange(12),
        best_error_by_head,
        color="#B91C1C",
        marker="o",
        linewidth=2,
        label="best train-free envelope",
    )
    error_axis.axhline(0.5, color="#B91C1C", linestyle=":")
    mass_axis.set_xticks(np.arange(12), [str(index) for index in range(12)])
    mass_axis.set_xlabel("Attention head")
    mass_axis.set_ylabel("Oracle critical attention mass (%)")
    error_axis.set_ylabel("Best aggregate output error (%)", color="#B91C1C")
    mass_axis.set_title("D  Critical mass does not certify tail quality", loc="left", fontweight="bold")
    handles_a, labels_a = mass_axis.get_legend_handles_labels()
    handles_b, labels_b = error_axis.get_legend_handles_labels()
    mass_axis.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, fontsize=8)
    mass_axis.grid(axis="y", alpha=0.18)

    fig.suptitle(
        "F81 train-free residual-tail failure concentration\n"
        "Post-hoc sample/head envelopes; no frozen deployment or latency claim",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = output_dir / "trainfree_tail_oracle_head_diagnostics.png"
    pdf = output_dir / "trainfree_tail_oracle_head_diagnostics.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_results(
    summary: list[dict[str, str]],
    envelope: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    ax_coreset, ax_poly, ax_cov, ax_gate, ax_pareto, ax_tail = axes.ravel()
    colors = {
        "k_only": "#6B7280",
        "joint_kv": "#0072B2",
        "value_aware_kv_thw": "#D55E00",
        "mean": "#009E73",
        "midrange": "#CC79A7",
    }

    coreset = [row for row in summary if row["family"] == "value_aware_coreset"]
    for (variant, density), rows in grouped(coreset, ("variant", "density")).items():
        rows = sorted(rows, key=lambda row: number(row, "landmarks"))
        ax_coreset.plot(
            [number(row, "landmarks") for row in rows],
            [100 * number(row, "aggregate_output_relative_l2") for row in rows],
            marker="o",
            color=colors.get(variant, "#333333"),
            linestyle="-" if math.isclose(float(density), 0.25) else "--",
            label=f"{variant}; {100 * float(density):.1f}%",
        )
    ax_coreset.axhline(0.5, color="#B91C1C", linestyle=":")
    ax_coreset.set_xlabel("Tail landmarks")
    ax_coreset.set_ylabel("Aggregate output error (%)")
    ax_coreset.set_title("A  Value-aware coreset capacity", loc="left", fontweight="bold")
    ax_coreset.legend(frameon=False, fontsize=7)
    ax_coreset.grid(alpha=0.18)

    polynomial = [
        row for row in summary if row["family"] == "residual_tail_polynomial"
    ]
    for (center, density), rows in grouped(polynomial, ("variant", "density")).items():
        rows = sorted(rows, key=lambda row: number(row, "order"))
        ax_poly.plot(
            [number(row, "order") for row in rows],
            [100 * number(row, "aggregate_output_relative_l2") for row in rows],
            marker="o",
            color=colors.get(center, "#333333"),
            linestyle="-" if math.isclose(float(density), 0.25) else "--",
            label=f"{center}; {100 * float(density):.1f}%",
        )
    ax_poly.axhline(0.5, color="#B91C1C", linestyle=":")
    ax_poly.set_xlabel("Taylor order")
    ax_poly.set_ylabel("Aggregate output error (%)")
    ax_poly.set_yscale("log")
    ax_poly.set_title("B  Residual-tail polynomial", loc="left", fontweight="bold")
    ax_poly.legend(frameon=False, fontsize=8)
    ax_poly.grid(alpha=0.18)

    covariance = [row for row in summary if row["family"] == "lowrank_covariance"]
    lowrank = [row for row in covariance if row["variant"] == "lowrank_gaussian"]
    for (components, density), rows in grouped(lowrank, ("components", "density")).items():
        rows = sorted(rows, key=lambda row: number(row, "rank"))
        ax_cov.plot(
            [number(row, "rank") for row in rows],
            [100 * number(row, "aggregate_output_relative_l2") for row in rows],
            marker="o",
            linestyle="-" if math.isclose(float(density), 0.25) else "--",
            label=f"components={components}; {100 * float(density):.1f}%",
        )
    baselines = [row for row in covariance if row["variant"] != "lowrank_gaussian"]
    for variant in ("centroid", "diag_gaussian"):
        candidates = [row for row in baselines if row["variant"] == variant]
        if candidates:
            row = best_row(candidates)
            ax_cov.scatter(
                [0],
                [100 * number(row, "aggregate_output_relative_l2")],
                marker="x" if variant == "centroid" else "+",
                s=70,
                label=f"best {variant}",
            )
    ax_cov.axhline(0.5, color="#B91C1C", linestyle=":")
    ax_cov.set_xlabel("Covariance rank")
    ax_cov.set_ylabel("Aggregate output error (%)")
    ax_cov.set_yscale("log")
    ax_cov.set_title("C  Full-covariance moments", loc="left", fontweight="bold")
    ax_cov.legend(frameon=False, fontsize=7)
    ax_cov.grid(alpha=0.18)

    envelope_by_family = {row["family"]: row for row in envelope}
    x = np.arange(len(FAMILY_ORDER))
    aggregate = [
        100 * number(envelope_by_family[family], "aggregate_output_relative_l2")
        for family in FAMILY_ORDER
    ]
    worst = [
        100 * number(envelope_by_family[family], "record_error_max")
        for family in FAMILY_ORDER
    ]
    ax_gate.bar(x - 0.18, aggregate, 0.36, label="Post-hoc aggregate")
    ax_gate.bar(x + 0.18, worst, 0.36, label="Post-hoc worst")
    ax_gate.axhline(0.5, color="#B91C1C", linestyle=":", label="0.5% aggregate gate")
    ax_gate.axhline(1.0, color="#B91C1C", linestyle="--", label="1% worst gate")
    ax_gate.set_xticks(x, ["coreset", "polynomial", "covariance"])
    ax_gate.set_ylabel("Output error (%)")
    ax_gate.set_title("D  Per-record oracle envelopes", loc="left", fontweight="bold")
    ax_gate.legend(frameon=False, fontsize=7)
    ax_gate.grid(axis="y", alpha=0.18)

    deployable_work = []
    for row in summary:
        work = number(row, "projected_query_work_ratio_mean")
        if math.isfinite(work):
            deployable_work.append(row)
    family_colors = {
        "value_aware_coreset": "#D55E00",
        "lowrank_covariance": "#0072B2",
    }
    for family in FAMILY_ORDER:
        rows = [row for row in deployable_work if row["family"] == family]
        if not rows:
            continue
        ax_pareto.scatter(
            [number(row, "projected_query_work_ratio_mean") for row in rows],
            [100 * number(row, "aggregate_output_relative_l2") for row in rows],
            alpha=0.65,
            s=32,
            label=family.replace("value_aware_", "").replace("lowrank_", ""),
            color=family_colors[family],
        )
    ax_pareto.axvline(0.5, color="#B91C1C", linestyle=":")
    ax_pareto.axhline(0.5, color="#B91C1C", linestyle=":")
    ax_pareto.set_xlabel("Projected query work / dense")
    ax_pareto.set_ylabel("Aggregate output error (%)")
    ax_pareto.set_yscale("log")
    ax_pareto.set_title("E  Arithmetic-quality screen", loc="left", fontweight="bold")
    ax_pareto.legend(frameon=False, fontsize=8)
    ax_pareto.grid(alpha=0.18)

    for (center, density), rows in grouped(polynomial, ("variant", "density")).items():
        rows = sorted(rows, key=lambda row: number(row, "order"))
        ax_tail.plot(
            [number(row, "order") for row in rows],
            [100 * number(row, "negative_tail_weight_fraction_mean") for row in rows],
            marker="o",
            color=colors.get(center, "#333333"),
            linestyle="-" if math.isclose(float(density), 0.25) else "--",
            label=f"negative: {center}; {100 * float(density):.1f}%",
        )
    ax_tail.set_xlabel("Taylor order")
    ax_tail.set_ylabel("Negative tail weights (%)")
    ax_tail.set_title("F  Polynomial validity diagnostic", loc="left", fontweight="bold")
    ax_tail.legend(frameon=False, fontsize=7)
    ax_tail.grid(alpha=0.18)

    fig.suptitle(
        "F81 stronger train-free residual-tail oracle screen\n"
        "All envelopes are post-hoc capacity diagnostics, not frozen test estimates",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = output_dir / "trainfree_tail_oracle_analysis.png"
    pdf = output_dir / "trainfree_tail_oracle_analysis.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir.resolve()
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    success = json.loads((probe_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    manifest = json.loads((probe_dir / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((probe_dir / "decision.json").read_text(encoding="utf-8"))
    if success.get("status") != "SUCCESS":
        raise ValueError("probe did not complete successfully")
    summary_path = probe_dir / "trainfree_tail_oracle_summary.csv"
    envelope_path = probe_dir / "trainfree_tail_oracle_envelope_summary.csv"
    summary = read_csv(summary_path)
    envelope = read_csv(envelope_path)
    envelope_heads_path = probe_dir / "trainfree_tail_oracle_envelope_heads.csv"
    envelope_heads = read_csv(envelope_heads_path)
    by_head = aggregate_envelope_records(envelope_heads, ("family", "head"))
    by_sample = aggregate_envelope_records(envelope_heads, ("family", "sample_id"))
    choices = candidate_counts(envelope_heads)
    fixed_best = [
        best_row([row for row in summary if row["family"] == family])
        for family in FAMILY_ORDER
    ]
    atomic_write_csv(output_dir / "plot_config_source.csv", summary)
    atomic_write_csv(output_dir / "plot_envelope_source.csv", envelope)
    atomic_write_csv(output_dir / "envelope_by_head.csv", by_head)
    atomic_write_csv(output_dir / "envelope_by_sample.csv", by_sample)
    atomic_write_csv(output_dir / "envelope_candidate_counts.csv", choices)
    atomic_write_csv(output_dir / "fixed_best_configurations.csv", fixed_best)
    png, pdf = plot_results(summary, envelope, output_dir, args.dpi)
    head_png, head_pdf = plot_head_diagnostics(
        envelope_heads, by_head, by_sample, output_dir, args.dpi
    )

    best = {row["family"]: row for row in fixed_best}
    envelope_by_family = {row["family"]: row for row in envelope}
    lines = [
        "# Stronger Train-Free Residual-Tail Oracle Report",
        "",
        f"Run kind: `{manifest['run_kind']}`.",
        "",
        "Every result below is a post-hoc function-class capacity diagnostic. Dense",
        "attention selects the critical mask, and each sample/head envelope may choose",
        "a different candidate. It is not a frozen test or deployment result.",
        "",
        "## Fixed-Configuration Diagnostics",
        "",
        "| Family | Registered-set post-hoc best fixed configuration | Aggregate | Worst |",
        "|---|---|---:|---:|",
    ]
    for family in sorted(best):
        row = best[family]
        lines.append(
            f"| {family} | {config_label(row)} | "
            f"{100 * number(row, 'aggregate_output_relative_l2'):.3f}% | "
            f"{100 * number(row, 'record_error_max'):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Per-Record Oracle Envelope",
            "",
            "| Family | Aggregate | Worst | Oracle gate | Query-work proxy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in sorted(envelope_by_family):
        row = envelope_by_family[family]
        work = number(row, "projected_query_work_ratio_mean")
        work_text = f"{work:.3f}" if math.isfinite(work) else "not implemented"
        lines.append(
            f"| {family} | {100 * number(row, 'aggregate_output_relative_l2'):.3f}% | "
            f"{100 * number(row, 'record_error_max'):.3f}% | "
            f"{row['oracle_quality_gate']} | {work_text} |"
        )
    lines.extend(
        [
            "",
            "## Sample-Wise Post-Hoc Envelope",
            "",
            "| Sample | Coreset | Polynomial | Covariance |",
            "|---|---:|---:|---:|",
        ]
    )
    sample_lookup = {
        (str(row["family"]), str(row["sample_id"])): row for row in by_sample
    }
    for sample_id in sorted({str(row["sample_id"]) for row in by_sample}):
        lines.append(
            f"| {sample_id} | "
            f"{100 * float(sample_lookup[('value_aware_coreset', sample_id)]['aggregate_output_relative_l2']):.3f}% | "
            f"{100 * float(sample_lookup[('residual_tail_polynomial', sample_id)]['aggregate_output_relative_l2']):.3f}% | "
            f"{100 * float(sample_lookup[('lowrank_covariance', sample_id)]['aggregate_output_relative_l2']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Failure Concentration",
            "",
            "| Family | Easiest head (aggregate) | Hardest head (aggregate / worst) | Records <= 1% |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in FAMILY_ORDER:
        family_heads = [row for row in by_head if row["family"] == family]
        easiest = min(family_heads, key=lambda row: float(row["aggregate_output_relative_l2"]))
        hardest = max(family_heads, key=lambda row: float(row["aggregate_output_relative_l2"]))
        passing_records = sum(
            number(row, "aggregate_output_relative_l2") <= 0.01
            for row in envelope_heads
            if row["family"] == family
        )
        lines.append(
            f"| {family} | h{easiest['head']}: "
            f"{100 * float(easiest['aggregate_output_relative_l2']):.3f}% | "
            f"h{hardest['head']}: {100 * float(hardest['aggregate_output_relative_l2']):.3f}% / "
            f"{100 * float(hardest['record_error_max']):.3f}% | {passing_records}/48 |"
        )
    polynomial = [
        row for row in summary if row["family"] == "residual_tail_polynomial"
    ]
    score_range = min(number(row, "tail_score_range_mean") for row in polynomial)
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            f"Probe decision: `{decision['status']}`.",
            "",
            f"The smallest recorded mean residual-tail score range is `{score_range:.3f}`.",
            "The residual score interval is therefore not narrow after removing the",
            "critical blocks; fourth-order Taylor approximation is still far outside",
            "the oracle quality gate. Odd midrange expansions also create signed tail",
            "weights and occasional non-positive shared denominators.",
            "",
            "The full-covariance Gaussian variants are numerically finite and their",
            "covariance products are covered by exact reconstruction tests, but they",
            "are usually worse than centroid/diagonal moments. This supports model",
            "mismatch rather than a missing covariance rank as the failure mechanism.",
            "",
            "Polynomial arithmetic speed is intentionally unreported because no",
            "TensorSketch/random-Maclaurin feature realization was implemented.",
            "Coreset query work omits dense-oracle leverage and clustering; covariance",
            "query work omits online moment formation and SVD. No H200 latency claim is",
            "authorized by this experiment.",
            "",
            "If all envelopes fail the 0.5% aggregate and 1% worst gate, stop the",
            "train-free tail family and move to a small learned Q/K-conditioned",
            "sparse-linear tail with frozen base QKV.",
        ]
    )
    atomic_write_text(output_dir / "report.md", "\n".join(lines) + "\n")
    analysis_manifest = {
        "schema_version": 1,
        "probe_dir": str(probe_dir),
        "probe_success_sha256": file_sha256(probe_dir / "SUCCESS.json"),
        "probe_manifest_sha256": file_sha256(probe_dir / "manifest.json"),
        "summary_sha256": file_sha256(summary_path),
        "envelope_sha256": file_sha256(envelope_path),
        "envelope_heads_sha256": file_sha256(envelope_heads_path),
        "run_kind": manifest["run_kind"],
        "decision": decision,
        "figure": png.name,
        "pdf": pdf.name,
        "head_figure": head_png.name,
        "head_pdf": head_pdf.name,
        "claim_boundary": "posthoc_oracle_capacity_only",
    }
    atomic_write_json(output_dir / "manifest.json", analysis_manifest)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "figure": png.name,
            "head_figure": head_png.name,
            "decision": decision["status"],
        },
    )
    print(json.dumps(analysis_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
