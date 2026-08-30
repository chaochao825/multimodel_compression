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


METHOD_LABELS = {
    "target_risk_run1": "Dense-gradient order (run 1)",
    "compact_contiguous": "Compact, contiguous positions",
    "compact_original_position": "Compact, original positions",
    "fixed_repeated": "Fixed length, repeated quotient",
}
COLORS = {
    "target_risk_run1": "#6B7280",
    "compact_contiguous": "#D55E00",
    "compact_original_position": "#0072B2",
    "fixed_repeated": "#009E73",
}
MARKERS = {
    "target_risk_run1": "o",
    "compact_contiguous": "s",
    "compact_original_position": "D",
    "fixed_repeated": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty plot-data CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frontiers(analysis_dir: Path) -> list[dict[str, object]]:
    target = json.loads(
        (
            analysis_dir
            / "target_risk_budget_frontier_exposed_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    geometry = json.loads(
        (
            analysis_dir
            / "group_compaction_geometry_exposed_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    for group_count_text, metrics in target["budget_summaries"].items():
        rows.append(
            {
                "method": "target_risk_run1",
                "refined_group_count": int(group_count_text),
                "token_retention": float(metrics["token_retention"]),
                "agreement": float(metrics["agreement"]),
                "mismatch_count": int(metrics["mismatch_count"]),
                "harmful_count": int(metrics["harmful_count"]),
                "candidate_kl_mean": float(metrics["candidate_kl_mean"]),
                "candidate_kl_p95": float(metrics["candidate_kl_p95"]),
            }
        )
    for method, summaries in geometry["layout_summaries"].items():
        for group_count_text, metrics in summaries.items():
            rows.append(
                {
                    "method": method,
                    "refined_group_count": int(group_count_text),
                    "token_retention": float(metrics["reader_token_retention"]),
                    "agreement": float(metrics["agreement"]),
                    "mismatch_count": int(metrics["mismatch_count"]),
                    "harmful_count": int(metrics["harmful_count"]),
                    "candidate_kl_mean": float(metrics["candidate_kl_mean"]),
                    "candidate_kl_p95": float(metrics["candidate_kl_p95"]),
                }
            )
    return sorted(
        rows,
        key=lambda row: (str(row["method"]), int(row["refined_group_count"])),
    )


def load_match_matrix(
    analysis_dir: Path,
) -> tuple[list[str], list[int], np.ndarray, list[dict[str, object]]]:
    rows = read_csv(
        analysis_dir
        / "group_compaction_geometry_exposed_v1"
        / "geometry_rows.csv"
    )
    positioned = [
        row for row in rows if row["layout"] == "compact_original_position"
    ]
    sample_ids = sorted(
        {row["sample_id"] for row in positioned},
        key=lambda sample_id: min(
            int(row["sample_position"])
            for row in positioned
            if row["sample_id"] == sample_id
        ),
    )
    group_counts = sorted({int(row["refined_group_count"]) for row in positioned})
    lookup = {
        (row["sample_id"], int(row["refined_group_count"])): int(
            row["prediction_match"]
        )
        for row in positioned
    }
    matrix = np.asarray(
        [[lookup[(sample_id, group_count)] for group_count in group_counts] for sample_id in sample_ids]
    )
    bound_rows = [
        {
            "sample_id": sample_id,
            "sample_order": sample_index + 1,
            "refined_group_count": group_count,
            "prediction_match": int(matrix[sample_index, group_index]),
        }
        for sample_index, sample_id in enumerate(sample_ids)
        for group_index, group_count in enumerate(group_counts)
    ]
    return sample_ids, group_counts, matrix, bound_rows


def load_run_instability(analysis_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frontier = read_csv(
        analysis_dir
        / "target_risk_budget_frontier_exposed_v1"
        / "frontier_rows.csv"
    )
    geometry = read_csv(
        analysis_dir
        / "group_compaction_geometry_exposed_v1"
        / "geometry_rows.csv"
    )
    contiguous = [row for row in geometry if row["layout"] == "compact_contiguous"]
    geometry_lookup = {
        (row["sample_id"], int(row["refined_group_count"])): row
        for row in contiguous
    }
    detail_rows: list[dict[str, object]] = []
    for first in frontier:
        key = (first["sample_id"], int(first["refined_group_count"]))
        second = geometry_lookup[key]
        detail_rows.append(
            {
                "sample_id": key[0],
                "refined_group_count": key[1],
                "run1_prediction_match": int(first["prediction_match"]),
                "run2_prediction_match": int(second["prediction_match"]),
                "prediction_changed": int(
                    int(first["approximate_index"])
                    != int(second["approximate_index"])
                ),
                "run1_candidate_kl": float(first["candidate_kl"]),
                "run2_candidate_kl": float(second["candidate_kl"]),
                "absolute_kl_delta": abs(
                    float(first["candidate_kl"])
                    - float(second["candidate_kl"])
                ),
            }
        )
    summary_rows = []
    for group_count in sorted({int(row["refined_group_count"]) for row in detail_rows}):
        rows = [
            row
            for row in detail_rows
            if int(row["refined_group_count"]) == group_count
        ]
        summary_rows.append(
            {
                "refined_group_count": group_count,
                "prediction_change_count": sum(
                    int(row["prediction_changed"]) for row in rows
                ),
                "mean_absolute_kl_delta": float(
                    np.mean([float(row["absolute_kl_delta"]) for row in rows])
                ),
                "maximum_absolute_kl_delta": max(
                    float(row["absolute_kl_delta"]) for row in rows
                ),
            }
        )
    return detail_rows, summary_rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = load_frontiers(args.analysis_dir)
    sample_ids, group_counts, match_matrix, match_rows = load_match_matrix(
        args.analysis_dir
    )
    instability_rows, instability_summary = load_run_instability(args.analysis_dir)
    write_csv(args.output_dir / "target_risk_compaction_frontiers.csv", frontier_rows)
    write_csv(args.output_dir / "positioned_path_match_matrix.csv", match_rows)
    write_csv(args.output_dir / "target_risk_run_instability.csv", instability_rows)
    write_csv(
        args.output_dir / "target_risk_run_instability_summary.csv",
        instability_summary,
    )

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        grouped[str(row["method"])].append(row)

    line_methods = (
        "target_risk_run1",
        "compact_contiguous",
        "compact_original_position",
    )
    for method in line_methods:
        rows = sorted(grouped[method], key=lambda row: int(row["refined_group_count"]))
        x = [100 * float(row["token_retention"]) for row in rows]
        panel_a.plot(
            x,
            [100 * float(row["agreement"]) for row in rows],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
        compressed_rows = [
            row for row in rows if int(row["refined_group_count"]) < 392
        ]
        compressed_x = [
            100 * float(row["token_retention"]) for row in compressed_rows
        ]
        panel_b.plot(
            compressed_x,
            [float(row["candidate_kl_mean"]) for row in compressed_rows],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=4.5,
            label=f"{METHOD_LABELS[method]} mean",
        )
        panel_b.plot(
            compressed_x,
            [float(row["candidate_kl_p95"]) for row in compressed_rows],
            color=COLORS[method],
            linestyle="--",
            linewidth=1.2,
            alpha=0.72,
        )

    panel_a.axhline(100, color="#222222", linewidth=0.8, linestyle=":")
    panel_a.set_xlabel("Reader token retention (%)")
    panel_a.set_ylabel("Dense-decision agreement (%)")
    panel_a.set_ylim(68, 102)
    panel_a.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    panel_a.legend(loc="lower right", frameon=False)

    panel_b.axhline(0.01, color="#222222", linewidth=0.8, linestyle=":")
    panel_b.axhline(0.02, color="#222222", linewidth=0.8, linestyle="-.")
    panel_b.set_xlabel("Reader token retention (%)")
    panel_b.set_ylabel("Candidate KL (solid mean, dashed P95)")
    panel_b.set_yscale("log")
    panel_b.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    image = panel_c.imshow(
        match_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#D55E00", "#009E73"]),
        vmin=0,
        vmax=1,
    )
    del image
    panel_c.set_xlabel("Exact four-token groups")
    panel_c.set_ylabel("Exposed question order")
    panel_c.set_xticks(range(len(group_counts)), labels=group_counts, rotation=45)
    panel_c.set_yticks([0, 5, 11, 17, 23], labels=[1, 6, 12, 18, 24])
    panel_c.spines["top"].set_visible(True)
    panel_c.spines["right"].set_visible(True)

    budget = 196
    bar_methods = ("compact_contiguous", "compact_original_position", "fixed_repeated")
    budget_rows = {
        method: next(
            row
            for row in grouped[method]
            if int(row["refined_group_count"]) == budget
        )
        for method in bar_methods
    }
    positions = np.arange(len(bar_methods))
    agreements = [100 * float(budget_rows[method]["agreement"]) for method in bar_methods]
    bars = panel_d.bar(
        positions,
        agreements,
        color=[COLORS[method] for method in bar_methods],
        width=0.62,
    )
    panel_d.set_xticks(
        positions,
        labels=["Contiguous\n62.5% tokens", "Original pos.\n62.5% tokens", "Repeated\n100% slots"],
    )
    panel_d.set_ylabel("Dense-decision agreement (%)")
    panel_d.set_ylim(84, 102)
    panel_d.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    for bar, agreement, method in zip(bars, agreements, bar_methods):
        mean_kl = float(budget_rows[method]["candidate_kl_mean"])
        p95_kl = float(budget_rows[method]["candidate_kl_p95"])
        panel_d.text(
            bar.get_x() + bar.get_width() / 2,
            agreement + 0.45,
            f"{agreement:.1f}%\nKL {mean_kl:.3f}/{p95_kl:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for label, axis in zip(("a", "b", "c", "d"), axes.flatten()):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    output_stem = args.output_dir / "target_risk_compaction_geometry_audit"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
