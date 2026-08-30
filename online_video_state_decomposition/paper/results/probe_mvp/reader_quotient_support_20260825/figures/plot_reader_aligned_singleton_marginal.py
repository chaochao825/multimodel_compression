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
        raise ValueError("cannot write empty plot data")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    singleton_dir = args.analysis_dir / "reader_aligned_singleton_marginal_exposed_v1"
    singleton_summary = json.loads(
        (singleton_dir / "summary.json").read_text(encoding="utf-8")
    )
    geometry_summary = json.loads(
        (
            args.analysis_dir
            / "group_compaction_geometry_exposed_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    path_rows = read_csv(singleton_dir / "reader_aligned_path_rows.csv")
    marginal_rows = read_csv(singleton_dir / "singleton_group_marginals.csv")

    frontier_rows: list[dict[str, object]] = []
    methods = {
        "Dense-gradient, original positions": geometry_summary["layout_summaries"][
            "compact_original_position"
        ],
        "Reader singleton, original positions": singleton_summary["budget_summaries"],
    }
    for method, summaries in methods.items():
        for group_count_text, metrics in summaries.items():
            retention_key = (
                "reader_token_retention"
                if "reader_token_retention" in metrics
                else "token_retention"
            )
            frontier_rows.append(
                {
                    "method": method,
                    "refined_group_count": int(group_count_text),
                    "token_retention": float(metrics[retention_key]),
                    "agreement": float(metrics["agreement"]),
                    "candidate_kl_mean": float(metrics["candidate_kl_mean"]),
                    "candidate_kl_p95": float(metrics["candidate_kl_p95"]),
                    "harmful_count": int(metrics["harmful_count"]),
                }
            )
    write_csv(args.output_dir / "reader_aligned_frontier_comparison.csv", frontier_rows)

    by_sample_marginals: dict[str, list[float]] = defaultdict(list)
    base_kl_by_sample: dict[str, float] = {}
    for row in marginal_rows:
        sample_id = row["sample_id"]
        by_sample_marginals[sample_id].append(float(row["singleton_kl_benefit"]))
        base_kl_by_sample[sample_id] = float(row["base_candidate_kl"])
    path_lookup = {
        (row["sample_id"], int(row["refined_group_count"])): row
        for row in path_rows
    }
    interaction_rows: list[dict[str, object]] = []
    for sample_id, benefits in by_sample_marginals.items():
        ordered = sorted(benefits, reverse=True)
        for group_count in (49, 98, 196):
            predicted_kl = base_kl_by_sample[sample_id] - sum(
                ordered[:group_count]
            )
            actual_kl = float(path_lookup[(sample_id, group_count)]["candidate_kl"])
            interaction_rows.append(
                {
                    "sample_id": sample_id,
                    "refined_group_count": group_count,
                    "base_candidate_kl": base_kl_by_sample[sample_id],
                    "additive_predicted_kl": predicted_kl,
                    "actual_joint_kl": actual_kl,
                    "interaction_residual": actual_kl - predicted_kl,
                }
            )
    write_csv(args.output_dir / "reader_aligned_interaction_gap.csv", interaction_rows)

    positive_rows = [
        {
            "sample_id": sample_id,
            "base_candidate_kl": base_kl_by_sample[sample_id],
            "positive_singleton_count": sum(benefit > 0 for benefit in benefits),
            "positive_singleton_fraction": sum(benefit > 0 for benefit in benefits)
            / len(benefits),
        }
        for sample_id, benefits in by_sample_marginals.items()
    ]
    write_csv(args.output_dir / "reader_aligned_positive_singletons.csv", positive_rows)

    sample_ids = sorted(
        {row["sample_id"] for row in path_rows},
        key=lambda sample_id: int(
            next(row["sample_position"] for row in path_rows if row["sample_id"] == sample_id)
        ),
    )
    group_counts = sorted({int(row["refined_group_count"]) for row in path_rows})
    matrix = np.asarray(
        [
            [
                int(path_lookup[(sample_id, group_count)]["prediction_match"])
                for group_count in group_counts
            ]
            for sample_id in sample_ids
        ]
    )
    heatmap_rows = [
        {
            "sample_id": sample_id,
            "sample_order": sample_index + 1,
            "refined_group_count": group_count,
            "prediction_match": int(matrix[sample_index, group_index]),
        }
        for sample_index, sample_id in enumerate(sample_ids)
        for group_index, group_count in enumerate(group_counts)
    ]
    write_csv(args.output_dir / "reader_aligned_match_matrix.csv", heatmap_rows)

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()
    colors = {
        "Dense-gradient, original positions": "#0072B2",
        "Reader singleton, original positions": "#CC79A7",
    }
    markers = {
        "Dense-gradient, original positions": "D",
        "Reader singleton, original positions": "o",
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        grouped[str(row["method"])].append(row)
    for method, rows in grouped.items():
        rows.sort(key=lambda row: int(row["refined_group_count"]))
        x = [100 * float(row["token_retention"]) for row in rows]
        panel_a.plot(
            x,
            [100 * float(row["agreement"]) for row in rows],
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=4.5,
            label=method,
        )
        compressed = [row for row in rows if int(row["refined_group_count"]) < 392]
        compressed_x = [100 * float(row["token_retention"]) for row in compressed]
        panel_b.plot(
            compressed_x,
            [float(row["candidate_kl_mean"]) for row in compressed],
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=4.5,
            label=f"{method} mean",
        )
        panel_b.plot(
            compressed_x,
            [float(row["candidate_kl_p95"]) for row in compressed],
            color=colors[method],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
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

    panel_c.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#D55E00", "#009E73"]),
        vmin=0,
        vmax=1,
    )
    panel_c.set_xlabel("Exact four-token groups")
    panel_c.set_ylabel("Exposed question order")
    panel_c.set_xticks(range(len(group_counts)), labels=group_counts, rotation=45)
    panel_c.set_yticks([0, 5, 11, 17, 23], labels=[1, 6, 12, 18, 24])
    panel_c.spines["top"].set_visible(True)
    panel_c.spines["right"].set_visible(True)

    interaction_by_budget = {
        group_count: [
            float(row["interaction_residual"])
            for row in interaction_rows
            if int(row["refined_group_count"]) == group_count
        ]
        for group_count in (49, 98, 196)
    }
    panel_d.boxplot(
        [interaction_by_budget[group_count] for group_count in (49, 98, 196)],
        tick_labels=[49, 98, 196],
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#E69F00", "alpha": 0.65},
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.55},
    )
    panel_d.axhline(0, color="#222222", linewidth=0.8, linestyle=":")
    panel_d.set_xlabel("Exact groups selected by empty-support singleton score")
    panel_d.set_ylabel("Interaction residual in candidate KL")
    panel_d.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    for index, group_count in enumerate((49, 98, 196), start=1):
        panel_d.text(
            index,
            max(interaction_by_budget[group_count]) + 0.08,
            f"mean {np.mean(interaction_by_budget[group_count]):.2f}",
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

    output_stem = args.output_dir / "reader_aligned_singleton_marginal_audit"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
