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


MODES = ("positioned_equal_mass", "positioned_group_mass")
GROUP_COUNTS = (0, 49, 98, 147, 196)
EXPECTED_PATH_ROWS = 24 * len(MODES) * len(GROUP_COUNTS)
EXPECTED_MARGINAL_ROWS = (
    24 * len(MODES) * sum(392 - count for count in GROUP_COUNTS[:-1])
)


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
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def path_regressions(rows: list[dict[str, str]]) -> tuple[int, int]:
    ordered = sorted(rows, key=lambda row: int(row["selected_group_count"]))
    match_to_mismatch = 0
    kl_increase = 0
    for previous, current in zip(ordered, ordered[1:]):
        if int(previous["prediction_match"]) and not int(current["prediction_match"]):
            match_to_mismatch += 1
        if float(current["candidate_kl"]) > float(previous["candidate_kl"]) + 1e-6:
            kl_increase += 1
    return match_to_mismatch, kl_increase


def validate_and_build_frontier(
    summary: dict[str, object],
    path_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    if len(path_rows) != EXPECTED_PATH_ROWS:
        raise ValueError(
            f"expected {EXPECTED_PATH_ROWS} path rows, found {len(path_rows)}"
        )
    sample_ids = {row["sample_id"] for row in path_rows}
    if len(sample_ids) != 24:
        raise ValueError(f"expected 24 samples, found {len(sample_ids)}")

    frontier_rows: list[dict[str, object]] = []
    registered = summary["summaries"]
    for mode in MODES:
        for group_count in GROUP_COUNTS:
            rows = [
                row
                for row in path_rows
                if row["mode"] == mode
                and int(row["selected_group_count"]) == group_count
            ]
            if len(rows) != 24:
                raise ValueError(
                    f"expected 24 rows for {mode}/{group_count}, found {len(rows)}"
                )
            kl_values = np.asarray(
                [float(row["candidate_kl"]) for row in rows], dtype=np.float64
            )
            recomputed = {
                "agreement": sum(int(row["prediction_match"]) for row in rows) / 24,
                "mismatch_count": sum(not int(row["prediction_match"]) for row in rows),
                "harmful_count": sum(
                    int(row["baseline_correct"]) and not int(row["approximate_correct"])
                    for row in rows
                ),
                "candidate_kl_mean": float(kl_values.mean()),
                "candidate_kl_p95": float(np.quantile(kl_values, 0.95)),
                "token_retention": float(rows[0]["token_retention"]),
            }
            recorded = registered[mode][str(group_count)]
            for key, value in recomputed.items():
                if not np.isclose(float(recorded[key]), float(value), atol=1e-12):
                    raise ValueError(
                        f"summary mismatch for {mode}/{group_count}/{key}: "
                        f"{recorded[key]} != {value}"
                    )
            frontier_rows.append(
                {
                    "method": mode,
                    "selected_group_count": group_count,
                    **recomputed,
                }
            )
    return frontier_rows


def build_regression_rows(
    path_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in path_rows:
        grouped[(row["mode"], row["sample_id"])].append(row)

    result: list[dict[str, object]] = []
    for mode in MODES:
        mode_pairs = [
            path_regressions(rows)
            for (row_mode, _sample_id), rows in grouped.items()
            if row_mode == mode
        ]
        result.append(
            {
                "mode": mode,
                "sample_count": len(mode_pairs),
                "match_to_mismatch_count": sum(pair[0] for pair in mode_pairs),
                "kl_increase_count": sum(pair[1] for pair in mode_pairs),
                "samples_with_match_regression": sum(
                    pair[0] > 0 for pair in mode_pairs
                ),
                "samples_with_kl_regression": sum(pair[1] > 0 for pair in mode_pairs),
            }
        )
    return result


def build_marginal_summary(
    marginal_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    if len(marginal_rows) != EXPECTED_MARGINAL_ROWS:
        raise ValueError(
            f"expected {EXPECTED_MARGINAL_ROWS} marginal rows, "
            f"found {len(marginal_rows)}"
        )
    result: list[dict[str, object]] = []
    for mode in MODES:
        for group_count in GROUP_COUNTS[:-1]:
            rows = [
                row
                for row in marginal_rows
                if row["mode"] == mode
                and int(row["current_group_count"]) == group_count
            ]
            selected = [row for row in rows if int(row["selected_next_batch"])]
            expected = 24 * (392 - group_count)
            if len(rows) != expected or len(selected) != 24 * 49:
                raise ValueError(
                    f"unexpected marginal shape for {mode}/{group_count}: "
                    f"all={len(rows)}, selected={len(selected)}"
                )
            all_benefits = np.asarray(
                [float(row["conditional_kl_benefit"]) for row in rows]
            )
            selected_benefits = np.asarray(
                [float(row["conditional_kl_benefit"]) for row in selected]
            )
            result.append(
                {
                    "mode": mode,
                    "current_group_count": group_count,
                    "remaining_candidate_count": len(rows),
                    "selected_candidate_count": len(selected),
                    "all_positive_fraction": float(np.mean(all_benefits > 0)),
                    "selected_positive_fraction": float(np.mean(selected_benefits > 0)),
                    "selected_benefit_mean": float(selected_benefits.mean()),
                    "selected_benefit_p10": float(np.quantile(selected_benefits, 0.10)),
                    "selected_benefit_median": float(
                        np.quantile(selected_benefits, 0.50)
                    ),
                    "selected_benefit_p90": float(np.quantile(selected_benefits, 0.90)),
                }
            )
    return result


def build_batch_interactions(
    path_rows: list[dict[str, str]],
    marginal_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    path_lookup = {
        (row["mode"], row["sample_id"], int(row["selected_group_count"])): row
        for row in path_rows
    }
    selected_benefits: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in marginal_rows:
        if int(row["selected_next_batch"]):
            selected_benefits[
                (
                    row["mode"],
                    row["sample_id"],
                    int(row["current_group_count"]),
                )
            ].append(float(row["conditional_kl_benefit"]))

    result: list[dict[str, object]] = []
    sample_ids = sorted({row["sample_id"] for row in path_rows})
    for mode in MODES:
        for sample_id in sample_ids:
            for current_count, next_count in zip(GROUP_COUNTS, GROUP_COUNTS[1:]):
                benefits = selected_benefits[(mode, sample_id, current_count)]
                if len(benefits) != 49:
                    raise ValueError(
                        f"expected 49 selected benefits for "
                        f"{mode}/{sample_id}/{current_count}"
                    )
                current_kl = float(
                    path_lookup[(mode, sample_id, current_count)]["candidate_kl"]
                )
                actual_next_kl = float(
                    path_lookup[(mode, sample_id, next_count)]["candidate_kl"]
                )
                predicted_next_kl = current_kl - sum(benefits)
                result.append(
                    {
                        "mode": mode,
                        "sample_id": sample_id,
                        "current_group_count": current_count,
                        "next_group_count": next_count,
                        "current_candidate_kl": current_kl,
                        "selected_benefit_sum": sum(benefits),
                        "additive_predicted_next_kl": predicted_next_kl,
                        "actual_next_kl": actual_next_kl,
                        "batch_interaction_residual": actual_next_kl
                        - predicted_next_kl,
                    }
                )
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_dir = args.analysis_dir / "batched_current_support_marginal_exposed_v1"
    static_dir = args.analysis_dir / "reader_aligned_singleton_marginal_exposed_v1"
    summary = json.loads((current_dir / "summary.json").read_text(encoding="utf-8"))
    path_rows = read_csv(current_dir / "current_support_path_rows.csv")
    marginal_rows = read_csv(current_dir / "current_support_marginals.csv")
    static_summary = json.loads(
        (static_dir / "summary.json").read_text(encoding="utf-8")
    )

    frontier_rows = validate_and_build_frontier(summary, path_rows)
    static_budgets = static_summary["budget_summaries"]
    for group_count in GROUP_COUNTS:
        if str(group_count) not in static_budgets:
            continue
        metrics = static_budgets[str(group_count)]
        frontier_rows.append(
            {
                "method": "static_empty_support_singleton",
                "selected_group_count": group_count,
                "agreement": float(metrics["agreement"]),
                "mismatch_count": int(metrics["mismatch_count"]),
                "harmful_count": int(metrics["harmful_count"]),
                "candidate_kl_mean": float(metrics["candidate_kl_mean"]),
                "candidate_kl_p95": float(metrics["candidate_kl_p95"]),
                "token_retention": float(metrics["token_retention"]),
            }
        )
    regression_rows = build_regression_rows(path_rows)
    marginal_summary = build_marginal_summary(marginal_rows)
    batch_interactions = build_batch_interactions(path_rows, marginal_rows)

    sample_order = {row["sample_id"]: int(row["sample_position"]) for row in path_rows}
    path_lookup = {
        (row["mode"], row["sample_id"], int(row["selected_group_count"])): row
        for row in path_rows
    }
    ordered_samples = sorted(sample_order, key=sample_order.__getitem__)
    matrix = np.asarray(
        [
            [
                int(path_lookup[(mode, sample_id, count)]["prediction_match"])
                for mode in MODES
                for count in GROUP_COUNTS
            ]
            for sample_id in ordered_samples
        ]
    )
    heatmap_rows = [
        {
            "sample_id": sample_id,
            "sample_order": sample_index + 1,
            "mode": mode,
            "selected_group_count": count,
            "prediction_match": int(
                path_lookup[(mode, sample_id, count)]["prediction_match"]
            ),
        }
        for sample_index, sample_id in enumerate(ordered_samples)
        for mode in MODES
        for count in GROUP_COUNTS
    ]

    write_csv(args.output_dir / "current_support_frontier.csv", frontier_rows)
    write_csv(args.output_dir / "current_support_path_regressions.csv", regression_rows)
    write_csv(
        args.output_dir / "current_support_marginal_summary.csv", marginal_summary
    )
    write_csv(
        args.output_dir / "current_support_batch_interactions.csv",
        batch_interactions,
    )
    write_csv(args.output_dir / "current_support_match_matrix.csv", heatmap_rows)

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), constrained_layout=True)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()
    labels = {
        "static_empty_support_singleton": "Static singleton (historical SDPA)",
        "positioned_equal_mass": "Current-support, equal mass",
        "positioned_group_mass": "Current-support, group mass",
    }
    colors = {
        "static_empty_support_singleton": "#CC79A7",
        "positioned_equal_mass": "#0072B2",
        "positioned_group_mass": "#E69F00",
    }
    markers = {
        "static_empty_support_singleton": "D",
        "positioned_equal_mass": "o",
        "positioned_group_mass": "s",
    }
    grouped_frontier: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        grouped_frontier[str(row["method"])].append(row)
    for method, rows in grouped_frontier.items():
        rows.sort(key=lambda row: int(row["selected_group_count"]))
        x = [100 * float(row["token_retention"]) for row in rows]
        panel_a.plot(
            x,
            [100 * float(row["agreement"]) for row in rows],
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=4.5,
            label=labels[method],
        )
        panel_b.plot(
            x,
            [float(row["candidate_kl_mean"]) for row in rows],
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=4.5,
            label=f"{labels[method]} mean",
        )
        panel_b.plot(
            x,
            [float(row["candidate_kl_p95"]) for row in rows],
            color=colors[method],
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

    panel_c.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#D55E00", "#009E73"]),
        vmin=0,
        vmax=1,
    )
    panel_c.axvline(len(GROUP_COUNTS) - 0.5, color="white", linewidth=1.3)
    panel_c.set_xlabel("Exact four-token groups")
    panel_c.set_ylabel("Exposed question order")
    panel_c.set_xticks(
        range(len(MODES) * len(GROUP_COUNTS)),
        labels=[str(count) for _mode in MODES for count in GROUP_COUNTS],
        rotation=45,
    )
    panel_c.set_yticks([0, 5, 11, 17, 23], labels=[1, 6, 12, 18, 24])
    panel_c.text(2, -1.8, "Equal mass", ha="center", fontsize=8.5)
    panel_c.text(7, -1.8, "Group mass", ha="center", fontsize=8.5)
    panel_c.spines["top"].set_visible(True)
    panel_c.spines["right"].set_visible(True)

    mode_offsets = {"positioned_equal_mass": -0.12, "positioned_group_mass": 0.12}
    for mode in MODES:
        rows = [row for row in batch_interactions if row["mode"] == mode]
        grouped_interactions = [
            np.asarray(
                [
                    float(row["batch_interaction_residual"])
                    for row in rows
                    if int(row["current_group_count"]) == group_count
                ]
            )
            for group_count in GROUP_COUNTS[:-1]
        ]
        x = np.arange(len(grouped_interactions), dtype=np.float64) + mode_offsets[mode]
        median = np.asarray(
            [float(np.quantile(values, 0.50)) for values in grouped_interactions]
        )
        lower = median - np.asarray(
            [float(np.quantile(values, 0.10)) for values in grouped_interactions]
        )
        upper = (
            np.asarray(
                [float(np.quantile(values, 0.90)) for values in grouped_interactions]
            )
            - median
        )
        panel_d.errorbar(
            x,
            median,
            yerr=np.vstack((lower, upper)),
            color=colors[mode],
            marker=markers[mode],
            linewidth=1.5,
            capsize=2.5,
            label=labels[mode],
        )
    panel_d.axhline(0, color="#222222", linewidth=0.8, linestyle=":")
    panel_d.set_xticks(range(4), labels=GROUP_COUNTS[:-1])
    panel_d.set_xlabel("Current exact groups before selecting next 49")
    panel_d.set_ylabel("49-group interaction residual (median, P10-P90)")
    panel_d.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    panel_d.legend(loc="best", frameon=False)

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

    output_stem = args.output_dir / "batched_current_support_marginal_audit"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "path_rows": len(path_rows),
                "marginal_rows": len(marginal_rows),
                "strict_budgets": summary["strict_budgets"],
                "regressions": regression_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
