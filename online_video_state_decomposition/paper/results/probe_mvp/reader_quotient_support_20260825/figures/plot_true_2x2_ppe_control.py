from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINE = "representative_position"
CANDIDATE = "ppe_center_ranked_k4"
METHODS = (BASELINE, CANDIDATE)
EXPECTED_ROWS = 24 * len(METHODS)


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
        raise ValueError("cannot write empty PPE plot data")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_sided_sign_pvalue(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    return sum(math.comb(non_ties, count) for count in range(wins, non_ties + 1)) / (
        2**non_ties
    )


def bootstrap_paired(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int = 20_000,
) -> dict[str, float]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(draws, baseline.size))
    sampled_baseline = baseline[indices].mean(axis=1)
    sampled_candidate = candidate[indices].mean(axis=1)
    sampled_delta = sampled_candidate - sampled_baseline
    sampled_ratio = sampled_candidate / sampled_baseline
    return {
        "mean_kl_delta": float((candidate - baseline).mean()),
        "mean_kl_delta_ci_low": float(np.quantile(sampled_delta, 0.025)),
        "mean_kl_delta_ci_high": float(np.quantile(sampled_delta, 0.975)),
        "mean_kl_ratio": float(candidate.mean() / baseline.mean()),
        "mean_kl_ratio_ci_low": float(np.quantile(sampled_ratio, 0.025)),
        "mean_kl_ratio_ci_high": float(np.quantile(sampled_ratio, 0.975)),
    }


def validate_and_summarize(
    recorded: dict[str, object], rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if float(recorded["maximum_baseline_kl_repeat_error"]) != 0.0:
        raise ValueError("PPE baseline did not exactly reproduce the geometry result")
    if int(recorded["baseline_prediction_repeat_mismatches"]) != 0:
        raise ValueError("PPE baseline prediction identity changed")
    if float(recorded["maximum_standard_rotary_logit_error"]) != 0.0:
        raise ValueError("frequency-wise scalar RoPE did not exactly reproduce Qwen2")

    sample_ids = {row["sample_id"] for row in rows}
    if len(sample_ids) != 24:
        raise ValueError(f"expected 24 samples, found {len(sample_ids)}")
    lookups: dict[str, dict[str, dict[str, str]]] = {}
    aggregate_rows: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 24:
            raise ValueError(f"expected 24 rows for {method}")
        lookup = {row["sample_id"]: row for row in selected}
        if set(lookup) != sample_ids:
            raise ValueError(f"sample identity mismatch for {method}")
        lookups[method] = lookup
        kl = np.asarray([float(row["candidate_kl"]) for row in selected])
        recomputed = {
            "agreement": sum(int(row["prediction_match"]) for row in selected) / 24,
            "mismatch_count": sum(not int(row["prediction_match"]) for row in selected),
            "harmful_count": sum(int(row["harmful"]) for row in selected),
            "candidate_kl_mean": float(kl.mean()),
            "candidate_kl_p95": float(np.quantile(kl, 0.95)),
            "token_retention": float(selected[0]["token_retention"]),
        }
        for key, value in recomputed.items():
            if not np.isclose(
                float(recorded["summaries"][method][key]), float(value), atol=1e-12
            ):
                raise ValueError(f"summary mismatch for {method}/{key}")
        aggregate_rows.append({"method": method, **recomputed})

    ordered_ids = sorted(
        sample_ids,
        key=lambda sample_id: int(lookups[BASELINE][sample_id]["sample_position"]),
    )
    paired_rows: list[dict[str, object]] = []
    baseline_kl = np.asarray(
        [
            float(lookups[BASELINE][sample_id]["candidate_kl"])
            for sample_id in ordered_ids
        ]
    )
    candidate_kl = np.asarray(
        [
            float(lookups[CANDIDATE][sample_id]["candidate_kl"])
            for sample_id in ordered_ids
        ]
    )
    for sample_id, baseline_value, candidate_value in zip(
        ordered_ids, baseline_kl, candidate_kl
    ):
        baseline_row = lookups[BASELINE][sample_id]
        candidate_row = lookups[CANDIDATE][sample_id]
        paired_rows.append(
            {
                "sample_id": sample_id,
                "sample_position": int(baseline_row["sample_position"]),
                "baseline_candidate_kl": baseline_value,
                "ppe_candidate_kl": candidate_value,
                "ppe_minus_baseline_kl": candidate_value - baseline_value,
                "ppe_kl_win": int(candidate_value < baseline_value),
                "baseline_prediction_match": int(baseline_row["prediction_match"]),
                "ppe_prediction_match": int(candidate_row["prediction_match"]),
                "baseline_harmful": int(baseline_row["harmful"]),
                "ppe_harmful": int(candidate_row["harmful"]),
            }
        )

    statistics: dict[str, object] = bootstrap_paired(
        baseline_kl, candidate_kl, seed=20260830
    )
    wins = int(np.sum(candidate_kl < baseline_kl))
    losses = int(np.sum(candidate_kl > baseline_kl))
    ties = int(candidate_kl.size - wins - losses)
    statistics.update(
        {
            "paired_kl_wins": wins,
            "paired_kl_losses": losses,
            "paired_kl_ties": ties,
            "one_sided_sign_pvalue": one_sided_sign_pvalue(wins, wins + losses),
            **recorded["comparison"],
        }
    )
    if not np.isclose(
        float(statistics["mean_kl_ratio"]),
        float(recorded["comparison"]["mean_kl_ratio"]),
        atol=1e-12,
    ):
        raise ValueError("recorded PPE mean ratio changed")
    aggregate_rows.append({"method": "paired_ppe_vs_baseline", **statistics})
    return aggregate_rows, paired_rows, statistics


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.analysis_dir / "true_2x2_ppe_exposed_v1"
    recorded = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(result_dir / "true_2x2_ppe_rows.csv")
    aggregate_rows, paired_rows, statistics = validate_and_summarize(recorded, rows)
    write_csv(args.output_dir / "true_2x2_ppe_summary.csv", aggregate_rows)
    write_csv(args.output_dir / "true_2x2_ppe_paired.csv", paired_rows)

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()
    labels = {BASELINE: "Representative position", CANDIDATE: "PPE K=4"}
    colors = {BASELINE: "#9CA3AF", CANDIDATE: "#CC79A7"}

    method_rows = [
        next(row for row in aggregate_rows if row["method"] == method)
        for method in METHODS
    ]
    x = np.arange(len(METHODS), dtype=np.float64)
    bars = panel_a.bar(
        x,
        [float(row["candidate_kl_mean"]) for row in method_rows],
        color=[colors[method] for method in METHODS],
        width=0.58,
    )
    panel_a.scatter(
        x,
        [float(row["candidate_kl_p95"]) for row in method_rows],
        marker="D",
        s=28,
        color="#111827",
        zorder=3,
        label="P95",
    )
    for bar in bars:
        panel_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    panel_a.set_xticks(x, labels=[labels[method] for method in METHODS])
    panel_a.set_ylabel("Candidate KL (bar mean, diamond P95)")
    panel_a.set_yscale("log")
    panel_a.legend(frameon=False, loc="upper left")
    panel_a.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    baseline_values = np.asarray(
        [float(row["baseline_candidate_kl"]) for row in paired_rows]
    )
    ppe_values = np.asarray([float(row["ppe_candidate_kl"]) for row in paired_rows])
    transition_colors = [
        "#009E73"
        if int(row["ppe_prediction_match"]) > int(row["baseline_prediction_match"])
        else "#D55E00"
        if int(row["ppe_prediction_match"]) < int(row["baseline_prediction_match"])
        else "#0072B2"
        for row in paired_rows
    ]
    panel_b.scatter(
        baseline_values,
        ppe_values,
        color=transition_colors,
        s=30,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    limits = (1e-5, 0.55)
    panel_b.plot(limits, limits, color="#222222", linestyle=":", linewidth=1)
    panel_b.set_xscale("log")
    panel_b.set_yscale("log")
    panel_b.set_xlim(limits)
    panel_b.set_ylim(limits)
    panel_b.set_xlabel("Representative-position KL")
    panel_b.set_ylabel("PPE KL")
    panel_b.grid(color="#D1D5DB", linewidth=0.5, alpha=0.55)
    panel_b.text(
        0.03,
        0.94,
        "green: match gain   orange: match loss",
        transform=panel_b.transAxes,
        va="top",
        fontsize=8,
    )

    sorted_pairs = sorted(
        paired_rows, key=lambda row: float(row["ppe_minus_baseline_kl"])
    )
    deltas = np.asarray([float(row["ppe_minus_baseline_kl"]) for row in sorted_pairs])
    panel_c.bar(
        np.arange(1, deltas.size + 1),
        deltas,
        color=["#009E73" if value < 0 else "#D55E00" for value in deltas],
        width=0.8,
    )
    panel_c.axhline(0, color="#222222", linestyle=":", linewidth=1)
    panel_c.set_xlabel("Exposed samples sorted by paired KL change")
    panel_c.set_ylabel("PPE KL - representative-position KL")
    panel_c.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    ratio = float(statistics["mean_kl_ratio"])
    lower = ratio - float(statistics["mean_kl_ratio_ci_low"])
    upper = float(statistics["mean_kl_ratio_ci_high"]) - ratio
    panel_d.errorbar(
        [0],
        [ratio],
        yerr=np.asarray([[lower], [upper]]),
        fmt="o",
        color="#111827",
        markerfacecolor="#CC79A7",
        markersize=8,
        linewidth=1.5,
        capsize=4,
    )
    panel_d.axhline(
        0.8, color="#009E73", linestyle="--", linewidth=1, label="Strict ratio gate"
    )
    panel_d.axhline(
        1.0, color="#222222", linestyle=":", linewidth=1, label="No KL increase"
    )
    panel_d.set_xlim(-0.7, 0.7)
    panel_d.set_ylim(0.45, 1.8)
    panel_d.set_xticks([0], labels=["PPE / representative"])
    panel_d.set_ylabel("Mean KL ratio (95% paired bootstrap CI)")
    panel_d.legend(frameon=False, loc="upper left")
    panel_d.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    panel_d.text(
        0.03,
        0.05,
        "mismatch 5 to 3\nharmful 1 to 2\npaired KL wins 10/24",
        transform=panel_d.transAxes,
        va="bottom",
        fontsize=8.5,
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

    output_stem = args.output_dir / "true_2x2_ppe_control"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    svg_path = output_stem.with_suffix(".svg")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": recorded["decision"],
                "paired_statistics": statistics,
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
