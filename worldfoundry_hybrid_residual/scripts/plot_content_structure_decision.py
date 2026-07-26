#!/usr/bin/env python3
"""Create the final six-panel decision figure for structured attention probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
YELLOW = "#E69F00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--displacement-dir", type=Path, required=True)
    parser.add_argument("--confidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-share", type=float, default=0.5388)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    block = read_csv(args.block_dir / "block_moment_marginal_summary.csv")
    block_heads = read_csv(args.block_dir / "block_moment_marginal_heads.csv")
    displacement = read_csv(
        args.displacement_dir / "local_displacement_mixture_roles.csv"
    )
    confidence = read_csv(args.confidence_dir / "confidence_summary.csv")
    confidence_decision = json.loads(
        (args.confidence_dir / "confidence_decision.json").read_text()
    )
    source: list[dict[str, object]] = []

    style()
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)

    # a: capacity Pareto frontier.
    axis = axes[0, 0]
    for router, marker in (("moment", "o"), ("oracle_mass", "x")):
        rows = sorted(
            [row for row in block if row["split"] == "test" and row["router"] == router],
            key=lambda row: float(row["attention_work_ratio_mean"]),
        )
        axis.scatter(
            [float(row["attention_work_ratio_mean"]) for row in rows],
            [100 * float(row["aggregate_output_relative_l2"]) for row in rows],
            color=BLUE,
            marker=marker,
            s=35,
            label=router.replace("_mass", ""),
        )
        for row in rows:
            source.append(
                {
                    "panel": "capacity_pareto",
                    "router": router,
                    "density": row["density"],
                    "tail_group_size": row["tail_group_size"],
                    "work_ratio": row["attention_work_ratio_mean"],
                    "aggregate_error_percent": 100
                    * float(row["aggregate_output_relative_l2"]),
                }
            )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.axvline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Arithmetic work / dense")
    axis.set_ylabel("Aggregate output error (%)")
    axis.legend(loc="upper right")
    panel_label(axis, "a")

    # b: oracle routing does not remove the approximation bottleneck.
    axis = axes[0, 1]
    lookup = {
        (row["density"], row["tail_group_size"], row["router"]): row
        for row in block
        if row["split"] == "test"
    }
    pairs = []
    for density, tail, _ in lookup:
        moment = lookup.get((density, tail, "moment"))
        oracle = lookup.get((density, tail, "oracle_mass"))
        key = (density, tail)
        if moment and oracle and key not in {(item[0], item[1]) for item in pairs}:
            pairs.append((density, tail, oracle, moment))
    for density, tail, oracle, moment in pairs:
        x = 100 * float(oracle["aggregate_output_relative_l2"])
        y = 100 * float(moment["aggregate_output_relative_l2"])
        axis.scatter(x, y, color=ORANGE, s=35)
        axis.annotate(f"d={density}, g={tail}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
        source.append(
            {
                "panel": "router_gap",
                "density": density,
                "tail_group_size": tail,
                "oracle_error_percent": x,
                "moment_error_percent": y,
            }
        )
    low = min([min(100 * float(pair[2]["aggregate_output_relative_l2"]), 100 * float(pair[3]["aggregate_output_relative_l2"])) for pair in pairs])
    high = max([max(100 * float(pair[2]["aggregate_output_relative_l2"]), 100 * float(pair[3]["aggregate_output_relative_l2"])) for pair in pairs])
    axis.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Oracle-router error (%)")
    axis.set_ylabel("Moment-router error (%)")
    panel_label(axis, "b")

    # c: displacement capacity and gate gap.
    axis = axes[0, 2]
    selected = [
        row
        for row in displacement
        if row["split"] == "test"
        and row["head_role"] == "localized"
        and row["prediction"] in ("oracle_nonnegative", "ridge_nonnegative")
    ]
    for prediction, color in (("oracle_nonnegative", GREEN), ("ridge_nonnegative", PINK)):
        rows = sorted(
            [row for row in selected if row["prediction"] == prediction],
            key=lambda row: int(row["rank"]),
        )
        ranks = [int(row["rank"]) for row in rows]
        errors = [100 * float(row["aggregate_full_output_relative_l2"]) for row in rows]
        worst = [100 * float(row["record_error_max"]) for row in rows]
        axis.plot(ranks, errors, marker="o", color=color, label=prediction.split("_")[0])
        axis.fill_between(ranks, errors, worst, color=color, alpha=0.13)
        for row, rank, error, maximum in zip(rows, ranks, errors, worst):
            source.append(
                {
                    "panel": "displacement_capacity",
                    "prediction": prediction,
                    "rank": rank,
                    "aggregate_error_percent": error,
                    "worst_error_percent": maximum,
                }
            )
    axis.axhline(2.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("Displacement expert rank")
    axis.set_ylabel("Localized output error (%)")
    axis.legend()
    panel_label(axis, "c")

    # d: conservatively calibrated confidence/fallback policy.
    axis = axes[1, 0]
    test_confidence = [row for row in confidence if row["split"] == "test"]
    colors = {"0.125": BLUE, "0.25": GREEN, "0.375": ORANGE}
    chosen = confidence_decision["chosen_without_test_selection"]
    chosen_key = (
        chosen["density"],
        chosen["tail_group_size"],
    ) if chosen else None
    for row in test_confidence:
        x = float(row["fallback_adjusted_arithmetic_speedup"])
        y = 100 * float(row["record_error_max"])
        key = (row["density"], row["tail_group_size"])
        axis.scatter(
            x,
            y,
            color=colors[row["density"]],
            marker="*" if key == chosen_key else "o",
            s=90 if key == chosen_key else 35,
        )
        source.append(
            {
                "panel": "confidence_fallback",
                "density": row["density"],
                "tail_group_size": row["tail_group_size"],
                "coverage": row["coverage"],
                "arithmetic_speedup": x,
                "worst_error_percent": y,
            }
        )
    axis.axhline(1.5, color="#333333", linestyle="--", linewidth=1, label="validation safety gate")
    axis.axvline(1.5, color="#777777", linestyle=":", linewidth=1, label="speed gate")
    axis.set_xlim(1.0, 1.55)
    axis.set_xlabel("Fallback-adjusted arithmetic speedup")
    axis.set_ylabel("Test worst-record error (%)")
    axis.legend(fontsize=7)
    panel_label(axis, "d")

    # e: head-role coverage imposes a hard speed ceiling.
    axis = axes[1, 1]
    representative = [
        row
        for row in block_heads
        if row["split"] == "test"
        and row["router"] == "moment"
        and row["method"] == "centroid"
        and row["density"] == "0.375"
        and row["tail_group_size"] == "8"
    ]
    counts = Counter(row["head_role"] for row in representative)
    roles = ["localized", "transitional", "diffuse"]
    fractions = [counts[role] / len(representative) for role in roles]
    candidate_work = 0.4687805250305245
    candidate_speed = [1 / ((1 - fraction) + fraction * candidate_work) for fraction in fractions]
    free_speed = [1 / (1 - fraction) for fraction in fractions]
    positions = list(range(len(roles)))
    axis.bar([p - 0.18 for p in positions], candidate_speed, width=0.36, color=BLUE, label="candidate work")
    axis.bar([p + 0.18 for p in positions], free_speed, width=0.36, color=YELLOW, label="branch is free")
    axis.axhline(1.5, color="#333333", linestyle="--", linewidth=1)
    axis.set_xticks(positions, roles, rotation=15)
    axis.set_ylabel("Whole-attention arithmetic speedup")
    axis.legend(fontsize=7)
    panel_label(axis, "e")
    for role, fraction, candidate, free in zip(roles, fractions, candidate_speed, free_speed):
        source.append(
            {
                "panel": "role_ceiling",
                "head_role": role,
                "fraction": fraction,
                "candidate_speedup": candidate,
                "free_branch_speedup": free,
            }
        )

    # f: measured attention share converted to denoiser Amdahl bounds.
    axis = axes[1, 2]
    local = [1.0 + 0.05 * index for index in range(61)]
    denoiser = [
        1 / ((1 - args.attention_share) + args.attention_share / speed)
        for speed in local
    ]
    axis.plot(local, denoiser, color=YELLOW, linewidth=2)
    for speed in (1.15, 1.5, 2.0):
        bound = 1 / ((1 - args.attention_share) + args.attention_share / speed)
        axis.scatter(speed, bound, color="#222222", s=25)
        axis.annotate(f"{speed:g}x -> {bound:.3f}x", (speed, bound), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Whole-attention speedup")
    axis.set_ylabel("Denoiser speedup upper bound")
    panel_label(axis, "f")
    for speed, bound in zip(local, denoiser):
        source.append(
            {
                "panel": "amdahl",
                "attention_share": args.attention_share,
                "attention_speedup": speed,
                "denoiser_speedup_upper_bound": bound,
            }
        )

    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"content_structure_decision.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    write_csv(args.output_dir / "content_structure_decision_data.csv", source)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "block_dir": str(args.block_dir.resolve()),
                "displacement_dir": str(args.displacement_dir.resolve()),
                "confidence_dir": str(args.confidence_dir.resolve()),
                "attention_share": args.attention_share,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
