#!/usr/bin/env python3
"""Analyze EXP-046 without opening the locked final identity split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "selection"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(input_dir: Path, split: str) -> list[dict[str, object]]:
    aggregate_path = input_dir / "all_cell_metrics.csv"
    paths = [aggregate_path] if aggregate_path.is_file() else sorted(
        input_dir.glob("p*_seed*/cell_metrics.csv")
    )
    if not paths:
        raise FileNotFoundError(f"no EXP-046 metrics found under {input_dir}")
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row["split"] == split)
    if not rows:
        raise ValueError(f"no rows found for split {split}")
    return rows


def expected_sample_ids(config: dict[str, object], split: str) -> set[str]:
    return {
        f"p{int(item['prompt_index']):02d}_seed{int(item['seed'])}"
        for item in config["sample_plan"]
        if item["split"] == split
    }


def validate_rows(
    rows: list[dict[str, object]], config: dict[str, object], split: str
) -> tuple[set[str], bool]:
    ranks = (0, *(int(value) for value in config["ranks"]))
    expected_samples = expected_sample_ids(config, split)
    observed_samples = {row["sample_id"] for row in rows}
    if not observed_samples.issubset(expected_samples):
        raise ValueError("metrics contain identities outside the frozen split")
    keys: set[tuple[object, ...]] = set()
    rank_groups: dict[tuple[object, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["target_visible"] not in (True, "True", "true", "1"):
            raise ValueError("EXP-046 metrics must remain target-visible capacity rows")
        numeric_fields = (
            "error_sq",
            "residual_target_sq",
            "output_target_sq",
            "output_relative_l2",
            "defect_remaining_energy",
            "render_to_exact_macs",
            "base_plus_render_to_exact_macs",
        )
        if not all(math.isfinite(float(row[name])) for name in numeric_fields):
            raise ValueError("metrics contain non-finite values")
        expected_output = (
            float(row["error_sq"]) / max(float(row["output_target_sq"]), 1e-30)
        ) ** 0.5
        if not math.isclose(
            float(row["output_relative_l2"]),
            expected_output,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("output-relative L2 uses the wrong denominator")
        key = (
            row["sample_id"],
            int(row["block"]),
            int(row["target_step"]),
            int(row["branch"]),
            int(row["horizon"]),
            int(row["rank"]),
        )
        if key in keys:
            raise ValueError(f"duplicate metric cell: {key}")
        keys.add(key)
        rank_groups[key[:-1]].append((int(row["rank"]), float(row["error_sq"])))
    expected_keys = {
        (sample, block, step, branch, horizon, rank)
        for sample in observed_samples
        for block in (int(value) for value in config["blocks"])
        for step in (int(value) for value in config["target_steps"])
        for branch in (int(value) for value in config["branches"])
        for horizon in (int(value) for value in config["horizons"])
        for rank in ranks
    }
    missing = expected_keys.difference(keys)
    extra = keys.difference(expected_keys)
    if missing or extra:
        raise ValueError(
            f"metric grid mismatch: missing={len(missing)} extra={len(extra)}"
        )
    for key, values in rank_groups.items():
        ordered = sorted(values)
        if tuple(rank for rank, _ in ordered) != ranks:
            raise ValueError(f"rank grid mismatch for {key}")
        for previous, current in zip(ordered, ordered[1:]):
            if current[1] > previous[1] * (1 + 1e-5):
                raise ValueError(f"rank-state error is not monotonic for {key}")
    selection_complete = (
        split == "selection"
        and observed_samples == expected_samples
        and len(observed_samples) == int(config["gate"]["required_selection_identities"])
    )
    return observed_samples, selection_complete


def aggregate_cells(
    rows: list[dict[str, object]], config: dict[str, object]
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["rank"]),
                int(row["block"]),
                int(row["target_step"]),
                int(row["horizon"]),
            )
        ].append(row)
    gate = config["gate"]
    output: list[dict[str, object]] = []
    for key in sorted(groups):
        rank, block, step, horizon = key
        group = groups[key]
        error_sq = sum(float(row["error_sq"]) for row in group)
        output_target_sq = sum(float(row["output_target_sq"]) for row in group)
        residual_target_sq = sum(float(row["residual_target_sq"]) for row in group)
        aggregate_output = (error_sq / max(output_target_sq, 1e-30)) ** 0.5
        worst_output = max(float(row["output_relative_l2"]) for row in group)
        aggregate_residual = (error_sq / max(residual_target_sq, 1e-30)) ** 0.5
        max_cost = max(float(row["render_to_exact_macs"]) for row in group)
        quality_pass = (
            aggregate_output
            <= float(gate["maximum_aggregate_output_relative_l2"])
            and worst_output <= float(gate["maximum_worst_output_relative_l2"])
        )
        cost_pass = max_cost <= float(gate["maximum_render_to_exact_macs"])
        output.append(
            {
                "rank": rank,
                "block": block,
                "target_step": step,
                "horizon": horizon,
                "aggregate_output_relative_l2": aggregate_output,
                "worst_output_relative_l2": worst_output,
                "aggregate_residual_relative_l2": aggregate_residual,
                "mean_defect_remaining_energy": sum(
                    float(row["defect_remaining_energy"]) for row in group
                )
                / len(group),
                "maximum_render_to_exact_macs": max_cost,
                "sample_count": len({row["sample_id"] for row in group}),
                "row_count": len(group),
                "quality_pass": quality_pass,
                "cost_pass": cost_pass,
                "cell_pass": quality_pass and cost_pass,
            }
        )
    return output


def rank_summaries(
    rows: list[dict[str, object]], cell_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rank in sorted({int(row["rank"]) for row in rows}):
        selected = [row for row in rows if int(row["rank"]) == rank]
        error_sq = sum(float(row["error_sq"]) for row in selected)
        output_target_sq = sum(float(row["output_target_sq"]) for row in selected)
        residual_target_sq = sum(float(row["residual_target_sq"]) for row in selected)
        cells = [row for row in cell_rows if int(row["rank"]) == rank]
        summaries.append(
            {
                "rank": rank,
                "aggregate_output_relative_l2": (
                    error_sq / max(output_target_sq, 1e-30)
                )
                ** 0.5,
                "worst_output_relative_l2": max(
                    float(row["output_relative_l2"]) for row in selected
                ),
                "aggregate_residual_relative_l2": (
                    error_sq / max(residual_target_sq, 1e-30)
                )
                ** 0.5,
                "mean_defect_remaining_energy": sum(
                    float(row["defect_remaining_energy"]) for row in selected
                )
                / len(selected),
                "passing_cells": sum(bool(row["cell_pass"]) for row in cells),
                "total_cells": len(cells),
                "maximum_render_to_exact_macs": max(
                    float(row["render_to_exact_macs"]) for row in selected
                ),
                "maximum_base_plus_render_to_exact_macs": max(
                    float(row["base_plus_render_to_exact_macs"]) for row in selected
                ),
            }
        )
    return summaries


def build_gate(
    cell_rows: list[dict[str, object]],
    config: dict[str, object],
    selection_complete: bool,
) -> tuple[list[dict[str, object]], str]:
    gate = config["gate"]
    decision_rank = int(gate["decision_rank"])
    diagnostic_rank = int(gate["diagnostic_rank"])
    required = int(gate["minimum_passing_layers_per_step_horizon"])
    coverage: list[dict[str, object]] = []
    for rank in (decision_rank, diagnostic_rank):
        for step in (int(value) for value in config["target_steps"]):
            for horizon in (int(value) for value in config["horizons"]):
                selected = [
                    row
                    for row in cell_rows
                    if int(row["rank"]) == rank
                    and int(row["target_step"]) == step
                    and int(row["horizon"]) == horizon
                ]
                passing = sum(bool(row["cell_pass"]) for row in selected)
                coverage.append(
                    {
                        "rank": rank,
                        "target_step": step,
                        "horizon": horizon,
                        "passing_layers": passing,
                        "total_layers": len(selected),
                        "coverage_pass": passing >= required,
                    }
                )
    decision_pass = all(
        bool(row["coverage_pass"])
        for row in coverage
        if int(row["rank"]) == decision_rank
    )
    diagnostic_pass = all(
        bool(row["coverage_pass"])
        for row in coverage
        if int(row["rank"]) == diagnostic_rank
    )
    if not selection_complete:
        decision = "INCOMPLETE"
    elif decision_pass:
        decision = "PASS"
    elif diagnostic_pass:
        decision = "BOUNDARY"
    else:
        decision = "FAIL"
    return coverage, decision


def minimum_rank_rows(
    cell_rows: list[dict[str, object]], config: dict[str, object]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for block in (int(value) for value in config["blocks"]):
        for step in (int(value) for value in config["target_steps"]):
            for horizon in (int(value) for value in config["horizons"]):
                selected = sorted(
                    (
                        row
                        for row in cell_rows
                        if int(row["block"]) == block
                        and int(row["target_step"]) == step
                        and int(row["horizon"]) == horizon
                    ),
                    key=lambda row: int(row["rank"]),
                )
                passing = [row for row in selected if bool(row["cell_pass"])]
                output.append(
                    {
                        "block": block,
                        "target_step": step,
                        "horizon": horizon,
                        "minimum_passing_rank": (
                            int(passing[0]["rank"]) if passing else "none"
                        ),
                    }
                )
    return output


def plot_rank_curve(
    summaries: list[dict[str, object]], config: dict[str, object], path: Path
) -> None:
    ranks = np.array([int(row["rank"]) for row in summaries])
    aggregate = 100 * np.array(
        [float(row["aggregate_output_relative_l2"]) for row in summaries]
    )
    worst = 100 * np.array(
        [float(row["worst_output_relative_l2"]) for row in summaries]
    )
    gate = config["gate"]
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(ranks, aggregate, marker="o", label="aggregate")
    axis.plot(ranks, worst, marker="s", label="worst identity/branch")
    axis.axhline(
        100 * float(gate["maximum_aggregate_output_relative_l2"]),
        color="#00796b",
        linestyle="--",
        label="aggregate gate",
    )
    axis.axhline(
        100 * float(gate["maximum_worst_output_relative_l2"]),
        color="#c62828",
        linestyle=":",
        label="worst gate",
    )
    axis.set_xlabel("Target-visible defect rank")
    axis.set_ylabel("Block-output relative L2 (%)")
    axis.set_title("EXP-046 rank-state capacity frontier")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_decision_heatmap(
    cell_rows: list[dict[str, object]], config: dict[str, object], path: Path
) -> None:
    rank = int(config["gate"]["decision_rank"])
    blocks = [int(value) for value in config["blocks"]]
    columns = [
        (int(step), int(horizon))
        for step in config["target_steps"]
        for horizon in config["horizons"]
    ]
    matrix = np.empty((len(blocks), len(columns)))
    labels: list[list[str]] = []
    for block in blocks:
        label_row: list[str] = []
        for column_index, (step, horizon) in enumerate(columns):
            row = next(
                item
                for item in cell_rows
                if int(item["rank"]) == rank
                and int(item["block"]) == block
                and int(item["target_step"]) == step
                and int(item["horizon"]) == horizon
            )
            aggregate = 100 * float(row["aggregate_output_relative_l2"])
            worst = 100 * float(row["worst_output_relative_l2"])
            matrix[blocks.index(block), column_index] = aggregate
            label_row.append(f"{aggregate:.2f}\n{worst:.2f}")
        labels.append(label_row)
    figure, axis = plt.subplots(figsize=(10.4, 7.4))
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(2.0, matrix.max()))
    axis.set_xticks(
        range(len(columns)), [f"step {step}\nH{horizon}" for step, horizon in columns]
    )
    axis.set_yticks(range(len(blocks)), [f"L{block}" for block in blocks])
    axis.set_title(f"Rank-{rank} output L2: aggregate / worst (%)")
    for y_index in range(len(blocks)):
        for x_index in range(len(columns)):
            axis.text(
                x_index,
                y_index,
                labels[y_index][x_index],
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, label="aggregate output L2 (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    *,
    split: str,
    decision: str,
    sample_count: int,
    summaries: list[dict[str, object]],
    coverage: list[dict[str, object]],
    config: dict[str, object],
) -> None:
    decision_rank = int(config["gate"]["decision_rank"])
    selected = next(row for row in summaries if int(row["rank"]) == decision_rank)
    lines = [
        "# EXP-046 whole-block rank-state capacity 分析",
        "",
        f"- 数据 split：`{split}`",
        f"- identity 数：`{sample_count}`",
        f"- G-025 决策：`{decision}`",
        "- 这是 target-visible 表示容量上界，不是可部署 observer、rollout 或速度结果。",
        "- 状态因子没有保存；held-out target 只用于本次 SVD 容量测量。",
        "",
        f"## Rank-{decision_rank} 总体结果",
        "",
        f"- 聚合 block-output L2：`{100 * float(selected['aggregate_output_relative_l2']):.3f}%`",
        f"- 最坏 identity/branch L2：`{100 * float(selected['worst_output_relative_l2']):.3f}%`",
        f"- 通过 cell：`{selected['passing_cells']}/{selected['total_cells']}`",
        f"- 最大 state-render / estimated exact MAC：`{100 * float(selected['maximum_render_to_exact_macs']):.3f}%`",
        "",
        "## 覆盖 Gate",
        "",
        "| rank | step | horizon | passing layers | Gate |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['rank']} | {row['target_step']} | H{row['horizon']} | "
            f"{row['passing_layers']}/{row['total_layers']} | "
            f"{'PASS' if row['coverage_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Rank frontier",
            "",
            "| rank | aggregate output L2 | worst output L2 | remaining defect | pass cells | render/exact |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['rank']} | {100 * float(row['aggregate_output_relative_l2']):.3f}% | "
            f"{100 * float(row['worst_output_relative_l2']):.3f}% | "
            f"{100 * float(row['mean_defect_remaining_energy']):.3f}% | "
            f"{row['passing_cells']}/{row['total_cells']} | "
            f"{100 * float(row['maximum_render_to_exact_macs']):.3f}% |"
        )
    if decision == "PASS":
        lines.extend(
            [
                "",
                "结果只允许开启独立的 current-h coordinate-observability Gate；尚不允许训练 recurrent student。",
            ]
        )
    elif decision == "BOUNDARY":
        lines.extend(
            [
                "",
                "Rank 96 仅形成成本/容量边界，不能挽救 rank-64 主张，需新的架构决策。",
            ]
        )
    elif decision == "FAIL":
        lines.extend(
            [
                "",
                "在当前 renderer 下停止 rank-64 whole-block state 路线；该结论不否定训练原生状态或完整 few-step student。",
            ]
        )
    else:
        lines.extend(["", "当前 split 不构成正式 G-025 结论。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_rows(args.input_dir, args.split)
    samples, selection_complete = validate_rows(rows, config, args.split)
    cells = aggregate_cells(rows, config)
    summaries = rank_summaries(rows, cells)
    coverage, decision = build_gate(cells, config, selection_complete)
    minimum_ranks = minimum_rank_rows(cells, config)
    write_csv(args.out_dir / "cell_rank_metrics.csv", cells)
    write_csv(args.out_dir / "rank_summary.csv", summaries)
    write_csv(args.out_dir / "coverage_gate.csv", coverage)
    write_csv(args.out_dir / "minimum_passing_rank.csv", minimum_ranks)
    plot_rank_curve(summaries, config, args.out_dir / "rank_capacity_frontier.png")
    plot_decision_heatmap(cells, config, args.out_dir / "rank64_layer_map.png")
    write_report(
        args.out_dir / "report.zh-CN.md",
        split=args.split,
        decision=decision,
        sample_count=len(samples),
        summaries=summaries,
        coverage=coverage,
        config=config,
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "split": args.split,
        "sample_count": len(samples),
        "selection_complete": selection_complete,
        "decision": decision,
        "final_split_opened": False,
    }
    (args.out_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
