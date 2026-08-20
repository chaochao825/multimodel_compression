#!/usr/bin/env python3
"""Aggregate EXP-045 metrics, apply G-024, and render diagnostic plots."""

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

from current_input_observability_core import oracle_recovery_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "selection"), required=True)
    return parser.parse_args()


def read_rows(input_dir: Path, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("p*_seed*/cell_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw["split"] != split:
                    continue
                rows.append(
                    {
                        **raw,
                        "block": int(raw["block"]),
                        "target_step": int(raw["target_step"]),
                        "branch": int(raw["branch"]),
                        "horizon": int(raw["horizon"]),
                        "relative_l2": float(raw["relative_l2"]),
                        "error_sq": float(raw["error_sq"]),
                        "target_sq": float(raw["target_sq"]),
                        "output_relative_l2": float(raw["output_relative_l2"]),
                        "output_target_sq": float(raw["output_target_sq"]),
                        "total_runtime_macs": int(raw["total_runtime_macs"]),
                        "observable_macs": int(raw["observable_macs"]),
                    }
                )
    if not rows:
        raise ValueError(f"no {split} metrics found under {input_dir}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_output_relative_l2(rows: list[dict[str, object]]) -> float:
    error_sq = sum(
        float(row["output_relative_l2"]) ** 2 * float(row["output_target_sq"])
        for row in rows
    )
    target_sq = sum(float(row["output_target_sq"]) for row in rows)
    return (error_sq / max(target_sq, 1e-30)) ** 0.5


def aggregate(
    rows: list[dict[str, object]], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key not in groups:
            groups[key] = {
                **{name: row[name] for name in keys},
                "error_sq": 0.0,
                "target_sq": 0.0,
                "sample_ids": set(),
                "row_count": 0,
                "runtime_macs_sum": 0,
                "observable_macs_sum": 0,
            }
        group = groups[key]
        group["error_sq"] = float(group["error_sq"]) + float(row["error_sq"])
        group["target_sq"] = float(group["target_sq"]) + float(row["target_sq"])
        group["sample_ids"].add(row["sample_id"])
        group["row_count"] = int(group["row_count"]) + 1
        group["runtime_macs_sum"] = int(group["runtime_macs_sum"]) + int(
            row["total_runtime_macs"]
        )
        group["observable_macs_sum"] = int(group["observable_macs_sum"]) + int(
            row["observable_macs"]
        )
    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        group = groups[key]
        target_sq = float(group["target_sq"])
        output.append(
            {
                **{name: group[name] for name in keys},
                "relative_l2": (float(group["error_sq"]) / max(target_sq, 1e-30))
                ** 0.5,
                "error_sq": group["error_sq"],
                "target_sq": target_sq,
                "sample_count": len(group["sample_ids"]),
                "row_count": group["row_count"],
                "mean_runtime_macs": int(group["runtime_macs_sum"])
                // int(group["row_count"]),
                "mean_observable_macs": int(group["observable_macs_sum"])
                // int(group["row_count"]),
            }
        )
    return output


def indexed_risk(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
) -> dict[tuple[object, ...], float]:
    return {
        tuple(row[name] for name in keys): float(row["relative_l2"])
        for row in aggregate(rows, keys)
    }


def build_gate_tables(
    rows: list[dict[str, object]], config: dict[str, object], split: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], str, str]:
    target_steps = tuple(int(value) for value in config["target_steps"])
    blocks = tuple(int(value) for value in config["blocks"])
    branches = tuple(int(value) for value in config["branches"])
    gate = config["gate"]
    methods = tuple(str(value) for value in config["gate_candidate_methods"])
    layer_risks = indexed_risk(
        rows, ("method", "block", "target_step", "branch", "horizon")
    )
    aggregate_risks = indexed_risk(rows, ("method", "target_step", "horizon"))
    branch_risks = indexed_risk(
        rows, ("method", "target_step", "branch", "horizon")
    )
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    complete_selection = split == "selection" and len(sample_ids) == 4

    layer_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    for method in methods:
        counts: dict[int, int] = {}
        recoveries: dict[int, float] = {}
        branch_harm_max = 0.0
        open_loop_ratio_max = 0.0
        missing = False
        for target_step in target_steps:
            passing_layers = 0
            ar2_aggregate = aggregate_risks[("ar2", target_step, 1)]
            method_key = (method, target_step, 1)
            oracle_key = ("oracle_transport75_token_ls", target_step, 1)
            if method_key not in aggregate_risks or oracle_key not in aggregate_risks:
                missing = True
                recoveries[target_step] = float("-inf")
            else:
                recoveries[target_step] = oracle_recovery_fraction(
                    ar2_aggregate,
                    aggregate_risks[method_key],
                    aggregate_risks[oracle_key],
                )
            for block in blocks:
                method_error_sq = 0.0
                method_target_sq = 0.0
                ar2_error_sq = 0.0
                ar2_target_sq = 0.0
                for row in rows:
                    if (
                        int(row["block"]) == block
                        and int(row["target_step"]) == target_step
                        and int(row["horizon"]) == 1
                    ):
                        if row["method"] == method:
                            method_error_sq += float(row["error_sq"])
                            method_target_sq += float(row["target_sq"])
                        elif row["method"] == "ar2":
                            ar2_error_sq += float(row["error_sq"])
                            ar2_target_sq += float(row["target_sq"])
                if method_target_sq == 0 or ar2_target_sq == 0:
                    missing = True
                    reduction = 0.0
                    method_risk = float("inf")
                    ar2_risk = float("inf")
                else:
                    method_risk = (method_error_sq / method_target_sq) ** 0.5
                    ar2_risk = (ar2_error_sq / ar2_target_sq) ** 0.5
                    reduction = ar2_risk / max(method_risk, 1e-30)
                passed = reduction >= float(gate["minimum_risk_reduction"])
                passing_layers += int(passed)
                layer_rows.append(
                    {
                        "method": method,
                        "block": block,
                        "target_step": target_step,
                        "ar2_risk": ar2_risk,
                        "method_risk": method_risk,
                        "risk_reduction": reduction,
                        "risk_gate_pass": passed,
                    }
                )
            counts[target_step] = passing_layers
            for branch in branches:
                method_branch_key = (method, target_step, branch, 1)
                ar2_branch_key = ("ar2", target_step, branch, 1)
                if method_branch_key not in branch_risks:
                    missing = True
                    branch_harm_max = float("inf")
                else:
                    branch_harm_max = max(
                        branch_harm_max,
                        branch_risks[method_branch_key]
                        / max(branch_risks[ar2_branch_key], 1e-30),
                    )
            for horizon in (2, 3):
                candidate_key = (method, target_step, horizon)
                ar2_key = ("ar2", target_step, horizon)
                if candidate_key not in aggregate_risks:
                    missing = True
                    open_loop_ratio_max = float("inf")
                else:
                    open_loop_ratio_max = max(
                        open_loop_ratio_max,
                        aggregate_risks[candidate_key]
                        / max(aggregate_risks[ar2_key], 1e-30),
                    )
        coverage_pass = all(
            counts[step] >= int(gate["minimum_passing_layers_per_step"])
            for step in target_steps
        )
        recovery_pass = all(
            recoveries[step] >= float(gate["minimum_oracle_recovery"])
            for step in target_steps
        )
        branch_pass = branch_harm_max <= float(gate["maximum_branch_harm_ratio"])
        open_loop_pass = open_loop_ratio_max <= float(
            gate["maximum_open_loop_risk_ratio"]
        )
        formal_pass = (
            complete_selection
            and not missing
            and coverage_pass
            and recovery_pass
            and branch_pass
            and open_loop_pass
        )
        horizon1_rows = [
            row
            for row in rows
            if row["method"] == method and int(row["horizon"]) == 1
        ]
        total_error_sq = sum(float(row["error_sq"]) for row in horizon1_rows)
        total_target_sq = sum(float(row["target_sq"]) for row in horizon1_rows)
        mean_runtime_macs = sum(
            int(row["total_runtime_macs"]) for row in horizon1_rows
        ) // max(len(horizon1_rows), 1)
        method_rows.append(
            {
                "method": method,
                "selection_complete": complete_selection,
                "step4_passing_layers": counts[target_steps[0]],
                "step6_passing_layers": counts[target_steps[1]],
                "step4_oracle_recovery": recoveries[target_steps[0]],
                "step6_oracle_recovery": recoveries[target_steps[1]],
                "maximum_branch_harm_ratio": branch_harm_max,
                "maximum_open_loop_risk_ratio": open_loop_ratio_max,
                "coverage_pass": coverage_pass,
                "recovery_pass": recovery_pass,
                "branch_pass": branch_pass,
                "open_loop_pass": open_loop_pass,
                "missing_registered_metrics": missing,
                "aggregate_h1_residual_risk": (
                    total_error_sq / max(total_target_sq, 1e-30)
                )
                ** 0.5,
                "aggregate_h1_output_relative_l2": aggregate_output_relative_l2(
                    horizon1_rows
                ),
                "mean_h1_runtime_macs": mean_runtime_macs,
                "g024_pass": formal_pass,
            }
        )
    ranked = sorted(
        method_rows,
        key=lambda row: (
            not bool(row["g024_pass"]),
            -min(int(row["step4_passing_layers"]), int(row["step6_passing_layers"])),
            -min(
                float(row["step4_oracle_recovery"]),
                float(row["step6_oracle_recovery"]),
            ),
            float(row["maximum_open_loop_risk_ratio"]),
            str(row["method"]),
        ),
    )
    best_method = str(ranked[0]["method"])
    if not complete_selection:
        decision = "INCOMPLETE"
    elif any(bool(row["g024_pass"]) for row in method_rows):
        decision = "PASS"
    else:
        decision = "FAIL"
    return layer_rows, method_rows, decision, best_method


def plot_layer_reduction(
    layer_rows: list[dict[str, object]], best_method: str, path: Path
) -> None:
    selected = [row for row in layer_rows if row["method"] == best_method]
    blocks = sorted({int(row["block"]) for row in selected})
    steps = sorted({int(row["target_step"]) for row in selected})
    matrix = np.array(
        [
            [
                next(
                    float(row["risk_reduction"])
                    for row in selected
                    if int(row["block"]) == block
                    and int(row["target_step"]) == step
                )
                for step in steps
            ]
            for block in blocks
        ]
    )
    figure, axis = plt.subplots(figsize=(6.2, 7.0))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=max(2.5, matrix.max()))
    axis.set_xticks(range(len(steps)), [f"step {step}" for step in steps])
    axis.set_yticks(range(len(blocks)), [f"L{block}" for block in blocks])
    axis.set_title(f"Matched AR(2) risk reduction: {best_method}")
    for y_index in range(len(blocks)):
        for x_index in range(len(steps)):
            axis.text(
                x_index,
                y_index,
                f"{matrix[y_index, x_index]:.2f}x",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="AR(2) risk / method risk")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_open_loop(
    rows: list[dict[str, object]], methods: list[str], path: Path
) -> None:
    risks = indexed_risk(rows, ("method", "horizon"))
    horizons = (1, 2, 3)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    x_values = np.arange(len(horizons))
    for method in methods:
        ratios = []
        for horizon in horizons:
            key = (method, horizon)
            ar2_key = ("ar2", horizon)
            ratios.append(
                np.nan
                if key not in risks
                else risks[key] / max(risks[ar2_key], 1e-30)
            )
        axis.plot(x_values, ratios, marker="o", label=method)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x_values, ["H1", "H2", "H3"])
    axis.set_ylabel("Risk / matched AR(2) risk")
    axis.set_title("Open-loop stability guard")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_quality_cost(method_rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for row in method_rows:
        x_value = float(row["mean_h1_runtime_macs"]) / 1e9
        y_value = 100 * float(row["aggregate_h1_output_relative_l2"])
        axis.scatter(x_value, y_value, s=50)
        axis.annotate(
            str(row["method"]),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Estimated H1 runtime MACs (billions)")
    axis.set_ylabel("Block-output relative L2 (%)")
    axis.set_title("Observable-inclusive quality/cost diagnostic")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    *,
    split: str,
    decision: str,
    best_method: str,
    sample_count: int,
    layer_count: int,
    method_rows: list[dict[str, object]],
) -> None:
    best = next(row for row in method_rows if row["method"] == best_method)
    lines = [
        "# EXP-045 current-input observability 分析",
        "",
        f"- 数据 split：`{split}`",
        f"- identity 数：`{sample_count}`",
        f"- G-024 决策：`{decision}`",
        f"- 当前最强因果方法：`{best_method}`",
        "- 本报告只讨论 Wan 去噪采样步之间的 block residual 可观测性，不讨论视频物理时间预测。",
        "",
        "## 最强方法 Gate 分解",
        "",
        f"- step 4 通过层数：`{best['step4_passing_layers']}/{layer_count}`",
        f"- step 6 通过层数：`{best['step6_passing_layers']}/{layer_count}`",
        f"- step 4 oracle 恢复率：`{float(best['step4_oracle_recovery']):.3f}`",
        f"- step 6 oracle 恢复率：`{float(best['step6_oracle_recovery']):.3f}`",
        f"- 最大 CFG branch harm ratio：`{float(best['maximum_branch_harm_ratio']):.3f}`",
        f"- 最大 H2/H3 risk ratio：`{float(best['maximum_open_loop_risk_ratio']):.3f}`",
        "",
        "Oracle 恢复率严格使用 `(R_AR2-R_method)/(R_AR2-R_oracle)`；target-visible oracle 仅是上界，不能成为运行时方法。",
        "",
        "## 方法总表",
        "",
        "| 方法 | L@step4 | L@step6 | recovery4 | recovery6 | output L2 | MACs | open-loop | G-024 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(method_rows, key=lambda value: str(value["method"])):
        lines.append(
            f"| {row['method']} | {row['step4_passing_layers']} | "
            f"{row['step6_passing_layers']} | "
            f"{float(row['step4_oracle_recovery']):.3f} | "
            f"{float(row['step6_oracle_recovery']):.3f} | "
            f"{100 * float(row['aggregate_h1_output_relative_l2']):.3f}% | "
            f"{float(row['mean_h1_runtime_macs']) / 1e9:.3f}G | "
            f"{float(row['maximum_open_loop_risk_ratio']):.3f} | "
            f"{'PASS' if row['g024_pass'] else 'FAIL'} |"
        )
    if decision == "INCOMPLETE":
        lines.extend(
            [
                "",
                "当前不是正式 Gate 结论：selection identity 尚未完整达到 4 个。",
            ]
        )
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
    aggregated_rows = aggregate(
        rows, ("method", "block", "target_step", "branch", "horizon")
    )
    layer_rows, method_rows, decision, best_method = build_gate_tables(
        rows, config, args.split
    )
    write_csv(args.out_dir / "aggregated_cells.csv", aggregated_rows)
    write_csv(args.out_dir / "layer_gate.csv", layer_rows)
    write_csv(args.out_dir / "method_gate.csv", method_rows)
    plot_layer_reduction(
        layer_rows, best_method, args.out_dir / "risk_reduction_by_layer.png"
    )
    plot_open_loop(
        rows,
        [str(row["method"]) for row in method_rows],
        args.out_dir / "open_loop_stability.png",
    )
    plot_quality_cost(method_rows, args.out_dir / "quality_cost.png")
    sample_count = len({str(row["sample_id"]) for row in rows})
    write_report(
        args.out_dir / "report.zh-CN.md",
        split=args.split,
        decision=decision,
        best_method=best_method,
        sample_count=sample_count,
        layer_count=len(config["blocks"]),
        method_rows=method_rows,
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "split": args.split,
        "sample_count": sample_count,
        "decision": decision,
        "best_method": best_method,
        "final_split_opened": False,
    }
    (args.out_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
