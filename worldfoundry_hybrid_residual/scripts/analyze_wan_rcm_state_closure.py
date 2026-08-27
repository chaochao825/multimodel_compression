#!/usr/bin/env python3
"""Analyze the frozen EXP-048 selection endpoint and apply G-027."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRING_FIELDS = {
    "sample_id",
    "model",
    "input_trajectory",
    "basis_scope",
    "method",
}
INTEGER_FIELDS = {
    "sample_index",
    "seed",
    "block",
    "rank",
    "target_stage",
    "horizon",
    "state_bytes_bf16",
    "basis_bytes_bf16",
    "estimated_macs",
}
FLOAT_FIELDS = {
    "relative_l2",
    "output_relative_l2",
    "error_sq",
    "target_sq",
    "output_target_sq",
}
REQUIRED_FIELDS = STRING_FIELDS | INTEGER_FIELDS | FLOAT_FIELDS
MODELS = ("teacher", "rcm")
TRAJECTORIES = ("native4", "rcm4")
BLOCKS = tuple(range(20, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-048" or config["gate_id"] != "G-027":
        raise ValueError("config is not the frozen EXP-048/G-027 configuration")
    return config


def load_rows(path: Path, config: dict[str, object]) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != REQUIRED_FIELDS:
            raise ValueError("selection metric schema does not match EXP-048")
        rows = []
        for raw in reader:
            row: dict[str, object] = {}
            for field in STRING_FIELDS:
                row[field] = raw[field]
            for field in INTEGER_FIELDS:
                row[field] = int(raw[field])
            for field in FLOAT_FIELDS:
                value = float(raw[field])
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite metric in {field}")
                row[field] = value
            rows.append(row)

    expected_samples = tuple(int(value) for value in config["splits"]["selection"])
    if tuple(sorted({int(row["sample_index"]) for row in rows})) != expected_samples:
        raise ValueError("selection metrics do not contain the frozen identities")
    ranks = tuple(int(value) for value in config["state"]["ranks"])
    expected_per_cell = 4 + 8 + 11
    expected_rows = (
        len(expected_samples)
        * len(MODELS)
        * len(TRAJECTORIES)
        * 2
        * len(BLOCKS)
        * len(ranks)
        * expected_per_cell
    )
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} metric rows, found {len(rows)}")
    keys = [
        (
            row["sample_index"],
            row["model"],
            row["input_trajectory"],
            row["basis_scope"],
            row["block"],
            row["rank"],
            row["method"],
            row["target_stage"],
            row["horizon"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("selection metrics contain duplicate cells")
    return rows


def subset(rows: list[dict[str, object]], **filters: object) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if all(row[field] == expected for field, expected in filters.items())
    ]
    if not selected:
        raise ValueError(f"empty metric subset: {filters}")
    return selected


def output_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    error_sq = sum(float(row["error_sq"]) for row in rows)
    output_target_sq = sum(float(row["output_target_sq"]) for row in rows)
    return {
        "aggregate": math.sqrt(error_sq / max(output_target_sq, 1e-30)),
        "worst": max(float(row["output_relative_l2"]) for row in rows),
        "rows": len(rows),
    }


def energy_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    error_sq = sum(float(row["error_sq"]) for row in rows)
    target_sq = sum(float(row["target_sq"]) for row in rows)
    output_target_sq = sum(float(row["output_target_sq"]) for row in rows)
    return {
        "residual_relative_l2": math.sqrt(error_sq / max(target_sq, 1e-30)),
        "output_relative_l2": math.sqrt(error_sq / max(output_target_sq, 1e-30)),
        "captured_residual_energy": 1.0 - error_sq / max(target_sq, 1e-30),
        "residual_to_output_scale": math.sqrt(target_sq / max(output_target_sq, 1e-30)),
        "worst_output_relative_l2": max(
            float(row["output_relative_l2"]) for row in rows
        ),
        "rows": len(rows),
    }


def metric_slice(
    rows: list[dict[str, object]],
    *,
    model: str,
    trajectory: str,
    block: int | None,
    rank: int,
    method: str,
    horizon: int | None = None,
    basis_scope: str = "model_specific",
    target_stages: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    filters: dict[str, object] = {
        "model": model,
        "input_trajectory": trajectory,
        "basis_scope": basis_scope,
        "rank": rank,
        "method": method,
    }
    if block is not None:
        filters["block"] = block
    if horizon is not None:
        filters["horizon"] = horizon
    selected = subset(rows, **filters)
    if target_stages is not None:
        selected = [row for row in selected if int(row["target_stage"]) in target_stages]
        if not selected:
            raise ValueError("target-stage filtering produced an empty metric subset")
    return selected


def relative_improvement(reference: float, candidate: float) -> float:
    return 1.0 - candidate / max(reference, 1e-30)


def block_summary(
    rows: list[dict[str, object]], thresholds: dict[str, object]
) -> list[dict[str, object]]:
    summaries = []
    for block in BLOCKS:
        capacity = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="capacity",
            )
        )
        h1 = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="drift",
                horizon=1,
            )
        )
        h2 = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="drift",
                horizon=2,
            )
        )
        h3 = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="drift",
                horizon=3,
            )
        )
        weight_values: dict[str, float] = {}
        for trajectory in TRAJECTORIES:
            teacher = output_summary(
                metric_slice(
                    rows,
                    model="teacher",
                    trajectory=trajectory,
                    block=block,
                    rank=64,
                    method="drift",
                    horizon=1,
                )
            )["aggregate"]
            rcm = output_summary(
                metric_slice(
                    rows,
                    model="rcm",
                    trajectory=trajectory,
                    block=block,
                    rank=64,
                    method="drift",
                    horizon=1,
                )
            )["aggregate"]
            weight_values[f"teacher_{trajectory}"] = teacher
            weight_values[f"rcm_{trajectory}"] = rcm
            weight_values[f"weight_improvement_{trajectory}"] = relative_improvement(
                teacher, rcm
            )

        shared_h1 = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="drift",
                horizon=1,
                basis_scope="shared",
            )
        )["aggregate"]
        drift_comparable = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="drift",
                horizon=1,
                target_stages=(2, 3),
            )
        )["aggregate"]
        ar2 = output_summary(
            metric_slice(
                rows,
                model="rcm",
                trajectory="rcm4",
                block=block,
                rank=64,
                method="ar2_drift",
                horizon=1,
            )
        )["aggregate"]
        values: dict[str, object] = {
            "block": block,
            "capacity_aggregate": capacity["aggregate"],
            "capacity_worst": capacity["worst"],
            "h1_aggregate": h1["aggregate"],
            "h1_worst": h1["worst"],
            "h2_aggregate": h2["aggregate"],
            "h2_worst": h2["worst"],
            "h3_aggregate": h3["aggregate"],
            "h3_worst": h3["worst"],
            "two_lag_advantage": relative_improvement(drift_comparable, ar2),
            "shared_basis_penalty": shared_h1 / max(float(h1["aggregate"]), 1e-30) - 1.0,
        }
        values.update(weight_values)
        values["capacity_pass"] = (
            float(capacity["aggregate"])
            <= float(thresholds["maximum_capacity_aggregate"])
            and float(capacity["worst"])
            <= float(thresholds["maximum_capacity_worst"])
        )
        values["h1_pass"] = (
            float(h1["aggregate"]) <= float(thresholds["maximum_h1_aggregate"])
            and float(h1["worst"]) <= float(thresholds["maximum_h1_worst"])
        )
        values["weight_pass"] = all(
            float(values[f"weight_improvement_{trajectory}"])
            >= float(thresholds["minimum_weight_improvement"]) - 1.0
            for trajectory in TRAJECTORIES
        )
        summaries.append(values)
    return summaries


def pooled_gate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    h2 = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="drift",
            horizon=2,
        )
    )
    h3 = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="drift",
            horizon=3,
        )
    )
    drift = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="drift",
            horizon=1,
            target_stages=(2, 3),
        )
    )["aggregate"]
    ar2 = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="ar2_drift",
            horizon=1,
        )
    )["aggregate"]
    model_specific = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="drift",
            horizon=1,
        )
    )["aggregate"]
    shared = output_summary(
        metric_slice(
            rows,
            model="rcm",
            trajectory="rcm4",
            block=None,
            rank=64,
            method="drift",
            horizon=1,
            basis_scope="shared",
        )
    )["aggregate"]
    return {
        "h2": h2,
        "h3": h3,
        "two_lag_advantage": relative_improvement(float(drift), float(ar2)),
        "shared_basis_penalty": float(shared) / max(float(model_specific), 1e-30) - 1.0,
    }


def cross_effects(rows: list[dict[str, object]]) -> list[dict[str, float | int]]:
    effects = []
    for block in BLOCKS:
        risk = {}
        for model in MODELS:
            for trajectory in TRAJECTORIES:
                summary = output_summary(
                    metric_slice(
                        rows,
                        model=model,
                        trajectory=trajectory,
                        block=block,
                        rank=64,
                        method="drift",
                        horizon=1,
                    )
                )
                risk[(model, trajectory)] = float(summary["aggregate"]) ** 2
        tt = risk[("teacher", "native4")]
        tr = risk[("teacher", "rcm4")]
        rt = risk[("rcm", "native4")]
        rr = risk[("rcm", "rcm4")]
        effects.append(
            {
                "block": block,
                "teacher_native4_risk": tt,
                "teacher_rcm4_risk": tr,
                "rcm_native4_risk": rt,
                "rcm_rcm4_risk": rr,
                "weight_log_effect_native4": math.log(rt / max(tt, 1e-30)),
                "weight_log_effect_rcm4": math.log(rr / max(tr, 1e-30)),
                "trajectory_log_effect_teacher": math.log(tr / max(tt, 1e-30)),
                "trajectory_log_effect_rcm": math.log(rr / max(rt, 1e-30)),
                "interaction_log_risk": math.log(rr)
                - math.log(rt)
                - math.log(tr)
                + math.log(tt),
            }
        )
    return effects


def rank_sweep(rows: list[dict[str, object]], ranks: tuple[int, ...]) -> list[dict[str, object]]:
    summaries = []
    for rank in ranks:
        for method, horizon in (("capacity", None), ("drift", 1)):
            summary = output_summary(
                metric_slice(
                    rows,
                    model="rcm",
                    trajectory="rcm4",
                    block=None,
                    rank=rank,
                    method=method,
                    horizon=horizon,
                )
            )
            summaries.append(
                {
                    "rank": rank,
                    "method": method,
                    "aggregate": summary["aggregate"],
                    "worst": summary["worst"],
                }
            )
    return summaries


def mechanism_sweep(
    rows: list[dict[str, object]], ranks: tuple[int, ...]
) -> list[dict[str, object]]:
    summaries = []
    for model in MODELS:
        for trajectory in TRAJECTORIES:
            for rank in ranks:
                for method, horizon in (("capacity", None), ("drift", 1)):
                    metrics = energy_summary(
                        metric_slice(
                            rows,
                            model=model,
                            trajectory=trajectory,
                            block=None,
                            rank=rank,
                            method=method,
                            horizon=horizon,
                        )
                    )
                    summaries.append(
                        {
                            "model": model,
                            "input_trajectory": trajectory,
                            "rank": rank,
                            "method": method,
                            **metrics,
                        }
                    )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_h1_cross(rows: list[dict[str, object]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    styles = {
        ("teacher", "native4"): ("#7A7265", "o"),
        ("teacher", "rcm4"): ("#D9822B", "s"),
        ("rcm", "native4"): ("#2C7A7B", "^"),
        ("rcm", "rcm4"): ("#1D3557", "D"),
    }
    for model in MODELS:
        for trajectory in TRAJECTORIES:
            values = [
                100
                * float(
                    output_summary(
                        metric_slice(
                            rows,
                            model=model,
                            trajectory=trajectory,
                            block=block,
                            rank=64,
                            method="drift",
                            horizon=1,
                        )
                    )["aggregate"]
                )
                for block in BLOCKS
            ]
            color, marker = styles[(model, trajectory)]
            axis.plot(BLOCKS, values, marker=marker, color=color, label=f"{model}/{trajectory}")
    axis.axhline(1.0, color="#B23A48", linestyle="--", linewidth=1.2, label="G-027 aggregate")
    axis.set_xlabel("Wan block")
    axis.set_ylabel("H1 whole-block relative L2 (%)")
    axis.set_title("EXP-048 held-out first-order closure cross")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_open_loop(summary: list[dict[str, object]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, color, marker in (
        ("h1", "#1D3557", "o"),
        ("h2", "#D9822B", "s"),
        ("h3", "#B23A48", "^"),
    ):
        axis.plot(
            BLOCKS,
            [100 * float(row[f"{name}_aggregate"]) for row in summary],
            color=color,
            marker=marker,
            label=name.upper(),
        )
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axhline(2.0, color="#666666", linestyle=":", linewidth=1.0)
    axis.set_xlabel("Wan block")
    axis.set_ylabel("Whole-block relative L2 (%)")
    axis.set_title("rCM weights on rCM4: open-loop state closure")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rank_sweep(summary: list[dict[str, object]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for method, color, marker in (
        ("capacity", "#2C7A7B", "o"),
        ("drift", "#1D3557", "s"),
    ):
        selected = [row for row in summary if row["method"] == method]
        axis.plot(
            [int(row["rank"]) for row in selected],
            [100 * float(row["aggregate"]) for row in selected],
            color=color,
            marker=marker,
            label=method,
        )
    axis.set_xlabel("State rank")
    axis.set_ylabel("Pooled whole-block relative L2 (%)")
    axis.set_title("EXP-048 payload sweep: rCM/rCM4")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_report(summary: dict[str, object]) -> str:
    gate = summary["gate"]
    pooled = summary["pooled"]
    return f"""# EXP-048 distillation-induced state-closure result

## Verdict

G-027 outcome: **{gate['outcome']}**.

- Capacity-pass layers: `{gate['capacity_pass_layers']}` / 10.
- H1-pass layers: `{gate['h1_pass_layers']}` / 10.
- Joint capacity/H1-pass layers: `{gate['joint_capacity_h1_layers']}` / 10.
- Weight-improvement-pass layers on both trajectories: `{gate['weight_pass_layers']}` / 10.
- Pooled H2 aggregate/worst: `{100 * pooled['h2']['aggregate']:.3f}%` / `{100 * pooled['h2']['worst']:.3f}%`.
- Pooled H3 aggregate/worst: `{100 * pooled['h3']['aggregate']:.3f}%` / `{100 * pooled['h3']['worst']:.3f}%`.
- Two-lag H1 advantage: `{100 * pooled['two_lag_advantage']:.3f}%`.
- Shared-basis H1 penalty: `{100 * pooled['shared_basis_penalty']:.3f}%`.

## Interpretation

This endpoint tests denoising-time low-rate state closure, not physical-time video memory.
All bases and transition coefficients are calibration-only; the four selection identities
are used only once for fixed evaluation. Whole-block-output error, rather than residual
energy capture, determines every G-027 threshold. The 2x2 cross separates the effect of
rCM weights from the effect of evaluating on the rCM4 latent trajectory.

No latency claim is made because GPU2 retained a stopped co-resident process during the
numerical experiment.
"""


def gate_decision(
    blocks: list[dict[str, object]],
    pooled: dict[str, object],
    thresholds: dict[str, object],
) -> dict[str, object]:
    capacity_pass = [int(row["block"]) for row in blocks if row["capacity_pass"]]
    h1_pass = [int(row["block"]) for row in blocks if row["h1_pass"]]
    joint_pass = sorted(set(capacity_pass) & set(h1_pass))
    weight_pass = [int(row["block"]) for row in blocks if row["weight_pass"]]
    stability_pass = (
        float(pooled["h2"]["aggregate"])
        <= float(thresholds["maximum_open_loop_aggregate"])
        and float(pooled["h2"]["worst"])
        <= float(thresholds["maximum_open_loop_worst"])
        and float(pooled["h3"]["aggregate"])
        <= float(thresholds["maximum_open_loop_aggregate"])
        and float(pooled["h3"]["worst"])
        <= float(thresholds["maximum_open_loop_worst"])
    )
    history_pass = float(pooled["two_lag_advantage"]) <= (
        float(thresholds["maximum_history_advantage"]) - 1.0
    )
    shared_pass = float(pooled["shared_basis_penalty"]) <= (
        float(thresholds["maximum_shared_basis_penalty"]) - 1.0
    )
    minimum_layers = int(thresholds["minimum_passing_layers"])
    full_pass = (
        len(joint_pass) >= minimum_layers
        and len(weight_pass) >= minimum_layers
        and stability_pass
        and history_pass
        and shared_pass
    )
    outcome = (
        "pass"
        if full_pass
        else "directional-only"
        if len(weight_pass) >= minimum_layers
        else "null/adverse"
    )
    return {
        "outcome": outcome,
        "capacity_pass_layers": capacity_pass,
        "h1_pass_layers": h1_pass,
        "joint_capacity_h1_layers": joint_pass,
        "weight_pass_layers": weight_pass,
        "stability_pass": stability_pass,
        "history_pass": history_pass,
        "shared_basis_pass": shared_pass,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    rows = load_rows(args.metrics.resolve(), config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output_dir}")
    output_dir.mkdir(parents=True)

    thresholds = config["gate"]
    blocks = block_summary(rows, thresholds)
    pooled = pooled_gate_metrics(rows)
    effects = cross_effects(rows)
    ranks = tuple(int(value) for value in config["state"]["ranks"])
    sweep = rank_sweep(rows, ranks)
    mechanisms = mechanism_sweep(rows, ranks)
    gate = gate_decision(blocks, pooled, thresholds)
    summary = {
        "experiment_id": "EXP-048",
        "gate_id": "G-027",
        "gate": gate,
        "pooled": pooled,
        "rank_sweep": sweep,
        "mechanism_sweep": mechanisms,
    }
    write_csv(output_dir / "block_gate_summary.csv", blocks)
    write_csv(output_dir / "cross_effects.csv", effects)
    write_csv(output_dir / "rank_sweep.csv", sweep)
    write_csv(output_dir / "mechanism_sweep.csv", mechanisms)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    plot_h1_cross(rows, output_dir / "h1_cross_by_block.png")
    plot_open_loop(blocks, output_dir / "open_loop_by_block.png")
    plot_rank_sweep(sweep, output_dir / "rank_sweep.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
