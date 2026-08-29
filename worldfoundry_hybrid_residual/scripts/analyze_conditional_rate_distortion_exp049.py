#!/usr/bin/env python3
"""Run the exposed-data Stage-0 screen for EXP-049."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from conditional_rate_distortion_core import (
    ModuleTrajectory,
    apply_diagonal_field,
    apply_scalar_ar2,
    error_terms,
    fit_diagonal_field,
    fit_scalar_ar2,
    oracle_gap_recovery,
    zero_cost_speedup_ceiling,
)


CellKey = tuple[str, int, int, int]
TrajectoryKey = tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_payload(payload_root: Path, sample_id: str) -> Path:
    matches = sorted(
        payload_root.glob(f"{sample_id}*/wan_module_trajectory_samples.pt")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one payload for {sample_id}, found {len(matches)}"
        )
    return matches[0]


def _stack(samples: list[dict[str, object]], field: str, device: str) -> torch.Tensor:
    tensors = [sample[field] for sample in samples]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(f"capture field {field} must contain tensors")
    return torch.stack(tensors).to(device=device, dtype=torch.float32)


def load_trajectories(
    path: Path,
    sample_id: str,
    targets: tuple[str, ...],
    device: str,
) -> dict[TrajectoryKey, ModuleTrajectory]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported capture schema in {path}")
    if len(payload["runs"]) != 1:
        raise ValueError(f"screen payload must contain one dense run: {path}")
    run = payload["runs"][0]
    trajectories: dict[TrajectoryKey, ModuleTrajectory] = {}
    for cell in run["cells"]:
        block = int(cell["block"])
        branch = int(cell["branch"])
        samples = sorted(cell["samples"], key=lambda item: int(item["step"]))
        if [int(sample["step"]) for sample in samples] != list(range(len(samples))):
            raise ValueError(f"non-contiguous steps in {sample_id} block {block}")
        for sample in samples:
            if float(sample["exact_replay_relative_l2"]) > 1e-5:
                raise RuntimeError("persisted capture fails exact replay parity")
            if float(sample["sampled_additive_floor_relative_l2"]) > 1e-2:
                raise RuntimeError("persisted capture fails additive decomposition")
        block_input = _stack(samples, "block_input", device)
        block_output = _stack(samples, "block_output", device)
        self_attn = _stack(samples, "self_attn_contribution", device)
        cross_attn = _stack(samples, "cross_attn_contribution", device)
        ffn = _stack(samples, "ffn_contribution", device)
        target_map = {
            "self_attn": (block_input, self_attn),
            "ffn": (block_input + self_attn + cross_attn, ffn),
            "whole_block": (
                block_input,
                _stack(samples, "block_residual", device),
            ),
        }
        for target_name in targets:
            interface, target = target_map[target_name]
            trajectory = ModuleTrajectory(
                sample_id=sample_id,
                target_name=target_name,
                block=block,
                branch=branch,
                interface=interface,
                target=target,
                block_output=block_output,
            )
            trajectory.validate()
            trajectories[(target_name, block, branch)] = trajectory
    return trajectories


def fit_artifacts(
    calibration: list[dict[TrajectoryKey, ModuleTrajectory]],
    target_steps: tuple[int, ...],
    ridge: float,
) -> tuple[dict[CellKey, object], dict[CellKey, object]]:
    if not calibration:
        raise ValueError("calibration split cannot be empty")
    keys = set(calibration[0])
    if any(set(item) != keys for item in calibration[1:]):
        raise ValueError("calibration payloads do not share trajectory cells")
    diagonal: dict[CellKey, object] = {}
    ar2: dict[CellKey, object] = {}
    for target_name, block, branch in sorted(keys):
        trajectories = [item[(target_name, block, branch)] for item in calibration]
        for target_step in target_steps:
            key = (target_name, block, branch, target_step)
            diagonal[key] = fit_diagonal_field(trajectories, target_step, ridge)
            ar2[key] = fit_scalar_ar2(trajectories, target_step, ridge)
    return diagonal, ar2


def evaluate_split(
    split: str,
    payloads: list[dict[TrajectoryKey, ModuleTrajectory]],
    target_steps: tuple[int, ...],
    diagonal: dict[CellKey, object],
    ar2: dict[CellKey, object],
    ridge: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectories in payloads:
        for (target_name, block, branch), trajectory in sorted(trajectories.items()):
            for target_step in target_steps:
                key = (target_name, block, branch, target_step)
                predictions = {
                    "reuse": trajectory.target[target_step - 1],
                    "ar2": apply_scalar_ar2(ar2[key], trajectory, target_step),
                    "diagonal": apply_diagonal_field(
                        diagonal[key], trajectory, target_step
                    ),
                    "target_visible_diagonal": apply_diagonal_field(
                        fit_diagonal_field([trajectory], target_step, ridge),
                        trajectory,
                        target_step,
                    ),
                }
                for method, prediction in predictions.items():
                    metrics = error_terms(prediction, trajectory, target_step)
                    rows.append(
                        {
                            "split": split,
                            "sample_id": trajectory.sample_id,
                            "target": target_name,
                            "block": block,
                            "branch": branch,
                            "target_step": target_step,
                            "method": method,
                            **metrics,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_risk(rows: list[dict[str, object]]) -> float:
    error_sq = sum(float(row["error_sq"]) for row in rows)
    reference_sq = sum(float(row["block_output_sq"]) for row in rows)
    return (error_sq / max(reference_sq, 1e-30)) ** 0.5


def worst_identity_branch_risk(rows: list[dict[str, object]]) -> float:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["sample_id"]), int(row["branch"]))].append(row)
    if not groups:
        return float("inf")
    return max(aggregate_risk(group) for group in groups.values())


def method_rows(
    rows: list[dict[str, object]],
    method: str,
    target: str,
    selected: set[CellKey],
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and row["target"] == target
        and (
            str(row["target"]),
            int(row["block"]),
            int(row["branch"]),
            int(row["target_step"]),
        )
        in selected
    ]


def selection_risk_index(
    rows: list[dict[str, object]], method: str
) -> dict[CellKey, float]:
    groups: dict[CellKey, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        key = (
            str(row["target"]),
            int(row["block"]),
            int(row["branch"]),
            int(row["target_step"]),
        )
        groups[key].append(row)
    return {key: aggregate_risk(group) for key, group in groups.items()}


def build_frontier(
    selection_rows: list[dict[str, object]],
    heldout_rows: list[dict[str, object]],
    config: dict[str, object],
) -> list[dict[str, object]]:
    selection_index = selection_risk_index(selection_rows, "diagonal")
    targets = tuple(str(value) for value in config["targets"])
    thresholds = tuple(float(value) for value in config["frontier_thresholds"])
    runtime_share = config["optimistic_target_runtime_share"]
    screen_gate = config["screen_gate"]
    frontier: list[dict[str, object]] = []
    for target in targets:
        target_keys = {key for key in selection_index if key[0] == target}
        for threshold in thresholds:
            selected = {
                key for key in target_keys if selection_index[key] <= threshold
            }
            selected_fraction = len(selected) / max(len(target_keys), 1)
            method_metrics: dict[str, tuple[float, float]] = {}
            for method in ("ar2", "diagonal", "target_visible_diagonal"):
                selected_rows = method_rows(heldout_rows, method, target, selected)
                method_metrics[method] = (
                    aggregate_risk(selected_rows) if selected_rows else float("inf"),
                    worst_identity_branch_risk(selected_rows),
                )
            ar2_risk, _ = method_metrics["ar2"]
            deployable_risk, deployable_worst = method_metrics["diagonal"]
            oracle_risk, oracle_worst = method_metrics["target_visible_diagonal"]
            recovery = oracle_gap_recovery(ar2_risk, deployable_risk, oracle_risk)
            ceiling = zero_cost_speedup_ceiling(
                float(runtime_share[target]), selected_fraction
            )
            quality_pass = (
                oracle_risk <= float(screen_gate["target_visible_aggregate"])
                and oracle_worst <= float(screen_gate["target_visible_worst"])
                and deployable_risk <= float(screen_gate["deployable_aggregate"])
                and deployable_worst <= float(screen_gate["deployable_worst"])
                and recovery >= float(screen_gate["minimum_oracle_recovery"])
            )
            frontier.append(
                {
                    "target": target,
                    "selection_threshold": threshold,
                    "selected_cells": len(selected),
                    "total_cells": len(target_keys),
                    "selected_fraction": selected_fraction,
                    "ar2_risk": ar2_risk,
                    "deployable_risk": deployable_risk,
                    "deployable_worst": deployable_worst,
                    "target_visible_risk": oracle_risk,
                    "target_visible_worst": oracle_worst,
                    "oracle_gap_recovery": recovery,
                    "optimistic_target_runtime_share": float(runtime_share[target]),
                    "zero_renderer_e2e_speedup_ceiling": ceiling,
                    "local_quality_pass": quality_pass,
                    "screen_pass": quality_pass
                    and ceiling
                    >= float(screen_gate["minimum_e2e_speedup_ceiling"]),
                    "evidence_scope": "exposed_sampled_rows_local_additive_screen",
                }
            )
    return frontier


def serialize_artifacts(
    diagonal: dict[CellKey, object], ar2: dict[CellKey, object]
) -> dict[str, object]:
    return {
        "diagonal": {key: value.coefficients.cpu() for key, value in diagonal.items()},
        "ar2": {
            key: torch.tensor([value.lag1, value.lag2], dtype=torch.float64)
            for key, value in ar2.items()
        },
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    targets = tuple(str(value) for value in config["targets"])
    target_steps = tuple(int(value) for value in config["target_steps"])
    split_plan = config["screen_split"]

    calibration = [
        load_trajectories(
            resolve_payload(args.payload_root, sample_id),
            sample_id,
            targets,
            args.device,
        )
        for sample_id in split_plan["calibration"]
    ]
    diagonal, ar2 = fit_artifacts(
        calibration, target_steps, float(config["ridge"])
    )
    torch.save(serialize_artifacts(diagonal, ar2), args.out_dir / "frozen_fields.pt")

    selection_payloads = [
        load_trajectories(
            resolve_payload(args.payload_root, sample_id),
            sample_id,
            targets,
            args.device,
        )
        for sample_id in split_plan["selection"]
    ]
    selection_rows = evaluate_split(
        "selection",
        selection_payloads,
        target_steps,
        diagonal,
        ar2,
        float(config["ridge"]),
    )

    heldout_payloads = [
        load_trajectories(
            resolve_payload(args.payload_root, sample_id),
            sample_id,
            targets,
            args.device,
        )
        for sample_id in split_plan["heldout"]
    ]
    heldout_rows = evaluate_split(
        "heldout",
        heldout_payloads,
        target_steps,
        diagonal,
        ar2,
        float(config["ridge"]),
    )
    all_rows = selection_rows + heldout_rows
    write_csv(args.out_dir / "local_rows.csv", all_rows)
    frontier = build_frontier(selection_rows, heldout_rows, config)
    write_csv(args.out_dir / "local_frontier.csv", frontier)

    policy_threshold = float(config["screen_policy_threshold"])
    policy_rows = [
        row
        for row in frontier
        if float(row["selection_threshold"]) == policy_threshold
    ]
    promoted = [str(row["target"]) for row in policy_rows if row["screen_pass"]]
    summary = {
        "experiment_id": config["experiment_id"],
        "stage": "exposed_local_screen",
        "device": args.device,
        "calibration_ids": split_plan["calibration"],
        "selection_ids": split_plan["selection"],
        "heldout_ids": split_plan["heldout"],
        "row_count": len(all_rows),
        "frontier_row_count": len(frontier),
        "policy_threshold": policy_threshold,
        "policy_rows": policy_rows,
        "promoted_targets": promoted,
        "gate_eligible": False,
        "limitation": (
            "EXP-003 sampled-row additive local screen; no suffix intervention "
            "or H200 candidate timing"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
