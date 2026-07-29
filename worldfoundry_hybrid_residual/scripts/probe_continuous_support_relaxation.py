#!/usr/bin/env python3
"""Continuous tile-weight upper bound for worst support-manifold records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiment_artifacts import atomic_write_csv, atomic_write_json, file_sha256, require_fresh_output_dir
from support_manifold_oracle_core import (
    GroupProblem,
    adaptive_tail,
    atom_statistics,
    contiguous_partition,
    optimize_support,
    slice_statistics,
)


TOPK_MULTIPLIERS = tuple(1.0 + 0.25 * index for index in range(13))


def multiplier_label(multiplier: float) -> str:
    return f"{multiplier:g}".replace(".", "p") + "x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-records", type=Path, required=True)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", default="hierarchical32")
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument(
        "--selection-density",
        type=float,
        help="Density used only to select registered worst records; defaults to --density.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--records-per-cell", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--refit-steps", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_worst_records(
    rows: list[dict[str, str]], family: str, density: float, records_per_cell: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row["cell"].startswith("layer14_")
            and row["family"] == family
            and math.isclose(float(row["density_target"]), density)
        ):
            grouped[row["cell"]].append(row)
    if not grouped:
        raise ValueError("no matching layer-14 support records")
    selected = []
    for cell, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["adaptive_output_relative_l2"]), reverse=True)
        selected.extend(ordered[:records_per_cell])
    return selected


def resolve_capture(index_path: Path, row: dict[str, str]) -> Path:
    matches = [
        item
        for item in read_csv(index_path)
        if item["sample_id"] == row["sample_id"]
        and item["branch"] == row["branch"]
        and int(item["layer"]) == int(row["layer"])
        and int(item["sampling_step"]) == int(row["sampling_step"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one capture for relaxation record; found {len(matches)}")
    path = Path(matches[0]["path"])
    if not path.is_absolute():
        path = index_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def budgeted_sigmoid(logits: torch.Tensor, budget: int, temperature: float) -> torch.Tensor:
    if logits.ndim != 1 or not 0 < budget < logits.numel() or temperature <= 0:
        raise ValueError("invalid budgeted sigmoid inputs")
    with torch.no_grad():
        lower = float(logits.min()) - 40.0 * temperature
        upper = float(logits.max()) + 40.0 * temperature
        for _ in range(64):
            threshold = (lower + upper) / 2.0
            total = float(torch.sigmoid((logits - threshold) / temperature).sum())
            if total > budget:
                lower = threshold
            else:
                upper = threshold
        threshold = (lower + upper) / 2.0
    return torch.sigmoid((logits - threshold) / temperature)


def weighted_defect(
    groups: tuple[GroupProblem, ...], weights: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    defects = []
    for problem, group_weights in zip(groups, weights):
        numerator = torch.einsum("b,bqd->qd", group_weights, problem.statistics.contributions)
        mass = torch.einsum("b,bq->q", group_weights, problem.statistics.mass)
        defects.append(problem.reference - numerator / mass.clamp_min(1e-30).unsqueeze(1))
    return torch.cat(defects)


def hard_defect(groups: tuple[GroupProblem, ...], logits: tuple[torch.Tensor, ...]) -> torch.Tensor:
    defects = []
    for problem, values in zip(groups, logits):
        selected = values.topk(problem.budget).indices
        numerator = problem.statistics.contributions.index_select(0, selected).sum(dim=0)
        mass = problem.statistics.mass.index_select(0, selected).sum(dim=0)
        defects.append(problem.reference - numerator / mass.clamp_min(1e-30).unsqueeze(1))
    return torch.cat(defects)


def weighted_topk_defect(
    groups: tuple[GroupProblem, ...],
    weights: tuple[torch.Tensor, ...],
    budget_multiplier: float,
) -> torch.Tensor:
    if budget_multiplier <= 0:
        raise ValueError("budget multiplier must be positive")
    defects = []
    for problem, values in zip(groups, weights):
        count = min(values.numel(), int(round(problem.budget * budget_multiplier)))
        selected = values.topk(count).indices
        selected_weights = values.index_select(0, selected)
        numerator = torch.einsum(
            "b,bqd->qd",
            selected_weights,
            problem.statistics.contributions.index_select(0, selected),
        )
        mass = torch.einsum(
            "b,bq->q", selected_weights, problem.statistics.mass.index_select(0, selected)
        )
        defects.append(problem.reference - numerator / mass.clamp_min(1e-30).unsqueeze(1))
    return torch.cat(defects)


def selected_weight_defect(
    groups: tuple[GroupProblem, ...],
    selections: tuple[torch.Tensor, ...],
    selected_weights: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    defects = []
    for problem, selected, values in zip(groups, selections, selected_weights):
        numerator = torch.einsum(
            "b,bqd->qd",
            values,
            problem.statistics.contributions.index_select(0, selected),
        )
        mass = torch.einsum(
            "b,bq->q", values, problem.statistics.mass.index_select(0, selected)
        )
        defects.append(problem.reference - numerator / mass.clamp_min(1e-30).unsqueeze(1))
    return torch.cat(defects)


def refit_weighted_topk(
    groups: tuple[GroupProblem, ...],
    initial_weights: tuple[torch.Tensor, ...],
    budget_multiplier: float,
    rank: int,
    steps: int,
    learning_rate: float,
) -> float:
    if steps < 0 or learning_rate <= 0:
        raise ValueError("invalid weighted top-k refit settings")
    selections = tuple(
        values.topk(min(values.numel(), int(round(problem.budget * budget_multiplier)))).indices
        for problem, values in zip(groups, initial_weights)
    )
    parameters = []
    for selected, values in zip(selections, initial_weights):
        bounded = values.index_select(0, selected).clamp(1e-4, 1.0 - 1e-4)
        parameters.append(torch.nn.Parameter(torch.logit(bounded)))
    reference_sq = sum(float(problem.reference.square().sum()) for problem in groups)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    best = float("inf")
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        weights = tuple(torch.sigmoid(parameter) for parameter in parameters)
        loss = residual_sq(selected_weight_defect(groups, selections, weights), rank)
        value = float(loss.detach())
        best = min(best, value)
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return math.sqrt(best / max(reference_sq, 1e-30))


def residual_sq(defect: torch.Tensor, rank: int) -> torch.Tensor:
    singular = torch.linalg.svdvals(defect)
    return singular[rank:].square().sum()


def optimize_relaxation(
    groups: tuple[GroupProblem, ...],
    initial_selections: tuple[torch.Tensor, ...],
    rank: int,
    steps: int,
    restarts: int,
    learning_rate: float,
    temperature: float,
    refit_steps: int = 80,
) -> dict[str, object]:
    reference_sq = sum(float(problem.reference.square().sum()) for problem in groups)
    hard_logits = []
    for problem, selected in zip(groups, initial_selections):
        logits = torch.zeros(
            problem.statistics.contributions.shape[0], device=problem.reference.device
        )
        logits[selected] = 1.0
        hard_logits.append(logits)
    hard_boundary = hard_defect(groups, tuple(hard_logits))
    hard_boundary_relative = math.sqrt(float(residual_sq(hard_boundary, rank)) / reference_sq)
    total_budget = sum(problem.budget for problem in groups)
    best: dict[str, object] | None = {
        "restart": -1,
        "solution_kind": "hard_feasible_boundary",
        "initial_fractional_output_relative_l2": hard_boundary_relative,
        "fractional_output_relative_l2": hard_boundary_relative,
        "thresholded_hard_output_relative_l2": hard_boundary_relative,
        "weight_sum": float(total_budget),
        "weight_effective_atoms": float(total_budget),
        "weight_near_binary_fraction": 1.0,
        "weight_dynamic_scalars": 0,
        **{
            f"weighted_topk_{multiplier_label(multiplier)}_output_relative_l2": hard_boundary_relative
            for multiplier in TOPK_MULTIPLIERS
        },
    }
    best_weights: tuple[torch.Tensor, ...] | None = None
    base_scores = [
        problem.statistics.contributions.square().sum(dim=(1, 2)) for problem in groups
    ]
    for restart in range(restarts):
        parameters = []
        for group_index, (scores, selected) in enumerate(zip(base_scores, initial_selections)):
            if restart == 0:
                normalized = torch.full_like(scores, -2.0)
                normalized[selected] = 2.0
            else:
                normalized = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(20260729 + 1009 * restart + group_index)
            noise = torch.randn(scores.shape, generator=generator, device=scores.device) * 0.05
            parameters.append(torch.nn.Parameter(normalized + noise))
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        initial_loss = None
        best_loss = float("inf")
        best_parameters: tuple[torch.Tensor, ...] | None = None
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            weights = tuple(
                budgeted_sigmoid(logits, problem.budget, temperature)
                for logits, problem in zip(parameters, groups)
            )
            defect = weighted_defect(groups, weights)
            loss = residual_sq(defect, rank) / max(reference_sq, 1e-30)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_parameters = tuple(parameter.detach().clone() for parameter in parameters)
            if step == steps:
                break
            loss.backward()
            optimizer.step()
        assert best_parameters is not None
        with torch.no_grad():
            weights = tuple(
                budgeted_sigmoid(logits, problem.budget, temperature)
                for logits, problem in zip(best_parameters, groups)
            )
            fractional = weighted_defect(groups, weights)
            hard = hard_defect(groups, best_parameters)
            fractional_sq = float(residual_sq(fractional, rank))
            hard_sq = float(residual_sq(hard, rank))
            weight_values = torch.cat(weights)
            candidate = {
                "restart": restart,
                "solution_kind": "fractional_relaxation",
                "initial_fractional_output_relative_l2": math.sqrt(
                    float(initial_loss or 0.0)
                ),
                "fractional_output_relative_l2": math.sqrt(fractional_sq / reference_sq),
                "thresholded_hard_output_relative_l2": math.sqrt(hard_sq / reference_sq),
                "weight_sum": float(weight_values.sum()),
                "weight_effective_atoms": float(weight_values.sum().square() / weight_values.square().sum()),
                "weight_near_binary_fraction": float(
                    ((weight_values < 0.05) | (weight_values > 0.95)).float().mean()
                ),
                "weight_dynamic_scalars": weight_values.numel(),
            }
            for multiplier in TOPK_MULTIPLIERS:
                topk = weighted_topk_defect(groups, weights, multiplier)
                candidate[
                    f"weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
                ] = math.sqrt(
                    float(residual_sq(topk, rank)) / reference_sq
                )
            if float(candidate["fractional_output_relative_l2"]) < float(best["fractional_output_relative_l2"]):
                best = candidate
                best_weights = tuple(weight.detach().clone() for weight in weights)
    assert best is not None
    if best_weights is not None:
        refit_cache: dict[tuple[int, ...], float] = {}
        for multiplier in TOPK_MULTIPLIERS:
            counts = tuple(
                min(weight.numel(), int(round(problem.budget * multiplier)))
                for problem, weight in zip(groups, best_weights)
            )
            if counts not in refit_cache:
                refit_cache[counts] = refit_weighted_topk(
                    groups,
                    best_weights,
                    multiplier,
                    rank,
                    refit_steps,
                    learning_rate,
                )
            best[
                f"refit_weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
            ] = refit_cache[counts]
    else:
        for multiplier in TOPK_MULTIPLIERS:
            best[
                f"refit_weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
            ] = hard_boundary_relative
    return best


def process_record(
    row: dict[str, str],
    capture_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    head = int(row["head"])
    start = int(row["query_start"])
    q = payload["q"][0, :, head].to(device=device, dtype=torch.float32)
    k = payload["k"][0, :, head].to(device=device, dtype=torch.float32)
    v = payload["v"][0, :, head].to(device=device, dtype=torch.float32)
    scale = float(payload.get("softmax_scale", q.shape[1] ** -0.5))
    attention = torch.softmax(q[start : start + 64] @ k.T * scale, dim=1)
    reference = attention @ v
    statistics = atom_statistics(attention, v, contiguous_partition(v.shape[0], 32))
    budget = max(1, int(round(args.density * v.shape[0] / 32)))
    groups = (
        GroupProblem(reference[:32], slice_statistics(statistics, 0, 32), budget),
        GroupProblem(reference[32:], slice_statistics(statistics, 32, 64), budget),
    )
    hard = optimize_support(groups, args.rank, alternations=1, swap_steps=1, shortlist=8)
    relaxed = optimize_relaxation(
        groups,
        hard.selections,
        args.rank,
        args.steps,
        args.restarts,
        args.learning_rate,
        args.temperature,
        args.refit_steps,
    )
    return {
        "sample_id": row["sample_id"],
        "cell": row["cell"],
        "layer": int(row["layer"]),
        "sampling_step": int(row["sampling_step"]),
        "head": head,
        "tile_index": int(row["tile_index"]),
        "query_start": start,
        "density": args.density,
        "selection_density": (
            args.selection_density if args.selection_density is not None else args.density
        ),
        "rank": args.rank,
        "discrete_screen_output_relative_l2": float(row["adaptive_output_relative_l2"]),
        "discrete_recomputed_output_relative_l2": math.sqrt(
            hard.residual_sq / float(reference.square().sum())
        ),
        **relaxed,
        "oracle_access": "worst-record selection and all fractional tile weights inspect heldout dense AV",
        "deployability": "nondeployable continuous upper bound; thresholded_hard is diagnostic only",
    }


def main() -> None:
    args = parse_args()
    require_fresh_output_dir(args.output_dir)
    records_path = args.support_records.resolve()
    capture_index = args.capture_index.resolve()
    selection_density = (
        args.selection_density if args.selection_density is not None else args.density
    )
    selected = select_worst_records(
        read_csv(records_path), args.family, selection_density, args.records_per_cell
    )
    device = torch.device(args.device)
    rows = []
    capture_hashes = {}
    started = time.time()
    for index, row in enumerate(selected):
        path = resolve_capture(capture_index, row)
        capture_hashes[str(path)] = file_sha256(path)
        result = process_record(row, path, args, device)
        rows.append(result)
        print(
            f"[support-relax] {index + 1}/{len(selected)} cell={row['cell']} "
            f"hard={100*float(result['discrete_recomputed_output_relative_l2']):.3f}% "
            f"fractional={100*float(result['fractional_output_relative_l2']):.3f}%",
            flush=True,
        )
    atomic_write_csv(args.output_dir / "continuous_support_relaxation.csv", rows)
    decision = {
        "all_fractional_records_pass_1pct": all(
            float(row["fractional_output_relative_l2"]) <= 0.01 for row in rows
        ),
        "all_thresholded_records_pass_1pct": all(
            float(row["thresholded_hard_output_relative_l2"]) <= 0.01 for row in rows
        ),
        "max_fractional_output_relative_l2": max(
            float(row["fractional_output_relative_l2"]) for row in rows
        ),
        "max_thresholded_hard_output_relative_l2": max(
            float(row["thresholded_hard_output_relative_l2"]) for row in rows
        ),
        "max_weighted_topk_output_relative_l2": {
            multiplier_label(multiplier): max(
                float(
                    row[
                        f"weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
                    ]
                )
                for row in rows
            )
            for multiplier in TOPK_MULTIPLIERS
        },
        "max_refit_weighted_topk_output_relative_l2": {
            multiplier_label(multiplier): max(
                float(
                    row[
                        f"refit_weighted_topk_{multiplier_label(multiplier)}_output_relative_l2"
                    ]
                )
                for row in rows
            )
            for multiplier in TOPK_MULTIPLIERS
        },
        "verdict": "CONTINUOUS_SUPPORT_UPPER_BOUND_ONLY",
    }
    atomic_write_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args)
        | {
            "support_records": str(args.support_records),
            "capture_index": str(args.capture_index),
            "output_dir": str(args.output_dir),
        },
        "support_records_sha256": file_sha256(records_path),
        "capture_index_sha256": file_sha256(capture_index),
        "capture_sha256": capture_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "claim_boundary": "Post-hoc nondeployable continuous tile-weight upper bound on records selected as worst by the completed discrete screen.",
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    artifacts = {
        name: file_sha256(args.output_dir / name)
        for name in ("continuous_support_relaxation.csv", "decision.json", "manifest.json")
    }
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {"verdict": decision["verdict"], "artifact_sha256": artifacts},
    )


if __name__ == "__main__":
    main()
