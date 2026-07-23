#!/usr/bin/env python3
"""Build conservative tri-mode combinations from measured single-cell probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Cell:
    action: str
    step_start: int
    step_end: int
    block_start: int
    block_end: int
    quality_loss: float
    saving: float
    branches: tuple[int, ...] = (0, 1)
    forecast_scale: float = 0.0

    @property
    def position(self) -> tuple[int, int, int, int]:
        return self.step_start, self.step_end, self.block_start, self.block_end

    @property
    def label(self) -> str:
        branch_suffix = "" if self.branches == (0, 1) else "_br" + "".join(
            map(str, self.branches)
        )
        forecast_suffix = (
            ""
            if self.forecast_scale == 0.0
            else f"_f{'m' if self.forecast_scale < 0 else ''}"
            f"{round(abs(self.forecast_scale) * 100):03d}"
        )
        return (
            f"{self.action.lower()}_s{self.step_start:02d}-{self.step_end:02d}_"
            f"b{self.block_start:02d}-{self.block_end:02d}"
            f"{branch_suffix}{forecast_suffix}"
        )

    @property
    def slots(self) -> frozenset[tuple[int, int, int]]:
        return frozenset(
            (step, block, branch)
            for step in range(self.step_start, self.step_end + 1)
            for block in range(self.block_start, self.block_end + 1)
            for branch in self.branches
        )


@dataclass(frozen=True)
class BeamState:
    selected: tuple[int, ...]
    quality_loss: float
    saving: float
    last_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--source-schedules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-loss-budget", type=float, default=0.02)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def global_action_costs(
    method_rows: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    methods = {row["method"]: row for row in method_rows}
    costs: dict[str, dict[str, float]] = {}
    for action in ("Q", "C"):
        row = methods.get(f"all_{action.lower()}")
        if row is None:
            continue
        speedup = float(row["speedup_geomean"])
        count_field = "requested_quant_mean" if action == "Q" else "executed_cache_mean"
        costs[action] = {
            "fractional_saving": 1.0 - 1.0 / speedup,
            "action_calls": float(row[count_field]),
        }
    return costs


def build_cells(
    cell_rows: list[dict[str, str]], action_costs: Mapping[str, Mapping[str, float]]
) -> list[Cell]:
    cells: list[Cell] = []
    for row in cell_rows:
        action = row["action"]
        anchor = action_costs.get(action)
        if anchor is None or anchor["fractional_saving"] <= 0 or anchor["action_calls"] <= 0:
            continue
        count_field = "requested_quant_mean" if action == "Q" else "executed_cache_mean"
        action_calls = float(row[count_field])
        nominal_saving = (
            float(anchor["fractional_saving"])
            * action_calls
            / float(anchor["action_calls"])
        )
        cells.append(
            Cell(
                action=action,
                step_start=int(row["step_start"]),
                step_end=int(row["step_end"]),
                block_start=int(row["block_start"]),
                block_end=int(row["block_end"]),
                quality_loss=max(0.0, float(row["quality_loss_worst"])),
                saving=nominal_saving,
                branches=tuple(json.loads(row.get("branches") or "[0, 1]")),
                forecast_scale=float(row.get("forecast_scale") or 0.0),
            )
        )
    return sorted(cells, key=lambda cell: (cell.position, cell.action))


def beam_search(
    cells: list[Cell],
    *,
    quality_loss_budget: float,
    beam_width: int,
    max_cells: int,
) -> list[BeamState]:
    if quality_loss_budget < 0 or beam_width <= 0 or max_cells <= 0:
        raise ValueError("budget must be nonnegative and search sizes positive")
    beam = [BeamState((), 0.0, 0.0, -1)]
    all_states: dict[tuple[int, ...], BeamState] = {(): beam[0]}
    for _ in range(max_cells):
        expanded: list[BeamState] = []
        for state in beam:
            used_slots = frozenset().union(
                *(cells[index].slots for index in state.selected)
            )
            cache_scales = {
                cells[index].forecast_scale
                for index in state.selected
                if cells[index].action == "C"
            }
            for index in range(state.last_index + 1, len(cells)):
                cell = cells[index]
                if cell.slots & used_slots:
                    continue
                if cell.action == "C" and cache_scales and cell.forecast_scale not in cache_scales:
                    continue
                quality_loss = state.quality_loss + cell.quality_loss
                if quality_loss > quality_loss_budget:
                    continue
                candidate = BeamState(
                    selected=state.selected + (index,),
                    quality_loss=quality_loss,
                    saving=state.saving + cell.saving,
                    last_index=index,
                )
                expanded.append(candidate)
                all_states[candidate.selected] = candidate
        if not expanded:
            break
        expanded.sort(
            key=lambda state: (state.saving, -state.quality_loss, len(state.selected)),
            reverse=True,
        )
        beam = expanded[:beam_width]
    return sorted(
        (state for state in all_states.values() if state.selected),
        key=lambda state: (state.saving, -state.quality_loss, len(state.selected)),
        reverse=True,
    )


def state_row(index: int, state: BeamState, cells: list[Cell]) -> dict[str, object]:
    selected = [cells[cell_index] for cell_index in state.selected]
    predicted_speedup = (
        1.0 / (1.0 - state.saving) if state.saving < 1.0 else math.inf
    )
    return {
        "rank": index,
        "name": f"beam_{index:02d}",
        "cell_count": len(selected),
        "predicted_quality_loss_upper": state.quality_loss,
        "predicted_ssim_lower": 1.0 - state.quality_loss,
        "predicted_fractional_latency_saving": state.saving,
        "predicted_speedup": predicted_speedup,
        "selected_cells": ";".join(cell.label for cell in selected),
        "actions": "".join(cell.action for cell in selected),
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    analysis_dir = args.analysis_dir.resolve()
    source = json.loads(args.source_schedules.read_text(encoding="utf-8"))
    method_rows = read_csv(analysis_dir / "method_summary.csv")
    measured_cells = read_csv(analysis_dir / "tri_mode_cell_metrics.csv")
    action_costs = global_action_costs(method_rows)
    cells = build_cells(measured_cells, action_costs)
    states = beam_search(
        cells,
        quality_loss_budget=args.quality_loss_budget,
        beam_width=args.beam_width,
        max_cells=args.max_cells,
    )[: args.top_k]
    if not states:
        raise RuntimeError(
            "no positive-saving combination survived; inspect all-Q/all-C H200 anchors"
        )

    rows = [state_row(index, state, cells) for index, state in enumerate(states)]
    schedules: list[dict[str, object]] = [
        {
            "name": "dense",
            "default_action": "D",
            "metadata": {"family": "reference", "cell_count": 0},
        }
    ]
    for index, state in enumerate(states):
        selected = [cells[cell_index] for cell_index in state.selected]
        schedules.append(
            {
                "name": f"beam_{index:02d}",
                "default_action": "D",
                "forecast_scale": next(
                    (cell.forecast_scale for cell in selected if cell.action == "C"),
                    0.0,
                ),
                "overrides": [
                    {
                        "action": cell.action,
                        "step_start": cell.step_start,
                        "step_end": cell.step_end,
                        "block_start": cell.block_start,
                        "block_end": cell.block_end,
                        "branches": list(cell.branches),
                    }
                    for cell in selected
                ],
                "metadata": {
                    "family": "beam_surrogate_candidate",
                    "cell_count": len(selected),
                    "predicted_quality_loss_upper": state.quality_loss,
                    "predicted_fractional_latency_saving": state.saving,
                    "selected_cells": [cell.label for cell in selected],
                    "warning": "surrogate proposal; requires actual rollout validation",
                },
            }
        )
    payload = {
        "schema_version": 1,
        "sampling_steps": int(source["sampling_steps"]),
        "block_count": int(source["block_count"]),
        "source_schedule_file": str(args.source_schedules.resolve()),
        "search": {
            "quality_loss_budget": args.quality_loss_budget,
            "beam_width": args.beam_width,
            "max_cells": args.max_cells,
            "top_k": args.top_k,
            "action_global_cost_anchors": action_costs,
            "quality_model": "sum of worst single-cell (1-SSIM); conservative surrogate only",
            "latency_model": "all-action H200 anchor saving distributed over grouped cells",
        },
        "schedules": schedules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output.with_suffix(".csv"), rows)
    print(
        f"wrote {len(states)} rollout candidates to {args.output}; "
        f"usable actions={sorted(action_costs)}"
    )


if __name__ == "__main__":
    main()
