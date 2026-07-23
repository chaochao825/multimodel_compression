#!/usr/bin/env python3
"""Build grouped single-cell D/Q/C schedules for an Oracle screening run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--block-count", type=int, default=30)
    parser.add_argument("--step-group-size", type=int, default=4)
    parser.add_argument("--block-group-size", type=int, default=6)
    parser.add_argument("--probe-step-start", type=int, default=0)
    parser.add_argument("--probe-step-end", type=int)
    parser.add_argument("--probe-steps", default="")
    parser.add_argument("--probe-blocks", default="")
    parser.add_argument("--branches", default="0,1")
    parser.add_argument("--actions", default="Q,C")
    parser.add_argument("--forecast-scale", type=float, default=0.0)
    parser.add_argument("--forecast-scales", default="")
    parser.add_argument("--skip-global-anchors", action="store_true")
    return parser.parse_args()


def groups(size: int, group_size: int, *, start: int = 0) -> list[tuple[int, int]]:
    if size <= 0 or group_size <= 0 or start < 0 or start >= size:
        raise ValueError("sizes must be positive")
    return [
        (start, min(start + group_size, size) - 1)
        for start in range(start, size, group_size)
    ]


def scale_suffix(scale: float, *, always: bool) -> str:
    if not always and scale == 0.0:
        return ""
    sign = "m" if scale < 0 else ""
    return f"_f{sign}{round(abs(scale) * 100):03d}"


def main() -> None:
    args = parse_args()
    actions = [token.strip().upper() for token in args.actions.split(",") if token.strip()]
    if not actions or any(action not in {"Q", "C"} for action in actions):
        raise ValueError("--actions must contain Q and/or C")
    requested_forecast_scales = [
        float(token) for token in args.forecast_scales.split(",") if token.strip()
    ]
    if not requested_forecast_scales:
        requested_forecast_scales = [args.forecast_scale]
    requested_forecast_scales = sorted(set(requested_forecast_scales))
    explicit_steps = sorted(
        {int(token) for token in args.probe_steps.split(",") if token.strip()}
    )
    explicit_blocks = sorted(
        {int(token) for token in args.probe_blocks.split(",") if token.strip()}
    )
    branches = tuple(
        sorted({int(token) for token in args.branches.split(",") if token.strip()})
    )
    if not branches or any(branch not in (0, 1) for branch in branches):
        raise ValueError("--branches must contain 0 and/or 1")
    if explicit_steps:
        if any(step < 0 or step >= args.sampling_steps for step in explicit_steps):
            raise ValueError("explicit probe steps lie outside sampling steps")
        step_groups = [(step, step) for step in explicit_steps]
    else:
        probe_step_end = (
            args.sampling_steps - 1
            if args.probe_step_end is None
            else args.probe_step_end
        )
        if not 0 <= args.probe_step_start <= probe_step_end < args.sampling_steps:
            raise ValueError("probe step range must lie inside sampling steps")
        step_groups = [
            (first, min(first + args.step_group_size - 1, probe_step_end))
            for first in range(
                args.probe_step_start, probe_step_end + 1, args.step_group_size
            )
        ]
    if explicit_blocks:
        if any(block < 0 or block >= args.block_count for block in explicit_blocks):
            raise ValueError("explicit probe blocks lie outside model blocks")
        block_groups = [(block, block) for block in explicit_blocks]
    else:
        block_groups = groups(args.block_count, args.block_group_size)
    branch_suffix = "" if branches == (0, 1) else "_br" + "".join(map(str, branches))
    schedules: list[dict[str, object]] = [
        {
            "name": "dense",
            "default_action": "D",
            "metadata": {"family": "reference", "cell_count": 0},
        }
    ]
    if not args.skip_global_anchors:
        for action in actions:
            scales = requested_forecast_scales if action == "C" else [0.0]
            for forecast_scale in scales:
                suffix = scale_suffix(
                    forecast_scale, always=len(scales) > 1 or forecast_scale != 0.0
                )
                schedules.append(
                    {
                        "name": f"all_{action.lower()}{branch_suffix}{suffix}",
                        "default_action": action if branches == (0, 1) else "D",
                        "forecast_scale": forecast_scale if action == "C" else 0.0,
                        "overrides": (
                            []
                            if branches == (0, 1)
                            else [{"action": action, "branches": list(branches)}]
                        ),
                        "metadata": {
                            "family": "global_cost_anchor",
                            "action": action,
                            "cell_count": len(step_groups) * len(block_groups),
                            "branches": list(branches),
                            "forecast_scale": forecast_scale,
                        },
                    }
                )
    for action in actions:
        scales = requested_forecast_scales if action == "C" else [0.0]
        for forecast_scale in scales:
            suffix = scale_suffix(
                forecast_scale, always=len(scales) > 1 or forecast_scale != 0.0
            )
            for step_start, step_end in step_groups:
                for block_start, block_end in block_groups:
                    name = (
                        f"{action.lower()}_s{step_start:02d}-{step_end:02d}_"
                        f"b{block_start:02d}-{block_end:02d}{branch_suffix}{suffix}"
                    )
                    schedules.append(
                        {
                            "name": name,
                            "default_action": "D",
                            "forecast_scale": forecast_scale if action == "C" else 0.0,
                            "overrides": [
                                {
                                    "action": action,
                                    "step_start": step_start,
                                    "step_end": step_end,
                                    "block_start": block_start,
                                    "block_end": block_end,
                                    "branches": list(branches),
                                }
                            ],
                            "metadata": {
                                "family": "single_cell",
                                "action": action,
                                "cell_count": 1,
                                "step_start": step_start,
                                "step_end": step_end,
                                "block_start": block_start,
                                "block_end": block_end,
                                "branches": list(branches),
                                "forecast_scale": forecast_scale,
                            },
                        }
                    )
    payload = {
        "schema_version": 1,
        "sampling_steps": args.sampling_steps,
        "block_count": args.block_count,
        "step_groups": step_groups,
        "block_groups": block_groups,
        "actions": actions,
        "branches": list(branches),
        "forecast_scales": requested_forecast_scales,
        "schedules": schedules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(schedules)} schedules "
        f"({len(step_groups)} step groups x {len(block_groups)} block groups) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
