#!/usr/bin/env python3
"""Mutually exclusive dense, FP8, and cache actions for Wan block rollouts."""

from __future__ import annotations

import json
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


DENSE_ACTION = "D"
QUANT_ACTION = "Q"
CACHE_ACTION = "C"
VALID_ACTIONS = frozenset({DENSE_ACTION, QUANT_ACTION, CACHE_ACTION})


def _action(value: object, *, field: str) -> str:
    action = str(value).upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"{field} must be one of {sorted(VALID_ACTIONS)}")
    return action


def _inclusive_range(
    record: Mapping[str, object],
    *,
    plural: str,
    start: str,
    end: str,
    limit: int,
) -> tuple[int, ...]:
    if plural in record:
        raw = record[plural]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"{plural} must be a sequence")
        values = tuple(sorted({int(value) for value in raw}))
    elif start in record or end in record:
        first = int(record.get(start, 0))
        last = int(record.get(end, limit - 1))
        if last < first:
            raise ValueError(f"{end} must be at least {start}")
        values = tuple(range(first, last + 1))
    else:
        values = tuple(range(limit))
    if not values:
        raise ValueError(f"{plural} cannot be empty")
    if any(value < 0 or value >= limit for value in values):
        raise ValueError(f"{plural} must lie in [0, {limit - 1}]")
    return values


@dataclass(frozen=True)
class ActionOverride:
    action: str
    steps: tuple[int, ...]
    blocks: tuple[int, ...]
    branches: tuple[int, ...]


class TriModeSchedule:
    """Validated block-step-CFG action table.

    Overrides are explicit and conflicting overlaps are rejected. Cache and
    quantization therefore cannot be requested for the same block-step-branch.
    """

    def __init__(
        self,
        *,
        name: str,
        sampling_steps: int,
        block_count: int,
        default_action: str,
        overrides: Sequence[ActionOverride],
        forecast_scale: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("schedule name cannot be empty")
        if sampling_steps <= 0 or block_count <= 0:
            raise ValueError("sampling_steps and block_count must be positive")
        if not torch.isfinite(torch.tensor(float(forecast_scale))):
            raise ValueError("forecast_scale must be finite")
        self.name = name.strip()
        self.sampling_steps = int(sampling_steps)
        self.block_count = int(block_count)
        self.default_action = _action(default_action, field="default_action")
        self.overrides = tuple(overrides)
        self.forecast_scale = float(forecast_scale)
        self.metadata = dict(metadata or {})
        explicit: dict[tuple[int, int, int], str] = {}
        for override in self.overrides:
            action = _action(override.action, field="override action")
            for step in override.steps:
                for block in override.blocks:
                    for branch in override.branches:
                        key = (int(step), int(block), int(branch))
                        previous = explicit.get(key)
                        if previous is not None and previous != action:
                            raise ValueError(
                                "conflicting actions for "
                                f"step={step}, block={block}, branch={branch}: "
                                f"{previous} versus {action}"
                            )
                        explicit[key] = action
        self._explicit = explicit
        self.cache_blocks = frozenset(
            block
            for (step, block, branch), action in explicit.items()
            if action == CACHE_ACTION
        )
        if self.default_action == CACHE_ACTION:
            self.cache_blocks = frozenset(range(self.block_count))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        sampling_steps: int,
        block_count: int,
    ) -> "TriModeSchedule":
        raw_overrides = payload.get("overrides", [])
        if not isinstance(raw_overrides, Sequence) or isinstance(
            raw_overrides, (str, bytes)
        ):
            raise TypeError("overrides must be a sequence")
        overrides: list[ActionOverride] = []
        for index, raw in enumerate(raw_overrides):
            if not isinstance(raw, Mapping):
                raise TypeError(f"override {index} must be an object")
            steps = _inclusive_range(
                raw,
                plural="steps",
                start="step_start",
                end="step_end",
                limit=sampling_steps,
            )
            blocks = _inclusive_range(
                raw,
                plural="blocks",
                start="block_start",
                end="block_end",
                limit=block_count,
            )
            branches = _inclusive_range(
                raw,
                plural="branches",
                start="branch_start",
                end="branch_end",
                limit=2,
            )
            overrides.append(
                ActionOverride(
                    action=_action(raw.get("action"), field=f"override {index} action"),
                    steps=steps,
                    blocks=blocks,
                    branches=branches,
                )
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            name=str(payload.get("name", "")),
            sampling_steps=sampling_steps,
            block_count=block_count,
            default_action=_action(
                payload.get("default_action", DENSE_ACTION),
                field="default_action",
            ),
            overrides=overrides,
            forecast_scale=float(payload.get("forecast_scale", 0.0)),
            metadata=metadata,
        )

    def action(self, step: int, block: int, branch: int) -> str:
        if not 0 <= step < self.sampling_steps:
            raise IndexError(f"step {step} lies outside this schedule")
        if not 0 <= block < self.block_count:
            raise IndexError(f"block {block} lies outside this schedule")
        if branch not in (0, 1):
            raise IndexError(f"CFG branch must be 0 or 1, got {branch}")
        return self._explicit.get((step, block, branch), self.default_action)

    def action_counts(self) -> dict[str, int]:
        counts = {action: 0 for action in sorted(VALID_ACTIONS)}
        for step in range(self.sampling_steps):
            for block in range(self.block_count):
                for branch in range(2):
                    counts[self.action(step, block, branch)] += 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sampling_steps": self.sampling_steps,
            "block_count": self.block_count,
            "default_action": self.default_action,
            "forecast_scale": self.forecast_scale,
            "overrides": [asdict(override) for override in self.overrides],
            "metadata": self.metadata,
            "action_counts": self.action_counts(),
        }


def load_schedule_bundle(
    path: Path,
    *,
    sampling_steps: int,
    block_count: int,
) -> list[TriModeSchedule]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("schedule bundle must be a JSON object")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schedule bundle schema_version")
    raw_schedules = payload.get("schedules")
    if not isinstance(raw_schedules, Sequence) or isinstance(
        raw_schedules, (str, bytes)
    ):
        raise TypeError("schedule bundle requires a schedules list")
    schedules = [
        TriModeSchedule.from_dict(
            raw,
            sampling_steps=sampling_steps,
            block_count=block_count,
        )
        for raw in raw_schedules
        if isinstance(raw, Mapping)
    ]
    if len(schedules) != len(raw_schedules):
        raise TypeError("every schedule must be an object")
    names = [schedule.name for schedule in schedules]
    if len(set(names)) != len(names):
        raise ValueError("schedule names must be unique")
    if not schedules:
        raise ValueError("schedule bundle cannot be empty")
    return schedules


@dataclass
class _CacheState:
    latest: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    age: int = 0


@dataclass(frozen=True)
class ActionEvent:
    call_index: int
    step: int
    block: int
    branch: int
    requested_action: str
    executed_action: str
    cache_age_before: int
    cache_age_after: int
    cache_refreshed: bool
    fallback_reason: str


class TriModeBlockController:
    """Install a mutually exclusive tri-mode state machine on Wan blocks."""

    def __init__(
        self,
        model: torch.nn.Module,
        linear_controller: object,
        *,
        sampling_steps: int,
    ) -> None:
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if not hasattr(model, "blocks") or not len(model.blocks):
            raise ValueError("tri-mode execution requires model.blocks")
        required = set(range(len(model.blocks)))
        available = set(getattr(linear_controller, "block_indices", ()))
        if available != required:
            raise ValueError(
                "tri-mode Q actions require injected linears in every block; "
                f"missing={sorted(required - available)}, extra={sorted(available - required)}"
            )
        self.model = model
        self.linear_controller = linear_controller
        self.sampling_steps = int(sampling_steps)
        self.block_count = len(model.blocks)
        self.original_model_forward = model.forward
        self.original_block_forwards = [block.forward for block in model.blocks]
        self.schedule: TriModeSchedule | None = None
        self.call_index = 0
        self.current_call = -1
        self.current_step = -1
        self.current_branch = -1
        self._cache: dict[tuple[int, int], _CacheState] = {}
        self.events: list[ActionEvent] = []
        self._install()

    def _install(self) -> None:
        def model_wrapped(
            model_self: torch.nn.Module,
            *args: object,
            **kwargs: object,
        ) -> object:
            return self._model_forward(*args, **kwargs)

        self.model.forward = types.MethodType(model_wrapped, self.model)
        for block_index, (block, original) in enumerate(
            zip(self.model.blocks, self.original_block_forwards, strict=True)
        ):
            def block_wrapped(
                block_self: torch.nn.Module,
                x: torch.Tensor,
                *args: object,
                _block_index: int = block_index,
                _original: object = original,
                **kwargs: object,
            ) -> torch.Tensor:
                return self._block_forward(
                    _block_index,
                    _original,
                    block_self,
                    x,
                    *args,
                    **kwargs,
                )

            block.forward = types.MethodType(block_wrapped, block)

    def begin(self, schedule: TriModeSchedule) -> None:
        if schedule.sampling_steps != self.sampling_steps:
            raise ValueError("schedule sampling_steps does not match controller")
        if schedule.block_count != self.block_count:
            raise ValueError("schedule block_count does not match controller")
        self.schedule = schedule
        self.call_index = 0
        self.current_call = -1
        self.current_step = -1
        self.current_branch = -1
        self._cache.clear()
        self.events.clear()
        self.linear_controller.set_mode("dense")
        if hasattr(self.linear_controller, "reset_runtime_stats"):
            self.linear_controller.reset_runtime_stats()

    def _model_forward(self, *args: object, **kwargs: object) -> object:
        if self.schedule is None:
            raise RuntimeError("tri-mode controller must begin with a schedule")
        expected_calls = self.sampling_steps * 2
        if self.call_index >= expected_calls:
            raise RuntimeError(
                f"model received more than the expected {expected_calls} CFG calls"
            )
        self.current_call = self.call_index
        self.current_step = self.call_index // 2
        self.current_branch = self.call_index % 2
        self.call_index += 1
        try:
            return self.original_model_forward(*args, **kwargs)
        finally:
            self.current_call = -1
            self.current_step = -1
            self.current_branch = -1

    def _execute_recompute(
        self,
        *,
        block_index: int,
        action: str,
        original: object,
        block_self: torch.nn.Module,
        x: torch.Tensor,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        mode = "dense" if action == DENSE_ACTION else "fp8"
        self.linear_controller.set_block_mode(block_index, mode)
        output = original(x, *args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wan block forward must return a tensor")
        return output

    def _update_cache(
        self,
        *,
        block_index: int,
        branch: int,
        x: torch.Tensor,
        output: torch.Tensor,
        force: bool = False,
    ) -> bool:
        assert self.schedule is not None
        if block_index not in self.schedule.cache_blocks:
            return False
        lookahead = 2 if self.schedule.forecast_scale else 1
        needed = force or any(
            self.schedule.action(next_step, block_index, branch) == CACHE_ACTION
            for next_step in range(
                self.current_step + 1,
                min(self.current_step + lookahead + 1, self.sampling_steps),
            )
        )
        if not needed:
            return False
        state = self._cache.setdefault((block_index, branch), _CacheState())
        residual = (output - x).detach()
        state.previous = state.latest
        state.latest = residual
        state.age = 0
        return True

    def _block_forward(
        self,
        block_index: int,
        original: object,
        block_self: torch.nn.Module,
        x: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.schedule is None or self.current_step < 0:
            raise RuntimeError("block executed outside a scheduled model forward")
        requested = self.schedule.action(
            self.current_step,
            block_index,
            self.current_branch,
        )
        state = self._cache.get((block_index, self.current_branch))
        age_before = state.age if state is not None else 0
        fallback_reason = ""
        executed = requested
        cache_refreshed = False

        if requested == CACHE_ACTION:
            valid = (
                state is not None
                and state.latest is not None
                and state.latest.shape == x.shape
            )
            if valid:
                assert state is not None and state.latest is not None
                residual = state.latest
                if state.previous is not None and self.schedule.forecast_scale:
                    residual = residual + self.schedule.forecast_scale * (
                        residual - state.previous
                    )
                output = x + residual.to(device=x.device, dtype=x.dtype)
                state.age += 1
            else:
                executed = DENSE_ACTION
                fallback_reason = "cache-uninitialized-or-shape-mismatch"
                output = self._execute_recompute(
                    block_index=block_index,
                    action=executed,
                    original=original,
                    block_self=block_self,
                    x=x,
                    args=args,
                    kwargs=kwargs,
                )
                cache_refreshed = self._update_cache(
                    block_index=block_index,
                    branch=self.current_branch,
                    x=x,
                    output=output,
                    force=True,
                )
        else:
            output = self._execute_recompute(
                block_index=block_index,
                action=requested,
                original=original,
                block_self=block_self,
                x=x,
                args=args,
                kwargs=kwargs,
            )
            cache_refreshed = self._update_cache(
                block_index=block_index,
                branch=self.current_branch,
                x=x,
                output=output,
            )

        final_state = self._cache.get((block_index, self.current_branch))
        self.events.append(
            ActionEvent(
                call_index=self.current_call,
                step=self.current_step,
                block=block_index,
                branch=self.current_branch,
                requested_action=requested,
                executed_action=executed,
                cache_age_before=age_before,
                cache_age_after=final_state.age if final_state is not None else 0,
                cache_refreshed=cache_refreshed,
                fallback_reason=fallback_reason,
            )
        )
        return output

    def stats(self) -> dict[str, object]:
        requested = {action: 0 for action in sorted(VALID_ACTIONS)}
        executed = {action: 0 for action in sorted(VALID_ACTIONS)}
        for event in self.events:
            requested[event.requested_action] += 1
            executed[event.executed_action] += 1
        return {
            "tri_mode_schedule": self.schedule.name if self.schedule else "",
            "tri_mode_model_calls": self.call_index,
            "tri_mode_block_calls": len(self.events),
            "tri_mode_requested_dense": requested[DENSE_ACTION],
            "tri_mode_requested_quant": requested[QUANT_ACTION],
            "tri_mode_requested_cache": requested[CACHE_ACTION],
            "tri_mode_executed_dense": executed[DENSE_ACTION],
            "tri_mode_executed_quant": executed[QUANT_ACTION],
            "tri_mode_executed_cache": executed[CACHE_ACTION],
            "tri_mode_cache_fallbacks": sum(
                bool(event.fallback_reason) for event in self.events
            ),
            "tri_mode_cache_refreshes": sum(event.cache_refreshed for event in self.events),
            "tri_mode_max_cache_age": max(
                (event.cache_age_after for event in self.events), default=0
            ),
        }

    def assert_complete(self) -> None:
        expected_model_calls = self.sampling_steps * 2
        expected_block_calls = expected_model_calls * self.block_count
        if self.call_index != expected_model_calls:
            raise RuntimeError(
                "scheduled rollout did not execute the expected CFG calls: "
                f"observed={self.call_index}, expected={expected_model_calls}"
            )
        if len(self.events) != expected_block_calls:
            raise RuntimeError(
                "scheduled rollout did not execute the expected block calls: "
                f"observed={len(self.events)}, expected={expected_block_calls}"
            )

    def write_events(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def restore(self) -> None:
        for block, original in zip(
            self.model.blocks, self.original_block_forwards, strict=True
        ):
            block.forward = original
        self.model.forward = self.original_model_forward


__all__ = [
    "ActionEvent",
    "ActionOverride",
    "CACHE_ACTION",
    "DENSE_ACTION",
    "QUANT_ACTION",
    "TriModeBlockController",
    "TriModeSchedule",
    "VALID_ACTIONS",
    "load_schedule_bundle",
]
