#!/usr/bin/env python3
"""Dense-trajectory counterfactual collection for Q/cache activation defects."""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class _ResidualHistory:
    latest: torch.Tensor | None = None
    previous: torch.Tensor | None = None


class ActivationDefectController:
    """Collect counterfactual defects while returning the exact dense output."""

    def __init__(
        self,
        model: torch.nn.Module,
        linear_controller: object,
        *,
        sampling_steps: int,
        steps: Iterable[int],
        blocks: Iterable[int],
        branches: Iterable[int] = (0, 1),
        forecast_scales: Iterable[float] = (0.0, 0.5, 1.0),
        sample_rows: int = 256,
        attention_dispatcher: object | None = None,
    ) -> None:
        if sampling_steps <= 0 or sample_rows <= 0:
            raise ValueError("sampling_steps and sample_rows must be positive")
        if not hasattr(model, "blocks") or not len(model.blocks):
            raise ValueError("activation defect probing requires model.blocks")
        self.model = model
        self.linear_controller = linear_controller
        self.sampling_steps = int(sampling_steps)
        self.block_count = len(model.blocks)
        self.steps = frozenset(int(step) for step in steps)
        self.blocks = frozenset(int(block) for block in blocks)
        self.branches = frozenset(int(branch) for branch in branches)
        self.forecast_scales = tuple(sorted(set(float(value) for value in forecast_scales)))
        self.sample_rows = int(sample_rows)
        self.attention_dispatcher = attention_dispatcher
        if not self.steps or any(step < 0 or step >= self.sampling_steps for step in self.steps):
            raise ValueError("probe steps lie outside the trajectory")
        if not self.blocks or any(block < 0 or block >= self.block_count for block in self.blocks):
            raise ValueError("probe blocks lie outside the model")
        if not self.branches or any(branch not in (0, 1) for branch in self.branches):
            raise ValueError("probe branches must contain 0 and/or 1")
        available = set(getattr(linear_controller, "block_indices", ()))
        if available != set(range(self.block_count)):
            raise ValueError("Q counterfactuals require injected linears in every block")

        self.original_model_forward = model.forward
        self.original_block_forwards = [block.forward for block in model.blocks]
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1
        self.run_id = ""
        self._history: dict[tuple[int, int], _ResidualHistory] = {}
        self.records: list[dict[str, object]] = []
        self._install()

    def _install(self) -> None:
        def model_wrapped(
            model_self: torch.nn.Module, *args: object, **kwargs: object
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
                    _block_index, _original, x, *args, **kwargs
                )

            block.forward = types.MethodType(block_wrapped, block)

    def begin_run(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        self.run_id = run_id
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1
        self._history.clear()
        self.linear_controller.set_mode("dense")

    def _model_forward(self, *args: object, **kwargs: object) -> object:
        if not self.run_id:
            raise RuntimeError("begin_run must be called before model execution")
        expected = self.sampling_steps * 2
        if self.call_index >= expected:
            raise RuntimeError(f"more than {expected} model calls in audit rollout")
        self.current_step = self.call_index // 2
        self.current_branch = self.call_index % 2
        self.call_index += 1
        try:
            return self.original_model_forward(*args, **kwargs)
        finally:
            self.current_step = -1
            self.current_branch = -1

    def _dispatcher_counts(self) -> tuple[int, int] | None:
        if self.attention_dispatcher is None:
            return None
        return (
            int(getattr(self.attention_dispatcher, "self_calls")),
            int(getattr(self.attention_dispatcher, "cross_calls")),
        )

    def _restore_dispatcher_counts(self, counts: tuple[int, int] | None) -> None:
        if counts is None or self.attention_dispatcher is None:
            return
        self.attention_dispatcher.self_calls = counts[0]
        self.attention_dispatcher.cross_calls = counts[1]

    def _capture(
        self,
        *,
        kind: str,
        block: int,
        defect: torch.Tensor,
        forecast_scale: float = 0.0,
    ) -> None:
        flat = defect.detach().float().reshape(-1, defect.shape[-1])
        if flat.shape[0] > self.sample_rows:
            indices = torch.linspace(
                0,
                flat.shape[0] - 1,
                self.sample_rows,
                device=flat.device,
                dtype=torch.float64,
            ).round().long()
            flat = flat.index_select(0, indices)
        sample = flat.to(device="cpu", dtype=torch.float16)
        self.records.append(
            {
                "run_id": self.run_id,
                "kind": kind,
                "step": self.current_step,
                "block": block,
                "branch": self.current_branch,
                "forecast_scale": forecast_scale,
                "rows": sample.shape[0],
                "features": sample.shape[1],
                "full_shape": tuple(defect.shape),
                "sample_energy": float(sample.float().square().sum().item()),
                "sample": sample,
            }
        )

    def _block_forward(
        self,
        block_index: int,
        original: object,
        x: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.current_step < 0:
            raise RuntimeError("block executed outside an audit model forward")
        selected = (
            self.current_step in self.steps
            and block_index in self.blocks
            and self.current_branch in self.branches
        )
        self.linear_controller.set_block_mode(block_index, "dense")
        dense = original(x, *args, **kwargs)
        if not isinstance(dense, torch.Tensor):
            raise TypeError("Wan block forward must return a tensor")

        history = self._history.get((block_index, self.current_branch))
        if selected and history is not None and history.latest is not None:
            stale = x + history.latest.to(device=x.device, dtype=x.dtype)
            self._capture(kind="C", block=block_index, defect=dense - stale)
            if history.previous is not None:
                delta = history.latest - history.previous
                for scale in self.forecast_scales:
                    if scale == 0.0:
                        continue
                    forecast = x + (history.latest + scale * delta).to(
                        device=x.device, dtype=x.dtype
                    )
                    self._capture(
                        kind="C_FORECAST",
                        block=block_index,
                        defect=dense - forecast,
                        forecast_scale=scale,
                    )

        if selected:
            counts = self._dispatcher_counts()
            self.linear_controller.set_block_mode(block_index, "fp8")
            quantized = original(x, *args, **kwargs)
            self.linear_controller.set_block_mode(block_index, "dense")
            self._restore_dispatcher_counts(counts)
            if not isinstance(quantized, torch.Tensor):
                raise TypeError("Wan block counterfactual must return a tensor")
            self._capture(kind="Q", block=block_index, defect=dense - quantized)

        if block_index in self.blocks:
            state = self._history.setdefault(
                (block_index, self.current_branch), _ResidualHistory()
            )
            state.previous = state.latest
            state.latest = (dense - x).detach()
        return dense

    def assert_complete(self) -> None:
        expected = self.sampling_steps * 2
        if self.call_index != expected:
            raise RuntimeError(
                f"audit rollout observed {self.call_index} model calls, expected {expected}"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sampling_steps": self.sampling_steps,
                "block_count": self.block_count,
                "steps": sorted(self.steps),
                "blocks": sorted(self.blocks),
                "branches": sorted(self.branches),
                "forecast_scales": list(self.forecast_scales),
                "sample_rows": self.sample_rows,
                "records": self.records,
            },
            path,
        )

    def restore(self) -> None:
        for block, original in zip(
            self.model.blocks, self.original_block_forwards, strict=True
        ):
            block.forward = original
        self.model.forward = self.original_model_forward


__all__ = ["ActivationDefectController"]
