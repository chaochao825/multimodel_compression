#!/usr/bin/env python3
"""Capture EXP-045 Wan observables while returning unchanged dense outputs."""

from __future__ import annotations

import types
from typing import Iterable

import torch
import torch.nn.functional as functional

from current_input_observability_core import CellTrajectory


class WanCurrentInputObservabilityRecorder:
    """Keep one identity in host memory and expose layer/branch trajectories."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sampling_steps: int,
        capture_steps: Iterable[int],
        blocks: Iterable[int],
        branches: Iterable[int],
        qk_rows: int,
    ) -> None:
        if sampling_steps <= 0 or qk_rows <= 0:
            raise ValueError("sampling_steps and qk_rows must be positive")
        if not hasattr(model, "blocks") or not len(model.blocks):
            raise ValueError("Wan observability capture requires model.blocks")
        self.model = model
        self.sampling_steps = int(sampling_steps)
        self.capture_steps = tuple(sorted(set(int(value) for value in capture_steps)))
        self.blocks = tuple(sorted(set(int(value) for value in blocks)))
        self.branches = tuple(sorted(set(int(value) for value in branches)))
        self.qk_rows = int(qk_rows)
        if not self.capture_steps or any(
            value < 0 or value >= self.sampling_steps for value in self.capture_steps
        ):
            raise ValueError("capture_steps lie outside the sampler")
        if not self.blocks or any(
            value < 0 or value >= len(model.blocks) for value in self.blocks
        ):
            raise ValueError("capture blocks lie outside the model")
        if not self.branches or any(value not in (0, 1) for value in self.branches):
            raise ValueError("branches must contain 0 and/or 1")

        self.original_model_forward = model.forward
        self.original_block_forwards = {
            block: model.blocks[block].forward for block in self.blocks
        }
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1
        self.run_id = ""
        self.run_metadata: dict[str, object] = {}
        self.timesteps: tuple[float, ...] = ()
        self.grid_size: tuple[int, int, int] | None = None
        self.records: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        self._install()

    def _install(self) -> None:
        def model_wrapped(
            model_self: torch.nn.Module, *args: object, **kwargs: object
        ) -> object:
            return self._model_forward(*args, **kwargs)

        self.model.forward = types.MethodType(model_wrapped, self.model)
        for block_index in self.blocks:
            block = self.model.blocks[block_index]
            original = self.original_block_forwards[block_index]

            def block_wrapped(
                block_self: torch.nn.Module,
                x: torch.Tensor,
                *args: object,
                _block_index: int = block_index,
                _original: object = original,
                **kwargs: object,
            ) -> torch.Tensor:
                return self._block_forward(
                    _block_index, block_self, _original, x, *args, **kwargs
                )

            block.forward = types.MethodType(block_wrapped, block)

    def begin_run(
        self,
        run_id: str,
        timesteps: Iterable[float],
        metadata: dict[str, object],
    ) -> None:
        if self.run_id:
            raise RuntimeError("the previous observability run must end first")
        timestep_values = tuple(float(value) for value in timesteps)
        if len(timestep_values) != self.sampling_steps:
            raise ValueError("timestep count does not match sampling_steps")
        if not run_id:
            raise ValueError("run_id cannot be empty")
        self.run_id = run_id
        self.run_metadata = dict(metadata)
        self.timesteps = timestep_values
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1
        self.grid_size = None
        self.records.clear()

    def _model_forward(self, *args: object, **kwargs: object) -> object:
        if not self.run_id:
            raise RuntimeError("begin_run must be called before model execution")
        expected_calls = self.sampling_steps * 2
        if self.call_index >= expected_calls:
            raise RuntimeError("model received more CFG calls than registered")
        self.current_step = self.call_index // 2
        self.current_branch = self.call_index % 2
        self.call_index += 1
        try:
            return self.original_model_forward(*args, **kwargs)
        finally:
            self.current_step = -1
            self.current_branch = -1

    def _projection_indices(self, feature_count: int, device: torch.device) -> torch.Tensor:
        if self.qk_rows > feature_count:
            raise ValueError("qk_rows exceeds the projection output width")
        return torch.linspace(
            0,
            feature_count - 1,
            self.qk_rows,
            device=device,
            dtype=torch.float64,
        ).round().long()

    @staticmethod
    def _host_half(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().squeeze(0).to(device="cpu", dtype=torch.float16)

    def _qk_sketch(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        attention_input = block.norm1(x) * (1 + modulation[:, 1]) + modulation[:, 0]
        q_projection = block.self_attn.q
        k_projection = block.self_attn.k
        if not isinstance(q_projection, torch.nn.Linear) or not isinstance(
            k_projection, torch.nn.Linear
        ):
            raise TypeError("fixed-row sketch requires linear Wan Q/K projections")
        indices = self._projection_indices(q_projection.out_features, x.device)
        q_bias = (
            None
            if q_projection.bias is None
            else q_projection.bias.index_select(0, indices)
        )
        k_bias = (
            None
            if k_projection.bias is None
            else k_projection.bias.index_select(0, indices)
        )
        q_sketch = functional.linear(
            attention_input,
            q_projection.weight.index_select(0, indices),
            q_bias,
        )
        k_sketch = functional.linear(
            attention_input,
            k_projection.weight.index_select(0, indices),
            k_bias,
        )
        return torch.cat((q_sketch, k_sketch), dim=-1)

    def _block_forward(
        self,
        block_index: int,
        block: torch.nn.Module,
        original: object,
        x: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.current_step < 0:
            raise RuntimeError("selected block executed outside an active model call")
        should_capture = (
            self.current_step in self.capture_steps
            and self.current_branch in self.branches
        )
        if not should_capture:
            return original(x, *args, **kwargs)
        e = kwargs["e"]
        grid_size = kwargs["grid_sizes"]
        if not isinstance(e, torch.Tensor):
            raise TypeError("Wan timestep conditioning must be a tensor")
        current_grid = tuple(int(value) for value in grid_size)
        if self.grid_size is None:
            self.grid_size = current_grid
        elif self.grid_size != current_grid:
            raise RuntimeError("Wan token geometry changed inside one trajectory")
        modulation = block.modulation + e
        qk_sketch = self._qk_sketch(block, x, modulation)
        output = original(x, *args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wan block forward must return a tensor")
        key = (block_index, self.current_step, self.current_branch)
        if key in self.records:
            raise RuntimeError(f"duplicate observability cell: {key}")
        self.records[key] = {
            "block_input": self._host_half(x),
            "residual": self._host_half(output - x),
            "adaln": modulation.detach().squeeze(0).to(
                device="cpu", dtype=torch.float32
            ),
            "qk_sketch": self._host_half(qk_sketch),
        }
        return output

    def assert_complete(self) -> None:
        expected_calls = self.sampling_steps * 2
        if self.call_index != expected_calls:
            raise RuntimeError(
                f"captured {self.call_index} CFG calls, expected {expected_calls}"
            )
        expected_keys = {
            (block, step, branch)
            for block in self.blocks
            for step in self.capture_steps
            for branch in self.branches
        }
        missing = expected_keys.difference(self.records)
        extra = set(self.records).difference(expected_keys)
        if missing or extra:
            raise RuntimeError(
                f"observability cells mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

    def cell_trajectory(
        self,
        block: int,
        branch: int,
        *,
        device: torch.device,
    ) -> CellTrajectory:
        self.assert_complete()
        if self.grid_size is None:
            raise RuntimeError("capture did not observe a Wan token grid")
        ordered = [self.records[(block, step, branch)] for step in self.capture_steps]
        expected_steps = tuple(range(self.capture_steps[-1] + 1))
        if self.capture_steps != expected_steps:
            raise ValueError("cell_trajectory currently requires contiguous steps from zero")
        trajectory = CellTrajectory(
            block_input=torch.stack(
                [record["block_input"] for record in ordered]
            ).to(device=device, dtype=torch.float32),
            residual=torch.stack([record["residual"] for record in ordered]).to(
                device=device, dtype=torch.float32
            ),
            adaln=torch.stack([record["adaln"] for record in ordered]).to(
                device=device, dtype=torch.float32
            ),
            qk_sketch=torch.stack(
                [record["qk_sketch"] for record in ordered]
            ).to(device=device, dtype=torch.float32),
            thw=self.grid_size,
        )
        trajectory.validate()
        return trajectory

    def estimated_host_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for record in self.records.values()
            for tensor in record.values()
        )

    def end_run(self) -> dict[str, object]:
        self.assert_complete()
        summary = {
            "run_id": self.run_id,
            "metadata": self.run_metadata,
            "timesteps": list(self.timesteps),
            "grid_size": list(self.grid_size or ()),
            "record_count": len(self.records),
            "estimated_host_bytes": self.estimated_host_bytes(),
            "qk_rows_per_projection": self.qk_rows,
        }
        self.run_id = ""
        self.run_metadata = {}
        self.timesteps = ()
        self.current_step = -1
        self.current_branch = -1
        return summary

    def clear(self) -> None:
        self.records.clear()
        self.grid_size = None

    def restore(self) -> None:
        self.model.forward = self.original_model_forward
        for block_index, original in self.original_block_forwards.items():
            self.model.blocks[block_index].forward = original
