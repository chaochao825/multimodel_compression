#!/usr/bin/env python3
"""Non-invasive block capture for the official Wan network used by EXP-048."""

from __future__ import annotations

import types
from typing import Iterable

import torch

from rcm_state_closure_core import ClosureTrajectory


class WanBlockSequenceRecorder:
    """Capture exact block inputs/residuals while returning unchanged outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        blocks: Iterable[int],
        expected_steps: int,
        sampled_tokens: int | None,
    ) -> None:
        if not hasattr(model, "blocks") or not len(model.blocks):
            raise ValueError("Wan block capture requires model.blocks")
        self.model = model
        self.blocks = tuple(sorted(set(int(value) for value in blocks)))
        self.expected_steps = int(expected_steps)
        self.sampled_tokens = sampled_tokens
        if self.expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if not self.blocks or any(
            value < 0 or value >= len(model.blocks) for value in self.blocks
        ):
            raise ValueError("capture blocks lie outside the Wan model")
        if sampled_tokens is not None and sampled_tokens <= 0:
            raise ValueError("sampled_tokens must be positive when provided")
        self.original_forwards = {
            block: model.blocks[block].forward for block in self.blocks
        }
        self.active = False
        self.records: dict[int, dict[str, list[torch.Tensor]]] = {}
        self._install()

    def _install(self) -> None:
        for block_index in self.blocks:
            block = self.model.blocks[block_index]
            original = self.original_forwards[block_index]

            def wrapped(
                block_self: torch.nn.Module,
                x: torch.Tensor,
                *args: object,
                _block_index: int = block_index,
                _original: object = original,
                **kwargs: object,
            ) -> torch.Tensor:
                return self._forward(
                    _block_index, block_self, _original, x, *args, **kwargs
                )

            block.forward = types.MethodType(wrapped, block)

    def begin(self) -> None:
        if self.active:
            raise RuntimeError("a capture sequence is already active")
        self.records = {
            block: {"block_input": [], "residual": []} for block in self.blocks
        }
        self.active = True

    def _select_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sampled_tokens is None:
            return tensor
        token_count = int(tensor.shape[1])
        if self.sampled_tokens > token_count:
            raise ValueError("sampled_tokens exceeds the Wan token count")
        indices = torch.linspace(
            0,
            token_count - 1,
            self.sampled_tokens,
            device=tensor.device,
            dtype=torch.float64,
        ).round().long()
        return tensor.index_select(1, indices)

    @staticmethod
    def _host_bfloat16(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().squeeze(0).to(device="cpu", dtype=torch.bfloat16)

    def _forward(
        self,
        block_index: int,
        block: torch.nn.Module,
        original: object,
        x: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        output = original(x, *args, **kwargs)
        if not self.active:
            return output
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wan block forward must return a tensor")
        self.records[block_index]["block_input"].append(
            self._host_bfloat16(self._select_tokens(x))
        )
        self.records[block_index]["residual"].append(
            self._host_bfloat16(self._select_tokens(output - x))
        )
        return output

    def end(self) -> dict[int, ClosureTrajectory]:
        if not self.active:
            raise RuntimeError("no capture sequence is active")
        self.active = False
        output: dict[int, ClosureTrajectory] = {}
        for block in self.blocks:
            record = self.records[block]
            if len(record["block_input"]) != self.expected_steps:
                raise RuntimeError(
                    f"block {block} captured {len(record['block_input'])} steps, "
                    f"expected {self.expected_steps}"
                )
            trajectory = ClosureTrajectory(
                block_input=torch.stack(record["block_input"]),
                residual=torch.stack(record["residual"]),
            )
            trajectory.validate()
            output[block] = trajectory
        return output

    def restore(self) -> None:
        self.active = False
        for block, original in self.original_forwards.items():
            self.model.blocks[block].forward = original
