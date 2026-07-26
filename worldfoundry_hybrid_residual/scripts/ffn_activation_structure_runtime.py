#!/usr/bin/env python3
"""Capture sampled Wan FFN activations without changing the dense trajectory."""

from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Iterable

import torch


class FFNActivationStructureController:
    """Collect FFN samples and selected THW spectra from an exact dense run."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sampling_steps: int,
        frame_num: int,
        grid_height: int,
        grid_width: int,
        blocks: Iterable[int],
        branches: Iterable[int] = (0, 1),
        spectral_steps: Iterable[int] = (0, 5, 10, 15, 19),
        sample_rows: int = 16,
        spectral_channels: int = 32,
        spectral_density: float = 0.125,
        seed: int = 20260726,
    ) -> None:
        if sampling_steps <= 0 or sample_rows <= 0 or spectral_channels <= 0:
            raise ValueError("sampling_steps, sample_rows, and spectral_channels must be positive")
        if not 0.0 < spectral_density <= 1.0:
            raise ValueError("spectral_density must lie in (0, 1]")
        if not hasattr(model, "blocks") or not len(model.blocks):
            raise ValueError("Wan activation probing requires model.blocks")
        self.model = model
        self.sampling_steps = int(sampling_steps)
        self.frame_num = int(frame_num)
        self.grid = ((frame_num - 1) // 4 + 1, int(grid_height), int(grid_width))
        self.blocks = frozenset(int(block) for block in blocks)
        self.branches = frozenset(int(branch) for branch in branches)
        self.spectral_steps = frozenset(int(step) for step in spectral_steps)
        self.sample_rows = int(sample_rows)
        self.spectral_channels = int(spectral_channels)
        self.spectral_density = float(spectral_density)
        self.seed = int(seed)
        if any(block < 0 or block >= len(model.blocks) for block in self.blocks):
            raise ValueError("probe block lies outside the model")
        if any(branch not in (0, 1) for branch in self.branches):
            raise ValueError("branches must contain only 0 and/or 1")
        if any(step < 0 or step >= sampling_steps for step in self.spectral_steps):
            raise ValueError("spectral step lies outside the trajectory")

        self.original_model_forward = model.forward
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1
        self.run_id = ""
        self.records: list[dict[str, object]] = []
        self.spectrum_records: list[dict[str, object]] = []
        self._mask_cache: dict[tuple[torch.device, tuple[int, int, int]], torch.Tensor] = {}
        self._install()

    def _install(self) -> None:
        def model_wrapped(model_self: torch.nn.Module, *args: object, **kwargs: object) -> object:
            return self._model_forward(*args, **kwargs)

        self.model.forward = types.MethodType(model_wrapped, self.model)
        for block_index in sorted(self.blocks):
            ffn = self.model.blocks[block_index].ffn

            def up_pre_hook(
                module: torch.nn.Module,
                inputs: tuple[object, ...],
                _block: int = block_index,
            ) -> None:
                self._hook_tensor(_block, "ffn_input", inputs[0])

            def down_pre_hook(
                module: torch.nn.Module,
                inputs: tuple[object, ...],
                _block: int = block_index,
            ) -> None:
                self._hook_tensor(_block, "ffn_hidden_post_gelu", inputs[0])

            def down_hook(
                module: torch.nn.Module,
                inputs: tuple[object, ...],
                output: object,
                _block: int = block_index,
            ) -> None:
                self._hook_tensor(_block, "ffn_output", output)

            self.hooks.append(ffn[0].register_forward_pre_hook(up_pre_hook))
            self.hooks.append(ffn[2].register_forward_pre_hook(down_pre_hook))
            self.hooks.append(ffn[2].register_forward_hook(down_hook))

    def begin_run(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        self.run_id = run_id
        self.call_index = 0
        self.current_step = -1
        self.current_branch = -1

    def _model_forward(self, *args: object, **kwargs: object) -> object:
        if not self.run_id:
            raise RuntimeError("begin_run must be called before model execution")
        expected = self.sampling_steps * 2
        if self.call_index >= expected:
            raise RuntimeError(f"more than {expected} model calls in activation audit")
        self.current_step = self.call_index // 2
        self.current_branch = self.call_index % 2
        self.call_index += 1
        try:
            return self.original_model_forward(*args, **kwargs)
        finally:
            self.current_step = -1
            self.current_branch = -1

    def _selected(self, block: int) -> bool:
        return (
            self.current_step >= 0
            and block in self.blocks
            and self.current_branch in self.branches
        )

    def _hook_tensor(self, block: int, signal: str, value: object) -> None:
        if not self._selected(block):
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{signal} hook expected a tensor")
        flat = value.detach().reshape(-1, value.shape[-1])
        expected_tokens = math.prod(self.grid)
        if flat.shape[0] != expected_tokens:
            raise ValueError(
                f"{signal} token count {flat.shape[0]} does not match grid {self.grid}"
            )
        indices = torch.linspace(
            0, flat.shape[0] - 1, self.sample_rows, device=flat.device, dtype=torch.float64
        ).round().long()
        sample = flat.index_select(0, indices).to(device="cpu", dtype=torch.float16)
        self.records.append(
            {
                "run_id": self.run_id,
                "step": self.current_step,
                "block": block,
                "branch": self.current_branch,
                "signal": signal,
                "full_shape": tuple(value.shape),
                "rows": sample.shape[0],
                "features": sample.shape[1],
                "sample": sample,
            }
        )
        if self.current_step in self.spectral_steps:
            self._capture_spectrum(block, signal, flat)

    def _lowpass_mask(self, device: torch.device) -> torch.Tensor:
        cache_key = (device, self.grid)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        axes = [torch.fft.fftfreq(size, device=device).abs() / 0.5 for size in self.grid]
        radius = (
            axes[0][:, None, None].square()
            + axes[1][None, :, None].square()
            + axes[2][None, None, :].square()
        )
        requested = max(1, min(radius.numel(), round(radius.numel() * self.spectral_density)))
        threshold = radius.flatten().sort().values[requested - 1]
        mask = radius <= threshold
        self._mask_cache[cache_key] = mask
        return mask

    def _capture_spectrum(self, block: int, signal: str, flat: torch.Tensor) -> None:
        channels = min(self.spectral_channels, flat.shape[1])
        channel_indices = torch.linspace(
            0, flat.shape[1] - 1, channels, device=flat.device, dtype=torch.float64
        ).round().long()
        grid = flat.index_select(1, channel_indices).float().reshape(*self.grid, channels)
        centered = grid - grid.mean(dim=(0, 1, 2), keepdim=True)
        mask = self._lowpass_mask(flat.device)
        signal_code = {
            "ffn_input": 1,
            "ffn_hidden_post_gelu": 2,
            "ffn_output": 3,
        }[signal]
        generator = torch.Generator(device=flat.device).manual_seed(
            self.seed
            + self.current_step * 100_003
            + block * 1_009
            + self.current_branch * 97
            + signal_code
        )
        permutation = torch.randperm(centered.shape[0] * centered.shape[1] * centered.shape[2], generator=generator, device=flat.device)
        shuffled = centered.reshape(-1, channels).index_select(0, permutation).reshape_as(centered)
        for control, candidate in (("original", centered), ("token_shuffled", shuffled)):
            transformed = torch.fft.fftn(candidate, dim=(0, 1, 2), norm="ortho")
            power = transformed.abs().square().sum(dim=-1)
            total = power.sum().clamp_min(1e-30)
            row: dict[str, object] = {
                "run_id": self.run_id,
                "step": self.current_step,
                "block": block,
                "branch": self.current_branch,
                "signal": signal,
                "control": control,
                "requested_density": self.spectral_density,
                "actual_density": float(mask.float().mean()),
                "lowpass_retained_energy": float(power[mask].sum() / total),
                "channels_sampled": channels,
            }
            variance = candidate.square().mean().clamp_min(1e-30)
            for axis, name in enumerate(("temporal", "height", "width")):
                if candidate.shape[axis] <= 1:
                    row[f"{name}_normalized_difference"] = float("nan")
                    continue
                left = candidate.narrow(axis, 0, candidate.shape[axis] - 1)
                right = candidate.narrow(axis, 1, candidate.shape[axis] - 1)
                row[f"{name}_normalized_difference"] = float(
                    (right - left).square().mean() / (2.0 * variance)
                )
            self.spectrum_records.append(row)

    def assert_complete(self) -> None:
        expected = self.sampling_steps * 2
        if self.call_index != expected:
            raise RuntimeError(
                f"activation audit observed {self.call_index} model calls, expected {expected}"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sampling_steps": self.sampling_steps,
                "frame_num": self.frame_num,
                "grid": self.grid,
                "blocks": sorted(self.blocks),
                "branches": sorted(self.branches),
                "spectral_steps": sorted(self.spectral_steps),
                "sample_rows": self.sample_rows,
                "spectral_channels": self.spectral_channels,
                "spectral_density": self.spectral_density,
                "records": self.records,
                "spectrum_records": self.spectrum_records,
            },
            path,
        )

    def restore(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.model.forward = self.original_model_forward


__all__ = ["FFNActivationStructureController"]
