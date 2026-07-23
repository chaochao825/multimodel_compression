#!/usr/bin/env python3
"""World Foundry FP8 linear with low-rank and static row-block residuals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.acceleration.quantization import Float8Linear

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _static_fp8_quantize_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        numel,
        fp8_max: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        scale = tl.load(scale_ptr)
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        values = values / scale
        values = tl.maximum(tl.minimum(values, fp8_max), -fp8_max)
        tl.store(output_ptr + offsets, values, mask=mask)


def _quantize_static_fp8(
    input: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    fp8_max = float(torch.finfo(fp8_dtype).max)
    if triton is not None and input.is_cuda:
        output = torch.empty_like(input, dtype=fp8_dtype)
        block_size = 1024
        grid = (triton.cdiv(input.numel(), block_size),)
        _static_fp8_quantize_kernel[grid](
            input,
            output,
            scale,
            input.numel(),
            fp8_max=fp8_max,
            block_size=block_size,
            num_warps=4,
        )
        return output
    return input.float().div(scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)


VALID_MODES = frozenset({"dense", "fp8", "hybrid"})


@dataclass(frozen=True)
class ReplacementSummary:
    names: tuple[str, ...]
    setup_seconds: float
    fp8_weight_values: int
    low_rank_values: int
    sparse_values: int
    selected_row_blocks: int


def _randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversample: int = 8,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(int(rank), *matrix.shape)
    if rank <= 0:
        return (
            matrix.new_empty((matrix.shape[0], 0)),
            matrix.new_empty((0, matrix.shape[1])),
        )
    q = min(rank + oversample, *matrix.shape)
    devices = [matrix.device.index] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    order = torch.argsort(s, descending=True)[:rank]
    up = u[:, order] * s[order]
    down = v[:, order].T
    return up, down


class HybridResidualLinear(Float8Linear):
    """Switchable dense, FP8, and FP8 plus structured residual linear.

    The FP8 payload and decode scale come from World Foundry. A one-step dense
    warmup calibrates a static input scale. The high-precision residual first
    uses low rank, then stores the highest-energy contiguous output-row blocks
    from the remaining quantization error.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        residual_rank: int,
        budget_rank: int,
        row_block_size: int,
        seed: int,
        static_scale_margin: float = 1.05,
    ) -> None:
        if residual_rank < 0 or budget_rank < residual_rank:
            raise ValueError("budget_rank must be at least residual_rank")
        if row_block_size <= 0:
            raise ValueError("row_block_size must be positive")
        if static_scale_margin < 1.0:
            raise ValueError("static_scale_margin must be at least one")
        source = weight.detach().float()
        super().__init__(weight, bias, keep_dense_fallback=True)
        quantized = self.weight_fp8.float() * self.weight_scale.float()
        error = source - quantized
        up, down = _randomized_svd(error, residual_rank, seed=seed)
        low_rank = up @ down if residual_rank else torch.zeros_like(error)
        remainder = error - low_rank

        value_budget = budget_rank * (self.out_features + self.in_features)
        low_rank_values = residual_rank * (self.out_features + self.in_features)
        sparse_budget = max(0, value_budget - low_rank_values)
        block_values = row_block_size * self.in_features
        available_blocks = self.out_features // row_block_size
        selected_blocks = min(available_blocks, sparse_budget // block_values)
        if selected_blocks:
            usable_rows = available_blocks * row_block_size
            blocks = remainder[:usable_rows].view(
                available_blocks, row_block_size, self.in_features
            )
            energy = blocks.square().sum(dim=(1, 2))
            block_ids = torch.topk(energy, k=selected_blocks).indices.sort().values
            offsets = torch.arange(row_block_size, device=weight.device)
            row_indices = (block_ids[:, None] * row_block_size + offsets).reshape(-1)
            sparse_weight = remainder[row_indices]
        else:
            block_ids = torch.empty(0, device=weight.device, dtype=torch.long)
            row_indices = torch.empty(0, device=weight.device, dtype=torch.long)
            sparse_weight = remainder.new_empty((0, self.in_features))

        compute_dtype = weight.dtype
        self.register_buffer("residual_up", up.to(dtype=compute_dtype).contiguous())
        self.register_buffer("residual_down", down.to(dtype=compute_dtype).contiguous())
        self.register_buffer("sparse_row_indices", row_indices.contiguous())
        self.register_buffer("sparse_weight", sparse_weight.to(dtype=compute_dtype).contiguous())
        self.register_buffer("selected_block_ids", block_ids.contiguous())
        self.register_buffer(
            "static_input_scale",
            torch.zeros(1, device=weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "calibration_amax",
            torch.zeros(1, device=weight.device, dtype=torch.float32),
        )
        self.residual_rank = int(residual_rank)
        self.budget_rank = int(budget_rank)
        self.row_block_size = int(row_block_size)
        self.static_scale_margin = float(static_scale_margin)
        self.static_scale_ready = False
        self.mode = "dense"
        self.calibration_enabled = False
        self.calibration_observations = 0
        self.dense_calls = 0
        self.fp8_calls = 0
        self.residual_calls = 0

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        residual_rank: int,
        budget_rank: int,
        row_block_size: int,
        seed: int,
        static_scale_margin: float = 1.05,
    ) -> "HybridResidualLinear":
        return cls(
            layer.weight,
            layer.bias,
            residual_rank=residual_rank,
            budget_rank=budget_rank,
            row_block_size=row_block_size,
            seed=seed,
            static_scale_margin=static_scale_margin,
        )

    @property
    def low_rank_values(self) -> int:
        return self.residual_rank * (self.in_features + self.out_features)

    @property
    def sparse_values(self) -> int:
        return int(self.sparse_weight.numel())

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown hybrid residual mode: {mode}")
        self.mode = mode
        self.low_precision_enabled = mode != "dense"

    def reset_runtime_stats(self) -> None:
        self.dense_calls = 0
        self.fp8_calls = 0
        self.residual_calls = 0

    def set_calibration(self, enabled: bool) -> None:
        self.calibration_enabled = bool(enabled)

    def finalize_calibration(self) -> None:
        if self.calibration_observations <= 0:
            raise RuntimeError("static FP8 calibration did not observe any inputs")
        fp8_max = float(torch.finfo(self.fp8_dtype).max)
        scale = (
            self.calibration_amax.float()
            * self.static_scale_margin
            / fp8_max
        ).clamp_min(torch.finfo(torch.float32).tiny)
        self.static_input_scale.copy_(scale)
        self.static_scale_ready = True
        self.calibration_enabled = False

    def _dense_forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("dense fallback was not retained")
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, self.weight.to(dtype=input.dtype), bias)

    def _fp8_static_forward(self, input: torch.Tensor) -> tuple[torch.Tensor, bool]:
        hardware_path = (
            input.device.type == "cuda"
            and not torch.is_grad_enabled()
            and input.dtype in {torch.float16, torch.bfloat16}
            and input.shape[-1] == self.in_features
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
            and self._hardware_eligible
        )
        if not hardware_path:
            return self._dense_forward(input), False
        if not self.static_scale_ready:
            return super().forward(input), True
        original_shape = input.shape
        flattened = input.reshape(-1, self.in_features).contiguous()
        input_fp8 = _quantize_static_fp8(
            flattened,
            self.static_input_scale,
            self.fp8_dtype,
        )
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        output = torch._scaled_mm(
            input_fp8,
            self.weight_fp8.t(),
            self.static_input_scale,
            self.weight_scale,
            bias=bias,
            out_dtype=input.dtype,
            use_fast_accum=False,
        )
        return output.reshape(*original_shape[:-1], self.out_features), True

    def _residual_forward(self, input: torch.Tensor) -> torch.Tensor:
        down = self.residual_down.to(dtype=input.dtype)
        up = self.residual_up.to(dtype=input.dtype)
        output = F.linear(F.linear(input, down), up)
        if self.sparse_row_indices.numel():
            partial = F.linear(input, self.sparse_weight.to(dtype=input.dtype))
            output.index_add_(-1, self.sparse_row_indices, partial)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {input.shape[-1]}"
            )
        if self.calibration_enabled:
            observed = input.detach().float().abs().amax().reshape(1)
            self.calibration_amax.copy_(torch.maximum(self.calibration_amax, observed))
            self.calibration_observations += 1
        if self.mode == "dense" or self.calibration_enabled:
            self.dense_calls += 1
            return self._dense_forward(input)
        output, used_fp8 = self._fp8_static_forward(input)
        if used_fp8:
            self.fp8_calls += 1
        else:
            self.dense_calls += 1
        if self.mode == "hybrid" and used_fp8:
            self.residual_calls += 1
            output = output + self._residual_forward(input)
        return output


class HybridResidualController:
    """Manage all injected hybrid linears as one experiment component."""

    def __init__(self, modules: Iterable[tuple[str, HybridResidualLinear]], setup_seconds: float) -> None:
        self.modules = tuple(modules)
        self.setup_seconds = float(setup_seconds)
        by_block: dict[int, list[HybridResidualLinear]] = {}
        for name, module in self.modules:
            fields = name.split(".")
            if len(fields) < 2 or fields[0] != "blocks":
                raise ValueError(f"cannot recover block index from module name: {name}")
            block_index = int(fields[1])
            by_block.setdefault(block_index, []).append(module)
        self._by_block = {
            block_index: tuple(block_modules)
            for block_index, block_modules in by_block.items()
        }

    def set_mode(self, mode: str) -> None:
        for _, module in self.modules:
            module.set_mode(mode)

    @property
    def block_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_block))

    def set_block_mode(self, block_index: int, mode: str) -> None:
        modules = self._by_block.get(int(block_index))
        if modules is None:
            raise KeyError(f"no injected linears for block {block_index}")
        for module in modules:
            module.set_mode(mode)

    def reset_runtime_stats(self) -> None:
        for _, module in self.modules:
            module.reset_runtime_stats()

    def set_calibration(self, enabled: bool) -> None:
        for _, module in self.modules:
            module.set_calibration(enabled)

    def finalize_calibration(self) -> None:
        for _, module in self.modules:
            module.finalize_calibration()

    def summary(self) -> ReplacementSummary:
        return ReplacementSummary(
            names=tuple(name for name, _ in self.modules),
            setup_seconds=self.setup_seconds,
            fp8_weight_values=sum(module.weight_fp8.numel() for _, module in self.modules),
            low_rank_values=sum(module.low_rank_values for _, module in self.modules),
            sparse_values=sum(module.sparse_values for _, module in self.modules),
            selected_row_blocks=sum(
                module.selected_block_ids.numel() for _, module in self.modules
            ),
        )

    def runtime_stats(self) -> dict[str, object]:
        scales = [float(module.static_input_scale.item()) for _, module in self.modules]
        return {
            "hybrid_linear_count": len(self.modules),
            "hybrid_dense_calls": sum(module.dense_calls for _, module in self.modules),
            "hybrid_fp8_calls": sum(module.fp8_calls for _, module in self.modules),
            "hybrid_residual_calls": sum(
                module.residual_calls for _, module in self.modules
            ),
            "static_scale_min": min(scales) if scales else math.nan,
            "static_scale_max": max(scales) if scales else math.nan,
        }


def _parse_block_selection(blocks: str, total: int) -> tuple[int, ...]:
    if blocks.strip().lower() == "all":
        return tuple(range(total))
    selected = tuple(sorted({int(token) for token in blocks.split(",") if token.strip()}))
    if any(index < 0 or index >= total for index in selected):
        raise ValueError(f"block selection must lie in [0, {total - 1}]")
    return selected


def replace_wan_linears(
    model: nn.Module,
    *,
    scope: str = "self_attn",
    targets: Iterable[str] | None = None,
    blocks: str = "all",
    residual_rank: int = 8,
    budget_rank: int = 16,
    row_block_size: int = 8,
    seed: int = 20260723,
    static_scale_margin: float = 1.05,
) -> HybridResidualController:
    if scope == "self_attn":
        target_map: dict[str, str | int] = {name: name for name in ("q", "k", "v", "o")}
        selected_targets = tuple(targets or ("q", "o"))
    elif scope == "ffn":
        target_map = {"up": 0, "down": 2}
        selected_targets = tuple(targets or ("up", "down"))
    else:
        raise ValueError(f"unknown Wan linear scope: {scope}")
    unknown = set(selected_targets) - set(target_map)
    if unknown:
        raise ValueError(f"unknown {scope} targets: {sorted(unknown)}")
    selected_blocks = _parse_block_selection(blocks, len(model.blocks))
    modules: list[tuple[str, HybridResidualLinear]] = []
    started = time.perf_counter()
    for block_index in selected_blocks:
        container = (
            model.blocks[block_index].self_attn
            if scope == "self_attn"
            else model.blocks[block_index].ffn
        )
        for target_index, target in enumerate(selected_targets):
            selector = target_map[target]
            original = (
                getattr(container, selector)
                if isinstance(selector, str)
                else container[selector]
            )
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"blocks.{block_index}.{scope}.{target} is not nn.Linear"
                )
            replacement = HybridResidualLinear.from_linear(
                original,
                residual_rank=residual_rank,
                budget_rank=budget_rank,
                row_block_size=row_block_size,
                seed=seed + block_index * 17 + target_index,
                static_scale_margin=static_scale_margin,
            )
            if isinstance(selector, str):
                setattr(container, selector, replacement)
            else:
                container[selector] = replacement
            modules.append((f"blocks.{block_index}.{scope}.{target}", replacement))
            del original
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return HybridResidualController(modules, time.perf_counter() - started)


def replace_wan_self_attention_linears(
    model: nn.Module,
    **kwargs: object,
) -> HybridResidualController:
    return replace_wan_linears(model, scope="self_attn", **kwargs)


__all__ = [
    "HybridResidualController",
    "HybridResidualLinear",
    "ReplacementSummary",
    "replace_wan_linears",
    "replace_wan_self_attention_linears",
]
