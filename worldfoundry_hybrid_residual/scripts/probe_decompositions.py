#!/usr/bin/env python3
"""Probe RobuQ and structured residual decompositions on real DiT weights."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


LOG2_3 = math.log2(3.0)


@dataclass
class Factors:
    u: torch.Tensor
    s: torch.Tensor
    v: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.s.numel())


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_hadamard_module():
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(__file__).resolve().parent / "hadamard_utils.py",
        project_root / "src" / "RobuQ" / "quant" / "quantized_modules" / "hadamard_utils.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("robuq_hadamard_utils", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    raise FileNotFoundError(
        "RobuQ hadamard_utils.py was not found; place it under src/RobuQ/quant/quantized_modules"
    )


def random_hadamard(
    value: torch.Tensor,
    signs: torch.Tensor,
    hadamard_module,
) -> torch.Tensor:
    # Use the exact normalized mixed-radix transform shipped by RobuQ. The
    # pure PyTorch implementation works without the optional CUDA extension.
    return hadamard_module.matmul_hadU(value * signs)


def ternary_groupwise(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    out_features, in_features = weight.shape
    groups = math.ceil(in_features / group_size)
    padded = groups * group_size
    pad = padded - in_features
    grouped = F.pad(weight, (0, pad)).view(out_features, groups, group_size)
    counts = torch.full(
        (groups,), group_size, dtype=weight.dtype, device=weight.device
    )
    if pad:
        counts[-1] = group_size - pad
    scale = grouped.abs().sum(dim=-1, keepdim=True) / counts.view(1, groups, 1)
    discrete = torch.round(grouped / scale.clamp_min(torch.finfo(weight.dtype).tiny))
    quantized = discrete.clamp(-1, 1) * scale
    return quantized.view(out_features, padded)[:, :in_features]


def fp8_tensorwise(weight: torch.Tensor) -> torch.Tensor:
    dtype = torch.float8_e4m3fn
    limit = float(torch.finfo(dtype).max)
    scale = (weight.abs().amax() / limit).clamp_min(torch.finfo(torch.float32).tiny)
    return (weight / scale).clamp(-limit, limit).to(dtype).float() * scale


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversample: int = 12,
    niter: int = 4,
) -> Factors:
    rank = min(int(rank), *matrix.shape)
    if rank <= 0:
        return Factors(
            matrix.new_empty((matrix.shape[0], 0)),
            matrix.new_empty((0,)),
            matrix.new_empty((matrix.shape[1], 0)),
        )
    q = min(rank + oversample, *matrix.shape)
    devices = [matrix.device.index] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    order = torch.argsort(s, descending=True)[:rank]
    return Factors(u[:, order], s[order], v[:, order])


def lowrank_dense(factors: Factors, rank: int | None = None) -> torch.Tensor:
    rank = factors.rank if rank is None else min(int(rank), factors.rank)
    if rank == 0:
        return factors.u.new_zeros((factors.u.shape[0], factors.v.shape[0]))
    return (factors.u[:, :rank] * factors.s[:rank]) @ factors.v[:, :rank].T


def bcm_project(weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    if out_features % block_size or in_features % block_size:
        raise ValueError(
            f"shape {tuple(weight.shape)} is not divisible by block size {block_size}"
        )
    out_blocks = out_features // block_size
    in_blocks = in_features // block_size
    blocks = (
        weight.view(out_blocks, block_size, in_blocks, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    rows = torch.arange(block_size, device=weight.device).view(1, block_size)
    offsets = torch.arange(block_size, device=weight.device).view(block_size, 1)
    row_index = rows.expand(block_size, block_size)
    col_index = (row_index + offsets) % block_size
    generators = blocks[..., row_index, col_index].mean(dim=-1)
    dense_index = (
        torch.arange(block_size, device=weight.device).view(1, block_size)
        - torch.arange(block_size, device=weight.device).view(block_size, 1)
    ) % block_size
    projected_blocks = generators[..., dense_index]
    projected = (
        projected_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(out_features, in_features)
    )
    return projected, generators


def top_output_block_residual(
    residual: torch.Tensor,
    parameter_budget: int,
    block_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    out_features, in_features = residual.shape
    if out_features % block_rows:
        raise ValueError("output width must be divisible by block_rows")
    block_count = out_features // block_rows
    selected_count = min(block_count, parameter_budget // (block_rows * in_features))
    if selected_count <= 0:
        return residual.new_zeros(residual.shape), residual.new_empty((0,), dtype=torch.long), 0
    blocks = residual.view(block_count, block_rows, in_features)
    energy = blocks.float().square().sum(dim=(1, 2))
    selected_blocks = torch.topk(energy, k=selected_count).indices.sort().values
    sparse = residual.new_zeros(residual.shape)
    row_indices = []
    for block in selected_blocks.tolist():
        start = block * block_rows
        stop = start + block_rows
        sparse[start:stop] = residual[start:stop]
        row_indices.extend(range(start, stop))
    indices = torch.tensor(row_indices, device=residual.device, dtype=torch.long)
    metadata_bits = selected_count * max(1, math.ceil(math.log2(max(2, block_count))))
    return sparse, indices, metadata_bits


def top_tile_residual(
    residual: torch.Tensor,
    parameter_budget: int,
    tile_size: int,
) -> tuple[torch.Tensor, int, int]:
    out_features, in_features = residual.shape
    if out_features % tile_size or in_features % tile_size:
        raise ValueError("matrix shape must be divisible by tile_size")
    out_blocks = out_features // tile_size
    in_blocks = in_features // tile_size
    tile_budget = min(out_blocks * in_blocks, parameter_budget // (tile_size * tile_size))
    if tile_budget <= 0:
        return residual.new_zeros(residual.shape), 0, 0
    blocks = (
        residual.view(out_blocks, tile_size, in_blocks, tile_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    flat = blocks.view(out_blocks * in_blocks, tile_size, tile_size)
    energy = flat.float().square().sum(dim=(1, 2))
    selected = torch.topk(energy, k=tile_budget).indices
    sparse_flat = torch.zeros_like(flat)
    sparse_flat[selected] = flat[selected]
    sparse = (
        sparse_flat.view(out_blocks, in_blocks, tile_size, tile_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(out_features, in_features)
    )
    total_tiles = out_blocks * in_blocks
    metadata_bits = tile_budget * max(1, math.ceil(math.log2(max(2, total_tiles))))
    return sparse, tile_budget, metadata_bits


def spectral_norm_power(error: torch.Tensor, steps: int = 16) -> float:
    generator = torch.Generator(device=error.device)
    generator.manual_seed(9917)
    vector = torch.randn(
        error.shape[1], device=error.device, dtype=error.dtype, generator=generator
    )
    vector = vector / vector.norm().clamp_min(1e-12)
    for _ in range(steps):
        left = error @ vector
        left = left / left.norm().clamp_min(1e-12)
        vector = error.T @ left
        vector = vector / vector.norm().clamp_min(1e-12)
    return float((error @ vector).norm().item())


def main_storage_bits(
    kind: str,
    shape: tuple[int, int],
    group_size: int,
) -> float:
    out_features, in_features = shape
    count = out_features * in_features
    if kind == "ternary":
        scale_count = out_features * math.ceil(in_features / group_size)
        return count * LOG2_3 + scale_count * 16
    if kind == "fp8":
        return count * 8 + 32
    if kind == "bf16":
        return count * 16
    raise ValueError(kind)


def evaluate(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    activation: torch.Tensor,
) -> dict[str, float]:
    error = reference - estimate
    ref_norm = reference.float().norm().clamp_min(1e-12)
    error_norm = error.float().norm()
    with torch.no_grad():
        ref_output = activation @ reference.T
        estimate_output = activation @ estimate.T
        output_error = ref_output - estimate_output
        activation_error = output_error.float().norm() / ref_output.float().norm().clamp_min(1e-12)
        cosine = F.cosine_similarity(
            ref_output.float().reshape(1, -1),
            estimate_output.float().reshape(1, -1),
        ).item()
    return {
        "relative_fro_error": float((error_norm / ref_norm).item()),
        "spectral_error_proxy": spectral_norm_power(error.float()),
        "activation_relative_l2": float(activation_error.item()),
        "activation_cosine": float(cosine),
        "absolute_error_norm": float(error_norm.item()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_weight(checkpoint: str, key: str, device: torch.device) -> torch.Tensor:
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"{key!r} is absent from {checkpoint}")
        return handle.get_tensor(key).to(device=device, dtype=torch.float32)


def load_activation(
    path: str,
    in_features: int,
    samples: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, str]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    candidates = []
    if isinstance(obj, dict):
        for name in ("q", "k", "v", "hidden_states", "x", "input"):
            value = obj.get(name)
            if torch.is_tensor(value):
                candidates.append((name, value))
    for name, value in candidates:
        flattened = value.reshape(-1, value.shape[-2] * value.shape[-1]) if value.ndim >= 4 else value.reshape(-1, value.shape[-1])
        if flattened.shape[-1] != in_features:
            continue
        count = min(samples, flattened.shape[0])
        indices = torch.linspace(0, flattened.shape[0] - 1, count).long()
        return flattened[indices].to(device=device, dtype=torch.float32), f"real:{name}"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(samples, in_features, generator=generator, device=device), "synthetic:gaussian"


def record_spectrum(
    rows: list[dict],
    key: str,
    source: str,
    factors: Factors,
    total_norm_sq: float,
    ranks: Iterable[int],
) -> None:
    singular_sq = factors.s.float().square()
    cumulative = torch.cumsum(singular_sq, dim=0)
    for rank in ranks:
        if rank <= factors.rank:
            rows.append(
                {
                    "weight_key": key,
                    "source": source,
                    "rank": rank,
                    "energy_ratio": float((cumulative[rank - 1] / max(total_norm_sq, 1e-30)).item()),
                }
            )


def run_weight(
    *,
    checkpoint: str,
    activation_path: str,
    key: str,
    device: torch.device,
    ranks: list[int],
    block_sizes: list[int],
    spectrum_ranks: list[int],
    rank_budget: int,
    group_size: int,
    activation_samples: int,
    block_rows: int,
    tile_size: int,
    seed: int,
    hadamard_module,
) -> tuple[list[dict], list[dict], list[dict]]:
    print(f"[probe] loading {key}", flush=True)
    weight = load_weight(checkpoint, key, device)
    activation, activation_source = load_activation(
        activation_path, weight.shape[1], activation_samples, device, seed
    )
    torch.manual_seed(seed)
    signs = torch.randint(0, 2, (weight.shape[1],), device=device, dtype=torch.int64)
    signs = signs.to(torch.float32).mul_(2).sub_(1)

    no_had_main = ternary_groupwise(weight, group_size)
    no_had_eval = evaluate(weight, no_had_main, activation)
    weight_rot = random_hadamard(weight, signs, hadamard_module)
    activation_rot = random_hadamard(activation, signs, hadamard_module)
    ternary_main = ternary_groupwise(weight_rot, group_size)
    fp8_main = fp8_tensorwise(weight_rot)
    ternary_error = weight_rot - ternary_main
    fp8_error = weight_rot - fp8_main
    ternary_error_norm = float(ternary_error.float().norm().item())
    fp8_error_norm = float(fp8_error.float().norm().item())
    out_features, in_features = weight.shape
    target_params = rank_budget * (out_features + in_features)
    max_rank = max(max(ranks), max(spectrum_ranks), rank_budget)

    metrics: list[dict] = []
    spectra: list[dict] = []
    structures: list[dict] = []

    def add_method(
        name: str,
        family: str,
        estimate: torch.Tensor,
        *,
        main_kind: str,
        high_precision_params: int,
        metadata_bits: int = 0,
        launch_groups: int = 1,
        baseline_error_norm: float,
        notes: str = "",
        eval_override: dict[str, float] | None = None,
    ) -> None:
        measured = eval_override if eval_override is not None else evaluate(weight_rot, estimate, activation_rot)
        residual_bits = high_precision_params * 16 + metadata_bits
        total_bits = main_storage_bits(main_kind, (out_features, in_features), group_size) + residual_bits
        captured = 1.0 - (measured["absolute_error_norm"] / max(baseline_error_norm, 1e-30)) ** 2
        metrics.append(
            {
                "weight_key": key,
                "shape": f"{out_features}x{in_features}",
                "activation_source": activation_source,
                "method": name,
                "family": family,
                "main_kind": main_kind,
                "high_precision_params": high_precision_params,
                "lr16_budget_params": target_params,
                "budget_ratio_vs_lr16": high_precision_params / max(target_params, 1),
                "metadata_bits": metadata_bits,
                "total_estimated_bits": total_bits,
                "stored_fp16_param_equivalent": total_bits / 16.0,
                "residual_launch_groups_proxy": launch_groups,
                "captured_main_error_energy": captured,
                "notes": notes,
                **measured,
            }
        )
        print(
            f"[probe] {key} {name}: fro={measured['relative_fro_error']:.5f} "
            f"act={measured['activation_relative_l2']:.5f} params={high_precision_params}",
            flush=True,
        )

    add_method(
        "ternary_no_hadamard",
        "main_only",
        no_had_main,
        main_kind="ternary",
        high_precision_params=0,
        baseline_error_norm=float((weight - no_had_main).norm().item()),
        notes="Official RobuQ groupwise ternary rule without rotation",
        eval_override=no_had_eval,
    )
    add_method(
        "ternary_hadamard",
        "main_only",
        ternary_main,
        main_kind="ternary",
        high_precision_params=0,
        baseline_error_norm=ternary_error_norm,
        notes="RobuQ-compatible randomized Hadamard rotation",
    )
    add_method(
        "fp8_hadamard",
        "main_only",
        fp8_main,
        main_kind="fp8",
        high_precision_params=0,
        baseline_error_norm=fp8_error_norm,
        notes="H200-native tensorwise E4M3 weight proxy",
    )

    weight_factors = randomized_svd(weight_rot, max_rank, seed=seed + 10)
    ternary_error_factors = randomized_svd(ternary_error, max_rank, seed=seed + 20)
    fp8_error_factors = randomized_svd(fp8_error, max_rank, seed=seed + 30)
    record_spectrum(
        spectra, key, "hadamard_weight", weight_factors, float(weight_rot.square().sum().item()), spectrum_ranks
    )
    record_spectrum(
        spectra, key, "ternary_error", ternary_error_factors, float(ternary_error.square().sum().item()), spectrum_ranks
    )
    record_spectrum(
        spectra, key, "fp8_error", fp8_error_factors, float(fp8_error.square().sum().item()), spectrum_ranks
    )

    for rank in ranks:
        lowrank_weight = lowrank_dense(weight_factors, rank)
        robuq_main = ternary_groupwise(weight_rot - lowrank_weight, group_size)
        add_method(
            f"robuq_weight_svd_r{rank}",
            "robuq",
            robuq_main + lowrank_weight,
            main_kind="ternary",
            high_precision_params=rank * (out_features + in_features),
            baseline_error_norm=ternary_error_norm,
            notes="Paper-faithful initialization before QAT; randomized top-SVD probe",
        )
        qer_lowrank = lowrank_dense(ternary_error_factors, rank)
        add_method(
            f"ternary_qer_svd_r{rank}",
            "qer_lowrank",
            ternary_main + qer_lowrank,
            main_kind="ternary",
            high_precision_params=rank * (out_features + in_features),
            baseline_error_norm=ternary_error_norm,
            notes="Training-free SVD fitted to fixed ternary quantization error",
        )
        fp8_lowrank = lowrank_dense(fp8_error_factors, rank)
        add_method(
            f"fp8_qer_svd_r{rank}",
            "qer_lowrank",
            fp8_main + fp8_lowrank,
            main_kind="fp8",
            high_precision_params=rank * (out_features + in_features),
            baseline_error_norm=fp8_error_norm,
            notes="Training-free SVD fitted to fixed FP8 weight error",
        )

    for block_size in block_sizes:
        if out_features % block_size or in_features % block_size:
            continue
        bcm, generators = bcm_project(ternary_error, block_size)
        bcm_params = int(generators.numel())
        residual_after_bcm = ternary_error - bcm
        remaining = max(0, target_params - bcm_params)
        budget_rank = remaining // (out_features + in_features)
        needed_rank = max(max(spectrum_ranks), max(ranks), budget_rank)
        after_bcm_factors = randomized_svd(
            residual_after_bcm, needed_rank, seed=seed + 100 + block_size
        )
        record_spectrum(
            spectra,
            key,
            f"ternary_error_after_bcm_b{block_size}",
            after_bcm_factors,
            float(residual_after_bcm.square().sum().item()),
            spectrum_ranks,
        )
        bcm_energy = float(bcm.square().sum().item())
        ternary_energy = float(ternary_error.square().sum().item())
        add_method(
            f"ternary_qer_bcm_b{block_size}",
            "qer_bcm",
            ternary_main + bcm,
            main_kind="ternary",
            high_precision_params=bcm_params,
            baseline_error_norm=ternary_error_norm,
            notes="Orthogonal full-grid block-circulant projection of quantization error",
        )
        if budget_rank > 0:
            correction = lowrank_dense(after_bcm_factors, budget_rank)
            add_method(
                f"ternary_qer_bcm_b{block_size}_svd_r{budget_rank}_budget_lr{rank_budget}",
                "qer_bcm_lowrank",
                ternary_main + bcm + correction,
                main_kind="ternary",
                high_precision_params=bcm_params + budget_rank * (out_features + in_features),
                baseline_error_norm=ternary_error_norm,
                notes="BCM-first hybrid constrained to the rank-16 FP16 parameter budget",
            )
            lr_first = lowrank_dense(ternary_error_factors, budget_rank)
            bcm_after_lr, generators_after_lr = bcm_project(ternary_error - lr_first, block_size)
            add_method(
                f"ternary_qer_svd_r{budget_rank}_bcm_b{block_size}_budget_lr{rank_budget}",
                "qer_lowrank_bcm",
                ternary_main + lr_first + bcm_after_lr,
                main_kind="ternary",
                high_precision_params=int(generators_after_lr.numel())
                + budget_rank * (out_features + in_features),
                baseline_error_norm=ternary_error_norm,
                notes="Low-rank-first ordering exposes overlap/redundancy with BCM",
            )
            bcm_after_lr_energy = float(bcm_after_lr.square().sum().item())
        else:
            bcm_after_lr_energy = 0.0
        expanded_rank = min(rank_budget, after_bcm_factors.rank)
        if expanded_rank > 0:
            correction = lowrank_dense(after_bcm_factors, expanded_rank)
            add_method(
                f"ternary_qer_bcm_b{block_size}_svd_r{expanded_rank}_expanded",
                "qer_bcm_lowrank",
                ternary_main + bcm + correction,
                main_kind="ternary",
                high_precision_params=bcm_params + expanded_rank * (out_features + in_features),
                baseline_error_norm=ternary_error_norm,
                notes="Expanded hybrid; not a same-parameter comparison",
            )
        structures.append(
            {
                "weight_key": key,
                "shape": f"{out_features}x{in_features}",
                "source": "ternary_error",
                "block_size": block_size,
                "bcm_params": bcm_params,
                "bcm_energy_ratio": bcm_energy / max(ternary_energy, 1e-30),
                "budget_rank_after_bcm": budget_rank,
                "bcm_after_lowrank_energy_ratio": bcm_after_lr_energy / max(ternary_energy, 1e-30),
            }
        )

        fp8_bcm, fp8_generators = bcm_project(fp8_error, block_size)
        fp8_remaining = max(0, target_params - int(fp8_generators.numel()))
        fp8_rank = fp8_remaining // (out_features + in_features)
        if fp8_rank > 0:
            fp8_after = fp8_error - fp8_bcm
            fp8_after_factors = randomized_svd(
                fp8_after, fp8_rank, seed=seed + 300 + block_size
            )
            fp8_correction = lowrank_dense(fp8_after_factors, fp8_rank)
            add_method(
                f"fp8_qer_bcm_b{block_size}_svd_r{fp8_rank}_budget_lr{rank_budget}",
                "qer_bcm_lowrank",
                fp8_main + fp8_bcm + fp8_correction,
                main_kind="fp8",
                high_precision_params=int(fp8_generators.numel())
                + fp8_rank * (out_features + in_features),
                baseline_error_norm=fp8_error_norm,
                notes="H200-native main format with same-budget structured residual",
            )

    base_rank = max(1, rank_budget // 2)
    base_lr = lowrank_dense(ternary_error_factors, base_rank)
    residual_after_lr = ternary_error - base_lr
    used_lr_params = base_rank * (out_features + in_features)
    sparse_budget = max(0, target_params - used_lr_params)
    if out_features % block_rows == 0:
        row_sparse, row_indices, row_metadata = top_output_block_residual(
            residual_after_lr, sparse_budget, block_rows
        )
        row_params = int(row_indices.numel()) * in_features
        if row_params:
            add_method(
                f"ternary_qer_svd_r{base_rank}_rowsparse_b{block_rows}_budget_lr{rank_budget}",
                "qer_lowrank_sparse",
                ternary_main + base_lr + row_sparse,
                main_kind="ternary",
                high_precision_params=used_lr_params + row_params,
                metadata_bits=row_metadata,
                launch_groups=max(1, int(row_indices.numel()) // block_rows),
                baseline_error_norm=ternary_error_norm,
                notes="Coarse output-row blocks: one small GEMM plus scatter per selected block group",
            )
            structures.append(
                {
                    "weight_key": key,
                    "shape": f"{out_features}x{in_features}",
                    "source": "ternary_error_after_lowrank",
                    "block_size": block_rows,
                    "structure": "row_sparse",
                    "selected_params": row_params,
                    "captured_residual_energy_ratio": float(
                        row_sparse.square().sum().item()
                        / max(residual_after_lr.square().sum().item(), 1e-30)
                    ),
                }
            )
    if out_features % tile_size == 0 and in_features % tile_size == 0:
        tile_sparse, tile_count, tile_metadata = top_tile_residual(
            residual_after_lr, sparse_budget, tile_size
        )
        tile_params = tile_count * tile_size * tile_size
        if tile_params:
            add_method(
                f"ternary_qer_svd_r{base_rank}_tilesparse_b{tile_size}_budget_lr{rank_budget}",
                "qer_lowrank_sparse",
                ternary_main + base_lr + tile_sparse,
                main_kind="ternary",
                high_precision_params=used_lr_params + tile_params,
                metadata_bits=tile_metadata,
                launch_groups=tile_count,
                baseline_error_norm=ternary_error_norm,
                notes="Accuracy-oriented top tiles; launch count flags GPU fragmentation",
            )
            structures.append(
                {
                    "weight_key": key,
                    "shape": f"{out_features}x{in_features}",
                    "source": "ternary_error_after_lowrank",
                    "block_size": tile_size,
                    "structure": "tile_sparse",
                    "selected_params": tile_params,
                    "selected_tiles": tile_count,
                    "captured_residual_energy_ratio": float(
                        tile_sparse.square().sum().item()
                        / max(residual_after_lr.square().sum().item(), 1e-30)
                    ),
                }
            )

    fp8_base_lr = lowrank_dense(fp8_error_factors, base_rank)
    fp8_residual_after_lr = fp8_error - fp8_base_lr
    if out_features % block_rows == 0:
        fp8_row, fp8_rows, fp8_metadata = top_output_block_residual(
            fp8_residual_after_lr, sparse_budget, block_rows
        )
        fp8_row_params = int(fp8_rows.numel()) * in_features
        if fp8_row_params:
            add_method(
                f"fp8_qer_svd_r{base_rank}_rowsparse_b{block_rows}_budget_lr{rank_budget}",
                "qer_lowrank_sparse",
                fp8_main + fp8_base_lr + fp8_row,
                main_kind="fp8",
                high_precision_params=used_lr_params + fp8_row_params,
                metadata_bits=fp8_metadata,
                launch_groups=max(1, int(fp8_rows.numel()) // block_rows),
                baseline_error_norm=fp8_error_norm,
                notes="Native-FP8 main plus GPU-coarse sparse correction",
            )

    del weight, activation, weight_rot, activation_rot
    torch.cuda.empty_cache()
    return metrics, spectra, structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--keys", required=True, help="Comma-separated safetensor keys")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--ranks", default="8,16,32")
    parser.add_argument("--block-sizes", default="64,128,256")
    parser.add_argument("--spectrum-ranks", default="1,2,4,8,16,32,64")
    parser.add_argument("--rank-budget", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-samples", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This probe is intended for the isolated CUDA experiment environment")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    hadamard_module, hadamard_path = load_hadamard_module()

    all_metrics: list[dict] = []
    all_spectra: list[dict] = []
    all_structures: list[dict] = []
    started = time.time()
    keys = parse_strings(args.keys)
    for index, key in enumerate(keys):
        metrics, spectra, structures = run_weight(
            checkpoint=args.checkpoint,
            activation_path=args.activation,
            key=key,
            device=device,
            ranks=parse_ints(args.ranks),
            block_sizes=parse_ints(args.block_sizes),
            spectrum_ranks=parse_ints(args.spectrum_ranks),
            rank_budget=args.rank_budget,
            group_size=args.group_size,
            activation_samples=args.activation_samples,
            block_rows=args.block_rows,
            tile_size=args.tile_size,
            seed=args.seed + index * 1000,
            hadamard_module=hadamard_module,
        )
        all_metrics.extend(metrics)
        all_spectra.extend(spectra)
        all_structures.extend(structures)
        write_csv(output_dir / "decomposition_metrics.partial.csv", all_metrics)

    write_csv(output_dir / "decomposition_metrics.csv", all_metrics)
    write_csv(output_dir / "residual_spectrum.csv", all_spectra)
    write_csv(output_dir / "structure_stats.csv", all_structures)
    manifest = {
        "arguments": vars(args),
        "checkpoint": os.path.realpath(args.checkpoint),
        "hadamard_source": str(hadamard_path),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "python": platform.python_version(),
        "elapsed_seconds": time.time() - started,
        "methodology": {
            "ternary": "RobuQ groupwise mean-absolute scale, round and clamp to {-1,0,1}",
            "fp8": "tensorwise E4M3 fake quantization matching WorldFoundry weight scaling",
            "svd": "torch.svd_lowrank with oversampling=12 and power_iterations=4",
            "bcm": "orthogonal projection of every dense block onto the circulant subspace",
            "hardware_scope": "decomposition only; actual H200 timings are produced by benchmark_h200.py",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[probe] wrote {output_dir} in {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
