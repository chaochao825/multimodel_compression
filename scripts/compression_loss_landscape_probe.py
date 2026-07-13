#!/usr/bin/env python3
"""Probe compression loss landscapes and parameter-efficient compensation.

This probe operates on the eight distinct saved attention matrices.  It treats
each matrix as a target and evaluates:

* block size and cross-block kernel sharing;
* shared kernel + column/full per-block scales;
* low-rank dictionaries over the block-circulant kernel tensor;
* global/row-block/per-block quantization scales;
* sink, low-rank, local-cyclic, and sparse residual compensation;
* budget-capped residual allocation, fit-order controls, and Pareto efficiency.

All fitted structures are per-target oracle diagnostics.  The repository did
not save V, logits, model weights, or per-example task outputs for these maps,
so new variants can only be evaluated with attention-space losses.  Existing
true-V output-error scalars are streamed from the original CSV files and kept
as anchors; they are not recomputed for new variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from hybrid_attention_decomposition import local_cyclic_projection, topk_sparse_component


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
VALUE_BITS = 16
SCALE_BITS = 16
COMMON_BLOCK_SIZE = 4
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
QUANT_BITS = (2, 3, 4, 6, 8)
QUANT_SCHEMES = ("global", "block_row", "per_block")
COMPENSATION_NAMES = ("sink", "lowrank", "local", "sparse")
TRUE_V_ANCHOR_PATHS = (
    LOG_DIR / "structured_attention_probe_vit_20260703.csv",
    LOG_DIR / "structured_attention_probe_qwen_20260703.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def row_normalize_nonnegative(matrix: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(matrix, dtype=np.float64), 0.0, None)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom <= 1e-15:
        return 0.0
    return float(np.linalg.norm(target - approx) / denom)


def mean_row_kl(target: np.ndarray, approx: np.ndarray, eps: float = 1e-12) -> float:
    left = row_normalize_nonnegative(target)
    right = row_normalize_nonnegative(approx)
    left_safe = np.clip(left, eps, None)
    right_safe = np.clip(right, eps, None)
    left_safe /= left_safe.sum(axis=1, keepdims=True)
    right_safe /= right_safe.sum(axis=1, keepdims=True)
    value = float(
        np.mean(np.sum(left_safe * (np.log(left_safe) - np.log(right_safe)), axis=1))
    )
    return max(0.0, value)


def mean_row_js(target: np.ndarray, approx: np.ndarray, eps: float = 1e-12) -> float:
    left = row_normalize_nonnegative(target)
    right = row_normalize_nonnegative(approx)
    midpoint = 0.5 * (left + right)
    return 0.5 * mean_row_kl(left, midpoint, eps) + 0.5 * mean_row_kl(
        right, midpoint, eps
    )


def mean_row_tv(target: np.ndarray, approx: np.ndarray) -> float:
    return float(0.5 * np.mean(np.sum(np.abs(target - approx), axis=1)))


def evaluate_candidate(attention: np.ndarray, raw_approx: np.ndarray) -> Dict[str, float]:
    raw = np.clip(np.asarray(raw_approx, dtype=np.float64), 0.0, None)
    approx = row_normalize_nonnegative(raw)
    rel = relative_error(attention, approx)
    return {
        "raw_relative_fro_error": relative_error(attention, raw),
        "relative_fro_error": rel,
        "normalized_mse": rel * rel,
        "mean_row_kl": mean_row_kl(attention, approx),
        "mean_row_js": mean_row_js(attention, approx),
        "mean_row_tv": mean_row_tv(attention, approx),
        "raw_row_sum_mae": float(np.mean(np.abs(raw.sum(axis=1) - 1.0))),
    }


def valid_block_sizes(n: int) -> List[int]:
    return [size for size in range(2, n + 1) if n % size == 0]


def offset_index(block_size: int) -> np.ndarray:
    idx = np.arange(block_size, dtype=np.int64)
    return (idx[None, :] - idx[:, None]) % block_size


def extract_block_kernels(matrix: np.ndarray, block_size: int) -> Tuple[np.ndarray, int]:
    n = int(matrix.shape[0])
    if matrix.shape != (n, n) or n % block_size != 0:
        raise ValueError(f"shape={matrix.shape} is incompatible with block_size={block_size}")
    blocks_per_axis = n // block_size
    blocks = (
        np.asarray(matrix, dtype=np.float64)
        .reshape(blocks_per_axis, block_size, blocks_per_axis, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(blocks_per_axis * blocks_per_axis, block_size, block_size)
    )
    offsets = offset_index(block_size)
    kernels = np.stack(
        [blocks[:, offsets == offset].mean(axis=1) for offset in range(block_size)],
        axis=1,
    )
    return kernels, blocks_per_axis


def expand_block_kernels(kernels: np.ndarray, blocks_per_axis: int, block_size: int) -> np.ndarray:
    offsets = offset_index(block_size)
    blocks = kernels[:, offsets.reshape(-1)].reshape(-1, block_size, block_size)
    return (
        blocks.reshape(blocks_per_axis, blocks_per_axis, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(blocks_per_axis * block_size, blocks_per_axis * block_size)
    )


def rank_approximation(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, float]:
    u, singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    used = min(int(rank), singular.size)
    approx = (u[:, :used] * singular[:used]) @ vt[:used]
    energy = float(np.sum(singular[:used] ** 2) / max(np.sum(singular**2), 1e-15))
    return approx, energy


def rank1_scale_factors(kernels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    u, singular, vt = np.linalg.svd(np.asarray(kernels, dtype=np.float64), full_matrices=False)
    root = math.sqrt(float(singular[0]))
    scales = u[:, 0] * root
    kernel = vt[0] * root
    if float(kernel.sum()) < 0.0:
        scales = -scales
        kernel = -kernel
    scales = np.clip(scales, 0.0, None)
    kernel = np.clip(kernel, 0.0, None)
    mean_scale = float(np.mean(scales))
    if mean_scale > 1e-15:
        scales = scales / mean_scale
        kernel = kernel * mean_scale
    captured = float(singular[0] ** 2 / max(np.sum(singular**2), 1e-15))
    return scales, kernel, captured


def gamma_scaled_kernels(
    kernels: np.ndarray,
    gamma: float,
    reoptimize_kernel: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    joint_scales, joint_kernel, captured = rank1_scale_factors(kernels)
    if reoptimize_kernel:
        fitted_scales = joint_scales
        base_kernel = joint_kernel
    else:
        # This path starts exactly at the optimal shared kernel (gamma=0) and
        # changes only block amplitudes.  At gamma=1 the scales are the
        # least-squares optimum for that frozen shared shape.
        base_kernel = np.asarray(kernels, dtype=np.float64).mean(axis=0)
        denom = float(np.dot(base_kernel, base_kernel))
        fitted_scales = np.clip(
            np.asarray(kernels, dtype=np.float64) @ base_kernel / max(denom, 1e-15),
            0.0,
            None,
        )
    scales = np.clip(1.0 + float(gamma) * (fitted_scales - 1.0), 0.0, None)
    if reoptimize_kernel:
        denom = float(np.dot(scales, scales))
        kernel = scales @ kernels / max(denom, 1e-15)
    else:
        kernel = base_kernel
    approx = scales[:, None] * kernel[None, :]
    direction = fitted_scales - 1.0
    coefficient_second_derivative = (
        2.0 * float(np.dot(direction, direction)) * float(np.dot(base_kernel, base_kernel))
    )
    return approx, {
        "scale_mean": float(np.mean(fitted_scales)),
        "scale_std": float(np.std(fitted_scales)),
        "scale_cv": float(np.std(fitted_scales) / max(np.mean(fitted_scales), 1e-15)),
        "scale_min": float(np.min(fitted_scales)),
        "scale_max": float(np.max(fitted_scales)),
        "rank1_kernel_energy_capture": captured,
        "fixed_kernel_coefficient_sse_second_derivative": coefficient_second_derivative,
        "fixed_kernel_matrix_sse_second_derivative": (
            kernels.shape[1] * coefficient_second_derivative
        ),
        "gamma_quadratic_note": (
            "The fixed-kernel SSE path is quadratic only while interpolated scales remain unclipped; "
            "the reoptimized-kernel path is not described by this curvature."
        ),
    }


def column_scaled_kernels(kernels: np.ndarray, blocks_per_axis: int) -> Tuple[np.ndarray, float]:
    tensor = kernels.reshape(blocks_per_axis, blocks_per_axis, kernels.shape[1])
    column_kernels = tensor.mean(axis=0)
    scales, kernel, captured = rank1_scale_factors(column_kernels)
    approx = np.broadcast_to(
        scales[None, :, None] * kernel[None, None, :],
        tensor.shape,
    )
    return approx.reshape(kernels.shape), captured


def quantize_kernels(
    kernels: np.ndarray,
    bits: int,
    scheme: str,
    blocks_per_axis: int,
) -> Tuple[np.ndarray, int]:
    values = np.clip(np.asarray(kernels, dtype=np.float64), 0.0, None)
    block_count = values.shape[0]
    if scheme == "global":
        groups = np.zeros(block_count, dtype=np.int64)
    elif scheme == "block_row":
        groups = np.arange(block_count, dtype=np.int64) // blocks_per_axis
    elif scheme == "per_block":
        groups = np.arange(block_count, dtype=np.int64)
    else:
        raise ValueError(f"unknown quantization scale scheme: {scheme}")
    quantized = np.zeros_like(values)
    qmax = (1 << int(bits)) - 1
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        vmax = float(values[mask].max(initial=0.0))
        scale = float(np.float16(vmax / qmax if vmax > 0.0 else 1.0))
        if scale == 0.0:
            # The ideal scale underflowed in the declared FP16 payload.  The
            # bit-exact decoded group is therefore all zeros; avoid inf*0.
            quantized[mask] = 0.0
            continue
        quantized[mask] = np.round(values[mask] / scale).clip(0, qmax) * scale
    return quantized, int(unique_groups.size)


def full_grid_cyclic_projection(attention: np.ndarray, grid_shape: Sequence[int]) -> np.ndarray:
    projected, _ = local_cyclic_projection(attention, grid_shape, radius=max(int(v) for v in grid_shape))
    return projected


def svd_relu_component(matrix: np.ndarray, rank: int, forbidden_cols: np.ndarray) -> np.ndarray:
    if rank <= 0:
        return np.zeros_like(matrix, dtype=np.float64)
    u, singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    used = min(int(rank), singular.size)
    out = np.clip((u[:, :used] * singular[:used]) @ vt[:used], 0.0, None)
    if forbidden_cols.size:
        out[:, forbidden_cols] = 0.0
    return out


def balanced_hyperparameters(n: int) -> Tuple[int, int, int]:
    k = max(2, n // 48)
    rank = 2 if n <= 64 else 4
    return k, rank, k


def parameter_fields(
    n: int,
    parameter_bits: int,
    stored_value_count: int,
    identifiable_value_count: int | None = None,
    index_bits: int = 0,
    scale_count: int = 0,
) -> Dict[str, float | int | None]:
    dense_bits = n * n * VALUE_BITS
    return {
        "stored_value_count": int(stored_value_count),
        "identifiable_value_count": (
            int(identifiable_value_count) if identifiable_value_count is not None else None
        ),
        "index_bits_total": int(index_bits),
        "scale_count": int(scale_count),
        "parameter_bits": int(parameter_bits),
        "parameter_equivalent_fp16": float(parameter_bits / VALUE_BITS),
        "parameter_fraction_of_dense_fp16": float(parameter_bits / dense_bits),
        "compression_ratio_vs_dense_fp16": (
            None if parameter_bits == 0 else float(dense_bits / parameter_bits)
        ),
    }


def base_row(example: Mapping[str, object]) -> Dict[str, object]:
    return {
        "key": example["key"],
        "label": example["label"],
        "map_id": example["map_id"],
        "family": example["family"],
        "source_input_id": example["source_input_id"],
        "grid_shape": "x".join(str(v) for v in example["grid_shape"]),
        "n": int(example["attention"].shape[0]),  # type: ignore[index]
    }


def source_input_id(map_id: str) -> str:
    parts = str(map_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else str(map_id)


def candidate_row(
    example: Mapping[str, object],
    category: str,
    method: str,
    raw_approx: np.ndarray,
    parameter_bits: int,
    stored_value_count: int,
    identifiable_value_count: int | None = None,
    index_bits: int = 0,
    scale_count: int = 0,
    target_fitted_oracle: bool = True,
    **metadata: object,
) -> dict:
    attention = np.asarray(example["attention"], dtype=np.float64)
    return {
        **base_row(example),
        "category": category,
        "method": method,
        **metadata,
        **parameter_fields(
            int(attention.shape[0]),
            parameter_bits,
            stored_value_count,
            identifiable_value_count,
            index_bits,
            scale_count,
        ),
        **evaluate_candidate(attention, raw_approx),
        "target_fitted_oracle": bool(target_fitted_oracle),
    }


def load_examples() -> Tuple[List[dict], List[Path]]:
    files: List[Path] = []
    examples: List[dict] = []
    structured_sources = [
        (
            "vit",
            LOG_DIR / "structured_attention_visual_vit_examples_hybrid_20260704.json",
            LOG_DIR / "structured_attention_visual_vit_examples_hybrid_20260704.npz",
        ),
        (
            "qwen",
            LOG_DIR / "structured_attention_visual_qwen_examples_hybrid_20260704.json",
            LOG_DIR / "structured_attention_visual_qwen_examples_hybrid_20260704.npz",
        ),
    ]
    for prefix, meta_path, npz_path in structured_sources:
        files.extend([meta_path, npz_path])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path)
        for item in meta["items"]:
            examples.append(
                {
                    "key": f"{prefix}_{item['key']}",
                    "label": item["label"],
                    "map_id": item["map_id"],
                    "source_input_id": source_input_id(item["map_id"]),
                    "family": "vit" if prefix == "vit" else "qwen3vl_visual",
                    "grid_shape": tuple(int(v) for v in item["grid_shape"]),
                    "attention": arrays[f"{item['key']}_attention"].astype(np.float64),
                    "source": repo_relative(npz_path),
                }
            )

    extra_meta_path = LOG_DIR / "qwen3vl_attention_visual_examples.json"
    extra_npz_path = LOG_DIR / "qwen3vl_attention_visual_examples.npz"
    files.extend([extra_meta_path, extra_npz_path])
    meta = json.loads(extra_meta_path.read_text(encoding="utf-8"))
    arrays = np.load(extra_npz_path)
    grid_shape = tuple(int(v) for v in meta["grid_thw"][0][1:])
    video_index = 1  # VP4GtrEsefk.mp4 is the second video in the structured probe.
    for item in meta["items"]:
        map_id = f"qwen:v{video_index}:f{item['frame']}:l{item['layer']}:h{item['head']}"
        examples.append(
            {
                "key": f"qwen_extra_{item['key']}",
                "label": f"Qwen VP4 L{item['layer']} H{item['head']} F{item['frame']}",
                "map_id": map_id,
                "source_input_id": source_input_id(map_id),
                "family": "qwen3vl_visual",
                "grid_shape": grid_shape,
                "attention": arrays[f"{item['key']}_attention"].astype(np.float64),
                "source": repo_relative(extra_npz_path),
            }
        )

    seen: Dict[str, str] = {}
    unique_examples: List[dict] = []
    for example in examples:
        digest = hashlib.sha256(np.asarray(example["attention"], dtype=np.float32).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen[digest] = str(example["key"])
        example["attention_sha256"] = digest
        unique_examples.append(example)
    return unique_examples, files


def add_backbone_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    attention = np.asarray(example["attention"], dtype=np.float64)
    n = int(attention.shape[0])
    dense_bits = n * n * VALUE_BITS
    rows.append(
        candidate_row(
            example,
            "baseline",
            "dense",
            attention,
            dense_bits,
            n * n,
            n * n - n,
            target_fitted_oracle=True,
            block_size=None,
            kernel_rank=None,
        )
    )
    uniform = np.full_like(attention, 1.0 / n)
    rows.append(
        candidate_row(
            example,
            "baseline",
            "uniform_no_payload",
            uniform,
            0,
            0,
            0,
            target_fitted_oracle=False,
            block_size=None,
            kernel_rank=0,
        )
    )
    grid = full_grid_cyclic_projection(attention, example["grid_shape"])
    rows.append(
        candidate_row(
            example,
            "backbone",
            "grid_bccb",
            grid,
            n * VALUE_BITS,
            n,
            n - 1,
            block_size="grid",
            kernel_rank=None,
        )
    )

    for block_size in valid_block_sizes(n):
        kernels, blocks_per_axis = extract_block_kernels(attention, block_size)
        block_count = kernels.shape[0]
        shared_kernel = kernels.mean(axis=0)
        shared_kernels = np.broadcast_to(shared_kernel[None, :], kernels.shape)
        shared_matrix = expand_block_kernels(shared_kernels, blocks_per_axis, block_size)
        rows.append(
            candidate_row(
                example,
                "backbone",
                "shared_kernel",
                shared_matrix,
                block_size * VALUE_BITS,
                block_size,
                block_size - 1,
                block_size=block_size,
                kernel_rank=1,
                scale_rank=0,
                blocks_per_axis=blocks_per_axis,
                block_count=block_count,
            )
        )

        column_kernels, column_capture = column_scaled_kernels(kernels, blocks_per_axis)
        column_matrix = expand_block_kernels(column_kernels, blocks_per_axis, block_size)
        column_values = block_size + blocks_per_axis
        rows.append(
            candidate_row(
                example,
                "backbone",
                "column_block_scale",
                column_matrix,
                column_values * VALUE_BITS,
                column_values,
                (block_size - 1) + (blocks_per_axis - 1),
                scale_count=blocks_per_axis,
                block_size=block_size,
                kernel_rank=1,
                kernel_energy_capture=column_capture,
                blocks_per_axis=blocks_per_axis,
                block_count=block_count,
            )
        )

        rank1_kernels, scale_stats = gamma_scaled_kernels(kernels, 1.0, reoptimize_kernel=True)
        rank1_matrix = expand_block_kernels(rank1_kernels, blocks_per_axis, block_size)
        rank1_values = block_count + block_size
        rank1_identifiable = (block_size - 1) + blocks_per_axis * (blocks_per_axis - 1)
        rows.append(
            candidate_row(
                example,
                "backbone",
                "block_scale_rank1",
                rank1_matrix,
                rank1_values * VALUE_BITS,
                rank1_values,
                rank1_identifiable,
                scale_count=block_count,
                block_size=block_size,
                kernel_rank=1,
                blocks_per_axis=blocks_per_axis,
                block_count=block_count,
                **scale_stats,
            )
        )

        max_rank = min(kernels.shape)
        for rank in sorted({value for value in (2, 3, 4, 8) if value <= max_rank}):
            approx_kernels, energy = rank_approximation(kernels, rank)
            matrix = expand_block_kernels(approx_kernels, blocks_per_axis, block_size)
            stored = rank * (block_count + block_size)
            coefficient_manifold_dof = rank * (block_count + block_size - rank)
            rows.append(
                candidate_row(
                    example,
                    "backbone",
                    f"block_kernel_rank{rank}",
                    matrix,
                    stored * VALUE_BITS,
                    stored,
                    None,
                    block_size=block_size,
                    kernel_rank=rank,
                    kernel_energy_capture=energy,
                    coefficient_manifold_dof_pre_row_normalization=coefficient_manifold_dof,
                    generic_row_normalized_manifold_dof_upper_bound=max(
                        0, coefficient_manifold_dof - blocks_per_axis
                    ),
                    identifiable_note=(
                        "Signed low-rank factors are clipped before row normalization; the active-set boundary "
                        "makes a single exact continuous identifiable-DOF count inappropriate."
                    ),
                    blocks_per_axis=blocks_per_axis,
                    block_count=block_count,
                    signed_factor_then_clamp=True,
                )
            )

        independent_matrix = expand_block_kernels(kernels, blocks_per_axis, block_size)
        independent_values = block_count * block_size
        independent_identifiable = independent_values - blocks_per_axis
        rows.append(
            candidate_row(
                example,
                "backbone",
                "independent_block_circulant",
                independent_matrix,
                independent_values * VALUE_BITS,
                independent_values,
                independent_identifiable,
                block_size=block_size,
                kernel_rank=max_rank,
                blocks_per_axis=blocks_per_axis,
                block_count=block_count,
            )
        )
        rows.append(
            candidate_row(
                example,
                "negative_control",
                "independent_plus_redundant_scale",
                independent_matrix,
                (independent_values + block_count) * VALUE_BITS,
                independent_values + block_count,
                independent_identifiable,
                scale_count=block_count,
                block_size=block_size,
                kernel_rank=max_rank,
                redundancy_note="Per-block scale is algebraically absorbable into each independent kernel.",
                blocks_per_axis=blocks_per_axis,
                block_count=block_count,
            )
        )

        for gamma in GAMMAS:
            for reoptimize in (False, True):
                gamma_kernels, stats = gamma_scaled_kernels(kernels, gamma, reoptimize)
                gamma_matrix = expand_block_kernels(gamma_kernels, blocks_per_axis, block_size)
                stored = block_size if gamma == 0.0 else rank1_values
                identifiable = block_size - 1 if gamma == 0.0 else rank1_identifiable
                rows.append(
                    candidate_row(
                        example,
                        "scale_landscape",
                        "block_scale_gamma_reoptimized" if reoptimize else "block_scale_gamma_fixed_kernel",
                        gamma_matrix,
                        stored * VALUE_BITS,
                        stored,
                        identifiable,
                        scale_count=0 if gamma == 0.0 else block_count,
                        block_size=block_size,
                        kernel_rank=1,
                        scale_gamma=gamma,
                        reoptimized_kernel=reoptimize,
                        blocks_per_axis=blocks_per_axis,
                        block_count=block_count,
                        **stats,
                    )
                )

        independent_fp16_metrics = evaluate_candidate(attention, independent_matrix)
        for bits in QUANT_BITS:
            for scheme in QUANT_SCHEMES:
                quantized_kernels, scale_count = quantize_kernels(
                    kernels, bits, scheme, blocks_per_axis
                )
                quantized_matrix = expand_block_kernels(
                    quantized_kernels, blocks_per_axis, block_size
                )
                parameter_bits = int(kernels.size * bits + scale_count * SCALE_BITS)
                if scheme in ("global", "block_row"):
                    functional_scale_count = 0
                else:
                    functional_scale_count = block_count - blocks_per_axis
                minimal_functional_bits = int(
                    kernels.size * bits + functional_scale_count * SCALE_BITS
                )
                row = candidate_row(
                    example,
                    "quantization",
                    f"quantized_independent_{scheme}",
                    quantized_matrix,
                    parameter_bits,
                    int(kernels.size),
                    None,
                    scale_count=scale_count,
                    block_size=block_size,
                    kernel_rank=max_rank,
                    quant_bits=bits,
                    scale_scheme=scheme,
                    blocks_per_axis=blocks_per_axis,
                    block_count=block_count,
                    quant_code_slots=int(kernels.size),
                    continuous_scale_dof_after_row_normalization=functional_scale_count,
                    minimal_functional_scale_count=functional_scale_count,
                    minimal_functional_parameter_bits=minimal_functional_bits,
                    minimal_functional_parameter_fraction_of_dense_fp16=float(
                        minimal_functional_bits / dense_bits
                    ),
                    quantization_payload_note=(
                        "parameter_bits includes every raw decoder scale. After explicit row normalization, a "
                        "global scale and one common scale per query block-row cancel; the minimal functional "
                        "payload is reported separately. Quantization codes are discrete slots, not continuous DOF."
                    ),
                )
                row["excess_normalized_mse_vs_fp16_independent"] = float(
                    row["normalized_mse"] - independent_fp16_metrics["normalized_mse"]
                )
                rows.append(row)


def fit_block_scale_backbone(
    target: np.ndarray,
    block_size: int,
    forbidden_cols: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    kernels, blocks_per_axis = extract_block_kernels(target, block_size)
    rank1_kernels, _ = gamma_scaled_kernels(kernels, 1.0, reoptimize_kernel=True)
    matrix = expand_block_kernels(rank1_kernels, blocks_per_axis, block_size)
    if forbidden_cols.size:
        matrix[:, forbidden_cols] = 0.0
    stored = kernels.shape[0] + block_size
    identifiable = (block_size - 1) + blocks_per_axis * (blocks_per_axis - 1)
    return matrix, stored, identifiable


def compensation_variant(
    example: Mapping[str, object],
    block_size: int,
    selected: Sequence[str],
    sink_k: int,
    residual_rank: int,
    sparse_k: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    attention = np.asarray(example["attention"], dtype=np.float64)
    n = int(attention.shape[0])
    index_width = int(math.ceil(math.log2(max(n, 2))))
    raw = np.zeros_like(attention)
    forbidden_cols = np.asarray([], dtype=np.int64)
    value_count = 0
    index_bits = 0
    local_params = 0

    if "sink" in selected:
        forbidden_cols = np.argsort(-attention.sum(axis=0))[: int(sink_k)]
        sink = np.zeros_like(attention)
        sink[:, forbidden_cols] = attention[:, forbidden_cols]
        raw += sink
        value_count += n * int(sink_k)
        index_bits += int(sink_k) * index_width

    residual = np.clip(attention - raw, 0.0, None)
    backbone, backbone_values, backbone_identifiable = fit_block_scale_backbone(
        residual, block_size, forbidden_cols
    )
    raw += backbone
    value_count += backbone_values

    if "lowrank" in selected:
        residual = np.clip(attention - raw, 0.0, None)
        lowrank = svd_relu_component(residual, residual_rank, forbidden_cols)
        raw += lowrank
        value_count += int(residual_rank) * (2 * n + 1)

    if "local" in selected:
        residual = np.clip(attention - raw, 0.0, None)
        local, local_params = local_cyclic_projection(
            residual, example["grid_shape"], radius=1
        )
        if forbidden_cols.size:
            local[:, forbidden_cols] = 0.0
        raw += local
        value_count += int(local_params)

    if "sparse" in selected:
        residual = np.clip(attention - raw, 0.0, None)
        sparse = topk_sparse_component(residual, sparse_k)
        if forbidden_cols.size:
            sparse[:, forbidden_cols] = 0.0
        raw += sparse
        value_count += n * int(sparse_k)
        index_bits += n * int(sparse_k) * index_width

    return raw, {
        "value_count": value_count,
        "index_bits": index_bits,
        "backbone_identifiable": backbone_identifiable,
        "local_params": int(local_params),
    }


def single_component_budget_variant(
    example: Mapping[str, object],
    block_size: int,
    method: str,
    amount: int,
    component_first: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Fit one residual family on either side of the same block-scale backbone.

    Reporting both orders makes the otherwise hidden greedy-fit advantage
    visible without selecting the better order in the summary.
    """

    attention = np.asarray(example["attention"], dtype=np.float64)
    n = int(attention.shape[0])
    index_width = int(math.ceil(math.log2(max(n, 2))))

    def fit_component(residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        forbidden = np.asarray([], dtype=np.int64)
        if method == "sink":
            forbidden = np.argsort(-residual.sum(axis=0))[: int(amount)]
            component = np.zeros_like(residual)
            component[:, forbidden] = residual[:, forbidden]
            return component, forbidden, n * int(amount), int(amount) * index_width
        if method == "lowrank":
            component = svd_relu_component(residual, int(amount), forbidden)
            return component, forbidden, int(amount) * (2 * n + 1), 0
        if method == "sparse":
            component = topk_sparse_component(residual, int(amount))
            return (
                component,
                forbidden,
                n * int(amount),
                n * int(amount) * index_width,
            )
        raise ValueError(f"unsupported budget component: {method}")

    if component_first:
        component, forbidden_cols, component_values, index_bits = fit_component(attention)
        residual = np.clip(attention - component, 0.0, None)
        backbone, backbone_values, backbone_identifiable = fit_block_scale_backbone(
            residual, block_size, forbidden_cols
        )
        raw = component + backbone
    else:
        backbone, backbone_values, backbone_identifiable = fit_block_scale_backbone(
            attention, block_size, np.asarray([], dtype=np.int64)
        )
        residual = np.clip(attention - backbone, 0.0, None)
        component, _, component_values, index_bits = fit_component(residual)
        raw = backbone + component

    return raw, {
        "value_count": int(backbone_values + component_values),
        "index_bits": int(index_bits),
        "backbone_identifiable": int(backbone_identifiable),
    }


def add_compensation_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    attention = np.asarray(example["attention"], dtype=np.float64)
    n = int(attention.shape[0])
    if n % COMMON_BLOCK_SIZE != 0:
        return
    sink_k, residual_rank, sparse_k = balanced_hyperparameters(n)
    for mask in range(1 << len(COMPENSATION_NAMES)):
        selected = [
            name
            for idx, name in enumerate(COMPENSATION_NAMES)
            if mask & (1 << idx)
        ]
        raw, budget = compensation_variant(
            example,
            COMMON_BLOCK_SIZE,
            selected,
            sink_k,
            residual_rank,
            sparse_k,
        )
        parameter_bits = int(budget["value_count"] * VALUE_BITS + budget["index_bits"])
        rows.append(
            candidate_row(
                example,
                "sequential_compensation_factorial",
                "block_scale" + ("+" + "+".join(selected) if selected else ""),
                raw,
                parameter_bits,
                budget["value_count"],
                None,
                budget["index_bits"],
                block_size=COMMON_BLOCK_SIZE,
                kernel_rank=1,
                compensation_mask=mask,
                compensation="+".join(selected) if selected else "none",
                sink_k=sink_k if "sink" in selected else 0,
                residual_rank=residual_rank if "lowrank" in selected else 0,
                sparse_k=sparse_k if "sparse" in selected else 0,
                local_radius=1 if "local" in selected else 0,
                oracle_support=bool("sink" in selected or "sparse" in selected),
                signed_factor_then_clamp=bool("lowrank" in selected),
                fit_order="sink>block_scale_refit>lowrank>local>sparse",
                comparison_design="fixed-dose sequential factorial; not equal-budget",
                interpretation_note=(
                    "Each subset is refitted in the named greedy order. Main effects include backbone refitting and "
                    "interactions include order/nonlinear-normalization effects; they are not orthogonal projections."
                ),
            )
        )

    kernels, _ = extract_block_kernels(attention, COMMON_BLOCK_SIZE)
    independent_bits = int(kernels.size * VALUE_BITS)
    base_values = kernels.shape[0] + COMMON_BLOCK_SIZE
    base_bits = base_values * VALUE_BITS
    index_width = int(math.ceil(math.log2(max(n, 2))))
    budget_capped_specs = {
        "sink": max(0, (independent_bits - base_bits) // (n * VALUE_BITS + index_width)),
        "lowrank": max(0, (independent_bits - base_bits) // ((2 * n + 1) * VALUE_BITS)),
        "sparse": max(
            0,
            (independent_bits - base_bits) // (n * (VALUE_BITS + index_width)),
        ),
    }
    for method, amount in budget_capped_specs.items():
        if amount <= 0:
            continue
        for component_first in (False, True):
            order = "component_first" if component_first else "backbone_first"
            raw, budget = single_component_budget_variant(
                example,
                COMMON_BLOCK_SIZE,
                method,
                int(amount),
                component_first,
            )
            parameter_bits = int(
                budget["value_count"] * VALUE_BITS + budget["index_bits"]
            )
            rows.append(
                candidate_row(
                    example,
                    "budget_capped_compensation",
                    f"budget_capped_{method}_{order}",
                    raw,
                    parameter_bits,
                    budget["value_count"],
                    None,
                    budget["index_bits"],
                    block_size=COMMON_BLOCK_SIZE,
                    kernel_rank=1,
                    budget_cap_bits=independent_bits,
                    budget_utilization=float(parameter_bits / independent_bits),
                    compensation_amount=int(amount),
                    component_fit_order=order,
                    oracle_support=method in ("sink", "sparse"),
                    signed_factor_then_clamp=method == "lowrank",
                    comparison_design=(
                        "integer-sized budget-capped target-fitted diagnostic; both component/backbone orders are "
                        "reported and neither is selected as a deployable winner"
                    ),
                )
            )


def load_true_v_anchors(examples: Sequence[Mapping[str, object]]) -> List[dict]:
    targets = {str(example["map_id"]): example for example in examples}
    best: Dict[Tuple[str, str, str], dict] = {}
    candidate_counts: Dict[Tuple[str, str, str], int] = {}
    allowed = {
        "grid_cyclic_bccb",
        "flat_block_circulant",
        "permuted_flat_block_circulant",
        "monarch_like_mask_proxy",
    }
    for path in TRUE_V_ANCHOR_PATHS:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                map_id = row.get("map_id", "")
                method = row.get("method", "")
                if (
                    map_id not in targets
                    or row.get("matrix_kind") != "attention"
                    or method not in allowed
                    or not row.get("output_relative_error")
                ):
                    continue
                if method == "flat_block_circulant" and row.get("permutation") != "identity":
                    continue
                block_size = str(row.get("block_size", ""))
                key = (map_id, method, block_size)
                candidate_counts[key] = candidate_counts.get(key, 0) + 1
                original_scope = str(row.get("scope", ""))
                item = {
                    "key": targets[map_id]["key"],
                    "label": targets[map_id]["label"],
                    "map_id": map_id,
                    "family": targets[map_id]["family"],
                    "source_input_id": targets[map_id]["source_input_id"],
                    "method": method,
                    "block_size": block_size,
                    "permutation": row.get("permutation"),
                    "relative_fro_error": float(row["relative_fro_error"]),
                    "true_v_output_relative_error": float(row["output_relative_error"]),
                    "params": int(row["params"]),
                    "dense_params": int(row["dense_params"]),
                    "compression_ratio": float(row["compression_ratio"]),
                    "original_scope": original_scope,
                    "original_note": row.get("note"),
                    "proxy_definition": row.get("proxy_definition"),
                    "selection_metric": "minimum relative_fro_error on the same target A",
                    "is_attention_only_rollout": "attention_only_rollout" in original_scope,
                    "scope_note": (
                        "Saved scalar from the original run; preserve original_scope because some ViT later-layer "
                        "values are attention-only rollout rather than an exact head output. V is not persisted and "
                        "new variants cannot be evaluated with true V."
                    ),
                }
                old = best.get(key)
                if old is None or item["relative_fro_error"] < old["relative_fro_error"]:
                    best[key] = item
    for key, item in best.items():
        item["selection_candidates"] = candidate_counts[key]
        item["target_selected_oracle"] = candidate_counts[key] > 1
    return list(best.values())


def load_hybrid_true_v_anchors(
    examples: Sequence[Mapping[str, object]],
) -> List[dict]:
    """Load saved true-V scalars for the earlier hybrid analysis framework.

    These rows deliberately retain the original nominal parameter accounting.
    The clipped/capped global component was not a strict low-rank stored matrix,
    so the values are comparison anchors rather than valid Pareto candidates.
    """

    targets = {str(example["map_id"]): example for example in examples}
    paths = [
        LOG_DIR / "structured_attention_visual_vit_examples_hybrid_20260704.json",
        LOG_DIR / "structured_attention_visual_qwen_examples_hybrid_20260704.json",
    ]
    allowed = ("hybrid_tiny", "hybrid_small", "hybrid_balanced", "hybrid_plus")
    anchors: List[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            map_id = str(item.get("map_id", ""))
            if map_id not in targets:
                continue
            for method in allowed:
                metric = item.get("metrics", {}).get(method)
                if not metric or metric.get("output_relative_error") is None:
                    continue
                anchors.append(
                    {
                        "key": targets[map_id]["key"],
                        "label": targets[map_id]["label"],
                        "map_id": map_id,
                        "family": targets[map_id]["family"],
                        "source_input_id": targets[map_id]["source_input_id"],
                        "method": method,
                        "block_size": "hybrid",
                        "permutation": "identity",
                        "relative_fro_error": float(metric["relative_fro_error"]),
                        "true_v_output_relative_error": float(
                            metric["output_relative_error"]
                        ),
                        "params": int(metric["nominal_budget_params"]),
                        "dense_params": int(metric["dense_params"]),
                        "compression_ratio": float(metric["nominal_budget_ratio"]),
                        "original_scope": "earlier_oracle_hybrid_framework",
                        "selection_metric": "fixed named hybrid configuration on the same target A",
                        "selection_candidates": 1,
                        "target_selected_oracle": True,
                        "is_attention_only_rollout": bool(
                            targets[map_id]["family"] == "vit" and ":l0:" not in map_id
                        ),
                        "scope_note": (
                            "Saved head-level A@V scalar from the earlier oracle hybrid framework. "
                            "Its parameter count is nominal only: the clipped/capped global component "
                            "was not a strict rank-constrained stored representation, and target-derived "
                            "sink/sparse supports are oracle."
                        ),
                    }
                )
    return anchors


def mark_pareto(rows: List[dict]) -> List[dict]:
    eligible_categories = {
        "baseline",
        "backbone",
        "quantization",
        "sequential_compensation_factorial",
        "budget_capped_compensation",
    }
    by_key: Dict[str, List[dict]] = {}
    for row in rows:
        row["pareto_nrmse"] = False
        if row["category"] in eligible_categories and int(row["parameter_bits"]) >= 0:
            by_key.setdefault(str(row["key"]), []).append(row)
    pareto: List[dict] = []
    for items in by_key.values():
        for candidate in items:
            bits = int(candidate["parameter_bits"])
            loss = float(candidate["normalized_mse"])
            dominated = any(
                int(other["parameter_bits"]) <= bits
                and float(other["normalized_mse"]) <= loss
                and (
                    int(other["parameter_bits"]) < bits
                    or float(other["normalized_mse"]) < loss
                )
                for other in items
                if other is not candidate
            )
            if not dominated:
                candidate["pareto_nrmse"] = True
                pareto.append(candidate)
    return pareto


def mean_float(rows: Iterable[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def source_balanced_mean(
    rows: Iterable[Mapping[str, object]], key: str
) -> float | None:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        grouped.setdefault(str(row["source_input_id"]), []).append(float(row[key]))
    source_means = [float(np.mean(values)) for values in grouped.values()]
    return float(np.mean(source_means)) if source_means else None


def summarize(rows: Sequence[dict], examples: Sequence[Mapping[str, object]]) -> dict:
    common = [row for row in rows if row.get("block_size") == COMMON_BLOCK_SIZE]
    shared = [row for row in common if row["method"] == "shared_kernel"]
    scaled = [row for row in common if row["method"] == "block_scale_rank1"]
    independent = [
        row for row in common if row["method"] == "independent_block_circulant"
    ]
    by_key_shared = {row["key"]: row for row in shared}
    by_key_scaled = {row["key"]: row for row in scaled}
    by_key_independent = {row["key"]: row for row in independent}
    scale_reductions = []
    recoverable_gap_fractions = []
    for key in sorted(set(by_key_shared) & set(by_key_scaled)):
        base_loss = float(by_key_shared[key]["normalized_mse"])
        scaled_loss = float(by_key_scaled[key]["normalized_mse"])
        scale_reductions.append((base_loss - scaled_loss) / max(base_loss, 1e-15))
        if key in by_key_independent:
            independent_loss = float(by_key_independent[key]["normalized_mse"])
            gap = base_loss - independent_loss
            if gap > 1e-15:
                recoverable_gap_fractions.append((base_loss - scaled_loss) / gap)

    quantization = []
    for bits in QUANT_BITS:
        for scheme in QUANT_SCHEMES:
            selected = [
                row
                for row in common
                if row.get("quant_bits") == bits and row.get("scale_scheme") == scheme
            ]
            quantization.append(
                {
                    "bits": bits,
                    "scale_scheme": scheme,
                    "examples": len(selected),
                    "mean_normalized_mse": mean_float(selected, "normalized_mse"),
                    "mean_excess_mse_vs_fp16_independent": mean_float(
                        selected, "excess_normalized_mse_vs_fp16_independent"
                    ),
                    "mean_parameter_fraction": mean_float(
                        selected, "parameter_fraction_of_dense_fp16"
                    ),
                    "mean_minimal_functional_parameter_fraction": mean_float(
                        selected, "minimal_functional_parameter_fraction_of_dense_fp16"
                    ),
                }
            )

    rank_curve = []
    for method in (
        "shared_kernel",
        "column_block_scale",
        "block_scale_rank1",
        "block_kernel_rank2",
        "block_kernel_rank3",
        "block_kernel_rank4",
        "independent_block_circulant",
    ):
        selected = [row for row in common if row["method"] == method]
        if selected:
            rank_curve.append(
                {
                    "method": method,
                    "examples": len(selected),
                    "mean_normalized_mse": mean_float(selected, "normalized_mse"),
                    "mean_relative_fro_error": mean_float(selected, "relative_fro_error"),
                    "mean_parameter_fraction": mean_float(
                        selected, "parameter_fraction_of_dense_fp16"
                    ),
                }
            )

    gamma_landscape = []
    for method in ("block_scale_gamma_fixed_kernel", "block_scale_gamma_reoptimized"):
        for gamma in GAMMAS:
            selected = [
                row
                for row in common
                if row["method"] == method and row.get("scale_gamma") == gamma
            ]
            gamma_landscape.append(
                {
                    "method": method,
                    "gamma": gamma,
                    "examples": len(selected),
                    "mean_normalized_mse": mean_float(selected, "normalized_mse"),
                    "mean_raw_relative_fro_error": mean_float(
                        selected, "raw_relative_fro_error"
                    ),
                }
            )

    compensation = []
    masks = sorted(
        {
            int(row["compensation_mask"])
            for row in common
            if row.get("compensation_mask") is not None
        }
    )
    for mask in masks:
        selected = [row for row in common if row.get("compensation_mask") == mask]
        compensation.append(
            {
                "mask": mask,
                "compensation": selected[0]["compensation"] if selected else None,
                "examples": len(selected),
                "mean_normalized_mse": mean_float(selected, "normalized_mse"),
                "mean_relative_fro_error": mean_float(selected, "relative_fro_error"),
                "mean_parameter_fraction": mean_float(
                    selected, "parameter_fraction_of_dense_fp16"
                ),
            }
        )

    sequential_pair_interactions = []
    component_bits = {name: 1 << index for index, name in enumerate(COMPENSATION_NAMES)}
    rows_by_key_mask = {
        (str(row["key"]), int(row["compensation_mask"])): row
        for row in common
        if row.get("compensation_mask") is not None
    }
    for left_index, left in enumerate(COMPENSATION_NAMES):
        for right in COMPENSATION_NAMES[left_index + 1 :]:
            per_example = []
            for example in examples:
                key = str(example["key"])
                required = (
                    (key, 0),
                    (key, component_bits[left]),
                    (key, component_bits[right]),
                    (key, component_bits[left] | component_bits[right]),
                )
                if all(item in rows_by_key_mask for item in required):
                    loss0, loss_left, loss_right, loss_pair = (
                        float(rows_by_key_mask[item]["normalized_mse"])
                        for item in required
                    )
                    per_example.append(loss_pair - loss_left - loss_right + loss0)
            sequential_pair_interactions.append(
                {
                    "left": left,
                    "right": right,
                    "examples": len(per_example),
                    "mean_loss_interaction": (
                        float(np.mean(per_example)) if per_example else None
                    ),
                    "interpretation": (
                        "Positive means less loss reduction than additive effects predict. The statistic includes "
                        "greedy refitting order, clipping, and row normalization."
                    ),
                }
            )

    budget_capped = []
    for method in (
        "block_scale_rank1",
        "independent_block_circulant",
        "budget_capped_sink_backbone_first",
        "budget_capped_sink_component_first",
        "budget_capped_lowrank_backbone_first",
        "budget_capped_lowrank_component_first",
        "budget_capped_sparse_backbone_first",
        "budget_capped_sparse_component_first",
    ):
        selected = [row for row in common if row["method"] == method]
        budget_capped.append(
            {
                "method": method,
                "examples": len(selected),
                "mean_relative_fro_error": mean_float(selected, "relative_fro_error"),
                "mean_normalized_mse": mean_float(selected, "normalized_mse"),
                "mean_parameter_fraction": mean_float(
                    selected, "parameter_fraction_of_dense_fp16"
                ),
                "mean_budget_utilization": mean_float(selected, "budget_utilization"),
                "min_budget_utilization": (
                    min(float(row["budget_utilization"]) for row in selected)
                    if selected and selected[0].get("budget_utilization") is not None
                    else None
                ),
                "max_budget_utilization": (
                    max(float(row["budget_utilization"]) for row in selected)
                    if selected and selected[0].get("budget_utilization") is not None
                    else None
                ),
            }
        )

    budget_envelope = []
    budgets = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.125, 0.20, 0.25, 0.50, 1.0)
    for budget in budgets:
        per_example = []
        winners = []
        per_source: Dict[str, List[float]] = {}
        for example in examples:
            candidates = [
                row
                for row in rows
                if row["key"] == example["key"]
                and int(row["parameter_bits"]) >= 0
                and float(row["parameter_fraction_of_dense_fp16"]) <= budget
                and row["category"]
                in {
                    "backbone",
                    "quantization",
                    "sequential_compensation_factorial",
                    "budget_capped_compensation",
                    "baseline",
                }
            ]
            if candidates:
                winner = min(candidates, key=lambda row: float(row["normalized_mse"]))
                winner_loss = float(winner["normalized_mse"])
                per_example.append(winner_loss)
                per_source.setdefault(str(example["source_input_id"]), []).append(
                    winner_loss
                )
                winners.append(str(winner["method"]))
        budget_envelope.append(
            {
                "parameter_fraction_budget": budget,
                "examples": len(per_example),
                "mean_best_normalized_mse": float(np.mean(per_example)) if per_example else None,
                "source_balanced_mean_best_normalized_mse": (
                    float(np.mean([np.mean(values) for values in per_source.values()]))
                    if per_source
                    else None
                ),
                "winner_counts": {
                    method: winners.count(method) for method in sorted(set(winners))
                },
            }
        )

    return {
        "examples": len(examples),
        "source_inputs": len({str(example["source_input_id"]) for example in examples}),
        "common_block_size": COMMON_BLOCK_SIZE,
        "common_block_scale": {
            "mean_shared_relative_fro_error": mean_float(shared, "relative_fro_error"),
            "source_balanced_shared_relative_fro_error": source_balanced_mean(
                shared, "relative_fro_error"
            ),
            "mean_block_scale_relative_fro_error": mean_float(scaled, "relative_fro_error"),
            "source_balanced_block_scale_relative_fro_error": source_balanced_mean(
                scaled, "relative_fro_error"
            ),
            "mean_independent_relative_fro_error": mean_float(
                independent, "relative_fro_error"
            ),
            "source_balanced_independent_relative_fro_error": source_balanced_mean(
                independent, "relative_fro_error"
            ),
            "mean_relative_normalized_mse_reduction_shared_to_scale": (
                float(np.mean(scale_reductions)) if scale_reductions else None
            ),
            "mean_fraction_of_shared_to_independent_mse_gap_recovered": (
                float(np.mean(recoverable_gap_fractions))
                if recoverable_gap_fractions
                else None
            ),
            "mean_scale_cv": mean_float(scaled, "scale_cv"),
            "mean_scale_parameter_fraction": mean_float(
                scaled, "parameter_fraction_of_dense_fp16"
            ),
            "mean_independent_parameter_fraction": mean_float(
                independent, "parameter_fraction_of_dense_fp16"
            ),
        },
        "common_block_scale_by_family": [
            {
                "family": family,
                "examples": sum(row["family"] == family for row in scaled),
                "mean_shared_relative_fro_error": mean_float(
                    [row for row in shared if row["family"] == family],
                    "relative_fro_error",
                ),
                "mean_block_scale_relative_fro_error": mean_float(
                    [row for row in scaled if row["family"] == family],
                    "relative_fro_error",
                ),
                "mean_independent_relative_fro_error": mean_float(
                    [row for row in independent if row["family"] == family],
                    "relative_fro_error",
                ),
            }
            for family in sorted({str(row["family"]) for row in scaled})
        ],
        "common_block_scale_by_source_input": [
            {
                "source_input_id": source_id,
                "maps": sum(row["source_input_id"] == source_id for row in scaled),
                "mean_shared_relative_fro_error": mean_float(
                    [row for row in shared if row["source_input_id"] == source_id],
                    "relative_fro_error",
                ),
                "mean_block_scale_relative_fro_error": mean_float(
                    [row for row in scaled if row["source_input_id"] == source_id],
                    "relative_fro_error",
                ),
                "mean_independent_relative_fro_error": mean_float(
                    [row for row in independent if row["source_input_id"] == source_id],
                    "relative_fro_error",
                ),
            }
            for source_id in sorted({str(row["source_input_id"]) for row in scaled})
        ],
        "rank_curve_b4": rank_curve,
        "scale_gamma_landscape_b4": gamma_landscape,
        "quantization_b4": quantization,
        "sequential_compensation_factorial_b4": compensation,
        "sequential_pair_interactions_b4": sequential_pair_interactions,
        "budget_capped_b4": budget_capped,
        "budget_envelope": budget_envelope,
        "scope_note": (
            "Eight hand-picked maps come from four source inputs (two ViT maps share one CIFAR image and four Qwen "
            "maps share one video); map-weighted means are not independent-sample generalization estimates. All new "
            "losses are per-target attention-space oracle fits. Except for explicitly rounded quantization "
            "scales/codes, parameter bits are nominal ideal-packed payload estimates: continuous factors are fitted "
            "in float64 but costed as FP16 and indices use ceil(log2(n)) packing. Router/predictor parameters, "
            "activation generation, decoder work, and latency are not included."
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=LOG_DIR / "compression_loss_landscape_20260712.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=LOG_DIR / "compression_loss_landscape_20260712.csv",
    )
    parser.add_argument(
        "--output-pareto-csv",
        type=Path,
        default=LOG_DIR / "compression_loss_pareto_20260712.csv",
    )
    parser.add_argument(
        "--output-anchor-csv",
        type=Path,
        default=LOG_DIR / "compression_loss_true_v_anchors_20260712.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    examples, source_files = load_examples()
    rows: List[dict] = []
    for example in examples:
        add_backbone_rows(example, rows)
        add_compensation_rows(example, rows)
    pareto = mark_pareto(rows)
    anchors = load_true_v_anchors(examples) + load_hybrid_true_v_anchors(examples)
    summary = summarize(rows, examples)
    all_source_files = list(source_files) + list(TRUE_V_ANCHOR_PATHS)
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": {
            "hybrid_attention_decomposition.py": sha256_file(
                ROOT / "scripts" / "hybrid_attention_decomposition.py"
            )
        },
        "source_sha256": {
            repo_relative(path): sha256_file(path) for path in all_source_files
        },
        "matrix_source_sha256": {
            repo_relative(path): sha256_file(path) for path in source_files
        },
        "anchor_source_sha256": {
            repo_relative(path): sha256_file(path) for path in TRUE_V_ANCHOR_PATHS
        },
        "constants": {
            "value_bits": VALUE_BITS,
            "scale_bits": SCALE_BITS,
            "common_block_size": COMMON_BLOCK_SIZE,
            "gammas": list(GAMMAS),
            "quant_bits": list(QUANT_BITS),
            "quant_schemes": list(QUANT_SCHEMES),
        },
        "method_notes": {
            "independent_scale_negative_control": (
                "A free scale multiplying each independently fitted block kernel is algebraically absorbable into that kernel; "
                "it adds stored scalars but cannot lower the representation loss."
            ),
            "block_scale": (
                "block_scale_rank1 is the rank-1 SVD of the matrix of independently projected block kernels: one shared "
                "circulant shape plus one amplitude per block."
            ),
            "attention_identifiability": (
                "Row normalization removes one common amplitude per query block row. Stored scalar counts and identifiable "
                "degrees of freedom are therefore both reported when available."
            ),
            "quantization": (
                "Unsigned uniform quantization of the independent non-negative block kernels. Each scale is rounded to "
                "FP16 before integer-code assignment and reconstruction, and its payload is included in parameter_bits."
            ),
            "loss_scope": (
                "New variants use attention-space NRMSE/KL/JS/TV only. Isotropic random-V expected squared output error "
                "energy ratio equals matrix NRMSE, but the expectation of a finite-dimensional error ratio need not; "
                "true V can differ, and saved true-V scalars are anchors only."
            ),
            "task_scope": (
                "No local model weights, V, logits, labels, or per-example task outputs are available. This probe cannot "
                "claim task-loss, denoising-loss, or end-to-end latency improvements."
            ),
        },
        "examples": [
            {
                key: value
                for key, value in example.items()
                if key != "attention"
            }
            for example in examples
        ],
        "rows": rows,
        "pareto_rows": pareto,
        "true_v_anchors": anchors,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_csv(args.output_pareto_csv, pareto)
    write_csv(args.output_anchor_csv, anchors)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
