#!/usr/bin/env python3
"""Probe parameter-efficient protection and repair for sparse attention.

The experiment asks whether pruning/sparsity can use an analogue of a
quantization scale: retain a very small amount of amplitude or tail-state so
that cheap support/code payloads do not destroy the numerical range.

Three complementary diagnostics are evaluated on the eight saved attention
maps used by ``compression_loss_landscape_probe.py``:

* row-top-k pruning with probability-mass and column-prior tail repair;
* structured block pruning with one scalar per pruned block or block row;
* low-bit sparse residual/outlier codecs around the block-scale backbone,
  including fit-order controls, loss-aware folded scales, and error feedback.

All support, scales, priors, and codecs are fitted to the same target A.  The
results are attention-space representation diagnostics, not held-out routing,
model-weight pruning, task-loss, or latency measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import compression_loss_landscape_probe as core


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
VALUE_BITS = 16
BLOCK_SIZE = 4
ROW_KEEP_RATIOS = (0.05, 0.10, 0.20, 0.40)
BLOCK_KEEP_RATIOS = (0.05, 0.10, 0.25, 0.50)
SPARSE_CAPS = (0.05, 0.10, 0.15, 0.20, 0.25)
SPARSE_QUANT_BITS = (2, 3, 4)
SCALE_SCHEMES = ("global", "block_row", "per_row")
GAIN_GRID = np.linspace(0.0, 3.0, 61, dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def fp16_round(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    limit = float(np.finfo(np.float16).max)
    return np.clip(array, -limit, limit).astype(np.float16).astype(np.float64)


def row_topk_mask(matrix: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    used = min(max(int(k), 1), values.shape[1])
    indices = np.argpartition(values, -used, axis=1)[:, -used:]
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def uniform_complement_template(mask: np.ndarray) -> np.ndarray:
    complement = ~np.asarray(mask, dtype=bool)
    counts = complement.sum(axis=1, keepdims=True)
    return complement.astype(np.float64) / np.maximum(counts, 1)


def column_prior_template(
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    prior = np.clip(fp16_round(np.asarray(target, dtype=np.float64).mean(axis=0)), 0.0, None)
    if float(prior.sum()) <= 1e-15:
        prior = np.ones(target.shape[1], dtype=np.float64)
    template = np.zeros_like(target, dtype=np.float64)
    complement = ~mask
    for row in range(target.shape[0]):
        weights = prior * complement[row]
        denom = float(weights.sum())
        if denom <= 1e-15:
            weights = complement[row].astype(np.float64)
            denom = float(weights.sum())
        template[row] = weights / max(denom, 1e-15)
    return template, prior


def optimal_uniform_gate(
    target: np.ndarray,
    kept_probability: np.ndarray,
    tail_template: np.ndarray,
    row_group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    direction = tail_template - kept_probability
    gains = np.zeros((target.shape[0], 1), dtype=np.float64)
    for start in range(0, target.shape[0], row_group_size):
        stop = min(start + row_group_size, target.shape[0])
        residual = target[start:stop] - kept_probability[start:stop]
        delta = direction[start:stop]
        gain = float(np.sum(residual * delta) / max(np.sum(delta * delta), 1e-15))
        gains[start:stop] = np.clip(gain, 0.0, 1.0)
    gains = np.clip(fp16_round(gains), 0.0, 1.0)
    return (1.0 - gains) * kept_probability + gains * tail_template, gains


def retained_column_group_scale(
    target: np.ndarray,
    kept_probability: np.ndarray,
    group_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.asarray(kept_probability, dtype=np.float64).copy()
    scales: List[float] = []
    for start in range(0, target.shape[1], group_width):
        stop = min(start + group_width, target.shape[1])
        target_mass = float(target[:, start:stop].sum())
        kept_mass = float(kept_probability[:, start:stop].sum())
        scale = target_mass / kept_mass if kept_mass > 1e-15 else 1.0
        scale = float(fp16_round(scale))
        out[:, start:stop] *= scale
        scales.append(scale)
    return core.row_normalize_nonnegative(out), np.asarray(scales, dtype=np.float64)


def add_extra_sparse_with_budget(
    target: np.ndarray,
    base_mask: np.ndarray,
    repair_budget_bits: int,
    index_width: int,
) -> Tuple[np.ndarray, int, int]:
    # Variable-row repair entries use an ideal COO row+column index.
    bits_per_entry = VALUE_BITS + 2 * index_width
    count = min(int(repair_budget_bits // bits_per_entry), int((~base_mask).sum()))
    extra_mask = np.zeros_like(base_mask, dtype=bool)
    if count > 0:
        candidates = np.where(~base_mask, target, -np.inf).reshape(-1)
        indices = np.argpartition(candidates, -count)[-count:]
        extra_mask.reshape(-1)[indices] = True
    decoded = fp16_round(target) * (base_mask | extra_mask)
    return decoded, count, count * bits_per_entry


def quantized_kept_shape(
    target: np.ndarray,
    mask: np.ndarray,
    bits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept = np.asarray(target, dtype=np.float64) * mask
    kept_mass = kept.sum(axis=1, keepdims=True)
    shape = kept / np.maximum(kept_mass, 1e-15)
    row_max = shape.max(axis=1, keepdims=True)
    qmax = (1 << int(bits)) - 1
    codes = np.rint(shape / np.maximum(row_max, 1e-15) * qmax) * mask
    decoded_shape = core.row_normalize_nonnegative(codes)
    # The row maximum scale is a row-normalization gauge and need not be decoded.
    return decoded_shape, np.clip(fp16_round(kept_mass), 0.0, 1.0), codes


def block_topk_mask(
    target: np.ndarray,
    block_size: int,
    keep_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = target.shape[0]
    blocks_per_axis = n // block_size
    tensor = (
        target.reshape(blocks_per_axis, block_size, blocks_per_axis, block_size)
        .transpose(0, 2, 1, 3)
    )
    scores = tensor.sum(axis=(2, 3))
    keep = min(
        blocks_per_axis,
        max(1, int(round(blocks_per_axis * float(keep_ratio)))),
    )
    indices = np.argpartition(scores, -keep, axis=1)[:, -keep:]
    block_mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(block_mask, indices, True, axis=1)
    mask = (
        np.broadcast_to(
            block_mask[:, :, None, None],
            (blocks_per_axis, blocks_per_axis, block_size, block_size),
        )
        .transpose(0, 2, 1, 3)
        .reshape(n, n)
    )
    return mask, block_mask, scores


def block_mass_repair(
    target: np.ndarray,
    block_mask: np.ndarray,
    block_size: int,
    per_row: bool,
) -> np.ndarray:
    n = target.shape[0]
    blocks_per_axis = n // block_size
    out = fp16_round(target) * np.repeat(
        np.repeat(block_mask, block_size, axis=0), block_size, axis=1
    )
    for block_row in range(blocks_per_axis):
        rows = slice(block_row * block_size, (block_row + 1) * block_size)
        for block_col in range(blocks_per_axis):
            if block_mask[block_row, block_col]:
                continue
            cols = slice(block_col * block_size, (block_col + 1) * block_size)
            block = target[rows, cols]
            if per_row:
                masses = np.clip(fp16_round(block.sum(axis=1, keepdims=True)), 0.0, None)
                out[rows, cols] = masses / block_size
            else:
                mass = max(float(fp16_round(float(block.sum()))), 0.0)
                out[rows, cols] = mass / (block_size * block_size)
    return out


def add_extra_blocks_with_budget(
    target: np.ndarray,
    block_mask: np.ndarray,
    scores: np.ndarray,
    block_size: int,
    repair_budget_bits: int,
) -> Tuple[np.ndarray, int, int]:
    blocks_per_axis = block_mask.shape[0]
    axis_index_width = int(math.ceil(math.log2(max(blocks_per_axis, 2))))
    bits_per_block = block_size * block_size * VALUE_BITS + 2 * axis_index_width
    count = min(int(repair_budget_bits // bits_per_block), int((~block_mask).sum()))
    expanded = block_mask.copy()
    if count > 0:
        candidates = np.where(~block_mask, scores, -np.inf).reshape(-1)
        indices = np.argpartition(candidates, -count)[-count:]
        expanded.reshape(-1)[indices] = True
    mask = np.repeat(np.repeat(expanded, block_size, axis=0), block_size, axis=1)
    return fp16_round(target) * mask, count, count * bits_per_block


def sparse_scale_group_count(n: int, scheme: str) -> int:
    if scheme == "global":
        return 1
    if scheme == "block_row":
        return n // BLOCK_SIZE
    if scheme == "per_row":
        return n
    raise ValueError(f"unknown sparse scale scheme: {scheme}")


def quantize_sparse_values(
    sparse: np.ndarray,
    mask: np.ndarray,
    bits: int,
    scheme: str,
) -> Tuple[np.ndarray, int, np.ndarray]:
    n = sparse.shape[0]
    decoded = np.zeros_like(sparse, dtype=np.float64)
    group_scales: List[float] = []
    if scheme == "global":
        groups = [(0, n)]
    elif scheme == "block_row":
        groups = [(start, min(start + BLOCK_SIZE, n)) for start in range(0, n, BLOCK_SIZE)]
    elif scheme == "per_row":
        groups = [(row, row + 1) for row in range(n)]
    else:
        raise ValueError(f"unknown sparse scale scheme: {scheme}")

    for start, stop in groups:
        local_mask = mask[start:stop]
        values = sparse[start:stop][local_mask]
        if values.size == 0:
            scale = 0.0
            group_scales.append(scale)
            continue
        if bits == 0:
            scale = float(fp16_round(float(values.mean())))
            decoded[start:stop][local_mask] = scale
        else:
            qmax = (1 << int(bits)) - 1
            scale = float(fp16_round(float(values.max()) / qmax))
            if scale > 0.0:
                codes = np.rint(values / scale).clip(0, qmax)
                decoded[start:stop][local_mask] = codes * scale
        group_scales.append(scale)
    return decoded, len(groups), np.asarray(group_scales, dtype=np.float64)


def optimize_folded_group_gain(
    target: np.ndarray,
    backbone: np.ndarray,
    sparse: np.ndarray,
    scheme: str,
    bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = target.shape[0]
    if scheme == "global":
        groups = [(0, n)]
    elif scheme == "block_row":
        groups = [(start, min(start + BLOCK_SIZE, n)) for start in range(0, n, BLOCK_SIZE)]
    elif scheme == "per_row":
        groups = [(row, row + 1) for row in range(n)]
    else:
        raise ValueError(f"unknown gain scheme: {scheme}")
    adjusted = np.zeros_like(sparse, dtype=np.float64)
    gains: List[float] = []
    for start, stop in groups:
        base_part = backbone[start:stop]
        sparse_part = sparse[start:stop]
        if bits == VALUE_BITS:
            # The gain is folded into each stored FP16 sparse value, so every
            # candidate must be rounded again after scaling.
            sparse_candidates = fp16_round(
                GAIN_GRID[:, None, None] * sparse_part[None, :, :]
            )
        else:
            # Recover the fixed integer codes and replace the existing FP16
            # group scale with one representable folded scale.  No extra gain
            # scalar is stored or counted.
            if bits == 0:
                base_scale = float(np.max(sparse_part, initial=0.0))
                codes = (sparse_part > 0.0).astype(np.float64)
            else:
                qmax = (1 << int(bits)) - 1
                base_scale = float(np.max(sparse_part, initial=0.0)) / qmax
                base_scale = float(fp16_round(base_scale))
                codes = (
                    np.rint(sparse_part / base_scale).clip(0, qmax)
                    if base_scale > 0.0
                    else np.zeros_like(sparse_part)
                )
            folded_scales = fp16_round(base_scale * GAIN_GRID)
            sparse_candidates = codes[None, :, :] * folded_scales[:, None, None]
        candidates = base_part[None, :, :] + sparse_candidates
        candidates = np.clip(candidates, 0.0, None)
        candidates /= np.maximum(candidates.sum(axis=2, keepdims=True), 1e-12)
        losses = np.sum((candidates - target[start:stop][None, :, :]) ** 2, axis=(1, 2))
        best = int(np.argmin(losses))
        gain = float(fp16_round(float(GAIN_GRID[best])))
        gains.append(gain)
        adjusted[start:stop] = sparse_candidates[best]
    return adjusted, np.asarray(gains, dtype=np.float64)


def sparse_k_for_cap(
    n: int,
    cap_fraction: float,
    quant_bits: int,
    scale_count: int,
    stages: int,
) -> int:
    dense_bits = n * n * VALUE_BITS
    blocks = (n // BLOCK_SIZE) ** 2
    backbone_bits = (blocks + BLOCK_SIZE) * VALUE_BITS
    index_width = int(math.ceil(math.log2(max(n, 2))))
    available = int(cap_fraction * dense_bits) - backbone_bits - stages * scale_count * VALUE_BITS
    per_k_bits = stages * n * (index_width + quant_bits)
    return max(0, min(n, available // max(per_k_bits, 1)))


def build_sparse_stages(
    target: np.ndarray,
    cap_fraction: float,
    bits: int,
    scheme: str,
    order: str,
    stages: int,
    loss_aware: bool,
) -> Tuple[np.ndarray, Dict[str, object]] | None:
    n = target.shape[0]
    scale_count = 0 if bits == VALUE_BITS else sparse_scale_group_count(n, scheme)
    k = sparse_k_for_cap(n, cap_fraction, bits, scale_count, stages)
    if k <= 0:
        return None

    empty = np.asarray([], dtype=np.int64)
    components: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    if order == "backbone_first":
        backbone, backbone_values, _ = core.fit_block_scale_backbone(
            target, BLOCK_SIZE, empty
        )
        running = backbone.copy()
    elif order == "component_first":
        backbone = np.zeros_like(target, dtype=np.float64)
        backbone_values = (n // BLOCK_SIZE) ** 2 + BLOCK_SIZE
        running = np.zeros_like(target, dtype=np.float64)
    else:
        raise ValueError(f"unknown fit order: {order}")

    for _ in range(stages):
        residual = np.clip(target - running, 0.0, None)
        mask = row_topk_mask(residual, k)
        exact = fp16_round(residual) * mask
        if bits == VALUE_BITS:
            decoded = exact
        else:
            decoded, _, _ = quantize_sparse_values(exact, mask, bits, scheme)
        components.append(decoded)
        masks.append(mask)
        running += decoded

    sparse_sum = np.sum(components, axis=0)
    if order == "component_first":
        residual = np.clip(target - sparse_sum, 0.0, None)
        backbone, backbone_values, _ = core.fit_block_scale_backbone(
            residual, BLOCK_SIZE, empty
        )

    gain_vectors: List[np.ndarray] = []
    if loss_aware:
        # Gauss-Seidel coordinate updates make every stage see the already
        # adjusted earlier stages.  Because gain=1 is in GAIN_GRID and each
        # candidate is decoded with the declared payload, an update cannot
        # worsen the represented row-normalized loss.
        components = [component.copy() for component in components]
        for index in range(len(components)):
            other = backbone + np.sum(
                [value for j, value in enumerate(components) if j != index],
                axis=0,
            )
            gain_scheme = scheme if bits != VALUE_BITS else "global"
            component, gains = optimize_folded_group_gain(
                target, other, components[index], gain_scheme, bits
            )
            components[index] = component
            gain_vectors.append(gains)
        sparse_sum = np.sum(components, axis=0)

    index_width = int(math.ceil(math.log2(max(n, 2))))
    backbone_bits = backbone_values * VALUE_BITS
    index_bits = stages * n * k * index_width
    code_bits = stages * n * k * bits
    scale_bits = stages * scale_count * VALUE_BITS
    parameter_bits = backbone_bits + index_bits + code_bits + scale_bits
    unique_mask = np.logical_or.reduce(masks)
    return backbone + sparse_sum, {
        "k": k,
        "stages": stages,
        "backbone_bits": backbone_bits,
        "index_bits": index_bits,
        "code_bits": code_bits,
        "scale_bits": scale_bits,
        "scale_count": stages * scale_count,
        "parameter_bits": parameter_bits,
        "code_slots": stages * n * k,
        "unique_nonzero_fraction": float(unique_mask.mean()),
        "nominal_stage_nonzero_fraction": float(stages * k / n),
        "loss_aware_folded_scale": bool(loss_aware),
        "gain_min": (
            min(float(gain.min()) for gain in gain_vectors) if gain_vectors else None
        ),
        "gain_max": (
            max(float(gain.max()) for gain in gain_vectors) if gain_vectors else None
        ),
        "gain_mean": (
            float(np.mean(np.concatenate(gain_vectors))) if gain_vectors else None
        ),
    }


def sparse_budget_metadata(budget: Mapping[str, object]) -> Dict[str, object]:
    renamed: Dict[str, object] = {}
    for key, value in budget.items():
        if key in {
            "backbone_bits",
            "index_bits",
            "code_bits",
            "scale_bits",
            "scale_count",
            "parameter_bits",
        }:
            renamed[f"codec_{key}"] = value
        else:
            renamed[key] = value
    return renamed


def append_candidate(
    rows: List[dict],
    example: Mapping[str, object],
    category: str,
    method: str,
    raw: np.ndarray,
    parameter_bits: int,
    stored_slots: int,
    index_bits: int,
    base_bits: int,
    repair_bits: int,
    reference_nrmse: float | None,
    scale_count: int = 0,
    target_fitted_oracle: bool = True,
    **metadata: object,
) -> dict:
    row = core.candidate_row(
        example,
        category,
        method,
        raw,
        int(parameter_bits),
        int(stored_slots),
        None,
        int(index_bits),
        int(scale_count),
        target_fitted_oracle=target_fitted_oracle,
        **metadata,
    )
    dense_bits = int(row["n"]) ** 2 * VALUE_BITS
    row["raw_normalized_mse"] = float(row["raw_relative_fro_error"] ** 2)
    row["base_parameter_bits"] = int(base_bits)
    row["repair_parameter_bits"] = int(repair_bits)
    row["incremental_parameter_fraction_of_dense"] = float(repair_bits / dense_bits)
    row["reference_normalized_mse"] = reference_nrmse
    if reference_nrmse is not None:
        reduction = float(reference_nrmse - row["normalized_mse"])
        row["normalized_mse_reduction"] = reduction
        row["relative_normalized_mse_reduction"] = float(
            reduction / max(reference_nrmse, 1e-15)
        )
        row["mse_reduction_per_one_percent_dense_bits"] = (
            float(reduction / (100.0 * repair_bits / dense_bits))
            if repair_bits > 0
            else None
        )
    else:
        row["normalized_mse_reduction"] = None
        row["relative_normalized_mse_reduction"] = None
        row["mse_reduction_per_one_percent_dense_bits"] = None
    rows.append(row)
    return row


def add_row_pruning_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    target = core.row_normalize_nonnegative(np.asarray(example["attention"], dtype=np.float64))
    n = target.shape[0]
    index_width = int(math.ceil(math.log2(max(n, 2))))
    dense_bits = n * n * VALUE_BITS
    for keep_ratio in ROW_KEEP_RATIOS:
        k = min(n - 1, max(1, int(round(n * keep_ratio))))
        mask = row_topk_mask(target, k)
        kept = fp16_round(target) * mask
        base_bits = n * k * (VALUE_BITS + index_width)
        base_metrics = core.evaluate_candidate(target, kept)
        base_loss = float(base_metrics["normalized_mse"])
        common = {
            "keep_ratio": keep_ratio,
            "k": k,
            "index_width": index_width,
            "support_nnz": n * k,
            "support_fraction": float(k / n),
            "payload_scope": "dynamic attention values + ideal-packed column indices",
        }
        append_candidate(
            rows,
            example,
            "row_topk_pruning",
            "row_topk_renormalized",
            kept,
            base_bits,
            n * k,
            n * k * index_width,
            base_bits,
            0,
            None,
            **common,
        )

        row_scales = 1.0 + np.clip(1.0 - kept.sum(axis=1, keepdims=True), 0.0, 1.0)
        append_candidate(
            rows,
            example,
            "negative_control",
            "row_topk_plus_redundant_row_scale",
            kept * row_scales,
            base_bits + n * VALUE_BITS,
            n * k + n,
            n * k * index_width,
            base_bits,
            n * VALUE_BITS,
            base_loss,
            scale_count=n,
            redundancy_note="A common positive scale per row is removed exactly by row normalization.",
            **common,
        )

        kept_probability = core.row_normalize_nonnegative(kept)
        uniform_tail = uniform_complement_template(mask)
        global_mix, global_gains = optimal_uniform_gate(
            target, kept_probability, uniform_tail, n
        )
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_global_uniform_gate",
            global_mix,
            base_bits + VALUE_BITS,
            n * k + 1,
            n * k * index_width,
            base_bits,
            VALUE_BITS,
            base_loss,
            scale_count=1,
            gate_mean=float(global_gains.mean()),
            **common,
        )

        block_mix, block_gains = optimal_uniform_gate(
            target, kept_probability, uniform_tail, BLOCK_SIZE
        )
        block_gate_count = n // BLOCK_SIZE
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_query_block_uniform_gate",
            block_mix,
            base_bits + block_gate_count * VALUE_BITS,
            n * k + block_gate_count,
            n * k * index_width,
            base_bits,
            block_gate_count * VALUE_BITS,
            base_loss,
            scale_count=block_gate_count,
            gate_mean=float(block_gains.mean()),
            **common,
        )

        # Attention rows have known unit mass.  With decoded absolute kept
        # values, the missing mass is derived for free and spread implicitly.
        tail_mass = np.clip(1.0 - kept.sum(axis=1, keepdims=True), 0.0, 1.0)
        mass_uniform = kept + tail_mass * uniform_tail
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_mass_conserving_uniform_tail",
            mass_uniform,
            base_bits,
            n * k,
            n * k * index_width,
            base_bits,
            0,
            base_loss,
            deterministic_repair=True,
            repair_note="Missing row mass is 1-sum(decoded kept values); no stored scalar is required.",
            **common,
        )

        prior_tail, prior = column_prior_template(target, mask)
        mass_prior = kept + tail_mass * prior_tail
        prior_bits = n * VALUE_BITS
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_column_prior_tail",
            mass_prior,
            base_bits + prior_bits,
            n * k + n,
            n * k * index_width,
            base_bits,
            prior_bits,
            base_loss,
            scale_count=n,
            prior_l1=float(prior.sum()),
            repair_note="One global FP16 column prior distributes the derived missing mass on pruned support.",
            **common,
        )

        group_scaled, group_scales = retained_column_group_scale(
            target, kept_probability, BLOCK_SIZE
        )
        group_bits = group_scales.size * VALUE_BITS
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_retained_column_group_scale",
            group_scaled,
            base_bits + group_bits,
            n * k + group_scales.size,
            n * k * index_width,
            base_bits,
            group_bits,
            base_loss,
            scale_count=int(group_scales.size),
            repair_note="Scales only retained support; cannot restore deleted directions.",
            **common,
        )

        extra_sparse, extra_count, extra_bits = add_extra_sparse_with_budget(
            target, mask, prior_bits, index_width
        )
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_equal_prior_bits_extra_sparse",
            extra_sparse,
            base_bits + extra_bits,
            n * k + extra_count,
            n * k * index_width + extra_count * 2 * index_width,
            base_bits,
            extra_bits,
            base_loss,
            extra_entries=extra_count,
            repair_budget_cap_bits=prior_bits,
            **common,
        )

        residual = np.clip(target - kept, 0.0, None)
        rank1 = core.svd_relu_component(residual, 1, np.asarray([], dtype=np.int64))
        rank1_bits = (2 * n + 1) * VALUE_BITS
        append_candidate(
            rows,
            example,
            "row_topk_repair",
            "row_topk_plus_rank1_tail",
            kept + rank1,
            base_bits + rank1_bits,
            n * k + 2 * n + 1,
            n * k * index_width,
            base_bits,
            rank1_bits,
            base_loss,
            signed_factor_then_clamp=True,
            execution_note="ReLU after a signed rank-1 factor can destroy strict low-rank execution.",
            **common,
        )

        for bits in SPARSE_QUANT_BITS:
            decoded_shape, protected_mass, codes = quantized_kept_shape(
                target, mask, bits
            )
            quant_base_bits = n * k * (index_width + bits)
            quant_metrics = core.evaluate_candidate(target, decoded_shape)
            quant_loss = float(quant_metrics["normalized_mse"])
            qcommon = {
                **common,
                "quant_bits": bits,
                "code_slots": n * k,
                "row_scale_gauge_elided": True,
                "integer_code_max": int(codes.max(initial=0.0)),
            }
            append_candidate(
                rows,
                example,
                "quantized_row_topk",
                f"row_topk_q{bits}_shape_only",
                decoded_shape,
                quant_base_bits,
                n * k,
                n * k * index_width,
                quant_base_bits,
                0,
                None,
                **qcommon,
            )
            mass_bits = n * VALUE_BITS
            mass_only = protected_mass * decoded_shape
            append_candidate(
                rows,
                example,
                "quantized_row_topk",
                f"row_topk_q{bits}_mass_scale_only",
                mass_only,
                quant_base_bits + mass_bits,
                n * k + n,
                n * k * index_width,
                quant_base_bits,
                mass_bits,
                quant_loss,
                scale_count=n,
                amplitude_protected_raw_path=True,
                repair_note="Row mass improves the unnormalized A@V path but cancels if the sparse row is renormalized alone.",
                **qcommon,
            )
            mass_uniform_q = mass_only + (1.0 - protected_mass) * uniform_tail
            append_candidate(
                rows,
                example,
                "quantized_row_topk",
                f"row_topk_q{bits}_mass_uniform_tail",
                mass_uniform_q,
                quant_base_bits + mass_bits,
                n * k + n,
                n * k * index_width,
                quant_base_bits,
                mass_bits,
                quant_loss,
                scale_count=n,
                **qcommon,
            )
            prior_tail_q, _ = column_prior_template(target, mask)
            mass_prior_q = mass_only + (1.0 - protected_mass) * prior_tail_q
            prior_mass_bits = 2 * n * VALUE_BITS
            append_candidate(
                rows,
                example,
                "quantized_row_topk",
                f"row_topk_q{bits}_mass_prior_tail",
                mass_prior_q,
                quant_base_bits + prior_mass_bits,
                n * k + 2 * n,
                n * k * index_width,
                quant_base_bits,
                prior_mass_bits,
                quant_loss,
                scale_count=2 * n,
                **qcommon,
            )


def add_block_pruning_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    target = core.row_normalize_nonnegative(np.asarray(example["attention"], dtype=np.float64))
    n = target.shape[0]
    blocks_per_axis = n // BLOCK_SIZE
    col_index_width = int(math.ceil(math.log2(max(blocks_per_axis, 2))))
    for keep_ratio in BLOCK_KEEP_RATIOS:
        mask, block_mask, scores = block_topk_mask(target, BLOCK_SIZE, keep_ratio)
        kept_blocks = int(block_mask.sum())
        pruned_blocks = int((~block_mask).sum())
        base_bits = kept_blocks * (BLOCK_SIZE * BLOCK_SIZE * VALUE_BITS + col_index_width)
        kept = fp16_round(target) * mask
        base_loss = float(core.evaluate_candidate(target, kept)["normalized_mse"])
        common = {
            "keep_ratio": keep_ratio,
            "block_size": BLOCK_SIZE,
            "kept_blocks": kept_blocks,
            "pruned_blocks": pruned_blocks,
            "block_support_fraction": float(kept_blocks / block_mask.size),
        }
        append_candidate(
            rows,
            example,
            "block_pruning",
            "block_topk_renormalized",
            kept,
            base_bits,
            kept_blocks * BLOCK_SIZE * BLOCK_SIZE,
            kept_blocks * col_index_width,
            base_bits,
            0,
            None,
            **common,
        )
        block_mass_bits = pruned_blocks * VALUE_BITS
        block_mass = block_mass_repair(target, block_mask, BLOCK_SIZE, per_row=False)
        append_candidate(
            rows,
            example,
            "block_pruning_repair",
            "block_topk_plus_block_mass",
            block_mass,
            base_bits + block_mass_bits,
            kept_blocks * BLOCK_SIZE * BLOCK_SIZE + pruned_blocks,
            kept_blocks * col_index_width,
            base_bits,
            block_mass_bits,
            base_loss,
            scale_count=pruned_blocks,
            repair_note="One FP16 total-mass scalar reconstructs each pruned block uniformly.",
            **common,
        )
        row_mass_bits = pruned_blocks * BLOCK_SIZE * VALUE_BITS
        row_mass = block_mass_repair(target, block_mask, BLOCK_SIZE, per_row=True)
        append_candidate(
            rows,
            example,
            "block_pruning_repair",
            "block_topk_plus_row_mass",
            row_mass,
            base_bits + row_mass_bits,
            kept_blocks * BLOCK_SIZE * BLOCK_SIZE + pruned_blocks * BLOCK_SIZE,
            kept_blocks * col_index_width,
            base_bits,
            row_mass_bits,
            base_loss,
            scale_count=pruned_blocks * BLOCK_SIZE,
            repair_note="One FP16 mass per row within every pruned block; uniform only across block columns.",
            **common,
        )
        extra, extra_count, extra_bits = add_extra_blocks_with_budget(
            target, block_mask, scores, BLOCK_SIZE, block_mass_bits
        )
        append_candidate(
            rows,
            example,
            "block_pruning_repair",
            "block_topk_plus_equal_block_mass_bits_extra_blocks",
            extra,
            base_bits + extra_bits,
            (kept_blocks + extra_count) * BLOCK_SIZE * BLOCK_SIZE,
            kept_blocks * col_index_width + extra_count * 2 * col_index_width,
            base_bits,
            extra_bits,
            base_loss,
            extra_blocks=extra_count,
            repair_budget_cap_bits=block_mass_bits,
            **common,
        )


def add_sparse_codec_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    target = core.row_normalize_nonnegative(np.asarray(example["attention"], dtype=np.float64))
    n = target.shape[0]
    backbone, backbone_values, _ = core.fit_block_scale_backbone(
        target, BLOCK_SIZE, np.asarray([], dtype=np.int64)
    )
    backbone_bits = backbone_values * VALUE_BITS
    backbone_loss = float(core.evaluate_candidate(target, backbone)["normalized_mse"])
    for cap in SPARSE_CAPS:
        for order in ("backbone_first", "component_first"):
            exact = build_sparse_stages(
                target,
                cap,
                VALUE_BITS,
                "global",
                order,
                stages=1,
                loss_aware=False,
            )
            if exact is not None:
                raw, budget = exact
                append_candidate(
                    rows,
                    example,
                    "scale_protected_sparse_residual",
                    "sparse_residual_fp16_exact",
                    raw,
                    int(budget["parameter_bits"]),
                    backbone_values + int(budget["code_slots"]),
                    int(budget["index_bits"]),
                    backbone_bits,
                    int(budget["parameter_bits"]) - backbone_bits,
                    backbone_loss,
                    cap_fraction=cap,
                    order=order,
                    quant_bits=VALUE_BITS,
                    scale_scheme="none",
                    **sparse_budget_metadata(budget),
                )
                exact_aware = build_sparse_stages(
                    target,
                    cap,
                    VALUE_BITS,
                    "global",
                    order,
                    stages=1,
                    loss_aware=True,
                )
                assert exact_aware is not None
                raw, aware_budget = exact_aware
                append_candidate(
                    rows,
                    example,
                    "scale_protected_sparse_residual",
                    "sparse_residual_fp16_loss_aware_global_gain",
                    raw,
                    int(aware_budget["parameter_bits"]),
                    backbone_values + int(aware_budget["code_slots"]),
                    int(aware_budget["index_bits"]),
                    backbone_bits,
                    int(aware_budget["parameter_bits"]) - backbone_bits,
                    backbone_loss,
                    cap_fraction=cap,
                    order=order,
                    quant_bits=VALUE_BITS,
                    scale_scheme="global_gain_folded_into_values",
                    **sparse_budget_metadata(aware_budget),
                )

            for scheme in SCALE_SCHEMES:
                for loss_aware in (False, True):
                    built = build_sparse_stages(
                        target,
                        cap,
                        0,
                        scheme,
                        order,
                        stages=1,
                        loss_aware=loss_aware,
                    )
                    if built is None:
                        continue
                    raw, budget = built
                    method = (
                        f"sparse_residual_binary_{scheme}_"
                        + ("loss_aware" if loss_aware else "mean_scale")
                    )
                    append_candidate(
                        rows,
                        example,
                        "scale_protected_sparse_residual",
                        method,
                        raw,
                        int(budget["parameter_bits"]),
                        backbone_values
                        + int(budget["code_slots"])
                        + int(budget["scale_count"]),
                        int(budget["index_bits"]),
                        backbone_bits,
                        int(budget["parameter_bits"]) - backbone_bits,
                        backbone_loss,
                        scale_count=int(budget["scale_count"]),
                        cap_fraction=cap,
                        order=order,
                        quant_bits=0,
                        scale_scheme=scheme,
                        **sparse_budget_metadata(budget),
                    )

            for bits in SPARSE_QUANT_BITS:
                for scheme in SCALE_SCHEMES:
                    for loss_aware in (False, True):
                        built = build_sparse_stages(
                            target,
                            cap,
                            bits,
                            scheme,
                            order,
                            stages=1,
                            loss_aware=loss_aware,
                        )
                        if built is None:
                            continue
                        raw, budget = built
                        method = (
                            f"sparse_residual_q{bits}_{scheme}_"
                            + ("loss_aware" if loss_aware else "max_scale")
                        )
                        append_candidate(
                            rows,
                            example,
                            "scale_protected_sparse_residual",
                            method,
                            raw,
                            int(budget["parameter_bits"]),
                            backbone_values
                            + int(budget["code_slots"])
                            + int(budget["scale_count"]),
                            int(budget["index_bits"]),
                            backbone_bits,
                            int(budget["parameter_bits"]) - backbone_bits,
                            backbone_loss,
                            scale_count=int(budget["scale_count"]),
                            cap_fraction=cap,
                            order=order,
                            quant_bits=bits,
                            scale_scheme=scheme,
                            **sparse_budget_metadata(budget),
                        )

        # Selected multi-stage codecs keep the combinatorial sweep bounded.
        for bits, scheme, stages in (
            (2, "per_row", 2),
            (3, "global", 4),
            (4, "block_row", 2),
            (4, "block_row", 4),
        ):
            built = build_sparse_stages(
                target,
                cap,
                bits,
                scheme,
                "component_first",
                stages=stages,
                loss_aware=True,
            )
            if built is None:
                continue
            raw, budget = built
            append_candidate(
                rows,
                example,
                "sparse_error_feedback",
                f"sparse_error_feedback_q{bits}_{scheme}_t{stages}",
                raw,
                int(budget["parameter_bits"]),
                backbone_values
                + int(budget["code_slots"])
                + int(budget["scale_count"]),
                int(budget["index_bits"]),
                backbone_bits,
                int(budget["parameter_bits"]) - backbone_bits,
                backbone_loss,
                scale_count=int(budget["scale_count"]),
                cap_fraction=cap,
                order="component_first",
                quant_bits=bits,
                scale_scheme=scheme,
                **sparse_budget_metadata(budget),
            )


def add_reference_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    target = core.row_normalize_nonnegative(np.asarray(example["attention"], dtype=np.float64))
    n = target.shape[0]
    append_candidate(
        rows,
        example,
        "reference",
        "uniform_no_payload",
        np.full_like(target, 1.0 / n),
        0,
        0,
        0,
        0,
        0,
        None,
        target_fitted_oracle=False,
    )
    append_candidate(
        rows,
        example,
        "reference",
        "dense_fp16_equivalent",
        target,
        n * n * VALUE_BITS,
        n * n,
        0,
        n * n * VALUE_BITS,
        0,
        None,
    )
    backbone, values, _ = core.fit_block_scale_backbone(
        target, BLOCK_SIZE, np.asarray([], dtype=np.int64)
    )
    append_candidate(
        rows,
        example,
        "reference",
        "block_scale_rank1",
        backbone,
        values * VALUE_BITS,
        values,
        0,
        values * VALUE_BITS,
        0,
        None,
        block_size=BLOCK_SIZE,
    )


def mark_pareto(rows: List[dict]) -> List[dict]:
    by_key: Dict[str, List[dict]] = {}
    for row in rows:
        row["pareto_nrmse"] = False
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


def aggregate(rows: Sequence[dict]) -> dict:
    return {
        "examples": len({str(row["key"]) for row in rows}),
        "mean_normalized_mse": mean_float(rows, "normalized_mse"),
        "mean_raw_normalized_mse": mean_float(rows, "raw_normalized_mse"),
        "mean_parameter_fraction": mean_float(
            rows, "parameter_fraction_of_dense_fp16"
        ),
        "mean_repair_parameter_fraction": mean_float(
            rows, "incremental_parameter_fraction_of_dense"
        ),
        "mean_relative_normalized_mse_reduction": mean_float(
            rows, "relative_normalized_mse_reduction"
        ),
    }


def select_rows(
    rows: Sequence[dict],
    method: str,
    **filters: object,
) -> List[dict]:
    return [
        row
        for row in rows
        if row["method"] == method
        and all(row.get(key) == value for key, value in filters.items())
    ]


def calibrated_global_gain_summary(rows: Sequence[dict]) -> dict:
    # The loss-aware exact sparse row already stores the selected gain in the
    # values.  This summary measures whether one gain transfers across the four
    # source inputs instead of choosing it per target.
    selected = select_rows(
        rows,
        "sparse_residual_fp16_exact",
        cap_fraction=0.25,
        order="backbone_first",
    )
    if not selected:
        return {}
    # Reconstruct the target/backbone/sparse path from the saved examples to
    # avoid serializing matrices in each result row.
    examples, _ = core.load_examples()
    by_key = {str(example["key"]): example for example in examples}
    paths: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in selected:
        example = by_key[str(row["key"])]
        target = core.row_normalize_nonnegative(
            np.asarray(example["attention"], dtype=np.float64)
        )
        backbone, _, _ = core.fit_block_scale_backbone(
            target, BLOCK_SIZE, np.asarray([], dtype=np.int64)
        )
        residual = np.clip(target - backbone, 0.0, None)
        mask = row_topk_mask(residual, int(row["k"]))
        sparse = fp16_round(residual) * mask
        paths[str(row["key"])] = (target, backbone, sparse)

    source_ids = sorted({str(row["source_input_id"]) for row in selected})
    evaluations: List[dict] = []
    for held_out in source_ids:
        training = [row for row in selected if row["source_input_id"] != held_out]
        held = [row for row in selected if row["source_input_id"] == held_out]
        losses = []
        for gain in GAIN_GRID:
            source_means = []
            for source in sorted({str(row["source_input_id"]) for row in training}):
                local = []
                for row in training:
                    if row["source_input_id"] != source:
                        continue
                    target, backbone, sparse = paths[str(row["key"])]
                    local.append(
                        core.evaluate_candidate(target, backbone + gain * sparse)[
                            "normalized_mse"
                        ]
                    )
                source_means.append(float(np.mean(local)))
            losses.append(float(np.mean(source_means)))
        chosen = float(GAIN_GRID[int(np.argmin(losses))])
        chosen = float(fp16_round(chosen))
        for row in held:
            target, backbone, sparse = paths[str(row["key"])]
            metrics = core.evaluate_candidate(target, backbone + chosen * sparse)
            evaluations.append(
                {
                    "key": row["key"],
                    "source_input_id": held_out,
                    "calibrated_gain": chosen,
                    "normalized_mse": metrics["normalized_mse"],
                }
            )
    return {
        "held_out_source_inputs": len(source_ids),
        "rows": evaluations,
        "mean_normalized_mse": mean_float(evaluations, "normalized_mse"),
        "mean_calibrated_gain": mean_float(evaluations, "calibrated_gain"),
        "gain_min": min(float(row["calibrated_gain"]) for row in evaluations),
        "gain_max": max(float(row["calibrated_gain"]) for row in evaluations),
        "note": (
            "Four-source-input leave-one-source-out calibration; support remains target-fitted, so this only tests gain transfer."
        ),
    }


def summarize(rows: Sequence[dict], examples: Sequence[Mapping[str, object]]) -> dict:
    row_methods = (
        "row_topk_renormalized",
        "row_topk_plus_global_uniform_gate",
        "row_topk_plus_query_block_uniform_gate",
        "row_topk_plus_mass_conserving_uniform_tail",
        "row_topk_plus_column_prior_tail",
        "row_topk_plus_equal_prior_bits_extra_sparse",
        "row_topk_retained_column_group_scale",
    )
    row_pruning = []
    for keep_ratio in ROW_KEEP_RATIOS:
        for method in row_methods:
            selected = select_rows(rows, method, keep_ratio=keep_ratio)
            if selected:
                row_pruning.append(
                    {"keep_ratio": keep_ratio, "method": method, **aggregate(selected)}
                )

    quantized = []
    for bits in SPARSE_QUANT_BITS:
        for suffix in ("shape_only", "mass_scale_only", "mass_uniform_tail", "mass_prior_tail"):
            method = f"row_topk_q{bits}_{suffix}"
            selected = select_rows(rows, method, keep_ratio=0.10)
            if selected:
                quantized.append(
                    {"quant_bits": bits, "method": method, **aggregate(selected)}
                )

    block = []
    for keep_ratio in BLOCK_KEEP_RATIOS:
        for method in (
            "block_topk_renormalized",
            "block_topk_plus_block_mass",
            "block_topk_plus_row_mass",
            "block_topk_plus_equal_block_mass_bits_extra_blocks",
        ):
            selected = select_rows(rows, method, keep_ratio=keep_ratio)
            if selected:
                block.append(
                    {"keep_ratio": keep_ratio, "method": method, **aggregate(selected)}
                )

    sparse = []
    selected_methods = (
        "sparse_residual_fp16_exact",
        "sparse_residual_fp16_loss_aware_global_gain",
        "sparse_residual_binary_per_row_loss_aware",
        "sparse_residual_q2_per_row_loss_aware",
        "sparse_residual_q3_per_row_loss_aware",
        "sparse_residual_q4_block_row_loss_aware",
        "sparse_residual_q4_per_row_loss_aware",
        "sparse_error_feedback_q4_block_row_t4",
    )
    for cap in SPARSE_CAPS:
        for order in ("backbone_first", "component_first"):
            for method in selected_methods:
                selected = select_rows(rows, method, cap_fraction=cap, order=order)
                if selected:
                    sparse.append(
                        {
                            "cap_fraction": cap,
                            "order": order,
                            "method": method,
                            "mean_k": mean_float(selected, "k"),
                            **aggregate(selected),
                        }
                    )

    budgets = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0)
    envelope = []
    for budget in budgets:
        losses = []
        winners: List[str] = []
        per_source: Dict[str, List[float]] = {}
        for example in examples:
            candidates = [
                row
                for row in rows
                if row["key"] == example["key"]
                and float(row["parameter_fraction_of_dense_fp16"]) <= budget
            ]
            if not candidates:
                continue
            winner = min(candidates, key=lambda row: float(row["normalized_mse"]))
            loss = float(winner["normalized_mse"])
            losses.append(loss)
            winners.append(str(winner["method"]))
            per_source.setdefault(str(example["source_input_id"]), []).append(loss)
        envelope.append(
            {
                "parameter_fraction_budget": budget,
                "examples": len(losses),
                "mean_best_normalized_mse": float(np.mean(losses)) if losses else None,
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

    negative = [
        row
        for row in rows
        if row["method"] == "row_topk_plus_redundant_row_scale"
    ]
    return {
        "examples": len(examples),
        "source_inputs": len({str(example["source_input_id"]) for example in examples}),
        "row_pruning": row_pruning,
        "quantized_row_pruning_keep10": quantized,
        "block_pruning": block,
        "scale_protected_sparse": sparse,
        "calibrated_global_gain": calibrated_global_gain_summary(rows),
        "budget_envelope": envelope,
        "negative_control_max_loss_difference": max(
            abs(float(row["normalized_mse_reduction"])) for row in negative
        ),
        "scope_note": (
            "Eight hand-picked attention maps from four source inputs. Supports and all per-target scales/priors are fitted "
            "on the evaluated A. Payloads are ideal-packed dynamic representation bits; router, selection, decoder compute, "
            "and latency are excluded."
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
        default=LOG_DIR / "sparsity_repair_probe_20260712.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=LOG_DIR / "sparsity_repair_probe_20260712.csv",
    )
    parser.add_argument(
        "--output-pareto-csv",
        type=Path,
        default=LOG_DIR / "sparsity_repair_pareto_20260712.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    examples, source_files = core.load_examples()
    rows: List[dict] = []
    for example in examples:
        add_reference_rows(example, rows)
        add_row_pruning_rows(example, rows)
        add_block_pruning_rows(example, rows)
        add_sparse_codec_rows(example, rows)
    pareto = mark_pareto(rows)
    summary = summarize(rows, examples)
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": {
            "scripts/compression_loss_landscape_probe.py": sha256_file(
                ROOT / "scripts" / "compression_loss_landscape_probe.py"
            ),
            "scripts/hybrid_attention_decomposition.py": sha256_file(
                ROOT / "scripts" / "hybrid_attention_decomposition.py"
            ),
        },
        "source_sha256": {
            repo_relative(path): sha256_file(path) for path in source_files
        },
        "constants": {
            "value_bits": VALUE_BITS,
            "block_size": BLOCK_SIZE,
            "row_keep_ratios": list(ROW_KEEP_RATIOS),
            "block_keep_ratios": list(BLOCK_KEEP_RATIOS),
            "sparse_caps": list(SPARSE_CAPS),
            "sparse_quant_bits": list(SPARSE_QUANT_BITS),
            "scale_schemes": list(SCALE_SCHEMES),
            "gain_grid": GAIN_GRID.tolist(),
        },
        "method_notes": {
            "mass_protection": (
                "For exact post-softmax top-k values, missing row mass is derived as 1-sum(kept), so a uniform tail repair "
                "requires no stored scalar. Quantized shape codes need one FP16 row-mass value to protect amplitude."
            ),
            "scale_protected_sparse": (
                "Low-bit sparse residual codes use global/query-block/per-row FP16 scales. Loss-aware gains are folded into "
                "the existing scale/value payload and therefore add no bits."
            ),
            "direction_limit": (
                "Scaling retained support repairs radial/amplitude error only. Uniform/prior tails, extra sparse entries, "
                "or the structured backbone are required to restore deleted directions."
            ),
            "execution": (
                "Uniform and global-prior tails are dense if materialized, but can be applied implicitly using global/block "
                "reductions minus kept-support corrections. No kernel latency is measured here."
            ),
            "deployment_scope": (
                "Every support and per-target repair is fitted to A. Dynamic-attention deployment requires a router/online "
                "encoder; static weight pruning would instead calibrate or learn channel/block statistics offline."
            ),
        },
        "examples": [
            {key: value for key, value in example.items() if key != "attention"}
            for example in examples
        ],
        "rows": rows,
        "pareto_rows": pareto,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_csv, rows)
    write_csv(args.output_pareto_csv, pareto)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
