#!/usr/bin/env python3
"""Probe Hessian-orthogonal compression and matched-rate combinations.

The probe has two layers:

1. Measure the error-direction Gram matrix of representative structured,
   pruning, and quantization codecs on the eight saved attention maps under
   both the Frobenius Hessian and a local KL Hessian.
2. Treat each attention row as a softmax-logit vector and study pruning plus
   retained-logit quantization under the damped softmax Fisher Hessian.  The
   experiment compares naive pruning, one-scale compensation, full
   OBS/Schur-complement compensation, bounded cross-null folded scales, and
   Hessian-loss-optimal folded scales with exact ideal-packed bit accounting.

All masks, codes, corrections, and scales are fitted to the evaluated target.
This is a controlled local-curvature diagnostic, not a model-weight, held-out
task-loss, accuracy, serialized-kernel, or latency result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import compression_loss_landscape_probe as core
import sparsity_repair_probe as sparse


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
VALUE_BITS = 16
SCALE_BITS = 16
BLOCK_SIZE = 4
PRUNE_FRACTIONS = tuple(round(0.05 * index, 2) for index in range(1, 17))
QUANT_BITS = tuple(range(2, 13))
MIXED_PRECISION_PARTS = 64
PAYLOAD_CAPS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
KL_HESSIAN_FLOORS = (1e-8, 1e-6, 1e-4)
FISHER_DAMPING_REL = 1e-4
SCALE_SEARCH_MULTIPLIER = 3.0
TAYLOR_RATIO_RANGE = (0.8, 1.25)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def fp16_round(values: np.ndarray | float) -> np.ndarray:
    return sparse.fp16_round(values)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-300)


def centered_fp16_logits(probability: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logits = np.log(np.maximum(np.asarray(probability, dtype=np.float64), 1e-12))
    logits -= logits.mean(axis=1, keepdims=True)
    logits = fp16_round(logits)
    return logits, softmax_rows(logits)


def mean_row_kl(target: np.ndarray, approx: np.ndarray) -> float:
    p = np.maximum(np.asarray(target, dtype=np.float64), 1e-300)
    q = np.maximum(np.asarray(approx, dtype=np.float64), 1e-300)
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def fisher_damping(probability: np.ndarray) -> np.ndarray:
    diagonal_mean = np.mean(probability * (1.0 - probability), axis=1, keepdims=True)
    return np.maximum(FISHER_DAMPING_REL * diagonal_mean, 1e-12)


def fisher_apply(
    probability: np.ndarray,
    delta: np.ndarray,
    damping: np.ndarray | float = 0.0,
) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    return p * d - p * np.sum(p * d, axis=1, keepdims=True) + damping * d


def fisher_inner(
    probability: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    damping: np.ndarray | float = 0.0,
) -> float:
    applied = fisher_apply(probability, right, damping)
    return float(np.mean(np.sum(np.asarray(left, dtype=np.float64) * applied, axis=1)))


def fisher_quadratic(
    probability: np.ndarray,
    delta: np.ndarray,
    damping: np.ndarray | float = 0.0,
) -> float:
    return 0.5 * fisher_inner(probability, delta, delta, damping)


def metric_cosine(left: np.ndarray, right: np.ndarray, weight: np.ndarray | None = None) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if weight is None:
        numerator = float(np.sum(a * b))
        left_norm = float(np.sum(a * a))
        right_norm = float(np.sum(b * b))
    else:
        w = np.asarray(weight, dtype=np.float64)
        numerator = float(np.sum(w * a * b))
        left_norm = float(np.sum(w * a * a))
        right_norm = float(np.sum(w * b * b))
    return numerator / max(math.sqrt(max(left_norm * right_norm, 0.0)), 1e-30)


def fisher_cosine(
    probability: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    damping: np.ndarray,
) -> float:
    numerator = fisher_inner(probability, left, right, damping)
    denominator = math.sqrt(
        max(
            fisher_inner(probability, left, left, damping)
            * fisher_inner(probability, right, right, damping),
            0.0,
        )
    )
    return numerator / max(denominator, 1e-30)


def representative_compression_errors(target: np.ndarray) -> Dict[str, np.ndarray]:
    attention = core.row_normalize_nonnegative(target)
    n = attention.shape[0]
    empty = np.asarray([], dtype=np.int64)

    structured_raw, _, _ = core.fit_block_scale_backbone(attention, BLOCK_SIZE, empty)
    structured = core.row_normalize_nonnegative(structured_raw)

    keep = min(n - 1, max(1, int(round(0.10 * n))))
    mask = sparse.row_topk_mask(attention, keep)
    kept = fp16_round(attention) * mask
    missing = np.clip(1.0 - kept.sum(axis=1, keepdims=True), 0.0, 1.0)
    pruned = core.row_normalize_nonnegative(
        kept + missing * sparse.uniform_complement_template(mask)
    )

    kernels, blocks_per_axis = core.extract_block_kernels(attention, BLOCK_SIZE)
    quantized_kernels, _ = core.quantize_kernels(
        kernels, 4, "global", blocks_per_axis
    )
    quantized = core.row_normalize_nonnegative(
        core.expand_block_kernels(quantized_kernels, blocks_per_axis, BLOCK_SIZE)
    )

    return {
        "structured": structured - attention,
        "pruning": pruned - attention,
        "quantization": quantized - attention,
    }


def error_gram_rows(examples: Sequence[Mapping[str, object]]) -> List[dict]:
    rows: List[dict] = []
    for example in examples:
        target = core.row_normalize_nonnegative(
            np.asarray(example["attention"], dtype=np.float64)
        )
        errors = representative_compression_errors(target)
        names = tuple(errors)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                common = {
                    "key": example["key"],
                    "label": example["label"],
                    "map_id": example["map_id"],
                    "family": example["family"],
                    "source_input_id": example["source_input_id"],
                    "left": left,
                    "right": right,
                }
                rows.append(
                    {
                        **common,
                        "metric": "frobenius_hessian",
                        "probability_floor": None,
                        "hessian_cosine": metric_cosine(errors[left], errors[right]),
                    }
                )
                for floor in KL_HESSIAN_FLOORS:
                    rows.append(
                        {
                            **common,
                            "metric": "local_kl_hessian",
                            "probability_floor": floor,
                            "hessian_cosine": metric_cosine(
                                errors[left],
                                errors[right],
                                1.0 / np.maximum(target, floor),
                            ),
                        }
                    )
    return rows


def hessian_sensitivity_mask(
    logits: np.ndarray,
    probability: np.ndarray,
    damping: np.ndarray,
    prune_fraction: float,
) -> np.ndarray:
    n = logits.shape[1]
    prune_count = min(n - 1, max(1, int(round(n * float(prune_fraction)))))
    diagonal = probability * (1.0 - probability) + damping
    score = logits * logits * diagonal
    order = np.argsort(score, axis=1)
    mask = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(mask, order[:, :prune_count], True, axis=1)
    return mask


def naive_pruning(logits: np.ndarray, pruned: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(logits, dtype=np.float64)
    delta[pruned] = -logits[pruned]
    return delta


def one_scale_compensated_pruning(
    logits: np.ndarray,
    probability: np.ndarray,
    pruned: np.ndarray,
    damping: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    delta = naive_pruning(logits, pruned)
    retained_direction = np.where(~pruned, logits, 0.0)
    numerator = -np.sum(
        retained_direction * fisher_apply(probability, delta, damping), axis=1
    )
    denominator = np.sum(
        retained_direction
        * fisher_apply(probability, retained_direction, damping),
        axis=1,
    )
    alpha = numerator / np.maximum(denominator, 1e-30)
    alpha = fp16_round(alpha)[:, None]
    return delta + alpha * retained_direction, alpha[:, 0]


def obs_compensated_pruning(
    logits: np.ndarray,
    probability: np.ndarray,
    pruned: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    """Exact retained-space correction for diag(p)-pp^T+lambda I.

    For each row, H_RR is diagonal minus rank one, so the Schur-complement
    correction is evaluated with the Sherman-Morrison identity rather than a
    dense inverse.
    """

    delta = naive_pruning(logits, pruned)
    for row in range(logits.shape[0]):
        retained = ~pruned[row]
        removed = pruned[row]
        if not retained.any() or not removed.any():
            continue
        p_retained = probability[row, retained]
        weighted_removed = float(
            probability[row, removed] @ logits[row, removed]
        )
        diagonal_inverse_times_p = p_retained / (
            p_retained + float(damping[row, 0])
        )
        denominator = max(
            1.0 - float(p_retained @ diagonal_inverse_times_p), 1e-12
        )
        delta[row, retained] = (
            -weighted_removed * diagonal_inverse_times_p / denominator
        )
    return delta


def signed_quant_codes(
    values: np.ndarray,
    retained: np.ndarray,
    bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax <= 0:
        raise ValueError("signed quantization needs at least two bits")
    magnitudes = np.where(retained, np.abs(values), 0.0)
    scales = fp16_round(magnitudes.max(axis=1, keepdims=True) / qmax)
    codes = np.rint(values / np.maximum(scales, 1e-15)).clip(-qmax, qmax)
    codes *= retained
    return codes.astype(np.float64), scales.astype(np.float64)


def folded_scale_quantization(
    logits: np.ndarray,
    probability: np.ndarray,
    prune_delta: np.ndarray,
    retained: np.ndarray,
    bits: int,
    damping: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    prequant = logits + prune_delta
    codes, max_scales = signed_quant_codes(prequant, retained, bits)
    selected_scales = max_scales.copy()
    boundary_hits = 0

    for row in range(logits.shape[0]):
        active = retained[row]
        if not active.any():
            selected_scales[row, 0] = 0.0
            continue
        scale_max = float(max_scales[row, 0])
        upper = SCALE_SEARCH_MULTIPLIER * scale_max
        code_direction = codes[row]
        retained_prequant = np.where(active, prequant[row], 0.0)

        if mode == "max_scale":
            scale = scale_max
        elif mode == "bounded_cross_null_scale":
            numerator = float(
                prune_delta[row]
                @ fisher_apply(
                    probability[row : row + 1],
                    retained_prequant[None, :],
                    damping[row : row + 1],
                )[0]
            )
            denominator = float(
                prune_delta[row]
                @ fisher_apply(
                    probability[row : row + 1],
                    code_direction[None, :],
                    damping[row : row + 1],
                )[0]
            )
            scale = numerator / denominator if abs(denominator) > 1e-30 else scale_max
        elif mode == "loss_optimal_scale":
            zero_scale_quant_error = -retained_prequant
            affine_base = prune_delta[row] + zero_scale_quant_error
            applied_codes = fisher_apply(
                probability[row : row + 1],
                code_direction[None, :],
                damping[row : row + 1],
            )[0]
            denominator = float(code_direction @ applied_codes)
            numerator = -float(affine_base @ applied_codes)
            scale = numerator / max(denominator, 1e-30)
        else:
            raise ValueError(f"unknown folded-scale mode: {mode}")

        unclipped = float(scale)
        scale = float(np.clip(scale, 0.0, upper))
        if abs(scale - unclipped) > 1e-12:
            boundary_hits += 1
        selected_scales[row, 0] = float(fp16_round(scale))

    decoded = prequant.copy()
    decoded[retained] = (codes * selected_scales)[retained]
    quant_delta = decoded - prequant
    return quant_delta, selected_scales[:, 0], boundary_hits


def per_row_damped_quadratic(
    probability: np.ndarray,
    delta: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    return 0.5 * np.sum(delta * fisher_apply(probability, delta, damping), axis=1)


def per_row_endpoint_kl(probability: np.ndarray, delta: np.ndarray) -> np.ndarray:
    baseline_logits = np.log(np.maximum(probability, 1e-300))
    baseline_logits -= baseline_logits.mean(axis=1, keepdims=True)
    decoded = softmax_rows(baseline_logits + delta)
    return np.sum(
        probability
        * np.log(np.maximum(probability, 1e-300) / np.maximum(decoded, 1e-300)),
        axis=1,
    )


def add_mixed_precision_quantization_rows(
    example: Mapping[str, object],
    rows: List[dict],
    logits: np.ndarray,
    probability: np.ndarray,
    damping: np.ndarray,
) -> None:
    """Add realizable row-wise q/(q+1)-bit single-quantization codecs.

    A row bitmap declares which rows receive the higher precision.  Candidate
    counts use up to sixty-four row-count increments, giving the single-method
    family a realizable candidate within one percent of each reported combo
    payload.
    Separate allocations optimize the local Hessian and exact endpoint KL.
    """

    n = logits.shape[1]
    dense_bits = n * n * VALUE_BITS
    retained = np.ones_like(logits, dtype=bool)
    zero = np.zeros_like(logits, dtype=np.float64)
    decoded: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for bits in QUANT_BITS:
        delta, _scales, _hits = folded_scale_quantization(
            logits,
            probability,
            zero,
            retained,
            bits,
            damping,
            "loss_optimal_scale",
        )
        decoded[bits] = (
            delta,
            np.stack(
                (
                    per_row_damped_quadratic(probability, delta, damping),
                    per_row_endpoint_kl(probability, delta),
                ),
                axis=1,
            ),
        )

    high_counts = sorted(
        {
            min(n - 1, max(1, int(round(n * part / MIXED_PRECISION_PARTS))))
            for part in range(1, MIXED_PRECISION_PARTS)
        }
    )
    for low_bits in QUANT_BITS[:-1]:
        high_bits = low_bits + 1
        low_delta, low_losses = decoded[low_bits]
        high_delta, high_losses = decoded[high_bits]
        for metric_index, allocation_metric in (
            (0, "damped_hessian"),
            (1, "endpoint_kl"),
        ):
            benefit = low_losses[:, metric_index] - high_losses[:, metric_index]
            order = np.argsort(-benefit)
            for high_count in high_counts:
                high_rows = order[:high_count]
                delta = low_delta.copy()
                delta[high_rows] = high_delta[high_rows]
                metrics = perturbation_metrics(probability, zero, delta, damping)
                # One row bitmap declares q+1 rows.  Both precisions reuse the
                # already counted per-row FP16 quantization scale.
                parameter_bits = (
                    n * n * low_bits
                    + high_count * n
                    + n * SCALE_BITS
                    + n
                )
                rows.append(
                    base_result_row(
                        example,
                        "single_quantization",
                        f"dense_mixed_q{low_bits}_q{high_bits}_{allocation_metric}_h{high_count}",
                        parameter_bits,
                        dense_bits,
                        metrics,
                        quant_bits=float(low_bits + high_count / n),
                        low_quant_bits=low_bits,
                        high_quant_bits=high_bits,
                        high_precision_rows=high_count,
                        bitwidth_bitmap_bits=n,
                        allocation_metric=allocation_metric,
                        prune_fraction=0.0,
                        retained_count=n,
                        scale_mode="loss_optimal_scale",
                        scale_count=n,
                        representation=(
                            "row-wise mixed q/(q+1) signed codes + row-bitwidth bitmap + one FP16 scale per row"
                        ),
                    )
                )


def support_payload_bits(n: int, retained_count: int) -> Tuple[int, str, int]:
    index_width = int(math.ceil(math.log2(max(n, 2))))
    bitmap_per_row = n
    coo_per_row = retained_count * index_width
    if bitmap_per_row <= coo_per_row:
        return n * bitmap_per_row, "bitmap", index_width
    return n * coo_per_row, "fixed_row_coo", index_width


def perturbation_metrics(
    probability: np.ndarray,
    prune_delta: np.ndarray,
    quant_delta: np.ndarray,
    damping: np.ndarray,
) -> Dict[str, float | bool]:
    total = prune_delta + quant_delta
    prune_self = fisher_quadratic(probability, prune_delta, damping)
    quant_self = fisher_quadratic(probability, quant_delta, damping)
    cross = fisher_inner(probability, prune_delta, quant_delta, damping)
    total_damped = fisher_quadratic(probability, total, damping)
    total_fisher = fisher_quadratic(probability, total, 0.0)
    rho = fisher_cosine(probability, prune_delta, quant_delta, damping)

    baseline_logits = np.log(np.maximum(probability, 1e-300))
    baseline_logits -= baseline_logits.mean(axis=1, keepdims=True)
    total_kl = mean_row_kl(probability, softmax_rows(baseline_logits + total))
    prune_kl = mean_row_kl(probability, softmax_rows(baseline_logits + prune_delta))
    quant_kl = mean_row_kl(probability, softmax_rows(baseline_logits + quant_delta))
    kl_interaction = total_kl - prune_kl - quant_kl
    ratio = total_kl / max(total_fisher, 1e-30)

    return {
        "mean_damped_hessian_quadratic": total_damped,
        "mean_fisher_quadratic": total_fisher,
        "mean_actual_kl": total_kl,
        "mean_prune_hessian_self": prune_self,
        "mean_quant_hessian_self": quant_self,
        "mean_hessian_cross": cross,
        "hessian_correlation": rho,
        "hessian_interaction_fraction": cross
        / max(prune_self + quant_self, 1e-30),
        "mean_prune_only_kl": prune_kl,
        "mean_fixed_quant_vector_only_kl": quant_kl,
        "mean_actual_kl_interaction": kl_interaction,
        "taylor_actual_to_quadratic_ratio": ratio,
        "inside_taylor_comfort_zone": bool(
            TAYLOR_RATIO_RANGE[0] <= ratio <= TAYLOR_RATIO_RANGE[1]
        ),
    }


def base_result_row(
    example: Mapping[str, object],
    category: str,
    method: str,
    parameter_bits: int,
    dense_bits: int,
    metrics: Mapping[str, object],
    **metadata: object,
) -> dict:
    return {
        "key": example["key"],
        "label": example["label"],
        "map_id": example["map_id"],
        "family": example["family"],
        "source_input_id": example["source_input_id"],
        "n": int(np.asarray(example["attention"]).shape[0]),
        "category": category,
        "method": method,
        "parameter_bits": int(parameter_bits),
        "dense_fp16_bits": int(dense_bits),
        "parameter_fraction_of_dense_fp16": float(parameter_bits / dense_bits),
        "compression_ratio_vs_dense_fp16": float(dense_bits / max(parameter_bits, 1)),
        "target_fitted_oracle": True,
        **metrics,
        **metadata,
    }


def add_logit_compression_rows(example: Mapping[str, object], rows: List[dict]) -> None:
    saved_probability = core.row_normalize_nonnegative(
        np.asarray(example["attention"], dtype=np.float64)
    )
    logits, probability = centered_fp16_logits(saved_probability)
    n = logits.shape[1]
    dense_bits = n * n * VALUE_BITS
    damping = fisher_damping(probability)
    all_retained = np.ones_like(logits, dtype=bool)
    zero = np.zeros_like(logits, dtype=np.float64)

    for bits in QUANT_BITS:
        for scale_mode in ("max_scale", "loss_optimal_scale"):
            quant_delta, scales, boundary_hits = folded_scale_quantization(
                logits,
                probability,
                zero,
                all_retained,
                bits,
                damping,
                scale_mode,
            )
            parameter_bits = n * n * bits + n * SCALE_BITS
            metrics = perturbation_metrics(probability, zero, quant_delta, damping)
            rows.append(
                base_result_row(
                    example,
                    "single_quantization",
                    f"dense_q{bits}_{scale_mode}",
                    parameter_bits,
                    dense_bits,
                    metrics,
                    quant_bits=bits,
                    prune_fraction=0.0,
                    retained_count=n,
                    scale_mode=scale_mode,
                    scale_count=n,
                    scale_boundary_hits=boundary_hits,
                    mean_abs_scale=float(np.mean(np.abs(scales))),
                    representation="dense signed codes + one FP16 scale per row",
                )
            )

    add_mixed_precision_quantization_rows(
        example, rows, logits, probability, damping
    )

    for prune_fraction in PRUNE_FRACTIONS:
        pruned = hessian_sensitivity_mask(
            logits, probability, damping, prune_fraction
        )
        retained = ~pruned
        retained_count = int(retained.sum(axis=1)[0])
        if not np.all(retained.sum(axis=1) == retained_count):
            raise AssertionError("fixed prune fraction must retain a fixed row count")
        support_bits, support_encoding, index_width = support_payload_bits(
            n, retained_count
        )

        naive_delta = naive_pruning(logits, pruned)
        scale_delta, scale_alpha = one_scale_compensated_pruning(
            logits, probability, pruned, damping
        )
        obs_delta = obs_compensated_pruning(logits, probability, pruned, damping)
        compensation = {
            "naive": (naive_delta, 0),
            "one_scale": (scale_delta, n),
            "obs": (obs_delta, retained_count * n),
        }

        for compensation_name, (prune_delta, correction_dof) in compensation.items():
            representable_logits = logits + prune_delta
            representable_logits[pruned] = 0.0
            representable_logits[retained] = fp16_round(
                representable_logits[retained]
            )
            representable_delta = representable_logits - logits
            metrics = perturbation_metrics(
                probability, representable_delta, zero, damping
            )
            parameter_bits = (
                n * retained_count * VALUE_BITS + support_bits
            )
            rows.append(
                base_result_row(
                    example,
                    "single_pruning",
                    f"{compensation_name}_prune_p{int(round(100 * prune_fraction)):02d}_fp16",
                    parameter_bits,
                    dense_bits,
                    metrics,
                    quant_bits=VALUE_BITS,
                    prune_fraction=prune_fraction,
                    retained_count=retained_count,
                    support_bits=support_bits,
                    support_encoding=support_encoding,
                    index_width=index_width,
                    compensation=compensation_name,
                    correction_dof=correction_dof,
                    correction_payload_bits=0,
                    correction_note=(
                        "Correction is folded into retained FP16 values; DOF is an encoder diagnostic, not extra payload."
                    ),
                    mean_abs_scale_alpha=(
                        float(np.mean(np.abs(scale_alpha)))
                        if compensation_name == "one_scale"
                        else None
                    ),
                )
            )

        combo_specs = (
            ("naive", naive_delta, "max_scale"),
            ("one_scale", scale_delta, "max_scale"),
            ("obs", obs_delta, "max_scale"),
            ("naive", naive_delta, "bounded_cross_null_scale"),
            ("naive", naive_delta, "loss_optimal_scale"),
        )
        for bits in QUANT_BITS:
            parameter_bits = (
                n * retained_count * bits + support_bits + n * SCALE_BITS
            )
            for compensation_name, prune_delta, scale_mode in combo_specs:
                quant_delta, scales, boundary_hits = folded_scale_quantization(
                    logits,
                    probability,
                    prune_delta,
                    retained,
                    bits,
                    damping,
                    scale_mode,
                )
                metrics = perturbation_metrics(
                    probability, prune_delta, quant_delta, damping
                )
                rows.append(
                    base_result_row(
                        example,
                        "combined_prune_quant",
                        f"{compensation_name}_prune_p{int(round(100 * prune_fraction)):02d}_q{bits}_{scale_mode}",
                        parameter_bits,
                        dense_bits,
                        metrics,
                        quant_bits=bits,
                        prune_fraction=prune_fraction,
                        retained_count=retained_count,
                        support_bits=support_bits,
                        support_encoding=support_encoding,
                        index_width=index_width,
                        compensation=compensation_name,
                        scale_mode=scale_mode,
                        scale_count=n,
                        scale_boundary_hits=boundary_hits,
                        mean_abs_scale=float(np.mean(np.abs(scales))),
                        correction_payload_bits=0,
                        representation=(
                            "fixed pruning support + signed retained codes + one folded FP16 scale per row"
                        ),
                    )
                )


def best_row(rows: Sequence[dict], metric: str) -> dict:
    return min(rows, key=lambda row: float(row[metric]))


def matched_rate_rows(rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    by_key: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_key[str(row["key"])].append(row)

    for key, candidates in by_key.items():
        dense_bits = int(candidates[0]["dense_fp16_bits"])
        source_input_id = str(candidates[0]["source_input_id"])
        single = [
            row
            for row in candidates
            if row["category"] in ("single_quantization", "single_pruning")
        ]
        combo = [
            row
            for row in candidates
            if row["category"] == "combined_prune_quant"
        ]
        for comparison_scope in ("all_candidates", "taylor_comfort_only"):
            if comparison_scope == "taylor_comfort_only":
                scoped_single = [
                    row for row in single if bool(row["inside_taylor_comfort_zone"])
                ]
                scoped_combo = [
                    row for row in combo if bool(row["inside_taylor_comfort_zone"])
                ]
            else:
                scoped_single = single
                scoped_combo = combo
            for cap in PAYLOAD_CAPS:
                cap_bits = int(cap * dense_bits)
                feasible_combo = [
                    row
                    for row in scoped_combo
                    if int(row["parameter_bits"]) <= cap_bits
                ]
                if not feasible_combo:
                    continue
                matchable_combo = []
                for combo_candidate in feasible_combo:
                    combo_bits = int(combo_candidate["parameter_bits"])
                    feasible_single = [
                        row
                        for row in scoped_single
                        if int(row["parameter_bits"]) <= combo_bits
                    ]
                    if not feasible_single:
                        continue
                    closest_single = max(
                        feasible_single, key=lambda row: int(row["parameter_bits"])
                    )
                    coverage = float(
                        int(closest_single["parameter_bits"]) / combo_bits
                    )
                    if coverage >= 0.99:
                        matchable_combo.append(
                            (
                                combo_candidate,
                                feasible_single,
                                closest_single,
                                coverage,
                            )
                        )
                if not matchable_combo:
                    continue
                for metric in (
                    "mean_damped_hessian_quadratic",
                    "mean_actual_kl",
                ):
                    (
                        combo_winner,
                        feasible_single,
                        closest_single,
                        coverage,
                    ) = min(
                        matchable_combo,
                        key=lambda item: float(item[0][metric]),
                    )
                    matched_bits = int(combo_winner["parameter_bits"])
                    single_winner = best_row(feasible_single, metric)
                    single_loss = float(single_winner[metric])
                    combo_loss = float(combo_winner[metric])
                    single_hessian = float(
                        single_winner["mean_damped_hessian_quadratic"]
                    )
                    combo_hessian = float(
                        combo_winner["mean_damped_hessian_quadratic"]
                    )
                    single_kl = float(single_winner["mean_actual_kl"])
                    combo_kl = float(combo_winner["mean_actual_kl"])
                    out.append(
                        {
                            "key": key,
                            "source_input_id": source_input_id,
                            "comparison_scope": comparison_scope,
                            "selection_metric": metric,
                            "payload_cap_fraction": cap,
                            "matched_parameter_bits": matched_bits,
                            "matched_payload_fraction": float(matched_bits / dense_bits),
                            "matched_compression_ratio": float(dense_bits / matched_bits),
                            "closest_single_parameter_bits": int(
                                closest_single["parameter_bits"]
                            ),
                            "single_rate_coverage": coverage,
                            "strict_rate_coverage_pass": True,
                            "single_method": single_winner["method"],
                            "single_category": single_winner["category"],
                            "single_parameter_bits": int(single_winner["parameter_bits"]),
                            "selected_single_rate_coverage": float(
                                int(single_winner["parameter_bits"]) / matched_bits
                            ),
                            "single_quant_bits": single_winner.get("quant_bits"),
                            "single_prune_fraction": single_winner.get(
                                "prune_fraction"
                            ),
                            "single_loss": single_loss,
                            "combo_method": combo_winner["method"],
                            "combo_parameter_bits": matched_bits,
                            "combo_quant_bits": combo_winner.get("quant_bits"),
                            "combo_prune_fraction": combo_winner.get(
                                "prune_fraction"
                            ),
                            "combo_compensation": combo_winner.get("compensation"),
                            "combo_scale_mode": combo_winner.get("scale_mode"),
                            "combo_loss": combo_loss,
                            "relative_combo_gain": float(
                                (single_loss - combo_loss) / max(single_loss, 1e-30)
                            ),
                            "combo_wins": bool(combo_loss < single_loss),
                            "single_damped_hessian_quadratic": single_hessian,
                            "combo_damped_hessian_quadratic": combo_hessian,
                            "same_selection_relative_hessian_gain": float(
                                (single_hessian - combo_hessian)
                                / max(single_hessian, 1e-30)
                            ),
                            "single_endpoint_kl": single_kl,
                            "combo_endpoint_kl": combo_kl,
                            "same_selection_relative_endpoint_kl_gain": float(
                                (single_kl - combo_kl) / max(single_kl, 1e-30)
                            ),
                            "single_inside_taylor_comfort_zone": bool(
                                single_winner["inside_taylor_comfort_zone"]
                            ),
                            "combo_inside_taylor_comfort_zone": bool(
                                combo_winner["inside_taylor_comfort_zone"]
                            ),
                            "combo_hessian_correlation": float(
                                combo_winner["hessian_correlation"]
                            ),
                        }
                    )
    return out


def source_balanced_mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_input_id"])].append(float(row[key]))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def aggregate_matched(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, float], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["comparison_scope"]),
                str(row["selection_metric"]),
                float(row["payload_cap_fraction"]),
            )
        ].append(row)
    out = []
    for (comparison_scope, metric, cap), selected in sorted(grouped.items()):
        single_methods = [str(row["single_method"]) for row in selected]
        combo_methods = [str(row["combo_method"]) for row in selected]
        out.append(
            {
                "comparison_scope": comparison_scope,
                "selection_metric": metric,
                "payload_cap_fraction": cap,
                "examples": len(selected),
                "source_inputs": len(
                    {str(row["source_input_id"]) for row in selected}
                ),
                "mean_matched_payload_fraction": float(
                    np.mean([row["matched_payload_fraction"] for row in selected])
                ),
                "mean_single_loss": float(np.mean([row["single_loss"] for row in selected])),
                "mean_combo_loss": float(np.mean([row["combo_loss"] for row in selected])),
                "mean_relative_combo_gain": float(
                    np.mean([row["relative_combo_gain"] for row in selected])
                ),
                "source_balanced_relative_combo_gain": source_balanced_mean(
                    selected, "relative_combo_gain"
                ),
                "mean_same_selection_relative_hessian_gain": float(
                    np.mean(
                        [row["same_selection_relative_hessian_gain"] for row in selected]
                    )
                ),
                "source_balanced_same_selection_relative_hessian_gain": source_balanced_mean(
                    selected, "same_selection_relative_hessian_gain"
                ),
                "mean_same_selection_relative_endpoint_kl_gain": float(
                    np.mean(
                        [
                            row["same_selection_relative_endpoint_kl_gain"]
                            for row in selected
                        ]
                    )
                ),
                "source_balanced_same_selection_relative_endpoint_kl_gain": source_balanced_mean(
                    selected, "same_selection_relative_endpoint_kl_gain"
                ),
                "combo_win_count": int(sum(bool(row["combo_wins"]) for row in selected)),
                "same_selection_hessian_win_count": int(
                    sum(
                        float(row["combo_damped_hessian_quadratic"])
                        < float(row["single_damped_hessian_quadratic"])
                        for row in selected
                    )
                ),
                "same_selection_endpoint_kl_win_count": int(
                    sum(
                        float(row["combo_endpoint_kl"])
                        < float(row["single_endpoint_kl"])
                        for row in selected
                    )
                ),
                "comfort_zone_count": int(
                    sum(bool(row["combo_inside_taylor_comfort_zone"]) for row in selected)
                ),
                "mean_abs_combo_hessian_correlation": float(
                    np.mean([abs(row["combo_hessian_correlation"]) for row in selected])
                ),
                "min_single_rate_coverage": float(
                    np.min([row["single_rate_coverage"] for row in selected])
                ),
                "mean_single_rate_coverage": float(
                    np.mean([row["single_rate_coverage"] for row in selected])
                ),
                "min_selected_single_rate_coverage": float(
                    np.min(
                        [row["selected_single_rate_coverage"] for row in selected]
                    )
                ),
                "mean_selected_single_rate_coverage": float(
                    np.mean(
                        [row["selected_single_rate_coverage"] for row in selected]
                    )
                ),
                "single_winner_counts": {
                    method: single_methods.count(method)
                    for method in sorted(set(single_methods))
                },
                "combo_winner_counts": {
                    method: combo_methods.count(method)
                    for method in sorted(set(combo_methods))
                },
            }
        )
    return out


def summarize_gram(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, float | None, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["metric"]),
                row["probability_floor"],
                str(row["left"]),
                str(row["right"]),
            )
        ].append(row)
    out = []
    for (metric, floor, left, right), selected in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        values = np.asarray([row["hessian_cosine"] for row in selected], dtype=np.float64)
        out.append(
            {
                "metric": metric,
                "probability_floor": floor,
                "left": left,
                "right": right,
                "examples": len(selected),
                "mean_hessian_cosine": float(values.mean()),
                "mean_abs_hessian_cosine": float(np.abs(values).mean()),
                "min_hessian_cosine": float(values.min()),
                "max_hessian_cosine": float(values.max()),
                "near_orthogonal_count_abs_le_0p1": int(np.sum(np.abs(values) <= 0.1)),
            }
        )
    return out


def summarize_compression(rows: Sequence[dict]) -> dict:
    combo = [row for row in rows if row["category"] == "combined_prune_quant"]
    variants = {}
    for label, predicate in {
        "naive_max": lambda row: row["compensation"] == "naive" and row["scale_mode"] == "max_scale",
        "one_scale_max": lambda row: row["compensation"] == "one_scale" and row["scale_mode"] == "max_scale",
        "obs_max": lambda row: row["compensation"] == "obs" and row["scale_mode"] == "max_scale",
        "naive_bounded_cross_null": lambda row: row["compensation"] == "naive" and row["scale_mode"] == "bounded_cross_null_scale",
        "naive_loss_optimal": lambda row: row["compensation"] == "naive" and row["scale_mode"] == "loss_optimal_scale",
    }.items():
        selected = [row for row in combo if predicate(row)]
        variants[label] = {
            "rows": len(selected),
            "mean_abs_hessian_correlation": float(
                np.mean([abs(row["hessian_correlation"]) for row in selected])
            ),
            "max_abs_hessian_correlation": float(
                np.max([abs(row["hessian_correlation"]) for row in selected])
            ),
            "mean_damped_hessian_quadratic": float(
                np.mean([row["mean_damped_hessian_quadratic"] for row in selected])
            ),
            "mean_actual_kl": float(np.mean([row["mean_actual_kl"] for row in selected])),
            "comfort_zone_fraction": float(
                np.mean([bool(row["inside_taylor_comfort_zone"]) for row in selected])
            ),
            "scale_boundary_hits": int(
                np.sum([int(row.get("scale_boundary_hits", 0)) for row in selected])
            ),
            "scale_count": int(
                np.sum([int(row.get("scale_count", 0)) for row in selected])
            ),
            "scale_boundary_hit_fraction": float(
                np.sum([int(row.get("scale_boundary_hits", 0)) for row in selected])
                / max(
                    np.sum([int(row.get("scale_count", 0)) for row in selected]),
                    1,
                )
            ),
        }
    return variants


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=LOG_DIR / "hessian_orthogonal_compression_20260712.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=LOG_DIR / "hessian_orthogonal_compression_20260712.csv",
    )
    parser.add_argument(
        "--output-gram-csv",
        type=Path,
        default=LOG_DIR / "hessian_compression_error_gram_20260712.csv",
    )
    parser.add_argument(
        "--output-matched-csv",
        type=Path,
        default=LOG_DIR / "hessian_matched_rate_20260712.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    examples, source_files = core.load_examples()
    gram = error_gram_rows(examples)
    rows: List[dict] = []
    for example in examples:
        add_logit_compression_rows(example, rows)
    matched = matched_rate_rows(rows)
    summary = {
        "examples": len(examples),
        "source_inputs": len({str(example["source_input_id"]) for example in examples}),
        "error_gram": summarize_gram(gram),
        "compression_variants": summarize_compression(rows),
        "matched_rate": aggregate_matched(matched),
        "scope_note": (
            "Eight target-fitted attention maps from four source inputs. The logit experiment uses a damped softmax "
            "Fisher Hessian and exact KL endpoint check. Masks/codes/corrections/scales are oracle-fitted; payload is "
            "ideal-packed and excludes encoder/Hessian construction, decoder compute, task loss, and latency."
        ),
    }
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
            "scripts/sparsity_repair_probe.py": sha256_file(
                ROOT / "scripts" / "sparsity_repair_probe.py"
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
            "scale_bits": SCALE_BITS,
            "block_size": BLOCK_SIZE,
            "prune_fractions": list(PRUNE_FRACTIONS),
            "quant_bits": list(QUANT_BITS),
            "mixed_precision_parts": MIXED_PRECISION_PARTS,
            "payload_caps": list(PAYLOAD_CAPS),
            "kl_hessian_floors": list(KL_HESSIAN_FLOORS),
            "fisher_damping_relative": FISHER_DAMPING_REL,
            "scale_search_multiplier": SCALE_SEARCH_MULTIPLIER,
            "taylor_ratio_range": list(TAYLOR_RATIO_RANGE),
        },
        "method_notes": {
            "orthogonality": (
                "Pairwise Hessian orthogonality is evaluated on compression errors delta=C(A)-A, not on positive fitted components."
            ),
            "obs": (
                "OBS correction solves H_RR d_R=-H_RP d_P. Therefore the corrected pruning perturbation is exactly "
                "H-orthogonal to any later perturbation supported only on retained coordinates before finite precision."
            ),
            "one_scale": (
                "One-scale compensation only projects the pruning residual off one retained-logit scale direction; it "
                "cannot guarantee orthogonality to arbitrary elementwise quantization error."
            ),
            "folded_scales": (
                "The bounded cross-null and loss-optimal scales replace the already counted FP16 row scale after fixing "
                "integer codes. Cross-null is only approximate after clipping to the representable search interval and "
                "FP16 scale rounding; its boundary-hit fraction and residual Hessian correlation are reported."
            ),
            "matched_rate": (
                "For each combo winner, the single-method envelope is restricted to no more than the combo's actual "
                "ideal-packed bits and must contain a realizable codec using at least 99% of that budget. Results are "
                "reported both over all candidates and after requiring both methods to pass the endpoint Taylor comfort check."
            ),
        },
        "examples": [
            {key: value for key, value in example.items() if key != "attention"}
            for example in examples
        ],
        "error_gram_rows": gram,
        "rows": rows,
        "matched_rate_rows": matched,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_csv, rows)
    write_csv(args.output_gram_csv, gram)
    write_csv(args.output_matched_csv, matched)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
