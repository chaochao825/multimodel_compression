#!/usr/bin/env python3
"""Probe activation-shaped PTQ with low-rank and block-sparse corrections."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_ints(text: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    lower = 0 if allow_zero else 1
    if not values or any(value < lower for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_floats(text: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    lower = 0.0 if allow_zero else torch.finfo(torch.float32).tiny
    if not values or any(value < lower for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative floats")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=lambda value: parse_ints(value, allow_zero=True), default=(0, 24))
    parser.add_argument(
        "--projection",
        choices=("down",),
        default="down",
        help="MVP evaluates the linear post-GELU down projection only.",
    )
    parser.add_argument("--bits", type=parse_ints, default=(4, 8))
    parser.add_argument("--ranks", type=parse_ints, default=(8, 16))
    parser.add_argument(
        "--clip-multipliers",
        type=parse_floats,
        default=(0.55, 0.65, 0.75, 0.85, 1.0),
    )
    parser.add_argument(
        "--sparse-ratios",
        type=lambda value: parse_floats(value, allow_zero=True),
        default=(0.01, 0.02),
    )
    parser.add_argument("--block-out", type=int, default=64)
    parser.add_argument("--block-in", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--alternating-iterations", type=int, default=2)
    parser.add_argument(
        "--shaping-strengths",
        type=lambda value: parse_floats(value, allow_zero=True),
        default=(1.0,),
        help="Relaxation factors beta for Q(W - beta*L - S).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def symmetric_channel_quantize(weight: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    if bits < 2 or clip <= 0.0 or clip > 1.0:
        raise ValueError("bits must be >= 2 and clip must be in (0, 1]")
    limit = float(2 ** (bits - 1) - 1)
    bound = weight.abs().amax(dim=1, keepdim=True) * clip
    scale = (bound / limit).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.round((weight / scale).clamp(-limit, limit)) * scale


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((estimate.float() - reference.float()).norm() / denominator)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten()
    right_flat = right.float().flatten()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) <= torch.finfo(torch.float32).tiny:
        return 0.0
    return float(torch.dot(left_flat, right_flat) / denominator)


def truncated_basis(
    matrix: torch.Tensor, rank: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if matrix.ndim != 2 or rank <= 0:
        raise ValueError("matrix must be 2D and rank positive")
    used = min(rank, matrix.shape[0], matrix.shape[1])
    q = min(max(used + 4, used), min(matrix.shape))
    torch.manual_seed(seed)
    u, singular, v = torch.pca_lowrank(matrix.float(), q=q, center=False, niter=2)
    total = float(matrix.float().square().sum())
    energy = 1.0 if total <= 1e-30 else float(singular[:used].square().sum()) / total
    return u[:, :used], singular[:used], v[:, :used], energy


def fit_activation_defect_lr(
    inputs: torch.Tensor,
    defect: torch.Tensor,
    rank: int,
    ridge: float,
    seed: int,
) -> dict[str, torch.Tensor | float]:
    if inputs.shape[0] != defect.shape[0]:
        raise ValueError("input and defect row counts differ")
    left, singular, output_basis, oracle_energy = truncated_basis(defect, rank, seed)
    coefficients = left * singular
    gram = inputs @ inputs.T
    regularization = ridge * float(gram.diagonal().mean().clamp_min(1e-12))
    gram.diagonal().add_(regularization)
    dual = torch.linalg.solve(gram, coefficients)
    input_factor = inputs.T @ dual
    return {
        "input_factor": input_factor,
        "output_basis": output_basis,
        "calibration_oracle_energy": oracle_energy,
    }


def apply_lr(
    inputs: torch.Tensor, input_factor: torch.Tensor, output_basis: torch.Tensor
) -> torch.Tensor:
    return (inputs @ input_factor) @ output_basis.T


def weight_error_lr(
    error: torch.Tensor, rank: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    output_basis, singular, input_basis, energy = truncated_basis(error, rank, seed)
    return input_basis * singular, output_basis, energy


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    used = min(left.shape[1], right.shape[1])
    if used == 0:
        return 1.0
    return float((left[:, :used].T @ right[:, :used]).square().sum() / used)


def select_block_sparse(
    error: torch.Tensor,
    inputs: torch.Tensor,
    ratio: float,
    block_out: int,
    block_in: int,
) -> tuple[torch.Tensor, int, int]:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("sparse block ratio must be in (0, 1]")
    output_features, input_features = error.shape
    input_energy = inputs.float().square().mean(dim=0)
    blocks: list[tuple[float, int, int, int, int]] = []
    for output_start in range(0, output_features, block_out):
        output_end = min(output_start + block_out, output_features)
        for input_start in range(0, input_features, block_in):
            input_end = min(input_start + block_in, input_features)
            block = error[output_start:output_end, input_start:input_end]
            score = float(
                (block.float().square() * input_energy[input_start:input_end]).sum()
            )
            blocks.append((score, output_start, output_end, input_start, input_end))
    selected_count = max(1, math.ceil(len(blocks) * ratio))
    selected = sorted(blocks, reverse=True)[:selected_count]
    sparse = torch.zeros_like(error)
    stored_values = 0
    for _, output_start, output_end, input_start, input_end in selected:
        sparse[output_start:output_end, input_start:input_end] = error[
            output_start:output_end, input_start:input_end
        ]
        stored_values += (output_end - output_start) * (input_end - input_start)
    return sparse, selected_count, stored_values


def gather_inputs(
    records: list[dict[str, object]],
    block: int,
    signal: str,
    run_id: str,
    parity: int | None,
    max_rows: int,
) -> torch.Tensor:
    selected = [
        record
        for record in records
        if int(record["block"]) == block
        and str(record["signal"]) == signal
        and str(record["run_id"]) == run_id
        and (parity is None or int(record["step"]) % 2 == parity)
    ]
    selected.sort(key=lambda record: (int(record["step"]), int(record["branch"])))
    if not selected:
        raise ValueError(
            f"no records for block={block} signal={signal} run={run_id} parity={parity}"
        )
    value = torch.cat([record["sample"].float() for record in selected], dim=0)
    return value[:max_rows] if max_rows > 0 else value


def dense_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: torch.nn.Module | None,
) -> torch.Tensor:
    output = F.linear(inputs, weight, bias)
    return activation(output) if activation is not None else output


def correction_metrics(
    inputs: torch.Tensor,
    reference: torch.Tensor,
    quantized: torch.Tensor,
    correction: torch.Tensor,
) -> dict[str, float]:
    del inputs
    quantization_error = quantized - reference
    estimate = quantized + correction
    residual = estimate - reference
    original_energy = float(quantization_error.float().square().sum())
    captured = 1.0 - float(residual.float().square().sum()) / max(original_energy, 1e-30)
    return {
        "relative_l2": relative_l2(estimate, reference),
        "defect_energy_captured": captured,
        "correction_vs_quant_error_cosine": cosine(correction, quantization_error),
        "residual_vs_correction_cosine": cosine(residual, correction),
    }


def storage_and_ops(
    weight: torch.Tensor,
    bits: int,
    rank: int,
    rows: int,
    sparse_values: int,
    sparse_blocks: int,
) -> dict[str, int]:
    input_features = weight.shape[1]
    output_features = weight.shape[0]
    lr_values = rank * (input_features + output_features)
    return {
        "estimated_stored_bits": int(
            weight.numel() * bits + lr_values * 16 + sparse_values * 16 + sparse_blocks * 32
        ),
        "estimated_extra_ops": int(
            2 * rows * rank * (input_features + output_features)
            + 2 * rows * sparse_values
        ),
        "lowrank_values": lr_values,
        "sparse_values": sparse_values,
        "sparse_blocks": sparse_blocks,
    }


def load_ffns(
    wan_source: Path, checkpoint: Path, blocks: tuple[int, ...]
) -> dict[int, torch.nn.Sequential]:
    sys.path.insert(0, str(wan_source))
    os.chdir(wan_source)
    from wan.modules.model import WanModel

    model = WanModel.from_pretrained(str(checkpoint))
    model.eval().requires_grad_(False)
    if max(blocks) >= len(model.blocks):
        raise ValueError("requested block exceeds checkpoint depth")
    ffns = {block: model.blocks[block].ffn.cpu().float() for block in blocks}
    del model
    gc.collect()
    return ffns


def candidate(
    inputs_fit: torch.Tensor,
    inputs_eval: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: torch.nn.Module | None,
    bits: int,
    clip: float,
    rank: int,
    ridge: float,
    seed: int,
    sparse_ratio: float = 0.0,
    block_out: int = 64,
    block_in: int = 128,
) -> dict[str, object]:
    quantized_weight = symmetric_channel_quantize(weight, bits, clip)
    sparse = torch.zeros_like(weight)
    sparse_blocks = 0
    sparse_values = 0
    if sparse_ratio > 0.0:
        sparse, sparse_blocks, sparse_values = select_block_sparse(
            weight - quantized_weight,
            inputs_fit,
            sparse_ratio,
            block_out,
            block_in,
        )
    main_weight = quantized_weight + sparse
    fit_reference = dense_projection(inputs_fit, weight, bias, activation)
    fit_quantized = dense_projection(inputs_fit, main_weight, bias, activation)
    fit_defect = fit_reference - fit_quantized
    factors = fit_activation_defect_lr(inputs_fit, fit_defect, rank, ridge, seed)
    eval_reference = dense_projection(inputs_eval, weight, bias, activation)
    eval_quantized = dense_projection(inputs_eval, main_weight, bias, activation)
    correction = apply_lr(
        inputs_eval,
        factors["input_factor"],
        factors["output_basis"],
    )
    metrics = correction_metrics(inputs_eval, eval_reference, eval_quantized, correction)
    return {
        "clip": clip,
        "sparse_ratio": sparse_ratio,
        "sparse": sparse,
        "sparse_blocks": sparse_blocks,
        "sparse_values": sparse_values,
        "quantized_weight": quantized_weight,
        "main_weight": main_weight,
        "factors": factors,
        "metrics": metrics,
    }


def alternating_candidate(
    inputs_fit: torch.Tensor,
    inputs_eval: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: torch.nn.Module | None,
    bits: int,
    clip: float,
    rank: int,
    ridge: float,
    seed: int,
    iterations: int,
    sparse_ratio: float = 0.0,
    block_out: int = 64,
    block_in: int = 128,
    shaping_strength: float = 1.0,
) -> dict[str, object]:
    """Alternate Q(W-beta*L-S), activation-defect LR, and sparse selection."""
    if iterations <= 0:
        raise ValueError("alternating iterations must be positive")
    if shaping_strength < 0.0 or shaping_strength > 1.0:
        raise ValueError("shaping strength must be in [0, 1]")
    if activation is not None:
        raise ValueError("residual weight shaping currently supports linear projections only")
    lowrank_weight = torch.zeros_like(weight)
    sparse = torch.zeros_like(weight)
    sparse_blocks = 0
    sparse_values = 0
    fit_reference = dense_projection(inputs_fit, weight, bias, activation)
    factors: dict[str, torch.Tensor | float] | None = None

    for iteration in range(iterations):
        quantized_residual = symmetric_channel_quantize(
            weight - shaping_strength * lowrank_weight - sparse, bits, clip
        )
        base_weight = quantized_residual + sparse
        fit_base = dense_projection(inputs_fit, base_weight, bias, activation)
        factors = fit_activation_defect_lr(
            inputs_fit,
            fit_reference - fit_base,
            rank,
            ridge,
            seed + iteration,
        )
        lowrank_weight = (
            factors["output_basis"] @ factors["input_factor"].T
        ).contiguous()
        if sparse_ratio > 0.0:
            sparse, sparse_blocks, sparse_values = select_block_sparse(
                weight - quantized_residual - lowrank_weight,
                inputs_fit,
                sparse_ratio,
                block_out,
                block_in,
            )

    quantized_residual = symmetric_channel_quantize(
        weight - shaping_strength * lowrank_weight - sparse, bits, clip
    )
    base_weight = quantized_residual + sparse
    fit_base = dense_projection(inputs_fit, base_weight, bias, activation)
    factors = fit_activation_defect_lr(
        inputs_fit,
        fit_reference - fit_base,
        rank,
        ridge,
        seed + iterations,
    )
    eval_reference = dense_projection(inputs_eval, weight, bias, activation)
    eval_base = dense_projection(inputs_eval, base_weight, bias, activation)
    correction = apply_lr(
        inputs_eval,
        factors["input_factor"],
        factors["output_basis"],
    )
    metrics = correction_metrics(inputs_eval, eval_reference, eval_base, correction)
    return {
        "clip": clip,
        "sparse_ratio": sparse_ratio,
        "sparse": sparse,
        "sparse_blocks": sparse_blocks,
        "sparse_values": sparse_values,
        "quantized_weight": quantized_residual,
        "main_weight": base_weight,
        "factors": factors,
        "metrics": metrics,
        "alternating_iterations": iterations,
        "shaping_strength": shaping_strength,
    }


def main() -> None:
    args = parse_args()
    if (
        args.block_out <= 0
        or args.block_in <= 0
        or args.ridge <= 0.0
        or args.alternating_iterations <= 0
    ):
        raise ValueError("block sizes and ridge must be positive")
    if any(value <= 0.0 or value > 1.0 for value in args.clip_multipliers):
        raise ValueError("clip multipliers must be in (0, 1]")
    if any(value <= 0.0 or value > 1.0 for value in args.sparse_ratios):
        raise ValueError("sparse ratios must be in (0, 1]")
    if any(value < 0.0 or value > 1.0 for value in args.shaping_strengths):
        raise ValueError("shaping strengths must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = list(payload["records"])
    run_ids = sorted({str(record["run_id"]) for record in records})
    if len(run_ids) < 2:
        raise ValueError("joint shaping requires at least two independent runs")
    calibration_run = run_ids[0]
    test_runs = run_ids[1:]
    signal = "ffn_hidden_post_gelu" if args.projection == "down" else "ffn_input"
    ffns = load_ffns(args.wan_source.resolve(), args.checkpoint.resolve(), args.blocks)
    rows: list[dict[str, object]] = []
    started = time.time()

    for block in args.blocks:
        ffn = ffns[block]
        linear = ffn[2] if args.projection == "down" else ffn[0]
        activation = None if args.projection == "down" else ffn[1]
        weight = linear.weight.detach().to(device=device, dtype=torch.float32)
        bias = (
            linear.bias.detach().to(device=device, dtype=torch.float32)
            if linear.bias is not None
            else None
        )
        x_fit = gather_inputs(
            records, block, signal, calibration_run, 0, args.max_rows_per_split
        ).to(device)
        x_validation = gather_inputs(
            records, block, signal, calibration_run, 1, args.max_rows_per_split
        ).to(device)
        x_test = torch.cat(
            [
                gather_inputs(records, block, signal, run_id, None, args.max_rows_per_split)
                for run_id in test_runs
            ],
            dim=0,
        ).to(device)
        y_validation = dense_projection(x_validation, weight, bias, activation)
        y_test = dense_projection(x_test, weight, bias, activation)

        for bits in args.bits:
            ptq_candidates = []
            for clip in args.clip_multipliers:
                quantized_weight = symmetric_channel_quantize(weight, bits, clip)
                validation_quantized = dense_projection(
                    x_validation, quantized_weight, bias, activation
                )
                ptq_candidates.append(
                    (relative_l2(validation_quantized, y_validation), clip, quantized_weight)
                )
            ptq_validation, ptq_clip, ptq_weight = min(ptq_candidates, key=lambda item: item[0])
            ptq_test_output = dense_projection(x_test, ptq_weight, bias, activation)
            ptq_test = relative_l2(ptq_test_output, y_test)
            rows.append(
                {
                    "block": block,
                    "projection": args.projection,
                    "bits": bits,
                    "method": "ptq_only",
                    "rank": 0,
                    "clip": ptq_clip,
                    "sparse_ratio": 0.0,
                    "validation_relative_l2": ptq_validation,
                    "test_relative_l2": ptq_test,
                    "defect_energy_captured_test": 0.0,
                    "subspace_overlap": "",
                    "correction_vs_quant_error_cosine": "",
                    "residual_vs_correction_cosine": "",
                    **storage_and_ops(weight, bits, 0, x_test.shape[0], 0, 0),
                }
            )

            for rank in args.ranks:
                error = weight - ptq_weight
                weight_input, weight_output, weight_energy = weight_error_lr(
                    error, rank, args.seed + block * 1000 + bits * 10 + rank
                )
                weight_correction = apply_lr(x_test, weight_input, weight_output)
                weight_metrics = correction_metrics(
                    x_test, y_test, ptq_test_output, weight_correction
                )
                rows.append(
                    {
                        "block": block,
                        "projection": args.projection,
                        "bits": bits,
                        "method": "ptq_weight_svd",
                        "rank": rank,
                        "clip": ptq_clip,
                        "sparse_ratio": 0.0,
                        "validation_relative_l2": "",
                        "test_relative_l2": weight_metrics["relative_l2"],
                        "defect_energy_captured_test": weight_metrics["defect_energy_captured"],
                        "weight_error_rank_energy": weight_energy,
                        "subspace_overlap": "",
                        "correction_vs_quant_error_cosine": weight_metrics[
                            "correction_vs_quant_error_cosine"
                        ],
                        "residual_vs_correction_cosine": weight_metrics[
                            "residual_vs_correction_cosine"
                        ],
                        **storage_and_ops(weight, bits, rank, x_test.shape[0], 0, 0),
                    }
                )

                alternating_candidates = [
                    alternating_candidate(
                        x_fit,
                        x_validation,
                        weight,
                        bias,
                        activation,
                        bits,
                        clip,
                        rank,
                        args.ridge,
                        args.seed
                        + block * 1000
                        + bits * 100
                        + rank * 10
                        + strength_index * len(args.clip_multipliers)
                        + index,
                        args.alternating_iterations,
                        shaping_strength=shaping_strength,
                    )
                    for strength_index, shaping_strength in enumerate(
                        args.shaping_strengths
                    )
                    for index, clip in enumerate(args.clip_multipliers)
                ]
                alternating = min(
                    alternating_candidates,
                    key=lambda item: float(item["metrics"]["relative_l2"]),
                )
                alternating_test_base = dense_projection(
                    x_test, alternating["main_weight"], bias, activation
                )
                alternating_test_correction = apply_lr(
                    x_test,
                    alternating["factors"]["input_factor"],
                    alternating["factors"]["output_basis"],
                )
                alternating_metrics = correction_metrics(
                    x_test,
                    y_test,
                    alternating_test_base,
                    alternating_test_correction,
                )
                _, _, alternating_test_basis, _ = truncated_basis(
                    y_test - alternating_test_base,
                    rank,
                    args.seed + block * 1000 + bits * 100 + rank * 10 + 7,
                )
                rows.append(
                    {
                        "block": block,
                        "projection": args.projection,
                        "bits": bits,
                        "method": "joint_residual_shaped_ptq_lr",
                        "rank": rank,
                        "clip": alternating["clip"],
                        "sparse_ratio": 0.0,
                        "alternating_iterations": args.alternating_iterations,
                        "shaping_strength": alternating["shaping_strength"],
                        "validation_relative_l2": alternating["metrics"][
                            "relative_l2"
                        ],
                        "test_relative_l2": alternating_metrics["relative_l2"],
                        "defect_energy_captured_test": alternating_metrics[
                            "defect_energy_captured"
                        ],
                        "calibration_defect_rank_energy": alternating["factors"][
                            "calibration_oracle_energy"
                        ],
                        "subspace_overlap": subspace_overlap(
                            alternating["factors"]["output_basis"],
                            alternating_test_basis,
                        ),
                        "correction_vs_quant_error_cosine": alternating_metrics[
                            "correction_vs_quant_error_cosine"
                        ],
                        "residual_vs_correction_cosine": alternating_metrics[
                            "residual_vs_correction_cosine"
                        ],
                        **storage_and_ops(
                            weight, bits, rank, x_test.shape[0], 0, 0
                        ),
                    }
                )

                fit_reference = dense_projection(x_fit, weight, bias, activation)
                fit_ptq = dense_projection(x_fit, ptq_weight, bias, activation)
                activation_factors = fit_activation_defect_lr(
                    x_fit,
                    fit_reference - fit_ptq,
                    rank,
                    args.ridge,
                    args.seed + block * 1000 + bits * 10 + rank + 1,
                )
                activation_correction = apply_lr(
                    x_test,
                    activation_factors["input_factor"],
                    activation_factors["output_basis"],
                )
                activation_metrics = correction_metrics(
                    x_test, y_test, ptq_test_output, activation_correction
                )
                _, _, test_output_basis, _ = truncated_basis(
                    y_test - ptq_test_output,
                    rank,
                    args.seed + block * 1000 + bits * 10 + rank + 2,
                )
                activation_overlap = subspace_overlap(
                    activation_factors["output_basis"], test_output_basis
                )
                rows.append(
                    {
                        "block": block,
                        "projection": args.projection,
                        "bits": bits,
                        "method": "ptq_activation_lr",
                        "rank": rank,
                        "clip": ptq_clip,
                        "sparse_ratio": 0.0,
                        "validation_relative_l2": "",
                        "test_relative_l2": activation_metrics["relative_l2"],
                        "defect_energy_captured_test": activation_metrics[
                            "defect_energy_captured"
                        ],
                        "calibration_defect_rank_energy": activation_factors[
                            "calibration_oracle_energy"
                        ],
                        "subspace_overlap": activation_overlap,
                        "correction_vs_quant_error_cosine": activation_metrics[
                            "correction_vs_quant_error_cosine"
                        ],
                        "residual_vs_correction_cosine": activation_metrics[
                            "residual_vs_correction_cosine"
                        ],
                        **storage_and_ops(weight, bits, rank, x_test.shape[0], 0, 0),
                    }
                )

                shaped_candidates = [
                    candidate(
                        x_fit,
                        x_validation,
                        weight,
                        bias,
                        activation,
                        bits,
                        clip,
                        rank,
                        args.ridge,
                        args.seed + block * 1000 + bits * 100 + rank * 10 + index,
                    )
                    for index, clip in enumerate(args.clip_multipliers)
                ]
                shaped = min(
                    shaped_candidates,
                    key=lambda item: float(item["metrics"]["relative_l2"]),
                )
                shaped_test_quant = dense_projection(
                    x_test, shaped["main_weight"], bias, activation
                )
                shaped_test_correction = apply_lr(
                    x_test,
                    shaped["factors"]["input_factor"],
                    shaped["factors"]["output_basis"],
                )
                shaped_metrics = correction_metrics(
                    x_test, y_test, shaped_test_quant, shaped_test_correction
                )
                _, _, shaped_test_basis, _ = truncated_basis(
                    y_test - shaped_test_quant,
                    rank,
                    args.seed + block * 1000 + bits * 100 + rank * 10 + 9,
                )
                rows.append(
                    {
                        "block": block,
                        "projection": args.projection,
                        "bits": bits,
                        "method": "joint_shaped_ptq_lr",
                        "rank": rank,
                        "clip": shaped["clip"],
                        "sparse_ratio": 0.0,
                        "validation_relative_l2": shaped["metrics"]["relative_l2"],
                        "test_relative_l2": shaped_metrics["relative_l2"],
                        "defect_energy_captured_test": shaped_metrics[
                            "defect_energy_captured"
                        ],
                        "calibration_defect_rank_energy": shaped["factors"][
                            "calibration_oracle_energy"
                        ],
                        "subspace_overlap": subspace_overlap(
                            shaped["factors"]["output_basis"], shaped_test_basis
                        ),
                        "correction_vs_quant_error_cosine": shaped_metrics[
                            "correction_vs_quant_error_cosine"
                        ],
                        "residual_vs_correction_cosine": shaped_metrics[
                            "residual_vs_correction_cosine"
                        ],
                        **storage_and_ops(weight, bits, rank, x_test.shape[0], 0, 0),
                    }
                )

                for sparse_ratio in args.sparse_ratios:
                    sparse_candidates = [
                        candidate(
                            x_fit,
                            x_validation,
                            weight,
                            bias,
                            activation,
                            bits,
                            clip,
                            rank,
                            args.ridge,
                            args.seed
                            + block * 1000
                            + bits * 100
                            + rank * 10
                            + index,
                            sparse_ratio=sparse_ratio,
                            block_out=args.block_out,
                            block_in=args.block_in,
                        )
                        for index, clip in enumerate(args.clip_multipliers)
                    ]
                    shaped_sparse = min(
                        sparse_candidates,
                        key=lambda item: float(item["metrics"]["relative_l2"]),
                    )
                    sparse_test_quant = dense_projection(
                        x_test, shaped_sparse["main_weight"], bias, activation
                    )
                    sparse_test_correction = apply_lr(
                        x_test,
                        shaped_sparse["factors"]["input_factor"],
                        shaped_sparse["factors"]["output_basis"],
                    )
                    sparse_metrics = correction_metrics(
                        x_test, y_test, sparse_test_quant, sparse_test_correction
                    )
                    _, _, sparse_test_basis, _ = truncated_basis(
                        y_test - sparse_test_quant,
                        rank,
                        args.seed + block * 1000 + bits * 100 + rank * 10 + 8,
                    )
                    rows.append(
                        {
                            "block": block,
                            "projection": args.projection,
                            "bits": bits,
                            "method": "joint_shaped_ptq_lr_block_sparse",
                            "rank": rank,
                            "clip": shaped_sparse["clip"],
                            "sparse_ratio": sparse_ratio,
                            "validation_relative_l2": shaped_sparse["metrics"][
                                "relative_l2"
                            ],
                            "test_relative_l2": sparse_metrics["relative_l2"],
                            "defect_energy_captured_test": sparse_metrics[
                                "defect_energy_captured"
                            ],
                            "calibration_defect_rank_energy": shaped_sparse["factors"][
                                "calibration_oracle_energy"
                            ],
                            "subspace_overlap": subspace_overlap(
                                shaped_sparse["factors"]["output_basis"],
                                sparse_test_basis,
                            ),
                            "correction_vs_quant_error_cosine": sparse_metrics[
                                "correction_vs_quant_error_cosine"
                            ],
                            "residual_vs_correction_cosine": sparse_metrics[
                                "residual_vs_correction_cosine"
                            ],
                            **storage_and_ops(
                                weight,
                                bits,
                                rank,
                                x_test.shape[0],
                                int(shaped_sparse["sparse_values"]),
                                int(shaped_sparse["sparse_blocks"]),
                            ),
                        }
                    )

                    alternating_sparse_candidates = [
                        alternating_candidate(
                            x_fit,
                            x_validation,
                            weight,
                            bias,
                            activation,
                            bits,
                            clip,
                            rank,
                            args.ridge,
                            args.seed
                            + block * 1000
                            + bits * 100
                            + rank * 10
                            + strength_index * len(args.clip_multipliers)
                            + index,
                            args.alternating_iterations,
                            sparse_ratio=sparse_ratio,
                            block_out=args.block_out,
                            block_in=args.block_in,
                            shaping_strength=shaping_strength,
                        )
                        for strength_index, shaping_strength in enumerate(
                            args.shaping_strengths
                        )
                        for index, clip in enumerate(args.clip_multipliers)
                    ]
                    alternating_sparse = min(
                        alternating_sparse_candidates,
                        key=lambda item: float(item["metrics"]["relative_l2"]),
                    )
                    alternating_sparse_test_base = dense_projection(
                        x_test,
                        alternating_sparse["main_weight"],
                        bias,
                        activation,
                    )
                    alternating_sparse_test_correction = apply_lr(
                        x_test,
                        alternating_sparse["factors"]["input_factor"],
                        alternating_sparse["factors"]["output_basis"],
                    )
                    alternating_sparse_metrics = correction_metrics(
                        x_test,
                        y_test,
                        alternating_sparse_test_base,
                        alternating_sparse_test_correction,
                    )
                    _, _, alternating_sparse_test_basis, _ = truncated_basis(
                        y_test - alternating_sparse_test_base,
                        rank,
                        args.seed + block * 1000 + bits * 100 + rank * 10 + 6,
                    )
                    rows.append(
                        {
                            "block": block,
                            "projection": args.projection,
                            "bits": bits,
                            "method": "joint_residual_shaped_ptq_lr_block_sparse",
                            "rank": rank,
                            "clip": alternating_sparse["clip"],
                            "sparse_ratio": sparse_ratio,
                            "alternating_iterations": args.alternating_iterations,
                            "shaping_strength": alternating_sparse[
                                "shaping_strength"
                            ],
                            "validation_relative_l2": alternating_sparse["metrics"][
                                "relative_l2"
                            ],
                            "test_relative_l2": alternating_sparse_metrics[
                                "relative_l2"
                            ],
                            "defect_energy_captured_test": alternating_sparse_metrics[
                                "defect_energy_captured"
                            ],
                            "calibration_defect_rank_energy": alternating_sparse[
                                "factors"
                            ]["calibration_oracle_energy"],
                            "subspace_overlap": subspace_overlap(
                                alternating_sparse["factors"]["output_basis"],
                                alternating_sparse_test_basis,
                            ),
                            "correction_vs_quant_error_cosine": alternating_sparse_metrics[
                                "correction_vs_quant_error_cosine"
                            ],
                            "residual_vs_correction_cosine": alternating_sparse_metrics[
                                "residual_vs_correction_cosine"
                            ],
                            **storage_and_ops(
                                weight,
                                bits,
                                rank,
                                x_test.shape[0],
                                int(alternating_sparse["sparse_values"]),
                                int(alternating_sparse["sparse_blocks"]),
                            ),
                        }
                    )
            print(f"[joint-shaping] block={block} bits={bits} rows={len(rows)}", flush=True)
        ffn.cpu()
        del weight, bias, x_fit, x_validation, x_test, y_validation, y_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for row in rows:
        row["local_error_2pct_go"] = float(row["test_relative_l2"]) <= 0.02
        overlap = row.get("subspace_overlap", "")
        row["subspace_overlap_07_go"] = overlap != "" and float(overlap) >= 0.70
        row["rank_energy_07_go"] = float(row.get("defect_energy_captured_test", 0.0)) >= 0.70
        row["combined_gate"] = bool(
            row["local_error_2pct_go"]
            and row["subspace_overlap_07_go"]
            and row["rank_energy_07_go"]
        )
    write_csv(args.output_dir / "joint_quant_lr_shaping.csv", rows)
    best = min(rows, key=lambda row: float(row["test_relative_l2"]))
    manifest = {
        "scope": "held-out activation-shaped PTQ + low-rank + block-sparse FFN probe",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "calibration_run": calibration_run,
        "calibration_fit_steps": "even",
        "calibration_selection_steps": "odd",
        "test_runs": test_runs,
        "records": len(records),
        "best_row": best,
        "methods": sorted({str(row["method"]) for row in rows}),
        "gates": {
            "heldout_defect_energy": 0.70,
            "local_output_relative_l2": 0.02,
            "cross_run_subspace_overlap": 0.70,
        },
        "warnings": [
            "Sampled F17 activation rows are an expressivity screen, not end-to-end video quality.",
            "Estimated storage/ops assume fused low-rank and block-sparse correction kernels.",
            "The sparse selector uses diagonal activation covariance for block scoring.",
        ],
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
