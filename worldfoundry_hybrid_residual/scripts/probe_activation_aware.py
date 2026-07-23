#!/usr/bin/env python3
"""Probe activation-aware residual decompositions for Wan DiT weights.

The original probe fits residuals in Frobenius norm. This script adds a
held-out activation objective so that the comparison reflects the actual
linear outputs used by WorldFoundry rather than treating every input
direction equally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from probe_decompositions import (
    bcm_project,
    load_hadamard_module,
    lowrank_dense,
    random_hadamard,
    randomized_svd,
    ternary_groupwise,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_weight(path: str, key: str, device: torch.device) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device=device, dtype=torch.float32)


def load_activation_matrix(
    path: str,
    in_features: int,
    samples: int,
    device: torch.device,
) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    candidates: list[torch.Tensor] = []
    if isinstance(obj, dict):
        for name in ("q", "k", "v", "hidden_states", "x", "input"):
            value = obj.get(name)
            if torch.is_tensor(value):
                candidates.append(value)
    for value in candidates:
        flattened = value.reshape(-1, value.shape[-1])
        if flattened.shape[-1] != in_features:
            continue
        count = min(samples, flattened.shape[0])
        indices = torch.linspace(0, flattened.shape[0] - 1, count).long()
        return flattened[indices].to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260723)
    return torch.randn(samples, in_features, device=device, generator=generator)


def fit_activation_lowrank(
    error: torch.Tensor,
    inputs: torch.Tensor,
    rank: int,
    ridge: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a rank-r matrix for ||inputs @ (error-delta).T||_F.

    Work in the singular coordinates of the held-out input distribution.
    This avoids forming an ill-conditioned d-by-d inverse when the number of
    calibration samples is smaller than the input width. ``ridge`` is a
    relative ridge coefficient, scaled by the mean squared input singular
    value, so it behaves consistently across layers.
    """
    # X = Ux diag(sx) Vh.  The activation objective is equivalent to
    # approximating T = diag(sx) Vh E^T, then lifting with the regularized
    # inverse of diag(sx).  Rank is preserved by this construction.
    _, sx, vh = torch.linalg.svd(inputs, full_matrices=False)
    keep = sx > sx.max().clamp_min(1e-12) * 1e-6
    sx = sx[keep]
    vh = vh[keep]
    target = (sx[:, None] * (vh @ error.T))
    target_factors = randomized_svd(target, rank, seed=seed)
    gain = sx / (sx.square() + ridge * sx.square().mean().clamp_min(1e-12))
    middle = target_factors.u.T * gain.unsqueeze(0)
    middle = target_factors.s.unsqueeze(1) * middle
    lifted = target_factors.v @ middle @ vh
    factors = randomized_svd(lifted, rank, seed=seed + 1)
    return lowrank_dense(factors), factors.u, factors.v


def fit_activation_bcm(
    error: torch.Tensor,
    inputs: torch.Tensor,
    block_size: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit shared-generator block-circulant blocks to activation outputs."""
    out_features, in_features = error.shape
    if out_features % block_size or in_features % block_size:
        raise ValueError(f"shape {tuple(error.shape)} is not divisible by {block_size}")
    in_blocks = in_features // block_size
    out_blocks = out_features // block_size
    x_blocks = inputs.view(inputs.shape[0], in_blocks, block_size)
    shifted = torch.stack(
        [torch.roll(x_blocks, shifts=-offset, dims=-1) for offset in range(block_size)],
        dim=-1,
    )
    design = shifted.permute(0, 2, 1, 3).reshape(-1, in_features)
    target = (inputs @ error.T).view(inputs.shape[0], out_blocks, block_size)
    target = target.permute(0, 2, 1).reshape(-1, out_blocks)
    gram = design.T @ design
    identity = torch.eye(in_features, device=design.device, dtype=design.dtype)
    relative_ridge = ridge * gram.diag().mean().clamp_min(1e-12)
    generators_flat = torch.linalg.solve(
        gram + relative_ridge * identity, design.T @ target
    )
    generators = generators_flat.T.reshape(out_blocks, in_blocks, block_size)
    rows = torch.arange(block_size, device=error.device).view(1, block_size)
    offsets = torch.arange(block_size, device=error.device).view(block_size, 1)
    dense_index = (rows - offsets) % block_size
    blocks = generators[..., dense_index]
    dense = (
        blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(out_features, in_features)
    )
    return dense, generators


def activation_row_sparse(
    residual: torch.Tensor,
    inputs: torch.Tensor,
    parameter_budget: int,
    block_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = residual.shape
    if out_features % block_rows:
        raise ValueError("output width must be divisible by block_rows")
    block_count = out_features // block_rows
    selected_count = min(block_count, parameter_budget // (block_rows * in_features))
    if selected_count <= 0:
        return residual.new_zeros(residual.shape), residual.new_empty((0,), dtype=torch.long)
    blocks = residual.view(block_count, block_rows, in_features)
    output_energy = (inputs @ residual.T).view(inputs.shape[0], block_count, block_rows)
    energy = output_energy.float().square().sum(dim=(0, 2))
    selected = torch.topk(energy, k=selected_count).indices.sort().values
    sparse = residual.new_zeros(residual.shape)
    row_indices = []
    for block in selected.tolist():
        start = block * block_rows
        stop = start + block_rows
        sparse[start:stop] = blocks[block]
        row_indices.extend(range(start, stop))
    return sparse, torch.tensor(row_indices, device=residual.device, dtype=torch.long)


def evaluate(reference: torch.Tensor, estimate: torch.Tensor, inputs: torch.Tensor) -> dict[str, float]:
    reference_output = inputs @ reference.T
    estimate_output = inputs @ estimate.T
    error = reference - estimate
    output_error = reference_output - estimate_output
    return {
        "weight_rel_fro": float((error.norm() / reference.norm().clamp_min(1e-12)).item()),
        "activation_rel_l2": float(
            (output_error.norm() / reference_output.norm().clamp_min(1e-12)).item()
        ),
        "activation_cosine": float(
            F.cosine_similarity(
                reference_output.reshape(1, -1), estimate_output.reshape(1, -1)
            ).item()
        ),
    }


def add_metric(
    rows: list[dict],
    key: str,
    method: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    fit_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    params: int,
    budget: int,
    notes: str,
) -> None:
    fit = evaluate(reference, estimate, fit_inputs)
    holdout = evaluate(reference, estimate, eval_inputs)
    rows.append(
        {
            "weight_key": key,
            "method": method,
            "params": params,
            "budget": budget,
            "budget_ratio": params / max(budget, 1),
            "fit_weight_rel_fro": fit["weight_rel_fro"],
            "fit_activation_rel_l2": fit["activation_rel_l2"],
            "fit_activation_cosine": fit["activation_cosine"],
            "holdout_weight_rel_fro": holdout["weight_rel_fro"],
            "holdout_activation_rel_l2": holdout["activation_rel_l2"],
            "holdout_activation_cosine": holdout["activation_cosine"],
            "notes": notes,
        }
    )


def run_weight(
    checkpoint: str,
    activation_path: str,
    key: str,
    device: torch.device,
    ranks: list[int],
    block_sizes: list[int],
    rank_budget: int,
    group_size: int,
    samples: int,
    ridge: float,
    block_rows: int,
    seed: int,
    hadamard_module,
) -> tuple[list[dict], list[dict]]:
    weight = load_weight(checkpoint, key, device)
    activation = load_activation_matrix(activation_path, weight.shape[1], samples * 2, device)
    signs = torch.randint(0, 2, (weight.shape[1],), device=device, dtype=torch.int64).float().mul_(2).sub_(1)
    weight = random_hadamard(weight, signs, hadamard_module)
    activation = random_hadamard(activation, signs, hadamard_module).float()
    split = max(1, activation.shape[0] // 2)
    fit_inputs = activation[:split]
    eval_inputs = activation[split:] if activation.shape[0] > split else activation[:split]
    main = ternary_groupwise(weight, group_size)
    error = weight - main
    out_features, in_features = weight.shape
    budget = rank_budget * (out_features + in_features)
    rows: list[dict] = []
    structures: list[dict] = []
    add_metric(rows, key, "ternary_main", main, weight, fit_inputs, eval_inputs, 0, budget, "RobuQ ternary main branch")

    max_rank = max(ranks)
    fro_factors = randomized_svd(error, max_rank, seed=seed + 11)
    for rank in ranks:
        delta = lowrank_dense(fro_factors, rank)
        add_metric(rows, key, f"fro_lowrank_r{rank}", main + delta, weight, fit_inputs, eval_inputs, rank * (out_features + in_features), budget, "Frobenius SVD of quantization error")
        act_delta, _, _ = fit_activation_lowrank(error, fit_inputs, rank, ridge, seed + 101 + rank)
        add_metric(rows, key, f"activation_lowrank_r{rank}", main + act_delta, weight, fit_inputs, eval_inputs, rank * (out_features + in_features), budget, "Activation-aware SVD lifted through ridge regression")

    for block_size in block_sizes:
        if out_features % block_size or in_features % block_size:
            continue
        bcm_fro, generators = bcm_project(error, block_size)
        bcm_params = int(generators.numel())
        remaining_rank = max(0, (budget - bcm_params) // (out_features + in_features))
        add_metric(rows, key, f"fro_bcm_b{block_size}", main + bcm_fro, weight, fit_inputs, eval_inputs, bcm_params, budget, "Frobenius block-circulant projection")
        if remaining_rank:
            after = error - bcm_fro
            after_factors = randomized_svd(after, remaining_rank, seed=seed + 200 + block_size)
            delta = lowrank_dense(after_factors, remaining_rank)
            add_metric(rows, key, f"fro_bcm_plus_lr_b{block_size}_r{remaining_rank}", main + bcm_fro + delta, weight, fit_inputs, eval_inputs, bcm_params + remaining_rank * (out_features + in_features), budget, "Frobenius BCM followed by Frobenius low rank")

        act_bcm, act_generators = fit_activation_bcm(error, fit_inputs, block_size, ridge)
        act_params = int(act_generators.numel())
        add_metric(rows, key, f"activation_bcm_b{block_size}", main + act_bcm, weight, fit_inputs, eval_inputs, act_params, budget, "Activation-aware shared-generator least squares")
        if remaining_rank:
            after = error - act_bcm
            act_delta, _, _ = fit_activation_lowrank(after, fit_inputs, remaining_rank, ridge, seed + 300 + block_size)
            add_metric(rows, key, f"activation_bcm_plus_lr_b{block_size}_r{remaining_rank}", main + act_bcm + act_delta, weight, fit_inputs, eval_inputs, act_params + remaining_rank * (out_features + in_features), budget, "Activation-aware BCM followed by activation-aware low rank")
        structures.append(
            {
                "weight_key": key,
                "block_size": block_size,
                "bcm_params": bcm_params,
                "fro_bcm_capture": float((bcm_fro.square().sum() / error.square().sum().clamp_min(1e-12)).item()),
                "activation_bcm_params": act_params,
                "activation_bcm_fit_capture": float((1.0 - (fit_inputs @ (error - act_bcm).T).norm().square() / (fit_inputs @ error.T).norm().square().clamp_min(1e-12)).item()),
                "activation_bcm_holdout_capture": float((1.0 - (eval_inputs @ (error - act_bcm).T).norm().square() / (eval_inputs @ error.T).norm().square().clamp_min(1e-12)).item()),
            }
        )

    base_rank = max(1, rank_budget // 2)
    base_delta, _, _ = fit_activation_lowrank(error, fit_inputs, base_rank, ridge, seed + 401)
    sparse_budget = max(0, budget - base_rank * (out_features + in_features))
    if out_features % block_rows == 0 and sparse_budget:
        sparse, selected = activation_row_sparse(error - base_delta, fit_inputs, sparse_budget, block_rows)
        params = base_rank * (out_features + in_features) + int(selected.numel()) * in_features
        add_metric(rows, key, f"activation_lr_r{base_rank}_row_sparse_b{block_rows}", main + base_delta + sparse, weight, fit_inputs, eval_inputs, params, budget, "Activation-energy-selected output-row blocks")
    return rows, structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--ranks", default="4,8,16")
    parser.add_argument("--block-sizes", default="64,128")
    parser.add_argument("--rank-budget", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    hadamard_module, hadamard_path = load_hadamard_module()
    keys = [item.strip() for item in args.keys.split(",") if item.strip()]
    ranks = [int(item) for item in args.ranks.split(",") if item.strip()]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item.strip()]
    all_rows: list[dict] = []
    all_structures: list[dict] = []
    started = time.time()
    for index, key in enumerate(keys):
        print(f"[activation-aware] loading {key}", flush=True)
        rows, structures = run_weight(
            args.checkpoint,
            args.activation,
            key,
            device,
            ranks,
            block_sizes,
            args.rank_budget,
            args.group_size,
            args.samples,
            args.ridge,
            args.block_rows,
            args.seed + index * 1000,
            hadamard_module,
        )
        all_rows.extend(rows)
        all_structures.extend(structures)
        write_csv(output / "activation_aware_metrics.partial.csv", all_rows)
        write_csv(output / "activation_aware_structure.partial.csv", all_structures)
    write_csv(output / "activation_aware_metrics.csv", all_rows)
    write_csv(output / "activation_aware_structure.csv", all_structures)
    manifest = {
        "arguments": vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "hadamard_source": str(hadamard_path),
        "elapsed_seconds": time.time() - started,
        "methodology": {
            "split": "first half fit, second half held out",
            "activation_bcm": "ridge least squares over circulant generator features",
            "activation_lowrank": "top output-sample SVD lifted by ridge regression",
            "sparse": "output-row blocks selected by activation residual energy",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[activation-aware] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
