#!/usr/bin/env python3
"""Training-free alternating quantization-aware residual decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from probe_decompositions import (
    bcm_project,
    evaluate,
    load_activation,
    load_hadamard_module,
    load_weight,
    lowrank_dense,
    parse_strings,
    random_hadamard,
    randomized_svd,
    ternary_groupwise,
    top_output_block_residual,
    top_tile_residual,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_lowrank(matrix: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(matrix)
    return lowrank_dense(randomized_svd(matrix, rank, seed=seed), rank)


def run_method(
    *,
    name: str,
    weight: torch.Tensor,
    activation: torch.Tensor,
    iterations: int,
    group_size: int,
    rank: int,
    block_size: int | None,
    sparse_kind: str | None,
    sparse_budget: int,
    sparse_block: int,
    high_precision_params: int,
    seed: int,
) -> list[dict]:
    lowrank = fit_lowrank(weight, rank, seed)
    bcm = torch.zeros_like(weight)
    sparse = torch.zeros_like(weight)
    rows = []
    best_error = float("inf")
    for iteration in range(iterations + 1):
        main = ternary_groupwise(weight - lowrank - bcm - sparse, group_size)
        estimate = main + lowrank + bcm + sparse
        measured = evaluate(weight, estimate, activation)
        best_error = min(best_error, measured["activation_relative_l2"])
        rows.append(
            {
                "method": name,
                "iteration": iteration,
                "rank": rank,
                "block_size": block_size or 0,
                "sparse_kind": sparse_kind or "none",
                "high_precision_params": high_precision_params,
                "relative_fro_error": measured["relative_fro_error"],
                "activation_relative_l2": measured["activation_relative_l2"],
                "activation_cosine": measured["activation_cosine"],
                "spectral_error_proxy": measured["spectral_error_proxy"],
                "best_activation_relative_l2_so_far": best_error,
                "lowrank_norm": float(lowrank.norm().item()),
                "bcm_norm": float(bcm.norm().item()),
                "sparse_norm": float(sparse.norm().item()),
            }
        )
        if iteration == iterations:
            break

        target = weight - main
        if block_size is not None:
            bcm, _ = bcm_project(target - lowrank - sparse, block_size)
        if rank > 0:
            lowrank = fit_lowrank(
                target - bcm - sparse,
                rank,
                seed + 100 + iteration,
            )
        if sparse_kind == "row":
            sparse, _, _ = top_output_block_residual(
                target - bcm - lowrank,
                sparse_budget,
                sparse_block,
            )
        elif sparse_kind == "tile":
            sparse, _, _ = top_tile_residual(
                target - bcm - lowrank,
                sparse_budget,
                sparse_block,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rank-budget", type=int, default=16)
    parser.add_argument("--hybrid-block-size", type=int, default=256)
    parser.add_argument("--sparse-rank", type=int, default=8)
    parser.add_argument("--row-block-size", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    hadamard_module, hadamard_path = load_hadamard_module()
    all_rows = []
    started = time.time()

    for key_index, key in enumerate(parse_strings(args.keys)):
        seed = args.seed + key_index * 1000
        weight = load_weight(args.checkpoint, key, device)
        activation, activation_source = load_activation(
            args.activation,
            weight.shape[1],
            args.activation_samples,
            device,
            seed,
        )
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (weight.shape[1],), device=device).float().mul_(2).sub_(1)
        weight = random_hadamard(weight, signs, hadamard_module)
        activation = random_hadamard(activation, signs, hadamard_module)
        out_features, in_features = weight.shape
        target_params = args.rank_budget * (out_features + in_features)

        configs = [
            {
                "name": f"alternating_svd_r{args.rank_budget}",
                "rank": args.rank_budget,
                "block_size": None,
                "sparse_kind": None,
                "sparse_budget": 0,
                "sparse_block": args.row_block_size,
                "high_precision_params": target_params,
            }
        ]
        if (
            out_features % args.hybrid_block_size == 0
            and in_features % args.hybrid_block_size == 0
        ):
            bcm_params = out_features * in_features // args.hybrid_block_size
            hybrid_rank = max(0, (target_params - bcm_params) // (out_features + in_features))
            configs.append(
                {
                    "name": f"alternating_bcm_b{args.hybrid_block_size}_svd_r{hybrid_rank}",
                    "rank": hybrid_rank,
                    "block_size": args.hybrid_block_size,
                    "sparse_kind": None,
                    "sparse_budget": 0,
                    "sparse_block": args.row_block_size,
                    "high_precision_params": bcm_params + hybrid_rank * (out_features + in_features),
                }
            )
        sparse_lr_params = args.sparse_rank * (out_features + in_features)
        sparse_budget = max(0, target_params - sparse_lr_params)
        if out_features % args.row_block_size == 0:
            row_params = (
                sparse_budget // (args.row_block_size * in_features)
            ) * args.row_block_size * in_features
            configs.append(
                {
                    "name": f"alternating_svd_r{args.sparse_rank}_rowsparse_b{args.row_block_size}",
                    "rank": args.sparse_rank,
                    "block_size": None,
                    "sparse_kind": "row",
                    "sparse_budget": sparse_budget,
                    "sparse_block": args.row_block_size,
                    "high_precision_params": sparse_lr_params + row_params,
                }
            )
        if out_features % args.tile_size == 0 and in_features % args.tile_size == 0:
            tile_params = (
                sparse_budget // (args.tile_size * args.tile_size)
            ) * args.tile_size * args.tile_size
            configs.append(
                {
                    "name": f"alternating_svd_r{args.sparse_rank}_tilesparse_b{args.tile_size}",
                    "rank": args.sparse_rank,
                    "block_size": None,
                    "sparse_kind": "tile",
                    "sparse_budget": sparse_budget,
                    "sparse_block": args.tile_size,
                    "high_precision_params": sparse_lr_params + tile_params,
                }
            )

        for method_index, config in enumerate(configs):
            method_rows = run_method(
                weight=weight,
                activation=activation,
                iterations=args.iterations,
                group_size=args.group_size,
                seed=seed + method_index * 10000,
                **config,
            )
            for row in method_rows:
                row.update(
                    {
                        "weight_key": key,
                        "shape": f"{out_features}x{in_features}",
                        "activation_source": activation_source,
                        "lr16_budget_params": target_params,
                        "budget_ratio_vs_lr16": row["high_precision_params"] / target_params,
                    }
                )
                all_rows.append(row)
                print(
                    f"[alt] {key} {row['method']} iter={row['iteration']} "
                    f"act={row['activation_relative_l2']:.5f} fro={row['relative_fro_error']:.5f}",
                    flush=True,
                )
        write_csv(output_dir / "alternating_metrics.partial.csv", all_rows)
        del weight, activation
        torch.cuda.empty_cache()

    write_csv(output_dir / "alternating_metrics.csv", all_rows)
    manifest = {
        "arguments": vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device),
        "hadamard_source": str(hadamard_path),
        "elapsed_seconds": time.time() - started,
        "scope": "No gradients or calibration optimization; alternating projections only",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[alt] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
