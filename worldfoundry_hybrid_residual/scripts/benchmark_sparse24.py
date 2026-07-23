#!/usr/bin/env python3
"""Reproduce H200 PyTorch/cuSPARSELt 2:4 residual-kernel latency."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def prune_2_to_4(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[1] % 4:
        raise ValueError("input width must be divisible by four")
    grouped = weight.float().view(weight.shape[0], weight.shape[1] // 4, 4)
    selected = grouped.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, selected, True)
    return torch.where(mask, grouped, torch.zeros_like(grouped)).view_as(weight).to(torch.bfloat16)


def benchmark(fn, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end) / iterations))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--rows", default="512,2048,7800,32760")
    parser.add_argument("--features", type=int, default=1536)
    parser.add_argument("--rank-budget", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    dense_weight = torch.randn(
        args.features, args.features, device=device, dtype=torch.bfloat16
    )
    pruned_weight = prune_2_to_4(dense_weight).contiguous()
    sparse_weight = torch.sparse.to_sparse_semi_structured(pruned_weight)
    # Semi-structured tensors expose the narrow matrix transpose operation;
    # the generic `.T` property dispatches through permute and is unsupported.
    sparse_transposed = sparse_weight.t()
    dense_transposed = pruned_weight.T
    lr16_params = args.rank_budget * args.features * 2
    nonzero_values = pruned_weight.numel() // 2

    rows = []
    started = time.time()
    for row_count in [int(item) for item in args.rows.split(",") if item.strip()]:
        activation = torch.randn(
            row_count, args.features, device=device, dtype=torch.bfloat16
        )
        methods = {
            "bf16_dense_2to4_values": lambda: torch.mm(activation, dense_transposed),
            "cusparselt_2to4": lambda: torch.mm(activation, sparse_transposed),
        }
        timings = {}
        for name, fn in methods.items():
            samples = benchmark(fn, args.warmup, args.iterations, args.repeats)
            timings[name] = statistics.median(samples)
            rows.append(
                {
                    "rows": row_count,
                    "features": args.features,
                    "method": name,
                    "latency_ms_median": statistics.median(samples),
                    "latency_ms_min": min(samples),
                    "latency_ms_max": max(samples),
                    "speedup_vs_dense": 0.0,
                    "stored_nonzero_values": nonzero_values,
                    "value_budget_ratio_vs_lr16": nonzero_values / lr16_params,
                    "density": 0.5,
                }
            )
        for row in rows:
            if int(row["rows"]) == row_count:
                row["speedup_vs_dense"] = timings["bf16_dense_2to4_values"] / row["latency_ms_median"]
        print(
            f"[sparse24] rows={row_count} dense={timings['bf16_dense_2to4_values']:.4f}ms "
            f"sparse={timings['cusparselt_2to4']:.4f}ms",
            flush=True,
        )

    fields = list(rows[0])
    with (output / "sparse24_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "arguments": vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "sparse_tensor_type": type(sparse_weight).__name__,
        "elapsed_seconds": time.time() - started,
        "scope": "Residual branch kernel only; no FP8 main branch included",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
