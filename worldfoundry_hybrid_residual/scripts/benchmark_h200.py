#!/usr/bin/env python3
"""Measure H200 latency of native FP8 main weights plus residual branches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from probe_decompositions import (
    bcm_project,
    fp8_tensorwise,
    load_activation,
    load_hadamard_module,
    load_weight,
    lowrank_dense,
    random_hadamard,
    randomized_svd,
    top_output_block_residual,
)


def quantize_fp8_payload(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e4m3fn
    limit = float(torch.finfo(dtype).max)
    scale = (value.detach().float().abs().amax() / limit).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (value.float() / scale).clamp(-limit, limit).to(dtype)
    return quantized.contiguous(), scale.reshape(1).float()


def fp8_scaled_mm(
    input_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch._scaled_mm(
        input_fp8,
        weight_fp8.T,
        input_scale,
        weight_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )


def lowrank_forward(
    value: torch.Tensor,
    factors,
    rank: int,
) -> torch.Tensor:
    down = factors.v[:, :rank].T.to(torch.bfloat16)
    up = (factors.u[:, :rank] * factors.s[:rank]).to(torch.bfloat16)
    return F.linear(F.linear(value, down), up)


def bcm_fft_forward(
    value: torch.Tensor,
    generators: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    rows = value.shape[0]
    in_blocks = value.shape[1] // block_size
    out_blocks = generators.shape[0]
    # PyTorch 2.9 can create complex-half FFT outputs on SM90, but its batched
    # complex contraction backend does not implement ComplexHalf. Keep this
    # path honest and portable with complex64 instead of silently falling back
    # to a dense BCM matrix.
    x_blocks = value.float().view(rows, in_blocks, block_size)
    generator = generators.float()
    x_fft = torch.fft.rfft(x_blocks, dim=-1)
    g_fft = torch.fft.rfft(generator, dim=-1)
    y_fft = torch.einsum("mif,oif->mof", x_fft, g_fft.conj())
    return torch.fft.irfft(y_fft, n=block_size, dim=-1).reshape(rows, out_blocks * block_size).to(torch.bfloat16)


def bcm_fft_forward_cached(
    value: torch.Tensor,
    generator_fft: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply BCM using a generator spectrum prepared outside the timed path."""
    rows = value.shape[0]
    in_blocks = value.shape[1] // block_size
    out_blocks = generator_fft.shape[0]
    x_blocks = value.float().view(rows, in_blocks, block_size)
    x_fft = torch.fft.rfft(x_blocks, dim=-1)
    y_fft = torch.einsum("mif,oif->mof", x_fft, generator_fft.conj())
    return torch.fft.irfft(y_fft, n=block_size, dim=-1).reshape(
        rows, out_blocks * block_size
    ).to(torch.bfloat16)


def relative_output_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> tuple[float, float]:
    rel = (reference.float() - estimate.float()).norm() / reference.float().norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(
        reference.float().reshape(1, -1), estimate.float().reshape(1, -1)
    ).item()
    return float(rel.item()), float(cosine)


def time_cuda(fn, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / iterations))
    return samples


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--key", default="blocks.0.self_attn.q.weight")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--rows", default="512,2048,7800,32760")
    parser.add_argument("--rank-budget", type=int, default=16)
    parser.add_argument("--bcm-block-size", type=int, default=128)
    parser.add_argument("--row-block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    hadamard_module, hadamard_path = load_hadamard_module()

    weight = load_weight(args.checkpoint, args.key, device)
    max_rows = max(int(item) for item in args.rows.split(","))
    activation, source = load_activation(
        args.activation, weight.shape[1], max_rows, device, args.seed
    )
    if activation.shape[0] < max_rows:
        repeats = math.ceil(max_rows / activation.shape[0])
        activation = activation.repeat(repeats, 1)[:max_rows]
        source += ":repeated"
    torch.manual_seed(args.seed)
    signs = torch.randint(0, 2, (weight.shape[1],), device=device).float().mul_(2).sub_(1)
    weight = random_hadamard(weight, signs, hadamard_module)
    activation = random_hadamard(activation, signs, hadamard_module).to(torch.bfloat16)

    out_features, in_features = weight.shape
    target_params = args.rank_budget * (out_features + in_features)
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    weight_fp8, weight_scale = quantize_fp8_payload(weight)
    dequantized = fp8_tensorwise(weight)
    fp8_error = weight - dequantized
    max_factor_rank = max(args.rank_budget, args.rank_budget // 2)
    error_factors = randomized_svd(fp8_error, max_factor_rank, seed=args.seed + 1)

    bcm, generators = bcm_project(fp8_error, args.bcm_block_size)
    bcm_params = int(generators.numel())
    # FFT(generator) is a weight-preparation artifact, not a per-token cost.
    # Keep the original on-demand path for comparison and add the deployable
    # cached path separately.
    generators_fft = torch.fft.rfft(generators.float(), dim=-1).contiguous()
    bcm_rank = max(0, (target_params - bcm_params) // (out_features + in_features))
    bcm_factors = randomized_svd(
        fp8_error - bcm, max(1, bcm_rank), seed=args.seed + 2
    )

    base_rank = max(1, args.rank_budget // 2)
    base_lr = lowrank_dense(error_factors, base_rank)
    sparse_budget = max(0, target_params - base_rank * (out_features + in_features))
    row_sparse, row_indices, row_metadata = top_output_block_residual(
        fp8_error - base_lr, sparse_budget, args.row_block_size
    )
    row_weight = row_sparse[row_indices].to(torch.bfloat16).contiguous()

    rows_out: list[dict] = []
    started = time.time()
    for row_count in [int(item) for item in args.rows.split(",") if item.strip()]:
        x = activation[:row_count].contiguous()
        x_fp8, x_scale = quantize_fp8_payload(x)

        def dense():
            return F.linear(x, weight_bf16)

        def dynamic_main():
            payload, scale = quantize_fp8_payload(x)
            return fp8_scaled_mm(payload, weight_fp8, scale, weight_scale)

        def static_main():
            return fp8_scaled_mm(x_fp8, weight_fp8, x_scale, weight_scale)

        def dynamic_lr16():
            return dynamic_main() + lowrank_forward(x, error_factors, args.rank_budget)

        def static_lr16():
            return static_main() + lowrank_forward(x, error_factors, args.rank_budget)

        def dynamic_bcm_lr():
            output = dynamic_main() + bcm_fft_forward(x, generators, args.bcm_block_size)
            if bcm_rank:
                output = output + lowrank_forward(x, bcm_factors, bcm_rank)
            return output

        def static_bcm_lr():
            output = static_main() + bcm_fft_forward(x, generators, args.bcm_block_size)
            if bcm_rank:
                output = output + lowrank_forward(x, bcm_factors, bcm_rank)
            return output

        def dynamic_bcm_lr_cached():
            output = dynamic_main() + bcm_fft_forward_cached(
                x, generators_fft, args.bcm_block_size
            )
            if bcm_rank:
                output = output + lowrank_forward(x, bcm_factors, bcm_rank)
            return output

        def static_bcm_lr_cached():
            output = static_main() + bcm_fft_forward_cached(
                x, generators_fft, args.bcm_block_size
            )
            if bcm_rank:
                output = output + lowrank_forward(x, bcm_factors, bcm_rank)
            return output

        def row_branch() -> torch.Tensor:
            if not row_indices.numel():
                return x.new_zeros((x.shape[0], out_features))
            partial = F.linear(x, row_weight)
            output = x.new_zeros((x.shape[0], out_features))
            output.index_add_(1, row_indices, partial)
            return output

        def dynamic_lr_rows():
            return dynamic_main() + lowrank_forward(x, error_factors, base_rank) + row_branch()

        def static_lr_rows():
            return static_main() + lowrank_forward(x, error_factors, base_rank) + row_branch()

        methods = {
            "bf16_dense": (dense, "measured", 16, 0, 0),
            "fp8_dynamic_main": (dynamic_main, "measured", 8, 0, 0),
            "fp8_static_input_main": (static_main, "lower_bound", 8, 0, 0),
            f"fp8_dynamic_svd_r{args.rank_budget}": (
                dynamic_lr16,
                "measured",
                8,
                target_params,
                0,
            ),
            f"fp8_static_input_svd_r{args.rank_budget}": (
                static_lr16,
                "lower_bound",
                8,
                target_params,
                0,
            ),
            f"fp8_dynamic_bcm_b{args.bcm_block_size}_svd_r{bcm_rank}": (
                dynamic_bcm_lr,
                "measured",
                8,
                bcm_params + bcm_rank * (out_features + in_features),
                0,
            ),
            f"fp8_static_input_bcm_b{args.bcm_block_size}_svd_r{bcm_rank}": (
                static_bcm_lr,
                "lower_bound",
                8,
                bcm_params + bcm_rank * (out_features + in_features),
                0,
            ),
            f"fp8_dynamic_cached_bcm_b{args.bcm_block_size}_svd_r{bcm_rank}": (
                dynamic_bcm_lr_cached,
                "measured",
                8,
                bcm_params + bcm_rank * (out_features + in_features),
                0,
            ),
            f"fp8_static_input_cached_bcm_b{args.bcm_block_size}_svd_r{bcm_rank}": (
                static_bcm_lr_cached,
                "lower_bound",
                8,
                bcm_params + bcm_rank * (out_features + in_features),
                0,
            ),
            f"fp8_dynamic_svd_r{base_rank}_rowsparse_b{args.row_block_size}": (
                dynamic_lr_rows,
                "measured",
                8,
                base_rank * (out_features + in_features) + int(row_weight.numel()),
                row_metadata,
            ),
            f"fp8_static_input_svd_r{base_rank}_rowsparse_b{args.row_block_size}": (
                static_lr_rows,
                "lower_bound",
                8,
                base_rank * (out_features + in_features) + int(row_weight.numel()),
                row_metadata,
            ),
        }

        reference = dense()
        torch.cuda.synchronize()
        baseline_ms = None
        local_rows = []
        for name, (fn, scope, main_bits, residual_params, metadata_bits) in methods.items():
            try:
                estimate = fn()
                torch.cuda.synchronize()
                rel_l2, cosine = relative_output_metrics(reference, estimate)
                samples = time_cuda(fn, args.warmup, args.iterations, args.repeats)
                median_ms = statistics.median(samples)
                if name == "bf16_dense":
                    baseline_ms = median_ms
                local_rows.append(
                    {
                        "weight_key": args.key,
                        "shape": f"{out_features}x{in_features}",
                        "activation_source": source,
                        "rows": row_count,
                        "method": name,
                        "measurement_scope": scope,
                        "latency_ms_median": median_ms,
                        "latency_ms_min": min(samples),
                        "latency_ms_max": max(samples),
                        "output_relative_l2": rel_l2,
                        "output_cosine": cosine,
                        "main_weight_bits": main_bits,
                        "high_precision_params": residual_params,
                        "metadata_bits": metadata_bits,
                    }
                )
            except Exception as exc:
                local_rows.append(
                    {
                        "weight_key": args.key,
                        "shape": f"{out_features}x{in_features}",
                        "activation_source": source,
                        "rows": row_count,
                        "method": name,
                        "measurement_scope": scope,
                        "status": "error",
                        "error": repr(exc),
                    }
                )
        for row in local_rows:
            if baseline_ms and row.get("latency_ms_median"):
                row["speedup_vs_bf16"] = baseline_ms / row["latency_ms_median"]
                row["status"] = "ok"
        rows_out.extend(local_rows)
        write_csv(output_dir / "h200_benchmark.partial.csv", rows_out)
        print(
            f"[bench] rows={row_count}: "
            + ", ".join(
                f"{row['method']}={row.get('latency_ms_median', float('nan')):.4f}ms"
                for row in local_rows
            ),
            flush=True,
        )

    write_csv(output_dir / "h200_benchmark.csv", rows_out)
    manifest = {
        "arguments": vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "hadamard_source": str(hadamard_path),
        "elapsed_seconds": time.time() - started,
        "interpretation": {
            "measured": "Includes dynamic activation quantization and all eager residual kernels",
            "lower_bound": "Input activation was quantized before timing; not an end-to-end implementation",
            "cached_bcm": "Generator FFT is prepared once before timing; only input FFT, contraction, inverse FFT, and residual low-rank kernels are timed",
            "ternary": "Not timed because no native ternary H200 kernel is installed",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[bench] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
