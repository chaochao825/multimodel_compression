#!/usr/bin/env python3
"""Benchmark online FFT and fused FFN pointwise costs on Hopper."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.jit
    def _bias_gelu_tanh_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        rows,
        width,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        column = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = (row < rows) & (column < width)
        offsets = row * width + column
        value = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        bias = tl.load(bias_ptr + column, mask=column < width).to(tl.float32)
        value += bias
        inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
        tanh_inner = libdevice.tanh(inner)
        result = 0.5 * value * (1.0 + tanh_inner)
        tl.store(out_ptr + offsets, result, mask=mask)


def fused_bias_gelu_tanh(value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    output = torch.empty_like(value)
    block_m = 8
    block_n = 256
    grid = (triton.cdiv(value.shape[0], block_m), triton.cdiv(value.shape[1], block_n))
    _bias_gelu_tanh_kernel[grid](
        value,
        bias,
        output,
        value.shape[0],
        value.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output


def eager_bias_gelu_tanh(value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(value + bias, approximate="tanh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--rows", default="7800,32760")
    parser.add_argument("--widths", default="1536,8960")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iterations: int, repeats: int) -> list[float]:
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    rows_out: list[dict[str, object]] = []
    started = time.time()
    compiled_bias_gelu = None
    if hasattr(torch, "compile"):
        try:
            compiled_bias_gelu = torch.compile(
                eager_bias_gelu_tanh,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        except Exception:
            compiled_bias_gelu = None

    for row_count in [int(item) for item in args.rows.split(",") if item.strip()]:
        for width in [int(item) for item in args.widths.split(",") if item.strip()]:
            value = torch.randn(row_count, width, device=device, dtype=torch.bfloat16)
            bias = torch.randn(width, device=device, dtype=torch.bfloat16)
            reference = F.gelu(value + bias, approximate="tanh")
            methods: dict[str, Callable[[], torch.Tensor]] = {
                "copy": lambda value=value: value.clone(),
                "gelu_tanh": lambda value=value: F.gelu(value, approximate="tanh"),
                "bias_plus_gelu_tanh_eager": lambda value=value, bias=bias: F.gelu(
                    value + bias, approximate="tanh"
                ),
                "rfft_fp32_forward": lambda value=value: torch.fft.rfft(
                    value.float(), dim=-1, norm="ortho"
                ),
                "rfft_bf16_forward": lambda value=value: torch.fft.rfft(
                    value, dim=-1, norm="ortho"
                ),
                "rfft_fp32_roundtrip": lambda value=value, width=width: torch.fft.irfft(
                    torch.fft.rfft(value.float(), dim=-1, norm="ortho"),
                    n=width,
                    dim=-1,
                    norm="ortho",
                ).to(torch.bfloat16),
            }
            if triton is not None:
                methods["bias_plus_gelu_tanh_triton"] = lambda value=value, bias=bias: fused_bias_gelu_tanh(
                    value, bias
                )
            if compiled_bias_gelu is not None:
                methods["bias_plus_gelu_tanh_compile"] = lambda value=value, bias=bias: compiled_bias_gelu(
                    value, bias
                )
            for name, fn in methods.items():
                try:
                    estimate = fn().clone()
                    torch.cuda.synchronize()
                    samples = time_cuda(fn, args.warmup, args.iterations, args.repeats)
                    row = {
                        "rows": row_count,
                        "width": width,
                        "operation": name,
                        "latency_ms_median": statistics.median(samples),
                        "latency_ms_min": min(samples),
                        "latency_ms_max": max(samples),
                        "status": "ok",
                    }
                    if name.startswith("bias_plus_gelu"):
                        delta = estimate.float() - reference.float()
                        row["relative_l2_vs_torch"] = float(delta.norm() / reference.float().norm())
                        row["max_abs_vs_torch"] = float(delta.abs().max())
                    rows_out.append(row)
                except Exception as error:
                    rows_out.append(
                        {
                            "rows": row_count,
                            "width": width,
                            "operation": name,
                            "status": "error",
                            "error": repr(error),
                        }
                    )
            del value, bias, reference
            torch.cuda.empty_cache()
            write_csv(args.output_dir / "h200_ffn_transform_benchmark.partial.csv", rows_out)
            print(f"[h200] rows={row_count} width={width}", flush=True)

    hidden = 1536
    intermediate = 8960
    weight_up = torch.randn(intermediate, hidden, device=device, dtype=torch.bfloat16)
    bias_up = torch.randn(intermediate, device=device, dtype=torch.bfloat16)
    weight_down = torch.randn(hidden, intermediate, device=device, dtype=torch.bfloat16)
    bias_down = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    for row_count in [int(item) for item in args.rows.split(",") if item.strip()]:
        value = torch.randn(row_count, hidden, device=device, dtype=torch.bfloat16)

        def up_linear_no_bias() -> torch.Tensor:
            return F.linear(value, weight_up)

        def up_linear_bias() -> torch.Tensor:
            return F.linear(value, weight_up, bias_up)

        def up_linear_bias_gelu() -> torch.Tensor:
            return F.gelu(F.linear(value, weight_up, bias_up), approximate="tanh")

        def up_linear_triton_bias_gelu() -> torch.Tensor:
            return fused_bias_gelu_tanh(F.linear(value, weight_up), bias_up)

        def full_ffn_eager() -> torch.Tensor:
            hidden_value = F.gelu(
                F.linear(value, weight_up, bias_up), approximate="tanh"
            )
            return F.linear(hidden_value, weight_down, bias_down)

        methods = {
            "ffn_up_linear_no_bias": up_linear_no_bias,
            "ffn_up_linear_bias": up_linear_bias,
            "ffn_up_linear_bias_gelu_eager": up_linear_bias_gelu,
            "ffn_full_eager": full_ffn_eager,
        }
        if triton is not None:
            methods["ffn_up_linear_triton_bias_gelu"] = up_linear_triton_bias_gelu
        reference = up_linear_bias_gelu().clone()
        torch.cuda.synchronize()
        for name, fn in methods.items():
            try:
                estimate = fn().clone()
                torch.cuda.synchronize()
                samples = time_cuda(fn, args.warmup, args.iterations, args.repeats)
                row = {
                    "rows": row_count,
                    "width": intermediate,
                    "operation": name,
                    "scope": "wan_ffn_shape",
                    "latency_ms_median": statistics.median(samples),
                    "latency_ms_min": min(samples),
                    "latency_ms_max": max(samples),
                    "status": "ok",
                }
                if name == "ffn_up_linear_triton_bias_gelu":
                    delta = estimate.float() - reference.float()
                    row["relative_l2_vs_torch"] = float(
                        delta.norm() / reference.float().norm()
                    )
                    row["max_abs_vs_torch"] = float(delta.abs().max())
                rows_out.append(row)
            except Exception as error:
                rows_out.append(
                    {
                        "rows": row_count,
                        "width": intermediate,
                        "operation": name,
                        "scope": "wan_ffn_shape",
                        "status": "error",
                        "error": repr(error),
                    }
                )
        del value, reference
        torch.cuda.empty_cache()
        write_csv(args.output_dir / "h200_ffn_transform_benchmark.partial.csv", rows_out)
        print(f"[h200] Wan FFN rows={row_count}", flush=True)

    write_csv(args.output_dir / "h200_ffn_transform_benchmark.csv", rows_out)
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "triton": None if triton is None else triton.__version__,
        "elapsed_seconds": time.time() - started,
        "interpretation": "FFT timings include BF16-to-FP32 conversion and exclude spectral sparse contraction. The Triton pointwise path fuses bias and tanh-GELU after an ordinary no-bias GEMM; it is not yet a GEMM epilogue.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[h200] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
