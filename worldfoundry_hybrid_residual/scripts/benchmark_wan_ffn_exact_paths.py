#!/usr/bin/env python3
"""Benchmark exact execution paths for complete Wan FFN modules on one GPU.

The benchmark measures the checkpoint's real ``Linear -> GELU -> Linear``
modules.  It does not infer that ``torch.compile`` fused either GEMM epilogue;
the implementation selected by PyTorch is intentionally treated as opaque.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


DEFAULT_CASES = "F17:7800,F81:32760"
DEFAULT_LAYERS = "0,14,29"
DEFAULT_COMPILE_MODES = "default,reduce-overhead,max-autotune"
AMORTIZATION_CALLS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--compile-modes", default=DEFAULT_COMPILE_MODES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--graph-capture-warmup", type=int, default=3)
    parser.add_argument("--amortization-calls", type=int, default=AMORTIZATION_CALLS)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError("integer list must contain nonnegative values")
    if len(set(values)) != len(values):
        raise ValueError("integer list must not contain duplicates")
    return values


def parse_cases(raw: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        name, separator, rows = item.partition(":")
        if not separator or not name.strip() or int(rows) <= 0:
            raise ValueError(f"invalid case {item!r}; expected NAME:POSITIVE_ROWS")
        cases.append((name.strip(), int(rows)))
    if not cases or len({name for name, _ in cases}) != len(cases):
        raise ValueError("cases must contain unique names")
    return cases


def parse_compile_modes(raw: str) -> list[str]:
    allowed = {"default", "reduce-overhead", "max-autotune"}
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unsupported compile modes: {unknown}")
    return list(dict.fromkeys(modes))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def time_cuda_calls(
    function: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output
    return samples


@torch.inference_mode()
def correctness_metrics(
    estimate: torch.Tensor, reference: torch.Tensor
) -> dict[str, object]:
    bitwise_equal = bool(torch.equal(estimate, reference))
    delta = estimate.float() - reference.float()
    denominator = max(float(reference.float().norm()), torch.finfo(torch.float32).tiny)
    return {
        "bitwise_equal": bitwise_equal,
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm()) / denominator,
    }


def memory_snapshot(device: torch.device, baseline_allocated: int) -> dict[str, int]:
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
    }


def timing_summary(samples: list[float]) -> dict[str, float]:
    return {
        "latency_ms_min": min(samples),
        "latency_ms_median": statistics.median(samples),
        "latency_ms_p95": percentile(samples, 0.95),
        "latency_ms_max": max(samples),
        "latency_ms_mean": statistics.fmean(samples),
    }


def reset_compile_cache() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if reset is not None:
        reset()


@dataclass
class PreparedPath:
    name: str
    function: Callable[[], torch.Tensor]
    setup_ms: float
    notes: str
    keepalive: tuple[object, ...] = ()


@torch.inference_mode()
def prepare_eager(ffn: torch.nn.Module, value: torch.Tensor) -> PreparedPath:
    return PreparedPath(
        name="eager",
        function=lambda: ffn(value),
        setup_ms=0.0,
        notes="checkpoint FFN eager reference",
    )


@torch.inference_mode()
def prepare_compile(
    ffn: torch.nn.Module, value: torch.Tensor, mode: str
) -> PreparedPath:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    torch.cuda.synchronize()
    started = time.perf_counter()
    compiled = torch.compile(ffn, mode=mode, fullgraph=True, dynamic=False)
    first_output = compiled(value)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - started) * 1000.0
    return PreparedPath(
        name=f"compile_{mode}",
        function=lambda: compiled(value),
        setup_ms=setup_ms,
        notes=(
            "full FFN torch.compile path; backend fusion is implementation-dependent "
            "and no GEMM-epilogue fusion is claimed"
        ),
        keepalive=(compiled, first_output),
    )


@torch.inference_mode()
def prepare_cuda_graph(
    ffn: torch.nn.Module, value: torch.Tensor, capture_warmup: int
) -> PreparedPath:
    static_value = value.clone()
    side_stream = torch.cuda.Stream(device=value.device)
    side_stream.wait_stream(torch.cuda.current_stream(value.device))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.cuda.stream(side_stream):
        for _ in range(capture_warmup):
            static_output = ffn(static_value)
    torch.cuda.current_stream(value.device).wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = ffn(static_value)
    graph.replay()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - started) * 1000.0

    def replay() -> torch.Tensor:
        graph.replay()
        return static_output

    return PreparedPath(
        name="cuda_graph_eager_static",
        function=replay,
        setup_ms=setup_ms,
        notes=(
            "static-address CUDA Graph replay of the complete eager FFN; input-copy "
            "and integration into a larger graph are excluded"
        ),
        keepalive=(graph, static_value, static_output, side_stream),
    )


def benchmark_path(
    prepared: PreparedPath,
    reference: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    amortization_calls: int,
    baseline_allocated: int,
) -> dict[str, object]:
    estimate = prepared.function().detach().clone()
    torch.cuda.synchronize()
    metrics = correctness_metrics(estimate, reference)
    samples = time_cuda_calls(prepared.function, warmup, repeats)
    row: dict[str, object] = {
        "path": prepared.name,
        "status": "ok",
        "setup_ms": prepared.setup_ms,
        "amortization_calls": amortization_calls,
        "amortized_latency_ms": statistics.median(samples)
        + prepared.setup_ms / amortization_calls,
        "notes": prepared.notes,
    }
    row.update(timing_summary(samples))
    row.update(metrics)
    row.update(memory_snapshot(device, baseline_allocated))
    return row


def load_checkpoint_ffns(
    wan_source: Path, checkpoint: Path, layers: list[int]
) -> tuple[dict[int, torch.nn.Module], float, dict[str, int]]:
    sys.path.insert(0, str(wan_source))
    os.chdir(wan_source)
    from wan.modules.model import WanModel

    started = time.perf_counter()
    model = WanModel.from_pretrained(str(checkpoint))
    model.eval().requires_grad_(False)
    if max(layers) >= len(model.blocks):
        raise ValueError(f"layer index exceeds checkpoint depth {len(model.blocks)}")
    ffns = {layer: model.blocks[layer].ffn for layer in layers}
    dimensions = {
        "hidden_dim": int(ffns[layers[0]][0].in_features),
        "ffn_dim": int(ffns[layers[0]][0].out_features),
        "num_layers": len(model.blocks),
    }
    for layer, ffn in ffns.items():
        if not isinstance(ffn, torch.nn.Sequential) or len(ffn) != 3:
            raise TypeError(f"layer {layer} does not expose the expected complete FFN")
        if int(ffn[0].in_features) != dimensions["hidden_dim"]:
            raise ValueError("selected FFN layers have inconsistent hidden dimensions")
    load_seconds = time.perf_counter() - started
    del model
    gc.collect()
    return ffns, load_seconds, dimensions


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_capture_warmup < 1:
        raise ValueError("warmup must be nonnegative and repeat counts must be positive")
    if args.amortization_calls <= 0:
        raise ValueError("amortization calls must be positive")
    layers = parse_ints(args.layers)
    cases = parse_cases(args.cases)
    compile_modes = parse_compile_modes(args.compile_modes)
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    ffns, load_seconds, dimensions = load_checkpoint_ffns(
        args.wan_source, args.checkpoint, layers
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for layer in layers:
        ffn = ffns[layer].to(device=device, dtype=torch.bfloat16).eval()
        for case_index, (case, token_rows) in enumerate(cases):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + layer * 1009 + case_index
            )
            value = torch.randn(
                1,
                token_rows,
                dimensions["hidden_dim"],
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            with torch.inference_mode():
                reference = ffn(value).detach().clone()
            torch.cuda.synchronize()
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            preparations: list[tuple[str, Callable[[], PreparedPath]]] = [
                ("eager", lambda ffn=ffn, value=value: prepare_eager(ffn, value)),
                (
                    "cuda_graph_eager_static",
                    lambda ffn=ffn, value=value: prepare_cuda_graph(
                        ffn, value, args.graph_capture_warmup
                    ),
                ),
            ]
            preparations.extend(
                (
                    f"compile_{mode}",
                    lambda mode=mode, ffn=ffn, value=value: prepare_compile(
                        ffn, value, mode
                    ),
                )
                for mode in compile_modes
            )
            for path_name, prepare in preparations:
                torch.cuda.reset_peak_memory_stats(device)
                prepared: PreparedPath | None = None
                try:
                    prepared = prepare()
                    row = benchmark_path(
                        prepared,
                        reference,
                        device,
                        args.warmup,
                        args.repeats,
                        args.amortization_calls,
                        baseline_allocated,
                    )
                except Exception as error:
                    row = {
                        "path": path_name,
                        "status": "error",
                        "error": repr(error),
                        "setup_ms": "",
                        "notes": "path preparation or execution failed",
                    }
                row = {
                    "case": case,
                    "token_rows": token_rows,
                    "input_shape": f"1x{token_rows}x{dimensions['hidden_dim']}",
                    "layer": layer,
                    "hidden_dim": dimensions["hidden_dim"],
                    "ffn_dim": dimensions["ffn_dim"],
                    "dtype": str(value.dtype),
                } | row
                rows.append(row)
                write_csv(args.output_dir / "wan_ffn_exact_paths.partial.csv", rows)
                print(
                    f"[ffn-exact] case={case} layer={layer} path={path_name} "
                    f"status={row['status']}",
                    flush=True,
                )
                del prepared
                if path_name.startswith("compile_"):
                    reset_compile_cache()
                gc.collect()
                torch.cuda.empty_cache()
            del value, reference
            gc.collect()
            torch.cuda.empty_cache()
        ffn.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "wan_ffn_exact_paths.csv", rows)
    manifest = {
        "scope": "checkpoint-faithful complete Wan FFN exact-path microbenchmark",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": dict(cases),
        "layers": layers,
        "compile_modes_requested": compile_modes,
        "dimensions": dimensions,
        "model_load_seconds": load_seconds,
        "benchmark_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "warning": (
            "torch.compile is benchmarked as an opaque full-FFN execution path. "
            "These measurements do not establish GEMM-epilogue fusion. CUDA Graph "
            "numbers use static addresses and exclude an input copy."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ffn-exact] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
