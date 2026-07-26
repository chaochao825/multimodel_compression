#!/usr/bin/env python3
"""Benchmark THW transforms and simple pooling against FA3 on captured Wan QKV."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--density", type=float, default=0.125)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    return parser.parse_args()


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


def infer_grid(frame_num: int, tokens: int, height: int, width: int) -> tuple[int, int, int]:
    temporal = (frame_num - 1) // 4 + 1
    if temporal * height * width != tokens:
        raise ValueError("captured token count does not match THW grid")
    return temporal, height, width


def lowpass_mask(
    shape: tuple[int, int, int], density: float, device: torch.device
) -> torch.Tensor:
    axes = [torch.fft.fftfreq(size, device=device).abs() / 0.5 for size in shape]
    radius = (
        axes[0][:, None, None].square()
        + axes[1][None, :, None].square()
        + axes[2][None, None, :].square()
    )
    count = max(1, min(radius.numel(), round(radius.numel() * density)))
    threshold = radius.flatten().sort().values[count - 1]
    return radius <= threshold


def benchmark(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    repetitions: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples), max(samples)


def thw_lowpass(grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = grid.float()
    mean = value.mean(dim=(0, 1, 2), keepdim=True)
    transformed = torch.fft.fftn(value - mean, dim=(0, 1, 2), norm="ortho")
    filtered = torch.where(mask[..., None], transformed, torch.zeros((), device=value.device))
    return torch.fft.ifftn(filtered, dim=(0, 1, 2), norm="ortho").real + mean


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 < args.density <= 1.0:
        raise ValueError("density must lie in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from sageattention.fa3_wrapper import fa3

    rows: list[dict[str, object]] = []
    replay_metadata: list[dict[str, object]] = []
    started = time.time()
    for replay_path in args.replay:
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        q = payload["q"].to(device=device).contiguous()
        k = payload["k"].to(device=device).contiguous()
        v = payload["v"].to(device=device).contiguous()
        frame_num = int(metadata["frame_num"])
        shape = infer_grid(frame_num, q.shape[1], args.grid_height, args.grid_width)
        case = f"f{frame_num}_l{metadata.get('layer', 'unknown')}_t{metadata.get('timestep', 'unknown')}"
        replay_metadata.append({"path": str(replay_path), "case": case, **metadata})
        mask = lowpass_mask(shape, args.density, device)
        q_grid = q[0].reshape(*shape, -1)
        k_grid = k[0].reshape(*shape, -1)
        scale = float(payload.get("softmax_scale", q.shape[-1] ** -0.5))

        operations: dict[str, Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]] = {
            "fa3_bf16_attention": lambda: fa3(
                q, k, v, tensor_layout="NHD", sm_scale=scale
            ),
            "q_thw_lowpass_roundtrip_fp32": lambda: thw_lowpass(q_grid, mask),
            "qk_thw_lowpass_roundtrip_fp32": lambda: (
                thw_lowpass(q_grid, mask),
                thw_lowpass(k_grid, mask),
            ),
            "q_spatial_pool2_bf16": lambda: q_grid.reshape(
                shape[0], shape[1] // 2, 2, shape[2] // 2, 2, -1
            ).mean(dim=(2, 4)),
            "qk_spatial_pool2_bf16": lambda: (
                q_grid.reshape(shape[0], shape[1] // 2, 2, shape[2] // 2, 2, -1).mean(dim=(2, 4)),
                k_grid.reshape(shape[0], shape[1] // 2, 2, shape[2] // 2, 2, -1).mean(dim=(2, 4)),
            ),
        }
        for operation, function in operations.items():
            try:
                median, minimum, maximum = benchmark(
                    function, args.warmup, args.repetitions
                )
                rows.append(
                    {
                        "case": case,
                        "operation": operation,
                        "latency_ms_median": median,
                        "latency_ms_min": minimum,
                        "latency_ms_max": maximum,
                        "status": "ok",
                        "tokens": q.shape[1],
                        "heads": q.shape[2],
                        "head_dim": q.shape[3],
                        "requested_density": args.density,
                        "actual_density": float(mask.float().mean()),
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "case": case,
                        "operation": operation,
                        "status": "error",
                        "error": repr(error),
                    }
                )
            print(f"[thw-bench] case={case} operation={operation}", flush=True)
        del payload, q, k, v, q_grid, k_grid
        torch.cuda.empty_cache()

    by_case = {row["case"]: row for row in rows if row["operation"] == "fa3_bf16_attention"}
    for row in rows:
        baseline = by_case.get(row["case"])
        if row.get("status") == "ok" and baseline and baseline.get("status") == "ok":
            row["latency_ratio_vs_fa3"] = float(row["latency_ms_median"]) / float(
                baseline["latency_ms_median"]
            )
    write_csv(args.output_dir / "h200_thw_router_benchmark.csv", rows)
    manifest = {
        "arguments": {
            key: [str(item) for item in value] if key == "replay" else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "replays": replay_metadata,
        "methodology": {
            "timing": "CUDA events after warmup",
            "fft": "PyTorch FP32 full THW FFT, lowpass mask, inverse FFT",
            "pool": "eager BF16 2x2 spatial reshape and mean",
            "warning": "unfused PyTorch cost gate, not a custom-kernel upper bound",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[thw-bench] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
