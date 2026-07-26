#!/usr/bin/env python3
"""Measure whether DiT speculative verification scales sublinearly on H200.

Classic speculative decoding is only useful when verifying a block of drafts
costs much less than evaluating the same target states sequentially.  Long
video DiTs already expose large GEMMs and attention kernels, so this benchmark
tests that prerequisite directly on captured Wan QKV tensors.
"""

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


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,2,4"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def benchmark(
    function: Callable[[], torch.Tensor], warmup: int, repetitions: int
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from sageattention.fa3_wrapper import fa3

    rows: list[dict[str, object]] = []
    replay_metadata: list[dict[str, object]] = []
    started = time.time()
    for replay_path in args.replay:
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        frame_num = int(metadata.get("frame_num", 0))
        case = f"F{frame_num}" if frame_num else replay_path.stem
        replay_id = replay_path.stem
        scalar_metadata = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        replay_metadata.append(
            {"case": case, "replay_id": replay_id, "path": str(replay_path), **metadata}
        )
        q1 = payload["q"].to(device=device).contiguous()
        k1 = payload["k"].to(device=device).contiguous()
        v1 = payload["v"].to(device=device).contiguous()
        hidden = q1.shape[2] * q1.shape[3]
        if hidden != 1536:
            raise ValueError(f"expected Wan hidden size 1536, received {hidden}")
        scale = float(payload.get("softmax_scale", q1.shape[-1] ** -0.5))

        generator = torch.Generator(device=device).manual_seed(args.seed + frame_num)
        qkv_weight = torch.randn(
            hidden, 3 * hidden, generator=generator, device=device, dtype=q1.dtype
        ) / hidden**0.5
        up_weight = torch.randn(
            hidden, 8960, generator=generator, device=device, dtype=q1.dtype
        ) / hidden**0.5
        down_weight = torch.randn(
            8960, hidden, generator=generator, device=device, dtype=q1.dtype
        ) / 8960**0.5

        for batch in args.batches:
            q = q1.repeat(batch, 1, 1, 1).contiguous()
            k = k1.repeat(batch, 1, 1, 1).contiguous()
            v = v1.repeat(batch, 1, 1, 1).contiguous()
            x = q.flatten(2)
            operations: dict[str, Callable[[], torch.Tensor]] = {
                "fa3_bf16_attention": lambda q=q, k=k, v=v: fa3(
                    q, k, v, tensor_layout="NHD", sm_scale=scale
                ),
                "qkv_bf16_gemm": lambda x=x: x @ qkv_weight,
                "ffn_bf16_eager": lambda x=x: F.gelu(
                    x @ up_weight, approximate="tanh"
                )
                @ down_weight,
            }
            for operation, function in operations.items():
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    median, minimum, maximum = benchmark(
                        function, args.warmup, args.repetitions
                    )
                    rows.append(
                        {
                            "case": case,
                            "replay_id": replay_id,
                            "frame_num": frame_num,
                            **scalar_metadata,
                            "operation": operation,
                            "batch": batch,
                            "tokens": q.shape[1],
                            "heads": q.shape[2],
                            "head_dim": q.shape[3],
                            "latency_ms_median": median,
                            "latency_ms_min": minimum,
                            "latency_ms_max": maximum,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                            / (1024.0**2),
                            "status": "ok",
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "case": case,
                            "replay_id": replay_id,
                            "frame_num": frame_num,
                            **scalar_metadata,
                            "operation": operation,
                            "batch": batch,
                            "status": "error",
                            "error": repr(error),
                        }
                    )
                print(
                    f"[spec-batch] case={case} operation={operation} batch={batch}",
                    flush=True,
                )
            del q, k, v, x
            torch.cuda.empty_cache()
        del payload, q1, k1, v1, qkv_weight, up_weight, down_weight
        torch.cuda.empty_cache()

    result = rows
    for row in result:
        if row.get("status") != "ok":
            continue
        baseline = next(
            (
                candidate
                for candidate in result
                if candidate.get("status") == "ok"
                and candidate["replay_id"] == row["replay_id"]
                and candidate["operation"] == row["operation"]
                and candidate["batch"] == 1
            ),
            None,
        )
        if baseline is None:
            continue
        batch = int(row["batch"])
        base_ms = float(baseline["latency_ms_median"])
        current_ms = float(row["latency_ms_median"])
        row["latency_ratio_vs_batch1"] = current_ms / base_ms
        row["verification_parallel_efficiency"] = batch * base_ms / current_ms
        row["per_candidate_latency_ratio"] = current_ms / (batch * base_ms)

    write_csv(args.output_dir / "speculative_batch_benchmark.csv", result)
    manifest = {
        "arguments": {
            key: [str(item) for item in value]
            if key == "replay"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "replays": replay_metadata,
        "methodology": {
            "timing": "CUDA events after warmup",
            "attention": "FA3 BF16 on repeated captured QKV",
            "linear": "BF16 random-weight shape proxy using captured Q as block input",
            "interpretation": "efficiency above 1 means batched verification is sublinear; this is necessary but not sufficient for speculative end-to-end speedup",
            "warning": "operator batch scaling is an optimistic prerequisite test, not a full speculative rollout",
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
    print(f"[spec-batch] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
