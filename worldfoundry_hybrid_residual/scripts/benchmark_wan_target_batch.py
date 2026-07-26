#!/usr/bin/env python3
"""Measure full Wan denoiser batch scaling for speculative verification on H200."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp

from generate_wan_cfg_parallel import make_scheduler, sequence_length, target_shape
from generate_wan_h200_v4 import AttentionDispatcher, install_grid_compatibility, load_backends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-nums", default="17,81")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--prompt", default="A red panda runs through a bamboo forest.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer lists must contain positive values")
    return values


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
def benchmark(function: object, warmup: int, repetitions: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        output = function()
        del output
    torch.cuda.synchronize()
    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
        del output
    return float(np.median(timings)), float(np.min(timings)), float(np.max(timings))


def main() -> None:
    args = parse_args()
    frame_nums = parse_ints(args.frame_nums)
    batches = parse_ints(args.batches)
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    if any((frame_num - 1) % 4 for frame_num in frame_nums):
        raise ValueError("every frame count must be 4n+1")
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.wan_source))
    os.chdir(args.wan_source)
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    import wan.modules.model as wan_model_module

    compatibility = install_grid_compatibility(wan_model_module)
    backends = load_backends()
    if "fa3_bf16" not in backends:
        raise RuntimeError("FA3 BF16 backend is required")
    dispatcher = AttentionDispatcher(
        backends, hybrid_layer_count=30, sampling_steps=args.sampling_steps
    )
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = dispatcher
    load_start = time.perf_counter()
    pipeline = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"],
        checkpoint_dir=str(args.checkpoint),
        device_id=device.index or 0,
        rank=0,
        t5_cpu=False,
    )
    pipeline.model.to(device=device, dtype=WAN_CONFIGS["t2v-1.3B"].param_dtype)
    pipeline.text_encoder.model.to(device)
    context_one = pipeline.text_encoder([args.prompt], device)
    load_seconds = time.perf_counter() - load_start

    rows: list[dict[str, object]] = []
    try:
        for frame_num in frame_nums:
            shape = target_shape(pipeline, frame_num, (args.width, args.height))
            seq_len = sequence_length(pipeline, shape)
            scheduler, timesteps = make_scheduler(
                "unipc",
                pipeline.num_train_timesteps,
                args.sampling_steps,
                args.shift,
                device,
            )
            del scheduler
            timestep_one = torch.stack([timesteps[0]])
            generator = torch.Generator(device=device).manual_seed(args.seed + frame_num)
            latent_one = torch.randn(
                *shape, dtype=torch.float32, device=device, generator=generator
            ).unsqueeze(0)
            for batch_size in batches:
                latent = latent_one.repeat(batch_size, 1, 1, 1, 1).contiguous()
                timestep = timestep_one.repeat(batch_size)
                contexts = list(context_one) * batch_size
                dispatcher.begin("fa3_bf16")

                def run() -> torch.Tensor:
                    with amp.autocast(dtype=pipeline.param_dtype):
                        return pipeline.model(
                            latent,
                            t=timestep,
                            context=contexts,
                            seq_len=seq_len,
                        )

                torch.cuda.reset_peak_memory_stats(device)
                try:
                    median, minimum, maximum = benchmark(
                        run, args.warmup, args.repetitions
                    )
                    rows.append(
                        {
                            "case": f"F{frame_num}",
                            "replay_id": f"F{frame_num}_initial_noise_t{float(timesteps[0]):g}",
                            "frame_num": frame_num,
                            "operation": "full_wan_model",
                            "batch": batch_size,
                            "tokens": seq_len,
                            "timestep": float(timesteps[0]),
                            "latency_ms_median": median,
                            "latency_ms_min": minimum,
                            "latency_ms_max": maximum,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                            / (1024.0**2),
                            "status": "ok",
                        }
                    )
                except torch.cuda.OutOfMemoryError as error:
                    rows.append(
                        {
                            "case": f"F{frame_num}",
                            "replay_id": f"F{frame_num}_initial_noise_t{float(timesteps[0]):g}",
                            "frame_num": frame_num,
                            "operation": "full_wan_model",
                            "batch": batch_size,
                            "status": "error",
                            "error": repr(error),
                        }
                    )
                print(
                    f"[wan-target-batch] F{frame_num} batch={batch_size} "
                    f"status={rows[-1]['status']}",
                    flush=True,
                )
                del latent, timestep, contexts
                torch.cuda.empty_cache()
            del latent_one, timestep_one, timesteps
    finally:
        wan_model_module.flash_attention = original_attention

    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = next(
            (
                candidate
                for candidate in rows
                if candidate.get("status") == "ok"
                and candidate["replay_id"] == row["replay_id"]
                and candidate["batch"] == 1
            ),
            None,
        )
        if baseline is None:
            continue
        base_ms = float(baseline["latency_ms_median"])
        current_ms = float(row["latency_ms_median"])
        batch_size = int(row["batch"])
        row["latency_ratio_vs_batch1"] = current_ms / base_ms
        row["verification_parallel_efficiency"] = batch_size * base_ms / current_ms
        row["per_candidate_latency_ratio"] = current_ms / (batch_size * base_ms)

    write_csv(args.output_dir / "wan_target_batch_benchmark.csv", rows)
    manifest = {
        "scope": "full Wan denoiser batch-scaling prerequisite for future-step verification",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methodology": {
            "state": "initial-noise latent at the first 20-step UniPC timestep",
            "precision": "BF16 model with FA3 BF16 attention",
            "timing": "CUDA events around the complete 30-block Wan model forward",
            "warning": (
                "This measures target verification cost but not draft acceptance, "
                "continuous residual sampling, or end-to-end speculative rollout."
            ),
        },
        "load_seconds": load_seconds,
        "grid_compatibility_installed": compatibility,
        "available_attention_backends": sorted(backends),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
