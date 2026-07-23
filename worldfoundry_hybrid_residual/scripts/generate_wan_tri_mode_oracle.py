#!/usr/bin/env python3
"""Collect real Wan rollouts for grouped dense/FP8/cache block schedules."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from generate_wan_h200_v4 import (
    AttentionDispatcher,
    install_grid_compatibility,
    load_backends,
    save_video,
)
from trajectory_budget_runtime import TriModeBlockController, load_schedule_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--worldfoundry-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--schedule-names", default="")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seeds", default="20260723")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--static-scale-margin", type=float, default=1.05)
    parser.add_argument("--calibration-steps", type=int, default=20)
    parser.add_argument("--precision-warmup-steps", type=int, default=1)
    parser.add_argument("--alternate-schedule-order", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        prompts = ["A red panda runs through a bamboo forest while the camera tracks smoothly."]
    return prompts[: args.max_prompts] if args.max_prompts > 0 else prompts


def parse_seeds(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


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


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    if args.sampling_steps <= 0 or args.calibration_steps <= 0:
        raise ValueError("sampling and calibration steps must be positive")
    if args.precision_warmup_steps < 0 or args.repeats <= 0:
        raise ValueError("warmup steps must be nonnegative and repeats positive")
    args.wan_source = args.wan_source.resolve()
    args.worldfoundry_source = args.worldfoundry_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.schedule_file = args.schedule_file.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    events_dir = args.out_dir / "events"
    events_dir.mkdir()
    prompts = load_prompts(args)
    seeds = parse_seeds(args.seeds)
    schedules = load_schedule_bundle(
        args.schedule_file,
        sampling_steps=args.sampling_steps,
        block_count=30,
    )
    requested_names = {
        token.strip() for token in args.schedule_names.split(",") if token.strip()
    }
    if requested_names:
        schedules = [schedule for schedule in schedules if schedule.name in requested_names]
        missing = requested_names - {schedule.name for schedule in schedules}
        if missing:
            raise ValueError(f"unknown schedule names: {sorted(missing)}")
    if not schedules:
        raise ValueError("no schedules selected")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed_all(seeds[0])
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, str(args.worldfoundry_source))
    sys.path.insert(0, str(args.wan_source))
    os.chdir(args.wan_source)

    from worldfoundry_hybrid_residual import replace_wan_linears
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    import wan.modules.model as wan_model_module

    grid_compatibility_installed = install_grid_compatibility(wan_model_module)
    backends = load_backends()
    if "fa3_bf16" not in backends:
        raise RuntimeError("FA3 BF16 backend is unavailable")
    dispatcher = AttentionDispatcher(
        backends,
        hybrid_layer_count=30,
        sampling_steps=args.sampling_steps,
    )
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = dispatcher

    print(
        f"STAGE model_load_start schedules={len(schedules)} "
        f"prompts={len(prompts)} seeds={len(seeds)} repeats={args.repeats}",
        flush=True,
    )
    load_started = time.perf_counter()
    pipeline = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"],
        checkpoint_dir=str(args.checkpoint),
        device_id=device.index or 0,
        rank=0,
        t5_cpu=False,
    )
    pipeline.model.to(device=device, dtype=WAN_CONFIGS["t2v-1.3B"].param_dtype)
    load_seconds = time.perf_counter() - load_started
    print(f"STAGE model_load_done seconds={load_seconds:.3f}", flush=True)

    setup_started = time.perf_counter()
    linear_controller = replace_wan_linears(
        pipeline.model,
        scope="ffn",
        targets=("up", "down"),
        blocks="all",
        residual_rank=0,
        budget_rank=0,
        row_block_size=8,
        seed=seeds[0],
        static_scale_margin=args.static_scale_margin,
    )
    replacement_summary = linear_controller.summary()
    print(
        f"STAGE fp8_setup_done seconds={time.perf_counter() - setup_started:.3f} "
        f"linears={len(replacement_summary.names)}",
        flush=True,
    )

    linear_controller.set_mode("dense")
    linear_controller.set_calibration(True)
    dispatcher.begin("fa3_bf16")
    print(f"STAGE calibration_start steps={args.calibration_steps}", flush=True)
    torch.cuda.synchronize(device)
    calibration_started = time.perf_counter()
    calibration_video = pipeline.generate(
        input_prompt=prompts[0],
        size=(args.width, args.height),
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.calibration_steps,
        guide_scale=args.guide_scale,
        n_prompt=args.negative_prompt,
        seed=seeds[0],
        offload_model=False,
    )
    torch.cuda.synchronize(device)
    calibration_seconds = time.perf_counter() - calibration_started
    del calibration_video
    linear_controller.finalize_calibration()
    print(f"STAGE calibration_done seconds={calibration_seconds:.3f}", flush=True)

    precision_warmup_seconds = 0.0
    if args.precision_warmup_steps:
        linear_controller.set_mode("fp8")
        dispatcher.begin("fa3_bf16")
        torch.cuda.synchronize(device)
        precision_started = time.perf_counter()
        precision_video = pipeline.generate(
            input_prompt=prompts[0],
            size=(args.width, args.height),
            frame_num=args.frame_num,
            shift=args.shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.precision_warmup_steps,
            guide_scale=args.guide_scale,
            n_prompt=args.negative_prompt,
            seed=seeds[0],
            offload_model=False,
        )
        torch.cuda.synchronize(device)
        precision_warmup_seconds = time.perf_counter() - precision_started
        del precision_video
        print(
            f"STAGE precision_warmup_done seconds={precision_warmup_seconds:.3f}",
            flush=True,
        )
    linear_controller.set_mode("dense")
    tri_mode = TriModeBlockController(
        pipeline.model,
        linear_controller,
        sampling_steps=args.sampling_steps,
    )

    rows: list[dict[str, object]] = []
    run_index = 0
    try:
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                for repeat in range(args.repeats):
                    ordered = list(schedules)
                    if args.alternate_schedule_order and ordered:
                        cycle_index = run_index // len(ordered)
                        if cycle_index % 2:
                            ordered.reverse()
                    for order_index, schedule in enumerate(ordered):
                        run_index += 1
                        tri_mode.begin(schedule)
                        dispatcher.begin("fa3_bf16")
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        filename = (
                            f"p{prompt_index:02d}_{schedule.name}_seed{seed}_r{repeat}.mp4"
                        )
                        try:
                            video = pipeline.generate(
                                input_prompt=prompt,
                                size=(args.width, args.height),
                                frame_num=args.frame_num,
                                shift=args.shift,
                                sample_solver=args.sample_solver,
                                sampling_steps=args.sampling_steps,
                                guide_scale=args.guide_scale,
                                n_prompt=args.negative_prompt,
                                seed=seed,
                                offload_model=False,
                            )
                            tri_mode.assert_complete()
                            torch.cuda.synchronize(device)
                            seconds = time.perf_counter() - started
                            video_path = args.out_dir / filename
                            save_video(video, video_path, args.fps)
                            event_file = (
                                events_dir
                                / f"p{prompt_index:02d}_{schedule.name}_seed{seed}_r{repeat}.jsonl"
                            )
                            tri_mode.write_events(event_file)
                            row: dict[str, object] = {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "seed": seed,
                                "repeat": repeat,
                                "method": schedule.name,
                                "schedule_order_index": order_index,
                                "status": "ok",
                                "seconds_including_text_and_vae": seconds,
                                "self_attention_calls": dispatcher.self_calls,
                                "cross_attention_calls": dispatcher.cross_calls,
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                                / (1024.0**2),
                                "video_file": filename,
                                "video_sha256": sha256(video_path),
                                "event_file": str(event_file.relative_to(args.out_dir)),
                                "schedule_family": schedule.metadata.get("family", ""),
                                "schedule_action": schedule.metadata.get("action", ""),
                                "schedule_cell_count": schedule.metadata.get("cell_count", ""),
                                "schedule_metadata": json.dumps(
                                    schedule.metadata, sort_keys=True
                                ),
                                **tri_mode.stats(),
                                **linear_controller.runtime_stats(),
                            }
                            rows.append(row)
                            print(
                                f"DONE schedule={schedule.name} seconds={seconds:.3f} "
                                f"Q={row['tri_mode_executed_quant']} "
                                f"C={row['tri_mode_executed_cache']} "
                                f"fallbacks={row['tri_mode_cache_fallbacks']}",
                                flush=True,
                            )
                            del video
                        except Exception as error:
                            torch.cuda.synchronize(device)
                            traceback.print_exc()
                            rows.append(
                                {
                                    "prompt_index": prompt_index,
                                    "prompt": prompt,
                                    "seed": seed,
                                    "repeat": repeat,
                                    "method": schedule.name,
                                    "schedule_order_index": order_index,
                                    "status": "error",
                                    "seconds_including_text_and_vae": time.perf_counter()
                                    - started,
                                    "error": f"{type(error).__name__}: {error}",
                                    **tri_mode.stats(),
                                }
                            )
                        write_csv(args.out_dir / "generation_runs.partial.csv", rows)
        write_csv(args.out_dir / "generation_runs.csv", rows)
    finally:
        tri_mode.restore()
        wan_model_module.flash_attention = original_attention

    manifest = {
        "scope": "Wan2.1-T2V-1.3B grouped block-step-CFG tri-mode Oracle collection",
        "quality_claim": "counterfactual engineering evidence, not an official video benchmark",
        "operator_contract": {
            "D": "BF16 block with FA3 BF16 self-attention and BF16 FFN",
            "Q": "FA3 BF16 attention with static-input FP8 FFN up/down recompute",
            "C": "whole-block residual replay or first-order forecast; no Q compute",
            "exclusion": "exactly one action is executed per block-step-CFG branch",
        },
        "arguments": vars(args)
        | {
            "wan_source": str(args.wan_source),
            "worldfoundry_source": str(args.worldfoundry_source),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "schedule_file": str(args.schedule_file),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        },
        "schedules": [schedule.to_dict() for schedule in schedules],
        "prompts": prompts,
        "seeds": seeds,
        "replacement": dataclasses.asdict(replacement_summary),
        "load_seconds": load_seconds,
        "calibration_seconds": calibration_seconds,
        "precision_warmup_seconds": precision_warmup_seconds,
        "grid_compatibility_installed": grid_compatibility_installed,
        "checkpoint_weight_sha256": sha256(
            args.checkpoint / "diffusion_pytorch_model.safetensors"
        ),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    (args.out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
