#!/usr/bin/env python3
"""Collect Q/cache activation defects on an unchanged dense Wan trajectory."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from activation_defect_runtime import ActivationDefectController
from generate_wan_h200_v4 import (
    AttentionDispatcher,
    install_grid_compatibility,
    load_backends,
    save_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--worldfoundry-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--seeds", default="20260723,20260724")
    parser.add_argument("--steps", default="14,16,19")
    parser.add_argument("--blocks", default="6,12,24")
    parser.add_argument("--branches", default="0,1")
    parser.add_argument("--forecast-scales", default="0.5,0.75,1.0")
    parser.add_argument("--sample-rows", type=int, default=256)
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
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    result = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not result:
        raise ValueError("integer list cannot be empty")
    return result


def parse_floats(value: str) -> list[float]:
    result = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not result:
        raise ValueError("float list cannot be empty")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_prompts]
    if not prompts:
        raise ValueError("prompt file did not provide any prompts")
    seeds = parse_ints(args.seeds)
    steps = parse_ints(args.steps)
    blocks = parse_ints(args.blocks)
    branches = parse_ints(args.branches)
    forecast_scales = parse_floats(args.forecast_scales)
    args.wan_source = args.wan_source.resolve()
    args.worldfoundry_source = args.worldfoundry_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)

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
        backends, hybrid_layer_count=30, sampling_steps=args.sampling_steps
    )
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = dispatcher

    print("STAGE model_load_start", flush=True)
    started = time.perf_counter()
    pipeline = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"],
        checkpoint_dir=str(args.checkpoint),
        device_id=device.index or 0,
        rank=0,
        t5_cpu=False,
    )
    pipeline.model.to(device=device, dtype=WAN_CONFIGS["t2v-1.3B"].param_dtype)
    load_seconds = time.perf_counter() - started
    print(f"STAGE model_load_done seconds={load_seconds:.3f}", flush=True)

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
    replacement = linear_controller.summary()
    linear_controller.set_mode("dense")
    linear_controller.set_calibration(True)
    dispatcher.begin("fa3_bf16")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
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
    calibration_seconds = time.perf_counter() - started
    del calibration_video
    linear_controller.finalize_calibration()
    print(f"STAGE calibration_done seconds={calibration_seconds:.3f}", flush=True)

    if args.precision_warmup_steps:
        linear_controller.set_mode("fp8")
        dispatcher.begin("fa3_bf16")
        warmup_video = pipeline.generate(
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
        del warmup_video
    linear_controller.set_mode("dense")

    audit = ActivationDefectController(
        pipeline.model,
        linear_controller,
        sampling_steps=args.sampling_steps,
        steps=steps,
        blocks=blocks,
        branches=branches,
        forecast_scales=forecast_scales,
        sample_rows=args.sample_rows,
        attention_dispatcher=dispatcher,
    )
    run_rows: list[dict[str, object]] = []
    try:
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                run_id = f"p{prompt_index:02d}_seed{seed}"
                audit.begin_run(run_id)
                dispatcher.begin("fa3_bf16")
                torch.cuda.synchronize(device)
                started = time.perf_counter()
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
                audit.assert_complete()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                video_path = args.out_dir / f"{run_id}_dense_audit.mp4"
                save_video(video, video_path, args.fps)
                del video
                run_rows.append(
                    {
                        "run_id": run_id,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "seconds": seconds,
                        "self_attention_calls": dispatcher.self_calls,
                        "cross_attention_calls": dispatcher.cross_calls,
                        "records_total": len(audit.records),
                        "video_file": video_path.name,
                        "video_sha256": sha256(video_path),
                    }
                )
                print(
                    f"DONE run={run_id} seconds={seconds:.3f} "
                    f"records_total={len(audit.records)}",
                    flush=True,
                )
    finally:
        audit.restore()
        wan_model_module.flash_attention = original_attention

    audit.save(args.out_dir / "activation_defect_samples.pt")
    manifest = {
        "scope": "dense-trajectory Q/cache activation-defect subspace probe",
        "warning": "token-row sampled defects; not a final-video quality metric or a correction benchmark",
        "arguments": vars(args)
        | {
            "wan_source": str(args.wan_source),
            "worldfoundry_source": str(args.worldfoundry_source),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "prompt_file": str(args.prompt_file),
        },
        "prompts": prompts,
        "seeds": seeds,
        "steps": steps,
        "blocks": blocks,
        "branches": branches,
        "forecast_scales": forecast_scales,
        "records": len(audit.records),
        "runs": run_rows,
        "replacement": dataclasses.asdict(replacement),
        "load_seconds": load_seconds,
        "calibration_seconds": calibration_seconds,
        "grid_compatibility_installed": grid_compatibility_installed,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (args.out_dir / "activation_defect_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
