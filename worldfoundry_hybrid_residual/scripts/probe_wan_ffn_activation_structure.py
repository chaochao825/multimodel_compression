#!/usr/bin/env python3
"""Collect dense Wan FFN activations across steps, blocks, branches, and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

from ffn_activation_structure_runtime import FFNActivationStructureController
from generate_wan_h200_v4 import AttentionDispatcher, install_grid_compatibility, load_backends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--seeds", default="20260723,20260724")
    parser.add_argument("--blocks", default="0,12,24,29")
    parser.add_argument("--branches", default="0,1")
    parser.add_argument("--spectral-steps", default="0,5,10,15,19")
    parser.add_argument("--sample-rows", type=int, default=16)
    parser.add_argument("--spectral-channels", type=int, default=32)
    parser.add_argument("--spectral-density", type=float, default=0.125)
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


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
    blocks = parse_ints(args.blocks)
    branches = parse_ints(args.branches)
    spectral_steps = parse_ints(args.spectral_steps)
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, str(args.wan_source))
    os.chdir(args.wan_source)

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

    controller = FFNActivationStructureController(
        pipeline.model,
        sampling_steps=args.sampling_steps,
        frame_num=args.frame_num,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        blocks=blocks,
        branches=branches,
        spectral_steps=spectral_steps,
        sample_rows=args.sample_rows,
        spectral_channels=args.spectral_channels,
        spectral_density=args.spectral_density,
        seed=args.seed,
    )
    run_rows: list[dict[str, object]] = []
    try:
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                run_id = f"p{prompt_index:02d}_seed{seed}"
                controller.begin_run(run_id)
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
                controller.assert_complete()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                del video
                run_rows.append(
                    {
                        "run_id": run_id,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "seconds": seconds,
                        "self_attention_calls": dispatcher.self_calls,
                        "cross_attention_calls": dispatcher.cross_calls,
                        "activation_records_total": len(controller.records),
                        "spectrum_records_total": len(controller.spectrum_records),
                    }
                )
                print(
                    f"DONE run={run_id} seconds={seconds:.3f} "
                    f"records={len(controller.records)} spectra={len(controller.spectrum_records)}",
                    flush=True,
                )
    finally:
        controller.restore()
        wan_model_module.flash_attention = original_attention

    controller.save(args.out_dir / "ffn_activation_samples.pt")
    write_csv(args.out_dir / "ffn_token_spectrum.csv", controller.spectrum_records)
    manifest = {
        "scope": "exact dense F17 FFN activation structure and calibration probe",
        "warning": "sampled-row quantization and sampled-channel spectra are proxies, not rollout perturbations",
        "arguments": vars(args)
        | {
            "wan_source": str(args.wan_source),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "prompt_file": str(args.prompt_file),
        },
        "prompts": prompts,
        "seeds": seeds,
        "blocks": blocks,
        "branches": branches,
        "spectral_steps": spectral_steps,
        "runs": run_rows,
        "records": len(controller.records),
        "spectrum_records": len(controller.spectrum_records),
        "load_seconds": load_seconds,
        "grid_compatibility_installed": grid_compatibility_installed,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
