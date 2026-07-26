#!/usr/bin/env python3
"""Benchmark exact generation-local text K/V caching in Wan2.1 cross-attention."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.cuda.amp as amp

from exact_cross_attention_cache import ExactCrossAttentionKVCache
from generate_wan_cfg_parallel import load_prompts, make_scheduler, sequence_length, target_shape
from generate_wan_h200_v4 import (
    AttentionDispatcher,
    install_grid_compatibility,
    load_backends,
    save_video,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--methods", default="baseline,crossattn_kv_cache")
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=20260738)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--alternate-method-order", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
def generate(
    pipeline: object,
    prompt: str,
    negative_prompt: str,
    size: tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    device: torch.device,
    use_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    shape = target_shape(pipeline, frame_num, size)
    seq_len = sequence_length(pipeline, shape)
    if not negative_prompt:
        negative_prompt = pipeline.sample_neg_prompt
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([prompt], device)
    context_null = pipeline.text_encoder([negative_prompt], device)
    latent = torch.randn(*shape, dtype=torch.float32, device=device, generator=generator)
    scheduler, timesteps = make_scheduler(
        sample_solver, pipeline.num_train_timesteps, sampling_steps, shift, device
    )
    pipeline.model.to(device)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
    controller = ExactCrossAttentionKVCache(pipeline.model) if use_cache else None
    if controller is not None:
        controller.install()
    torch.cuda.synchronize(device)
    denoiser_start = time.perf_counter()
    try:
        with amp.autocast(dtype=pipeline.param_dtype), no_sync():
            for t in timesteps:
                timestep = torch.stack([t])
                model_input = latent.unsqueeze(0)
                if controller is not None:
                    controller.activate("conditional")
                conditional = pipeline.model(
                    model_input, t=timestep, context=context, seq_len=seq_len
                )[0]
                if controller is not None:
                    controller.activate("unconditional")
                unconditional = pipeline.model(
                    model_input, t=timestep, context=context_null, seq_len=seq_len
                )[0]
                guided = unconditional + guide_scale * (conditional - unconditional)
                next_latent = scheduler.step(
                    guided.unsqueeze(0),
                    t,
                    model_input,
                    return_dict=False,
                    generator=generator,
                )[0]
                latent = next_latent.squeeze(0)
                del timestep, model_input, conditional, unconditional, guided, next_latent
        torch.cuda.synchronize(device)
        denoiser_seconds = time.perf_counter() - denoiser_start
        cache_stats = controller.stats() if controller is not None else {
            "crossattn_cache_hits": 0,
            "crossattn_cache_misses": 0,
            "crossattn_cache_initialized_blocks": 0,
            "crossattn_cache_bytes": 0,
        }
    finally:
        if controller is not None:
            controller.restore()
            controller.clear()

    decode_start = time.perf_counter()
    video = pipeline.vae.decode([latent])[0]
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    return video, latent.detach().cpu(), {
        "seconds_including_text_and_vae": time.perf_counter() - total_start,
        "denoiser_seconds": denoiser_seconds,
        "decode_seconds": decode_seconds,
        **cache_stats,
    }


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or set(methods) - {"baseline", "crossattn_kv_cache"}:
        raise ValueError("--methods accepts only baseline,crossattn_kv_cache")
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    sys.path.insert(0, str(args.wan_source))
    os.chdir(args.wan_source)
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    import wan.modules.model as wan_model_module

    compatibility = install_grid_compatibility(wan_model_module)
    backends = load_backends()
    if "fa3_bf16" not in backends:
        raise RuntimeError("FA3 BF16 backend is required")
    dispatcher = AttentionDispatcher(backends, hybrid_layer_count=30, sampling_steps=args.sampling_steps)
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
    load_seconds = time.perf_counter() - load_start
    rows: list[dict[str, object]] = []
    try:
        for method in methods:
            dispatcher.begin("fa3_bf16")
            video, latent, _ = generate(
                pipeline,
                prompts[0],
                args.negative_prompt,
                (args.width, args.height),
                args.frame_num,
                args.shift,
                args.sample_solver,
                args.warmup_steps,
                args.guide_scale,
                args.seed,
                device,
                method == "crossattn_kv_cache",
            )
            del video, latent
            torch.cuda.empty_cache()

        for prompt_index, prompt in enumerate(prompts):
            seed = args.seed + prompt_index
            for repeat in range(args.repeats):
                order = list(methods)
                if args.alternate_method_order and repeat % 2:
                    order.reverse()
                for method_order_index, method in enumerate(order):
                    dispatcher.begin("fa3_bf16")
                    torch.cuda.reset_peak_memory_stats(device)
                    video, latent, timing = generate(
                        pipeline,
                        prompt,
                        args.negative_prompt,
                        (args.width, args.height),
                        args.frame_num,
                        args.shift,
                        args.sample_solver,
                        args.sampling_steps,
                        args.guide_scale,
                        seed,
                        device,
                        method == "crossattn_kv_cache",
                    )
                    suffix = f"_repeat{repeat:02d}" if args.repeats > 1 else ""
                    filename = f"{prompt_index:04d}_{method}_seed{seed}{suffix}.mp4"
                    latent_filename = f"{prompt_index:04d}_{method}_seed{seed}{suffix}.latent.pt"
                    save_video(video, args.out_dir / filename, args.fps)
                    torch.save({"latent": latent}, args.out_dir / latent_filename)
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "method": method,
                            "repeat": repeat,
                            "method_order_index": method_order_index,
                            "seed": seed,
                            "status": "ok",
                            **timing,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
                            "self_attention_calls": dispatcher.self_calls,
                            "cross_attention_calls": dispatcher.cross_calls,
                            "video_file": filename,
                            "video_sha256": sha256(args.out_dir / filename),
                            "latent_file": latent_filename,
                            "latent_sha256": sha256(args.out_dir / latent_filename),
                            "error": "",
                        }
                    )
                    print(
                        f"[crossattn-cache] method={method} prompt={prompt_index} repeat={repeat} "
                        f"seconds={timing['seconds_including_text_and_vae']:.3f}",
                        flush=True,
                    )
                    del video, latent
                    torch.cuda.empty_cache()
    finally:
        wan_model_module.flash_attention = original_attention

    write_rows(args.out_dir / "generation_runs.csv", rows)
    manifest = {
        "scope": "Wan2.1-T2V-1.3B exact generation-local per-branch per-block text K/V cache",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "cache_boundary": "one prompt generation; conditional and unconditional slots are disjoint",
        "warning": "monkey-patch probe only; formal integration should pass cache objects through model/block APIs",
        "load_seconds": load_seconds,
        "grid_compatibility_installed": compatibility,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    (args.out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
