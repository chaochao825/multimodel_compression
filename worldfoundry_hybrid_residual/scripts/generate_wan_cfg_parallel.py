#!/usr/bin/env python3
"""Benchmark exact two-GPU CFG branch parallelism for Wan2.1 on H200 NVL.

Run with two visible GPUs, for example:

    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
      scripts/generate_wan_cfg_parallel.py ...

The conditional and unconditional denoisers receive the same latent and run on
separate model replicas.  Rank 1 sends its unconditional prediction to rank 0;
rank 0 applies the original CFG arithmetic and UniPC/DPM++ scheduler, then
broadcasts the next latent.  No model approximation is introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

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
    parser.add_argument("--methods", default="sequential,cfg_parallel")
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--alternate-method-order", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()


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
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    return prompts


def make_scheduler(
    solver: str,
    train_steps: int,
    sampling_steps: int,
    shift: float,
    device: torch.device,
) -> tuple[object, torch.Tensor]:
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=train_steps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        return scheduler, scheduler.timesteps
    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=train_steps,
        shift=1,
        use_dynamic_shifting=False,
    )
    sigmas = get_sampling_sigmas(sampling_steps, shift)
    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
    return scheduler, timesteps


def target_shape(pipeline: object, frame_num: int, size: tuple[int, int]) -> tuple[int, ...]:
    return (
        pipeline.vae.model.z_dim,
        (frame_num - 1) // pipeline.vae_stride[0] + 1,
        size[1] // pipeline.vae_stride[1],
        size[0] // pipeline.vae_stride[2],
    )


def sequence_length(pipeline: object, shape: Sequence[int]) -> int:
    return (
        math.ceil(
            (shape[2] * shape[3])
            / (pipeline.patch_size[1] * pipeline.patch_size[2])
            * shape[1]
            / pipeline.sp_size
        )
        * pipeline.sp_size
    )


@torch.inference_mode()
def generate_cfg_parallel(
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
    rank: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    shape = target_shape(pipeline, frame_num, size)
    seq_len = sequence_length(pipeline, shape)
    if not negative_prompt:
        negative_prompt = pipeline.sample_neg_prompt

    generator = torch.Generator(device=device).manual_seed(seed)
    branch_prompt = prompt if rank == 0 else negative_prompt
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([branch_prompt], device)

    if rank == 0:
        latent = torch.randn(
            *shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    else:
        latent = torch.empty(*shape, dtype=torch.float32, device=device)
    dist.broadcast(latent, src=0)
    scheduler, timesteps = make_scheduler(
        sample_solver,
        pipeline.num_train_timesteps,
        sampling_steps,
        shift,
        device,
    )
    pipeline.model.to(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    denoiser_start = time.perf_counter()

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
    with amp.autocast(dtype=pipeline.param_dtype), no_sync():
        for t in timesteps:
            timestep = torch.stack([t])
            prediction = pipeline.model(
                latent.unsqueeze(0),
                t=timestep,
                context=context,
                seq_len=seq_len,
            )[0]
            if rank == 0:
                unconditional = torch.empty_like(prediction)
                dist.recv(unconditional, src=1)
                guided = unconditional + guide_scale * (prediction - unconditional)
                next_latent = scheduler.step(
                    guided.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                # UniPC retains a reference to its input sample. Rebinding is
                # required; an in-place copy would corrupt scheduler history.
                latent = next_latent.squeeze(0)
                del unconditional, guided, next_latent
            else:
                dist.send(prediction.contiguous(), dst=0)
            dist.broadcast(latent, src=0)
            del prediction, timestep

    torch.cuda.synchronize(device)
    denoiser_seconds = time.perf_counter() - denoiser_start
    video = None
    decode_seconds = 0.0
    if rank == 0:
        decode_start = time.perf_counter()
        video = pipeline.vae.decode([latent])[0]
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_start
    dist.barrier()
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_start
    final_latent = latent.detach().cpu() if rank == 0 else None
    return video, final_latent, {
        "seconds_including_text_and_vae": total_seconds,
        "denoiser_seconds": denoiser_seconds,
        "decode_seconds_rank0": decode_seconds,
    }


@torch.inference_mode()
def generate_sequential_reference(
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
    rank: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Run the original two-branch arithmetic on rank 0 without hidden collectives."""

    dist.barrier()
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    if rank != 0:
        dist.barrier()
        torch.cuda.synchronize(device)
        return None, None, {
            "seconds_including_text_and_vae": time.perf_counter() - total_start,
        }

    shape = target_shape(pipeline, frame_num, size)
    seq_len = sequence_length(pipeline, shape)
    if not negative_prompt:
        negative_prompt = pipeline.sample_neg_prompt
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([prompt], device)
    context_null = pipeline.text_encoder([negative_prompt], device)
    latent = torch.randn(
        *shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scheduler, timesteps = make_scheduler(
        sample_solver,
        pipeline.num_train_timesteps,
        sampling_steps,
        shift,
        device,
    )
    pipeline.model.to(device)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
    torch.cuda.synchronize(device)
    denoiser_start = time.perf_counter()
    with amp.autocast(dtype=pipeline.param_dtype), no_sync():
        for t in timesteps:
            timestep = torch.stack([t])
            model_input = latent.unsqueeze(0)
            conditional = pipeline.model(
                model_input,
                t=timestep,
                context=context,
                seq_len=seq_len,
            )[0]
            unconditional = pipeline.model(
                model_input,
                t=timestep,
                context=context_null,
                seq_len=seq_len,
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
    decode_start = time.perf_counter()
    video = pipeline.vae.decode([latent])[0]
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    final_latent = latent.detach().cpu()
    dist.barrier()
    torch.cuda.synchronize(device)
    return video, final_latent, {
        "seconds_including_text_and_vae": time.perf_counter() - total_start,
        "denoiser_seconds": denoiser_seconds,
        "decode_seconds_rank0": decode_seconds,
    }


def aggregate_worker_stats(
    device: torch.device, dispatcher: AttentionDispatcher
) -> dict[str, float]:
    counts = torch.tensor(
        [float(dispatcher.self_calls), float(dispatcher.cross_calls)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / (1024.0**2)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return {
        "self_attention_calls_all_ranks": float(counts[0].item()),
        "cross_attention_calls_all_ranks": float(counts[1].item()),
        "peak_allocated_mib_max_rank": float(peak.item()),
    }


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


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("generate_wan_cfg_parallel.py requires exactly two torchrun ranks")
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    if min(args.sampling_steps, args.warmup_steps, args.repeats) <= 0:
        raise ValueError("sampling steps, warmup steps, and repeats must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    prompts = load_prompts(args)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or set(methods) - {"sequential", "cfg_parallel"}:
        raise ValueError("--methods accepts only sequential,cfg_parallel")

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
        device_id=local_rank,
        rank=rank,
        t5_cpu=False,
    )
    pipeline.model.to(device=device, dtype=WAN_CONFIGS["t2v-1.3B"].param_dtype)
    dist.barrier()
    load_seconds = time.perf_counter() - load_start

    rows: list[dict[str, object]] = []
    try:
        # Warm both code paths so the comparison excludes first-use kernels.
        for method in methods:
            dispatcher.begin("fa3_bf16")
            if method == "cfg_parallel":
                video, latent, _ = generate_cfg_parallel(
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
                    rank,
                )
            else:
                video, latent, _ = generate_sequential_reference(
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
                    rank,
                )
            del video, latent
            torch.cuda.empty_cache()

        for prompt_index, prompt in enumerate(prompts):
            seed = args.seed + prompt_index
            for repeat in range(args.repeats):
                order = list(methods)
                if args.alternate_method_order and repeat % 2:
                    order.reverse()
                for method_index, method in enumerate(order):
                    dispatcher.begin("fa3_bf16")
                    torch.cuda.reset_peak_memory_stats(device)
                    dist.barrier()
                    if method == "cfg_parallel":
                        video, latent, timing = generate_cfg_parallel(
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
                            rank,
                        )
                    else:
                        video, latent, timing = generate_sequential_reference(
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
                            rank,
                        )
                    worker = aggregate_worker_stats(device, dispatcher)
                    if rank == 0:
                        filename = ""
                        digest = ""
                        latent_filename = ""
                        latent_digest = ""
                        if video is not None and latent is not None:
                            suffix = f"_repeat{repeat:02d}" if args.repeats > 1 else ""
                            filename = f"{prompt_index:04d}_{method}_seed{seed}{suffix}.mp4"
                            output_path = args.out_dir / filename
                            save_video(video, output_path, args.fps)
                            digest = sha256(output_path)
                            latent_filename = (
                                f"{prompt_index:04d}_{method}_seed{seed}{suffix}.latent.pt"
                            )
                            latent_path = args.out_dir / latent_filename
                            torch.save({"latent": latent}, latent_path)
                            latent_digest = sha256(latent_path)
                        rows.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "method": method,
                                "repeat": repeat,
                                "method_order_index": method_index,
                                "seed": seed,
                                "status": "ok",
                                **timing,
                                **worker,
                                "video_file": filename,
                                "video_sha256": digest,
                                "latent_file": latent_filename,
                                "latent_sha256": latent_digest,
                                "error": "",
                            }
                        )
                        print(
                            f"[cfg-parallel] method={method} prompt={prompt_index} "
                            f"repeat={repeat} seconds={timing.get('seconds_including_text_and_vae', float('nan')):.3f}",
                            flush=True,
                        )
                    del video, latent
                    torch.cuda.empty_cache()
    finally:
        wan_model_module.flash_attention = original_attention

    if rank == 0:
        write_rows(args.out_dir / "generation_runs.csv", rows)
        manifest = {
            "scope": "Wan2.1-T2V-1.3B paired sequential versus two-H200 CFG branch parallelism",
            "fidelity": (
                "model-exact design with the same sampler, seed, and CFG arithmetic; "
                "the paired latent/video files are the required numerical evidence"
            ),
            "timing": "wall time includes text encoding, denoising, communication, scheduler, and VAE decode",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "prompts": prompts,
            "methods": methods,
            "world_size": dist.get_world_size(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "grid_compatibility_installed": compatibility,
            "available_attention_backends": sorted(backends),
            "load_seconds_synchronized": load_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        (args.out_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
