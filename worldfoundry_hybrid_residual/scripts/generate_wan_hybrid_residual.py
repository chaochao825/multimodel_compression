#!/usr/bin/env python3
"""Paired Wan H200 experiment for FP8 + low-rank + row-sparse + TeaCache."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import platform
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch

from generate_wan_h200_v4 import (
    AttentionDispatcher,
    TeaCacheBlockController,
    install_grid_compatibility,
    load_backends,
    save_video,
    sha256,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--worldfoundry-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seeds", default="20260723,20260724")
    parser.add_argument(
        "--methods",
        default="dense,dense_cache008,fp8,hybrid,hybrid_cache008",
    )
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
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--precision-warmup-steps", type=int, default=1)
    parser.add_argument("--alternate-method-order", action="store_true")
    parser.add_argument("--linear-scope", choices=("self_attn", "ffn"), default="self_attn")
    parser.add_argument("--residual-targets", default="q,o")
    parser.add_argument("--residual-blocks", default="all")
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--budget-rank", type=int, default=16)
    parser.add_argument("--row-block-size", type=int, default=8)
    parser.add_argument("--static-scale-margin", type=float, default=1.05)
    parser.add_argument("--teacache-retention-calls", type=int, default=24)
    parser.add_argument("--precision-boundary-steps", type=int, default=2)
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
    return prompts[: args.max_prompts] if args.max_prompts > 0 else prompts


def parse_seeds(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def method_configuration(method: str) -> tuple[str, str, int, int]:
    if "_cache" in method:
        mode, cache_token = method.rsplit("_cache", 1)
        if len(cache_token) != 3 or not cache_token.isdigit():
            raise ValueError(f"cache method requires a three-digit suffix: {method}")
        cache_method = f"fa3_bf16_teacache_{cache_token}"
    else:
        mode, cache_method = method, "fa3_bf16"
    refresh_interval = 0
    middle_steps = 0
    if mode.startswith("hybrid_refresh"):
        refresh_token = mode.removeprefix("hybrid_refresh")
        if not refresh_token.isdigit() or int(refresh_token) <= 0:
            raise ValueError(f"invalid precision refresh method: {method}")
        mode = "hybrid"
        refresh_interval = int(refresh_token)
    elif mode.startswith("hybrid_middle"):
        middle_token = mode.removeprefix("hybrid_middle")
        if not middle_token.isdigit() or int(middle_token) <= 0:
            raise ValueError(f"invalid middle-step method: {method}")
        mode = "hybrid"
        middle_steps = int(middle_token)
    elif mode.startswith("fp8_middle"):
        middle_token = mode.removeprefix("fp8_middle")
        if not middle_token.isdigit() or int(middle_token) <= 0:
            raise ValueError(f"invalid FP8 middle-step method: {method}")
        mode = "fp8"
        middle_steps = int(middle_token)
    if mode not in {"dense", "fp8", "hybrid"}:
        raise ValueError(f"unknown method mode: {method}")
    return mode, cache_method, refresh_interval, middle_steps


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PrecisionRefreshController:
    """Switch injected linears at model-forward boundaries.

    Wan performs two classifier-free-guidance model calls per diffusion step.
    Both branches receive the same precision policy. Boundary and periodic
    dense calls bound the number of consecutive approximate diffusion steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        linear_controller: object,
        *,
        sampling_steps: int,
        boundary_steps: int,
    ) -> None:
        if boundary_steps < 0 or boundary_steps * 2 > sampling_steps:
            raise ValueError("precision boundary steps must lie in [0, sampling_steps / 2]")
        self.model = model
        self.linear_controller = linear_controller
        self.sampling_steps = sampling_steps
        self.boundary_steps = boundary_steps
        self.original_forward = model.forward
        self.base_mode = "dense"
        self.refresh_interval = 0
        self.middle_steps = 0
        self.call_index = 0
        self.dense_model_forwards = 0
        self.approximate_model_forwards = 0

        def wrapped(
            model_self: torch.nn.Module,
            *args: object,
            **kwargs: object,
        ) -> object:
            return self._forward(*args, **kwargs)

        model.forward = types.MethodType(wrapped, model)

    def begin(self, base_mode: str, refresh_interval: int, middle_steps: int) -> None:
        if middle_steps > self.sampling_steps:
            raise ValueError("middle steps cannot exceed sampling steps")
        self.base_mode = base_mode
        self.refresh_interval = refresh_interval
        self.middle_steps = middle_steps
        self.call_index = 0
        self.dense_model_forwards = 0
        self.approximate_model_forwards = 0
        self.linear_controller.set_mode(base_mode)

    def _selected_mode(self) -> str:
        step = self.call_index // 2
        if self.middle_steps:
            start = (self.sampling_steps - self.middle_steps) // 2
            return (
                self.base_mode
                if start <= step < start + self.middle_steps
                else "dense"
            )
        if self.base_mode != "hybrid":
            return self.base_mode
        if self.refresh_interval <= 0:
            return self.base_mode
        boundary = (
            step < self.boundary_steps
            or step >= self.sampling_steps - self.boundary_steps
        )
        periodic = step % self.refresh_interval == 0
        return "dense" if boundary or periodic else "hybrid"

    def _forward(self, *args: object, **kwargs: object) -> object:
        selected_mode = self._selected_mode()
        self.linear_controller.set_mode(selected_mode)
        if selected_mode == "dense":
            self.dense_model_forwards += 1
        else:
            self.approximate_model_forwards += 1
        self.call_index += 1
        return self.original_forward(*args, **kwargs)

    def stats(self) -> dict[str, object]:
        total = self.dense_model_forwards + self.approximate_model_forwards
        return {
            "precision_refresh_interval": self.refresh_interval,
            "precision_middle_steps": self.middle_steps,
            "precision_boundary_steps": self.boundary_steps,
            "precision_model_forwards": total,
            "precision_dense_model_forwards": self.dense_model_forwards,
            "precision_approximate_model_forwards": self.approximate_model_forwards,
            "precision_approximate_fraction": (
                self.approximate_model_forwards / total if total else 0.0
            ),
        }

    def restore(self) -> None:
        self.model.forward = self.original_forward


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    if args.sampling_steps <= 0 or args.warmup_steps <= 0:
        raise ValueError("sampling and calibration warmup steps must be positive")
    if args.precision_warmup_steps < 0:
        raise ValueError("precision warmup steps must be nonnegative")
    args.wan_source = args.wan_source.resolve()
    args.worldfoundry_source = args.worldfoundry_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    seeds = parse_seeds(args.seeds)
    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    configurations = {method: method_configuration(method) for method in methods}
    targets = tuple(token.strip() for token in args.residual_targets.split(",") if token.strip())

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
    dispatcher = AttentionDispatcher(backends, hybrid_layer_count=30, sampling_steps=args.sampling_steps)
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = dispatcher

    print(
        f"STAGE model_load_start device={device} prompts={len(prompts)} "
        f"seeds={len(seeds)} methods={len(methods)}",
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
    print(
        f"STAGE residual_setup_start scope={args.linear_scope} "
        f"targets={','.join(targets)} "
        f"blocks={args.residual_blocks}",
        flush=True,
    )
    controller = replace_wan_linears(
        pipeline.model,
        scope=args.linear_scope,
        targets=targets,
        blocks=args.residual_blocks,
        residual_rank=args.residual_rank,
        budget_rank=args.budget_rank,
        row_block_size=args.row_block_size,
        seed=seeds[0],
        static_scale_margin=args.static_scale_margin,
    )
    replacement_summary = controller.summary()
    print(
        f"STAGE residual_setup_done seconds={replacement_summary.setup_seconds:.3f} "
        f"linears={len(replacement_summary.names)} "
        f"low_rank_values={replacement_summary.low_rank_values} "
        f"sparse_values={replacement_summary.sparse_values}",
        flush=True,
    )
    teacache = TeaCacheBlockController(
        pipeline.model, retention_calls=args.teacache_retention_calls
    )

    controller.set_mode("dense")
    controller.set_calibration(True)
    controller.reset_runtime_stats()
    dispatcher.begin("fa3_bf16")
    teacache.begin("fa3_bf16")
    print(f"STAGE calibration_start steps={args.warmup_steps}", flush=True)
    torch.cuda.synchronize(device)
    warmup_started = time.perf_counter()
    warmup_video = pipeline.generate(
        input_prompt=prompts[0],
        size=(args.width, args.height),
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.warmup_steps,
        guide_scale=args.guide_scale,
        n_prompt=args.negative_prompt,
        seed=seeds[0],
        offload_model=False,
    )
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    del warmup_video
    controller.finalize_calibration()
    warmup_stats = controller.runtime_stats()
    print(
        f"STAGE calibration_done seconds={warmup_seconds:.3f} "
        f"scale_min={warmup_stats['static_scale_min']:.8f} "
        f"scale_max={warmup_stats['static_scale_max']:.8f}",
        flush=True,
    )
    torch.cuda.empty_cache()
    precision_warmup_seconds = 0.0
    if args.precision_warmup_steps:
        controller.set_mode("fp8")
        controller.reset_runtime_stats()
        dispatcher.begin("fa3_bf16")
        teacache.begin("fa3_bf16")
        print(
            f"STAGE precision_warmup_start steps={args.precision_warmup_steps}",
            flush=True,
        )
        torch.cuda.synchronize(device)
        precision_warmup_started = time.perf_counter()
        precision_warmup_video = pipeline.generate(
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
        precision_warmup_seconds = time.perf_counter() - precision_warmup_started
        del precision_warmup_video
        print(
            f"STAGE precision_warmup_done seconds={precision_warmup_seconds:.3f}",
            flush=True,
        )
        torch.cuda.empty_cache()
    precision = PrecisionRefreshController(
        pipeline.model,
        controller,
        sampling_steps=args.sampling_steps,
        boundary_steps=args.precision_boundary_steps,
    )

    rows: list[dict[str, object]] = []
    pair_counter = 0
    try:
        for prompt_index, prompt in enumerate(prompts):
            for seed_index, seed in enumerate(seeds):
                method_order = list(methods)
                if args.alternate_method_order and method_order:
                    shift = pair_counter % len(method_order)
                    method_order = method_order[shift:] + method_order[:shift]
                pair_counter += 1
                for method_order_index, method in enumerate(method_order):
                    mode, cache_method, refresh_interval, middle_steps = configurations[method]
                    controller.reset_runtime_stats()
                    precision.begin(mode, refresh_interval, middle_steps)
                    dispatcher.begin("fa3_bf16")
                    teacache.begin(cache_method)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
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
                        torch.cuda.synchronize(device)
                        seconds = time.perf_counter() - started
                        filename = f"p{prompt_index:02d}_{method}_seed{seed}.mp4"
                        video_path = args.out_dir / filename
                        save_video(video, video_path, args.fps)
                        row: dict[str, object] = {
                            "prompt_index": prompt_index,
                            "seed_index": seed_index,
                            "prompt": prompt,
                            "method": method,
                            "mode": mode,
                            "cache_method": cache_method,
                            "method_order_index": method_order_index,
                            "seed": seed,
                            "status": "ok",
                            "seconds_including_text_and_vae": seconds,
                            "self_attention_calls": dispatcher.self_calls,
                            "cross_attention_calls": dispatcher.cross_calls,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
                            "video_file": filename,
                            "video_sha256": sha256(video_path),
                            **teacache.stats(),
                            **precision.stats(),
                            **controller.runtime_stats(),
                        }
                        rows.append(row)
                        print(
                            f"DONE prompt={prompt_index} seed={seed} method={method} "
                            f"seconds={seconds:.3f} cache_hits={row['cached_model_forwards']} "
                            f"residual_calls={row['hybrid_residual_calls']}",
                            flush=True,
                        )
                        del video
                    except Exception as error:
                        torch.cuda.synchronize(device)
                        traceback.print_exc()
                        rows.append(
                            {
                                "prompt_index": prompt_index,
                                "seed_index": seed_index,
                                "prompt": prompt,
                                "method": method,
                                "mode": mode,
                                "cache_method": cache_method,
                                "method_order_index": method_order_index,
                                "seed": seed,
                                "status": "error",
                                "seconds_including_text_and_vae": time.perf_counter() - started,
                                "error": f"{type(error).__name__}: {error}",
                                **teacache.stats(),
                                **precision.stats(),
                                **controller.runtime_stats(),
                            }
                        )
                    write_csv(args.out_dir / "generation_runs.partial.csv", rows)
                    torch.cuda.empty_cache()
    finally:
        precision.restore()
        teacache.restore()
        wan_model_module.flash_attention = original_attention

    write_csv(args.out_dir / "generation_runs.csv", rows)
    manifest = {
        "scope": "paired Wan2.1-T2V-1.3B H200 experiment; FA3 BF16 attention fixed; selected linear scope and refresh policy vary",
        "quality_claim": "paired engineering evidence, not an official video benchmark",
        "arguments": vars(args)
        | {
            "wan_source": str(args.wan_source),
            "worldfoundry_source": str(args.worldfoundry_source),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        },
        "prompts": prompts,
        "seeds": seeds,
        "methods": methods,
        "method_configurations": configurations,
        "replacement": dataclasses.asdict(replacement_summary),
        "warmup_seconds": warmup_seconds,
        "precision_warmup_seconds": precision_warmup_seconds,
        "warmup_stats": warmup_stats,
        "load_seconds": load_seconds,
        "grid_compatibility_installed": grid_compatibility_installed,
        "attention_policy": "FA3 BF16 self-attention and dense SDPA cross-attention for every method",
        "cache_policy": "official-coefficient TeaCache block wrapper; cache hits bypass all 30 blocks and injected residual linears",
        "precision_policy": "CFG-paired dense boundary/periodic refresh; approximate intervals are bounded in diffusion-step space",
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
