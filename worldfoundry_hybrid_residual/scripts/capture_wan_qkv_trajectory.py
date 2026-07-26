#!/usr/bin/env python3
"""Capture full Wan self-attention Q/K/V on selected production UniPC trajectory cells."""

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
from typing import Callable

import torch
import torch.cuda.amp as amp

from generate_wan_cfg_parallel import load_prompts, make_scheduler, sequence_length, target_shape
from generate_wan_h200_v4 import AttentionDispatcher, install_grid_compatibility, load_backends


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return values


def parse_sample_plan(text: str) -> tuple[tuple[int, int], ...]:
    """Parse prompt-index:seed pairs without coupling text and noise changes."""
    values: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            prompt_index_text, seed_text = item.split(":", maxsplit=1)
            prompt_index = int(prompt_index_text)
            seed = int(seed_text)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                "sample plan must contain prompt_index:seed pairs"
            ) from error
        if prompt_index < 0 or seed < 0:
            raise argparse.ArgumentTypeError("sample plan values must be non-negative")
        values.append((prompt_index, seed))
    if not values:
        raise argparse.ArgumentTypeError("sample plan cannot be empty")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--capture-steps", type=parse_int_list, default=parse_int_list("0,9,19"))
    parser.add_argument("--capture-layers", type=parse_int_list, default=parse_int_list("0,14,29"))
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=20260740)
    parser.add_argument(
        "--sample-plan",
        type=parse_sample_plan,
        help=(
            "Explicit prompt_index:seed pairs. When omitted, each loaded prompt "
            "uses --seed + prompt_index."
        ),
    )
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


class QKVTrajectoryRecorder:
    def __init__(
        self,
        dispatcher: AttentionDispatcher,
        output_dir: Path,
        capture_steps: set[int],
        capture_layers: set[int],
        static_metadata: dict[str, object],
    ) -> None:
        self.dispatcher = dispatcher
        self.output_dir = output_dir
        self.capture_steps = capture_steps
        self.capture_layers = capture_layers
        self.static_metadata = static_metadata
        self.active: dict[str, object] = {}
        self.self_layer = 0
        self.captured: set[tuple[object, ...]] = set()
        self.index_rows: list[dict[str, object]] = []

    def begin(self, **metadata: object) -> None:
        self.active = metadata
        self.self_layer = 0
        self.dispatcher.begin("fa3_bf16")

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        is_self = q.shape[1] == k.shape[1]
        layer = self.self_layer if is_self else -1
        if is_self:
            self.self_layer += 1
        output = self.dispatcher(q, k, v, **kwargs)
        step = int(self.active.get("sampling_step", -1))
        if not is_self or step not in self.capture_steps or layer not in self.capture_layers:
            return output

        key = (
            self.active.get("sample_id"),
            step,
            self.active.get("branch"),
            layer,
        )
        if key in self.captured:
            raise RuntimeError(f"duplicate QKV capture cell: {key}")
        self.captured.add(key)
        timestep = float(self.active["timestep"])
        sample_id = str(self.active["sample_id"])
        branch = str(self.active["branch"])
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{sample_id}_step{step:02d}_t{timestep:g}_{branch}_l{layer:02d}_self.pt"
        )
        path = sample_dir / filename
        scale_object = kwargs.get("softmax_scale")
        softmax_scale = (
            float(scale_object) if scale_object is not None else 1.0 / math.sqrt(q.shape[-1])
        )
        metadata = {
            **self.static_metadata,
            **self.active,
            "layer": layer,
            "attention_kind": "self",
            "token_count": q.shape[1],
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "dtype": str(q.dtype),
        }
        payload = {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "softmax_scale": softmax_scale,
            "metadata": metadata,
        }
        torch.save(payload, path)
        bytes_on_disk = path.stat().st_size
        self.index_rows.append(
            {
                "sample_id": sample_id,
                "prompt_index": self.active["prompt_index"],
                "seed": self.active["seed"],
                "sampling_step": step,
                "timestep": timestep,
                "branch": branch,
                "layer": layer,
                "path": str(path),
                "bytes": bytes_on_disk,
            }
        )
        print(
            f"[qkv-capture] sample={sample_id} step={step} branch={branch} "
            f"layer={layer} gib={bytes_on_disk / (1024.0**3):.3f}",
            flush=True,
        )
        del payload
        return output


@torch.inference_mode()
def capture_sample(
    pipeline: object,
    recorder: QKVTrajectoryRecorder,
    prompt: str,
    negative_prompt: str,
    prompt_index: int,
    seed: int,
    sample_id: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    size = (args.width, args.height)
    shape = target_shape(pipeline, args.frame_num, size)
    seq_len = sequence_length(pipeline, shape)
    if not negative_prompt:
        negative_prompt = pipeline.sample_neg_prompt
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([prompt], device)
    context_null = pipeline.text_encoder([negative_prompt], device)
    latent = torch.randn(*shape, dtype=torch.float32, device=device, generator=generator)
    scheduler, timesteps = make_scheduler(
        args.sample_solver,
        pipeline.num_train_timesteps,
        args.sampling_steps,
        args.shift,
        device,
    )
    pipeline.model.to(device)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync: Callable[[], object] = getattr(pipeline.model, "no_sync", noop_no_sync)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with amp.autocast(dtype=pipeline.param_dtype), no_sync():
        for sampling_step, timestep in enumerate(timesteps):
            model_input = latent.unsqueeze(0)
            recorder.begin(
                sample_id=sample_id,
                prompt_index=prompt_index,
                seed=seed,
                prompt=prompt,
                sampling_step=sampling_step,
                timestep=float(timestep),
                branch="cond",
            )
            conditional = pipeline.model(
                model_input,
                t=torch.stack([timestep]),
                context=context,
                seq_len=seq_len,
            )[0]
            recorder.begin(
                sample_id=sample_id,
                prompt_index=prompt_index,
                seed=seed,
                prompt=prompt,
                sampling_step=sampling_step,
                timestep=float(timestep),
                branch="uncond",
            )
            unconditional = pipeline.model(
                model_input,
                t=torch.stack([timestep]),
                context=context_null,
                seq_len=seq_len,
            )[0]
            guided = unconditional + args.guide_scale * (conditional - unconditional)
            next_latent = scheduler.step(
                guided.unsqueeze(0),
                timestep,
                model_input,
                return_dict=False,
                generator=generator,
            )[0]
            latent = next_latent.squeeze(0)
            del model_input, conditional, unconditional, guided, next_latent
    torch.cuda.synchronize(device)
    sample_dir = args.out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": latent.detach().cpu()}, sample_dir / "final_latent.pt")
    return {
        "sample_id": sample_id,
        "prompt_index": prompt_index,
        "seed": seed,
        "prompt": prompt,
        "seconds": time.perf_counter() - started,
        "final_latent": str(sample_dir / "final_latent.pt"),
    }


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    if max(args.capture_steps) >= args.sampling_steps:
        raise ValueError("capture steps must be smaller than sampling steps")
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    sample_plan = (
        args.sample_plan
        if args.sample_plan is not None
        else tuple((index, args.seed + index) for index in range(len(prompts)))
    )
    unavailable = sorted({index for index, _ in sample_plan if index >= len(prompts)})
    if unavailable:
        raise ValueError(
            f"sample plan references prompt indices {unavailable}, but only "
            f"{len(prompts)} prompts were loaded"
        )
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
    latent_frames = (args.frame_num - 1) // 4 + 1
    grid_size = [latent_frames, args.height // 16, args.width // 16]
    recorder = QKVTrajectoryRecorder(
        dispatcher,
        args.out_dir,
        set(args.capture_steps),
        set(args.capture_layers),
        {
            "frame_num": args.frame_num,
            "input_height": args.height,
            "input_width": args.width,
            "grid_size": grid_size,
            "vae_stride": [4, 8, 8],
            "patch_size": [1, 2, 2],
            "token_flatten_order": "t,h,w",
            "sampling_steps": args.sampling_steps,
            "sample_solver": args.sample_solver,
        },
    )
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = recorder
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
    samples: list[dict[str, object]] = []
    try:
        for sample_index, (prompt_index, seed) in enumerate(sample_plan):
            prompt = prompts[prompt_index]
            sample_id = f"s{sample_index:02d}_p{prompt_index:02d}_seed{seed}"
            samples.append(
                capture_sample(
                    pipeline,
                    recorder,
                    prompt,
                    args.negative_prompt,
                    prompt_index,
                    seed,
                    sample_id,
                    args,
                    device,
                )
            )
            torch.cuda.empty_cache()
    finally:
        wan_model_module.flash_attention = original_attention

    write_rows(args.out_dir / "capture_index.csv", recorder.index_rows)
    write_rows(args.out_dir / "sample_runs.csv", samples)
    expected = len(sample_plan) * len(args.capture_steps) * 2 * len(args.capture_layers)
    manifest = {
        "scope": "full post-RoPE Q/K/V from selected cells on production UniPC CFG trajectories",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "prompts": prompts,
        "sample_plan": [
            {
                "sample_id": f"s{index:02d}_p{prompt_index:02d}_seed{seed}",
                "prompt_index": prompt_index,
                "seed": seed,
            }
            for index, (prompt_index, seed) in enumerate(sample_plan)
        ],
        "grid_size": grid_size,
        "capture_count": len(recorder.index_rows),
        "expected_capture_count": expected,
        "capture_complete": len(recorder.index_rows) == expected,
        "capture_bytes": sum(int(row["bytes"]) for row in recorder.index_rows),
        "load_seconds": load_seconds,
        "grid_compatibility_installed": compatibility,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    (args.out_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not manifest["capture_complete"]:
        raise RuntimeError(
            f"captured {len(recorder.index_rows)} cells, expected {expected}"
        )


if __name__ == "__main__":
    main()
