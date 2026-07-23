#!/usr/bin/env python3
"""Generate paired Wan2.1 videos with H200 attention backends.

Only video self-attention is replaced.  Text cross-attention remains dense
PyTorch Flash-SDPA, matching the precision/sparsity evidence in this branch.
All methods reuse the same checkpoint, prompt, sampler settings, and seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Callable, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


TEACACHE_SOURCE_COMMIT = "7c10efc4702c6b619f47805f7abe4a7a08085aa0"
TEACACHE_1P3B_REF_COEFFICIENTS = (
    -5.21862437e04,
    9.23041404e03,
    -5.28275948e02,
    1.36987616e01,
    -4.99875664e-02,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--methods", default="sdpa,fa3_bf16,fa3_fp8,sage_auto,sage_sm90_smooth"
    )
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed generations per prompt and method, all with the same seed.",
    )
    parser.add_argument(
        "--alternate-method-order",
        action="store_true",
        help="Reverse method order on odd repeats to reduce order/thermal bias.",
    )
    parser.add_argument(
        "--warmup-method",
        default="",
        help="Optional backend used for one unsaved warm-up generation.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--hybrid-layer-count",
        type=int,
        default=30,
        help="DiT blocks per model forward, used by FA3 hybrid schedules.",
    )
    parser.add_argument(
        "--teacache-retention-calls",
        type=int,
        default=10,
        help="Initial conditional/unconditional model calls that always execute.",
    )
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
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    return prompts


def torch_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kwargs: dict[str, object]
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2).to(v.dtype),
        k.transpose(1, 2).to(v.dtype),
        v.transpose(1, 2),
        attn_mask=None,
        dropout_p=float(kwargs.get("dropout_p", 0.0)),
        is_causal=bool(kwargs.get("causal", False)),
        scale=kwargs.get("softmax_scale"),
    )
    return output.transpose(1, 2).contiguous().type_as(q)


def load_backends() -> dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float | None], torch.Tensor]]:
    backends: dict[
        str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float | None], torch.Tensor]
    ] = {}
    try:
        from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90

        backends["sage_auto"] = lambda q, k, v, scale: sageattn(
            q, k, v, tensor_layout="NHD", sm_scale=scale
        )
        backends["sage_sm90_smooth"] = lambda q, k, v, scale: sageattn_qk_int8_pv_fp8_cuda_sm90(
            q,
            k,
            v,
            tensor_layout="NHD",
            sm_scale=scale,
            smooth_k=True,
            pv_accum_dtype="fp32+fp32",
        )
    except Exception:
        traceback.print_exc()
    try:
        from sageattention.fa3_wrapper import fa3, fa3_fp8

        backends["fa3_bf16"] = lambda q, k, v, scale: fa3(
            q, k, v, tensor_layout="NHD", sm_scale=scale
        )
        backends["fa3_fp8"] = lambda q, k, v, scale: fa3_fp8(
            q, k, v, tensor_layout="NHD", sm_scale=scale
        )
    except Exception:
        traceback.print_exc()
    return backends


class AttentionDispatcher:
    def __init__(
        self,
        backends: dict[str, Callable[..., torch.Tensor]],
        hybrid_layer_count: int,
        sampling_steps: int,
    ) -> None:
        self.backends = backends
        self.hybrid_layer_count = hybrid_layer_count
        self.sampling_steps = sampling_steps
        self.method = "sdpa"
        self.self_calls = 0
        self.cross_calls = 0

    def begin(self, method: str) -> None:
        self.method = method
        self.self_calls = 0
        self.cross_calls = 0

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        is_self = q.shape[1] == k.shape[1]
        if not is_self:
            self.cross_calls += 1
            return torch_sdpa(q, k, v, kwargs)
        self.self_calls += 1
        if self.method == "sdpa":
            return torch_sdpa(q, k, v, kwargs)
        backend_name = self.method
        if self.method.startswith("fa3_bf16_teacache_"):
            backend_name = "fa3_bf16"
        if self.method.startswith("fa3_hybrid_fp8_"):
            layer = (self.self_calls - 1) % self.hybrid_layer_count
            if self.method == "fa3_hybrid_fp8_every4":
                use_fp8 = layer % 4 == 0
            elif self.method == "fa3_hybrid_fp8_every2":
                use_fp8 = layer % 2 == 0
            elif self.method == "fa3_hybrid_fp8_late8":
                use_fp8 = layer >= self.hybrid_layer_count - 8
            elif self.method.startswith("fa3_hybrid_fp8_middle"):
                middle_steps = int(self.method.removeprefix("fa3_hybrid_fp8_middle"))
                if middle_steps <= 0 or middle_steps > self.sampling_steps:
                    raise RuntimeError(
                        f"middle-step FP8 count must lie in [1, {self.sampling_steps}]"
                    )
                model_forward = (self.self_calls - 1) // self.hybrid_layer_count
                diffusion_step = model_forward // 2
                first_fp8_step = (self.sampling_steps - middle_steps) // 2
                use_fp8 = (
                    first_fp8_step
                    <= diffusion_step
                    < first_fp8_step + middle_steps
                )
            else:
                raise RuntimeError(f"unknown FA3 hybrid schedule: {self.method}")
            backend_name = "fa3_fp8" if use_fp8 else "fa3_bf16"
        backend = self.backends.get(backend_name)
        if backend is None:
            raise RuntimeError(f"attention backend unavailable: {backend_name}")
        scale_object = kwargs.get("softmax_scale")
        scale = float(scale_object) if scale_object is not None else 1.0 / math.sqrt(q.shape[-1])
        return backend(q, k, v, scale).contiguous().type_as(q)


def teacache_threshold(method: str) -> float | None:
    """Parse ``fa3_bf16_teacache_005`` as a TeaCache threshold of 0.05."""

    prefix = "fa3_bf16_teacache_"
    if not method.startswith(prefix):
        return None
    token = method[len(prefix) :]
    if not token.isdigit() or len(token) != 3:
        raise ValueError(
            "TeaCache methods must use a three-digit percent suffix, "
            "for example fa3_bf16_teacache_005 for threshold 0.05"
        )
    threshold = int(token) / 100.0
    if threshold <= 0:
        raise ValueError("TeaCache threshold must be positive")
    return threshold


class TeaCacheBlockController:
    """Training-free residual reuse without replacing the Wan model forward.

    This is a block-level adaptation of the official TeaCache4Wan2.1 policy.
    The first block decides whether to reuse a full-stack residual and the last
    block records that residual on executed model calls.  Wrapping blocks keeps
    MonarchRT's embeddings, RoPE construction, head, and unpatchify code intact.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        retention_calls: int,
    ) -> None:
        if retention_calls < 0:
            raise ValueError("--teacache-retention-calls must be nonnegative")
        if not len(model.blocks):
            raise ValueError("TeaCache requires at least one Wan block")
        self.model = model
        self.retention_calls = retention_calls
        self.original_forwards = [block.forward for block in model.blocks]
        self.enabled = False
        self.threshold: float | None = None
        self.call_index = 0
        self.total_forwards = 0
        self.computed_forwards = 0
        self.cached_forwards = 0
        self.computed_call_indices: list[int] = []
        self.cached_call_indices: list[int] = []
        self.current_skip = False
        self.current_branch = 0
        self.current_input: torch.Tensor | None = None
        self.previous_e0: list[torch.Tensor | None] = [None, None]
        self.previous_residual: list[torch.Tensor | None] = [None, None]
        self.accumulated_distance = [0.0, 0.0]
        self._install()

    def _install(self) -> None:
        last_index = len(self.original_forwards) - 1
        for index, block in enumerate(self.model.blocks):
            original = self.original_forwards[index]

            def wrapped(
                block_self: torch.nn.Module,
                x: torch.Tensor,
                *args: object,
                _index: int = index,
                _original: Callable[..., torch.Tensor] = original,
                **kwargs: object,
            ) -> torch.Tensor:
                if _index == 0:
                    self._start_forward(x, kwargs.get("e"))
                    if self.current_skip:
                        residual = self.previous_residual[self.current_branch]
                        assert residual is not None
                        return x + residual
                elif self.current_skip:
                    return x

                output = _original(x, *args, **kwargs)
                if _index == last_index:
                    self._finish_forward(output)
                return output

            block.forward = types.MethodType(wrapped, block)

    def restore(self) -> None:
        for block, original in zip(self.model.blocks, self.original_forwards):
            block.forward = original

    def begin(self, method: str) -> None:
        self.threshold = teacache_threshold(method)
        self.enabled = self.threshold is not None
        self.call_index = 0
        self.total_forwards = 0
        self.computed_forwards = 0
        self.cached_forwards = 0
        self.computed_call_indices = []
        self.cached_call_indices = []
        self.current_skip = False
        self.current_branch = 0
        self.current_input = None
        self.previous_e0 = [None, None]
        self.previous_residual = [None, None]
        self.accumulated_distance = [0.0, 0.0]

    @staticmethod
    def _rescale(relative_l1: float) -> float:
        value = 0.0
        for coefficient in TEACACHE_1P3B_REF_COEFFICIENTS:
            value = value * relative_l1 + coefficient
        return value

    def _start_forward(self, x: torch.Tensor, e0: object) -> None:
        self.total_forwards += 1
        self.current_branch = self.call_index % 2
        self.current_skip = False
        self.current_input = None
        if not self.enabled:
            self.computed_forwards += 1
            self.computed_call_indices.append(self.call_index)
            self.call_index += 1
            return
        if not isinstance(e0, torch.Tensor):
            raise RuntimeError("TeaCache block wrapper requires the Wan e tensor")

        branch = self.current_branch
        previous_e0 = self.previous_e0[branch]
        previous_residual = self.previous_residual[branch]
        must_compute = (
            self.call_index < self.retention_calls
            or previous_e0 is None
            or previous_residual is None
        )
        if must_compute:
            self.accumulated_distance[branch] = 0.0
        else:
            current = e0.detach().float()
            previous = previous_e0.float()
            denominator = previous.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
            relative_l1 = float(((current - previous).abs().mean() / denominator).item())
            self.accumulated_distance[branch] += self._rescale(relative_l1)
            must_compute = self.accumulated_distance[branch] >= float(self.threshold)
            if must_compute:
                self.accumulated_distance[branch] = 0.0

        self.previous_e0[branch] = e0.detach().clone()
        self.current_skip = not must_compute
        if self.current_skip:
            self.cached_forwards += 1
            self.cached_call_indices.append(self.call_index)
        else:
            self.computed_forwards += 1
            self.computed_call_indices.append(self.call_index)
            self.current_input = x.detach().clone()
        self.call_index += 1

    def _finish_forward(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        if self.current_skip:
            raise RuntimeError("TeaCache reached the last block on a cached forward")
        if self.current_input is None:
            raise RuntimeError("TeaCache did not preserve the full-stack input")
        self.previous_residual[self.current_branch] = x.detach() - self.current_input
        self.current_input = None

    def stats(self) -> dict[str, object]:
        return {
            "teacache_threshold": self.threshold,
            "model_forwards": self.total_forwards,
            "computed_model_forwards": self.computed_forwards,
            "cached_model_forwards": self.cached_forwards,
            "computed_call_indices": ";".join(
                str(index) for index in self.computed_call_indices
            ),
            "cached_call_indices": ";".join(
                str(index) for index in self.cached_call_indices
            ),
            "cached_model_forward_fraction": (
                self.cached_forwards / self.total_forwards
                if self.total_forwards
                else 0.0
            ),
        }


def install_grid_compatibility(wan_model_module: object) -> bool:
    block_forward = wan_model_module.WanAttentionBlock.forward
    if "grid_sizes" in inspect.signature(block_forward).parameters:
        return False

    def compatible_block_forward(
        block_self: torch.nn.Module,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: Sequence[int],
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        rope_cache: object,
    ) -> torch.Tensor:
        return block_forward(block_self, x, e, grid_sizes, context, context_lens, rope_cache)

    wan_model_module.WanAttentionBlock.forward = compatible_block_forward
    return True


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    frames = video.detach().float().clamp(-1, 1).add(1).mul(127.5)
    frames = frames.permute(1, 2, 3, 0).round().to(torch.uint8).cpu().numpy()
    imageio.mimsave(path, list(frames), fps=fps, codec="libx264", quality=8)


def main() -> None:
    args = parse_args()
    if (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must be 4n+1")
    if args.sampling_steps <= 0:
        raise ValueError("--sampling-steps must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
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

    grid_compatibility_installed = install_grid_compatibility(wan_model_module)
    backends = load_backends()
    if args.hybrid_layer_count <= 0:
        raise ValueError("--hybrid-layer-count must be positive")
    if args.warmup_steps <= 0:
        raise ValueError("--warmup-steps must be positive")
    for method in methods:
        teacache_threshold(method)
    dispatcher = AttentionDispatcher(
        backends, args.hybrid_layer_count, args.sampling_steps
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
    load_seconds = time.perf_counter() - load_start
    teacache = TeaCacheBlockController(
        pipeline.model, retention_calls=args.teacache_retention_calls
    )

    warmup: dict[str, object] | None = None
    if args.warmup_method:
        dispatcher.begin(args.warmup_method)
        teacache.begin(args.warmup_method)
        torch.cuda.synchronize(device)
        warmup_start = time.perf_counter()
        warmup_video = pipeline.generate(
            input_prompt=prompts[0],
            size=(args.width, args.height),
            frame_num=args.frame_num,
            shift=args.shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.warmup_steps,
            guide_scale=args.guide_scale,
            n_prompt=args.negative_prompt,
            seed=args.seed,
            offload_model=False,
        )
        torch.cuda.synchronize(device)
        warmup = {
            "method": args.warmup_method,
            "steps": args.warmup_steps,
            "seconds": time.perf_counter() - warmup_start,
            "self_attention_calls": dispatcher.self_calls,
            "cross_attention_calls": dispatcher.cross_calls,
            **teacache.stats(),
        }
        del warmup_video
        torch.cuda.empty_cache()

    rows: list[dict[str, object]] = []
    try:
        for prompt_index, prompt in enumerate(prompts):
            seed = args.seed + prompt_index
            for repeat_index in range(args.repeats):
                method_order = list(methods)
                if args.alternate_method_order and repeat_index % 2:
                    method_order.reverse()
                for method_order_index, method in enumerate(method_order):
                    dispatcher.begin(method)
                    teacache.begin(method)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
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
                        seconds = time.perf_counter() - start
                        repeat_suffix = (
                            f"_repeat{repeat_index:02d}" if args.repeats > 1 else ""
                        )
                        filename = (
                            f"{prompt_index:04d}_{method}_seed{seed}"
                            f"{repeat_suffix}.mp4"
                        )
                        output_path = args.out_dir / filename
                        save_video(video, output_path, args.fps)
                        rows.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "method": method,
                                "repeat": repeat_index,
                                "method_order_index": method_order_index,
                                "seed": seed,
                                "status": "ok",
                                "seconds_including_text_and_vae": seconds,
                                "self_attention_calls": dispatcher.self_calls,
                                "cross_attention_calls": dispatcher.cross_calls,
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
                                "video_file": filename,
                                "video_sha256": sha256(output_path),
                                **teacache.stats(),
                            }
                        )
                        print(
                            f"DONE prompt={prompt_index} repeat={repeat_index} "
                            f"order={method_order_index} method={method} "
                            f"seconds={seconds:.3f} self_calls={dispatcher.self_calls}",
                            flush=True,
                        )
                        del video
                    except Exception as error:
                        torch.cuda.synchronize(device)
                        seconds = time.perf_counter() - start
                        traceback.print_exc()
                        rows.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "method": method,
                                "repeat": repeat_index,
                                "method_order_index": method_order_index,
                                "seed": seed,
                                "status": "error",
                                "seconds_including_text_and_vae": seconds,
                                "self_attention_calls": dispatcher.self_calls,
                                "cross_attention_calls": dispatcher.cross_calls,
                                "error": f"{type(error).__name__}: {error}",
                                **teacache.stats(),
                            }
                        )
                    torch.cuda.empty_cache()
    finally:
        teacache.restore()
        wan_model_module.flash_attention = original_attention

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (args.out_dir / "generation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "scope": "paired decoded Wan2.1-T2V-1.3B videos on H200; only self-attention backend changes; cross-attention remains dense SDPA",
        "quality_claim": "screening evidence only unless evaluated over a dimension-balanced VBench prompt set",
        "generation_timing_caveat": "wall time includes text encoding, diffusion, and VAE decode; optional alternating order balances but does not randomize method order",
        "arguments": vars(args)
        | {
            "wan_source": str(args.wan_source),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        },
        "prompts": prompts,
        "methods": methods,
        "available_low_precision_backends": sorted(backends),
        "cross_attention_policy": "torch_sdpa",
        "teacache_policy": {
            "source": "official ali-vilab/TeaCache TeaCache4Wan2.1",
            "source_commit": TEACACHE_SOURCE_COMMIT,
            "adaptation": "first/last Wan block wrappers; MonarchRT model forward remains unchanged",
            "distance_signal": "per-branch relative L1 of e0 with official 1.3B use_ref_steps polynomial",
            "coefficients": TEACACHE_1P3B_REF_COEFFICIENTS,
            "retention_calls": args.teacache_retention_calls,
            "precision": "all computed attention calls use the selected backend; cached calls reuse a BF16 full-stack residual",
        },
        "grid_compatibility_installed": grid_compatibility_installed,
        "warmup": warmup,
        "load_seconds": load_seconds,
        "checkpoint_weight_sha256": sha256(args.checkpoint / "diffusion_pytorch_model.safetensors"),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    (args.out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
