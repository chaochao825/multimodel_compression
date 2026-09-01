#!/usr/bin/env python3
"""Run one frozen EXP-047 Wan/rCM method on a single visible H200."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_json,
    file_sha256,
    require_fresh_output_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("teacher20", "native4", "rcm4")
STAGES = ("f17-smoke", "f81-timing", "formal")
RCM_TRAINING_METADATA_KEYS = (
    "accum_image_sample_counter",
    "accum_iteration",
    "accum_train_in_hours",
    "accum_video_sample_counter",
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    stage: str
    prompt: str
    prompt_index: int
    seed: int
    num_frames: int
    warmups: int
    repeats: int
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--prompt-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-047" or config["gate_id"] != "G-026":
        raise ValueError("config is not the frozen EXP-047/G-026 configuration")
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"method order must remain frozen as {METHODS}")
    return config


def load_formal_prompts(config: dict[str, Any]) -> tuple[str, ...]:
    path = PROJECT_ROOT / config["generation"]["formal_prompt_file"]
    prompts = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(prompts) != 4:
        raise ValueError(f"EXP-047 requires exactly four formal prompts, got {len(prompts)}")
    return prompts


def resolve_run_spec(
    config: dict[str, Any],
    method: str,
    stage: str,
    prompt_index: int | None,
    seed: int | None,
    output_dir: Path | None,
) -> RunSpec:
    prompts = load_formal_prompts(config)
    output_root = Path(config["remote"]["output_root"])
    if stage == "f17-smoke":
        if prompt_index is not None or seed is not None:
            raise ValueError("f17-smoke uses only its frozen engineering identity")
        resolved_prompt = config["generation"]["smoke_prompt"]
        resolved_prompt_index = -1
        resolved_seed = int(config["generation"]["smoke_seed"])
        frames = 17
        warmups = 0
        repeats = 1
        default_output = output_root / "smoke_f17" / method
    elif stage == "f81-timing":
        if prompt_index is not None or seed is not None:
            raise ValueError("f81-timing uses only the frozen repeated-timing identity")
        resolved_prompt_index = int(config["timing"]["formal_timing_prompt_index"])
        resolved_prompt = prompts[resolved_prompt_index]
        resolved_seed = int(config["timing"]["formal_timing_seed"])
        frames = int(config["generation"]["num_frames"])
        warmups = int(config["timing"]["smoke_warmups"])
        repeats = int(config["timing"]["smoke_repeats"])
        default_output = output_root / "timing_f81" / method
    else:
        if prompt_index is None or seed is None:
            raise ValueError("formal stage requires --prompt-index and --seed")
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise ValueError(f"formal prompt index must lie in [0, {len(prompts) - 1}]")
        if seed not in config["generation"]["formal_seeds"]:
            raise ValueError("formal seed is not registered in EXP-047")
        resolved_prompt_index = prompt_index
        resolved_prompt = prompts[prompt_index]
        resolved_seed = seed
        frames = int(config["generation"]["num_frames"])
        warmups = 0
        repeats = 1
        default_output = output_root / "formal" / method / f"p{prompt_index:02d}_s{seed}"
    if (frames - 1) % 4:
        raise ValueError("Wan frame count must be 4n+1")
    return RunSpec(
        method=method,
        stage=stage,
        prompt=resolved_prompt,
        prompt_index=resolved_prompt_index,
        seed=resolved_seed,
        num_frames=frames,
        warmups=warmups,
        repeats=repeats,
        output_dir=(output_dir or default_output).resolve(),
    )


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("net.") if key.startswith("net.") else key: value
        for key, value in state_dict.items()
    }


def remove_rcm_training_metadata(state_dict: dict[str, Any]) -> dict[str, Any]:
    for key in RCM_TRAINING_METADATA_KEYS:
        if key in state_dict:
            del state_dict[key]
    return state_dict


def verify_source(config: dict[str, Any]) -> str:
    source_root = Path(config["remote"]["rcm_root"]).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["source"]["commit"]:
        raise RuntimeError(f"rCM source mismatch: expected {config['source']['commit']}, got {commit}")
    status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"rCM source is dirty:\n{status}")
    return commit


def verify_checkpoint(config: dict[str, Any], method: str) -> tuple[Path, str]:
    if method == "rcm4":
        path = Path(config["remote"]["checkpoint"]).resolve()
        expected = config["source"]["checkpoint_sha256"]
    else:
        path = Path(config["remote"]["teacher_model"]).resolve()
        expected = config["source"]["teacher_checkpoint_sha256"]
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"checkpoint mismatch for {method}: expected {expected}, got {actual}"
        )
    return path, actual


def configure_offline_model_cache(config: dict[str, Any]) -> Path:
    cache = Path(config["remote"]["hf_home"]).resolve()
    if not cache.is_dir():
        raise FileNotFoundError(f"offline Hugging Face cache does not exist: {cache}")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return cache


def install_flash_attention_output_compat(attention_module: Any) -> bool:
    """Keep the official FA3 path while normalizing its output to one tensor."""
    if not attention_module.FLASH_ATTN_3_AVAILABLE:
        return False

    flash_attn_func = attention_module.flash_attn_func

    def output_only(*args: Any, **kwargs: Any) -> Any:
        result = flash_attn_func(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    attention_module.flash_attn_func = output_only
    return True


def import_runtime(config: dict[str, Any]) -> dict[str, Any]:
    source_root = config["remote"]["rcm_root"]
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    import torch
    from einops import repeat
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.io import save_image_or_video
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    from rcm.inference.wan2pt1_t2v_rcm_infer import dit_configs
    from rcm.samplers.unipc import FlowUniPCMultistepSampler
    from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    import rcm.utils.attention as attention_runtime
    from rcm.utils.model_utils import init_weights_on_device, load_state_dict
    from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
    from safetensors.torch import load_file

    fa3_output_compat = install_flash_attention_output_compat(attention_runtime)

    return {
        "torch": torch,
        "repeat": repeat,
        "instantiate": instantiate,
        "save_image_or_video": save_image_or_video,
        "video_sizes": VIDEO_RES_SIZE_INFO,
        "dit_configs": dit_configs,
        "unipc": FlowUniPCMultistepSampler,
        "vae_interface": Wan2pt1VAEInterface,
        "init_weights": init_weights_on_device,
        "load_state_dict": load_state_dict,
        "clear_umt5": clear_umt5_memory,
        "get_umt5": get_umt5_embedding,
        "load_safetensors": load_file,
        "fa3_output_compat": fa3_output_compat,
    }


def synchronize(torch: Any, device: Any) -> None:
    torch.cuda.synchronize(device)


def load_pipeline(
    config: dict[str, Any], method: str, runtime: dict[str, Any], device: Any
) -> tuple[Any, Any, dict[str, Any]]:
    torch = runtime["torch"]
    method_config = config["methods"][method]
    start = time.perf_counter()
    with runtime["init_weights"]():
        network = runtime["instantiate"](
            runtime["dit_configs"][config["generation"]["model_size"]]
        ).eval()
    if method_config["kind"] == "native_unipc":
        state_dict = runtime["load_safetensors"](
            config["remote"]["teacher_model"], device="cpu"
        )
        state_dict["patch_embedding.weight"] = state_dict[
            "patch_embedding.weight"
        ].flatten(1)
        checkpoint_path = config["remote"]["teacher_model"]
    else:
        state_dict = runtime["load_state_dict"](config["remote"]["checkpoint"])
        checkpoint_path = config["remote"]["checkpoint"]
    state_dict = normalize_state_dict_keys(state_dict)
    if method_config["kind"] == "rcm":
        state_dict = remove_rcm_training_metadata(state_dict)
    incompatible = network.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    network.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    tokenizer = runtime["vae_interface"](vae_pth=config["remote"]["vae"])
    synchronize(torch, device)
    return network, tokenizer, {
        "load_seconds": time.perf_counter() - start,
        "checkpoint_path": checkpoint_path,
        "missing_keys": 0,
        "unexpected_keys": 0,
    }


def encode_prompt(
    config: dict[str, Any], method: str, prompt: str, runtime: dict[str, Any], device: Any
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    torch = runtime["torch"]
    repeat = runtime["repeat"]
    text_path = config["remote"]["text_encoder"]
    text = runtime["get_umt5"](checkpoint_path=text_path, prompts=prompt)
    text = text.to(device=device, dtype=torch.bfloat16)
    condition = {"crossattn_emb": repeat(text, "b l d -> (k b) l d", k=1)}
    uncondition = None
    if config["methods"][method]["kind"] == "native_unipc":
        negative = runtime["get_umt5"](
            checkpoint_path=text_path, prompts=NEGATIVE_PROMPT
        )
        negative = negative.to(device=device, dtype=torch.bfloat16)
        uncondition = {
            "crossattn_emb": repeat(negative, "b l d -> (k b) l d", k=1)
        }
    runtime["clear_umt5"]()
    synchronize(torch, device)
    return condition, uncondition


def make_initial_noise(
    config: dict[str, Any], spec: RunSpec, tokenizer: Any, runtime: dict[str, Any], device: Any
) -> tuple[Any, Any]:
    torch = runtime["torch"]
    width, height = runtime["video_sizes"][config["generation"]["resolution"]][
        config["generation"]["aspect_ratio"]
    ]
    state_shape = (
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(spec.num_frames),
        height // tokenizer.spatial_compression_factor,
        width // tokenizer.spatial_compression_factor,
    )
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    noise = torch.randn(
        1, *state_shape, dtype=torch.float32, device=device, generator=generator
    )
    return noise, generator


def denoise_native(
    config: dict[str, Any], method: str, network: Any, noise: Any,
    generator: Any, condition: dict[str, Any], uncondition: dict[str, Any],
    runtime: dict[str, Any], device: Any,
) -> tuple[Any, int]:
    torch = runtime["torch"]
    method_config = config["methods"][method]
    x = noise.to(torch.float64)
    sigma_max = 5000.0 / 5001.0
    shift = float(method_config["timestep_shift"])
    unshifted_sigma_max = sigma_max / (shift - (shift - 1.0) * sigma_max)
    sampler = runtime["unipc"](
        num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0
    )
    sampler.set_timesteps(
        num_inference_steps=int(method_config["num_steps"]),
        device=device,
        shift=shift,
    )
    ones = torch.ones(x.size(0), 1, device=device, dtype=x.dtype)
    with torch.inference_mode():
        for timestep in sampler.timesteps:
            timesteps = timestep * ones
            model_input = x.to(dtype=torch.bfloat16)
            conditional = network(
                x_B_C_T_H_W=model_input,
                timesteps_B_T=timesteps.to(dtype=torch.bfloat16),
                **condition,
            ).float()
            unconditional = network(
                x_B_C_T_H_W=model_input,
                timesteps_B_T=timesteps.to(dtype=torch.bfloat16),
                **uncondition,
            ).float()
            prediction = unconditional + float(method_config["guidance_scale"]) * (
                conditional - unconditional
            )
            x = sampler.step(prediction, timestep, x)
    synchronize(torch, device)
    return x.float(), 2 * int(method_config["num_steps"])


def denoise_rcm(
    config: dict[str, Any], network: Any, noise: Any, generator: Any,
    condition: dict[str, Any], runtime: dict[str, Any], device: Any,
) -> tuple[Any, int]:
    torch = runtime["torch"]
    method_config = config["methods"]["rcm4"]
    steps = int(method_config["num_steps"])
    middle = [1.5, 1.4, 1.0][: steps - 1]
    t_steps = torch.tensor(
        [math.atan(float(method_config["sigma_max"])), *middle, 0.0],
        dtype=torch.float64,
        device=device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    x = noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=device, dtype=x.dtype)
    with torch.inference_mode():
        for current, following in zip(t_steps[:-1], t_steps[1:]):
            prediction = network(
                x_B_C_T_H_W=x.to(dtype=torch.bfloat16),
                timesteps_B_T=(current.float() * ones * 1000).to(
                    dtype=torch.bfloat16
                ),
                **condition,
            ).to(torch.float64)
            x = (1 - following) * (x - current * prediction) + following * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
    synchronize(torch, device)
    return x.float(), steps


def execute_once(
    config: dict[str, Any], spec: RunSpec, network: Any, tokenizer: Any,
    runtime: dict[str, Any], device: Any, output_path: Path | None,
) -> dict[str, Any]:
    torch = runtime["torch"]
    synchronize(torch, device)
    total_start = time.perf_counter()
    text_start = time.perf_counter()
    condition, uncondition = encode_prompt(
        config, spec.method, spec.prompt, runtime, device
    )
    text_seconds = time.perf_counter() - text_start

    noise, generator = make_initial_noise(config, spec, tokenizer, runtime, device)
    synchronize(torch, device)
    denoiser_start = time.perf_counter()
    if config["methods"][spec.method]["kind"] == "native_unipc":
        if uncondition is None:
            raise RuntimeError("native UniPC requires the frozen negative condition")
        samples, forward_calls = denoise_native(
            config, spec.method, network, noise, generator, condition,
            uncondition, runtime, device,
        )
    else:
        samples, forward_calls = denoise_rcm(
            config, network, noise, generator, condition, runtime, device
        )
    denoiser_seconds = time.perf_counter() - denoiser_start

    vae_start = time.perf_counter()
    video = tokenizer.decode(samples)
    synchronize(torch, device)
    vae_seconds = time.perf_counter() - vae_start
    video = ((1.0 + video.float().clamp(-1, 1)) / 2.0)[0].cpu()

    serialization_seconds = 0.0
    if output_path is not None:
        serialization_start = time.perf_counter()
        runtime["save_image_or_video"](
            video, str(output_path), fps=int(config["generation"]["fps"])
        )
        serialization_seconds = time.perf_counter() - serialization_start
    warm_e2e_seconds = time.perf_counter() - total_start
    return {
        "text_seconds": text_seconds,
        "denoiser_seconds": denoiser_seconds,
        "vae_seconds": vae_seconds,
        "serialization_seconds": serialization_seconds,
        "warm_e2e_seconds": warm_e2e_seconds,
        "sampling_steps": int(config["methods"][spec.method]["num_steps"]),
        "network_forward_calls": forward_calls,
        "denoiser_seconds_per_step": denoiser_seconds
        / int(config["methods"][spec.method]["num_steps"]),
        "denoiser_seconds_per_forward": denoiser_seconds / forward_calls,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0**2),
    }


def timing_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "text_seconds",
        "denoiser_seconds",
        "vae_seconds",
        "serialization_seconds",
        "warm_e2e_seconds",
        "denoiser_seconds_per_step",
        "denoiser_seconds_per_forward",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    return {
        f"median_{field}": float(statistics.median(float(row[field]) for row in rows))
        for field in fields
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    spec = resolve_run_spec(
        config, args.method, args.stage, args.prompt_index, args.seed, args.output_dir
    )
    require_fresh_output_dir(spec.output_dir)
    log = JsonlEventLog(spec.output_dir / "events.jsonl", f"EXP-047-{spec.stage}-{spec.method}")
    log.emit("run_start", method=spec.method, stage=spec.stage)

    source_commit = verify_source(config)
    checkpoint_path, checkpoint_sha256 = verify_checkpoint(config, spec.method)
    log.emit(
        "checkpoint_verified",
        path=str(checkpoint_path),
        sha256=checkpoint_sha256,
    )
    offline_cache = configure_offline_model_cache(config)
    log.emit("offline_model_cache_configured", path=str(offline_cache))
    runtime = import_runtime(config)
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-047 requires one visible CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"EXP-047 requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    torch.set_grad_enabled(False)

    network, tokenizer, load_info = load_pipeline(
        config, spec.method, runtime, device
    )
    log.emit("pipeline_loaded", seconds=load_info["load_seconds"])

    for warmup_index in range(spec.warmups):
        execute_once(config, spec, network, tokenizer, runtime, device, None)
        log.emit("warmup_complete", warmup=warmup_index)
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for repeat_index in range(spec.repeats):
        torch.cuda.reset_peak_memory_stats(device)
        filename = (
            f"p{spec.prompt_index:02d}_{spec.method}_seed{spec.seed}"
            f"_repeat{repeat_index:02d}.mp4"
        )
        output_path = spec.output_dir / filename
        row = execute_once(
            config, spec, network, tokenizer, runtime, device, output_path
        )
        row.update(
            {
                "method": spec.method,
                "stage": spec.stage,
                "prompt_index": spec.prompt_index,
                "prompt": spec.prompt,
                "seed": spec.seed,
                "num_frames": spec.num_frames,
                "repeat": repeat_index,
                "video_file": filename,
                "status": "ok",
            }
        )
        rows.append(row)
        atomic_write_json(spec.output_dir / "rows.partial.json", rows)
        log.emit(
            "repeat_complete",
            repeat=repeat_index,
            denoiser_seconds=row["denoiser_seconds"],
            warm_e2e_seconds=row["warm_e2e_seconds"],
        )
        torch.cuda.empty_cache()

    gpu = torch.cuda.get_device_properties(device)
    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "method": spec.method,
        "stage": spec.stage,
        "prompt_index": spec.prompt_index,
        "prompt": spec.prompt,
        "seed": spec.seed,
        "num_frames": spec.num_frames,
        "warmups": spec.warmups,
        "repeats": spec.repeats,
        "source_repository": config["source"]["repository"],
        "source_commit": source_commit,
        "checkpoint_identity": checkpoint_sha256,
        "attention_backend_policy": (
            "official rCM dense BF16 dispatcher with FA3 tensor-output compatibility"
            if runtime["fa3_output_compat"]
            else "official rCM dense BF16 dispatcher with cuDNN/SDPA fallback"
        ),
        "load": load_info,
        "rows": rows,
        "summary": timing_summary(rows),
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
    }
    atomic_write_json(spec.output_dir / "generation_manifest.json", manifest)
    atomic_write_json(
        spec.output_dir / "SUCCESS.json",
        {"status": "complete", "rows": len(rows), "method": spec.method},
    )
    log.emit("run_complete", rows=len(rows))
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
