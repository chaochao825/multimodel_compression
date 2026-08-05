"""Capture bounded LongLive causal self-attention records without changing output."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from importlib.metadata import version as package_version
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch


def install_flash_attention_3_output_compatibility(module=None) -> bool:
    """Adapt FA3 variants that return ``(output, softmax_lse)``.

    LongLive v1 consumes only the output tensor. Some installed FA3 interfaces
    always expose the auxiliary log-sum-exp value, while the interface used by
    LongLive returned only the output. Dropping that unused auxiliary tensor
    preserves the attention kernel and its numerical output.
    """

    if module is None:
        import flash_attn_interface as module

    original = module.flash_attn_varlen_func
    if getattr(original, "_longlive_output_only_compat", False):
        return True

    def output_only(*args, **kwargs):
        result = original(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    output_only._longlive_output_only_compat = True
    module.flash_attn_varlen_func = output_only
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longlive-root", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prompt-id", action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class CaptureController:
    def __init__(self, protocol: dict[str, Any], output_dir: Path) -> None:
        capture = protocol["capture"]
        self.protocol = protocol
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layers = set(int(item) for item in capture["layer_indices"])
        self.starts = set(int(item) for item in capture["current_start_frames"])
        self.occurrences = set(int(item) for item in capture["denoising_call_indices"])
        self.frame_seq_len = int(capture["frame_seq_len"])
        self.sink_frames = int(capture["sink_frames"])
        self.tile_size = int(capture["query_tile_size"])
        self.tile_count = int(capture["query_tiles_per_record"])
        self.heads = tuple(int(item) for item in capture["heads"])
        self.call_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.active: dict[str, Any] | None = None
        self.prompt: dict[str, Any] | None = None
        self.saved_paths: list[Path] = []
        self.denoising_timesteps: tuple[int, ...] = ()

    def set_denoising_timesteps(self, timesteps) -> None:
        self.denoising_timesteps = tuple(int(item) for item in timesteps)

    def set_prompt(self, prompt: dict[str, Any]) -> None:
        self.prompt = prompt
        self.call_counts.clear()

    @contextmanager
    def module_call(self, layer: int, arguments: dict[str, Any]):
        current_start_tokens = int(arguments.get("current_start", 0))
        if current_start_tokens % self.frame_seq_len != 0:
            raise ValueError("current_start is not aligned to a latent frame")
        start_frame = current_start_tokens // self.frame_seq_len
        key = (layer, start_frame)
        self.call_counts[key] += 1
        occurrence = self.call_counts[key]
        enabled = (
            layer in self.layers
            and start_frame in self.starts
            and occurrence in self.occurrences
        )
        previous = self.active
        self.active = {
            "enabled": enabled,
            "layer": layer,
            "start_frame": start_frame,
            "occurrence": occurrence,
            "grid_sizes": arguments.get("grid_sizes"),
            "freqs": arguments.get("freqs"),
        }
        try:
            yield
        finally:
            self.active = previous

    def _query_indices(self, query_tokens: int, device: torch.device) -> torch.Tensor:
        if query_tokens % self.frame_seq_len != 0:
            raise ValueError("query token count is not frame aligned")
        query_frames = query_tokens // self.frame_seq_len
        maximum_start = self.frame_seq_len - self.tile_size
        if maximum_start < 0:
            raise ValueError("query tile is larger than one latent frame")
        if self.tile_count == 1:
            starts = [maximum_start // 2]
        else:
            starts = [
                round(index * maximum_start / (self.tile_count - 1))
                for index in range(self.tile_count)
            ]
        indices: list[int] = []
        for frame in range(query_frames):
            frame_offset = frame * self.frame_seq_len
            for start in starts:
                indices.extend(range(frame_offset + start, frame_offset + start + self.tile_size))
        return torch.tensor(indices, device=device, dtype=torch.long)

    def maybe_save(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dense_output: torch.Tensor,
    ) -> None:
        active = self.active
        if not active or not active["enabled"]:
            return
        if self.prompt is None:
            raise RuntimeError("capture prompt metadata is not set")
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError("capture protocol requires batch size one")
        if key.shape[1] % self.frame_seq_len != 0:
            raise ValueError("key token count is not frame aligned")
        if max(self.heads) >= query.shape[2]:
            raise ValueError("requested head index is unavailable")
        if active["grid_sizes"] is None or active["freqs"] is None:
            raise ValueError("grid_sizes and RoPE frequencies are required for capture")

        query_indices = self._query_indices(query.shape[1], query.device)
        head_indices = torch.tensor(self.heads, device=query.device, dtype=torch.long)
        selected_query = query[0].index_select(0, query_indices).index_select(1, head_indices)
        selected_output = dense_output[0].index_select(0, query_indices).index_select(1, head_indices)
        selected_key = key[0].index_select(1, head_indices)
        selected_value = value[0].index_select(1, head_indices)

        query_frames = query.shape[1] // self.frame_seq_len
        key_frames = key.shape[1] // self.frame_seq_len
        local_frames = key_frames - self.sink_frames
        local_end_frame = int(active["start_frame"]) + query_frames
        local_start_frame = local_end_frame - local_frames
        if local_frames < 0 or local_start_frame < self.sink_frames:
            raise ValueError("cannot infer disjoint sink/local absolute frame IDs")
        key_frame_ids = list(range(self.sink_frames)) + list(
            range(local_start_frame, local_end_frame)
        )
        if len(key_frame_ids) != key_frames:
            raise AssertionError("inferred key frame IDs do not match captured K/V")
        query_frame_ids = list(
            range(int(active["start_frame"]), int(active["start_frame"]) + query_frames)
        )
        grid_sizes = active["grid_sizes"].detach().cpu().tolist()
        rope_freqs = active["freqs"].detach().to(device="cpu", dtype=torch.complex64)

        metadata = {
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "prompt_id": self.prompt["id"],
            "prompt_split": self.prompt["split"],
            "prompt_text": self.prompt["text"],
            "seed": int(self.prompt["seed"]),
            "layer": int(active["layer"]),
            "current_start_frame": int(active["start_frame"]),
            "denoising_call_index": int(active["occurrence"]),
            "denoising_timestep": (
                self.denoising_timesteps[int(active["occurrence"]) - 1]
                if int(active["occurrence"]) <= len(self.denoising_timesteps)
                else None
            ),
            "frame_seq_len": self.frame_seq_len,
            "query_tokens_full": int(query.shape[1]),
            "query_tokens_saved": int(selected_query.shape[0]),
            "key_tokens": int(key.shape[1]),
            "key_frames": int(key.shape[1] // self.frame_seq_len),
            "query_frame_ids": query_frame_ids,
            "key_frame_ids": key_frame_ids,
            "grid_sizes": grid_sizes,
            "head_indices": list(self.heads),
            "head_dim": int(query.shape[-1]),
            "query_dtype": str(query.dtype),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
        }
        name = (
            f"{self.prompt['id']}__l{active['layer']:02d}"
            f"__f{active['start_frame']:03d}__c{active['occurrence']:02d}.pt"
        )
        path = self.output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite capture artifact: {path}")
        payload = {
            "metadata": metadata,
            "query": selected_query.detach().to(device="cpu", dtype=torch.bfloat16),
            "key": selected_key.detach().to(device="cpu", dtype=torch.bfloat16),
            "value": selected_value.detach().to(device="cpu", dtype=torch.bfloat16),
            "dense_output": selected_output.detach().to(device="cpu", dtype=torch.bfloat16),
            "rope_freqs": rope_freqs,
        }
        torch.save(payload, path)
        self.saved_paths.append(path)
        print(f"[capture] {path.name} q={tuple(selected_query.shape)} k={tuple(selected_key.shape)}")


def install_capture_hooks(model: torch.nn.Module, controller: CaptureController) -> None:
    import wan.modules.causal_model as causal_model
    import wan.modules.causal_model_infinity as causal_model_infinity

    for module_globals in (causal_model, causal_model_infinity):
        original_attention = module_globals.attention

        def attention_wrapper(
            query,
            key,
            value,
            *args,
            __original=original_attention,
            **kwargs,
        ):
            output = __original(query, key, value, *args, **kwargs)
            controller.maybe_save(query, key, value, output)
            return output

        module_globals.attention = attention_wrapper

    matched_layers: set[int] = set()
    pattern = re.compile(r"(?:^|\.)blocks\.(\d+)\.self_attn$")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if not match:
            continue
        layer = int(match.group(1))
        if layer not in controller.layers:
            continue
        original_forward = module.forward
        signature = inspect.signature(original_forward)

        def forward_wrapper(*args, __layer=layer, __original=original_forward, __signature=signature, **kwargs):
            bound = __signature.bind_partial(*args, **kwargs)
            with controller.module_call(__layer, dict(bound.arguments)):
                return __original(*args, **kwargs)

        module.forward = forward_wrapper
        matched_layers.add(layer)
    if matched_layers != controller.layers:
        raise RuntimeError(
            f"capture hook layer mismatch: expected {sorted(controller.layers)}, got {sorted(matched_layers)}"
        )


def load_longlive_pipeline(config, device: torch.device):
    from pipeline import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    import peft

    pipeline = CausalInferencePipeline(config, device=device)
    checkpoint = torch.load(config.generator_ckpt, map_location="cpu", weights_only=False)
    if "generator" in checkpoint or "generator_ema" in checkpoint:
        state_dict = checkpoint["generator_ema" if config.use_ema else "generator"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise ValueError("generator state dict is missing")
    pipeline.generator.load_state_dict(state_dict)

    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )
    lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu", weights_only=False)
    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
        lora_checkpoint = lora_checkpoint["generator_lora"]
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
    pipeline.is_lora_enabled = True
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    return pipeline


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty capture directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_commit = protocol["model"]["code_commit"]
    actual_commit = source_commit(args.longlive_root)
    if actual_commit != expected_commit:
        raise RuntimeError(f"LongLive commit mismatch: {actual_commit} != {expected_commit}")

    os.chdir(args.longlive_root)
    sys.path.insert(0, str(args.longlive_root))
    from omegaconf import OmegaConf
    from utils.misc import set_seed

    config = OmegaConf.load(args.runtime_config)
    generator_path = Path(str(config.generator_ckpt))
    lora_path = Path(str(config.lora_ckpt))
    checkpoint_hashes = {
        "generator_sha256": sha256_file(generator_path),
        "lora_sha256": sha256_file(lora_path),
    }
    for name, actual in checkpoint_hashes.items():
        expected = protocol["model"][name]
        if actual != expected:
            raise RuntimeError(f"{name} mismatch: {actual} != {expected}")

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit CUDA device such as cuda:2 is required")
    torch.cuda.set_device(device.index)
    torch.set_grad_enabled(False)

    fa3_output_compatibility = install_flash_attention_3_output_compatibility()
    pipeline = load_longlive_pipeline(config, device)
    controller = CaptureController(protocol, args.output_dir)
    controller.set_denoising_timesteps(pipeline.denoising_step_list.tolist())
    install_capture_hooks(pipeline.generator.model, controller)

    selected_ids = set(args.prompt_id)
    prompts = [
        prompt for prompt in protocol["capture"]["prompts"]
        if not selected_ids or prompt["id"] in selected_ids
    ]
    if not prompts:
        raise ValueError("no protocol prompts selected")

    for prompt in prompts:
        set_seed(int(prompt["seed"]))
        controller.set_prompt(prompt)
        generator = torch.Generator(device=device).manual_seed(int(prompt["seed"]))
        noise = torch.randn(
            [1, int(config.num_output_frames), 16, 60, 104],
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        print(f"[prompt] {prompt['id']} seed={prompt['seed']}")
        video, _ = pipeline.inference(
            noise=noise,
            text_prompts=[prompt["text"]],
            return_latents=True,
            low_memory=False,
            profile=False,
        )
        del video
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "source_commit": actual_commit,
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": sha256_file(args.runtime_config),
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "generator_checkpoint": str(generator_path),
        "generator_sha256": checkpoint_hashes["generator_sha256"],
        "lora_checkpoint": str(lora_path),
        "lora_sha256": checkpoint_hashes["lora_sha256"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fa3_output_only_compatibility": fa3_output_compatibility,
        "python_executable": sys.executable,
        "package_versions": {
            name: package_version(name)
            for name in ("transformers", "diffusers", "peft", "omegaconf", "accelerate")
        },
        "device": torch.cuda.get_device_name(device),
        "captures": [path.name for path in controller.saved_paths],
    }
    manifest_path = args.output_dir / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_path} captures={len(controller.saved_paths)}")


if __name__ == "__main__":
    main()
