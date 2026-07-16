from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=-1)
    parser.add_argument("--start-fraction", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--layers", default="0,8,16,26")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def install_register_fake_shim() -> None:
    if hasattr(torch.library, "register_fake"):
        return

    def register_fake(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    torch.library.register_fake = register_fake  # type: ignore[attr-defined]


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT_ERROR:{exc.__class__.__name__}:{exc}"
    return str(getattr(module, "__version__", "unknown"))


def sha256_file(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def read_video_segment(
    path: Path,
    *,
    frames: int,
    frame_stride: int,
    start_frame: int,
    start_fraction: float,
    size: int,
) -> tuple[np.ndarray, list[int], float, int]:
    import cv2

    if frames <= 0 or frames % 2:
        raise ValueError("frames must be a positive even number for temporal patch size 2")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    required_span = (frames - 1) * frame_stride
    max_start = max(total - required_span - 1, 0)
    if start_frame < 0:
        chosen_start = int(round(max_start * min(max(start_fraction, 0.0), 1.0)))
    else:
        chosen_start = min(max(start_frame, 0), max_start)
    indices = [chosen_start + index * frame_stride for index in range(frames)]
    decoded: list[np.ndarray] = []
    actual_indices: list[int] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        decoded.append(frame)
        actual_indices.append(int(index))
    capture.release()
    if len(decoded) != frames:
        raise RuntimeError(
            f"decoded {len(decoded)} of {frames} requested frames from {path}"
        )
    return np.stack(decoded), actual_indices, fps, total


def video_to_qwen_patches(
    video: np.ndarray,
    temporal_patch: int = 2,
    patch: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    time, height, width, channels = video.shape
    if channels != 3:
        raise ValueError(f"expected RGB video, got {video.shape}")
    time = (time // temporal_patch) * temporal_patch
    height = (height // patch) * patch
    width = (width // patch) * patch
    video = video[:time, :height, :width]
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    time_grid = time // temporal_patch
    height_grid = height // patch
    width_grid = width // patch
    patches = tensor.reshape(
        time_grid,
        temporal_patch,
        3,
        height_grid,
        patch,
        width_grid,
        patch,
    )
    patches = patches.permute(0, 3, 5, 2, 1, 4, 6).contiguous()
    hidden_states = patches.reshape(time_grid * height_grid * width_grid, -1)
    grid_thw = torch.tensor(
        [[time_grid, height_grid, width_grid]],
        dtype=torch.long,
    )
    return hidden_states, grid_thw


def load_visual_model(model_dir: Path, device: torch.device):
    install_register_fake_shim()
    from transformers import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeVisionModel,
    )

    config = Qwen3VLMoeConfig.from_pretrained(str(model_dir))
    config.vision_config._attn_implementation = "eager"
    model = Qwen3VLMoeVisionModel(config.vision_config)
    model.to(dtype=torch.bfloat16)

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        if key.startswith("model.visual."):
            by_shard.setdefault(shard, []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                state[key.removeprefix("model.visual.")] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "visual_shards": sorted(by_shard),
    }


def capture_hidden_states(
    model,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    layers: list[int],
) -> dict[int, np.ndarray]:
    captures: dict[int, torch.Tensor] = {}
    handles = []

    for layer in layers:
        if not 0 <= layer < len(model.blocks):
            raise ValueError(f"layer {layer} is outside [0,{len(model.blocks) - 1}]")

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captures[layer_index] = tensor.detach().to(
                    device="cpu",
                    dtype=torch.float16,
                )

            return hook

        handles.append(model.blocks[layer].register_forward_hook(make_hook(layer)))

    model_device = next(model.parameters()).device
    with torch.inference_mode():
        _ = model(hidden_states.to(model_device), grid_thw.to(model_device))
    for handle in handles:
        handle.remove()

    time_grid, height_grid, width_grid = [
        int(value) for value in grid_thw[0].tolist()
    ]
    arrays = {}
    for layer in layers:
        tensor = captures[layer]
        arrays[layer] = tensor.reshape(
            time_grid,
            height_grid,
            width_grid,
            tensor.shape[-1],
        ).numpy()
    return arrays


def main() -> int:
    args = parse_args()
    layers = [int(value) for value in args.layers.split(",") if value]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frames_rgb, frame_indices, fps, total_frames = read_video_segment(
        args.video,
        frames=args.frames,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
        start_fraction=args.start_fraction,
        size=args.size,
    )
    patch_input, grid_thw = video_to_qwen_patches(frames_rgb)
    model, load_info = load_visual_model(args.model_dir, device)
    captures = capture_hidden_states(model, patch_input, grid_thw, layers)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "frames_rgb": frames_rgb,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "grid_thw": grid_thw.numpy(),
    }
    for layer, hidden in captures.items():
        arrays[f"hidden_layer_{layer}"] = hidden
    np.savez_compressed(args.out_npz, **arrays)

    metadata = {
        "argv": sys.argv,
        "model_dir": str(args.model_dir),
        "video": str(args.video),
        "video_sha256": sha256_file(args.video),
        "video_size_bytes": args.video.stat().st_size,
        "model_index_sha256": sha256_file(
            args.model_dir / "model.safetensors.index.json"
        ),
        "frames": args.frames,
        "frame_stride": args.frame_stride,
        "frame_indices": frame_indices,
        "fps": fps,
        "total_frames": total_frames,
        "size": args.size,
        "grid_thw": grid_thw.tolist(),
        "temporal_patch": 2,
        "spatial_patch": 16,
        "layers": layers,
        "hidden_shapes": {
            str(layer): [int(value) for value in hidden.shape]
            for layer, hidden in captures.items()
        },
        "device_requested": args.device,
        "device_used": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "versions": {
            "python": sys.version,
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "safetensors": module_version("safetensors"),
            "numpy": module_version("numpy"),
            "cv2": module_version("cv2"),
        },
        "load_info": load_info,
        "scope": (
            "Consecutive Qwen3-VL visual-block hidden states. This export tests "
            "cross-frame latent dynamics and does not represent language-model KV state."
        ),
    }
    args.out_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_npz": str(args.out_npz),
                "out_json": str(args.out_json),
                "grid_thw": metadata["grid_thw"],
                "hidden_shapes": metadata["hidden_shapes"],
                "frame_indices": frame_indices,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
