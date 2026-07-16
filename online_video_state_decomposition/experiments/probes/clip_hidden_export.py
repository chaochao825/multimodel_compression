from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=-1)
    parser.add_argument("--start-fraction", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--layers", default="0,7,15,22,23")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT_ERROR:{exc.__class__.__name__}:{exc}"
    return str(getattr(module, "__version__", "unknown"))


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
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

    if frames <= 0:
        raise ValueError("frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    required_span = (frames - 1) * frame_stride
    max_start = max(total - required_span - 1, 0)
    chosen_start = (
        int(round(max_start * min(max(start_fraction, 0.0), 1.0)))
        if start_frame < 0
        else min(max(start_frame, 0), max_start)
    )
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


def capture_hidden_layers(
    model: torch.nn.Module,
    processor: object,
    frames_rgb: np.ndarray,
    *,
    layers: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]], tuple[int, int]]:
    captures: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for start in range(0, len(frames_rgb), batch_size):
        batch_frames = [
            frame for frame in frames_rgb[start : start + batch_size]
        ]
        inputs = processor(images=batch_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        with torch.inference_mode():
            outputs = model(pixel_values, output_hidden_states=True)
        for layer in layers:
            # hidden_states[0] is the embedding output; layer k is at index k+1.
            hidden = outputs.hidden_states[layer + 1][:, 1:]
            token_count = hidden.shape[1]
            side = int(round(token_count**0.5))
            if side * side != token_count:
                raise RuntimeError(
                    f"CLIP patch token count is not square: {token_count}"
                )
            captures[layer].append(
                hidden.reshape(
                    hidden.shape[0],
                    side,
                    side,
                    hidden.shape[-1],
                )
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )

    arrays: dict[str, np.ndarray] = {}
    hidden_shapes: dict[str, list[int]] = {}
    grid_shape = None
    for layer, chunks in captures.items():
        hidden = np.concatenate(chunks, axis=0)
        arrays[f"hidden_layer_{layer}"] = hidden
        hidden_shapes[str(layer)] = [int(value) for value in hidden.shape]
        grid_shape = (int(hidden.shape[1]), int(hidden.shape[2]))
    if grid_shape is None:
        raise RuntimeError("no layers were captured")
    return arrays, hidden_shapes, grid_shape


def main() -> int:
    args = parse_args()
    from transformers import CLIPImageProcessor, CLIPVisionModel

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
    processor = CLIPImageProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )
    model = CLIPVisionModel.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    layer_count = int(model.config.num_hidden_layers)
    for layer in layers:
        if not 0 <= layer < layer_count:
            raise ValueError(f"layer {layer} is outside [0,{layer_count - 1}]")

    arrays: dict[str, np.ndarray] = {
        "frames_rgb": frames_rgb,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
    }
    hidden_arrays, hidden_shapes, grid_shape = capture_hidden_layers(
        model,
        processor,
        frames_rgb,
        layers=layers,
        batch_size=args.batch_size,
        device=device,
    )
    arrays.update(hidden_arrays)
    arrays["grid_thw"] = np.asarray(
        [[args.frames, int(grid_shape[0]), int(grid_shape[1])]],
        dtype=np.int64,
    )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **arrays)
    metadata = {
        "argv": sys.argv,
        "model_dir": str(args.model_dir),
        "video": str(args.video),
        "video_sha256": sha256_file(args.video),
        "frames": args.frames,
        "frame_stride": args.frame_stride,
        "frame_indices": frame_indices,
        "fps": fps,
        "total_frames": total_frames,
        "size": args.size,
        "grid_thw": arrays["grid_thw"].tolist(),
        "temporal_patch": 1,
        "spatial_patch": int(model.config.patch_size),
        "layers": layers,
        "hidden_shapes": hidden_shapes,
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
            "numpy": module_version("numpy"),
            "cv2": module_version("cv2"),
        },
        "scope": (
            "Per-frame CLIP ViT hidden patch tokens used as the LLaVA-1.5 "
            "visual-tower cross-encoder probe. CLS tokens are excluded. "
            "Layer 22 matches LLaVA's configured vision_feature_layer=-2; "
            "layer 23 is retained as a final-block diagnostic."
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
                "hidden_shapes": hidden_shapes,
                "frame_indices": frame_indices,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
