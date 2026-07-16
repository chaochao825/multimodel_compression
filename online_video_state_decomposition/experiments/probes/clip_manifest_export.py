from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.clip_hidden_export import (  # noqa: E402
    capture_hidden_layers,
    module_version,
    read_video_segment,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--layers", default="0,7,15,22,23")
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "run_name",
        "video",
        "start_frame",
        "formal_frames",
        "formal_stride",
        "category",
    }
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def valid_existing(run_dir: Path, layers: list[int]) -> bool:
    metadata_path = run_dir / "metadata.json"
    npz_path = run_dir / "hidden.npz"
    if not metadata_path.exists() or not npz_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as archive:
            return (
                metadata.get("layers") == layers
                and all(
                    f"hidden_layer_{layer}" in archive.files
                    for layer in layers
                )
            )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    from transformers import CLIPImageProcessor, CLIPVisionModel

    layers = [int(value) for value in args.layers.split(",") if value]
    rows = read_manifest(args.manifest)
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.num_workers == args.worker_index
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    completed = []
    skipped = []
    failed = []
    for position, row in enumerate(rows, start=1):
        run_name = row["run_name"]
        run_dir = args.out_root / run_name
        if not args.overwrite and valid_existing(run_dir, layers):
            skipped.append(run_name)
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "skipped_existing",
                    }
                ),
                flush=True,
            )
            continue
        try:
            frames = int(row["formal_frames"])
            frame_stride = int(row["formal_stride"])
            start_frame = int(row["start_frame"])
            video = Path(row["video"])
            frames_rgb, frame_indices, fps, total_frames = read_video_segment(
                video,
                frames=frames,
                frame_stride=frame_stride,
                start_frame=start_frame,
                start_fraction=0.0,
                size=args.size,
            )
            hidden_arrays, hidden_shapes, grid_shape = capture_hidden_layers(
                model,
                processor,
                frames_rgb,
                layers=layers,
                batch_size=args.batch_size,
                device=device,
            )
            arrays: dict[str, np.ndarray] = {
                "frames_rgb": frames_rgb,
                "frame_indices": np.asarray(frame_indices, dtype=np.int64),
                **hidden_arrays,
                "grid_thw": np.asarray(
                    [[frames, grid_shape[0], grid_shape[1]]],
                    dtype=np.int64,
                ),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "hidden.npz", **arrays)
            metadata = {
                "manifest": str(args.manifest.resolve()),
                "manifest_row": row,
                "run_name": run_name,
                "category": row["category"],
                "model_dir": str(args.model_dir),
                "video": str(video),
                "video_sha256": sha256_file(video),
                "frames": frames,
                "frame_stride": frame_stride,
                "frame_indices": frame_indices,
                "fps": fps,
                "total_frames": total_frames,
                "size": args.size,
                "grid_thw": arrays["grid_thw"].tolist(),
                "temporal_patch": 1,
                "spatial_patch": int(model.config.patch_size),
                "layers": layers,
                "hidden_shapes": hidden_shapes,
                "worker_index": args.worker_index,
                "num_workers": args.num_workers,
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
                    "Manifest-batched CLIP hidden export. Layer 22 matches "
                    "LLaVA vision_feature_layer=-2; layer 23 is a diagnostic."
                ),
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            completed.append(run_name)
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "completed",
                        "grid": arrays["grid_thw"].tolist(),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failed.append(
                {
                    "run": run_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
    summary = {
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "assigned": len(rows),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / f"worker_{args.worker_index}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
