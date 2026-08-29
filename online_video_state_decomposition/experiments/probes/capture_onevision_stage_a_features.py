from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    encode_video_features,
    load_onevision_model,
    preprocess_video,
)
from mvbench_utils import decode_video_frames, uniform_frame_indices, video_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--expected-protocol-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_id"] != args.expected_protocol_id:
        raise ValueError("feature manifest protocol identity mismatch")
    if args.role not in manifest["roles"]:
        raise KeyError(f"unknown manifest role: {args.role}")
    samples = [
        entry
        for index, entry in enumerate(manifest["roles"][args.role])
        if index % args.shard_count == args.shard_index
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    completed = []
    started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        output_path = args.out_dir / f"{sample['sample_id']}.pt"
        if output_path.is_file() and not args.overwrite:
            completed.append(str(sample["sample_id"]))
            continue
        video_path = args.video_root / str(sample["relative_video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        total_frames, _ = video_metadata(video_path)
        pool_indices = uniform_frame_indices(total_frames, args.feature_pool_frames)
        frames, _, decoded_total = decode_video_frames(video_path, pool_indices)
        if decoded_total != total_frames:
            raise RuntimeError("video frame count changed during decoding")
        pixels = preprocess_video(
            processor,
            np.stack(frames),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            features = encode_video_features(model, pixels)
        payload = {
            "protocol_id": args.expected_protocol_id,
            "role": args.role,
            "domain": str(sample["domain"]),
            "sample_id": str(sample["sample_id"]),
            "pool_indices": pool_indices,
            "features": features.to(device="cpu", dtype=torch.float16),
        }
        temporary_path = output_path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
        completed.append(str(sample["sample_id"]))
        print(
            json.dumps(
                {
                    "event": "feature_ok",
                    "sample_id": sample["sample_id"],
                    "position": position,
                    "total": len(samples),
                }
            ),
            flush=True,
        )
    summary = {
        "protocol_id": args.expected_protocol_id,
        "role": args.role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_count": len(samples),
        "completed_sample_ids": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_pool_frames": args.feature_pool_frames,
    }
    write_json_atomic(
        args.out_dir / f"summary_shard_{args.shard_index}.json",
        summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
