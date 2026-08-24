from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from feature_memory_codec import fit_pca_codec, save_codec
from mvbench_onevision_utils import (
    decode_feature_pool,
    encode_video_features,
    expected_feature_state_bytes,
    load_onevision_model,
    preprocess_video,
)
from mvbench_utils import load_mvbench_samples, parse_csv_list


DEFAULT_TASKS = (
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=384)
    parser.add_argument("--residual-tokens", type=int, default=4)
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    tasks = parse_csv_list(args.tasks)
    samples = load_mvbench_samples(
        args.dataset_root,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        selection_seed=args.selection_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    feature_blocks = []
    sample_ids = []
    extraction_started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        frames, _, _ = decode_feature_pool(
            sample,
            sampled_frames=args.sampled_frames,
            feature_pool_frames=args.feature_pool_frames,
            frame_budget=args.frame_budget,
        )
        pixels = preprocess_video(
            processor,
            frames,
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            features = encode_video_features(model, pixels)
        feature_blocks.append(features.to(device="cpu", dtype=torch.float16))
        sample_ids.append(sample.sample_id)
        print(
            json.dumps(
                {
                    "event": "feature_ok",
                    "sample": sample.sample_id,
                    "position": position,
                    "total": len(samples),
                    "shape": list(features.shape),
                }
            ),
            flush=True,
        )

    source_shapes = {tuple(block.shape) for block in feature_blocks}
    if len(source_shapes) != 1:
        raise RuntimeError("OneVision feature shapes differ across samples")
    extraction_seconds = time.perf_counter() - extraction_started
    training_features = torch.cat(feature_blocks, dim=0).to(device)
    fit_started = time.perf_counter()
    codec, fit_metadata = fit_pca_codec(
        training_features,
        rank=args.rank,
        seed=args.selection_seed,
        niter=args.niter,
        storage_dtype=torch.float16,
    )
    fit_seconds = time.perf_counter() - fit_started
    frames, tokens_per_frame, hidden_size = next(iter(source_shapes))
    accounting = expected_feature_state_bytes(
        frames=frames,
        tokens_per_frame=tokens_per_frame,
        hidden_size=hidden_size,
        rank=args.rank,
        residual_tokens_per_frame=args.residual_tokens,
    )
    metadata = {
        **fit_metadata,
        "claim_tier": "calibration_only_onevision_feature_codec",
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "selection_seed": args.selection_seed,
        "sampled_frames": args.sampled_frames,
        "feature_pool_frames": args.feature_pool_frames,
        "frame_budget": args.frame_budget,
        "source_feature_shape": list(next(iter(source_shapes))),
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "extraction_seconds": extraction_seconds,
        "fit_seconds": fit_seconds,
        "state_accounting": accounting,
    }
    codec_path = args.out_dir / f"onevision_feature_pca_rank{args.rank}.pt"
    save_codec(codec, codec_path, metadata=metadata)
    write_json(
        args.out_dir / "fit_summary.json",
        {**metadata, "codec_path": str(codec_path.resolve())},
    )
    print(json.dumps({"codec_path": str(codec_path), **accounting}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
