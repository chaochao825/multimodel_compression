from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from mvbench_clip_memory import (
    encode_images,
    encode_text,
    fingerprint,
    write_json_atomic,
)
from mvbench_utils import (
    clip_candidate_prompts,
    clip_question_prompt,
    decode_video_frames,
    load_mvbench_samples_by_indices,
    parse_csv_list,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", default="calibration,evaluation")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_samples(
    dataset_root: Path,
    split_manifest: dict[str, object],
    split_names: list[str],
) -> tuple[list[object], dict[str, str]]:
    indices_by_task: dict[str, list[int]] = {
        str(task): []
        for task in split_manifest["tasks"]
    }
    sample_splits: dict[str, str] = {}
    for split_name in split_names:
        split = split_manifest[split_name]
        for task in split_manifest["tasks"]:
            for index in split[task]:
                indices_by_task[str(task)].append(int(index))
                sample_splits[f"{task}_{int(index):04d}"] = split_name
    samples = load_mvbench_samples_by_indices(
        dataset_root,
        indices_by_task=indices_by_task,
    )
    return samples, sample_splits


def main() -> int:
    args = parse_args()
    split_names = parse_csv_list(args.splits)
    split_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8")
    )
    for split_name in split_names:
        if split_name not in {"calibration", "evaluation"}:
            raise ValueError(f"unsupported split: {split_name}")
    samples, sample_splits = load_samples(
        args.dataset_root,
        split_manifest,
        split_names,
    )
    samples = shard_samples(
        samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "split_manifest_name": args.split_manifest.name,
        "split_manifest_sha256": path_sha256(args.split_manifest),
        "splits": split_names,
        "frames": args.frames,
    }
    config_fingerprint = fingerprint(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out_dir / "configuration.json",
        {
            **config,
            "configuration_fingerprint": config_fingerprint,
            "argv": sys.argv,
        },
    )

    from transformers import (
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizer,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        use_fast=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )

    completed = 0
    failures: list[dict[str, object]] = []
    for position, sample in enumerate(samples, start=1):
        npz_path = args.out_dir / "cache" / f"{sample.sample_id}.npz"
        metadata_path = args.out_dir / "cache" / f"{sample.sample_id}.json"
        if npz_path.exists() and metadata_path.exists() and not args.overwrite:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if metadata.get("configuration_fingerprint") == config_fingerprint:
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "resume",
                            "sample": sample.sample_id,
                            "position": position,
                            "total": len(samples),
                        }
                    ),
                    flush=True,
                )
                continue
        try:
            total_frames, fps = video_metadata(sample.video_path)
            frame_indices = uniform_frame_indices(total_frames, args.frames)
            decode_start = time.perf_counter()
            frames, decoded_fps, decoded_total = decode_video_frames(
                sample.video_path,
                frame_indices,
            )
            decode_seconds = time.perf_counter() - decode_start
            if decoded_total != total_frames:
                raise RuntimeError("video frame count changed during decoding")
            if decoded_fps > 0:
                fps = decoded_fps

            image_start = time.perf_counter()
            image_vectors = encode_images(
                model,
                image_processor,
                frames,
                batch_size=args.image_batch_size,
                device=device,
            )
            image_encode_seconds = time.perf_counter() - image_start

            text_start = time.perf_counter()
            question_vector = encode_text(
                model,
                tokenizer,
                [clip_question_prompt(sample)],
                device=device,
            )[0]
            candidate_vectors = encode_text(
                model,
                tokenizer,
                clip_candidate_prompts(sample),
                device=device,
            )
            text_encode_seconds = time.perf_counter() - text_start

            atomic_save_npz(
                npz_path,
                image_vectors=image_vectors.astype(np.float16),
                question_vector=question_vector.astype(np.float16),
                candidate_vectors=candidate_vectors.astype(np.float16),
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
            )
            write_json_atomic(
                metadata_path,
                {
                    "configuration_fingerprint": config_fingerprint,
                    "sample_id": sample.sample_id,
                    "split": sample_splits[sample.sample_id],
                    "task": sample.task,
                    "sample_index": sample.index,
                    "video": str(sample.video_path),
                    "question": sample.question,
                    "candidates": list(sample.candidates),
                    "answer": sample.answer,
                    "answer_index": sample.answer_index,
                    "fps": fps,
                    "total_frames": total_frames,
                    "sampled_frames": len(frame_indices),
                    "frame_indices": frame_indices,
                    "decode_seconds": decode_seconds,
                    "image_encode_seconds": image_encode_seconds,
                    "text_encode_seconds": text_encode_seconds,
                    "image_embedding_shape": list(image_vectors.shape),
                    "question_embedding_shape": list(question_vector.shape),
                    "candidate_embedding_shape": list(
                        candidate_vectors.shape
                    ),
                },
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "sample_ok",
                        "sample": sample.sample_id,
                        "split": sample_splits[sample.sample_id],
                        "position": position,
                        "total": len(samples),
                        "seconds": (
                            decode_seconds
                            + image_encode_seconds
                            + text_encode_seconds
                        ),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "sample": sample.sample_id,
                "task": sample.task,
                "type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            write_json_atomic(
                args.out_dir / "errors" / f"{sample.sample_id}.json",
                {
                    "configuration_fingerprint": config_fingerprint,
                    **failure,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "sample_fail",
                        "sample": sample.sample_id,
                        "type": type(exc).__name__,
                        "error": repr(exc),
                    }
                ),
                flush=True,
            )
            if args.fail_fast:
                raise

    shard_name = f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    write_json_atomic(
        args.out_dir / f"{shard_name}_status.json",
        {
            "configuration_fingerprint": config_fingerprint,
            "assigned_samples": len(samples),
            "completed_samples": completed,
            "failures": failures,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
