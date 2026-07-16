from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from feature_memory_codec import sample_feature_tokens
from mvbench_llava_anchor import (
    install_transformers_forward_compatibility,
    install_visual_pooling,
    write_json_atomic,
)
from mvbench_llava_feature_memory_anchor import encode_native_feature_pool
from mvbench_utils import (
    load_mvbench_samples_by_indices,
    parse_csv_list,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)


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
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--tokens-per-video", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((base_seed + offset) % (2**63 - 1))


def core_configuration(args: argparse.Namespace) -> dict[str, object]:
    split_bytes = args.split_manifest.read_bytes()
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "llava_source": str(args.llava_source.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "split": args.split,
        "tasks": parse_csv_list(args.tasks),
        "sampled_frames": args.sampled_frames,
        "feature_pool_frames": args.feature_pool_frames,
        "pool_grid": args.pool_grid,
        "tokens_per_video": args.tokens_per_video,
        "seed": args.seed,
        "label_free_feature_fit": True,
    }


def fingerprint(configuration: dict[str, object]) -> str:
    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_token_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.llava_source))
    configuration = core_configuration(args)
    configuration_fingerprint = fingerprint(configuration)
    split_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8")
    )
    if args.split not in split_manifest:
        raise KeyError(f"split {args.split!r} is absent from manifest")
    selected_tasks = [str(value) for value in configuration["tasks"]]
    split_indices = {
        task: split_manifest[args.split][task]
        for task in selected_tasks
    }
    samples = load_mvbench_samples_by_indices(
        args.dataset_root,
        indices_by_task=split_indices,
    )
    samples = shard_samples(
        samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out_dir / f"configuration_shard_{args.shard_index}.json",
        {
            **configuration,
            "configuration_fingerprint": configuration_fingerprint,
            "argv": sys.argv,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "shard_samples": len(samples),
        },
    )

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(str(args.model_dir))
    _, model, image_processor, _ = load_pretrained_model(
        str(args.model_dir),
        None,
        model_name,
        device_map="auto",
        device=args.device,
        torch_dtype=torch.float16,
    )
    install_transformers_forward_compatibility(model)
    install_visual_pooling(model, args.pool_grid)
    model.eval()

    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint_path = (
            args.out_dir / "tokens" / f"{sample.sample_id}.pt"
        )
        if checkpoint_path.exists() and not args.overwrite:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if (
                payload.get("configuration_fingerprint")
                == configuration_fingerprint
            ):
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
            total_frames, _ = video_metadata(sample.video_path)
            sampled_indices = uniform_frame_indices(
                total_frames,
                args.sampled_frames,
            )
            pool_indices = sampled_indices[-args.feature_pool_frames :]
            started = time.perf_counter()
            features, _, timing = encode_native_feature_pool(
                sample=sample,
                frame_indices=pool_indices,
                model=model,
                image_processor=image_processor,
            )
            sampled_tokens, token_positions = sample_feature_tokens(
                features,
                count=args.tokens_per_video,
                seed=stable_sample_seed(args.seed, sample.sample_id),
            )
            elapsed = time.perf_counter() - started
            save_token_checkpoint(
                checkpoint_path,
                {
                    "format_version": 1,
                    "configuration_fingerprint": (
                        configuration_fingerprint
                    ),
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "sample_index": sample.index,
                    "pool_frame_indices": pool_indices,
                    "source_feature_shape": list(features.shape),
                    "sampled_token_positions": (
                        token_positions.detach().cpu()
                    ),
                    "tokens": sampled_tokens.detach().cpu().half(),
                    "timing": timing,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "sample_ok",
                        "sample": sample.sample_id,
                        "position": position,
                        "total": len(samples),
                        "tokens": len(sampled_tokens),
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(
                json.dumps({"event": "failure", **failure}),
                flush=True,
            )
            if args.fail_fast:
                raise
    write_json_atomic(
        args.out_dir / f"failures_shard_{args.shard_index}.json",
        failures,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
