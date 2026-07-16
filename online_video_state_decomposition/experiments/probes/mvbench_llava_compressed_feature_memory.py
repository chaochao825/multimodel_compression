from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from feature_memory_codec import (
    EncodedFeatureMemory,
    encode_feature_memory,
    load_codec,
    reconstruct_feature_memory,
    relative_reconstruction_error,
)
from mvbench_llava_anchor import (
    install_transformers_forward_compatibility,
    install_visual_pooling,
    load_manifest_samples,
    write_json_atomic,
)
from mvbench_llava_feature_memory_anchor import (
    DEFAULT_POLICIES,
    DEFAULT_TASKS,
    encode_native_feature_pool,
    feature_cache_bytes,
    read_from_native_feature_pool,
    selected_positions,
)
from mvbench_utils import (
    parse_csv_list,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260717)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--residual-tokens", default="0,1,2,4")
    parser.add_argument("--exclude-full", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--include-subtitle", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_nonnegative_ints(value: str) -> list[int]:
    parsed = sorted({int(item) for item in parse_csv_list(value)})
    if parsed[0] < 0:
        raise ValueError("residual token counts must be non-negative")
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def variant_name(rank: int, residual_tokens: int) -> str:
    return f"pca_r{rank}_s{residual_tokens}"


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "llava_source": str(args.llava_source.resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "codec_path": str(args.codec_path.resolve()),
        "codec_sha256": sha256(args.codec_path),
        "tasks": parse_csv_list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "selection_seed": args.selection_seed,
        "sampled_frames": args.sampled_frames,
        "feature_pool_frames": args.feature_pool_frames,
        "frame_budget": args.frame_budget,
        "pool_grid": args.pool_grid,
        "policies": parse_csv_list(args.policies),
        "residual_tokens": parse_nonnegative_ints(
            args.residual_tokens
        ),
        "include_full": not args.exclude_full,
        "max_new_tokens": args.max_new_tokens,
        "include_subtitle": args.include_subtitle,
        "native_feature_memory": True,
        "compressed_native_feature_memory": True,
        "raw_frame_replay_at_read": False,
    }


def fingerprint(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_compressed_states(
    features: torch.Tensor,
    *,
    codec: object,
    residual_tokens: list[int],
) -> tuple[
    dict[str, EncodedFeatureMemory],
    dict[str, dict[str, float | int]],
]:
    states = {}
    metadata = {}
    for count in residual_tokens:
        synchronize()
        started = time.perf_counter()
        state = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=count,
        )
        synchronize()
        compression_seconds = time.perf_counter() - started
        reconstruction = reconstruct_feature_memory(
            state,
            codec,
            output_dtype=features.dtype,
        )
        name = variant_name(codec.rank, count)
        states[name] = state
        metadata[name] = {
            "compression_seconds": compression_seconds,
            "pool_reconstruction_relative_error": (
                relative_reconstruction_error(
                    features,
                    reconstruction,
                )
            ),
            "native_feature_state_bytes": state.stream_state_bytes,
            "latent_state_bytes": state.latent_bytes,
            "residual_value_bytes": state.residual_value_bytes,
            "residual_index_bytes": state.residual_index_bytes,
            "residual_tokens_per_frame": count,
        }
    return states, metadata


def run_sample(
    *,
    sample: object,
    policies: list[str],
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
    codec: object,
    residual_tokens: list[int],
    include_full: bool,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
    pool_grid: int,
    max_new_tokens: int,
    include_subtitle: bool,
) -> list[dict[str, object]]:
    total_frames, fps = video_metadata(sample.video_path)
    sampled_indices = uniform_frame_indices(total_frames, sampled_frames)
    pool_indices = sampled_indices[-feature_pool_frames:]
    sample_manifest = selection_manifest["samples"][sample.sample_id]
    selected_by_policy = {
        policy: [
            int(index)
            for index in sample_manifest["policies"][policy]
        ]
        for policy in policies
    }
    positions_by_policy = {
        policy: selected_positions(pool_indices, indices)
        for policy, indices in selected_by_policy.items()
    }
    features, image_sizes, write_timing = encode_native_feature_pool(
        sample=sample,
        frame_indices=pool_indices,
        model=model,
        image_processor=image_processor,
    )
    dense_cache_bytes = feature_cache_bytes(features)
    compressed_states, compressed_metadata = build_compressed_states(
        features,
        codec=codec,
        residual_tokens=residual_tokens,
    )
    accounting_by_policy = selection_manifest.get(
        "policy_accounting",
        {},
    )
    provisioned_selector_bytes = max(
        int(
            accounting_by_policy.get(policy, {}).get(
                "total_state_bytes",
                0,
            )
        )
        for policy in policies
    )
    if not getattr(model, "_compressed_feature_memory_warmed_up", False):
        warmup_policy = policies[0]
        warmup_positions = positions_by_policy[warmup_policy]
        warmup_tensor = torch.tensor(
            warmup_positions,
            device=features.device,
            dtype=torch.long,
        )
        read_from_native_feature_pool(
            sample=sample,
            policy=f"{warmup_policy}__warmup",
            selected_frame_indices=selected_by_policy[warmup_policy],
            selected_features=features.index_select(
                0,
                warmup_tensor,
            ).contiguous(),
            selected_image_sizes=[
                image_sizes[position]
                for position in warmup_positions
            ],
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            include_subtitle=include_subtitle,
        )
        model._compressed_feature_memory_warmed_up = True

    variants = []
    if include_full:
        variants.append("full")
    variants.extend(sorted(compressed_states))
    rows = []
    for variant in variants:
        state = compressed_states.get(variant)
        variant_metadata = compressed_metadata.get(
            variant,
            {
                "compression_seconds": 0.0,
                "pool_reconstruction_relative_error": 0.0,
                "native_feature_state_bytes": dense_cache_bytes,
                "latent_state_bytes": dense_cache_bytes,
                "residual_value_bytes": 0,
                "residual_index_bytes": 0,
                "residual_tokens_per_frame": 0,
            },
        )
        for policy in policies:
            positions = positions_by_policy[policy]
            position_tensor = torch.tensor(
                positions,
                device=features.device,
                dtype=torch.long,
            )
            reference_features = features.index_select(
                0,
                position_tensor,
            ).contiguous()
            synchronize()
            reconstruction_started = time.perf_counter()
            if state is None:
                selected_features = reference_features
            else:
                selected_features = reconstruct_feature_memory(
                    state,
                    codec,
                    frame_positions=position_tensor,
                    output_dtype=features.dtype,
                ).contiguous()
            synchronize()
            reconstruction_seconds = (
                time.perf_counter() - reconstruction_started
            )
            selected_error = relative_reconstruction_error(
                reference_features,
                selected_features,
            )
            composite_policy = f"{policy}__{variant}"
            result = read_from_native_feature_pool(
                sample=sample,
                policy=composite_policy,
                selected_frame_indices=selected_by_policy[policy],
                selected_features=selected_features,
                selected_image_sizes=[
                    image_sizes[position]
                    for position in positions
                ],
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                include_subtitle=include_subtitle,
            )
            feature_state_bytes = int(
                variant_metadata["native_feature_state_bytes"]
            )
            total_state_bytes = (
                feature_state_bytes + provisioned_selector_bytes
            )
            policy_accounting = accounting_by_policy.get(policy, {})
            result.update(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "sample_index": sample.index,
                    "video": str(sample.video_path),
                    "question": sample.question,
                    "candidates": list(sample.candidates),
                    "answer": sample.answer,
                    "answer_index": int(sample.answer_index),
                    "selection_policy": policy,
                    "memory_variant": variant,
                    "frame_indices": selected_by_policy[policy],
                    "pool_frame_indices": pool_indices,
                    "fps": fps,
                    "total_frames": total_frames,
                    "decode_seconds": write_timing["decode_seconds"],
                    "preprocess_seconds": write_timing[
                        "preprocess_seconds"
                    ],
                    "vision_encode_seconds": write_timing[
                        "vision_encode_seconds"
                    ],
                    "feature_cache_write_seconds": write_timing[
                        "write_seconds"
                    ],
                    "compression_seconds": float(
                        variant_metadata["compression_seconds"]
                    ),
                    "reconstruction_seconds": reconstruction_seconds,
                    "policy_seconds": (
                        write_timing["write_seconds"]
                        + float(variant_metadata["compression_seconds"])
                        + reconstruction_seconds
                        + float(result["inference_seconds"])
                    ),
                    "visual_tokens": (
                        frame_budget * pool_grid * pool_grid
                    ),
                    "selection_state_proxy_bytes": total_state_bytes,
                    "actual_selector_state_bytes": int(
                        policy_accounting.get("total_state_bytes", 0)
                    ),
                    "provisioned_selector_state_bytes": (
                        provisioned_selector_bytes
                    ),
                    "dense_feature_cache_bytes": dense_cache_bytes,
                    "native_feature_state_bytes": feature_state_bytes,
                    "latent_state_bytes": int(
                        variant_metadata["latent_state_bytes"]
                    ),
                    "residual_value_bytes": int(
                        variant_metadata["residual_value_bytes"]
                    ),
                    "residual_index_bytes": int(
                        variant_metadata["residual_index_bytes"]
                    ),
                    "codec_parameter_bytes": codec.parameter_bytes,
                    "codec_rank": codec.rank,
                    "residual_tokens_per_frame": int(
                        variant_metadata[
                            "residual_tokens_per_frame"
                        ]
                    ),
                    "pool_reconstruction_relative_error": float(
                        variant_metadata[
                            "pool_reconstruction_relative_error"
                        ]
                    ),
                    "selected_reconstruction_relative_error": (
                        selected_error
                    ),
                    "feature_state_compression_ratio": (
                        dense_cache_bytes / feature_state_bytes
                    ),
                    "total_state_compression_ratio": (
                        (dense_cache_bytes + provisioned_selector_bytes)
                        / total_state_bytes
                    ),
                    "llm_visual_token_bytes": (
                        frame_budget
                        * pool_grid
                        * pool_grid
                        * int(features.shape[-1])
                        * features.element_size()
                    ),
                    "selection_online_bounded": 1,
                    "visual_evidence_cache_counted": 1,
                    "raw_frame_replay_at_read": 0,
                    "matched_provisioned_state": 1,
                }
            )
            rows.append(result)
    return rows


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.llava_source))
    config = config_from_args(args)
    config_fingerprint = fingerprint(config)
    selection_manifest = json.loads(
        args.selection_manifest.read_text(encoding="utf-8")
    )
    policies = [str(value) for value in config["policies"]]
    samples = load_manifest_samples(
        args.dataset_root,
        selection_manifest,
        tasks=[str(value) for value in config["tasks"]],
        samples_per_task=args.samples_per_task,
        selection_seed=args.selection_seed,
    )
    samples = shard_samples(
        samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out_dir / "configuration.json",
        {
            **config,
            "configuration_fingerprint": config_fingerprint,
            "argv": sys.argv,
        },
    )

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(str(args.model_dir))
    tokenizer, model, image_processor, _ = load_pretrained_model(
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
    codec, codec_metadata = load_codec(
        args.codec_path,
        device=model.device,
        dtype=torch.float16,
    )
    write_json_atomic(
        args.out_dir / "codec_metadata.json",
        codec_metadata,
    )

    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint_path = (
            args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        )
        if checkpoint_path.exists() and not args.overwrite:
            payload = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            if payload.get("configuration_fingerprint") == config_fingerprint:
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
            rows = run_sample(
                sample=sample,
                policies=policies,
                selection_manifest=selection_manifest,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                codec=codec,
                residual_tokens=[
                    int(value)
                    for value in config["residual_tokens"]
                ],
                include_full=bool(config["include_full"]),
                sampled_frames=args.sampled_frames,
                feature_pool_frames=args.feature_pool_frames,
                frame_budget=args.frame_budget,
                pool_grid=args.pool_grid,
                max_new_tokens=args.max_new_tokens,
                include_subtitle=args.include_subtitle,
            )
            for row in rows:
                print(
                    json.dumps(
                        {
                            "event": "policy_ok",
                            "sample": sample.sample_id,
                            "policy": row["policy"],
                            "correct": row["correct"],
                            "parsed": row["parsed"],
                            "state_bytes": row[
                                "selection_state_proxy_bytes"
                            ],
                            "reconstruction_error": row[
                                "selected_reconstruction_relative_error"
                            ],
                            "seconds": row["policy_seconds"],
                        }
                    ),
                    flush=True,
                )
            write_json_atomic(
                checkpoint_path,
                {
                    "configuration_fingerprint": config_fingerprint,
                    "rows": rows,
                },
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
