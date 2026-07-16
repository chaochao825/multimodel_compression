from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import types
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

from mvbench_llava_anchor import (
    install_transformers_forward_compatibility,
    install_visual_pooling,
    load_manifest_samples,
    write_json_atomic,
)
from mvbench_utils import (
    choice_prompt,
    decode_video_frames,
    parse_choice_output,
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
DEFAULT_POLICIES = (
    "exact_recent",
    "recent_pool_query_topk",
    "recent_pool_query_mmr",
    "learned_recent_query_topk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260717)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--include-subtitle", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "llava_source": str(args.llava_source.resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": hashlib.sha256(
            args.selection_manifest.read_bytes()
        ).hexdigest(),
        "tasks": parse_csv_list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "selection_seed": args.selection_seed,
        "sampled_frames": args.sampled_frames,
        "feature_pool_frames": args.feature_pool_frames,
        "frame_budget": args.frame_budget,
        "pool_grid": args.pool_grid,
        "policies": parse_csv_list(args.policies),
        "max_new_tokens": args.max_new_tokens,
        "include_subtitle": args.include_subtitle,
        "native_feature_memory": True,
        "raw_frame_replay_at_read": False,
        "matched_provisioned_state": True,
    }


def fingerprint(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_positions(
    pool_indices: list[int],
    selected_indices: list[int],
) -> list[int]:
    position_by_index = {
        frame_index: position
        for position, frame_index in enumerate(pool_indices)
    }
    missing = [
        frame_index
        for frame_index in selected_indices
        if frame_index not in position_by_index
    ]
    if missing:
        raise ValueError(
            f"selected frames are outside the native feature pool: {missing}"
        )
    return [position_by_index[index] for index in selected_indices]


def feature_cache_bytes(features: torch.Tensor) -> int:
    return features.numel() * features.element_size()


@contextmanager
def use_cached_image_features(
    model: torch.nn.Module,
    features: torch.Tensor,
):
    original_encode = model.encode_images

    def cached_encode(
        _self: object,
        _images: torch.Tensor,
    ) -> torch.Tensor:
        return features

    model.encode_images = types.MethodType(cached_encode, model)
    try:
        yield
    finally:
        model.encode_images = original_encode


def build_prompt_inputs(
    *,
    sample: object,
    image_count: int,
    tokenizer: object,
    model: torch.nn.Module,
    include_subtitle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from llava import conversation as conversation_lib
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import tokenizer_image_token

    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
        )
    question = choice_prompt(
        sample,
        include_subtitle=include_subtitle,
    )
    query = "\n".join([image_token] * image_count) + "\n" + question
    conversation = conversation_lib.conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], query)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def encode_native_feature_pool(
    *,
    sample: object,
    frame_indices: list[int],
    model: torch.nn.Module,
    image_processor: object,
) -> tuple[torch.Tensor, list[tuple[int, int]], dict[str, float]]:
    from llava.mm_utils import process_images

    total_frames, _ = video_metadata(sample.video_path)
    write_start = time.perf_counter()
    decode_start = time.perf_counter()
    frames, _, decoded_total = decode_video_frames(
        sample.video_path,
        frame_indices,
    )
    decode_seconds = time.perf_counter() - decode_start
    if decoded_total != total_frames:
        raise RuntimeError("video frame count changed during native write")
    preprocess_start = time.perf_counter()
    images = [Image.fromarray(frame) for frame in frames]
    image_sizes = [image.size for image in images]
    image_tensor = process_images(
        images,
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)
    preprocess_seconds = time.perf_counter() - preprocess_start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_start = time.perf_counter()
    with torch.inference_mode():
        features = model.encode_images(image_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_start
    write_seconds = time.perf_counter() - write_start
    if features.ndim != 3 or features.shape[0] != len(frame_indices):
        raise RuntimeError(
            f"unexpected native feature shape: {tuple(features.shape)}"
        )
    return (
        features.detach(),
        image_sizes,
        {
            "write_seconds": write_seconds,
            "decode_seconds": decode_seconds,
            "preprocess_seconds": preprocess_seconds,
            "vision_encode_seconds": encode_seconds,
        },
    )


def read_from_native_feature_pool(
    *,
    sample: object,
    policy: str,
    selected_frame_indices: list[int],
    selected_features: torch.Tensor,
    selected_image_sizes: list[tuple[int, int]],
    tokenizer: object,
    model: torch.nn.Module,
    max_new_tokens: int,
    include_subtitle: bool,
) -> dict[str, object]:
    input_ids, attention_mask = build_prompt_inputs(
        sample=sample,
        image_count=len(selected_frame_indices),
        tokenizer=tokenizer,
        model=model,
        include_subtitle=include_subtitle,
    )
    dummy_images = torch.empty(
        (len(selected_frame_indices), 3, 1, 1),
        device=model.device,
        dtype=torch.float16,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    read_start = time.perf_counter()
    with use_cached_image_features(model, selected_features):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=dummy_images,
                image_sizes=selected_image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    read_seconds = time.perf_counter() - read_start
    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()
    predicted = parse_choice_output(output, sample.candidates)
    return {
        "policy": policy,
        "raw_output": output,
        "predicted_index": predicted,
        "prediction": (
            sample.candidates[predicted] if predicted is not None else ""
        ),
        "parsed": int(predicted is not None),
        "correct": int(predicted == int(sample.answer_index)),
        "inference_seconds": read_seconds,
    }


def run_sample(
    *,
    sample: object,
    policies: list[str],
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
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
    if any(
        len(indices) != frame_budget
        for indices in selected_by_policy.values()
    ):
        raise ValueError("selection manifest frame budget mismatch")
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
    cache_bytes = feature_cache_bytes(features)
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
    if not getattr(model, "_native_feature_memory_warmed_up", False):
        warmup_policy = policies[0]
        warmup_positions = positions_by_policy[warmup_policy]
        warmup_tensor = torch.tensor(
            warmup_positions,
            device=features.device,
            dtype=torch.long,
        )
        read_from_native_feature_pool(
            sample=sample,
            policy=warmup_policy,
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
        model._native_feature_memory_warmed_up = True
    rows = []
    for policy in policies:
        positions = positions_by_policy[policy]
        position_tensor = torch.tensor(
            positions,
            device=features.device,
            dtype=torch.long,
        )
        selected_features = features.index_select(
            0,
            position_tensor,
        ).contiguous()
        selected_sizes = [image_sizes[position] for position in positions]
        result = read_from_native_feature_pool(
            sample=sample,
            policy=policy,
            selected_frame_indices=selected_by_policy[policy],
            selected_features=selected_features,
            selected_image_sizes=selected_sizes,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            include_subtitle=include_subtitle,
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
                "policy_seconds": (
                    write_timing["write_seconds"]
                    + float(result["inference_seconds"])
                ),
                "visual_tokens": (
                    frame_budget * pool_grid * pool_grid
                ),
                "selection_state_proxy_bytes": (
                    cache_bytes + provisioned_selector_bytes
                ),
                "actual_selector_state_bytes": int(
                    policy_accounting.get("total_state_bytes", 0)
                ),
                "provisioned_selector_state_bytes": (
                    provisioned_selector_bytes
                ),
                "native_feature_cache_bytes": cache_bytes,
                "native_feature_pool_frames": feature_pool_frames,
                "native_tokens_per_frame": pool_grid * pool_grid,
                "native_feature_hidden_size": int(features.shape[-1]),
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
                            "seconds": row["policy_seconds"],
                            "write_seconds": row[
                                "feature_cache_write_seconds"
                            ],
                            "read_seconds": row["inference_seconds"],
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
