from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from feature_memory_codec import LowRankFeatureCodec, encode_feature_memory, load_codec
from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_reader_quotient_oracle import candidate_fisher_diagonal
from mvbench_onevision_utils import (
    build_prompt_batch,
    decode_feature_pool,
    direct_first_token_logits,
    encode_video_features,
    first_token_logits_from_features,
    load_onevision_model,
    preprocess_video,
)
from mvbench_reader_quotient_support_oracle import (
    candidate_token_ids,
    distribution_metrics,
    normalized_mixed_scores,
    relative_feature_error,
    sparse_reconstruction,
    support_overlap,
)
from mvbench_utils import load_mvbench_samples, parse_csv_list, shard_samples


DEFAULT_TASKS = (
    "fine_grained_pose",
    "object_interaction",
    "action_prediction",
    "egocentric_navigation",
    "moving_attribute",
)
DEFAULT_ALLOCATIONS = ((384, 4), (402, 3), (420, 2), (438, 1), (456, 0))


def parse_allocations(value: str) -> tuple[tuple[int, int], ...]:
    allocations = []
    for item in value.split(","):
        rank_text, support_text = item.strip().split(":", maxsplit=1)
        rank = int(rank_text)
        support = int(support_text)
        if rank <= 0 or support < 0:
            raise ValueError("allocation rank/support must be non-negative")
        allocations.append((rank, support))
    if not allocations or len(set(allocations)) != len(allocations):
        raise ValueError("allocations must be non-empty and unique")
    return tuple(allocations)


def allocation_id(rank: int, support: int) -> str:
    return f"r{rank}_s{support}"


def allocation_variants(
    allocations: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    variants = []
    for rank, support in allocations:
        identifier = allocation_id(rank, support)
        variants.append(f"euclidean_{identifier}")
        if support:
            variants.extend((f"fisher_{identifier}", f"mixed_{identifier}"))
    return tuple(variants)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument(
        "--allocations",
        default=",".join(f"{rank}:{support}" for rank, support in DEFAULT_ALLOCATIONS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def prefix_codec(codec: LowRankFeatureCodec, rank: int) -> LowRankFeatureCodec:
    if rank > codec.rank:
        raise ValueError(f"allocation rank {rank} exceeds codec rank {codec.rank}")
    return LowRankFeatureCodec(mean=codec.mean, basis=codec.basis[:, :rank])


def evaluate_sample(
    *,
    sample: object,
    processor: object,
    model: torch.nn.Module,
    max_codec: LowRankFeatureCodec,
    allocations: tuple[tuple[int, int], ...],
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    frames, pool_indices, selected_positions = decode_feature_pool(
        sample,
        sampled_frames=sampled_frames,
        feature_pool_frames=feature_pool_frames,
        frame_budget=frame_budget,
    )
    pixels = preprocess_video(processor, frames, device=device, dtype=dtype)
    vision_started = time.perf_counter()
    with torch.inference_mode():
        pool_features = encode_video_features(model, pixels)
    vision_seconds = time.perf_counter() - vision_started

    position_tensor = torch.tensor(
        selected_positions,
        device=pool_features.device,
        dtype=torch.long,
    )
    reference_features = pool_features.index_select(0, position_tensor).contiguous()
    selected_frames = frames[np.asarray(selected_positions, dtype=np.int64)]
    prompt_batch = build_prompt_batch(
        processor,
        sample,
        selected_frames,
        device=device,
        dtype=dtype,
    )
    token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
    with torch.inference_mode():
        direct_logits = direct_first_token_logits(
            model=model,
            prompt_batch=prompt_batch,
        ).detach()
        reference_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=reference_features,
        ).detach()
    injection_max_abs = float(
        torch.max(torch.abs(direct_logits.float() - reference_logits.float())).item()
    )
    if injection_max_abs > 1e-3:
        raise ValueError(f"manual feature injection changed logits by {injection_max_abs}")

    fisher_started = time.perf_counter()
    gradient_logits, fisher = candidate_fisher_diagonal(
        model=model,
        input_ids=prompt_batch["input_ids"],
        attention_mask=prompt_batch["attention_mask"],
        selected_features=reference_features,
        token_ids=token_ids,
    )
    fisher_seconds = time.perf_counter() - fisher_started
    instrumentation_max_abs = float(
        torch.max(torch.abs(reference_logits.float() - gradient_logits.float())).item()
    )
    if instrumentation_max_abs > 1e-3:
        raise ValueError(
            f"gradient instrumentation changed logits by {instrumentation_max_abs}"
        )

    reference_metrics = distribution_metrics(
        reference_logits,
        reference_logits,
        token_ids,
        int(sample.answer_index),
    )
    dense_state_bytes = int(pool_features.numel() * pool_features.element_size())
    rows = []
    for rank, support in allocations:
        codec = prefix_codec(max_codec, rank)
        latents = codec.encode(pool_features).to(torch.float16)
        base_pool = codec.decode(latents)
        residual_pool = pool_features.float() - base_pool
        selected_base = base_pool.index_select(0, position_tensor).float()
        selected_residual = residual_pool.index_select(0, position_tensor)
        selected_euclidean_scores = torch.sum(selected_residual.square(), dim=-1)
        identifier = allocation_id(rank, support)
        state = encode_feature_memory(
            pool_features,
            codec,
            residual_tokens_per_frame=support,
        )

        if support:
            euclidean_selected, euclidean_indices = sparse_reconstruction(
                selected_base,
                selected_residual,
                selected_euclidean_scores,
                support,
            )
            expected_indices = state.residual_indices.index_select(
                0,
                position_tensor,
            )
            if not torch.equal(euclidean_indices.long(), expected_indices.long()):
                raise ValueError(
                    f"Euclidean support differs from codec for {identifier}"
                )
            fisher_scores = torch.sum(fisher * selected_residual.square(), dim=-1)
            mixed_scores = normalized_mixed_scores(
                selected_euclidean_scores,
                fisher_scores,
            )
            fisher_selected, fisher_indices = sparse_reconstruction(
                selected_base,
                selected_residual,
                fisher_scores,
                support,
            )
            mixed_selected, mixed_indices = sparse_reconstruction(
                selected_base,
                selected_residual,
                mixed_scores,
                support,
            )
            reconstructions = (
                ("euclidean", euclidean_selected, 1.0),
                (
                    "fisher",
                    fisher_selected,
                    support_overlap(euclidean_indices, fisher_indices),
                ),
                (
                    "mixed",
                    mixed_selected,
                    support_overlap(euclidean_indices, mixed_indices),
                ),
            )
        else:
            reconstructions = (("euclidean", selected_base, 1.0),)

        for family, reconstruction, overlap in reconstructions:
            inference_started = time.perf_counter()
            with torch.inference_mode():
                approximate_logits = first_token_logits_from_features(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    features=reconstruction.to(dtype).contiguous(),
                ).detach()
            inference_seconds = time.perf_counter() - inference_started
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "variant": f"{family}_{identifier}",
                    "variant_family": family,
                    "allocation_id": identifier,
                    "rank": rank,
                    "residual_tokens_per_frame": support,
                    "answer_index": int(sample.answer_index),
                    "reference_candidate_prediction": int(
                        reference_metrics["candidate_prediction"]
                    ),
                    "reference_candidate_correct": int(
                        reference_metrics["candidate_correct"]
                    ),
                    "feature_relative_l2": relative_feature_error(
                        reference_features,
                        reconstruction,
                    ),
                    "support_overlap_with_euclidean": overlap,
                    "native_feature_state_bytes": int(state.stream_state_bytes),
                    "dense_native_feature_bytes": dense_state_bytes,
                    "state_compression_ratio": (
                        dense_state_bytes / state.stream_state_bytes
                    ),
                    "vision_seconds": vision_seconds,
                    "fisher_seconds": fisher_seconds,
                    "inference_seconds": inference_seconds,
                    "injection_max_abs": injection_max_abs,
                    "instrumentation_max_abs": instrumentation_max_abs,
                    **distribution_metrics(
                        reference_logits,
                        approximate_logits,
                        token_ids,
                        int(sample.answer_index),
                    ),
                }
            )

    metadata = {
        "sample_id": sample.sample_id,
        "pool_indices": pool_indices,
        "selected_positions": selected_positions,
        "pool_feature_shape": list(pool_features.shape),
        "reference_candidate_correct": int(reference_metrics["candidate_correct"]),
    }
    return rows, metadata


def main() -> int:
    args = parse_args()
    tasks = parse_csv_list(args.tasks)
    allocations = parse_allocations(args.allocations)
    variants = allocation_variants(allocations)
    all_samples = load_mvbench_samples(
        args.dataset_root,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        selection_seed=args.selection_seed,
    )
    samples = shard_samples(
        all_samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_index == 0:
        write_json_atomic(
            args.out_dir / "configuration.json",
            {
                "tasks": tasks,
                "samples_per_task": args.samples_per_task,
                "sample_ids": [sample.sample_id for sample in all_samples],
                "selection_seed": args.selection_seed,
                "sampled_frames": args.sampled_frames,
                "feature_pool_frames": args.feature_pool_frames,
                "frame_budget": args.frame_budget,
                "allocations": [list(item) for item in allocations],
                "variants": list(variants),
                "claim_tier": "onevision_equal_budget_rank_support_selection",
            },
        )
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    max_codec, codec_metadata = load_codec(
        args.codec_path,
        device=next(model.parameters()).device,
        dtype=torch.float16,
    )
    if max(rank for rank, _ in allocations) != max_codec.rank:
        raise ValueError("codec rank must equal the maximum allocation rank")
    if args.shard_index == 0:
        write_json_atomic(args.out_dir / "codec_metadata.json", codec_metadata)

    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint = args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        if checkpoint.exists() and not args.overwrite:
            print(json.dumps({"event": "resume", "sample": sample.sample_id}), flush=True)
            continue
        try:
            rows, metadata = evaluate_sample(
                sample=sample,
                processor=processor,
                model=model,
                max_codec=max_codec,
                allocations=allocations,
                sampled_frames=args.sampled_frames,
                feature_pool_frames=args.feature_pool_frames,
                frame_budget=args.frame_budget,
            )
            write_json_atomic(checkpoint, {"metadata": metadata, "rows": rows})
            print(
                json.dumps(
                    {
                        "event": "sample_ok",
                        "sample": sample.sample_id,
                        "position": position,
                        "total": len(samples),
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
            print(json.dumps({"event": "failure", **failure}), flush=True)
            if args.fail_fast:
                raise
    write_json_atomic(
        args.out_dir / f"failures_shard_{args.shard_index}.json",
        failures,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
