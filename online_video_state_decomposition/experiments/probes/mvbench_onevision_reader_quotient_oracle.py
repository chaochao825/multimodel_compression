from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from feature_memory_codec import encode_feature_memory, load_codec
from mvbench_llava_anchor import write_json_atomic
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
VARIANTS = ("pca_only", "euclidean_s4", "fisher_s4", "mixed_s4")


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
    parser.add_argument("--residual-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def candidate_fisher_diagonal(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_features: torch.Tensor,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    probe = selected_features.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = first_token_logits_from_features(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=probe,
        )
        indices = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
        candidate_logp = torch.log_softmax(
            logits.float().index_select(0, indices),
            dim=0,
        )
        probabilities = candidate_logp.detach().exp()
        fisher = torch.zeros_like(probe, dtype=torch.float32)
        for index in range(len(token_ids)):
            gradient = torch.autograd.grad(
                candidate_logp[index],
                probe,
                retain_graph=index + 1 < len(token_ids),
            )[0]
            fisher.add_(probabilities[index] * gradient.float().square())
    if not torch.isfinite(fisher).all():
        raise ValueError("candidate Fisher contains non-finite values")
    return logits.detach(), fisher.detach()


def evaluate_sample(
    *,
    sample: object,
    processor: object,
    model: torch.nn.Module,
    codec: object,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
    residual_tokens: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    frames, pool_indices, selected_positions = decode_feature_pool(
        sample,
        sampled_frames=sampled_frames,
        feature_pool_frames=feature_pool_frames,
        frame_budget=frame_budget,
    )
    pixels = preprocess_video(
        processor,
        frames,
        device=device,
        dtype=dtype,
    )
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

    latents = codec.encode(pool_features).to(torch.float16)
    base_pool = codec.decode(latents)
    residual_pool = pool_features.float() - base_pool
    euclidean_scores_pool = torch.sum(residual_pool.square(), dim=-1)
    euclidean_pool, euclidean_indices_pool = sparse_reconstruction(
        base_pool.float(),
        residual_pool,
        euclidean_scores_pool,
        residual_tokens,
    )
    codec_state = encode_feature_memory(
        pool_features,
        codec,
        residual_tokens_per_frame=residual_tokens,
    )
    pca_state = encode_feature_memory(
        pool_features,
        codec,
        residual_tokens_per_frame=0,
    )
    if not torch.equal(codec_state.residual_indices.long(), euclidean_indices_pool.long()):
        raise ValueError("Euclidean support does not match the existing codec")

    selected_base = base_pool.index_select(0, position_tensor).float()
    selected_residual = residual_pool.index_select(0, position_tensor)
    selected_euclidean_scores = euclidean_scores_pool.index_select(0, position_tensor)
    fisher_scores = torch.sum(fisher * selected_residual.square(), dim=-1)
    mixed_scores = normalized_mixed_scores(selected_euclidean_scores, fisher_scores)
    fisher_selected, fisher_indices = sparse_reconstruction(
        selected_base,
        selected_residual,
        fisher_scores,
        residual_tokens,
    )
    mixed_selected, mixed_indices = sparse_reconstruction(
        selected_base,
        selected_residual,
        mixed_scores,
        residual_tokens,
    )
    selected_euclidean = euclidean_pool.index_select(0, position_tensor)
    euclidean_indices = euclidean_indices_pool.index_select(0, position_tensor)
    reconstructions = {
        "pca_only": selected_base,
        "euclidean_s4": selected_euclidean,
        "fisher_s4": fisher_selected,
        "mixed_s4": mixed_selected,
    }
    overlaps = {
        "pca_only": 0.0,
        "euclidean_s4": 1.0,
        "fisher_s4": support_overlap(euclidean_indices, fisher_indices),
        "mixed_s4": support_overlap(euclidean_indices, mixed_indices),
    }
    state_bytes = {
        "pca_only": int(pca_state.stream_state_bytes),
        "euclidean_s4": int(codec_state.stream_state_bytes),
        "fisher_s4": int(codec_state.stream_state_bytes),
        "mixed_s4": int(codec_state.stream_state_bytes),
    }
    dense_state_bytes = int(pool_features.numel() * pool_features.element_size())
    reference_metrics = distribution_metrics(
        reference_logits,
        reference_logits,
        token_ids,
        int(sample.answer_index),
    )
    rows = []
    for variant in VARIANTS:
        reconstruction = reconstructions[variant].to(dtype).contiguous()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            approximate_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reconstruction,
            ).detach()
        inference_seconds = time.perf_counter() - inference_started
        rows.append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "variant": variant,
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
                "support_overlap_with_euclidean": overlaps[variant],
                "native_feature_state_bytes": state_bytes[variant],
                "dense_native_feature_bytes": dense_state_bytes,
                "state_compression_ratio": dense_state_bytes / state_bytes[variant],
                "residual_tokens_per_frame": residual_tokens,
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
        "reference_candidate_correct": int(
            reference_metrics["candidate_correct"]
        ),
    }
    return rows, metadata


def main() -> int:
    args = parse_args()
    tasks = parse_csv_list(args.tasks)
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
                "residual_tokens": args.residual_tokens,
                "variants": list(VARIANTS),
                "claim_tier": "onevision_transductive_native_fisher_capacity_replication",
            },
        )
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    codec, codec_metadata = load_codec(
        args.codec_path,
        device=next(model.parameters()).device,
        dtype=torch.float16,
    )
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
                codec=codec,
                sampled_frames=args.sampled_frames,
                feature_pool_frames=args.feature_pool_frames,
                frame_budget=args.frame_budget,
                residual_tokens=args.residual_tokens,
            )
            write_json_atomic(checkpoint, {"metadata": metadata, "rows": rows})
            print(
                json.dumps(
                    {
                        "event": "sample_ok",
                        "sample": sample.sample_id,
                        "position": position,
                        "total": len(samples),
                        "candidate_kl": {
                            row["variant"]: row["candidate_kl"] for row in rows
                        },
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
