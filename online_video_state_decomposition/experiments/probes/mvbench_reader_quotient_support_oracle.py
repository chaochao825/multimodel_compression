from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from feature_memory_codec import encode_feature_memory, load_codec
from mvbench_llava_anchor import (
    install_transformers_forward_compatibility,
    install_visual_pooling,
    load_manifest_samples,
    write_json_atomic,
)
from mvbench_llava_feature_memory_anchor import (
    build_prompt_inputs,
    encode_native_feature_pool,
    selected_positions,
    use_cached_image_features,
)
from mvbench_utils import parse_csv_list, shard_samples, uniform_frame_indices, video_metadata


DEFAULT_TASKS = (
    "object_shuffle",
    "character_order",
    "moving_count",
    "episodic_reasoning",
    "action_localization",
)
VARIANTS = ("pca_only", "euclidean_s4", "fisher_s4", "mixed_s4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260824)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--policy", default="exact_recent")
    parser.add_argument("--residual-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def candidate_token_ids(tokenizer: object, candidate_count: int) -> list[int]:
    if not 2 <= candidate_count <= 26:
        raise ValueError("candidate count must be between 2 and 26")
    token_ids = []
    for index in range(candidate_count):
        label = chr(ord("A") + index)
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate label {label} is not one token: {encoded}")
        token_ids.append(int(encoded[0]))
    return token_ids


def sparse_reconstruction(
    base: torch.Tensor,
    residual: torch.Tensor,
    scores: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base.shape != residual.shape:
        raise ValueError("base and residual shapes differ")
    if scores.shape != base.shape[:2]:
        raise ValueError("scores must have shape [frames, tokens]")
    if not 0 < count <= base.shape[1]:
        raise ValueError("residual token count is outside the valid range")
    indices = torch.topk(scores, k=count, dim=1, largest=True, sorted=True).indices
    gather = indices.unsqueeze(-1).expand(-1, -1, base.shape[-1])
    values = torch.gather(residual, dim=1, index=gather)
    reconstruction = base.clone()
    reconstruction.scatter_add_(1, gather, values)
    return reconstruction, indices


def normalized_mixed_scores(
    euclidean_scores: torch.Tensor,
    fisher_scores: torch.Tensor,
) -> torch.Tensor:
    if euclidean_scores.shape != fisher_scores.shape:
        raise ValueError("score shapes differ")
    epsilon = torch.finfo(torch.float32).eps
    euclidean_scale = euclidean_scores.float().mean(dim=1, keepdim=True).clamp_min(
        epsilon
    )
    fisher_scale = fisher_scores.float().mean(dim=1, keepdim=True).clamp_min(epsilon)
    return 0.5 * (
        euclidean_scores.float() / euclidean_scale
        + fisher_scores.float() / fisher_scale
    )


def distribution_metrics(
    reference_logits: torch.Tensor,
    approximate_logits: torch.Tensor,
    token_ids: list[int],
    answer_index: int,
) -> dict[str, float | int]:
    reference = reference_logits.float().flatten()
    approximate = approximate_logits.float().flatten()
    if reference.shape != approximate.shape:
        raise ValueError("reference and approximate logits have different shapes")
    indices = torch.tensor(token_ids, device=reference.device, dtype=torch.long)
    reference_vocab_logp = torch.log_softmax(reference, dim=0)
    approximate_vocab_logp = torch.log_softmax(approximate, dim=0)
    reference_vocab_p = reference_vocab_logp.exp()
    vocabulary_kl = torch.sum(
        reference_vocab_p * (reference_vocab_logp - approximate_vocab_logp)
    )
    reference_candidate_logp = torch.log_softmax(
        reference.index_select(0, indices), dim=0
    )
    approximate_candidate_logp = torch.log_softmax(
        approximate.index_select(0, indices), dim=0
    )
    reference_candidate_p = reference_candidate_logp.exp()
    candidate_kl = torch.sum(
        reference_candidate_p
        * (reference_candidate_logp - approximate_candidate_logp)
    )
    reference_prediction = int(torch.argmax(reference_candidate_logp).item())
    approximate_prediction = int(torch.argmax(approximate_candidate_logp).item())
    return {
        "candidate_kl": float(candidate_kl.item()),
        "vocabulary_kl": float(vocabulary_kl.item()),
        "candidate_top1_match": int(reference_prediction == approximate_prediction),
        "candidate_prediction": approximate_prediction,
        "candidate_correct": int(approximate_prediction == answer_index),
        "answer_logprob": float(approximate_candidate_logp[answer_index].item()),
        "answer_logprob_delta": float(
            (approximate_candidate_logp[answer_index]
            - reference_candidate_logp[answer_index]).item()
        ),
    }


def model_first_token_logits(
    *,
    sample: object,
    selected_features: torch.Tensor,
    selected_image_sizes: list[tuple[int, int]],
    tokenizer: object,
    model: torch.nn.Module,
    include_grad: bool,
) -> torch.Tensor:
    input_ids, attention_mask = build_prompt_inputs(
        sample=sample,
        image_count=selected_features.shape[0],
        tokenizer=tokenizer,
        model=model,
        include_subtitle=False,
    )
    dummy_images = torch.empty(
        (selected_features.shape[0], 3, 1, 1),
        device=model.device,
        dtype=torch.float16,
    )
    context = torch.enable_grad() if include_grad else torch.inference_mode()
    with use_cached_image_features(model, selected_features):
        with context:
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                images=dummy_images,
                image_sizes=selected_image_sizes,
                use_cache=False,
                return_dict=True,
            )
    return outputs.logits[0, -1]


def candidate_fisher_diagonal(
    *,
    sample: object,
    selected_features: torch.Tensor,
    selected_image_sizes: list[tuple[int, int]],
    tokenizer: object,
    model: torch.nn.Module,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    probe = selected_features.detach().clone().requires_grad_(True)
    logits = model_first_token_logits(
        sample=sample,
        selected_features=probe,
        selected_image_sizes=selected_image_sizes,
        tokenizer=tokenizer,
        model=model,
        include_grad=True,
    )
    indices = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    candidate_logp = torch.log_softmax(logits.float().index_select(0, indices), dim=0)
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


def relative_feature_error(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    numerator = torch.linalg.vector_norm(reference.float() - reconstruction.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return float((numerator / denominator).item())


def support_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("support shapes differ")
    overlaps = []
    for left_row, right_row in zip(left.tolist(), right.tolist(), strict=True):
        overlaps.append(len(set(left_row) & set(right_row)) / len(left_row))
    return float(sum(overlaps) / len(overlaps))


def evaluate_sample(
    *,
    sample: object,
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
    codec: object,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
    pool_grid: int,
    policy: str,
    residual_tokens: int,
) -> list[dict[str, object]]:
    total_frames, _ = video_metadata(sample.video_path)
    sampled_indices = uniform_frame_indices(total_frames, sampled_frames)
    pool_indices = sampled_indices[-feature_pool_frames:]
    selected_indices = [
        int(index)
        for index in selection_manifest["samples"][sample.sample_id]["policies"][policy]
    ]
    if len(selected_indices) != frame_budget:
        raise ValueError("selection manifest frame budget mismatch")
    positions = selected_positions(pool_indices, selected_indices)
    pool_features, image_sizes, _ = encode_native_feature_pool(
        sample=sample,
        frame_indices=pool_indices,
        model=model,
        image_processor=image_processor,
    )
    if pool_features.shape[1] != pool_grid**2:
        raise ValueError("native feature grid does not match pool grid")
    position_tensor = torch.tensor(positions, device=pool_features.device, dtype=torch.long)
    reference_features = pool_features.index_select(0, position_tensor).contiguous()
    selected_sizes = [image_sizes[position] for position in positions]
    token_ids = candidate_token_ids(tokenizer, len(sample.candidates))

    with torch.inference_mode():
        reference_logits = model_first_token_logits(
            sample=sample,
            selected_features=reference_features,
            selected_image_sizes=selected_sizes,
            tokenizer=tokenizer,
            model=model,
            include_grad=False,
        ).detach()
    started = time.perf_counter()
    gradient_logits, fisher = candidate_fisher_diagonal(
        sample=sample,
        selected_features=reference_features,
        selected_image_sizes=selected_sizes,
        tokenizer=tokenizer,
        model=model,
        token_ids=token_ids,
    )
    fisher_seconds = time.perf_counter() - started
    instrumentation_max_abs = float(
        torch.max(torch.abs(reference_logits.float() - gradient_logits.float())).item()
    )
    if instrumentation_max_abs > 1e-3:
        raise ValueError(
            f"gradient instrumentation changed logits by {instrumentation_max_abs}"
        )

    latents = codec.encode(pool_features).to(torch.float16)
    base_pool = codec.decode(latents).to(pool_features.dtype)
    residual_pool = pool_features.float() - base_pool.float()
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
    support_overlaps = {
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
    rows = []
    for variant in VARIANTS:
        reconstruction = reconstructions[variant].to(pool_features.dtype).contiguous()
        started = time.perf_counter()
        with torch.inference_mode():
            approximate_logits = model_first_token_logits(
                sample=sample,
                selected_features=reconstruction,
                selected_image_sizes=selected_sizes,
                tokenizer=tokenizer,
                model=model,
                include_grad=False,
            ).detach()
        inference_seconds = time.perf_counter() - started
        rows.append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "variant": variant,
                "answer_index": int(sample.answer_index),
                "feature_relative_l2": relative_feature_error(
                    reference_features, reconstruction
                ),
                "support_overlap_with_euclidean": support_overlaps[variant],
                "native_feature_state_bytes": state_bytes[variant],
                "residual_tokens_per_frame": residual_tokens,
                "fisher_seconds": fisher_seconds,
                "inference_seconds": inference_seconds,
                "instrumentation_max_abs": instrumentation_max_abs,
                **distribution_metrics(
                    reference_logits,
                    approximate_logits,
                    token_ids,
                    int(sample.answer_index),
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.llava_source))
    tasks = parse_csv_list(args.tasks)
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    samples = load_manifest_samples(
        args.dataset_root,
        selection_manifest,
        tasks=tasks,
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
            "tasks": tasks,
            "samples_per_task": args.samples_per_task,
            "selection_seed": args.selection_seed,
            "sampled_frames": args.sampled_frames,
            "feature_pool_frames": args.feature_pool_frames,
            "frame_budget": args.frame_budget,
            "pool_grid": args.pool_grid,
            "policy": args.policy,
            "residual_tokens": args.residual_tokens,
            "variants": list(VARIANTS),
            "claim_tier": "transductive_native_fisher_support_oracle",
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
    model.requires_grad_(False)
    codec, codec_metadata = load_codec(
        args.codec_path,
        device=model.device,
        dtype=torch.float16,
    )
    write_json_atomic(args.out_dir / "codec_metadata.json", codec_metadata)

    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint = args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        if checkpoint.exists() and not args.overwrite:
            print(json.dumps({"event": "resume", "sample": sample.sample_id}), flush=True)
            continue
        try:
            rows = evaluate_sample(
                sample=sample,
                selection_manifest=selection_manifest,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                codec=codec,
                sampled_frames=args.sampled_frames,
                feature_pool_frames=args.feature_pool_frames,
                frame_budget=args.frame_budget,
                pool_grid=args.pool_grid,
                policy=args.policy,
                residual_tokens=args.residual_tokens,
            )
            write_json_atomic(checkpoint, {"rows": rows})
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
        args.out_dir / f"failures_shard_{args.shard_index}.json", failures
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
