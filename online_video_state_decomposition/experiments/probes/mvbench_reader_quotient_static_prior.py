from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
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
    encode_native_feature_pool,
    selected_positions,
)
from mvbench_reader_quotient_support_oracle import (
    candidate_fisher_diagonal,
    candidate_token_ids,
    distribution_metrics,
    model_first_token_logits,
    normalized_mixed_scores,
    relative_feature_error,
    sparse_reconstruction,
    support_overlap,
)
from mvbench_utils import parse_csv_list, uniform_frame_indices, video_metadata


EVALUATION_VARIANTS = (
    "euclidean_s4",
    "position_s4",
    "channel_s4",
    "separable_s4",
    "static_fisher_s4",
    "mixed_static_s4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--manifest-samples-per-task", type=int, required=True)
    parser.add_argument("--take-per-task", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--exclude-sample-ids", type=Path)
    parser.add_argument("--prior-path", type=Path)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--policy", default="exact_recent")
    parser.add_argument("--residual-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_samples(
    *,
    dataset_root: Path,
    selection_manifest: dict[str, object],
    tasks: list[str],
    manifest_samples_per_task: int,
    take_per_task: int,
    selection_seed: int,
    excluded: set[str],
) -> list[object]:
    candidates = load_manifest_samples(
        dataset_root,
        selection_manifest,
        tasks=tasks,
        samples_per_task=manifest_samples_per_task,
        selection_seed=selection_seed,
    )
    grouped: dict[str, list[object]] = defaultdict(list)
    for sample in candidates:
        if sample.sample_id not in excluded:
            grouped[sample.task].append(sample)
    selected = []
    for task in tasks:
        available = grouped[task]
        if len(available) < take_per_task:
            raise ValueError(
                f"task {task} has {len(available)} eligible samples; "
                f"requires {take_per_task}"
            )
        selected.extend(available[:take_per_task])
    return selected


def prepare_sample(
    *,
    sample: object,
    selection_manifest: dict[str, object],
    model: torch.nn.Module,
    image_processor: object,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
    pool_grid: int,
    policy: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[int, int]],
]:
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
    return pool_features, reference_features, position_tensor, selected_sizes


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calibrate(
    *,
    samples: list[object],
    args: argparse.Namespace,
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
) -> None:
    fisher_sum = None
    frame_count = 0
    timing_rows = []
    for position, sample in enumerate(samples, start=1):
        _, reference_features, _, selected_sizes = prepare_sample(
            sample=sample,
            selection_manifest=selection_manifest,
            model=model,
            image_processor=image_processor,
            sampled_frames=args.sampled_frames,
            feature_pool_frames=args.feature_pool_frames,
            frame_budget=args.frame_budget,
            pool_grid=args.pool_grid,
            policy=args.policy,
        )
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
        synchronize()
        started = time.perf_counter()
        gradient_logits, fisher = candidate_fisher_diagonal(
            sample=sample,
            selected_features=reference_features,
            selected_image_sizes=selected_sizes,
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
        )
        synchronize()
        seconds = time.perf_counter() - started
        max_abs = float(
            torch.max(torch.abs(reference_logits.float() - gradient_logits.float())).item()
        )
        if max_abs > 1e-3:
            raise ValueError(f"gradient instrumentation changed logits by {max_abs}")
        sample_sum = fisher.sum(dim=0).double().cpu()
        fisher_sum = sample_sum if fisher_sum is None else fisher_sum + sample_sum
        frame_count += int(fisher.shape[0])
        timing_rows.append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "fisher_seconds": seconds,
                "instrumentation_max_abs": max_abs,
            }
        )
        print(
            json.dumps(
                {
                    "event": "calibration_ok",
                    "sample": sample.sample_id,
                    "position": position,
                    "total": len(samples),
                }
            ),
            flush=True,
        )
    if fisher_sum is None or frame_count == 0:
        raise ValueError("calibration produced no Fisher observations")
    fisher_mean = (fisher_sum / frame_count).float()
    if not torch.isfinite(fisher_mean).all() or not torch.any(fisher_mean > 0):
        raise ValueError("calibration Fisher prior is invalid")
    prior_path = args.out_dir / "static_fisher_prior.pt"
    torch.save(
        {
            "fisher_mean": fisher_mean,
            "sample_ids": [sample.sample_id for sample in samples],
            "tasks": parse_csv_list(args.tasks),
            "frame_count": frame_count,
            "token_grid": args.pool_grid,
            "hidden_size": int(fisher_mean.shape[1]),
        },
        prior_path,
    )
    write_json_atomic(
        args.out_dir / "calibration_summary.json",
        {
            "samples": len(samples),
            "sample_ids": [sample.sample_id for sample in samples],
            "frame_count": frame_count,
            "prior_shape": list(fisher_mean.shape),
            "prior_parameter_bytes_fp32": fisher_mean.numel() * fisher_mean.element_size(),
            "mean_fisher_seconds": sum(row["fisher_seconds"] for row in timing_rows)
            / len(timing_rows),
            "max_instrumentation_abs": max(
                row["instrumentation_max_abs"] for row in timing_rows
            ),
        },
    )


def evaluate_one(
    *,
    sample: object,
    args: argparse.Namespace,
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
    codec: object,
    prior: torch.Tensor,
) -> list[dict[str, object]]:
    pool_features, reference_features, position_tensor, selected_sizes = prepare_sample(
        sample=sample,
        selection_manifest=selection_manifest,
        model=model,
        image_processor=image_processor,
        sampled_frames=args.sampled_frames,
        feature_pool_frames=args.feature_pool_frames,
        frame_budget=args.frame_budget,
        pool_grid=args.pool_grid,
        policy=args.policy,
    )
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

    latents = codec.encode(pool_features).to(torch.float16)
    base_pool = codec.decode(latents).to(pool_features.dtype).float()
    residual_pool = pool_features.float() - base_pool
    euclidean_scores_pool = torch.sum(residual_pool.square(), dim=-1)
    euclidean_pool, euclidean_indices_pool = sparse_reconstruction(
        base_pool,
        residual_pool,
        euclidean_scores_pool,
        args.residual_tokens,
    )
    codec_state = encode_feature_memory(
        pool_features,
        codec,
        residual_tokens_per_frame=args.residual_tokens,
    )
    if not torch.equal(codec_state.residual_indices.long(), euclidean_indices_pool.long()):
        raise ValueError("Euclidean support does not match the existing codec")

    selected_base = base_pool.index_select(0, position_tensor)
    selected_residual = residual_pool.index_select(0, position_tensor)
    selected_euclidean_scores = euclidean_scores_pool.index_select(0, position_tensor)
    if prior.shape != selected_residual.shape[1:]:
        raise ValueError(
            f"prior shape {tuple(prior.shape)} does not match "
            f"token/channel shape {tuple(selected_residual.shape[1:])}"
        )
    position_weight = prior.mean(dim=1)
    channel_weight = prior.mean(dim=0)
    separable_weight = (
        position_weight[:, None]
        * channel_weight[None, :]
        / prior.mean().clamp_min(torch.finfo(torch.float32).tiny)
    )
    squared_residual = selected_residual.square()
    score_by_variant = {
        "position_s4": selected_euclidean_scores * position_weight[None, :],
        "channel_s4": torch.sum(squared_residual * channel_weight[None, None, :], dim=-1),
        "separable_s4": torch.sum(squared_residual * separable_weight[None, :, :], dim=-1),
        "static_fisher_s4": torch.sum(squared_residual * prior[None, :, :], dim=-1),
    }
    score_by_variant["mixed_static_s4"] = normalized_mixed_scores(
        selected_euclidean_scores,
        score_by_variant["static_fisher_s4"],
    )
    reconstructions = {
        "euclidean_s4": euclidean_pool.index_select(0, position_tensor),
    }
    supports = {
        "euclidean_s4": euclidean_indices_pool.index_select(0, position_tensor),
    }
    synchronize()
    scorer_started = time.perf_counter()
    for variant, scores in score_by_variant.items():
        reconstruction, indices = sparse_reconstruction(
            selected_base,
            selected_residual,
            scores,
            args.residual_tokens,
        )
        reconstructions[variant] = reconstruction
        supports[variant] = indices
    synchronize()
    scorer_seconds = time.perf_counter() - scorer_started

    rows = []
    for variant in EVALUATION_VARIANTS:
        reconstruction = reconstructions[variant].to(pool_features.dtype).contiguous()
        synchronize()
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
        synchronize()
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
                "support_overlap_with_euclidean": support_overlap(
                    supports["euclidean_s4"], supports[variant]
                ),
                "native_feature_state_bytes": int(codec_state.stream_state_bytes),
                "prior_parameter_bytes_fp32": prior.numel() * prior.element_size(),
                "scorer_seconds_all_candidates": scorer_seconds,
                "inference_seconds": inference_seconds,
                "instrumentation_max_abs": 0.0,
                **distribution_metrics(
                    reference_logits,
                    approximate_logits,
                    token_ids,
                    int(sample.answer_index),
                ),
            }
        )
    return rows


def evaluate(
    *,
    samples: list[object],
    args: argparse.Namespace,
    selection_manifest: dict[str, object],
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
    codec: object,
) -> None:
    if args.prior_path is None:
        raise ValueError("--prior-path is required in evaluate mode")
    payload = torch.load(args.prior_path, map_location="cpu", weights_only=True)
    calibration_ids = {str(sample_id) for sample_id in payload["sample_ids"]}
    evaluation_ids = {sample.sample_id for sample in samples}
    overlap = sorted(calibration_ids & evaluation_ids)
    if overlap:
        raise ValueError(f"calibration/evaluation sample overlap: {overlap}")
    prior = payload["fisher_mean"].to(model.device, dtype=torch.float32)
    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint = args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        if checkpoint.exists() and not args.overwrite:
            print(json.dumps({"event": "resume", "sample": sample.sample_id}), flush=True)
            continue
        try:
            rows = evaluate_one(
                sample=sample,
                args=args,
                selection_manifest=selection_manifest,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                codec=codec,
                prior=prior,
            )
            write_json_atomic(checkpoint, {"rows": rows})
            print(
                json.dumps(
                    {
                        "event": "evaluation_ok",
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
    write_json_atomic(args.out_dir / "failures.json", failures)
    if failures:
        raise RuntimeError(f"evaluation failed for {len(failures)} samples")


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.llava_source))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_csv_list(args.tasks)
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    excluded = set()
    if args.exclude_sample_ids is not None:
        excluded = {
            str(sample_id)
            for sample_id in json.loads(args.exclude_sample_ids.read_text(encoding="utf-8"))
        }
    samples = choose_samples(
        dataset_root=args.dataset_root,
        selection_manifest=selection_manifest,
        tasks=tasks,
        manifest_samples_per_task=args.manifest_samples_per_task,
        take_per_task=args.take_per_task,
        selection_seed=args.selection_seed,
        excluded=excluded,
    )
    write_json_atomic(
        args.out_dir / "configuration.json",
        {
            "mode": args.mode,
            "tasks": tasks,
            "manifest_samples_per_task": args.manifest_samples_per_task,
            "take_per_task": args.take_per_task,
            "selection_seed": args.selection_seed,
            "sample_ids": [sample.sample_id for sample in samples],
            "excluded_sample_count": len(excluded),
            "sampled_frames": args.sampled_frames,
            "feature_pool_frames": args.feature_pool_frames,
            "frame_budget": args.frame_budget,
            "pool_grid": args.pool_grid,
            "policy": args.policy,
            "residual_tokens": args.residual_tokens,
            "claim_tier": "calibration_only_query_agnostic_static_fisher_prior",
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
    if args.mode == "calibrate":
        calibrate(
            samples=samples,
            args=args,
            selection_manifest=selection_manifest,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
        )
    else:
        evaluate(
            samples=samples,
            args=args,
            selection_manifest=selection_manifest,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            codec=codec,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
