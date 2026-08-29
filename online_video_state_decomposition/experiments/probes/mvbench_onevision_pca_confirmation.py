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
    relative_feature_error,
)
from mvbench_utils import (
    load_mvbench_samples,
    normalize_text,
    parse_csv_list,
    shard_samples,
)


DEFAULT_TASKS = (
    "fine_grained_action",
    "action_antonym",
    "unexpected_action",
    "counterfactual_inference",
    "action_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20260829)
    parser.add_argument("--sampled-frames", type=int, default=32)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def select_eligible_samples(
    dataset_root: Path,
    *,
    tasks: list[str],
    samples_per_task: int,
    selection_seed: int,
) -> tuple[list[object], dict[str, int]]:
    raw_samples = load_mvbench_samples(
        dataset_root,
        tasks=tasks,
        samples_per_task=0,
        selection_seed=selection_seed,
    )
    rng = np.random.default_rng(selection_seed)
    selected = []
    excluded_by_task = {}
    for task in tasks:
        task_samples = [sample for sample in raw_samples if sample.task == task]
        eligible = [
            sample
            for sample in task_samples
            if 2 <= len(sample.candidates) <= 26
            and normalize_text(sample.answer)
            in {normalize_text(candidate) for candidate in sample.candidates}
        ]
        excluded_by_task[task] = len(task_samples) - len(eligible)
        if len(eligible) < samples_per_task:
            raise ValueError(
                f"task {task} has only {len(eligible)} eligible samples"
            )
        positions = sorted(
            int(value)
            for value in rng.choice(
                len(eligible),
                size=samples_per_task,
                replace=False,
            )
        )
        selected.extend(eligible[position] for position in positions)
    return selected, excluded_by_task


def evaluate_sample(
    *,
    sample: object,
    processor: object,
    model: torch.nn.Module,
    codec: object,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[dict[str, object], dict[str, object]]:
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

    positions = torch.tensor(
        selected_positions,
        device=pool_features.device,
        dtype=torch.long,
    )
    reference_features = pool_features.index_select(0, positions).contiguous()
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

    reference_metrics = distribution_metrics(
        reference_logits,
        reference_logits,
        token_ids,
        int(sample.answer_index),
    )
    latents = codec.encode(pool_features).to(torch.float16)
    reconstructed_pool = codec.decode(latents)
    reconstructed_features = reconstructed_pool.index_select(0, positions).contiguous()
    state = encode_feature_memory(
        pool_features,
        codec,
        residual_tokens_per_frame=0,
    )
    inference_started = time.perf_counter()
    with torch.inference_mode():
        approximate_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=reconstructed_features.to(dtype).contiguous(),
        ).detach()
    inference_seconds = time.perf_counter() - inference_started
    approximate_metrics = distribution_metrics(
        reference_logits,
        approximate_logits,
        token_ids,
        int(sample.answer_index),
    )
    dense_state_bytes = int(pool_features.numel() * pool_features.element_size())
    reference_correct = int(reference_metrics["candidate_correct"])
    candidate_correct = int(approximate_metrics["candidate_correct"])
    row = {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "answer_index": int(sample.answer_index),
        "reference_candidate_prediction": int(
            reference_metrics["candidate_prediction"]
        ),
        "reference_candidate_correct": reference_correct,
        "feature_relative_l2": relative_feature_error(
            reference_features,
            reconstructed_features,
        ),
        "native_feature_state_bytes": int(state.stream_state_bytes),
        "dense_native_feature_bytes": dense_state_bytes,
        "state_compression_ratio": dense_state_bytes / state.stream_state_bytes,
        "vision_seconds": vision_seconds,
        "inference_seconds": inference_seconds,
        "injection_max_abs": injection_max_abs,
        "prediction_match": int(
            reference_metrics["candidate_prediction"]
            == approximate_metrics["candidate_prediction"]
        ),
        "harmful_flip": int(reference_correct == 1 and candidate_correct == 0),
        "beneficial_flip": int(reference_correct == 0 and candidate_correct == 1),
        **approximate_metrics,
    }
    metadata = {
        "sample_id": sample.sample_id,
        "pool_indices": pool_indices,
        "selected_positions": selected_positions,
        "pool_feature_shape": list(pool_features.shape),
    }
    return row, metadata


def main() -> int:
    args = parse_args()
    tasks = parse_csv_list(args.tasks)
    if tuple(tasks) != DEFAULT_TASKS:
        raise ValueError("confirmation tasks differ from the frozen protocol")
    if args.samples_per_task != 100 or args.selection_seed != 20260829:
        raise ValueError("confirmation sample count or seed differs from the frozen protocol")
    all_samples, excluded_by_task = select_eligible_samples(
        args.dataset_root,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        selection_seed=args.selection_seed,
    )
    if len(all_samples) != 500:
        raise ValueError(f"expected 500 confirmation samples, found {len(all_samples)}")
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
                "excluded_by_task": excluded_by_task,
                "eligibility": "2_to_26_candidates_and_answer_in_candidates",
                "selection_seed": args.selection_seed,
                "sampled_frames": args.sampled_frames,
                "feature_pool_frames": args.feature_pool_frames,
                "frame_budget": args.frame_budget,
                "candidate": "pca_r456_s0",
                "claim_tier": "onevision_pca_r456_untouched_task_confirmation",
            },
        )
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    codec, codec_metadata = load_codec(
        args.codec_path,
        device=next(model.parameters()).device,
        dtype=torch.float16,
    )
    if codec.rank != 456:
        raise ValueError(f"confirmation requires rank 456, found rank {codec.rank}")
    if args.shard_index == 0:
        write_json_atomic(args.out_dir / "codec_metadata.json", codec_metadata)

    failures = []
    for position, sample in enumerate(samples, start=1):
        checkpoint = args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        if checkpoint.exists() and not args.overwrite:
            print(json.dumps({"event": "resume", "sample": sample.sample_id}), flush=True)
            continue
        try:
            row, metadata = evaluate_sample(
                sample=sample,
                processor=processor,
                model=model,
                codec=codec,
                sampled_frames=args.sampled_frames,
                feature_pool_frames=args.feature_pool_frames,
                frame_budget=args.frame_budget,
            )
            write_json_atomic(checkpoint, {"metadata": metadata, "row": row})
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
