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
from mvbench_utils import shard_samples
from videomme_onevision_domain_residual import (
    load_role_samples,
    role_duration_counts,
    validate_manifest,
)
from videomme_onevision_domain_residual_protocol import CANDIDATES, PROTOCOL_ID, RANK
from videomme_onevision_pca_replication import (
    VideoMMESample,
    decode_uniform_feature_pool,
    load_parquet_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("selection", "formal"), default="selection")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def evaluate_sample(
    *,
    sample: VideoMMESample,
    processor: object,
    model: torch.nn.Module,
    codecs: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    frames, pool_indices, selected_positions = decode_uniform_feature_pool(
        sample,
        feature_pool_frames=16,
        frame_budget=8,
    )
    pixels = preprocess_video(processor, frames, device=device, dtype=dtype)
    vision_started = time.perf_counter()
    with torch.inference_mode():
        pool_features = encode_video_features(model, pixels)
    vision_seconds = time.perf_counter() - vision_started
    positions = torch.tensor(selected_positions, device=device, dtype=torch.long)
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
        sample.answer_index,
    )

    rows = []
    for candidate_id in CANDIDATES:
        codec = codecs[candidate_id]
        latents = codec.encode(pool_features).to(torch.float16)
        reconstructed_pool = codec.decode(latents)
        reconstructed = reconstructed_pool.index_select(0, positions).contiguous()
        state = encode_feature_memory(
            pool_features,
            codec,
            residual_tokens_per_frame=0,
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reconstructed.to(dtype).contiguous(),
            ).detach()
        inference_seconds = time.perf_counter() - inference_started
        metrics = distribution_metrics(
            reference_logits,
            logits,
            token_ids,
            sample.answer_index,
        )
        reference_correct = int(reference_metrics["candidate_correct"])
        candidate_correct = int(metrics["candidate_correct"])
        dense_bytes = int(pool_features.numel() * pool_features.element_size())
        rows.append(
            {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "video_id": sample.video_id,
                "duration": sample.duration,
                "domain": sample.domain,
                "task_type": sample.task_type,
                "candidate": candidate_id,
                "answer_index": sample.answer_index,
                "reference_candidate_prediction": int(
                    reference_metrics["candidate_prediction"]
                ),
                "reference_candidate_correct": reference_correct,
                "feature_relative_l2": relative_feature_error(
                    reference_features,
                    reconstructed,
                ),
                "native_feature_state_bytes": int(state.stream_state_bytes),
                "dense_native_feature_bytes": dense_bytes,
                "state_compression_ratio": dense_bytes / state.stream_state_bytes,
                "vision_seconds": vision_seconds,
                "inference_seconds": inference_seconds,
                "injection_max_abs": injection_max_abs,
                "prediction_match": int(
                    reference_metrics["candidate_prediction"]
                    == metrics["candidate_prediction"]
                ),
                "harmful_flip": int(reference_correct == 1 and candidate_correct == 0),
                "beneficial_flip": int(
                    reference_correct == 0 and candidate_correct == 1
                ),
                **metrics,
            }
        )
    return rows, {
        "sample_id": sample.sample_id,
        "pool_indices": pool_indices,
        "selected_positions": selected_positions,
        "pool_feature_shape": list(pool_features.shape),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    records = load_parquet_records(args.parquet_path)
    all_samples = load_role_samples(
        records=records,
        video_root=args.video_root,
        manifest=manifest,
        role=args.role,
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
                "protocol_id": PROTOCOL_ID,
                "role": args.role,
                "candidates": list(CANDIDATES),
                "rank": RANK,
                "feature_pool_frames": 16,
                "frame_budget": 8,
                "frame_policy": "uniform16_pool_uniform8_reader",
                "sample_ids": [sample.sample_id for sample in all_samples],
                "video_ids": [sample.video_id for sample in all_samples],
                "duration_counts": role_duration_counts(all_samples),
            },
        )
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    device = next(model.parameters()).device
    codecs = {}
    codec_metadata = {}
    for candidate_id in CANDIDATES:
        codec, metadata = load_codec(
            args.codec_dir / f"{candidate_id}.pt",
            device=device,
            dtype=torch.float16,
        )
        if codec.rank != RANK or metadata["candidate"] != candidate_id:
            raise ValueError(f"codec identity mismatch for {candidate_id}")
        codecs[candidate_id] = codec
        codec_metadata[candidate_id] = metadata
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
                codecs=codecs,
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
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "video_id": sample.video_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                json.dumps(
                    {"event": "failure", "sample": sample.sample_id, "error": repr(error)}
                ),
                flush=True,
            )
            if args.fail_fast:
                break
    write_json_atomic(args.out_dir / f"failures_shard_{args.shard_index}.json", failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
