from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from dataclasses import dataclass
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
from mvbench_utils import (
    decode_video_frames,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)
from videomme_onevision_pca_protocol import (
    DURATIONS,
    FORMAL_SAMPLES_PER_DURATION,
    FORMAL_SEED,
    HISTORICAL_VIDEO_IDS,
    SPLIT_PROTOCOL_ID,
    option_text,
    record_is_eligible,
)


@dataclass(frozen=True)
class VideoMMESample:
    question_id: str
    video_id: str
    video_path: Path
    duration: str
    domain: str
    task_type: str
    question: str
    candidates: tuple[str, ...]
    correct_index: int

    @property
    def task(self) -> str:
        return self.duration

    @property
    def sample_id(self) -> str:
        return f"videomme_{self.question_id}"

    @property
    def answer(self) -> str:
        return self.candidates[self.correct_index]

    @property
    def answer_index(self) -> int:
        return self.correct_index

    @property
    def subtitle(self) -> str:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-pool-frames", type=int, default=16)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_parquet_records(path: Path) -> list[dict[str, object]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def load_split_samples(
    *,
    records: list[dict[str, object]],
    video_root: Path,
    split: dict[str, object],
) -> list[VideoMMESample]:
    if split["protocol_id"] != SPLIT_PROTOCOL_ID:
        raise ValueError("split protocol identity mismatch")
    if int(split["selection_seed"]) != FORMAL_SEED:
        raise ValueError("split seed mismatch")
    if int(split["samples_per_duration"]) != FORMAL_SAMPLES_PER_DURATION:
        raise ValueError("split sample count mismatch")
    if set(split["historical_video_exclusions"]) != HISTORICAL_VIDEO_IDS:
        raise ValueError("historical exclusion set mismatch")
    if split["frame_policy"] != "uniform16_pool_uniform8_reader":
        raise ValueError("frame policy mismatch")

    by_question = {str(record["question_id"]): record for record in records}
    samples: list[VideoMMESample] = []
    for entry in split["samples"]:
        question_id = str(entry["question_id"])
        if question_id not in by_question:
            raise KeyError(f"question {question_id} is absent from the parquet")
        record = by_question[question_id]
        video_id = str(record["videoID"])
        if (
            video_id != entry["video_id"]
            or str(record["duration"]) != entry["duration"]
            or str(record["domain"]) != entry["domain"]
            or str(record["task_type"]) != entry["task_type"]
        ):
            raise ValueError(f"split metadata mismatch for {question_id}")
        if not record_is_eligible(record):
            raise ValueError(f"question {question_id} is not eligible")
        options = tuple(
            option_text(option, chr(ord("A") + index))
            for index, option in enumerate(record["options"])
        )
        correct_index = ord(str(record["answer"]).strip().upper()) - ord("A")
        video_path = video_root / f"{video_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        samples.append(
            VideoMMESample(
                question_id=question_id,
                video_id=video_id,
                video_path=video_path,
                duration=str(record["duration"]),
                domain=str(record["domain"]),
                task_type=str(record["task_type"]),
                question=str(record["question"]),
                candidates=options,
                correct_index=correct_index,
            )
        )

    expected = FORMAL_SAMPLES_PER_DURATION * len(DURATIONS)
    if len(samples) != expected:
        raise ValueError(f"expected {expected} formal samples, found {len(samples)}")
    if len({sample.video_id for sample in samples}) != expected:
        raise ValueError("formal samples must use one question per video")
    counts = Counter(sample.duration for sample in samples)
    if counts != Counter({duration: FORMAL_SAMPLES_PER_DURATION for duration in DURATIONS}):
        raise ValueError("formal duration counts differ from the protocol")
    return samples


def decode_uniform_feature_pool(
    sample: VideoMMESample,
    *,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    if not 0 < frame_budget <= feature_pool_frames:
        raise ValueError("frame budgets must be positive and nested")
    total_frames, _ = video_metadata(sample.video_path)
    pool_indices = uniform_frame_indices(total_frames, feature_pool_frames)
    if len(pool_indices) != feature_pool_frames:
        raise ValueError("video has fewer frames than the feature pool")
    selected_positions = uniform_frame_indices(feature_pool_frames, frame_budget)
    frames, _, decoded_total = decode_video_frames(sample.video_path, pool_indices)
    if decoded_total != total_frames:
        raise RuntimeError("video frame count changed during decoding")
    return np.stack(frames), pool_indices, selected_positions


def evaluate_sample(
    *,
    sample: VideoMMESample,
    processor: object,
    model: torch.nn.Module,
    codec: object,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[dict[str, object], dict[str, object]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    frames, pool_indices, selected_positions = decode_uniform_feature_pool(
        sample,
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
        sample.answer_index,
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
        sample.answer_index,
    )
    dense_state_bytes = int(pool_features.numel() * pool_features.element_size())
    reference_correct = int(reference_metrics["candidate_correct"])
    candidate_correct = int(approximate_metrics["candidate_correct"])
    row = {
        "sample_id": sample.sample_id,
        "question_id": sample.question_id,
        "video_id": sample.video_id,
        "duration": sample.duration,
        "domain": sample.domain,
        "task_type": sample.task_type,
        "answer_index": sample.answer_index,
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
    if args.feature_pool_frames != 16 or args.frame_budget != 8:
        raise ValueError("frame budgets differ from the frozen protocol")
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    records = load_parquet_records(args.parquet_path)
    all_samples = load_split_samples(
        records=records,
        video_root=args.video_root,
        split=split,
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
                "benchmark": "Video-MME",
                "candidate": "pca_r456_s0",
                "claim_tier": "onevision_pca_r456_cross_domain_replication",
                "split_protocol_id": SPLIT_PROTOCOL_ID,
                "selection_seed": FORMAL_SEED,
                "samples_per_duration": FORMAL_SAMPLES_PER_DURATION,
                "sample_ids": [sample.sample_id for sample in all_samples],
                "video_ids": [sample.video_id for sample in all_samples],
                "duration_counts": Counter(sample.duration for sample in all_samples),
                "domain_counts": Counter(sample.domain for sample in all_samples),
                "task_type_counts": Counter(sample.task_type for sample in all_samples),
                "feature_pool_frames": args.feature_pool_frames,
                "frame_budget": args.frame_budget,
                "frame_policy": "uniform16_pool_uniform8_reader",
                "subtitles": "disabled",
            },
        )
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    codec, codec_metadata = load_codec(
        args.codec_path,
        device=next(model.parameters()).device,
        dtype=torch.float16,
    )
    if codec.rank != 456:
        raise ValueError(f"replication requires rank 456, found rank {codec.rank}")
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
                "video_id": sample.video_id,
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
