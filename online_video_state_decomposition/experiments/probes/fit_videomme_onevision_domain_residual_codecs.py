from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from feature_memory_codec import load_codec, save_codec
from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    encode_video_features,
    load_onevision_model,
    preprocess_video,
)
from videomme_onevision_domain_residual import (
    fit_domain_residual_codecs,
    load_role_samples,
    role_duration_counts,
)
from videomme_onevision_domain_residual_protocol import PROTOCOL_ID, SELECTION_SEED
from videomme_onevision_pca_replication import (
    decode_uniform_feature_pool,
    load_parquet_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-codec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    records = load_parquet_records(args.parquet_path)
    samples = load_role_samples(
        records=records,
        video_root=args.video_root,
        manifest=manifest,
        role="calibration",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    blocks = []
    extraction_started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        frames, _, _ = decode_uniform_feature_pool(
            sample,
            feature_pool_frames=16,
            frame_budget=8,
        )
        pixels = preprocess_video(processor, frames, device=device, dtype=dtype)
        with torch.inference_mode():
            features = encode_video_features(model, pixels)
        blocks.append(features.to(device="cpu", dtype=torch.float16))
        print(
            json.dumps(
                {
                    "event": "calibration_feature_ok",
                    "sample": sample.sample_id,
                    "position": position,
                    "total": len(samples),
                }
            ),
            flush=True,
        )
    extraction_seconds = time.perf_counter() - extraction_started
    calibration_features = torch.cat(blocks, dim=0).to(device)
    source_codec, source_metadata = load_codec(
        args.source_codec,
        device=device,
        dtype=torch.float16,
    )
    fit_started = time.perf_counter()
    candidates, fit_metadata = fit_domain_residual_codecs(
        features=calibration_features,
        source_codec=source_codec,
        seed=SELECTION_SEED,
        niter=args.niter,
    )
    fit_seconds = time.perf_counter() - fit_started
    candidate_dir = args.out_dir / "codecs"
    for candidate_id, codec in candidates.items():
        save_codec(
            codec,
            candidate_dir / f"{candidate_id}.pt",
            metadata={
                "protocol_id": PROTOCOL_ID,
                "candidate": candidate_id,
                "role": "calibration_only",
                "calibration_sample_ids": [sample.sample_id for sample in samples],
                "feature_pool_frames": 16,
                "frame_budget": 8,
                **fit_metadata["candidates"][candidate_id],
            },
        )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "calibration_samples": len(samples),
        "calibration_sample_ids": [sample.sample_id for sample in samples],
        "duration_counts": role_duration_counts(samples),
        "source_codec_metadata": source_metadata,
        "extraction_seconds": extraction_seconds,
        "fit_seconds": fit_seconds,
        **fit_metadata,
    }
    write_json_atomic(args.out_dir / "fit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
