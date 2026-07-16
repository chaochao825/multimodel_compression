from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from feature_memory_codec import (
    encode_feature_memory,
    fit_pca_codec,
    reconstruct_feature_memory,
    relative_reconstruction_error,
    save_codec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--expected-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    token_files = sorted(args.token_dir.glob("*.pt"))
    if args.expected_files > 0 and len(token_files) != args.expected_files:
        raise RuntimeError(
            f"expected {args.expected_files} token files, "
            f"found {len(token_files)}"
        )
    if not token_files:
        raise RuntimeError("no extracted token files found")

    token_blocks = []
    sample_ids = []
    fingerprints = set()
    source_shapes = set()
    file_hashes = {}
    for path in token_files:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"unsupported token file: {path}")
        token_blocks.append(payload["tokens"])
        sample_ids.append(str(payload["sample_id"]))
        fingerprints.add(str(payload["configuration_fingerprint"]))
        source_shapes.add(tuple(payload["source_feature_shape"]))
        file_hashes[path.name] = sha256(path)
    if len(fingerprints) != 1:
        raise RuntimeError("extracted tokens use multiple configurations")
    if len(source_shapes) != 1:
        raise RuntimeError("extracted native feature shapes differ")

    training_tokens = torch.cat(token_blocks, dim=0)
    training_features = training_tokens.unsqueeze(0).to(args.device)
    started = time.perf_counter()
    codec, fit_metadata = fit_pca_codec(
        training_features,
        rank=args.rank,
        seed=args.seed,
        niter=args.niter,
        storage_dtype=torch.float16,
    )
    state = encode_feature_memory(
        training_features,
        codec,
        residual_tokens_per_frame=0,
    )
    reconstruction = reconstruct_feature_memory(
        state,
        codec,
        output_dtype=torch.float16,
    )
    training_error = relative_reconstruction_error(
        training_features,
        reconstruction,
    )
    fit_seconds = time.perf_counter() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codec_path = args.out_dir / f"llava_feature_pca_rank{args.rank}.pt"
    metadata = {
        **fit_metadata,
        "extraction_configuration_fingerprint": next(
            iter(fingerprints)
        ),
        "source_feature_shape": list(next(iter(source_shapes))),
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "training_relative_reconstruction_error": training_error,
        "fit_seconds": fit_seconds,
        "token_file_sha256": file_hashes,
    }
    save_codec(codec, codec_path, metadata=metadata)
    summary = {
        **metadata,
        "codec_path": str(codec_path.resolve()),
        "codec_sha256": sha256(codec_path),
        "training_tensor_bytes_fp16": int(
            training_tokens.numel() * training_tokens.element_size()
        ),
    }
    write_json_atomic(args.out_dir / "fit_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
