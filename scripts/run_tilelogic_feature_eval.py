#!/usr/bin/env python3
"""Evaluate TileLogic-RVQ feature distortion, routing, and exact bit rate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch

from tilespec_ex.cache import (
    load_cache_manifest,
    load_cache_payload,
    manifest_sha256,
    validate_split_contract,
)
from tilespec_ex.core import (
    RETENTION_RATES,
    block_query_relevance,
    cosine_similarity,
    crop_boundary_mse,
    enumerate_blocks,
    normalized_mse,
)
from tilespec_ex.rate import RATE_STORAGE_POLICY
from tilespec_ex.tilelogic_codec import encode_base_vq, oracle_marginal_benefits
from tilespec_ex.tilelogic_methods import (
    ORIGINAL_CROP_TOKENS,
    build_tilelogic_variants,
    load_tilelogic_artifacts,
)


EVALUATION_FORMAT = "tilelogic_feature_evaluation_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return output


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _exception_budget(rate: float) -> tuple[int, int, int]:
    retained = round(ORIGINAL_CROP_TOKENS * rate)
    exception_tokens = round(retained * 0.25 / 4) * 4
    return retained, retained - exception_tokens, exception_tokens


def _weighted_nmse(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    channel_fisher: torch.Tensor,
) -> float:
    weights = channel_fisher.to(reference.device).reshape(1, 1, 1, -1)
    error = ((reference.float() - estimate.float()).square() * weights).sum()
    denominator = (reference.float().square() * weights).sum().clamp_min(1e-20)
    return float((error / denominator).item())


def _first_order_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    gradient: torch.Tensor,
) -> float:
    if reference.shape != estimate.shape or reference.shape != gradient.shape:
        raise ValueError("first-order tensors must share a shape")
    return float(
        ((reference.float() - estimate.float()) * gradient.float()).sum().abs().item()
    )


def _mode_summary(modes: torch.Tensor | None, stages: int) -> dict[str, int] | None:
    if modes is None:
        return None
    return {
        "drop": int((modes == 0).sum()),
        **{f"rvq{depth}": int((modes == depth).sum()) for depth in range(1, stages + 1)},
        "exact": int((modes == stages + 1).sum()),
    }


def _router_oracle_record(
    crops: torch.Tensor,
    thumbnail: torch.Tensor,
    query: torch.Tensor,
    gradient: torch.Tensor,
    artifacts: Any,
    rate: float,
    variants: list[Any],
) -> dict[str, Any]:
    _, base_tokens, _ = _exception_budget(rate)
    base = encode_base_vq(crops, base_tokens, artifacts.base).reconstructed
    residual = crops - base
    blocks, locations, energy = enumerate_blocks(residual)
    gradient_blocks, gradient_locations, _ = enumerate_blocks(gradient)
    if locations != gradient_locations:
        raise AssertionError("evaluation gradient and residual layouts differ")
    oracle = oracle_marginal_benefits(
        blocks, gradient_blocks, artifacts.residual_fisher
    )
    relevance = block_query_relevance(blocks, query)
    scores = {
        variant.method: variant.router_marginal_scores.detach().cpu().tolist()
        for variant in variants
        if variant.rate == rate and variant.router_marginal_scores is not None
    }
    return {
        "retention_rate": rate,
        "locations": [list(item) for item in locations],
        "oracle_marginal_benefits": oracle.detach().cpu().tolist(),
        "energy": energy.detach().cpu().tolist(),
        "query_relevance": relevance.detach().cpu().tolist(),
        "risk": (energy * relevance).detach().cpu().tolist(),
        "router_marginal_scores": scores,
    }


def _validate_training(
    training_dir: Path, cache_dir: Path
) -> dict[str, Any]:
    path = training_dir / "training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("format") != "tilelogic_rvq_training_v1" or not summary.get("complete"):
        raise RuntimeError("training summary is incomplete or unsupported")
    if summary.get("evaluation_entries_loaded") != 0:
        raise RuntimeError("training summary reports evaluation leakage")
    if summary.get("cache_manifest_sha256") != manifest_sha256(cache_dir):
        raise RuntimeError("training and evaluation cache manifests differ")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-cache-hashes", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    training_dir = args.training_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_summary = _validate_training(training_dir, cache_dir)
    entries = load_cache_manifest(cache_dir, verify_files=args.verify_cache_hashes)
    split_counts = validate_split_contract(
        entries,
        calibration_per_dataset=80,
        evaluation_per_dataset=120,
        oracle_per_dataset_split=16,
    )
    evaluation = [entry for entry in entries if entry.split == "evaluation"]
    if len(evaluation) != 360 or any(entry.split != "evaluation" for entry in evaluation):
        raise RuntimeError("feature evaluation must use exactly 360 evaluation entries")
    artifacts = load_tilelogic_artifacts(training_dir).to(args.device)
    fisher_state = torch.load(
        training_dir / "channel_fisher.pt", map_location="cpu", weights_only=True
    )
    if fisher_state.get("format") != "tilelogic_channel_fisher_v1":
        raise RuntimeError("unsupported Fisher artifact")
    channel_fisher = fisher_state["weights"].float().to(args.device)

    samples_path = output_dir / "feature_samples.jsonl"
    completed_records = _read_jsonl(samples_path)
    completed = {
        (str(record["dataset"]), int(record["dataset_index"]))
        for record in completed_records
    }
    if len(completed) != len(completed_records):
        raise RuntimeError("feature evaluation contains duplicate samples")
    expected_keys = {entry.key for entry in evaluation}
    if not completed <= expected_keys:
        raise RuntimeError("feature evaluation contains non-evaluation samples")

    started = time.perf_counter()
    new_records = 0
    for sequence_index, entry in enumerate(evaluation):
        if entry.key in completed:
            continue
        sample_started = time.perf_counter()
        payload = load_cache_payload(cache_dir, entry, device=args.device)
        crops = payload["crops"]
        thumbnail = payload["thumbnail"]
        query = payload["query"]
        variants = build_tilelogic_variants(
            thumbnail, crops, query, artifacts
        )
        gradient = payload.get("crop_gradient")
        variant_records = []
        for variant in variants:
            rate_metrics = variant.ledger.metrics(
                original_vectors=ORIGINAL_CROP_TOKENS,
                vector_dimension=crops.shape[-1],
                amortization_count=len(evaluation),
            )
            variant_records.append(
                {
                    "method": variant.method,
                    "retention_rate": variant.rate,
                    "scope": variant.scope,
                    "feature_nmse": normalized_mse(crops, variant.reconstructed),
                    "feature_cosine": cosine_similarity(crops, variant.reconstructed),
                    "boundary_mse": crop_boundary_mse(crops, variant.reconstructed),
                    "fisher_weighted_nmse": _weighted_nmse(
                        crops, variant.reconstructed, channel_fisher
                    ),
                    "oracle_first_order_abs": (
                        _first_order_error(crops, variant.reconstructed, gradient)
                        if gradient is not None
                        else None
                    ),
                    "base_tokens": variant.base_tokens,
                    "residual_budget_bits": variant.residual_budget_bits,
                    "residual_spent_bits": variant.residual_spent_bits,
                    "mode_counts": _mode_summary(
                        variant.residual_modes, artifacts.residual_fisher.stages
                    ),
                    "rate": rate_metrics,
                    "rate_components": variant.ledger.as_dict()["components"],
                }
            )
        router_oracle = []
        if gradient is not None:
            router_oracle = [
                _router_oracle_record(
                    crops,
                    thumbnail,
                    query,
                    gradient,
                    artifacts,
                    rate,
                    variants,
                )
                for rate in RETENTION_RATES
            ]
        record = {
            "format": EVALUATION_FORMAT,
            "dataset": entry.dataset,
            "dataset_index": entry.dataset_index,
            "sample_id": entry.sample_id,
            "image_sha256": entry.image_sha256,
            "split": entry.split,
            "oracle": entry.oracle,
            "variants": variant_records,
            "router_oracle": router_oracle,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        _append_jsonl(samples_path, record)
        completed.add(entry.key)
        new_records += 1
        print(
            json.dumps(
                {
                    "progress": f"{sequence_index + 1}/{len(evaluation)}",
                    "dataset": entry.dataset,
                    "dataset_index": entry.dataset_index,
                    "oracle": entry.oracle,
                    "seconds": round(time.perf_counter() - sample_started, 3),
                }
            ),
            flush=True,
        )

    summary = {
        "format": EVALUATION_FORMAT,
        "records": len(completed),
        "new_records": new_records,
        "expected_records": len(evaluation),
        "variants_per_record": 1 + len(RETENTION_RATES) * 11,
        "cache_manifest_sha256": manifest_sha256(cache_dir),
        "training_summary_sha256": hashlib.sha256(
            (training_dir / "training_summary.json").read_bytes()
        ).hexdigest(),
        "split_counts": split_counts,
        "calibration_records_loaded": 0,
        "evaluation_records_loaded": len(completed),
        "complete": len(completed) == len(evaluation),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "training_format": training_summary["format"],
        "rate_storage_policy": RATE_STORAGE_POLICY,
    }
    (output_dir / "feature_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
