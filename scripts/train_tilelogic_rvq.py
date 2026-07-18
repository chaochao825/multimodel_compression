#!/usr/bin/env python3
"""Fit TileLogic-RVQ codebooks and routers from calibration cache only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch

from tilespec_ex.cache import (
    CacheEntry,
    load_cache_manifest,
    load_cache_payload,
    manifest_sha256,
    validate_split_contract,
)
from tilespec_ex.core import RETENTION_RATES, enumerate_blocks
from tilespec_ex.routing import (
    BLOCK_FEATURE_NAMES,
    FeatureNormalizer,
    RouterMLP,
    block_router_features,
    fit_logic_router,
    fit_router_mlp,
    fixed_slot_mask,
)
from tilespec_ex.tilelogic_codec import (
    encode_base_vq,
    extract_tile_coefficients,
    oracle_marginal_benefits,
)
from tilespec_ex.vq import fit_residual_vq_codebook, fit_scaled_codebook


TRAINING_FORMAT = "tilelogic_rvq_training_v1"
ORIGINAL_CROP_TOKENS = 4 * 16 * 16


def _rate_key(rate: float) -> str:
    return f"rate_{rate:.3f}".replace(".", "p")


def _exception_budget(rate: float) -> tuple[int, int, int]:
    retained = round(ORIGINAL_CROP_TOKENS * rate)
    exception_tokens = round(retained * 0.25 / 4) * 4
    base_tokens = retained - exception_tokens
    if retained % 4 or base_tokens % 4 or exception_tokens % 4:
        raise ValueError("rate does not satisfy the four-tile/2x2 block contract")
    return retained, base_tokens, exception_tokens


def _save_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(state, temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


def _sample_rows(tensor: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if tensor.shape[0] <= count:
        return tensor
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(tensor.shape[0], generator=generator)[:count]
    return tensor[indices]


def _calibration_entries(entries: list[CacheEntry]) -> list[CacheEntry]:
    selected = [entry for entry in entries if entry.split == "calibration"]
    if any(entry.split != "calibration" for entry in selected):
        raise AssertionError("non-calibration entry entered training")
    return selected


def _empirical_fisher(
    cache_dir: Path, entries: list[CacheEntry]
) -> tuple[torch.Tensor, dict[str, Any]]:
    oracle_entries = [entry for entry in entries if entry.oracle]
    if not oracle_entries:
        raise RuntimeError("calibration cache has no oracle gradients")
    accumulator: torch.Tensor | None = None
    spatial_values = 0
    for entry in oracle_entries:
        payload = load_cache_payload(cache_dir, entry)
        gradient = payload["crop_gradient"].float()
        channel_sum = gradient.square().sum(dim=(0, 1, 2)).double()
        accumulator = channel_sum if accumulator is None else accumulator + channel_sum
        spatial_values += math.prod(gradient.shape[:-1])
    assert accumulator is not None
    raw = (accumulator / spatial_values).float().clamp_min(1e-20)
    lower = torch.quantile(raw, 0.01)
    upper = torch.quantile(raw, 0.99)
    clipped = raw.clamp(lower, upper)
    normalized = clipped / clipped.mean().clamp_min(1e-20)
    summary = {
        "oracle_entries": len(oracle_entries),
        "spatial_values": spatial_values,
        "raw_min": float(raw.min()),
        "raw_median": float(raw.median()),
        "raw_max": float(raw.max()),
        "clip_lower": float(lower),
        "clip_upper": float(upper),
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "normalized_mean": float(normalized.mean()),
    }
    return normalized, summary


def _collect_base_vectors(
    cache_dir: Path,
    entries: list[CacheEntry],
    *,
    max_vectors: int,
    device: str,
) -> torch.Tensor:
    maximum_retained = round(ORIGINAL_CROP_TOKENS * max(RETENTION_RATES))
    per_entry = max(1, math.ceil(max_vectors / len(entries)))
    collected = []
    for sequence, entry in enumerate(entries):
        payload = load_cache_payload(cache_dir, entry)
        crops = payload["crops"].to(device)
        coefficients = extract_tile_coefficients(crops, maximum_retained).cpu()
        collected.append(_sample_rows(coefficients, per_entry, 1009 + sequence))
    vectors = torch.cat(collected)
    return _sample_rows(vectors, max_vectors, 20260718)


def _collect_residual_vectors(
    cache_dir: Path,
    entries: list[CacheEntry],
    base_codebook: Any,
    *,
    max_blocks: int,
    device: str,
) -> torch.Tensor:
    per_entry_rate = max(1, math.ceil(max_blocks / (len(entries) * len(RETENTION_RATES))))
    collected = []
    codebook = base_codebook.to(device)
    sequence = 0
    for entry in entries:
        payload = load_cache_payload(cache_dir, entry)
        crops = payload["crops"].to(device)
        for rate in RETENTION_RATES:
            _, base_tokens, _ = _exception_budget(rate)
            base = encode_base_vq(crops, base_tokens, codebook).reconstructed
            blocks, _, _ = enumerate_blocks(crops - base)
            flat = blocks.reshape(blocks.shape[0], -1).cpu()
            collected.append(_sample_rows(flat, per_entry_rate, 3001 + sequence))
            sequence += 1
    vectors = torch.cat(collected)
    return _sample_rows(vectors, max_blocks, 20260719)


def _oracle_train_validation(entries: list[CacheEntry]) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    train: set[tuple[str, int]] = set()
    validation: set[tuple[str, int]] = set()
    for dataset in ("gqa", "textvqa", "chartqa"):
        selected = sorted(
            [entry for entry in entries if entry.dataset == dataset and entry.oracle],
            key=lambda entry: hashlib.sha256(
                f"router-split:{entry.dataset}:{entry.sample_id}".encode()
            ).hexdigest(),
        )
        if len(selected) < 4:
            raise RuntimeError("router split needs at least four oracle examples per dataset")
        validation_count = max(1, len(selected) // 4)
        validation.update(entry.key for entry in selected[:validation_count])
        train.update(entry.key for entry in selected[validation_count:])
    if train & validation:
        raise AssertionError("router train/validation overlap")
    return train, validation


def _router_records(
    cache_dir: Path,
    entries: list[CacheEntry],
    base_codebook: Any,
    residual_codebook: Any,
    *,
    device: str,
) -> dict[float, dict[str, Any]]:
    oracle_entries = [entry for entry in entries if entry.oracle]
    train_keys, validation_keys = _oracle_train_validation(entries)
    output: dict[float, dict[str, Any]] = {}
    base_codebook = base_codebook.to(device)
    residual_codebook = residual_codebook.to(device)
    for rate in RETENTION_RATES:
        output[rate] = {
            "features": [],
            "targets": [],
            "keys": [],
            "locations": None,
            "curvature_sum": None,
            "curvature_count": 0,
            "train_keys": train_keys,
            "validation_keys": validation_keys,
        }
    for entry in oracle_entries:
        payload = load_cache_payload(cache_dir, entry)
        crops = payload["crops"].to(device)
        thumbnail = payload["thumbnail"].to(device)
        query = payload["query"].to(device)
        gradients = payload["crop_gradient"].to(device)
        gradient_blocks, gradient_locations, _ = enumerate_blocks(gradients)
        for rate in RETENTION_RATES:
            _, base_tokens, _ = _exception_budget(rate)
            base = encode_base_vq(crops, base_tokens, base_codebook).reconstructed
            residual = crops - base
            blocks, locations, _ = enumerate_blocks(residual)
            if locations != gradient_locations:
                raise AssertionError("residual and gradient block layouts differ")
            targets = oracle_marginal_benefits(
                blocks, gradient_blocks, residual_codebook
            )
            features, feature_locations = block_router_features(
                crops, residual, query, thumbnail
            )
            if feature_locations != locations:
                raise AssertionError("router feature layout differs from residual")
            record = output[rate]
            if record["locations"] is None:
                record["locations"] = locations
                record["curvature_sum"] = torch.zeros(
                    len(locations), device=device, dtype=torch.float64
                )
            elif record["locations"] != locations:
                raise AssertionError("router location schema drift")
            baseline_importance = (
                blocks.float().reshape(blocks.shape[0], -1)
                * gradient_blocks.float().reshape(gradient_blocks.shape[0], -1)
            ).sum(dim=1).abs().double()
            record["curvature_sum"] += baseline_importance
            record["curvature_count"] += 1
            record["features"].append(features.cpu())
            record["targets"].append(targets.cpu())
            record["keys"].extend([entry.key] * len(locations))
    for rate, record in output.items():
        prior = (record["curvature_sum"] / record["curvature_count"]).float().cpu()
        prior = prior / prior.mean().clamp_min(1e-20)
        features = torch.cat(record["features"])
        targets = torch.cat(record["targets"])
        location_count = len(record["locations"])
        repeated_prior = prior.repeat(len(oracle_entries))
        if repeated_prior.shape[0] != features.shape[0]:
            raise AssertionError("curvature prior repetition mismatch")
        features[:, -1] = repeated_prior
        record["features"] = features
        record["targets"] = targets
        record["curvature_prior"] = prior
        record["location_count"] = location_count
    return output


def _train_routers(
    router_records: dict[float, dict[str, Any]],
    output_dir: Path,
    *,
    hidden_dim: int,
    epochs: int,
    device: str,
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    artifacts: dict[str, Any] = {}
    bundles: dict[float, dict[str, Any]] = {}
    for rate, record in router_records.items():
        keys = record["keys"]
        train_mask = torch.tensor([key in record["train_keys"] for key in keys])
        validation_mask = torch.tensor(
            [key in record["validation_keys"] for key in keys]
        )
        if bool((train_mask & validation_mask).any()) or not bool(train_mask.any()) or not bool(validation_mask.any()):
            raise RuntimeError("invalid router train/validation masks")
        features = record["features"].float()
        targets = record["targets"].float()
        normalizer = FeatureNormalizer.fit(features[train_mask])
        normalized = normalizer.transform(features)
        model = RouterMLP(
            input_dim=features.shape[1],
            output_dim=targets.shape[1],
            hidden_dim=hidden_dim,
        ).to(device)
        summary = fit_router_mlp(
            model,
            normalized[train_mask].to(device),
            targets[train_mask].to(device),
            normalized[validation_mask].to(device),
            targets[validation_mask].to(device),
            epochs=epochs,
        )
        model = model.cpu().eval()
        with torch.inference_mode():
            teacher = model(normalized)
        logic = fit_logic_router(
            features,
            teacher,
            max_depth=6,
            min_leaf=32,
        )
        _, _, exception_tokens = _exception_budget(rate)
        slots_per_tile = exception_tokens // 4 // 4
        fixed_mask = fixed_slot_mask(
            record["curvature_prior"],
            record["locations"],
            slots_per_tile=slots_per_tile,
        )
        bundle = {
            "format": "tilelogic_router_bundle_v1",
            "rate": rate,
            "feature_names": BLOCK_FEATURE_NAMES,
            "normalizer": normalizer.state_dict(),
            "mlp": model.export_state(),
            "logic": logic.state_dict(),
            "curvature_prior": record["curvature_prior"],
            "fixed_slot_mask": fixed_mask,
            "locations": record["locations"],
            "router_train_keys": sorted(record["train_keys"]),
            "router_validation_keys": sorted(record["validation_keys"]),
            "training_summary": summary.__dict__,
        }
        path = output_dir / f"router_{_rate_key(rate)}.pt"
        artifacts[f"router_{_rate_key(rate)}"] = _save_state(path, bundle)
        bundles[rate] = bundle
    return artifacts, bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-train-vectors", type=int, default=32768)
    parser.add_argument("--residual-train-blocks", type=int, default=8192)
    parser.add_argument("--base-codes", type=int, default=256)
    parser.add_argument("--residual-codes", type=int, default=256)
    parser.add_argument("--scale-levels", type=int, default=16)
    parser.add_argument("--rvq-stages", type=int, default=2)
    parser.add_argument("--kmeans-iterations", type=int, default=12)
    parser.add_argument("--router-hidden", type=int, default=32)
    parser.add_argument("--router-epochs", type=int, default=300)
    parser.add_argument("--verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    if min(
        args.base_train_vectors,
        args.residual_train_blocks,
        args.base_codes,
        args.residual_codes,
        args.scale_levels,
        args.rvq_stages,
        args.kmeans_iterations,
        args.router_hidden,
        args.router_epochs,
    ) <= 0:
        raise SystemExit("all training dimensions/counts must be positive")

    torch.manual_seed(20260718)
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_cache_manifest(cache_dir, verify_files=args.verify_cache_hashes)
    split_counts = validate_split_contract(
        entries,
        calibration_per_dataset=80,
        evaluation_per_dataset=120,
        oracle_per_dataset_split=16,
    )
    calibration = _calibration_entries(entries)
    if len(calibration) != 240:
        raise RuntimeError("training contract requires exactly 240 calibration entries")
    training_keys = sorted(entry.key for entry in calibration)
    evaluation_keys = {entry.key for entry in entries if entry.split == "evaluation"}
    if set(training_keys) & evaluation_keys:
        raise RuntimeError("calibration/evaluation leakage detected")

    started = time.perf_counter()
    fisher, fisher_summary = _empirical_fisher(cache_dir, calibration)
    artifacts: dict[str, Any] = {}
    artifacts["fisher"] = _save_state(
        output_dir / "channel_fisher.pt",
        {
            "format": "tilelogic_channel_fisher_v1",
            "weights": fisher,
            "summary": fisher_summary,
        },
    )

    base_vectors = _collect_base_vectors(
        cache_dir,
        calibration,
        max_vectors=args.base_train_vectors,
        device=args.device,
    )
    base_codebook = fit_scaled_codebook(
        base_vectors.to(args.device, torch.float32),
        num_codes=args.base_codes,
        num_scale_levels=args.scale_levels,
        metric_weights=fisher.to(args.device),
        iterations=args.kmeans_iterations,
        seed=20260718,
        batch_size=512,
    ).to("cpu")
    artifacts["base_codebook"] = _save_state(
        output_dir / "base_codebook.pt", base_codebook.state_dict()
    )
    del base_vectors
    torch.cuda.empty_cache()

    residual_vectors = _collect_residual_vectors(
        cache_dir,
        calibration,
        base_codebook,
        max_blocks=args.residual_train_blocks,
        device=args.device,
    )
    residual_metric = fisher.repeat(4)
    residual_unweighted = fit_residual_vq_codebook(
        residual_vectors.to(args.device, torch.float32),
        stages=args.rvq_stages,
        num_codes=args.residual_codes,
        num_scale_levels=args.scale_levels,
        metric_weights=None,
        iterations=args.kmeans_iterations,
        seed=20260718,
        batch_size=256,
    ).to("cpu")
    artifacts["residual_rvq_unweighted"] = _save_state(
        output_dir / "residual_rvq_unweighted.pt",
        residual_unweighted.state_dict(),
    )
    torch.cuda.empty_cache()
    residual_fisher = fit_residual_vq_codebook(
        residual_vectors.to(args.device, torch.float32),
        stages=args.rvq_stages,
        num_codes=args.residual_codes,
        num_scale_levels=args.scale_levels,
        metric_weights=residual_metric.to(args.device),
        iterations=args.kmeans_iterations,
        seed=20260718,
        batch_size=256,
    ).to("cpu")
    artifacts["residual_rvq_fisher"] = _save_state(
        output_dir / "residual_rvq_fisher.pt",
        residual_fisher.state_dict(),
    )
    del residual_vectors, residual_unweighted
    torch.cuda.empty_cache()

    router_records = _router_records(
        cache_dir,
        calibration,
        base_codebook,
        residual_fisher,
        device=args.device,
    )
    router_artifacts, _ = _train_routers(
        router_records,
        output_dir,
        hidden_dim=args.router_hidden,
        epochs=args.router_epochs,
        device=args.device,
    )
    artifacts.update(router_artifacts)

    training_key_payload = json.dumps(training_keys, separators=(",", ":"))
    summary = {
        "format": TRAINING_FORMAT,
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": manifest_sha256(cache_dir),
        "split_counts": split_counts,
        "training_entries": len(calibration),
        "training_keys_sha256": hashlib.sha256(
            training_key_payload.encode("utf-8")
        ).hexdigest(),
        "evaluation_entries_loaded": 0,
        "base_train_vectors": args.base_train_vectors,
        "residual_train_blocks": args.residual_train_blocks,
        "base_codes": args.base_codes,
        "residual_codes": args.residual_codes,
        "scale_levels": args.scale_levels,
        "rvq_stages": args.rvq_stages,
        "kmeans_iterations": args.kmeans_iterations,
        "router_hidden": args.router_hidden,
        "router_epochs": args.router_epochs,
        "feature_names": BLOCK_FEATURE_NAMES,
        "fisher_summary": fisher_summary,
        "artifacts": artifacts,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
