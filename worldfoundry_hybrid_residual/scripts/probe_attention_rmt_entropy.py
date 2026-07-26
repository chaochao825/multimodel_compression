#!/usr/bin/env python3
"""Probe F81 attention diffuseness and Q/K random-matrix anisotropy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from probe_geometry_sparse_attention import (
    DEFAULT_SPECS,
    geometry_mask,
    grid_from_metadata,
    stratified_query_indices,
)


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-samples", type=int, default=128)
    parser.add_argument("--tile-height", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=8)
    parser.add_argument("--geometry-mask", default="s3_temporal_pm2")
    parser.add_argument("--topk", type=parse_ints, default=parse_ints("64,128,256,512,1024,2048,4096"))
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def covariance_spectrum(matrix: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    standardized = centered / scale
    observations, dimensions = standardized.shape
    covariance = standardized.T @ standardized / max(observations - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    aspect = dimensions / observations
    mp_lower = (1.0 - math.sqrt(aspect)) ** 2
    mp_upper = (1.0 + math.sqrt(aspect)) ** 2
    return eigenvalues, mp_lower, mp_upper


def probability_metrics(
    probabilities: torch.Tensor, topk: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    tokens = probabilities.shape[-1]
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    participation = probabilities.square().sum(dim=-1).reciprocal()
    metrics = {
        "normalized_entropy": entropy / math.log(tokens),
        "entropy_support_fraction": entropy.exp() / tokens,
        "participation_support_fraction": participation / tokens,
        "max_probability": probabilities.max(dim=-1).values,
    }
    for count in topk:
        used = min(count, tokens)
        metrics[f"top{count}_mass"] = torch.topk(
            probabilities, used, dim=-1, sorted=False
        ).values.sum(dim=-1)
    return metrics


def summarize_tensor(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flattened = values.float().flatten().cpu().tolist()
    return {
        f"{prefix}_mean": sum(flattened) / len(flattened),
        f"{prefix}_p05": quantile(flattened, 0.05),
        f"{prefix}_p50": quantile(flattened, 0.50),
        f"{prefix}_p95": quantile(flattened, 0.95),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.query_samples <= 0:
        raise ValueError("query samples must be positive")
    payload = torch.load(args.replay.resolve(), map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all = payload["q"][0]
    k_all = payload["k"][0]
    if q_all.shape != k_all.shape or q_all.ndim != 3:
        raise RuntimeError("expected matching Q/K tensors with shape [tokens, heads, dim]")
    tokens, heads, dimension = q_all.shape
    if max(args.topk) > tokens:
        raise ValueError(f"top-k request exceeds token count {tokens}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shape = grid_from_metadata(metadata, tokens, fallback_height=30, fallback_width=52)
    query_indices = stratified_query_indices(
        shape, args.query_samples, args.tile_height, args.tile_width, device
    )
    specs = {spec.name: spec for spec in DEFAULT_SPECS}
    if args.geometry_mask not in specs:
        raise ValueError(f"unknown geometry mask: {args.geometry_mask}")
    selected_mask = geometry_mask(
        query_indices,
        shape,
        specs[args.geometry_mask],
        args.tile_height,
        args.tile_width,
        anchor_phase=int(metadata.get("layer", 0)),
    )[0]
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    cpu_queries = query_indices.cpu()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    head_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    started = time.time()

    for head in range(heads):
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        q_eigen, q_mp_lower, q_mp_upper = covariance_spectrum(q)
        k_eigen, k_mp_lower, k_mp_upper = covariance_spectrum(k)
        sampled_q = q.index_select(0, query_indices)
        scores = sampled_q @ k.T * scale
        probabilities = torch.softmax(scores, dim=-1)
        actual = probability_metrics(probabilities, args.topk)

        row_mean = scores.mean(dim=-1, keepdim=True)
        row_std = scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
        gaussian_scores = torch.randn(
            scores.shape, generator=generator, device=device, dtype=torch.float32
        ) * row_std + row_mean
        gaussian = probability_metrics(torch.softmax(gaussian_scores, dim=-1), args.topk)
        geometry_mass = (probabilities * selected_mask).sum(dim=-1)

        row: dict[str, object] = {
            "head": head,
            "tokens": tokens,
            "head_dim": dimension,
            "query_samples": len(query_indices),
            "geometry_mask": args.geometry_mask,
            **summarize_tensor(geometry_mass, "geometry_mass"),
        }
        for name, values in actual.items():
            row.update(summarize_tensor(values, f"actual_{name}"))
        for name, values in gaussian.items():
            row.update(summarize_tensor(values, f"gaussian_{name}"))
        for name, eigenvalues, lower, upper in (
            ("q", q_eigen, q_mp_lower, q_mp_upper),
            ("k", k_eigen, k_mp_lower, k_mp_upper),
        ):
            total = float(eigenvalues.sum())
            largest = float(eigenvalues[-1])
            row[f"{name}_mp_lower"] = lower
            row[f"{name}_mp_upper"] = upper
            row[f"{name}_eigen_max"] = largest
            row[f"{name}_stable_rank"] = total / max(largest, 1e-30)
            row[f"{name}_outliers_above_mp"] = int((eigenvalues > upper).sum())
            row[f"{name}_top8_energy"] = float(eigenvalues[-8:].sum()) / max(total, 1e-30)
            for index, eigenvalue in enumerate(eigenvalues.cpu().tolist()):
                eigen_rows.append(
                    {
                        "head": head,
                        "source": name,
                        "ascending_index": index,
                        "eigenvalue": eigenvalue,
                        "mp_lower": lower,
                        "mp_upper": upper,
                    }
                )
        head_rows.append(row)
        del q, k, sampled_q, scores, probabilities, gaussian_scores

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "attention_rmt_entropy_heads.csv", head_rows)
    write_csv(args.output_dir / "attention_rmt_eigenvalues.csv", eigen_rows)

    numeric_fields = [
        key
        for key, value in head_rows[0].items()
        if isinstance(value, (int, float)) and key not in {"head", "tokens", "head_dim", "query_samples"}
    ]
    aggregate = {
        key: sum(float(row[key]) for row in head_rows) / len(head_rows)
        for key in numeric_fields
    }
    summary = {
        "replay": str(args.replay.resolve()),
        "metadata": metadata,
        "grid_size": shape,
        "tokens": tokens,
        "heads": heads,
        "head_dim": dimension,
        "query_samples": args.query_samples,
        "geometry_mask": args.geometry_mask,
        "topk": list(args.topk),
        "head_mean_metrics": aggregate,
        "interpretation_contract": {
            "mp_test": (
                "Correlation eigenvalues outside the Marchenko-Pastur band indicate "
                "channel anisotropy, not token-support sparsity."
            ),
            "gaussian_test": (
                "The matched Gaussian baseline preserves each query's logit mean/std; "
                "similar support statistics imply weak content-selective sparsity beyond scale."
            ),
            "scope": (
                "One layer/step/branch replay is a structural screen, not cross-trajectory "
                "or end-to-end quality evidence."
            ),
        },
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    (args.output_dir / "attention_rmt_entropy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
