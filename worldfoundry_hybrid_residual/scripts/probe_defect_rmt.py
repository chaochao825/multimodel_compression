#!/usr/bin/env python3
"""Test whether Wan activation defects contain stable spiked-covariance structure."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--null-repeats", type=int, default=4)
    parser.add_argument("--stability-rank", type=int, default=16)
    parser.add_argument("--include-block-groups", action="store_true")
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def operator_name(record: dict[str, object]) -> str:
    kind = str(record["kind"])
    if kind != "C_FORECAST":
        return kind
    return f"C_FORECAST_{float(record['forecast_scale']):g}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mp_upper_edge(mean_variance: float, features: int, samples: int) -> float:
    if mean_variance < 0.0 or features <= 0 or samples <= 1:
        raise ValueError("invalid Marchenko-Pastur parameters")
    gamma = features / samples
    return mean_variance * (1.0 + math.sqrt(gamma)) ** 2


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return normalized projection overlap in [0, 1]."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("subspace bases must have the same matrix shape")
    rank = left.shape[1]
    if rank <= 0:
        raise ValueError("subspace rank must be positive")
    return float((left.T @ right).square().sum().item() / rank)


def sample_matrix(
    records: list[dict[str, object]], max_rows: int, generator: torch.Generator
) -> torch.Tensor:
    matrices = []
    for record in records:
        sample = record.get("sample")
        if not isinstance(sample, torch.Tensor):
            raise TypeError("defect record sample must be a tensor")
        matrices.append(sample.float())
    matrix = torch.cat(matrices, dim=0)
    if matrix.shape[0] > max_rows:
        indices = torch.randperm(matrix.shape[0], generator=generator)[:max_rows]
        matrix = matrix.index_select(0, indices)
    return matrix


def normalize_matrix(matrix: torch.Tensor, variant: str) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if variant == "centered":
        return centered
    if variant == "channel_standardized":
        scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min_(1e-8)
        return centered / scale
    raise ValueError(f"unknown normalization variant: {variant}")


def covariance_eigensystem(
    matrix: torch.Tensor, device: torch.device, eigenvectors: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    work = matrix.to(device=device, dtype=torch.float32)
    covariance = work.T @ work / max(1, work.shape[0] - 1)
    if eigenvectors:
        values, vectors = torch.linalg.eigh(covariance)
        return values.flip(0), vectors.flip(1)
    return torch.linalg.eigvalsh(covariance).flip(0), None


def permuted_null_maxima(
    matrix: torch.Tensor,
    device: torch.device,
    repeats: int,
    generator: torch.Generator,
) -> list[float]:
    if repeats <= 0:
        return []
    work = matrix.to(device=device, dtype=torch.float32)
    rows, features = work.shape
    base = torch.arange(rows, device=device).unsqueeze(1)
    maxima = []
    for _ in range(repeats):
        shifts = torch.randint(
            rows, (1, features), generator=generator, device="cpu"
        ).to(device)
        indices = (base + shifts) % rows
        permuted = torch.gather(work, 0, indices)
        covariance = permuted.T @ permuted / max(1, rows - 1)
        maxima.append(float(torch.linalg.eigvalsh(covariance)[-1].item()))
    return maxima


def top_subspace(
    matrix: torch.Tensor, rank: int, device: torch.device
) -> torch.Tensor:
    _, vectors = covariance_eigensystem(matrix, device, eigenvectors=True)
    if vectors is None:
        raise RuntimeError("eigenvectors were not produced")
    return vectors[:, : min(rank, vectors.shape[1])]


def main() -> None:
    args = parse_args()
    if args.max_rows <= 1 or args.null_repeats < 0 or args.stability_rank <= 0:
        raise ValueError("invalid row, null-repeat, or rank setting")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError("sample file contains no defect records")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        operator = operator_name(record)
        groups[(operator, "all_blocks")].append(record)
        if args.include_block_groups:
            groups[(operator, f"block_{int(record['block']):02d}")].append(record)

    summary_rows: list[dict[str, object]] = []
    eigenvalue_rows: list[dict[str, object]] = []
    seed_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for (operator, group), group_records in sorted(groups.items()):
        base = sample_matrix(group_records, args.max_rows, seed_generator)
        for variant in ("centered", "channel_standardized"):
            matrix = normalize_matrix(base, variant)
            eigenvalues, _ = covariance_eigensystem(matrix, device, eigenvectors=False)
            eigenvalues_cpu = eigenvalues.cpu().double()
            samples, features = matrix.shape
            mean_variance = float(eigenvalues_cpu.sum().item() / features)
            theoretical_edge = mp_upper_edge(mean_variance, features, samples)
            null_values = permuted_null_maxima(
                matrix, device, args.null_repeats, seed_generator
            )
            empirical_edge = (
                float(np.quantile(np.asarray(null_values), 0.95))
                if null_values
                else float("nan")
            )
            threshold = max(
                theoretical_edge,
                empirical_edge if math.isfinite(empirical_edge) else theoretical_edge,
            )
            spike_mask = eigenvalues_cpu > threshold
            total_energy = float(eigenvalues_cpu.sum().item())
            spike_energy = float(eigenvalues_cpu[spike_mask].sum().item())
            cumulative = torch.cumsum(eigenvalues_cpu, dim=0) / max(total_energy, 1e-30)

            run_bases = []
            by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
            for record in group_records:
                by_run[str(record["run_id"])].append(record)
            for run_id, run_records in sorted(by_run.items()):
                run_generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + sum(run_id.encode("utf-8"))
                )
                run_matrix = normalize_matrix(
                    sample_matrix(run_records, args.max_rows, run_generator), variant
                )
                run_bases.append(top_subspace(run_matrix, args.stability_rank, device))
            overlaps = [
                subspace_overlap(left, right)
                for left, right in itertools.combinations(run_bases, 2)
                if left.shape == right.shape
            ]

            summary_rows.append(
                {
                    "operator": operator,
                    "group": group,
                    "normalization": variant,
                    "records": len(group_records),
                    "runs": len(by_run),
                    "sampled_rows": samples,
                    "features": features,
                    "aspect_ratio_features_per_sample": features / samples,
                    "mean_variance": mean_variance,
                    "mp_upper_edge": theoretical_edge,
                    "empirical_null_p95_max": empirical_edge,
                    "spike_threshold": threshold,
                    "spike_count": int(spike_mask.sum().item()),
                    "spike_energy_ratio": spike_energy / max(total_energy, 1e-30),
                    "energy_rank_8": float(cumulative[min(7, features - 1)].item()),
                    "energy_rank_16": float(cumulative[min(15, features - 1)].item()),
                    "energy_rank_32": float(cumulative[min(31, features - 1)].item()),
                    "subspace_rank": min(args.stability_rank, features),
                    "subspace_overlap_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
                    "subspace_overlap_min": float(np.min(overlaps)) if overlaps else float("nan"),
                    "null_maxima": ";".join(f"{value:.9g}" for value in null_values),
                }
            )
            for index, value in enumerate(eigenvalues_cpu.tolist(), start=1):
                eigenvalue_rows.append(
                    {
                        "operator": operator,
                        "group": group,
                        "normalization": variant,
                        "index": index,
                        "eigenvalue": value,
                        "above_spike_threshold": int(value > threshold),
                    }
                )
            print(
                f"[defect-rmt] operator={operator} group={group} variant={variant} "
                f"spikes={int(spike_mask.sum().item())} "
                f"spike_energy={spike_energy / max(total_energy, 1e-30):.4f} "
                f"stability={float(np.mean(overlaps)) if overlaps else float('nan'):.4f}",
                flush=True,
            )

    write_csv(out_dir / "defect_rmt_summary.csv", summary_rows)
    write_csv(out_dir / "defect_rmt_eigenvalues.csv", eigenvalue_rows)
    manifest = {
        "scope": "RMT screen of token-row sampled Wan activation defects",
        "samples": str(args.samples.resolve()),
        "records": len(records),
        "arguments": vars(args) | {"samples": str(args.samples), "out_dir": str(args.out_dir)},
        "method": {
            "mp_edge": "mean covariance variance times (1 + sqrt(features / samples))^2",
            "empirical_null": "independent circular row shift per channel, 95th percentile of maximum eigenvalue",
            "stability": "mean normalized projection overlap of top-r defect covariance subspaces across run_id",
        },
        "warning": (
            "MP assumptions are approximate because video-token rows are correlated. "
            "Spikes are candidates for protection, not evidence that the remaining bulk is removable."
        ),
    }
    (out_dir / "defect_rmt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
