from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mvbench_llava_anchor import write_json_atomic
from onevision_reader_quotient_stage_a import (
    FeatureStatistics,
    centered_covariance,
    descending_eigenspace,
    eigengap_summary,
    merge_statistics,
    subspace_squared_cosines,
    tail_energy_fraction,
)


@dataclass(frozen=True)
class DomainSpectrum:
    name: str
    sample_ids: tuple[str, ...]
    full_statistics: FeatureStatistics
    full_covariance: torch.Tensor
    full_eigenvalues: torch.Tensor
    full_basis: torch.Tensor
    bootstrap_statistics: dict[int, list[FeatureStatistics]]
    bootstrap_eigenvalues: dict[int, list[torch.Tensor]]
    bootstrap_bases: dict[int, list[torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help="Domain feature directory as NAME=PATH.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--sample-sizes", default="20,60,120")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool-pair", default="")
    return parser.parse_args()


def parse_domains(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid domain specification: {value}")
        if name in output:
            raise ValueError(f"duplicate domain name: {name}")
        output[name] = Path(raw_path)
    return output


def parse_sample_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not sizes or sizes[0] <= 0:
        raise ValueError("sample sizes must be positive")
    return sizes


def bootstrap_multiplicities(
    *,
    video_count: int,
    sample_sizes: tuple[int, ...],
    replicates: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = {}
    for sample_size in sample_sizes:
        counts = np.zeros((replicates, video_count), dtype=np.int16)
        for replicate in range(replicates):
            indices = rng.choice(video_count, size=sample_size, replace=True)
            counts[replicate] = np.bincount(indices, minlength=video_count)
        output[sample_size] = counts
    return output


def empty_statistics(dimension: int, *, device: torch.device) -> FeatureStatistics:
    return FeatureStatistics(
        rows=0,
        feature_sum=torch.zeros(dimension, device=device, dtype=torch.float32),
        gram=torch.zeros(
            (dimension, dimension),
            device=device,
            dtype=torch.float32,
        ),
    )


def add_statistics(
    destination: FeatureStatistics,
    source: FeatureStatistics,
    *,
    multiplicity: int = 1,
) -> FeatureStatistics:
    return FeatureStatistics(
        rows=destination.rows + source.rows * multiplicity,
        feature_sum=destination.feature_sum + source.feature_sum * multiplicity,
        gram=destination.gram + source.gram * multiplicity,
    )


def load_domain_spectrum(
    name: str,
    feature_dir: Path,
    *,
    rank: int,
    sample_sizes: tuple[int, ...],
    replicates: int,
    seed: int,
    device: torch.device,
) -> DomainSpectrum:
    paths = sorted(feature_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no feature payloads found in {feature_dir}")
    if sample_sizes[-1] > len(paths):
        raise ValueError(
            f"domain {name} has {len(paths)} videos, fewer than sample size {sample_sizes[-1]}"
        )
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    dimension = int(first["features"].shape[-1])
    multiplicities = bootstrap_multiplicities(
        video_count=len(paths),
        sample_sizes=sample_sizes,
        replicates=replicates,
        seed=seed,
    )
    full = empty_statistics(dimension, device=device)
    bootstraps = {
        sample_size: [
            empty_statistics(dimension, device=device) for _ in range(replicates)
        ]
        for sample_size in sample_sizes
    }
    sample_ids = []
    for video_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        features = payload["features"].to(device=device, dtype=torch.float32)
        matrix = features.reshape(-1, dimension)
        statistics = FeatureStatistics(
            rows=matrix.shape[0],
            feature_sum=matrix.sum(dim=0),
            gram=matrix.transpose(0, 1) @ matrix,
        )
        full = add_statistics(full, statistics)
        for sample_size in sample_sizes:
            for replicate in range(replicates):
                multiplicity = int(multiplicities[sample_size][replicate, video_index])
                if multiplicity:
                    bootstraps[sample_size][replicate] = add_statistics(
                        bootstraps[sample_size][replicate],
                        statistics,
                        multiplicity=multiplicity,
                    )
        sample_ids.append(str(payload["sample_id"]))
        del features, matrix, statistics, payload

    full_covariance = centered_covariance(full)
    full_eigenvalues, full_basis = descending_eigenspace(
        full_covariance,
        rank=rank,
    )
    bootstrap_eigenvalues = {sample_size: [] for sample_size in sample_sizes}
    bootstrap_bases = {sample_size: [] for sample_size in sample_sizes}
    for sample_size in sample_sizes:
        for statistics in bootstraps[sample_size]:
            eigenvalues, basis = descending_eigenspace(
                centered_covariance(statistics),
                rank=rank,
            )
            bootstrap_eigenvalues[sample_size].append(eigenvalues.cpu())
            bootstrap_bases[sample_size].append(basis.cpu())
    return DomainSpectrum(
        name=name,
        sample_ids=tuple(sample_ids),
        full_statistics=FeatureStatistics(
            rows=full.rows,
            feature_sum=full.feature_sum.cpu(),
            gram=full.gram.cpu(),
        ),
        full_covariance=full_covariance.cpu(),
        full_eigenvalues=full_eigenvalues.cpu(),
        full_basis=full_basis.cpu(),
        bootstrap_statistics={
            sample_size: [
                FeatureStatistics(
                    rows=value.rows,
                    feature_sum=value.feature_sum.cpu(),
                    gram=value.gram.cpu(),
                )
                for value in values
            ]
            for sample_size, values in bootstraps.items()
        },
        bootstrap_eigenvalues=bootstrap_eigenvalues,
        bootstrap_bases=bootstrap_bases,
    )


def basis_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    squared_cosines = subspace_squared_cosines(left, right)
    return {
        "overlap": float(squared_cosines.mean().item()),
        "minimum_squared_cosine": float(squared_cosines.min().item()),
        "median_squared_cosine": float(squared_cosines.median().item()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    domains = parse_domains(args.domain)
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spectra = {}
    for domain_index, (name, feature_dir) in enumerate(domains.items()):
        spectra[name] = load_domain_spectrum(
            name,
            feature_dir,
            rank=args.rank,
            sample_sizes=sample_sizes,
            replicates=args.replicates,
            seed=args.seed + 1009 * domain_index,
            device=device,
        )

    domain_rows = []
    bootstrap_rows = []
    for name, spectrum in spectra.items():
        eigengap = eigengap_summary(spectrum.full_eigenvalues, rank=args.rank)
        domain_rows.append(
            {
                "domain": name,
                "videos": len(spectrum.sample_ids),
                "rows": spectrum.full_statistics.rows,
                "rank": args.rank,
                "tail_energy_fraction": tail_energy_fraction(
                    spectrum.full_eigenvalues,
                    rank=args.rank,
                ),
                "absolute_eigengap": eigengap["absolute_gap"],
                "relative_eigengap": eigengap["relative_gap"],
            }
        )
        for sample_size in sample_sizes:
            for replicate, basis in enumerate(spectrum.bootstrap_bases[sample_size]):
                comparison = basis_comparison(spectrum.full_basis, basis)
                bootstrap_rows.append(
                    {
                        "domain": name,
                        "sample_size": sample_size,
                        "replicate": replicate,
                        **comparison,
                    }
                )

    pair_rows = []
    names = tuple(spectra)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pair_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "sample_size": "full",
                    "replicate": -1,
                    **basis_comparison(
                        spectra[left_name].full_basis,
                        spectra[right_name].full_basis,
                    ),
                }
            )
            for sample_size in sample_sizes:
                for replicate in range(args.replicates):
                    pair_rows.append(
                        {
                            "left": left_name,
                            "right": right_name,
                            "sample_size": sample_size,
                            "replicate": replicate,
                            **basis_comparison(
                                spectra[left_name].bootstrap_bases[sample_size][replicate],
                                spectra[right_name].bootstrap_bases[sample_size][replicate],
                            ),
                        }
                    )

    pooled_rows = []
    if args.pool_pair:
        left_name, separator, right_name = args.pool_pair.partition(",")
        if not separator or left_name not in spectra or right_name not in spectra:
            raise ValueError("pool pair must name two loaded domains")
        for sample_size in sample_sizes:
            for replicate in range(args.replicates):
                statistics = merge_statistics(
                    [
                        spectra[left_name].bootstrap_statistics[sample_size][replicate],
                        spectra[right_name].bootstrap_statistics[sample_size][replicate],
                    ]
                )
                eigenvalues, basis = descending_eigenspace(
                    centered_covariance(
                        FeatureStatistics(
                            rows=statistics.rows,
                            feature_sum=statistics.feature_sum.to(device),
                            gram=statistics.gram.to(device),
                        )
                    ),
                    rank=args.rank,
                )
                pooled_rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "per_domain_sample_size": sample_size,
                        "total_sample_size": 2 * sample_size,
                        "replicate": replicate,
                        "tail_energy_fraction": tail_energy_fraction(
                            eigenvalues.cpu(),
                            rank=args.rank,
                        ),
                        "overlap_with_left": basis_comparison(
                            basis.cpu(), spectra[left_name].full_basis
                        )["overlap"],
                        "overlap_with_right": basis_comparison(
                            basis.cpu(), spectra[right_name].full_basis
                        )["overlap"],
                    }
                )

    write_csv(args.out_dir / "domain_spectra.csv", domain_rows)
    write_csv(args.out_dir / "bootstrap_stability.csv", bootstrap_rows)
    write_csv(args.out_dir / "cross_domain_overlap.csv", pair_rows)
    write_csv(args.out_dir / "pooled_spectra.csv", pooled_rows)
    artifact = {
        name: {
            "sample_ids": spectrum.sample_ids,
            "full_mean": spectrum.full_statistics.feature_sum
            / spectrum.full_statistics.rows,
            "full_covariance": spectrum.full_covariance,
            "full_eigenvalues": spectrum.full_eigenvalues,
            "full_basis": spectrum.full_basis,
            "bootstrap_eigenvalues": spectrum.bootstrap_eigenvalues,
            "bootstrap_bases": spectrum.bootstrap_bases,
        }
        for name, spectrum in spectra.items()
    }
    torch.save(artifact, args.out_dir / "spectral_artifacts.pt")
    summary = {
        "rank": args.rank,
        "sample_sizes": sample_sizes,
        "replicates": args.replicates,
        "seed": args.seed,
        "domain_spectra": domain_rows,
        "cross_domain_full": [
            row for row in pair_rows if row["sample_size"] == "full"
        ],
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
