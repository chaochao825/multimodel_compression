#!/usr/bin/env python3
"""Compare Wan FFN singular spectra with MP controls and mixed-bit splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_KEYS = ",".join(
    f"blocks.{block}.ffn.{linear}.weight"
    for block in (0, 15, 29)
    for linear in (0, 2)
)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("ranks must be nonnegative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keys", default=DEFAULT_KEYS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--spectrum-ranks",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16,32,64,128,256"),
    )
    parser.add_argument(
        "--split-ranks", type=parse_int_list, default=parse_int_list("0,8,16,32,64")
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


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


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected one safetensors file under {path}, found {len(candidates)}")
    return candidates[0]


def orient(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.shape[0] <= weight.shape[1]:
        return weight, False
    return weight.T.contiguous(), True


def biwhiten(matrix: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    result = matrix.clone()
    target_rms = matrix.square().mean().sqrt().clamp_min(1e-12)
    for _ in range(iterations):
        result = result / result.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        result = result / result.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    return result * target_rms


def normalized_eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    variance = matrix.square().mean().clamp_min(1e-30)
    gram = matrix @ matrix.T / (matrix.shape[1] * variance)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    return eigenvalues.clamp_min(0.0), eigenvectors


def effective_rank(eigenvalues: torch.Tensor) -> float:
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def groupwise_int4(matrix: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    columns = matrix.shape[1]
    padded_columns = math.ceil(columns / group_size) * group_size
    if padded_columns != columns:
        padded = torch.nn.functional.pad(matrix, (0, padded_columns - columns))
    else:
        padded = matrix
    grouped = padded.reshape(padded.shape[0], -1, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 7.0
    quantized = torch.round(grouped / scales).clamp(-7, 7) * scales
    restored = quantized.reshape(padded.shape)[:, :columns]
    return restored, scales.numel()


def tensor_fp8(matrix: torch.Tensor) -> torch.Tensor:
    scale = matrix.abs().amax().clamp_min(1e-12) / 448.0
    normalized = (matrix / scale).clamp(-448.0, 448.0)
    return normalized.to(torch.float8_e4m3fn).float() * scale


def relative_output_error(
    weight: torch.Tensor, estimate: torch.Tensor, inputs: torch.Tensor
) -> float:
    reference = inputs @ weight.T
    output = inputs @ estimate.T
    return float((output - reference).norm() / reference.norm().clamp_min(1e-30))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint)
    keys = tuple(item.strip() for item in args.keys.split(",") if item.strip())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    spectrum_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    quant_rows: list[dict[str, object]] = []
    started = time.time()

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        missing = sorted(set(keys) - set(handle.keys()))
        if missing:
            raise KeyError(f"checkpoint is missing keys: {missing}")

        for key_index, key in enumerate(keys):
            weight = handle.get_tensor(key).to(device=device, dtype=torch.float32)
            matrix, transposed = orient(weight)
            centered = matrix - matrix.mean()
            permutation = torch.randperm(centered.numel(), generator=generator, device=device)
            shuffled = centered.flatten().index_select(0, permutation).reshape_as(centered)
            gaussian = torch.randn(
                centered.shape, generator=generator, device=device, dtype=centered.dtype
            ) * centered.std(unbiased=False)
            variants = {
                "original_centered": centered,
                "entry_shuffled": shuffled,
                "matched_gaussian": gaussian,
                "biwhitened": biwhiten(centered),
            }

            for variant, candidate in variants.items():
                eigenvalues, _ = normalized_eigensystem(candidate)
                descending = eigenvalues.flip(0)
                cumulative = descending.cumsum(0) / descending.sum().clamp_min(1e-30)
                aspect = candidate.shape[0] / candidate.shape[1]
                mp_lower = (1.0 - math.sqrt(aspect)) ** 2
                mp_upper = (1.0 + math.sqrt(aspect)) ** 2
                above = int((eigenvalues > mp_upper * 1.05).sum().item())
                below = int((eigenvalues < mp_lower * 0.95).sum().item())
                stable_rank = float(eigenvalues.sum() / eigenvalues.max().clamp_min(1e-30))
                summary_rows.append(
                    {
                        "weight_key": key,
                        "variant": variant,
                        "rows": weight.shape[0],
                        "columns": weight.shape[1],
                        "oriented_rows": candidate.shape[0],
                        "oriented_columns": candidate.shape[1],
                        "aspect": aspect,
                        "mp_lower": mp_lower,
                        "mp_upper": mp_upper,
                        "eigen_min": float(eigenvalues.min()),
                        "eigen_median": float(eigenvalues.median()),
                        "eigen_max": float(eigenvalues.max()),
                        "above_mp_upper_5pct": above,
                        "below_mp_lower_5pct": below,
                        "stable_rank": stable_rank,
                        "effective_rank_entropy": effective_rank(eigenvalues),
                    }
                )
                for rank in args.spectrum_ranks:
                    if rank > cumulative.numel():
                        continue
                    spectrum_rows.append(
                        {
                            "weight_key": key,
                            "variant": variant,
                            "rank": rank,
                            "energy_ratio": float(cumulative[rank - 1]),
                        }
                    )
                for index, value in enumerate(descending.tolist(), start=1):
                    spectrum_rows.append(
                        {
                            "weight_key": key,
                            "variant": variant,
                            "rank": index,
                            "normalized_eigenvalue": value,
                            "record_kind": "full_spectrum",
                        }
                    )

            raw_eigenvalues, raw_vectors = torch.linalg.eigh(matrix @ matrix.T)
            raw_vectors = raw_vectors.flip(1)
            inputs = torch.randn(
                args.activation_samples,
                weight.shape[1],
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            fp8_estimate = tensor_fp8(weight)
            quant_rows.append(
                {
                    "weight_key": key,
                    "method": "fp8_tensor",
                    "rank": 0,
                    "weight_relative_fro": float((fp8_estimate - weight).norm() / weight.norm()),
                    "gaussian_output_relative_l2": relative_output_error(weight, fp8_estimate, inputs),
                    "stored_bits_ratio_vs_bf16": 0.5,
                }
            )

            for rank in args.split_ranks:
                if rank > matrix.shape[0]:
                    continue
                if rank == 0:
                    methods = [("int4_groupwise", torch.zeros_like(matrix))]
                else:
                    top_basis = raw_vectors[:, :rank]
                    spectral = (top_basis @ (top_basis.T @ matrix)).to(torch.bfloat16).float()
                    random_basis, _ = torch.linalg.qr(
                        torch.randn(
                            matrix.shape[0],
                            rank,
                            generator=generator,
                            device=device,
                            dtype=matrix.dtype,
                        ),
                        mode="reduced",
                    )
                    random_component = (
                        random_basis @ (random_basis.T @ matrix)
                    ).to(torch.bfloat16).float()
                    methods = [
                        ("spectral_fp16_plus_int4", spectral),
                        ("random_fp16_plus_int4", random_component),
                    ]
                for method, protected in methods:
                    quantized_residual, scale_count = groupwise_int4(
                        matrix - protected, args.group_size
                    )
                    estimate_matrix = protected + quantized_residual
                    estimate_weight = estimate_matrix.T.contiguous() if transposed else estimate_matrix
                    lowrank_bits = 16 * rank * (matrix.shape[0] + matrix.shape[1])
                    main_bits = 4 * matrix.numel() + 16 * scale_count
                    quant_rows.append(
                        {
                            "weight_key": key,
                            "method": method,
                            "rank": rank,
                            "weight_relative_fro": float(
                                (estimate_weight - weight).norm() / weight.norm().clamp_min(1e-30)
                            ),
                            "gaussian_output_relative_l2": relative_output_error(
                                weight, estimate_weight, inputs
                            ),
                            "stored_bits_ratio_vs_bf16": (lowrank_bits + main_bits)
                            / (16 * matrix.numel()),
                            "group_size": args.group_size,
                            "scale_count": scale_count,
                        }
                    )

            print(f"[mp] {key}", flush=True)
            del weight, matrix, centered, shuffled, gaussian, variants, permutation
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "weight_mp_summary.csv", summary_rows)
    write_csv(args.output_dir / "weight_spectrum.csv", spectrum_rows)
    write_csv(args.output_dir / "weight_mixed_bit.csv", quant_rows)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "checkpoint": str(checkpoint),
        "methodology": {
            "mp_normalization": "eigenvalues of AA^T / (q * element_variance), p<=q",
            "mp_outlier_gate": "outside theoretical edge by a 5 percent margin",
            "biwhitening": "five alternating row/column RMS normalizations",
            "mixed_bit": "BF16 rank-r left singular projection plus row-group INT4 residual",
            "random_control": "same-rank random orthonormal left subspace",
            "warning": "Gaussian activation error is a matrix proxy, not diffusion rollout quality",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
        "weights": len(keys),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[mp] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
