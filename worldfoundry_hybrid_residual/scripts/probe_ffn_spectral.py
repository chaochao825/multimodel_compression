#!/usr/bin/env python3
"""Audit whether Wan and Llama FFN weights have deployable FFT structure.

The probe operates on contiguous square weight blocks. Complex FFT payloads
are charged by independent real degrees of freedom, so a conjugate pair costs
two scalars. It reports both oracle top-k coefficients and structured
low-frequency masks; only the latter has a plausible static kernel layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_WAN_KEYS = (
    "blocks.0.ffn.0.weight",
    "blocks.0.ffn.2.weight",
    "blocks.15.ffn.0.weight",
    "blocks.15.ffn.2.weight",
    "blocks.29.ffn.0.weight",
    "blocks.29.ffn.2.weight",
)
DEFAULT_LLAMA_KEYS = (
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.15.mlp.gate_proj.weight",
    "model.layers.15.mlp.up_proj.weight",
    "model.layers.15.mlp.down_proj.weight",
    "model.layers.31.mlp.gate_proj.weight",
    "model.layers.31.mlp.up_proj.weight",
    "model.layers.31.mlp.down_proj.weight",
)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("densities must be in (0, 1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-checkpoint", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--blocks-per-tensor", type=int, default=40)
    parser.add_argument(
        "--densities",
        type=parse_float_list,
        default=parse_float_list("0.015625,0.0625,0.125,0.25"),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_single_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def load_sharded_tensor(model_dir: Path, key: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"][key]
    return load_single_tensor(model_dir / shard, key)


def tensor_role(key: str) -> str:
    if ".ffn.0." in key or "gate_proj" in key or "up_proj" in key:
        return "expand"
    return "contract"


def sample_blocks(
    tensor: torch.Tensor, block_size: int, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    rows, cols = tensor.shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    if row_blocks == 0 or col_blocks == 0:
        raise ValueError(f"tensor {tuple(tensor.shape)} is smaller than block size")
    candidates = [(row, col) for row in range(row_blocks) for col in range(col_blocks)]
    selected = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    output = []
    for index in selected:
        row, col = candidates[int(index)]
        block = tensor[
            row * block_size : (row + 1) * block_size,
            col * block_size : (col + 1) * block_size,
        ]
        output.append(block.float().numpy().astype(np.float64, copy=False))
    return output


def matched_gaussian(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(block.mean(), block.std(), size=block.shape)


def smooth_control(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = block.shape[0]
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    value = np.zeros_like(block)
    for _ in range(12):
        ky = int(rng.integers(0, 4))
        kx = int(rng.integers(0, 4))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        amplitude = float(rng.normal())
        value += amplitude * np.cos(2.0 * math.pi * (ky * y + kx * x) / n + phase)
    value -= value.mean()
    value *= block.std() / max(value.std(), np.finfo(np.float64).tiny)
    return value + block.mean()


def shuffled(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return block[rng.permutation(block.shape[0])][:, rng.permutation(block.shape[1])]


def greedy_energy(
    energies: np.ndarray, costs: np.ndarray, budget: int, order: np.ndarray
) -> tuple[float, int]:
    retained = 0.0
    used = 0
    for raw_index in order:
        index = int(raw_index)
        cost = int(costs[index])
        if used + cost > budget:
            continue
        retained += float(energies[index])
        used += cost
        if used >= budget:
            break
    return retained, used


def identity_topk(block: np.ndarray, density: float) -> tuple[float, float]:
    energies = np.square(block).reshape(-1)
    budget = max(1, int(round(density * block.size)))
    indices = np.argpartition(energies, -budget)[-budget:]
    return float(energies[indices].sum() / energies.sum()), budget / block.size


def axis_fft_candidates(
    block: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = block.shape[axis]
    coeff = np.fft.fft(block, axis=axis, norm="ortho")
    energies: list[float] = []
    costs: list[int] = []
    frequencies: list[float] = []
    for other in range(n):
        for frequency in range(n // 2 + 1):
            index = (other, frequency) if axis == 1 else (frequency, other)
            self_conjugate = frequency == 0 or (n % 2 == 0 and frequency == n // 2)
            cost = 1 if self_conjugate else 2
            energies.append(float(abs(coeff[index]) ** 2) * cost)
            costs.append(cost)
            frequencies.append(frequency / n)
    return (
        np.asarray(energies),
        np.asarray(costs),
        np.asarray(frequencies),
        coeff,
    )


def axis_fft_energy(
    block: np.ndarray, density: float, axis: int, low_frequency: bool
) -> tuple[float, float]:
    energies, costs, frequencies, coeff = axis_fft_candidates(block, axis)
    budget = max(1, int(round(density * block.size)))
    if low_frequency:
        order = np.argsort(frequencies, kind="stable")
    else:
        order = np.argsort(-(energies / costs), kind="stable")
    retained, used = greedy_energy(energies, costs, budget, order)
    total = float(np.square(np.abs(coeff)).sum())
    return retained / total, used / block.size


def fft2_candidates(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = block.shape[0]
    coeff = np.fft.fft2(block, norm="ortho")
    visited: set[tuple[int, int]] = set()
    energies: list[float] = []
    costs: list[int] = []
    frequencies: list[float] = []
    for row in range(n):
        for col in range(n):
            index = (row, col)
            if index in visited:
                continue
            partner = ((-row) % n, (-col) % n)
            visited.add(index)
            visited.add(partner)
            self_conjugate = index == partner
            cost = 1 if self_conjugate else 2
            energy = float(abs(coeff[index]) ** 2)
            if not self_conjugate:
                energy += float(abs(coeff[partner]) ** 2)
            fy = min(row, n - row) / n
            fx = min(col, n - col) / n
            energies.append(energy)
            costs.append(cost)
            frequencies.append(math.hypot(fy, fx))
    return (
        np.asarray(energies),
        np.asarray(costs),
        np.asarray(frequencies),
        coeff,
    )


def fft2_energy(
    block: np.ndarray, density: float, low_frequency: bool
) -> tuple[float, float]:
    energies, costs, frequencies, coeff = fft2_candidates(block)
    budget = max(1, int(round(density * block.size)))
    if low_frequency:
        order = np.argsort(frequencies, kind="stable")
    else:
        order = np.argsort(-(energies / costs), kind="stable")
    retained, used = greedy_energy(energies, costs, budget, order)
    total = float(np.square(np.abs(coeff)).sum())
    return retained / total, used / block.size


def circulant_projection_energy(block: np.ndarray) -> float:
    n = block.shape[0]
    generator = np.empty(n, dtype=np.float64)
    for offset in range(n):
        generator[offset] = np.mean([block[row, (row + offset) % n] for row in range(n)])
    projection = np.empty_like(block)
    for row in range(n):
        projection[row] = np.roll(generator, row)
    return float(np.square(projection).sum() / np.square(block).sum())


def neighbor_cosine(block: np.ndarray, axis: int) -> float:
    vectors = block if axis == 0 else block.T
    left = vectors[:-1]
    right = vectors[1:]
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float64).tiny)))


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


def evaluate_block(block: np.ndarray, densities: tuple[float, ...]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for density in densities:
        methods = {
            "identity_topk_oracle": identity_topk(block, density),
            "row_fft_topk_oracle": axis_fft_energy(block, density, 1, False),
            "row_fft_lowfreq_static": axis_fft_energy(block, density, 1, True),
            "column_fft_topk_oracle": axis_fft_energy(block, density, 0, False),
            "column_fft_lowfreq_static": axis_fft_energy(block, density, 0, True),
            "fft2_topk_oracle": fft2_energy(block, density, False),
            "fft2_lowfreq_static": fft2_energy(block, density, True),
        }
        for method, (energy, actual_density) in methods.items():
            rows.append(
                {
                    "method": method,
                    "target_scalar_density": density,
                    "actual_scalar_density": actual_density,
                    "retained_energy": energy,
                    "relative_fro_error": math.sqrt(max(0.0, 1.0 - energy)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.block_size <= 1 or args.block_size % 2:
        raise ValueError("block size must be an even integer greater than one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sources = []
    for key in DEFAULT_WAN_KEYS:
        sources.append(("wan2.1-1.3b", key, load_single_tensor(args.wan_checkpoint, key)))
    for key in DEFAULT_LLAMA_KEYS:
        sources.append(("llama2-7b", key, load_sharded_tensor(args.llama_dir, key)))

    detailed: list[dict[str, object]] = []
    structure: list[dict[str, object]] = []
    started = time.time()
    for model, key, tensor in sources:
        blocks = sample_blocks(tensor, args.block_size, args.blocks_per_tensor, rng)
        del tensor
        for block_index, original in enumerate(blocks):
            variants = {
                "original": original,
                "rowcol_shuffled": shuffled(original, rng),
                "matched_gaussian": matched_gaussian(original, rng),
                "smooth_positive_control": smooth_control(original, rng),
            }
            for control, block in variants.items():
                common = {
                    "model": model,
                    "weight_key": key,
                    "role": tensor_role(key),
                    "block_index": block_index,
                    "control": control,
                    "block_size": args.block_size,
                }
                for row in evaluate_block(block, args.densities):
                    detailed.append({**common, **row})
                structure.append(
                    {
                        **common,
                        "row_neighbor_cosine": neighbor_cosine(block, 0),
                        "column_neighbor_cosine": neighbor_cosine(block, 1),
                        "circulant_retained_energy": circulant_projection_energy(block),
                        "circulant_random_expectation": 1.0 / args.block_size,
                    }
                )
        print(f"[spectral] {model} {key}: {len(blocks)} blocks", flush=True)

    write_csv(args.output_dir / "ffn_spectral_detail.csv", detailed)
    write_csv(args.output_dir / "ffn_structure_detail.csv", structure)
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": time.time() - started,
        "budget_definition": "independent real scalar degrees; non-self-conjugate FFT pairs cost two",
        "evidence_boundary": "sampled contiguous weight blocks; no activation or end-to-end quality claim",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[spectral] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
