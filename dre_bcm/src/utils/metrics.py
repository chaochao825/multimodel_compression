import math
from typing import Dict

import numpy as np
import torch


def relative_fro_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    numerator = torch.linalg.norm(target - approx)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return (numerator / denominator).item()


def spectral_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    delta = target - approx
    numerator = torch.linalg.matrix_norm(delta, ord=2)
    denominator = torch.linalg.matrix_norm(target, ord=2).clamp_min(1e-12)
    return (numerator / denominator).item()


def energy_ratio(singular_values: torch.Tensor, rank: int) -> float:
    if singular_values.numel() == 0:
        return 0.0
    rank = min(rank, singular_values.numel())
    total = singular_values.square().sum().clamp_min(1e-12)
    kept = singular_values[:rank].square().sum()
    return (kept / total).item()


def quantize_symmetric(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    levels = 2 ** (num_bits - 1) - 1
    scale = values.abs().max().clamp_min(1e-12) / levels
    q = torch.round(values / scale).clamp(-levels - 1, levels)
    return q * scale


def empirical_entropy(values: torch.Tensor, bins: int = 256) -> float:
    flat = values.detach().float().reshape(-1).cpu().numpy()
    if flat.size == 0:
        return 0.0
    min_value = float(flat.min())
    max_value = float(flat.max())
    if math.isclose(min_value, max_value):
        return 0.0
    hist, _ = np.histogram(flat, bins=bins, range=(min_value, max_value), density=False)
    probs = hist.astype(np.float64)
    probs = probs[probs > 0]
    probs /= probs.sum()
    return float(-(probs * np.log2(probs)).sum())


def summarize_tensor(values: torch.Tensor) -> Dict[str, float]:
    tensor = values.detach().float().reshape(-1)
    return {
        "mean": tensor.mean().item(),
        "median": tensor.median().item(),
        "std": tensor.std(unbiased=False).item(),
        "max_abs": tensor.abs().max().item(),
    }
