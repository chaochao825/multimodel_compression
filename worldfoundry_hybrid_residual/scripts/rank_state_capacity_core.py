#!/usr/bin/env python3
"""Numerical core for the EXP-046 target-visible rank-state capacity Gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RankStateSpectrum:
    singular_values: torch.Tensor
    total_energy: float
    rows: int
    channels: int

    def error_sq(self, rank: int) -> float:
        if rank <= 0 or rank > self.singular_values.numel():
            raise ValueError("rank lies outside the computed spectrum")
        captured = float(self.singular_values[:rank].double().square().sum())
        return max(self.total_energy - captured, 0.0)


@torch.inference_mode()
def randomized_rank_state_spectrum(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> RankStateSpectrum:
    """Compute a deterministic randomized SVD spectrum without centering."""

    if matrix.ndim != 2:
        raise ValueError("rank-state matrix must be two-dimensional")
    if max_rank <= 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("invalid randomized SVD settings")
    rows, channels = matrix.shape
    available_rank = min(rows, channels)
    if max_rank > available_rank:
        raise ValueError("max_rank exceeds the matrix dimensions")

    work = matrix.float()
    sketch_width = min(max_rank + oversample, available_rank)
    generator = torch.Generator(device=work.device).manual_seed(seed)
    omega = torch.randn(
        channels,
        sketch_width,
        device=work.device,
        dtype=work.dtype,
        generator=generator,
    )
    basis, _ = torch.linalg.qr(work @ omega, mode="reduced")
    for _ in range(power_iterations):
        right_basis, _ = torch.linalg.qr(work.T @ basis, mode="reduced")
        basis, _ = torch.linalg.qr(work @ right_basis, mode="reduced")
    reduced = basis.T @ work
    singular_values = torch.linalg.svdvals(reduced)[:max_rank]
    return RankStateSpectrum(
        singular_values=singular_values,
        total_energy=float(work.double().square().sum()),
        rows=rows,
        channels=channels,
    )


def state_capacity_rows(
    spectrum: RankStateSpectrum,
    *,
    ranks: tuple[int, ...],
    residual_target_sq: float,
    output_target_sq: float,
    estimated_exact_block_macs: int,
) -> list[dict[str, float | int]]:
    if tuple(sorted(set(ranks))) != ranks:
        raise ValueError("ranks must be strictly increasing")
    if ranks[-1] > spectrum.singular_values.numel():
        raise ValueError("requested rank was not computed")
    if residual_target_sq <= 0 or output_target_sq <= 0:
        raise ValueError("target energies must be positive")
    rows: list[dict[str, float | int]] = []
    previous_error = float("inf")
    for rank in ranks:
        error_sq = spectrum.error_sq(rank)
        if error_sq > previous_error * (1 + 1e-6):
            raise RuntimeError("rank-state approximation error is not monotonic")
        previous_error = error_sq
        render_macs = 2 * spectrum.rows * spectrum.channels * rank
        rows.append(
            {
                "rank": rank,
                "error_sq": error_sq,
                "residual_relative_l2": (
                    error_sq / max(residual_target_sq, 1e-30)
                )
                ** 0.5,
                "output_relative_l2": (
                    error_sq / max(output_target_sq, 1e-30)
                )
                ** 0.5,
                "defect_remaining_energy": (
                    error_sq / max(spectrum.total_energy, 1e-30)
                ),
                "state_factor_values": rank * (spectrum.rows + spectrum.channels),
                "render_macs": render_macs,
                "render_to_exact_macs": render_macs
                / max(estimated_exact_block_macs, 1),
            }
        )
    return rows


def estimated_wan_block_macs(
    *, tokens: int, hidden_size: int, ffn_size: int
) -> int:
    qkvo = 4 * tokens * hidden_size * hidden_size
    ffn = 2 * tokens * hidden_size * ffn_size
    attention = 2 * tokens * tokens * hidden_size
    return qkvo + ffn + attention
