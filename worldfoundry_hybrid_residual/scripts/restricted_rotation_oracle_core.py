#!/usr/bin/env python3
"""Core transforms for the held-out restricted-rotation capacity oracle.

Every transform acts on an output-channel basis with shape ``[..., d, r]``.
The fitting objective is the actual sparse-attention output defect, not an
unweighted distance between subspaces.  These routines intentionally expose
post-hoc capacity only; they do not predict transform parameters from Q/K/V.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RotationCost:
    family: str
    generators: int
    dynamic_scalars: int
    rotation_macs: int
    tail_macs: int
    dense_attention_macs: int

    @property
    def work_ratio(self) -> float:
        return (self.rotation_macs + self.tail_macs) / self.dense_attention_macs


def orthonormalize(basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim < 2:
        raise ValueError("basis must have shape [..., channels, rank]")
    if basis.shape[-2] < basis.shape[-1]:
        raise ValueError("basis channel dimension must be at least its rank")
    q, r = torch.linalg.qr(basis, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * signs.unsqueeze(-2)


def right_singular_basis(defect: torch.Tensor, rank: int) -> torch.Tensor:
    if defect.ndim != 2:
        raise ValueError("defect must be a matrix")
    if rank <= 0 or rank > min(defect.shape):
        raise ValueError(f"rank must be in [1, {min(defect.shape)}]")
    covariance = defect.T @ defect
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    return eigenvectors[:, order[:rank]].contiguous()


def residual_after_basis(defect: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return defect - (defect @ basis) @ basis.transpose(-2, -1)


def batch_residual_squares(defects: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    if defects.ndim != 3 or bases.ndim != 3:
        raise ValueError("batched defects and bases must be rank-3")
    coefficients = torch.einsum("bqd,bdr->bqr", defects, bases)
    reconstructed = torch.einsum("bqr,bdr->bqd", coefficients, bases)
    return (defects - reconstructed).square().sum(dim=(1, 2))


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("subspace bases must have identical shapes")
    rank = left.shape[-1]
    cross = left.transpose(-2, -1) @ right
    return cross.square().sum(dim=(-2, -1)) / rank


def align_target_frame(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resolve the arbitrary within-subspace orientation of ``target``."""

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be matching rank-2 bases")
    u, _, vh = torch.linalg.svd(source.T @ target)
    frame_rotation = vh.T @ u.T
    return target @ frame_rotation


def apply_givens(basis: torch.Tensor, first: int, second: int, angle: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("Givens helper expects one rank-2 basis")
    if first == second:
        raise ValueError("Givens channel indices must differ")
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    first_row = basis[first].clone()
    second_row = basis[second].clone()
    output = basis.clone()
    output[first] = cosine * first_row - sine * second_row
    output[second] = sine * first_row + cosine * second_row
    return output


def greedy_givens_prefixes(
    source: torch.Tensor,
    defect: torch.Tensor,
    generator_counts: tuple[int, ...],
    grid_points: int = 129,
) -> tuple[dict[int, torch.Tensor], list[tuple[int, int, float]]]:
    """Greedily fit output-aware channel-pair rotations.

    Pair selection uses the covariance/projector commutator, and each selected
    angle is searched against the true projected defect.  Zero is included in
    the search grid, so a prefix cannot worsen the previous prefix.
    """

    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer >= 3")
    wanted = tuple(sorted(set(generator_counts)))
    if not wanted or wanted[0] <= 0:
        raise ValueError("generator counts must be positive")
    current = source.clone()
    covariance = defect.T @ defect
    outputs: dict[int, torch.Tensor] = {}
    parameters: list[tuple[int, int, float]] = []
    upper = torch.triu(torch.ones_like(covariance, dtype=torch.bool), diagonal=1)
    angles = torch.linspace(-math.pi, math.pi, grid_points, device=source.device)

    for step in range(1, wanted[-1] + 1):
        projector = current @ current.T
        commutator = covariance @ projector - projector @ covariance
        score = commutator.abs().masked_fill(~upper, float("-inf"))
        flat_index = int(score.argmax())
        first = flat_index // score.shape[1]
        second = flat_index % score.shape[1]

        first_row = current[first]
        second_row = current[second]
        candidates = current.unsqueeze(0).expand(grid_points, -1, -1).clone()
        cosine = torch.cos(angles).unsqueeze(1)
        sine = torch.sin(angles).unsqueeze(1)
        candidates[:, first] = cosine * first_row - sine * second_row
        candidates[:, second] = sine * first_row + cosine * second_row
        expanded_defect = defect.unsqueeze(0).expand(grid_points, -1, -1)
        residual_squares = batch_residual_squares(expanded_defect, candidates)
        best = int(residual_squares.argmin())
        current = candidates[best].clone()
        parameters.append((first, second, float(angles[best])))
        if step in wanted:
            outputs[step] = current.clone()
    return outputs, parameters


def householder_prefixes(
    source: torch.Tensor,
    target: torch.Tensor,
    generator_counts: tuple[int, ...],
) -> tuple[dict[int, torch.Tensor], list[torch.Tensor]]:
    """Construct prefixes that map a source frame to a target frame in <= r reflections."""

    wanted = tuple(sorted(set(generator_counts)))
    if not wanted or wanted[0] <= 0 or wanted[-1] > source.shape[1]:
        raise ValueError("Householder counts must be in [1, rank]")
    # Sparse renormalization can make ||defect|| / ||reference|| very large for
    # individual heads.  Construct the exact control in float64 so a 1e-5
    # projector error is not amplified into a percent-level output error.
    work_source = orthonormalize(source.to(dtype=torch.float64))
    work_target = orthonormalize(target.to(dtype=torch.float64))
    aligned_target = align_target_frame(work_source, work_target)
    current = work_source.clone()
    outputs: dict[int, torch.Tensor] = {}
    vectors: list[torch.Tensor] = []
    for step in range(1, wanted[-1] + 1):
        vector = current[:, step - 1] - aligned_target[:, step - 1]
        denominator = vector.square().sum()
        if float(denominator) > 1e-20:
            current = current - 2.0 * vector[:, None] * (vector @ current)[None, :] / denominator
        vectors.append(vector.clone())
        if step in wanted:
            outputs[step] = orthonormalize(current)
    return outputs, vectors


def _channel_permutation(channels: int, stage: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    multipliers = (1, 3, 5, 11, 17, 29, 43, 61)
    multiplier = multipliers[stage % len(multipliers)]
    while math.gcd(multiplier, channels) != 1:
        multiplier += 2
    permutation = (torch.arange(channels, device=device) * multiplier) % channels
    return permutation, permutation.argsort()


def _butterfly_pairs(channels: int, stage: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if channels <= 0 or channels & (channels - 1):
        raise ValueError("butterfly channel count must be a positive power of two")
    stride = 1 << (stage % int(math.log2(channels)))
    starts = torch.arange(0, channels, 2 * stride, device=device)
    offsets = torch.arange(stride, device=device)
    first = (starts[:, None] + offsets[None, :]).flatten()
    return first, first + stride


def apply_butterfly(source: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if source.ndim != 3 or angles.ndim != 3:
        raise ValueError("butterfly expects [batch,d,r] and [batch,M,d/2]")
    batch, channels, _ = source.shape
    if angles.shape[0] != batch or angles.shape[2] != channels // 2:
        raise ValueError("butterfly parameter shape mismatch")
    output = source
    for stage in range(angles.shape[1]):
        first, second = _butterfly_pairs(channels, stage, source.device)
        first_rows = output.index_select(1, first)
        second_rows = output.index_select(1, second)
        cosine = torch.cos(angles[:, stage]).unsqueeze(2)
        sine = torch.sin(angles[:, stage]).unsqueeze(2)
        updated = output.clone()
        updated[:, first] = cosine * first_rows - sine * second_rows
        updated[:, second] = sine * first_rows + cosine * second_rows
        output = updated
    return output


def apply_orthogonal_bcm(
    source: torch.Tensor,
    phases: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply a product of permuted orthogonal block-circulant factors."""

    if source.ndim != 3 or phases.ndim != 4:
        raise ValueError("BCM expects [batch,d,r] and [batch,M,blocks,phase]")
    batch, channels, rank = source.shape
    if channels % block_size:
        raise ValueError("channels must be divisible by BCM block size")
    blocks = channels // block_size
    interior = block_size // 2 - 1
    if phases.shape[0] != batch or phases.shape[2:] != (blocks, interior):
        raise ValueError("BCM phase shape mismatch")
    output = source
    endpoint = source.new_zeros((batch, blocks, 1))
    for stage in range(phases.shape[1]):
        permutation, inverse = _channel_permutation(channels, stage, source.device)
        grouped = output.index_select(1, permutation).reshape(batch, blocks, block_size, rank)
        spectrum = torch.fft.rfft(grouped, dim=2, norm="ortho")
        full_phase = torch.cat((endpoint, phases[:, stage], endpoint), dim=2)
        unit_phase = torch.polar(torch.ones_like(full_phase), full_phase).unsqueeze(3)
        grouped = torch.fft.irfft(spectrum * unit_phase, n=block_size, dim=2, norm="ortho")
        output = grouped.reshape(batch, channels, rank).index_select(1, inverse)
    return output


def apply_dcd(
    source: torch.Tensor,
    left_raw: torch.Tensor,
    right_raw: torch.Tensor,
    phases: torch.Tensor,
    max_log_scale: float,
) -> torch.Tensor:
    """Apply bounded diagonal-circulant-diagonal factors with stable QR steps."""

    if source.ndim != 3 or left_raw.ndim != 3 or right_raw.ndim != 3 or phases.ndim != 3:
        raise ValueError("DCD expects batched source and parameter tensors")
    batch, channels, rank = source.shape
    interior = channels // 2 - 1
    expected = (batch, left_raw.shape[1], channels)
    if left_raw.shape != expected or right_raw.shape != expected:
        raise ValueError("DCD diagonal shape mismatch")
    if phases.shape != (batch, left_raw.shape[1], interior):
        raise ValueError("DCD phase shape mismatch")
    output = source
    endpoint = source.new_zeros((batch, 1))
    for stage in range(left_raw.shape[1]):
        permutation, inverse = _channel_permutation(channels, stage, source.device)
        transformed = output.index_select(1, permutation)
        right_scale = torch.exp(max_log_scale * torch.tanh(right_raw[:, stage])).unsqueeze(2)
        left_scale = torch.exp(max_log_scale * torch.tanh(left_raw[:, stage])).unsqueeze(2)
        transformed = transformed * right_scale
        spectrum = torch.fft.rfft(transformed, dim=1, norm="ortho")
        full_phase = torch.cat((endpoint, phases[:, stage], endpoint), dim=1)
        unit_phase = torch.polar(torch.ones_like(full_phase), full_phase).unsqueeze(2)
        transformed = torch.fft.irfft(spectrum * unit_phase, n=channels, dim=1, norm="ortho")
        transformed = transformed * left_scale
        output = orthonormalize(transformed).index_select(1, inverse)
    return output


def _make_parameters(
    family: str,
    batch: int,
    channels: int,
    generators: int,
    block_size: int,
    device: torch.device,
    restart: int,
) -> list[torch.Tensor]:
    scale = 0.0 if restart == 0 else 0.01

    def parameter(shape: tuple[int, ...]) -> torch.Tensor:
        value = torch.randn(shape, device=device) * scale
        return value.requires_grad_(True)

    if family == "butterfly":
        return [parameter((batch, generators, channels // 2))]
    if family == "orthogonal_bcm":
        return [
            parameter((batch, generators, channels // block_size, block_size // 2 - 1))
        ]
    if family == "dcd":
        return [
            parameter((batch, generators, channels)),
            parameter((batch, generators, channels)),
            parameter((batch, generators, channels // 2 - 1)),
        ]
    raise ValueError(f"unsupported optimized family: {family}")


def _apply_parameterized(
    family: str,
    source: torch.Tensor,
    parameters: list[torch.Tensor],
    block_size: int,
    max_log_scale: float,
) -> torch.Tensor:
    if family == "butterfly":
        return apply_butterfly(source, parameters[0])
    if family == "orthogonal_bcm":
        return apply_orthogonal_bcm(source, parameters[0], block_size)
    if family == "dcd":
        return apply_dcd(source, parameters[0], parameters[1], parameters[2], max_log_scale)
    raise ValueError(f"unsupported optimized family: {family}")


def fit_parameterized_rotation(
    family: str,
    source: torch.Tensor,
    defects: torch.Tensor,
    generators: int,
    *,
    steps: int,
    learning_rate: float,
    restarts: int,
    block_size: int,
    max_log_scale: float,
    seed: int,
) -> torch.Tensor:
    """Fit independent post-hoc transforms for every item in one batch."""

    if source.ndim != 3 or defects.ndim != 3 or source.shape[0] != defects.shape[0]:
        raise ValueError("source and defects must be matching batched tensors")
    if generators <= 0 or steps <= 0 or restarts <= 0:
        raise ValueError("generators, steps, and restarts must be positive")
    batch, channels, _ = source.shape
    defect_energy = defects.square().sum(dim=(1, 2)).clamp_min(1e-30)
    best_basis = source.detach().clone()
    best_loss = batch_residual_squares(defects, source).detach() / defect_energy

    for restart in range(restarts):
        torch.manual_seed(seed + restart)
        parameters = _make_parameters(
            family, batch, channels, generators, block_size, source.device, restart
        )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for step in range(steps + 1):
            candidate = _apply_parameterized(
                family, source, parameters, block_size, max_log_scale
            )
            losses = batch_residual_squares(defects, candidate) / defect_energy
            improved = losses.detach() < best_loss
            if bool(improved.any()):
                best_loss = torch.where(improved, losses.detach(), best_loss)
                best_basis[improved] = candidate.detach()[improved]
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            losses.mean().backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
    return orthonormalize(best_basis)


def rotation_cost(
    family: str,
    generators: int,
    *,
    query_tokens: int,
    key_tokens: int,
    channels: int,
    rank: int,
    block_size: int,
) -> RotationCost:
    if min(query_tokens, key_tokens, channels, rank) <= 0 or generators < 0:
        raise ValueError("cost dimensions must be positive and generators non-negative")
    dense = 2 * query_tokens * key_tokens * channels
    tail = 2 * query_tokens * channels * rank
    if family == "frozen":
        dynamic, rotation = 0, 0
    elif family == "adaptive" or family == "full_procrustes":
        dynamic, rotation = channels * rank, 2 * channels * channels * rank
    elif family == "givens":
        dynamic, rotation = 3 * generators, 6 * generators * rank
    elif family == "householder":
        dynamic, rotation = channels * generators, 4 * generators * channels * rank
    elif family == "orthogonal_bcm":
        dynamic = generators * (channels // block_size) * (block_size // 2 - 1)
        rotation = int(5 * generators * channels * math.log2(block_size) * rank)
    elif family == "dcd":
        dynamic = generators * (2 * channels + channels // 2 - 1)
        rotation = int(
            generators * (5 * channels * math.log2(channels) * rank + 4 * channels * rank)
        )
    elif family == "butterfly":
        dynamic = generators * channels // 2
        rotation = 3 * generators * channels * rank
    else:
        raise ValueError(f"unsupported cost family: {family}")
    return RotationCost(family, generators, dynamic, rotation, tail, dense)
