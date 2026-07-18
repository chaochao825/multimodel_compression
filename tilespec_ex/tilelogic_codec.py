"""Shared encoding and reconstruction path for TileLogic-RVQ methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import (
    _per_tile_budget,
    _validate_tiles,
    dct2,
    enumerate_blocks,
    idct2,
    zigzag_coordinates,
)
from .vq import (
    ResidualVQCodebook,
    ScaledCodebook,
    ScalarQuantizationResult,
    symmetric_vector_quantize,
)


@dataclass(frozen=True)
class BaseVQEncoding:
    reconstructed: torch.Tensor
    reference_coefficients: torch.Tensor
    decoded_coefficients: torch.Tensor
    code_indices: torch.Tensor
    scale_indices: torch.Tensor


@dataclass(frozen=True)
class BaseScalarEncoding:
    reconstructed: torch.Tensor
    reference_coefficients: torch.Tensor
    quantization: ScalarQuantizationResult


@dataclass(frozen=True)
class ResidualEncoding:
    reconstructed: torch.Tensor
    decoded_blocks: torch.Tensor
    modes: torch.Tensor
    stage_indices: torch.Tensor
    scale_indices: torch.Tensor


def extract_tile_coefficients(
    crop_tiles: torch.Tensor, total_retained: int
) -> torch.Tensor:
    """Extract per-tile zig-zag DCT coefficients as [total_retained,C]."""

    tiles, height, width, _ = _validate_tiles(crop_tiles)
    per_tile = _per_tile_budget(total_retained, tiles)
    if per_tile > height * width:
        raise ValueError("retained coefficient count exceeds tile size")
    coordinates = zigzag_coordinates(height, width)[:per_tile]
    rows = torch.tensor(
        [item[0] for item in coordinates], device=crop_tiles.device, dtype=torch.long
    )
    cols = torch.tensor(
        [item[1] for item in coordinates], device=crop_tiles.device, dtype=torch.long
    )
    coefficients = [dct2(tile)[rows, cols] for tile in crop_tiles]
    return torch.cat(coefficients, dim=0).to(crop_tiles.dtype)


def reconstruct_tile_coefficients(
    compact: torch.Tensor,
    *,
    tiles: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Place compact coefficients back into sparse DCT grids and apply IDCT."""

    if compact.ndim != 2:
        raise ValueError("compact coefficients must have shape [N,C]")
    if tiles <= 0 or compact.shape[0] % tiles:
        raise ValueError("compact coefficient count must divide across tiles")
    per_tile = compact.shape[0] // tiles
    if per_tile > height * width:
        raise ValueError("too many compact coefficients for tile dimensions")
    coordinates = zigzag_coordinates(height, width)[:per_tile]
    rows = torch.tensor(
        [item[0] for item in coordinates], device=compact.device, dtype=torch.long
    )
    cols = torch.tensor(
        [item[1] for item in coordinates], device=compact.device, dtype=torch.long
    )
    output = []
    for tile in range(tiles):
        sparse = torch.zeros(
            (height, width, compact.shape[1]),
            device=compact.device,
            dtype=torch.float32,
        )
        start = tile * per_tile
        sparse[rows, cols] = compact[start : start + per_tile].float()
        output.append(idct2(sparse))
    return torch.stack(output)


def encode_base_vq(
    crop_tiles: torch.Tensor,
    total_retained: int,
    codebook: ScaledCodebook,
    *,
    batch_size: int = 1024,
) -> BaseVQEncoding:
    tiles, height, width, channels = _validate_tiles(crop_tiles)
    if channels != codebook.dimension:
        raise ValueError("crop channel dimension differs from base codebook")
    reference = extract_tile_coefficients(crop_tiles, total_retained).float()
    decoded, indices, scales = codebook.reconstruct(reference, batch_size=batch_size)
    reconstructed = reconstruct_tile_coefficients(
        decoded,
        tiles=tiles,
        height=height,
        width=width,
    ).to(crop_tiles.dtype)
    return BaseVQEncoding(reconstructed, reference, decoded, indices, scales)


def encode_base_scalar(
    crop_tiles: torch.Tensor,
    total_retained: int,
    *,
    bits: int = 4,
    scale_storage_bits: int = 32,
) -> BaseScalarEncoding:
    tiles, height, width, _ = _validate_tiles(crop_tiles)
    reference = extract_tile_coefficients(crop_tiles, total_retained).float()
    quantized = symmetric_vector_quantize(
        reference, bits=bits, scale_storage_bits=scale_storage_bits
    )
    reconstructed = reconstruct_tile_coefficients(
        quantized.reconstructed,
        tiles=tiles,
        height=height,
        width=width,
    ).to(crop_tiles.dtype)
    return BaseScalarEncoding(reconstructed, reference, quantized)


def encode_residual_modes(
    base: torch.Tensor,
    target: torch.Tensor,
    codebook: ResidualVQCodebook,
    modes: torch.Tensor,
    *,
    batch_size: int = 512,
) -> ResidualEncoding:
    """Apply modes 0=drop, 1..S=RVQ depth, S+1=exact residual."""

    if base.shape != target.shape:
        raise ValueError("base and target shapes differ")
    residual = target - base
    blocks, locations, _ = enumerate_blocks(residual)
    if modes.shape != (len(locations),):
        raise ValueError("modes must enumerate every residual block")
    if codebook.dimension != blocks[0].numel():
        raise ValueError("residual block dimension differs from codebook")
    flat = blocks.float().reshape(len(locations), -1)
    stage_indices, scale_indices = codebook.encode(flat, batch_size=batch_size)
    reconstructed, decoded_blocks = decode_residual_payload(
        base,
        blocks,
        locations,
        codebook,
        stage_indices,
        scale_indices,
        modes,
        output_dtype=target.dtype,
    )
    return ResidualEncoding(
        reconstructed,
        decoded_blocks,
        modes,
        stage_indices,
        scale_indices,
    )


def decode_residual_payload(
    base: torch.Tensor,
    residual_blocks: torch.Tensor,
    locations: tuple[tuple[int, int, int], ...],
    codebook: ResidualVQCodebook,
    stage_indices: torch.Tensor,
    scale_indices: torch.Tensor,
    modes: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode precomputed RVQ indices and scatter them into a tile buffer."""

    if residual_blocks.ndim != 4 or residual_blocks.shape[0] != len(locations):
        raise ValueError("residual blocks and locations differ")
    if modes.shape != (len(locations),):
        raise ValueError("modes must enumerate every residual block")
    if stage_indices.shape != (len(locations), codebook.stages):
        raise ValueError("stage index shape differs from residual codebook")
    if scale_indices.shape != (len(locations),):
        raise ValueError("scale index shape differs from residual blocks")
    flat = residual_blocks.float().reshape(len(locations), -1)
    decoded = torch.zeros_like(flat)
    rvq = (modes > 0) & (modes <= codebook.stages)
    if bool(rvq.any()):
        decoded[rvq] = codebook.decode(
            stage_indices[rvq],
            scale_indices[rvq],
            depths=modes[rvq],
        )
    exact = modes == codebook.stages + 1
    if bool(exact.any()):
        decoded[exact] = flat[exact]
    if bool(((modes < 0) | (modes > codebook.stages + 1)).any()):
        raise ValueError("residual mode is out of range")

    reconstructed = base.float().clone()
    decoded_blocks = decoded.reshape_as(residual_blocks)
    for index, (tile, row, col) in enumerate(locations):
        reconstructed[tile, row : row + 2, col : col + 2] += decoded_blocks[index]
    return reconstructed.to(output_dtype or base.dtype), decoded_blocks


def oracle_marginal_benefits(
    residual_blocks: torch.Tensor,
    gradient_blocks: torch.Tensor,
    codebook: ResidualVQCodebook,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return non-negative first-order benefits for RVQ1, RVQ2, ..., exact."""

    if residual_blocks.shape != gradient_blocks.shape or residual_blocks.ndim != 4:
        raise ValueError("residual and gradient blocks must share [B,2,2,C]")
    flat = residual_blocks.float().reshape(residual_blocks.shape[0], -1)
    gradients = gradient_blocks.float().reshape(gradient_blocks.shape[0], -1)
    reconstructions, _, _ = codebook.reconstructions_by_depth(
        flat, batch_size=batch_size
    )
    first_order_errors = [
        (gradients * (flat - reconstruction.float())).sum(dim=1).abs()
        for reconstruction in reconstructions
    ]
    benefits = []
    for stage in range(codebook.stages):
        benefits.append(
            (first_order_errors[stage] - first_order_errors[stage + 1]).clamp_min(0)
        )
    benefits.append(first_order_errors[-1])
    return torch.stack(benefits, dim=1)


def weighted_marginal_benefits(
    residual_blocks: torch.Tensor,
    codebook: ResidualVQCodebook,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return diagonal-metric distortion reductions for each coding upgrade."""

    if residual_blocks.ndim != 4:
        raise ValueError("residual_blocks must have shape [B,2,2,C]")
    flat = residual_blocks.float().reshape(residual_blocks.shape[0], -1)
    reconstructions, _, _ = codebook.reconstructions_by_depth(
        flat, batch_size=batch_size
    )
    weights = codebook.metric_weights.to(flat.device)
    errors = [
        ((flat - reconstruction.float()).square() * weights).sum(dim=1)
        for reconstruction in reconstructions
    ]
    benefits = [
        (errors[stage] - errors[stage + 1]).clamp_min(0)
        for stage in range(codebook.stages)
    ]
    benefits.append(errors[-1])
    return torch.stack(benefits, dim=1)
