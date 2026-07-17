"""Bit-exact budgeted spectral compression primitives for TileSpec-Ex.

The quality probe reconstructs the original visual-token grid so it can be
injected into an unchanged VLM.  ``compact`` contains the representation that
would be consumed by a later compact execution path; the two are deliberately
kept separate so fixed-length quality simulation is not reported as speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


METHODS = (
    "none",
    "average_pool",
    "global_lowpass",
    "tile_lowpass",
    "tile_energy_exception",
    "tile_risk_exception",
)
RETENTION_RATES = (0.125, 0.25)


@dataclass(frozen=True)
class CompressionResult:
    reconstructed: torch.Tensor
    compact: torch.Tensor
    retained_tokens: int
    base_tokens: int
    exception_tokens: int
    selected_blocks: tuple[tuple[int, int, int], ...]
    block_energy: torch.Tensor | None = None
    block_relevance: torch.Tensor | None = None
    block_score: torch.Tensor | None = None


STRUCTURE_VARIANTS = (
    "risk_token_unstructured",
    "risk_block_dynamic",
    "risk_block_fixed_slots",
)


def orthonormal_dct_matrix(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError("DCT size must be positive")
    positions = torch.arange(n, device=device, dtype=dtype) + 0.5
    frequencies = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos((math.pi / n) * frequencies * positions)
    matrix *= math.sqrt(2.0 / n)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    return matrix


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal 2-D DCT to ``[H, W, C]`` input."""

    if x.ndim != 3:
        raise ValueError(f"expected [H,W,C], got {tuple(x.shape)}")
    h, w, _ = x.shape
    work = x.float()
    dh = orthonormal_dct_matrix(h, device=x.device, dtype=work.dtype)
    dw = orthonormal_dct_matrix(w, device=x.device, dtype=work.dtype)
    return torch.einsum("ih,hwc,jw->ijc", dh, work, dw)


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.ndim != 3:
        raise ValueError(
            f"expected [H,W,C] coefficients, got {tuple(coefficients.shape)}"
        )
    h, w, _ = coefficients.shape
    work = coefficients.float()
    dh = orthonormal_dct_matrix(h, device=work.device, dtype=work.dtype)
    dw = orthonormal_dct_matrix(w, device=work.device, dtype=work.dtype)
    return torch.einsum("ih,ijc,jw->hwc", dh, work, dw)


def zigzag_coordinates(height: int, width: int) -> tuple[tuple[int, int], ...]:
    if height <= 0 or width <= 0:
        raise ValueError("zigzag dimensions must be positive")
    coordinates: list[tuple[int, int]] = []
    for diagonal in range(height + width - 1):
        current = [
            (row, diagonal - row)
            for row in range(height)
            if 0 <= diagonal - row < width
        ]
        if diagonal % 2 == 0:
            current.reverse()
        coordinates.extend(current)
    return tuple(coordinates)


def _lowpass(
    x: torch.Tensor, retained: int
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width, channels = x.shape
    total = height * width
    if not 0 < retained <= total:
        raise ValueError(f"retained must be in [1,{total}], got {retained}")
    coefficients = dct2(x)
    coordinates = zigzag_coordinates(height, width)[:retained]
    rows = torch.tensor([item[0] for item in coordinates], device=x.device)
    cols = torch.tensor([item[1] for item in coordinates], device=x.device)
    compact = coefficients[rows, cols]
    sparse = torch.zeros(
        (height, width, channels), device=x.device, dtype=coefficients.dtype
    )
    sparse[rows, cols] = compact
    return idct2(sparse).to(x.dtype), compact.to(x.dtype)


def _pool_shape(height: int, width: int, retained: int) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []
    target_aspect = height / width
    for out_height in range(1, min(height, retained) + 1):
        if retained % out_height:
            continue
        out_width = retained // out_height
        if out_width > width:
            continue
        aspect_error = abs(math.log((out_height / out_width) / target_aspect))
        candidates.append((aspect_error, out_height, out_width))
    if not candidates:
        raise ValueError(
            f"cannot make exact {retained}-cell pool for {height}x{width} tile"
        )
    _, out_height, out_width = min(candidates)
    return out_height, out_width


def _average_pool_tile(
    x: torch.Tensor, retained: int
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width, _ = x.shape
    out_height, out_width = _pool_shape(height, width, retained)
    nchw = x.permute(2, 0, 1).unsqueeze(0).float()
    compact_grid = F.adaptive_avg_pool2d(nchw, (out_height, out_width))
    reconstructed = F.interpolate(
        compact_grid,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    compact = compact_grid.squeeze(0).permute(1, 2, 0).reshape(retained, -1)
    return reconstructed.squeeze(0).permute(1, 2, 0).to(x.dtype), compact.to(x.dtype)


def _validate_tiles(crop_tiles: torch.Tensor) -> tuple[int, int, int, int]:
    if crop_tiles.ndim != 4:
        raise ValueError(
            f"expected [tiles,H,W,C], got {tuple(crop_tiles.shape)}"
        )
    tiles, height, width, channels = crop_tiles.shape
    if tiles != 4:
        raise ValueError("the minimal experiment requires exactly four crop tiles")
    if height % 2 or width % 2:
        raise ValueError("2x2 exception blocks require even tile dimensions")
    return tiles, height, width, channels


def stitch_crop_tiles(crop_tiles: torch.Tensor) -> torch.Tensor:
    """Stitch TL/TR/BL/BR feature tiles into one global feature grid."""

    _validate_tiles(crop_tiles)
    top = torch.cat((crop_tiles[0], crop_tiles[1]), dim=1)
    bottom = torch.cat((crop_tiles[2], crop_tiles[3]), dim=1)
    return torch.cat((top, bottom), dim=0)


def unstack_crop_grid(global_grid: torch.Tensor) -> torch.Tensor:
    if global_grid.ndim != 3:
        raise ValueError("global grid must be [H,W,C]")
    height, width, _ = global_grid.shape
    if height % 2 or width % 2:
        raise ValueError("global crop grid must split evenly into 2x2 tiles")
    half_height, half_width = height // 2, width // 2
    return torch.stack(
        (
            global_grid[:half_height, :half_width],
            global_grid[:half_height, half_width:],
            global_grid[half_height:, :half_width],
            global_grid[half_height:, half_width:],
        )
    )


def _per_tile_budget(total_retained: int, tile_count: int = 4) -> int:
    if total_retained % tile_count:
        raise ValueError(
            f"retained budget {total_retained} must divide across {tile_count} tiles"
        )
    return total_retained // tile_count


def _tile_lowpass(
    crop_tiles: torch.Tensor, total_retained: int
) -> tuple[torch.Tensor, torch.Tensor]:
    per_tile = _per_tile_budget(total_retained)
    reconstructions: list[torch.Tensor] = []
    compact: list[torch.Tensor] = []
    for tile in crop_tiles:
        reconstructed, coefficients = _lowpass(tile, per_tile)
        reconstructions.append(reconstructed)
        compact.append(coefficients)
    return torch.stack(reconstructions), torch.cat(compact)


def enumerate_blocks(
    residual: torch.Tensor, block_size: int = 2
) -> tuple[torch.Tensor, tuple[tuple[int, int, int], ...], torch.Tensor]:
    tiles, height, width, channels = _validate_tiles(residual)
    if block_size != 2:
        raise ValueError("V0 only supports 2x2 exception blocks")
    blocks: list[torch.Tensor] = []
    locations: list[tuple[int, int, int]] = []
    for tile_index in range(tiles):
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                blocks.append(
                    residual[
                        tile_index,
                        row : row + block_size,
                        col : col + block_size,
                    ]
                )
                locations.append((tile_index, row, col))
    block_tensor = torch.stack(blocks)
    flat = block_tensor.reshape(len(blocks), block_size * block_size, channels)
    energy = flat.float().square().sum(dim=(1, 2))
    return block_tensor, tuple(locations), energy


def block_query_relevance(
    blocks: torch.Tensor, query_embedding: torch.Tensor
) -> torch.Tensor:
    if query_embedding.ndim != 1:
        raise ValueError("query embedding must be one-dimensional")
    block_vectors = blocks.float().mean(dim=(1, 2))
    if block_vectors.shape[1] != query_embedding.numel():
        raise ValueError("query and visual embedding dimensions differ")
    block_vectors = F.normalize(block_vectors, dim=1, eps=1e-12)
    query = F.normalize(query_embedding.float(), dim=0, eps=1e-12)
    cosine = block_vectors @ query
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def _add_selected_blocks(
    base: torch.Tensor,
    residual: torch.Tensor,
    locations: Iterable[tuple[int, int, int]],
) -> torch.Tensor:
    reconstructed = base.clone()
    for tile, row, col in locations:
        reconstructed[tile, row : row + 2, col : col + 2] += residual[
            tile, row : row + 2, col : col + 2
        ]
    return reconstructed


def _exception_budget(
    original_tokens: int,
    retention_rate: float,
    exception_fraction: float,
) -> tuple[int, int, int]:
    retained = round(original_tokens * retention_rate)
    exception_tokens = round(retained * exception_fraction / 4) * 4
    base_tokens = retained - exception_tokens
    if retained % 4 or base_tokens % 4:
        raise ValueError("retained and base budgets must divide across four tiles")
    if exception_tokens <= 0 or exception_tokens >= retained:
        raise ValueError("exception budget must leave both base and residual tokens")
    return retained, base_tokens, exception_tokens


def _token_query_relevance(
    residual: torch.Tensor, query_embedding: torch.Tensor
) -> torch.Tensor:
    flat = residual.float().reshape(-1, residual.shape[-1])
    visual = F.normalize(flat, dim=1, eps=1e-12)
    query = F.normalize(query_embedding.float(), dim=0, eps=1e-12)
    return ((visual @ query + 1.0) * 0.5).clamp(0.0, 1.0)


def compress_risk_structure_variant(
    crop_tiles: torch.Tensor,
    variant: str,
    retention_rate: float,
    *,
    query_embedding: torch.Tensor,
    exception_fraction: float = 0.25,
) -> CompressionResult:
    """Risk-exception structural ablation under the main method's budget.

    ``risk_block_dynamic`` is identical to the headline risk-exception method.
    ``risk_token_unstructured`` selects arbitrary residual tokens, while
    ``risk_block_fixed_slots`` reserves the same number of block slots in every
    tile.  All three return the same exact number of compact vectors.
    """

    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(f"unknown structure variant: {variant}")
    tiles, height, width, channels = _validate_tiles(crop_tiles)
    original_tokens = tiles * height * width
    retained, base_tokens, exception_tokens = _exception_budget(
        original_tokens, retention_rate, exception_fraction
    )
    base, base_compact = _tile_lowpass(crop_tiles, base_tokens)
    residual = crop_tiles - base

    if variant == "risk_token_unstructured":
        flat_residual = residual.reshape(original_tokens, channels)
        energy = flat_residual.float().square().sum(dim=1)
        relevance = _token_query_relevance(residual, query_embedding)
        score = energy * relevance
        selected_indices = torch.topk(score, k=exception_tokens).indices
        reconstructed_flat = base.reshape(original_tokens, channels).clone()
        reconstructed_flat[selected_indices] += flat_residual[selected_indices]
        compact = torch.cat((base_compact, flat_residual[selected_indices]))
        locations = tuple(
            (
                index // (height * width),
                (index % (height * width)) // width,
                index % width,
            )
            for index in selected_indices.tolist()
        )
        return CompressionResult(
            reconstructed_flat.reshape_as(crop_tiles),
            compact,
            retained,
            base_tokens,
            exception_tokens,
            locations,
            block_energy=energy,
            block_relevance=relevance,
            block_score=score,
        )

    if variant == "risk_block_dynamic":
        return compress_crop_tiles(
            crop_tiles,
            "tile_risk_exception",
            retention_rate,
            query_embedding=query_embedding,
            exception_fraction=exception_fraction,
        )

    blocks, locations, energy = enumerate_blocks(residual)
    relevance = block_query_relevance(blocks, query_embedding)
    score = energy * relevance
    block_count = exception_tokens // 4
    if block_count % tiles:
        raise ValueError("fixed-slot block budget must divide across tiles")
    per_tile = block_count // tiles
    selected: list[int] = []
    for tile_index in range(tiles):
        candidates = torch.tensor(
            [
                index
                for index, location in enumerate(locations)
                if location[0] == tile_index
            ],
            device=score.device,
            dtype=torch.long,
        )
        local = torch.topk(score[candidates], k=per_tile).indices
        selected.extend(candidates[local].tolist())
    selected_indices = torch.tensor(selected, device=score.device, dtype=torch.long)
    selected_locations = tuple(locations[index] for index in selected)
    reconstructed = _add_selected_blocks(base, residual, selected_locations)
    compact = torch.cat(
        (base_compact, blocks[selected_indices].reshape(exception_tokens, channels))
    )
    return CompressionResult(
        reconstructed,
        compact,
        retained,
        base_tokens,
        exception_tokens,
        selected_locations,
        block_energy=energy,
        block_relevance=relevance,
        block_score=score,
    )


def compress_crop_tiles(
    crop_tiles: torch.Tensor,
    method: str,
    retention_rate: float,
    *,
    query_embedding: torch.Tensor | None = None,
    exception_fraction: float = 0.25,
) -> CompressionResult:
    """Compress four crop feature grids under an exact token budget."""

    tiles, height, width, channels = _validate_tiles(crop_tiles)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if not 0.0 < retention_rate <= 1.0:
        raise ValueError("retention_rate must be in (0,1]")
    original_tokens = tiles * height * width
    retained = original_tokens if method == "none" else round(
        original_tokens * retention_rate
    )
    if method != "none" and retained % tiles:
        raise ValueError("compressed budget must divide evenly across crop tiles")

    if method == "none":
        compact = crop_tiles.reshape(original_tokens, channels)
        return CompressionResult(
            crop_tiles.clone(), compact, original_tokens, original_tokens, 0, ()
        )

    if method == "average_pool":
        per_tile = _per_tile_budget(retained)
        reconstructed: list[torch.Tensor] = []
        compact: list[torch.Tensor] = []
        for tile in crop_tiles:
            tile_reconstructed, tile_compact = _average_pool_tile(tile, per_tile)
            reconstructed.append(tile_reconstructed)
            compact.append(tile_compact)
        return CompressionResult(
            torch.stack(reconstructed),
            torch.cat(compact),
            retained,
            retained,
            0,
            (),
        )

    if method == "global_lowpass":
        stitched = stitch_crop_tiles(crop_tiles)
        reconstructed, compact = _lowpass(stitched, retained)
        return CompressionResult(
            unstack_crop_grid(reconstructed), compact, retained, retained, 0, ()
        )

    if method == "tile_lowpass":
        reconstructed, compact = _tile_lowpass(crop_tiles, retained)
        return CompressionResult(
            reconstructed, compact, retained, retained, 0, ()
        )

    retained_check, base_tokens, exception_tokens = _exception_budget(
        original_tokens, retention_rate, exception_fraction
    )
    if retained_check != retained:
        raise AssertionError("inconsistent exception budget")
    base, base_compact = _tile_lowpass(crop_tiles, base_tokens)
    residual = crop_tiles - base
    blocks, locations, energy = enumerate_blocks(residual)
    block_count = exception_tokens // 4

    relevance: torch.Tensor | None = None
    if method == "tile_energy_exception":
        score = energy
    else:
        if query_embedding is None:
            raise ValueError("risk exception requires a query embedding")
        relevance = block_query_relevance(blocks, query_embedding)
        score = energy * relevance

    selected_indices = torch.topk(score, k=block_count, largest=True).indices
    selected_locations = tuple(locations[index] for index in selected_indices.tolist())
    reconstructed = _add_selected_blocks(base, residual, selected_locations)
    selected_blocks = blocks[selected_indices].reshape(exception_tokens, channels)
    compact = torch.cat((base_compact, selected_blocks))
    if compact.shape[0] != retained:
        raise AssertionError("compact representation violated the declared budget")
    return CompressionResult(
        reconstructed=reconstructed,
        compact=compact,
        retained_tokens=retained,
        base_tokens=base_tokens,
        exception_tokens=exception_tokens,
        selected_blocks=selected_locations,
        block_energy=energy,
        block_relevance=relevance,
        block_score=score,
    )


def normalized_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = (reference.float() - estimate.float()).square().sum()
    denominator = reference.float().square().sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def cosine_similarity(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    lhs = reference.float().reshape(-1)
    rhs = estimate.float().reshape(-1)
    return float(F.cosine_similarity(lhs, rhs, dim=0, eps=1e-12).item())


def crop_boundary_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """MSE in a one-token band around the two crop-tile seams."""

    stitched_reference = stitch_crop_tiles(reference).float()
    stitched_estimate = stitch_crop_tiles(estimate).float()
    height, width, _ = stitched_reference.shape
    row, col = height // 2, width // 2
    mask = torch.zeros((height, width), device=reference.device, dtype=torch.bool)
    mask[max(0, row - 1) : min(height, row + 1), :] = True
    mask[:, max(0, col - 1) : min(width, col + 1)] = True
    error = (stitched_reference - stitched_estimate).square().mean(dim=2)
    return float(error[mask].mean().item())
