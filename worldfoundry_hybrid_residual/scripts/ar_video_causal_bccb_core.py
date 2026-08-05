"""Causal spatial BCCB/Toeplitz operators for captured LongLive Q/K/V."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class CapturedQueryLayout:
    """Spatial positions and tile IDs for every saved query frame."""

    query_frames: int
    positions_per_frame: torch.Tensor
    tile_ids_per_frame: torch.Tensor
    tile_count: int
    tile_size: int

    @property
    def queries_per_frame(self) -> int:
        return int(self.positions_per_frame.numel())


def make_captured_query_layout(
    saved_query_tokens: int,
    frame_seq_len: int,
    tile_size: int,
    tile_count: int,
    device: torch.device | str = "cpu",
) -> CapturedQueryLayout:
    if min(saved_query_tokens, frame_seq_len, tile_size, tile_count) <= 0:
        raise ValueError("query layout dimensions must be positive")
    if tile_size > frame_seq_len:
        raise ValueError("tile size exceeds one frame")
    queries_per_frame = tile_size * tile_count
    if saved_query_tokens % queries_per_frame:
        raise ValueError("saved queries do not contain complete frame/tile groups")
    query_frames = saved_query_tokens // queries_per_frame
    maximum_start = frame_seq_len - tile_size
    starts = (
        [maximum_start // 2]
        if tile_count == 1
        else [
            round(index * maximum_start / (tile_count - 1))
            for index in range(tile_count)
        ]
    )
    positions = []
    tile_ids = []
    for tile_id, start in enumerate(starts):
        positions.extend(range(start, start + tile_size))
        tile_ids.extend([tile_id] * tile_size)
    return CapturedQueryLayout(
        query_frames=query_frames,
        positions_per_frame=torch.tensor(positions, dtype=torch.long, device=device),
        tile_ids_per_frame=torch.tensor(tile_ids, dtype=torch.long, device=device),
        tile_count=tile_count,
        tile_size=tile_size,
    )


def displacement_indices(
    query_positions: torch.Tensor,
    height: int,
    width: int,
    periodic: bool,
) -> tuple[torch.Tensor, int]:
    """Map each query/key pair to a spatial displacement bucket."""

    if query_positions.ndim != 1 or query_positions.dtype != torch.long:
        raise ValueError("query positions must be a one-dimensional long tensor")
    spatial = height * width
    if height <= 0 or width <= 0 or spatial <= 0:
        raise ValueError("height and width must be positive")
    if query_positions.numel() == 0:
        raise ValueError("at least one query position is required")
    if int(query_positions.min()) < 0 or int(query_positions.max()) >= spatial:
        raise ValueError("query position lies outside the spatial grid")

    key_positions = torch.arange(spatial, device=query_positions.device)
    query_y = torch.div(query_positions, width, rounding_mode="floor")[:, None]
    query_x = (query_positions % width)[:, None]
    key_y = torch.div(key_positions, width, rounding_mode="floor")[None, :]
    key_x = (key_positions % width)[None, :]
    if periodic:
        delta_y = (key_y - query_y) % height
        delta_x = (key_x - query_x) % width
        return delta_y * width + delta_x, spatial
    delta_y = key_y - query_y + height - 1
    delta_x = key_x - query_x + width - 1
    signed_width = 2 * width - 1
    return delta_y * signed_width + delta_x, (2 * height - 1) * signed_width


def project_logits_to_spatial_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    query_group_ids: torch.Tensor,
    height: int,
    width: int,
    periodic: bool,
) -> torch.Tensor:
    """Project QK logits onto group-conditioned displacement buckets.

    Returns `[groups, heads, displacement_bins]`. The operation uses only Q/K
    and is therefore runtime-observable; no dense attention output is read.
    """

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query/key must have shape [tokens, heads, dim]")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("query/key head dimensions do not match")
    if query_positions.shape != (query.shape[0],):
        raise ValueError("query position count does not match query tokens")
    if query_group_ids.shape != (query.shape[0],):
        raise ValueError("query group count does not match query tokens")
    if query_group_ids.dtype != torch.long or int(query_group_ids.min()) < 0:
        raise ValueError("query groups must be non-negative long IDs")
    if key.shape[0] != height * width:
        raise ValueError("key tokens do not match the spatial grid")

    bucket_index, bucket_count = displacement_indices(
        query_positions, height, width, periodic
    )
    logits = torch.einsum("qhd,khd->hqk", query.float(), key.float())
    logits.mul_(1.0 / math.sqrt(query.shape[-1]))
    groups = int(query_group_ids.max()) + 1
    kernels = torch.empty(
        (groups, query.shape[1], bucket_count),
        dtype=torch.float32,
        device=query.device,
    )
    for group in range(groups):
        selected = query_group_ids == group
        if not bool(selected.any()):
            raise ValueError(f"query group {group} is empty")
        indices = bucket_index[selected]
        flat_indices = indices.reshape(1, -1).expand(query.shape[1], -1)
        sums = torch.zeros(
            (query.shape[1], bucket_count), dtype=torch.float32, device=query.device
        )
        sums.scatter_add_(1, flat_indices, logits[:, selected].reshape(query.shape[1], -1))
        counts = torch.bincount(indices.reshape(-1), minlength=bucket_count).float()
        if bool((counts == 0).any()):
            raise ValueError("spatial projection produced an unobserved displacement")
        kernels[group] = sums / counts[None, :]
    return kernels


def logits_from_spatial_kernel(
    kernel: torch.Tensor,
    query_positions: torch.Tensor,
    query_group_ids: torch.Tensor,
    height: int,
    width: int,
    periodic: bool,
) -> torch.Tensor:
    """Reconstruct `[heads, queries, spatial_keys]` projected logits."""

    if kernel.ndim != 3:
        raise ValueError("kernel must have shape [groups, heads, bins]")
    if query_positions.shape != query_group_ids.shape:
        raise ValueError("query positions and groups must have the same shape")
    indices, bucket_count = displacement_indices(
        query_positions, height, width, periodic
    )
    if kernel.shape[-1] != bucket_count:
        raise ValueError("kernel displacement dimension does not match the grid")
    if int(query_group_ids.max()) >= kernel.shape[0]:
        raise ValueError("query group refers to a missing kernel")
    output = torch.empty(
        (kernel.shape[1], query_positions.numel(), height * width),
        dtype=kernel.dtype,
        device=kernel.device,
    )
    for group in range(kernel.shape[0]):
        selected = query_group_ids == group
        if bool(selected.any()):
            output[:, selected] = kernel[group][:, indices[selected]]
    return output


def build_spatial_kernel_bank(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: CapturedQueryLayout,
    key_frames: int,
    height: int,
    width: int,
    periodic: bool,
    query_groups: str,
) -> torch.Tensor:
    """Build `[q_frames, groups, k_frames, heads, bins]` QK kernels."""

    spatial = height * width
    if key.shape[0] != key_frames * spatial:
        raise ValueError("flat key tensor does not match frame/grid dimensions")
    if query.shape[0] != layout.query_frames * layout.queries_per_frame:
        raise ValueError("flat query tensor does not match the captured layout")
    if query_groups not in {"global", "capture_tiles"}:
        raise ValueError(f"unsupported query grouping: {query_groups}")
    query_view = query.reshape(
        layout.query_frames, layout.queries_per_frame, query.shape[1], query.shape[2]
    )
    key_view = key.reshape(key_frames, spatial, key.shape[1], key.shape[2])
    group_ids = (
        torch.zeros_like(layout.tile_ids_per_frame)
        if query_groups == "global"
        else layout.tile_ids_per_frame
    )
    banks = []
    for query_frame in range(layout.query_frames):
        per_key = []
        for key_frame in range(key_frames):
            per_key.append(
                project_logits_to_spatial_kernel(
                    query_view[query_frame],
                    key_view[key_frame],
                    layout.positions_per_frame,
                    group_ids,
                    height,
                    width,
                    periodic,
                )
            )
        banks.append(torch.stack(per_key, dim=1))
    return torch.stack(banks, dim=0)


def pool_kernel_bank_by_relative_frame(
    bank: torch.Tensor,
    query_frame_ids: Sequence[int],
    key_frame_ids: Sequence[int],
) -> torch.Tensor:
    """Share kernels between frame pairs with the same signed frame offset."""

    if bank.ndim != 5:
        raise ValueError("bank must have shape [q_frames, groups, k_frames, heads, bins]")
    if bank.shape[0] != len(query_frame_ids) or bank.shape[2] != len(key_frame_ids):
        raise ValueError("frame IDs do not match the kernel bank")
    pairs: dict[int, list[tuple[int, int]]] = {}
    for query_index, query_frame in enumerate(query_frame_ids):
        for key_index, key_frame in enumerate(key_frame_ids):
            offset = int(key_frame) - int(query_frame)
            pairs.setdefault(offset, []).append((query_index, key_index))
    output = torch.empty_like(bank)
    for members in pairs.values():
        pooled = torch.stack(
            [bank[query_index, :, key_index] for query_index, key_index in members]
        ).mean(dim=0)
        for query_index, key_index in members:
            output[query_index, :, key_index] = pooled
    return output


def structured_attention_from_kernel_bank(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bank: torch.Tensor,
    layout: CapturedQueryLayout,
    key_frames: int,
    height: int,
    width: int,
    periodic: bool,
    exact_frame_indices: Sequence[int] = (),
    event_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine exact and structured logits under one stable softmax."""

    spatial = height * width
    if key.shape != value.shape:
        raise ValueError("this probe requires matching K/V dimensions")
    if key.shape[0] != key_frames * spatial:
        raise ValueError("flat key/value tensors do not match the grid")
    if bank.shape[0] != layout.query_frames or bank.shape[2] != key_frames:
        raise ValueError("kernel bank frame dimensions do not match")
    if bank.shape[3] != query.shape[1]:
        raise ValueError("kernel bank head dimension does not match")
    if bank.shape[1] not in {1, layout.tile_count}:
        raise ValueError("kernel bank must use global or captured-tile query groups")
    if event_mask is None:
        event_mask = torch.zeros(
            (key_frames, spatial), dtype=torch.bool, device=query.device
        )
    if event_mask.shape != (key_frames, spatial) or event_mask.dtype != torch.bool:
        raise ValueError("event mask must be bool [key_frames, spatial]")
    exact_frames = set(int(frame) for frame in exact_frame_indices)
    if any(frame < 0 or frame >= key_frames for frame in exact_frames):
        raise ValueError("exact frame index is outside the key cache")

    query_view = query.reshape(
        layout.query_frames, layout.queries_per_frame, query.shape[1], query.shape[2]
    )
    key_view = key.reshape(key_frames, spatial, key.shape[1], key.shape[2]).float()
    key_flat = key_view.reshape(-1, key.shape[1], key.shape[2])
    value_flat = value.float()
    output = torch.empty_like(query_view)
    scale = 1.0 / math.sqrt(query.shape[-1])
    exact_mask = event_mask.clone()
    for frame in exact_frames:
        exact_mask[frame] = True
    exact_flat = exact_mask.reshape(-1)
    group_ids = (
        torch.zeros_like(layout.tile_ids_per_frame)
        if bank.shape[1] == 1
        else layout.tile_ids_per_frame
    )
    for query_frame in range(layout.query_frames):
        logit_parts = [
            logits_from_spatial_kernel(
                bank[query_frame, :, key_frame],
                layout.positions_per_frame,
                group_ids,
                height,
                width,
                periodic,
            )
            for key_frame in range(key_frames)
        ]
        logits = torch.cat(logit_parts, dim=-1)
        if bool(exact_flat.any()):
            true_logits = torch.einsum(
                "qhd,khd->hqk",
                query_view[query_frame].float(),
                key_flat[exact_flat],
            ) * scale
            logits[:, :, exact_flat] = true_logits
        row_max = logits.amax(dim=-1, keepdim=True)
        weights = torch.exp(logits - row_max)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        output[query_frame] = torch.einsum(
            "hqk,khd->qhd", weights, value_flat
        ) / denominator.transpose(0, 1)
    return output.reshape_as(query).to(value.dtype)


def exact_sink_recent_frames(
    key_frames: int, sink_frames: int, recent_frames: int
) -> tuple[int, ...]:
    if min(key_frames, sink_frames, recent_frames) < 0:
        raise ValueError("frame counts must be non-negative")
    if sink_frames + recent_frames > key_frames:
        raise ValueError("sink and recent regions overlap")
    return tuple(range(sink_frames)) + tuple(range(key_frames - recent_frames, key_frames))


def fft_arithmetic_reduction(
    query_frames: int,
    key_frames: int,
    height: int,
    width: int,
    query_groups: int,
    exact_frames: int,
    event_fraction_of_structured: float,
    periodic: bool,
    fft_constant_factor: float = 4.0,
) -> float:
    """Conservative symbolic FFT-vs-dense pair-cost proxy, not wall time."""

    if not 0.0 <= event_fraction_of_structured <= 1.0:
        raise ValueError("event fraction must be in [0, 1]")
    if exact_frames > key_frames or min(query_frames, key_frames, query_groups) <= 0:
        raise ValueError("invalid frame or group count")
    spatial = height * width
    padded_spatial = spatial if periodic else (2 * height) * (2 * width)
    structured_frames = key_frames - exact_frames
    exact_equivalent = exact_frames + event_fraction_of_structured * structured_frames
    structured_equivalent = structured_frames * (1.0 - event_fraction_of_structured)
    dense_cost = query_frames * key_frames * spatial * spatial
    exact_cost = query_frames * exact_equivalent * spatial * spatial
    fft_cost = (
        query_frames
        * structured_equivalent
        * fft_constant_factor
        * query_groups
        * padded_spatial
        * math.log2(padded_spatial)
    )
    return float(dense_cost / (exact_cost + fft_cost))
