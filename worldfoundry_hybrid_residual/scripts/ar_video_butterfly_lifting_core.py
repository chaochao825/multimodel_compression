"""Causal Butterfly-lifting compression for autoregressive video K/V memory.

The transform acts on cached K/V, not on the attention probability matrix.  A
dyadic lifting tree keeps one coarse spatial map and tile-sparse detail maps.
Optional cyclic shifts predict motion between sibling nodes; discarded details
therefore remove only prediction residuals.  Reconstructed K/V are read by the
original attention operator with its single softmax normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch


Shift = tuple[int, int]


@dataclass
class LiftingNode:
    """One node in an invertible binary lifting tree."""

    coarse_key: torch.Tensor
    coarse_value: torch.Tensor
    frame_indices: tuple[int, ...]
    left: "LiftingNode | None" = None
    right: "LiftingNode | None" = None
    detail_key: torch.Tensor | None = None
    detail_value: torch.Tensor | None = None
    shifts: torch.Tensor | None = None
    shift_scope: str | None = None
    window_shape: tuple[int, int] | None = None
    window_offset: tuple[int, int] = (0, 0)

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass(frozen=True)
class DetailSelection:
    """Tile masks and accounting for a sparsified lifting tree."""

    masks: dict[int, torch.Tensor]
    candidate_blocks: int
    retained_blocks: int
    retained_tokens: int


@dataclass(frozen=True)
class StorageEstimate:
    dense_bytes: int
    compressed_bytes: int
    exact_bytes: int
    coarse_bytes: int
    detail_bytes: int
    metadata_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / max(self.compressed_bytes, 1)


def _validate_kv(key: torch.Tensor, value: torch.Tensor) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("key/value must have shape [frames, spatial, heads, dim]")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("key/value frame, spatial, and head axes must match")
    if key.device != value.device:
        raise ValueError("key and value must share a device")


def middle_frame_indices(
    frames: int, sink_frames: int, recent_frames: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return disjoint exact and transform-coded frame indices."""

    if frames <= 0:
        raise ValueError("frames must be positive")
    if min(sink_frames, recent_frames) < 0:
        raise ValueError("sink/recent frame counts must be non-negative")
    if sink_frames + recent_frames > frames:
        raise ValueError("sink and recent regions overlap")
    exact = tuple(range(sink_frames)) + tuple(range(frames - recent_frames, frames))
    middle = tuple(range(sink_frames, frames - recent_frames))
    return exact, middle


def _rope_multipliers(
    frame_ids: Sequence[int],
    height: int,
    width: int,
    head_dim: int,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    if not frame_ids or min(height, width) <= 0 or head_dim % 2:
        raise ValueError("invalid RoPE geometry")
    complex_dim = head_dim // 2
    if rope_freqs.ndim != 2 or rope_freqs.shape[1] != complex_dim:
        raise ValueError("RoPE table does not match the head dimension")
    if not rope_freqs.is_complex():
        raise ValueError("RoPE table must be complex")
    if min(frame_ids) < 0 or max(max(frame_ids), height - 1, width - 1) >= rope_freqs.shape[0]:
        raise ValueError("RoPE position lies outside the frequency table")
    spatial_dim = complex_dim // 3
    temporal_dim = complex_dim - 2 * spatial_dim
    temporal, vertical, horizontal = rope_freqs.split(
        [temporal_dim, spatial_dim, spatial_dim], dim=1
    )
    frame_index = torch.tensor(frame_ids, device=rope_freqs.device, dtype=torch.long)
    return torch.cat(
        [
            temporal.index_select(0, frame_index)[:, None, None, :].expand(
                -1, height, width, -1
            ),
            vertical[:height][None, :, None, :].expand(len(frame_ids), -1, width, -1),
            horizontal[:width][None, None, :, :].expand(len(frame_ids), height, -1, -1),
        ],
        dim=-1,
    ).reshape(len(frame_ids), height * width, complex_dim)


def canonicalize_rope_keys(
    key: torch.Tensor,
    absolute_frame_ids: Sequence[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Remove captured 3D RoPE from post-RoPE keys."""

    if key.ndim != 4 or key.shape[0] != len(absolute_frame_ids):
        raise ValueError("key/frame IDs must match [frames, spatial, heads, dim]")
    if key.shape[1] != height * width:
        raise ValueError("key spatial axis does not match height x width")
    multiplier = _rope_multipliers(
        absolute_frame_ids, height, width, key.shape[-1], rope_freqs
    )
    complex_key = torch.view_as_complex(
        key.float().reshape(*key.shape[:-1], key.shape[-1] // 2, 2).contiguous()
    )
    canonical = complex_key * multiplier.conj().unsqueeze(2)
    return torch.view_as_real(canonical).flatten(-2)


def restore_rope_keys(
    canonical_key: torch.Tensor,
    absolute_frame_ids: Sequence[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply captured 3D RoPE to canonical keys."""

    if canonical_key.ndim != 4 or canonical_key.shape[0] != len(absolute_frame_ids):
        raise ValueError("canonical key/frame IDs do not match")
    multiplier = _rope_multipliers(
        absolute_frame_ids, height, width, canonical_key.shape[-1], rope_freqs
    )
    complex_key = torch.view_as_complex(
        canonical_key.float()
        .reshape(*canonical_key.shape[:-1], canonical_key.shape[-1] // 2, 2)
        .contiguous()
    )
    restored = complex_key * multiplier.unsqueeze(2)
    return torch.view_as_real(restored).flatten(-2).to(dtype)


def apply_cyclic_shift(field: torch.Tensor, shifts: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Apply one shared or one-per-head invertible 2D cyclic shift."""

    if field.ndim != 3 or field.shape[0] != height * width:
        raise ValueError("field must have shape [height*width, heads, dim]")
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape [1|heads, 2]")
    heads = field.shape[1]
    if shifts.shape[0] not in (1, heads):
        raise ValueError("shift count must be one or match heads")
    view = field.reshape(height, width, heads, field.shape[-1])
    if shifts.shape[0] == 1:
        dy, dx = (int(value) for value in shifts[0].tolist())
        return torch.roll(view, shifts=(dy, dx), dims=(0, 1)).reshape_as(field)
    output = torch.empty_like(view)
    for head in range(heads):
        dy, dx = (int(value) for value in shifts[head].tolist())
        output[:, :, head] = torch.roll(
            view[:, :, head], shifts=(dy, dx), dims=(0, 1)
        )
    return output.reshape_as(field)


def _window_slices(
    height: int, width: int, window_shape: tuple[int, int]
) -> list[tuple[slice, slice]]:
    window_height, window_width = window_shape
    if min(window_height, window_width) <= 0:
        raise ValueError("window dimensions must be positive")
    return [
        (
            slice(y, min(y + window_height, height)),
            slice(x, min(x + window_width, width)),
        )
        for y in range(0, height, window_height)
        for x in range(0, width, window_width)
    ]


def apply_window_cyclic_shift(
    field: torch.Tensor,
    shifts: torch.Tensor,
    height: int,
    width: int,
    window_shape: tuple[int, int],
    window_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Apply invertible per-window shifts under a shifted-window partition."""

    if field.ndim != 3 or field.shape[0] != height * width:
        raise ValueError("field must have shape [height*width, heads, dim]")
    windows = _window_slices(height, width, window_shape)
    if shifts.shape != (len(windows), 2):
        raise ValueError("window shifts must have shape [windows, 2]")
    offset_y, offset_x = window_offset
    view = field.reshape(height, width, field.shape[1], field.shape[2])
    partitioned = torch.roll(view, shifts=(-offset_y, -offset_x), dims=(0, 1))
    output = torch.empty_like(partitioned)
    for index, (vertical, horizontal) in enumerate(windows):
        dy, dx = (int(value) for value in shifts[index].tolist())
        output[vertical, horizontal] = torch.roll(
            partitioned[vertical, horizontal], shifts=(dy, dx), dims=(0, 1)
        )
    return torch.roll(output, shifts=(offset_y, offset_x), dims=(0, 1)).reshape_as(field)


def apply_prediction_shift(
    field: torch.Tensor,
    shifts: torch.Tensor,
    height: int,
    width: int,
    scope: str,
    window_shape: tuple[int, int] | None = None,
    window_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    if scope in {"identity", "shared", "per_head"}:
        return apply_cyclic_shift(field, shifts, height, width)
    if scope == "window_shared":
        if window_shape is None:
            raise ValueError("window_shared prediction requires a window shape")
        return apply_window_cyclic_shift(
            field, shifts, height, width, window_shape, window_offset
        )
    raise ValueError(f"unsupported shift scope: {scope}")


def invert_shifts(shifts: torch.Tensor) -> torch.Tensor:
    return -shifts


def choose_prediction_shifts(
    left_key: torch.Tensor,
    left_value: torch.Tensor,
    right_key: torch.Tensor,
    right_value: torch.Tensor,
    candidates: Sequence[Shift],
    height: int,
    width: int,
    scope: str,
    key_weight: float = 1.0,
    value_weight: float = 1.0,
    window_shape: tuple[int, int] | None = None,
    window_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Choose shifts from runtime K/V reconstruction energy only."""

    if scope not in {"identity", "shared", "per_head", "window_shared"}:
        raise ValueError(f"unsupported shift scope: {scope}")
    if not candidates:
        raise ValueError("at least one shift candidate is required")
    if min(key_weight, value_weight) < 0 or key_weight + value_weight <= 0:
        raise ValueError("K/V weights must be non-negative and not both zero")
    heads = left_key.shape[1]
    device = left_key.device
    if scope == "identity":
        return torch.zeros((1, 2), dtype=torch.long, device=device)
    if scope == "window_shared":
        if window_shape is None:
            raise ValueError("window_shared prediction requires a window shape")
        windows = _window_slices(height, width, window_shape)
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        per_candidate = []
        for shift in candidates:
            repeated = torch.tensor(
                [shift] * len(windows), dtype=torch.long, device=device
            )
            key_residual = right_key.float() - apply_window_cyclic_shift(
                left_key.float(),
                repeated,
                height,
                width,
                window_shape,
                window_offset,
            )
            value_residual = right_value.float() - apply_window_cyclic_shift(
                left_value.float(),
                repeated,
                height,
                width,
                window_shape,
                window_offset,
            )
            token_loss = (
                key_weight * key_residual.square().mean(dim=(1, 2))
                + value_weight * value_residual.square().mean(dim=(1, 2))
            ).reshape(height, width)
            partitioned_loss = torch.roll(
                token_loss, shifts=(-window_offset[0], -window_offset[1]), dims=(0, 1)
            )
            per_candidate.append(
                torch.stack(
                    [partitioned_loss[v, h].mean() for v, h in windows], dim=0
                )
            )
        selected = torch.stack(per_candidate, dim=0).argmin(dim=0)
        return candidate_tensor.index_select(0, selected)
    losses = []
    for shift in candidates:
        shift_tensor = torch.tensor([shift], dtype=torch.long, device=device)
        key_residual = right_key.float() - apply_prediction_shift(
            left_key.float(), shift_tensor, height, width, scope
        )
        value_residual = right_value.float() - apply_prediction_shift(
            left_value.float(), shift_tensor, height, width, scope
        )
        losses.append(
            key_weight * key_residual.square().mean(dim=(0, 2))
            + value_weight * value_residual.square().mean(dim=(0, 2))
        )
    loss = torch.stack(losses, dim=0)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    if scope == "shared":
        return candidate_tensor[loss.mean(dim=1).argmin()].reshape(1, 2)
    selected = loss.argmin(dim=0)
    return candidate_tensor.index_select(0, selected).reshape(heads, 2)


def _merge_nodes(
    left: LiftingNode,
    right: LiftingNode,
    candidates: Sequence[Shift],
    height: int,
    width: int,
    shift_scope: str,
    key_weight: float,
    value_weight: float,
    window_shape: tuple[int, int] | None,
    window_offset: tuple[int, int],
) -> LiftingNode:
    shifts = choose_prediction_shifts(
        left.coarse_key,
        left.coarse_value,
        right.coarse_key,
        right.coarse_value,
        candidates,
        height,
        width,
        shift_scope,
        key_weight,
        value_weight,
        window_shape,
        window_offset,
    )
    predicted_key = apply_prediction_shift(
        left.coarse_key,
        shifts,
        height,
        width,
        shift_scope,
        window_shape,
        window_offset,
    )
    predicted_value = apply_prediction_shift(
        left.coarse_value,
        shifts,
        height,
        width,
        shift_scope,
        window_shape,
        window_offset,
    )
    detail_key = right.coarse_key - predicted_key
    detail_value = right.coarse_value - predicted_value
    inverse = invert_shifts(shifts)
    coarse_key = left.coarse_key + 0.5 * apply_prediction_shift(
        detail_key,
        inverse,
        height,
        width,
        shift_scope,
        window_shape,
        window_offset,
    )
    coarse_value = left.coarse_value + 0.5 * apply_prediction_shift(
        detail_value,
        inverse,
        height,
        width,
        shift_scope,
        window_shape,
        window_offset,
    )
    return LiftingNode(
        coarse_key=coarse_key,
        coarse_value=coarse_value,
        frame_indices=left.frame_indices + right.frame_indices,
        left=left,
        right=right,
        detail_key=detail_key,
        detail_value=detail_value,
        shifts=shifts,
        shift_scope=shift_scope,
        window_shape=window_shape,
        window_offset=window_offset,
    )


def build_lifting_tree(
    key: torch.Tensor,
    value: torch.Tensor,
    frame_indices: Sequence[int],
    candidates: Sequence[Shift],
    height: int,
    width: int,
    shift_scope: str,
    key_weight: float = 1.0,
    value_weight: float = 1.0,
    window_shape: tuple[int, int] | None = None,
    window_offsets: Sequence[tuple[int, int]] = ((0, 0),),
) -> LiftingNode:
    """Build a balanced, causal dyadic lifting tree over selected frames."""

    _validate_kv(key, value)
    if key.shape[0] != len(frame_indices) or not frame_indices:
        raise ValueError("frame indices must match a non-empty K/V frame axis")
    if shift_scope == "window_shared" and window_shape is None:
        raise ValueError("window_shared lifting requires a window shape")
    if not window_offsets:
        raise ValueError("at least one window offset is required")
    nodes = [
        LiftingNode(key[index], value[index], (int(frame_indices[index]),))
        for index in range(key.shape[0])
    ]
    level = 0
    while len(nodes) > 1:
        next_level: list[LiftingNode] = []
        window_offset = tuple(window_offsets[level % len(window_offsets)])
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                next_level.append(nodes[index])
            else:
                next_level.append(
                    _merge_nodes(
                        nodes[index],
                        nodes[index + 1],
                        candidates,
                        height,
                        width,
                        shift_scope,
                        key_weight,
                        value_weight,
                        window_shape,
                        window_offset,
                    )
                )
        nodes = next_level
        level += 1
    return nodes[0]


def iter_merge_nodes(root: LiftingNode) -> Iterable[LiftingNode]:
    if root.is_leaf:
        return
    yield root
    assert root.left is not None and root.right is not None
    yield from iter_merge_nodes(root.left)
    yield from iter_merge_nodes(root.right)


def enumerate_detail_blocks(
    root: LiftingNode, tile_size: int
) -> list[tuple[LiftingNode, int, int]]:
    """Return a deterministic list of regular detail blocks."""

    if tile_size <= 0:
        raise ValueError("tile size must be positive")
    blocks: list[tuple[LiftingNode, int, int]] = []
    for node in iter_merge_nodes(root):
        spatial = node.coarse_key.shape[0]
        blocks.extend(
            (node, start, min(start + tile_size, spatial))
            for start in range(0, spatial, tile_size)
        )
    return blocks


def detail_selection_from_indices(
    root: LiftingNode, tile_size: int, selected_indices: Sequence[int]
) -> DetailSelection:
    """Build a detail selection from deterministic block indices."""

    blocks = enumerate_detail_blocks(root, tile_size)
    selected = tuple(int(index) for index in selected_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("detail block indices must be unique")
    if any(index < 0 or index >= len(blocks) for index in selected):
        raise ValueError("detail block index lies outside the tree")
    masks = {
        id(node): torch.zeros(
            node.coarse_key.shape[0], dtype=torch.bool, device=node.coarse_key.device
        )
        for node in iter_merge_nodes(root)
    }
    retained_tokens = 0
    for index in selected:
        node, start, end = blocks[index]
        masks[id(node)][start:end] = True
        retained_tokens += end - start
    return DetailSelection(
        masks=masks,
        candidate_blocks=len(blocks),
        retained_blocks=len(selected),
        retained_tokens=retained_tokens,
    )


def select_detail_tiles(
    root: LiftingNode,
    tile_size: int,
    fraction: float,
    key_weight: float = 1.0,
    value_weight: float = 1.0,
) -> DetailSelection:
    """Retain the highest-energy regular spatial detail tiles globally."""

    if tile_size <= 0:
        raise ValueError("tile size must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("detail fraction must lie in [0, 1]")
    if min(key_weight, value_weight) < 0 or key_weight + value_weight <= 0:
        raise ValueError("K/V weights must be non-negative and not both zero")
    candidates: list[tuple[float, int]] = []
    blocks = enumerate_detail_blocks(root, tile_size)
    for block_index, (node, start, end) in enumerate(blocks):
        assert node.detail_key is not None and node.detail_value is not None
        token_score = (
            key_weight * node.detail_key.float().square().mean(dim=(1, 2))
            + value_weight * node.detail_value.float().square().mean(dim=(1, 2))
        )
        candidates.append((float(token_score[start:end].mean().item()), block_index))
    budget = min(len(candidates), int(math.ceil(len(candidates) * fraction)))
    selected = [
        index
        for _, index in sorted(candidates, key=lambda item: item[0], reverse=True)[:budget]
    ]
    return detail_selection_from_indices(
        root,
        tile_size,
        selected,
    )


def reconstruct_lifting_tree(
    root: LiftingNode,
    selection: DetailSelection,
    height: int,
    width: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Decode approximate leaf K/V maps from one coarse map and sparse details."""

    keys: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}

    def decode(node: LiftingNode, coarse_key: torch.Tensor, coarse_value: torch.Tensor) -> None:
        if node.is_leaf:
            frame = node.frame_indices[0]
            keys[frame] = coarse_key
            values[frame] = coarse_value
            return
        assert node.left is not None and node.right is not None
        assert node.detail_key is not None and node.detail_value is not None
        assert node.shifts is not None
        assert node.shift_scope is not None
        mask = selection.masks[id(node)][:, None, None]
        detail_key = torch.where(mask, node.detail_key, torch.zeros_like(node.detail_key))
        detail_value = torch.where(
            mask, node.detail_value, torch.zeros_like(node.detail_value)
        )
        inverse = invert_shifts(node.shifts)
        left_key = coarse_key - 0.5 * apply_prediction_shift(
            detail_key,
            inverse,
            height,
            width,
            node.shift_scope,
            node.window_shape,
            node.window_offset,
        )
        left_value = coarse_value - 0.5 * apply_prediction_shift(
            detail_value,
            inverse,
            height,
            width,
            node.shift_scope,
            node.window_shape,
            node.window_offset,
        )
        right_key = apply_prediction_shift(
            left_key,
            node.shifts,
            height,
            width,
            node.shift_scope,
            node.window_shape,
            node.window_offset,
        ) + detail_key
        right_value = (
            apply_prediction_shift(
                left_value,
                node.shifts,
                height,
                width,
                node.shift_scope,
                node.window_shape,
                node.window_offset,
            )
            + detail_value
        )
        decode(node.left, left_key, left_value)
        decode(node.right, right_key, right_value)

    decode(root, root.coarse_key, root.coarse_value)
    return keys, values


def estimate_storage(
    frames: int,
    exact_frames: int,
    spatial: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    root: LiftingNode,
    selection: DetailSelection,
    element_bytes: int = 2,
    index_bytes: int = 4,
    padded_detail_tile_size: int | None = None,
) -> StorageEstimate:
    """Estimate materialized cache bytes, including sparse indices and shifts."""

    if padded_detail_tile_size is not None and padded_detail_tile_size <= 0:
        raise ValueError("padded detail tile size must be positive")
    per_token = heads * (key_dim + value_dim) * element_bytes
    dense = frames * spatial * per_token
    exact = exact_frames * spatial * per_token
    coarse = spatial * per_token
    retained_tokens = selection.retained_tokens
    if padded_detail_tile_size is not None:
        retained_tokens = selection.retained_blocks * padded_detail_tile_size
        if retained_tokens < selection.retained_tokens:
            raise ValueError("padded detail storage cannot be smaller than the payload")
    detail = retained_tokens * per_token
    merge_nodes = list(iter_merge_nodes(root))
    shift_scalars = sum(int(node.shifts.numel()) for node in merge_nodes if node.shifts is not None)
    metadata = selection.retained_blocks * index_bytes + shift_scalars
    return StorageEstimate(
        dense_bytes=dense,
        compressed_bytes=exact + coarse + detail + metadata,
        exact_bytes=exact,
        coarse_bytes=coarse,
        detail_bytes=detail,
        metadata_bytes=metadata,
    )
