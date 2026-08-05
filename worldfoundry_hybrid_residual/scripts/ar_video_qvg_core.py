"""Numerical helpers for the LongLive QuantVideoGen probe."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ar_video_residual_memory_core import _rope_multipliers


def transform_key_rope(
    key: torch.Tensor,
    absolute_frame_ids: Sequence[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply or remove captured 3D RoPE on `[frames, spatial, heads, dim]` K."""

    if key.ndim != 4:
        raise ValueError("key must have shape [frames, spatial, heads, dim]")
    if key.shape[0] != len(absolute_frame_ids):
        raise ValueError("absolute frame IDs do not match the key frame axis")
    if key.shape[1] != height * width:
        raise ValueError("height and width do not match the key spatial axis")
    multipliers = _rope_multipliers(
        absolute_frame_ids, height, width, key.shape[-1], rope_freqs
    )
    key_complex = torch.view_as_complex(
        key.float().reshape(*key.shape[:-1], key.shape[-1] // 2, 2).contiguous()
    )
    factor = multipliers.conj() if inverse else multipliers
    transformed = key_complex * factor.unsqueeze(2)
    return torch.view_as_real(transformed).flatten(-2).to(key.dtype)


def quantized_frame_indices(
    key_frames: int,
    exact_policy: str,
    sink_frames: int,
    recent_frames: int,
) -> tuple[int, ...]:
    """Return the frames encoded by a low-bit representation."""

    if key_frames <= 0:
        raise ValueError("key_frames must be positive")
    if min(sink_frames, recent_frames) < 0:
        raise ValueError("exact frame counts must be non-negative")
    if sink_frames + recent_frames > key_frames:
        raise ValueError("sink and recent exact regions overlap")
    if exact_policy == "none":
        return tuple(range(key_frames))
    if exact_policy != "sink_recent":
        raise ValueError(f"unsupported exact policy: {exact_policy}")
    return tuple(range(sink_frames, key_frames - recent_frames))


def gather_frames_for_qvg(tensor: torch.Tensor, frame_indices: Sequence[int]) -> torch.Tensor:
    """Convert selected `[F,S,H,D]` frames to QVG's `[1,H,F*S,D]` layout."""

    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [frames, spatial, heads, dim]")
    if not frame_indices:
        raise ValueError("at least one frame must be quantized")
    index = torch.tensor(frame_indices, device=tensor.device, dtype=torch.long)
    selected = tensor.index_select(0, index)
    return selected.permute(2, 0, 1, 3).reshape(
        1, tensor.shape[2], len(frame_indices) * tensor.shape[1], tensor.shape[3]
    ).contiguous()


def scatter_qvg_frames(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    frame_indices: Sequence[int],
) -> torch.Tensor:
    """Insert `[1,H,F*S,D]` reconstructions into `[F,S,H,D]` original data."""

    if original.ndim != 4 or reconstructed.ndim != 4:
        raise ValueError("original and reconstructed tensors must be rank four")
    frames, spatial, heads, dim = original.shape
    expected = (1, heads, len(frame_indices) * spatial, dim)
    if tuple(reconstructed.shape) != expected:
        raise ValueError(
            f"reconstructed shape {tuple(reconstructed.shape)} does not match {expected}"
        )
    output = original.clone()
    restored = reconstructed.reshape(1, heads, len(frame_indices), spatial, dim)[0]
    restored = restored.permute(1, 2, 0, 3).contiguous()
    index = torch.tensor(frame_indices, device=original.device, dtype=torch.long)
    output.index_copy_(0, index, restored.to(original.dtype))
    return output


def tensor_tree_nbytes(value: Any) -> int:
    """Count tensor storage in a nested packed state, without counting Python metadata."""

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def logical_rtn_bytes(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
    scale_bytes: int = 1,
) -> int:
    """Packed payload plus one scale per last-dimension quantization block."""

    if bits not in {2, 4, 8}:
        raise ValueError("bits must be 2, 4, or 8")
    if block_size <= 0 or tensor.shape[-1] % block_size:
        raise ValueError("block size must divide the last dimension")
    payload = math.ceil(tensor.numel() * bits / 8)
    scales = tensor.numel() // block_size * scale_bytes
    return int(payload + scales)


def relative_l2_by_head(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Relative L2 over every axis except the head axis of `[F,S,H,D]`."""

    if reference.shape != estimate.shape or reference.ndim != 4:
        raise ValueError("reference and estimate must match [frames, spatial, heads, dim]")
    difference = (reference.float() - estimate.float()).square().sum(dim=(0, 1, 3))
    energy = reference.float().square().sum(dim=(0, 1, 3)).clamp_min(1e-24)
    return (difference / energy).sqrt()


def compression_ratio(reference_tensors: Sequence[torch.Tensor], packed_bytes: int) -> float:
    if packed_bytes <= 0:
        raise ValueError("packed byte count must be positive")
    reference_bytes = sum(
        int(tensor.numel() * tensor.element_size()) for tensor in reference_tensors
    )
    return float(reference_bytes / packed_bytes)
