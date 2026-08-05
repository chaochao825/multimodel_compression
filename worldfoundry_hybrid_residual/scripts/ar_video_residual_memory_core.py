"""Core operators for the LongLive causal residual-memory probe.

The module keeps the numerical object explicit: exact and summarized branches
share one softmax normalizer. A temporal summary preserves every spatial token
position and only reduces the frame axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TemporalCompressionPlan:
    """Disjoint exact frames and temporally summarized frame groups."""

    exact_frames: tuple[int, ...]
    summary_groups: tuple[tuple[int, ...], ...]

    @property
    def covered_frames(self) -> tuple[int, ...]:
        frames = list(self.exact_frames)
        for group in self.summary_groups:
            frames.extend(group)
        return tuple(sorted(frames))


@dataclass(frozen=True)
class RepresentativeSet:
    """K/V representatives and their positive softmax multiplicities."""

    key: torch.Tensor
    value: torch.Tensor
    log_multiplicity: torch.Tensor
    exact_token_count: int
    summary_token_count: int

    @property
    def token_count(self) -> int:
        return int(self.key.shape[0])


def make_recency_plan(
    num_frames: int,
    sink_frames: int,
    recent_frames: int,
    max_summary_groups: int,
) -> TemporalCompressionPlan:
    """Build power-of-two recency groups over non-exact middle frames.

    Groups nearest the recent exact window are finest. The oldest residual
    interval is merged into the final group so the group count is bounded.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sink_frames < 0 or recent_frames < 0:
        raise ValueError("sink_frames and recent_frames must be non-negative")
    if sink_frames + recent_frames > num_frames:
        raise ValueError("exact frame regions overlap")
    if max_summary_groups < 0:
        raise ValueError("max_summary_groups must be non-negative")

    sink = tuple(range(sink_frames))
    recent_start = num_frames - recent_frames
    recent = tuple(range(recent_start, num_frames))
    middle = list(range(sink_frames, recent_start))
    if not middle:
        return TemporalCompressionPlan(tuple(sorted(sink + recent)), ())
    if max_summary_groups == 0:
        return TemporalCompressionPlan(tuple(sorted(sink + recent)), ())

    groups_newest_first: list[tuple[int, ...]] = []
    group_size = 1
    remaining = middle
    while remaining and len(groups_newest_first) < max_summary_groups:
        slots_left = max_summary_groups - len(groups_newest_first)
        if slots_left == 1:
            take = len(remaining)
        else:
            take = min(group_size, len(remaining) - (slots_left - 1))
            take = max(1, take)
        group = tuple(remaining[-take:])
        remaining = remaining[:-take]
        groups_newest_first.append(group)
        group_size *= 2

    if remaining:
        oldest = tuple(remaining) + groups_newest_first[-1]
        groups_newest_first[-1] = oldest

    groups = tuple(reversed(groups_newest_first))
    plan = TemporalCompressionPlan(tuple(sorted(sink + recent)), groups)
    expected = tuple(range(num_frames))
    if plan.covered_frames != expected:
        raise AssertionError("temporal plan does not cover every frame exactly once")
    return plan


def _validate_kv(key: torch.Tensor, value: torch.Tensor) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("key/value must have shape [frames, spatial, heads, dim]")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("key/value frame, spatial, and head axes must match")
    if key.device != value.device:
        raise ValueError("key and value must be on the same device")


def _rope_multipliers(
    frame_ids: Sequence[int],
    height: int,
    width: int,
    head_dim: int,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Return Wan 3D-RoPE complex multipliers for absolute frame IDs."""

    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if height <= 0 or width <= 0 or not frame_ids:
        raise ValueError("frame IDs and spatial dimensions must be non-empty")
    complex_dim = head_dim // 2
    if rope_freqs.ndim != 2 or rope_freqs.shape[1] != complex_dim:
        raise ValueError("RoPE frequency table does not match the head dimension")
    if not rope_freqs.is_complex():
        raise ValueError("RoPE frequency table must be complex")
    if min(frame_ids) < 0 or max(max(frame_ids), height - 1, width - 1) >= rope_freqs.shape[0]:
        raise ValueError("RoPE position is outside the captured frequency table")

    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    spatial_dim = complex_dim // 3
    temporal, vertical, horizontal = rope_freqs.split(
        [temporal_dim, spatial_dim, spatial_dim], dim=1
    )
    frame_index = torch.tensor(frame_ids, device=rope_freqs.device, dtype=torch.long)
    temporal_part = temporal.index_select(0, frame_index)
    multiplier = torch.cat(
        [
            temporal_part[:, None, None, :].expand(-1, height, width, -1),
            vertical[:height][None, :, None, :].expand(len(frame_ids), -1, width, -1),
            horizontal[:width][None, None, :, :].expand(len(frame_ids), height, -1, -1),
        ],
        dim=-1,
    )
    return multiplier.reshape(len(frame_ids), height * width, complex_dim)


def phase_align_keys_for_temporal_summaries(
    key: torch.Tensor,
    plan: TemporalCompressionPlan,
    absolute_frame_ids: Sequence[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Move summary-group K vectors to a common temporal RoPE center.

    Exact-frame and event-token keys must still come from the original `key`.
    The returned tensor is only intended for computing temporal summary means.
    """

    if key.ndim != 4:
        raise ValueError("key must have shape [frames, spatial, heads, dim]")
    if key.shape[0] != len(absolute_frame_ids):
        raise ValueError("absolute frame IDs do not match the key frame axis")
    if key.shape[1] != height * width:
        raise ValueError("height and width do not match the key spatial axis")
    if len(set(absolute_frame_ids)) != len(absolute_frame_ids) or tuple(
        sorted(absolute_frame_ids)
    ) != tuple(absolute_frame_ids):
        raise ValueError("absolute frame IDs must be strictly increasing")

    multipliers = _rope_multipliers(
        absolute_frame_ids, height, width, key.shape[-1], rope_freqs
    )
    key_complex = torch.view_as_complex(
        key.float().reshape(*key.shape[:-1], key.shape[-1] // 2, 2).contiguous()
    )
    canonical = key_complex * multipliers.conj().unsqueeze(2)
    aligned = key_complex.clone()
    for group in plan.summary_groups:
        group_absolute = [int(absolute_frame_ids[index]) for index in group]
        center = int(math.floor(sum(group_absolute) / len(group_absolute) + 0.5))
        center_multiplier = _rope_multipliers(
            [center], height, width, key.shape[-1], rope_freqs
        )[0]
        indices = torch.tensor(group, device=key.device, dtype=torch.long)
        centered = canonical.index_select(0, indices) * center_multiplier[None, :, None, :]
        aligned.index_copy_(0, indices, centered)
    return torch.view_as_real(aligned).flatten(-2).to(key.dtype)


def select_residual_event_tiles(
    key: torch.Tensor,
    value: torch.Tensor,
    plan: TemporalCompressionPlan,
    tile_size: int,
    tile_fraction: float,
    query: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select high-residual temporal-spatial tiles without dense AV labels.

    Scores combine K deviation from the temporal group mean and V deviation.
    Query norm only rescales the K term; no dense attention output is used.
    """

    _validate_kv(key, value)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= tile_fraction <= 1.0:
        raise ValueError("tile_fraction must be in [0, 1]")

    frames, spatial, _, dim = key.shape
    event_mask = torch.zeros((frames, spatial), dtype=torch.bool, device=key.device)
    candidates: list[tuple[float, int, int, int]] = []
    query_scale = 1.0
    if query is not None:
        if query.ndim != 3 or query.shape[-1] != dim:
            raise ValueError("query must have shape [queries, heads, key_dim]")
        query_scale = float(query.float().norm(dim=-1).mean().item() / math.sqrt(dim))

    for group in plan.summary_groups:
        indices = torch.tensor(group, device=key.device)
        group_key = key.index_select(0, indices).float()
        group_value = value.index_select(0, indices).float()
        mean_key = group_key.mean(dim=0, keepdim=True)
        mean_value = group_value.mean(dim=0, keepdim=True)
        key_score = (group_key - mean_key).square().mean(dim=(-1, -2)).sqrt()
        value_score = (group_value - mean_value).square().mean(dim=(-1, -2)).sqrt()
        token_score = query_scale * key_score + value_score
        for local_frame, frame in enumerate(group):
            for start in range(0, spatial, tile_size):
                end = min(start + tile_size, spatial)
                score = float(token_score[local_frame, start:end].mean().item())
                candidates.append((score, frame, start, end))

    if not candidates or tile_fraction == 0.0:
        return event_mask
    budget = int(math.ceil(len(candidates) * tile_fraction))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, frame, start, end in candidates[:budget]:
        event_mask[frame, start:end] = True
    return event_mask


def build_representatives(
    key: torch.Tensor,
    value: torch.Tensor,
    plan: TemporalCompressionPlan,
    event_mask: torch.Tensor | None = None,
    summary_key: torch.Tensor | None = None,
) -> RepresentativeSet:
    """Create exact tokens and spatially aligned temporal mean summaries."""

    _validate_kv(key, value)
    frames, spatial, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if event_mask is None:
        event_mask = torch.zeros((frames, spatial), dtype=torch.bool, device=key.device)
    if event_mask.shape != (frames, spatial) or event_mask.dtype != torch.bool:
        raise ValueError("event_mask must be bool with shape [frames, spatial]")
    if summary_key is None:
        summary_key = key
    if summary_key.shape != key.shape or summary_key.device != key.device:
        raise ValueError("summary_key must match the key shape and device")

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    log_parts: list[torch.Tensor] = []
    exact_count = 0
    summary_count = 0

    if plan.exact_frames:
        exact_indices = torch.tensor(plan.exact_frames, device=key.device)
        exact_key = key.index_select(0, exact_indices).reshape(-1, heads, key_dim)
        exact_value = value.index_select(0, exact_indices).reshape(-1, heads, value_dim)
        key_parts.append(exact_key)
        value_parts.append(exact_value)
        log_parts.append(torch.zeros(exact_key.shape[0], device=key.device))
        exact_count += int(exact_key.shape[0])

    for group in plan.summary_groups:
        indices = torch.tensor(group, device=key.device)
        group_key = key.index_select(0, indices)
        group_summary_key = summary_key.index_select(0, indices)
        group_value = value.index_select(0, indices)
        group_event = event_mask.index_select(0, indices)

        if bool(group_event.any()):
            selected_key = group_key[group_event]
            selected_value = group_value[group_event]
            key_parts.append(selected_key)
            value_parts.append(selected_value)
            log_parts.append(torch.zeros(selected_key.shape[0], device=key.device))
            exact_count += int(selected_key.shape[0])

        keep = ~group_event
        counts = keep.sum(dim=0)
        valid = counts > 0
        if bool(valid.any()):
            key_sum = (group_summary_key * keep[..., None, None]).sum(dim=0)
            value_sum = (group_value * keep[..., None, None]).sum(dim=0)
            denom = counts.clamp_min(1).to(key.dtype)[..., None, None]
            mean_key = (key_sum / denom)[valid]
            mean_value = (value_sum / denom.to(value.dtype))[valid]
            key_parts.append(mean_key)
            value_parts.append(mean_value)
            log_parts.append(counts[valid].float().log())
            summary_count += int(mean_key.shape[0])

    if not key_parts:
        raise ValueError("compression plan produced no representatives")
    rep_key = torch.cat(key_parts, dim=0)
    rep_value = torch.cat(value_parts, dim=0)
    log_multiplicity = torch.cat(log_parts, dim=0)
    return RepresentativeSet(
        key=rep_key,
        value=rep_value,
        log_multiplicity=log_multiplicity,
        exact_token_count=exact_count,
        summary_token_count=summary_count,
    )


def attention_from_representatives(
    query: torch.Tensor,
    representatives: RepresentativeSet,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Compute attention with one shared, numerically stable normalizer."""

    if query.ndim != 3:
        raise ValueError("query must have shape [queries, heads, dim]")
    key = representatives.key
    value = representatives.value
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("representative key/value must have shape [tokens, heads, dim]")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[:2] != value.shape[:2]:
        raise ValueError("key and value token/head dimensions must match")
    if representatives.log_multiplicity.shape != (key.shape[0],):
        raise ValueError("log multiplicity shape mismatch")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    scale = 1.0 / math.sqrt(query.shape[-1])
    outputs: list[torch.Tensor] = []
    key_f = key.float()
    value_f = value.float()
    log_weight = representatives.log_multiplicity.float()[None, None, :]
    for start in range(0, query.shape[0], query_chunk_size):
        query_f = query[start:start + query_chunk_size].float()
        logits = torch.einsum("qhd,khd->hqk", query_f, key_f) * scale
        logits = logits + log_weight
        row_max = logits.amax(dim=-1, keepdim=True)
        weights = torch.exp(logits - row_max)
        denominator = weights.sum(dim=-1, keepdim=True)
        output = torch.einsum("hqk,khd->qhd", weights, value_f)
        outputs.append(output / denominator.transpose(0, 1))
    return torch.cat(outputs, dim=0).to(value.dtype)


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Reference attention over flat [tokens, heads, dim] K/V."""

    reps = RepresentativeSet(
        key=key,
        value=value,
        log_multiplicity=torch.zeros(key.shape[0], device=key.device),
        exact_token_count=int(key.shape[0]),
        summary_token_count=0,
    )
    return attention_from_representatives(query, reps, query_chunk_size)


def relative_l2_by_head(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Return one relative L2 error per head."""

    if target.shape != estimate.shape or target.ndim != 3:
        raise ValueError("target and estimate must match [queries, heads, dim]")
    numerator = (target.float() - estimate.float()).square().sum(dim=(0, 2)).sqrt()
    denominator = target.float().square().sum(dim=(0, 2)).sqrt().clamp_min(1e-12)
    return numerator / denominator


def adaptive_rank_projection(defect: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-head best rank-r projection of a query-by-output defect matrix."""

    if defect.ndim != 3:
        raise ValueError("defect must have shape [queries, heads, value_dim]")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    projected = torch.zeros_like(defect)
    for head in range(defect.shape[1]):
        matrix = defect[:, head].float()
        effective_rank = min(rank, matrix.shape[0], matrix.shape[1])
        if effective_rank == 0:
            continue
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        approximation = (u[:, :effective_rank] * singular[:effective_rank]) @ vh[:effective_rank]
        projected[:, head] = approximation.to(defect.dtype)
    return projected


def fit_output_basis(defects: Iterable[torch.Tensor], rank: int) -> torch.Tensor:
    """Fit a calibration-only output-channel basis for each attention head."""

    defect_list = list(defects)
    if not defect_list:
        raise ValueError("at least one calibration defect is required")
    reference_shape = defect_list[0].shape[1:]
    if any(item.ndim != 3 or item.shape[1:] != reference_shape for item in defect_list):
        raise ValueError("all defects must share head and value dimensions")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    combined = torch.cat([item.float() for item in defect_list], dim=0)
    heads, value_dim = reference_shape
    effective_rank = min(rank, combined.shape[0], value_dim)
    basis = torch.zeros((heads, value_dim, effective_rank), device=combined.device)
    for head in range(heads):
        _, _, vh = torch.linalg.svd(combined[:, head], full_matrices=False)
        basis[head] = vh[:effective_rank].transpose(0, 1)
    return basis


def project_onto_output_basis(defect: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project a defect onto a frozen per-head output-channel basis."""

    if defect.ndim != 3 or basis.ndim != 3:
        raise ValueError("defect and basis must be rank-3 tensors")
    if defect.shape[1] != basis.shape[0] or defect.shape[2] != basis.shape[1]:
        raise ValueError("defect and basis dimensions do not match")
    coefficients = torch.einsum("qhd,hdr->qhr", defect.float(), basis.float())
    projection = torch.einsum("qhr,hdr->qhd", coefficients, basis.float())
    return projection.to(defect.dtype)


def arithmetic_reduction(dense_tokens: int, representative_tokens: int) -> float:
    if dense_tokens <= 0 or representative_tokens <= 0:
        raise ValueError("token counts must be positive")
    return float(dense_tokens / representative_tokens)
