"""Core utilities for residual-width-guided episodic write probes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TileCandidate:
    index: int
    frame: int
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start


def enumerate_summary_tiles(
    summary_groups: Iterable[Sequence[int]],
    spatial_tokens: int,
    tile_size: int,
) -> list[TileCandidate]:
    if spatial_tokens <= 0 or tile_size <= 0:
        raise ValueError("spatial_tokens and tile_size must be positive")
    frames = [int(frame) for group in summary_groups for frame in group]
    if len(frames) != len(set(frames)):
        raise ValueError("summary groups must not contain duplicate frames")
    candidates: list[TileCandidate] = []
    for frame in frames:
        if frame < 0:
            raise ValueError("frame indices must be non-negative")
        for start in range(0, spatial_tokens, tile_size):
            candidates.append(
                TileCandidate(
                    index=len(candidates),
                    frame=frame,
                    start=start,
                    end=min(start + tile_size, spatial_tokens),
                )
            )
    return candidates


def selection_budget(candidate_count: int, fraction: float) -> int:
    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return min(candidate_count, int(math.ceil(candidate_count * fraction)))


def select_top_indices(scores: torch.Tensor, budget: int) -> list[int]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if budget < 0 or budget > scores.numel():
        raise ValueError("budget lies outside the score vector")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    ordered = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index]), index),
    )
    return ordered[:budget]


def event_mask_from_indices(
    candidates: Sequence[TileCandidate],
    selected_indices: Iterable[int],
    frames: int,
    spatial_tokens: int,
    device: torch.device | str,
) -> torch.Tensor:
    if frames <= 0 or spatial_tokens <= 0:
        raise ValueError("frames and spatial_tokens must be positive")
    selected = list(selected_indices)
    if len(selected) != len(set(selected)):
        raise ValueError("selected tile indices must be unique")
    mask = torch.zeros(
        (frames, spatial_tokens), dtype=torch.bool, device=torch.device(device)
    )
    for index in selected:
        if index < 0 or index >= len(candidates):
            raise ValueError(f"candidate index outside range: {index}")
        candidate = candidates[index]
        if candidate.frame >= frames or candidate.end > spatial_tokens:
            raise ValueError("candidate lies outside the event mask")
        mask[candidate.frame, candidate.start : candidate.end] = True
    return mask


def indices_from_event_mask(
    candidates: Sequence[TileCandidate], event_mask: torch.Tensor
) -> list[int]:
    if event_mask.ndim != 2 or event_mask.dtype != torch.bool:
        raise ValueError("event_mask must be bool [frames, spatial]")
    selected = []
    for candidate in candidates:
        tile = event_mask[candidate.frame, candidate.start : candidate.end]
        if bool(tile.all()):
            selected.append(candidate.index)
        elif bool(tile.any()):
            raise ValueError("event mask contains a partial candidate tile")
    return selected


def tail_energy_by_head(defect: torch.Tensor, rank: int) -> torch.Tensor:
    """Return best rank-r residual Frobenius energy for each head."""

    if defect.ndim != 3:
        raise ValueError("defect must have shape [queries, heads, value_dim]")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    matrices = defect.float().permute(1, 0, 2)
    singular = torch.linalg.svdvals(matrices)
    used = min(rank, singular.shape[-1])
    return singular[:, used:].square().sum(dim=-1)


def normalized_tail_objective(
    defect: torch.Tensor, reference: torch.Tensor, rank: int
) -> torch.Tensor:
    if reference.shape != defect.shape:
        raise ValueError("reference and defect shapes must match")
    reference_energy = reference.float().square().sum(dim=(0, 2)).clamp_min(1e-24)
    return (tail_energy_by_head(defect, rank) / reference_energy).mean()


def selection_signature(indices: Iterable[int]) -> str:
    normalized = ",".join(str(index) for index in sorted(indices))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0
