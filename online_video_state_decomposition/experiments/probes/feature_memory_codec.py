from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isqrt
from pathlib import Path

import torch


ROUTED_STATE_FORMAT_VERSION = 1
ROUTED_TOKEN_LAYOUT = "row_major_square_grid"


@dataclass(frozen=True)
class LowRankFeatureCodec:
    mean: torch.Tensor
    basis: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("codec mean must be one-dimensional")
        if self.basis.ndim != 2:
            raise ValueError("codec basis must be two-dimensional")
        if self.basis.shape[0] != self.mean.shape[0]:
            raise ValueError("codec mean and basis hidden sizes differ")
        if self.basis.shape[1] <= 0:
            raise ValueError("codec rank must be positive")

    @property
    def hidden_size(self) -> int:
        return int(self.mean.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def parameter_bytes(self) -> int:
        return tensor_bytes(self.mean) + tensor_bytes(self.basis)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, hidden_size=self.hidden_size)
        centered = features.float() - self.mean.float()
        return centered @ self.basis.float()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim < 2 or latents.shape[-1] != self.rank:
            raise ValueError("latent tensor has the wrong rank dimension")
        return (
            latents.float() @ self.basis.float().transpose(0, 1)
            + self.mean.float()
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LowRankFeatureCodec:
        return LowRankFeatureCodec(
            mean=self.mean.to(device=device, dtype=dtype),
            basis=self.basis.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class EncodedFeatureMemory:
    latents: torch.Tensor
    residual_indices: torch.Tensor
    residual_values: torch.Tensor
    tokens_per_frame: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.latents.ndim != 3:
            raise ValueError("latents must have shape [frames, tokens, rank]")
        frames, tokens, _ = self.latents.shape
        if tokens != self.tokens_per_frame:
            raise ValueError("tokens_per_frame does not match latent shape")
        if self.residual_indices.ndim != 2:
            raise ValueError(
                "residual indices must have shape [frames, residual_tokens]"
            )
        if self.residual_values.ndim != 3:
            raise ValueError(
                "residual values must have shape "
                "[frames, residual_tokens, hidden]"
            )
        if self.residual_indices.shape != self.residual_values.shape[:2]:
            raise ValueError("residual index and value shapes differ")
        if self.residual_indices.shape[0] != frames:
            raise ValueError("residual frame count differs from latents")
        if self.residual_values.shape[-1] != self.hidden_size:
            raise ValueError("residual hidden size differs from metadata")
        if self.residual_indices.dtype not in {
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("residual indices must use an integer dtype")

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def rank(self) -> int:
        return int(self.latents.shape[-1])

    @property
    def residual_tokens_per_frame(self) -> int:
        return int(self.residual_indices.shape[1])

    @property
    def latent_bytes(self) -> int:
        return tensor_bytes(self.latents)

    @property
    def residual_value_bytes(self) -> int:
        return tensor_bytes(self.residual_values)

    @property
    def residual_index_bytes(self) -> int:
        return tensor_bytes(self.residual_indices)

    @property
    def stream_state_bytes(self) -> int:
        return (
            self.latent_bytes
            + self.residual_value_bytes
            + self.residual_index_bytes
        )


@dataclass(frozen=True)
class BudgetedResidualFeatureMemory:
    """Low-rank state plus one globally budgeted sparse residual archive."""

    latents: torch.Tensor
    residual_flat_indices: torch.Tensor
    residual_values: torch.Tensor
    tokens_per_frame: int
    hidden_size: int
    allocation: str

    def __post_init__(self) -> None:
        if self.latents.ndim != 3:
            raise ValueError("latents must have shape [frames, tokens, rank]")
        frames, tokens, _ = self.latents.shape
        if frames <= 0 or tokens != self.tokens_per_frame:
            raise ValueError("latent shape does not match memory metadata")
        if self.residual_flat_indices.ndim != 1:
            raise ValueError("residual indices must be one-dimensional")
        if self.residual_values.ndim != 2:
            raise ValueError(
                "residual values must have shape [residual_tokens, hidden]"
            )
        if len(self.residual_flat_indices) != len(self.residual_values):
            raise ValueError("residual index and value counts differ")
        if self.residual_values.shape[-1] != self.hidden_size:
            raise ValueError("residual hidden size differs from metadata")
        if self.residual_flat_indices.dtype not in {
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("residual indices must use an integer dtype")
        if len(self.residual_flat_indices):
            minimum = int(self.residual_flat_indices.min().item())
            maximum = int(self.residual_flat_indices.max().item())
            if minimum < 0 or maximum >= frames * tokens:
                raise ValueError("residual indices are outside the memory")
            if len(torch.unique(self.residual_flat_indices)) != len(
                self.residual_flat_indices
            ):
                raise ValueError("residual indices must be unique")
        if self.allocation not in {
            "global_energy",
            "temporal_novelty",
        }:
            raise ValueError(f"unsupported residual allocation: {self.allocation}")

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def rank(self) -> int:
        return int(self.latents.shape[-1])

    @property
    def residual_token_budget(self) -> int:
        return int(len(self.residual_flat_indices))

    @property
    def latent_bytes(self) -> int:
        return tensor_bytes(self.latents)

    @property
    def residual_value_bytes(self) -> int:
        return tensor_bytes(self.residual_values)

    @property
    def residual_index_bytes(self) -> int:
        return tensor_bytes(self.residual_flat_indices)

    @property
    def stream_state_bytes(self) -> int:
        return (
            self.latent_bytes
            + self.residual_value_bytes
            + self.residual_index_bytes
        )

    def residual_frame_counts(self) -> torch.Tensor:
        if not len(self.residual_flat_indices):
            return torch.zeros(
                self.frames,
                device=self.latents.device,
                dtype=torch.long,
            )
        frame_indices = self.residual_flat_indices.long() // self.tokens_per_frame
        return torch.bincount(frame_indices, minlength=self.frames)


@dataclass(frozen=True)
class PooledSparseResidualFeatureMemory:
    """Low-rank state plus diffuse and sparse per-frame innovations."""

    latents: torch.Tensor
    pooled_residual_values: torch.Tensor
    residual_indices: torch.Tensor
    residual_values: torch.Tensor
    tokens_per_frame: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.latents.ndim != 3:
            raise ValueError("latents must have shape [frames, tokens, rank]")
        frames, tokens, _ = self.latents.shape
        if tokens != self.tokens_per_frame:
            raise ValueError("tokens_per_frame does not match latent shape")
        if self.pooled_residual_values.shape != (frames, self.hidden_size):
            raise ValueError(
                "pooled residuals must have shape [frames, hidden]"
            )
        if self.residual_indices.ndim != 2:
            raise ValueError(
                "residual indices must have shape [frames, sparse_tokens]"
            )
        if self.residual_values.ndim != 3:
            raise ValueError(
                "residual values must have shape "
                "[frames, sparse_tokens, hidden]"
            )
        if self.residual_indices.shape != self.residual_values.shape[:2]:
            raise ValueError("residual index and value shapes differ")
        if self.residual_indices.shape[0] != frames:
            raise ValueError("residual frame count differs from latents")
        if self.residual_values.shape[-1] != self.hidden_size:
            raise ValueError("residual hidden size differs from metadata")
        if self.residual_indices.dtype not in {
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("residual indices must use an integer dtype")

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def rank(self) -> int:
        return int(self.latents.shape[-1])

    @property
    def sparse_residual_tokens_per_frame(self) -> int:
        return int(self.residual_indices.shape[1])

    @property
    def residual_vectors_per_frame(self) -> int:
        return 1 + self.sparse_residual_tokens_per_frame

    @property
    def latent_bytes(self) -> int:
        return tensor_bytes(self.latents)

    @property
    def residual_value_bytes(self) -> int:
        return tensor_bytes(self.pooled_residual_values) + tensor_bytes(
            self.residual_values
        )

    @property
    def residual_index_bytes(self) -> int:
        return tensor_bytes(self.residual_indices)

    @property
    def stream_state_bytes(self) -> int:
        return (
            self.latent_bytes
            + self.residual_value_bytes
            + self.residual_index_bytes
        )


@dataclass(frozen=True)
class SpatialGridResidualFeatureMemory:
    """Low-rank state plus a coarse spatial residual field per frame."""

    latents: torch.Tensor
    grid_residual_values: torch.Tensor
    token_grid_size: int
    residual_grid_size: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.latents.ndim != 3:
            raise ValueError("latents must have shape [frames, tokens, rank]")
        frames, tokens, _ = self.latents.shape
        if self.token_grid_size**2 != tokens:
            raise ValueError("token grid size does not match latent shape")
        if self.token_grid_size % self.residual_grid_size:
            raise ValueError("residual grid must divide the token grid")
        if self.grid_residual_values.shape != (
            frames,
            self.residual_grid_size,
            self.residual_grid_size,
            self.hidden_size,
        ):
            raise ValueError("grid residual shape does not match metadata")

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def rank(self) -> int:
        return int(self.latents.shape[-1])

    @property
    def tokens_per_frame(self) -> int:
        return self.token_grid_size**2

    @property
    def residual_vectors_per_frame(self) -> int:
        return self.residual_grid_size**2

    @property
    def latent_bytes(self) -> int:
        return tensor_bytes(self.latents)

    @property
    def residual_value_bytes(self) -> int:
        return tensor_bytes(self.grid_residual_values)

    @property
    def residual_index_bytes(self) -> int:
        return 0

    @property
    def stream_state_bytes(self) -> int:
        return self.latent_bytes + self.residual_value_bytes


@dataclass(frozen=True)
class RoutedSpatialSparseFeatureMemory:
    """Per-frame routing between a coarse grid and sparse innovations."""

    latents: torch.Tensor
    grid_mode: torch.Tensor
    residual_indices: torch.Tensor
    residual_values: torch.Tensor
    token_grid_size: int
    residual_grid_size: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.latents.ndim != 3:
            raise ValueError("latents must have shape [frames, tokens, rank]")
        frames, tokens, _ = self.latents.shape
        vectors = self.residual_grid_size**2
        if self.token_grid_size**2 != tokens:
            raise ValueError("token grid size does not match latent shape")
        if self.token_grid_size % self.residual_grid_size:
            raise ValueError("residual grid must divide the token grid")
        if self.grid_mode.shape != (frames,) or self.grid_mode.dtype != torch.bool:
            raise ValueError("grid mode must be a boolean vector over frames")
        if self.residual_indices.shape != (frames, vectors):
            raise ValueError("routed residual index shape is invalid")
        if self.residual_values.shape != (frames, vectors, self.hidden_size):
            raise ValueError("routed residual value shape is invalid")
        if self.residual_indices.dtype not in {
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("routed residual indices must be integer")

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def rank(self) -> int:
        return int(self.latents.shape[-1])

    @property
    def tokens_per_frame(self) -> int:
        return self.token_grid_size**2

    @property
    def residual_vectors_per_frame(self) -> int:
        return self.residual_grid_size**2

    @property
    def latent_bytes(self) -> int:
        return tensor_bytes(self.latents)

    @property
    def residual_value_bytes(self) -> int:
        return tensor_bytes(self.residual_values)

    @property
    def residual_index_slot_bytes(self) -> int:
        return tensor_bytes(self.residual_indices)

    @property
    def route_mask_bytes(self) -> int:
        return tensor_bytes(self.grid_mode)

    @property
    def residual_index_bytes(self) -> int:
        return self.residual_index_slot_bytes + self.route_mask_bytes

    @property
    def stream_state_bytes(self) -> int:
        return (
            self.latent_bytes
            + self.residual_value_bytes
            + self.residual_index_bytes
        )


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def dense_feature_bytes(features: torch.Tensor) -> int:
    _validate_features(features)
    return tensor_bytes(features)


def sample_feature_tokens(
    features: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_features(features)
    if count <= 0:
        raise ValueError("sample count must be positive")
    flattened = features.reshape(-1, features.shape[-1])
    retained = min(count, len(flattened))
    generator = torch.Generator(device=flattened.device)
    generator.manual_seed(seed)
    positions = torch.randperm(
        len(flattened),
        generator=generator,
        device=flattened.device,
    )[:retained]
    positions = torch.sort(positions).values
    return flattened.index_select(0, positions), positions


def _validate_features(
    features: torch.Tensor,
    *,
    hidden_size: int | None = None,
) -> None:
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    if hidden_size is not None and features.shape[-1] != hidden_size:
        raise ValueError("feature hidden size does not match codec")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")


def fit_pca_codec(
    features: torch.Tensor,
    *,
    rank: int,
    max_tokens: int | None = None,
    seed: int = 20260718,
    niter: int = 4,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[LowRankFeatureCodec, dict[str, float | int]]:
    _validate_features(features)
    flattened = features.reshape(-1, features.shape[-1]).float()
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if len(flattened) > max_tokens:
            generator = torch.Generator(device=flattened.device)
            generator.manual_seed(seed)
            positions = torch.randperm(
                len(flattened),
                generator=generator,
                device=flattened.device,
            )[:max_tokens]
            flattened = flattened.index_select(0, positions)
    if not 0 < rank <= min(flattened.shape):
        raise ValueError("rank exceeds the sampled feature matrix")
    mean = flattened.mean(dim=0)
    centered = flattened - mean
    _, singular_values, basis = torch.pca_lowrank(
        centered,
        q=rank,
        center=False,
        niter=niter,
    )
    total_energy = torch.sum(centered.square())
    retained_energy = torch.sum(singular_values.square())
    energy_ratio = (
        float((retained_energy / total_energy).item())
        if float(total_energy.item()) > 0.0
        else 1.0
    )
    codec = LowRankFeatureCodec(
        mean=mean.to(storage_dtype),
        basis=basis.to(storage_dtype),
    )
    return codec, {
        "training_tokens": int(len(flattened)),
        "hidden_size": int(flattened.shape[1]),
        "rank": rank,
        "explained_energy_ratio": energy_ratio,
        "model_parameter_bytes": codec.parameter_bytes,
        "seed": seed,
        "niter": niter,
    }


def encode_feature_memory(
    features: torch.Tensor,
    codec: LowRankFeatureCodec,
    *,
    residual_tokens_per_frame: int,
    storage_dtype: torch.dtype = torch.float16,
    index_dtype: torch.dtype = torch.int16,
) -> EncodedFeatureMemory:
    _validate_features(features, hidden_size=codec.hidden_size)
    frames, tokens_per_frame, hidden_size = features.shape
    if residual_tokens_per_frame < 0:
        raise ValueError("residual token count must be non-negative")
    if residual_tokens_per_frame > tokens_per_frame:
        raise ValueError("residual token count exceeds tokens per frame")
    if index_dtype == torch.int16 and tokens_per_frame > 32767:
        raise ValueError("int16 indices cannot address all frame tokens")

    latents = codec.encode(features).to(storage_dtype)
    reconstruction = codec.decode(latents)
    residual = features.float() - reconstruction
    if residual_tokens_per_frame:
        residual_energy = torch.sum(residual.square(), dim=-1)
        indices = torch.topk(
            residual_energy,
            k=residual_tokens_per_frame,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        gather_indices = indices.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_size,
        )
        residual_values = torch.gather(
            residual,
            dim=1,
            index=gather_indices,
        ).to(storage_dtype)
        residual_indices = indices.to(index_dtype)
    else:
        residual_indices = torch.empty(
            (frames, 0),
            device=features.device,
            dtype=index_dtype,
        )
        residual_values = torch.empty(
            (frames, 0, hidden_size),
            device=features.device,
            dtype=storage_dtype,
        )
    return EncodedFeatureMemory(
        latents=latents,
        residual_indices=residual_indices,
        residual_values=residual_values,
        tokens_per_frame=tokens_per_frame,
        hidden_size=hidden_size,
    )


def _budgeted_residual_scores(
    features: torch.Tensor,
    residual: torch.Tensor,
    *,
    allocation: str,
    temporal_novelty_weight: float,
) -> torch.Tensor:
    residual_energy = torch.sum(residual.square(), dim=-1)
    if allocation == "global_energy":
        return residual_energy
    if allocation != "temporal_novelty":
        raise ValueError(f"unsupported residual allocation: {allocation}")
    if temporal_novelty_weight < 0.0:
        raise ValueError("temporal novelty weight must be non-negative")

    temporal_energy = torch.zeros_like(residual_energy)
    if features.shape[0] > 1:
        temporal_energy[1:] = torch.sum(
            (features[1:].float() - features[:-1].float()).square(),
            dim=-1,
        )
    # Freeze each frame's scale when it arrives so the score is reproducible
    # by an online writer. Future frames never rescale earlier candidates.
    scale = torch.ones(
        features.shape[0],
        device=features.device,
        dtype=torch.float32,
    )
    if features.shape[0] > 1:
        transition_sums = temporal_energy[1:].sum(dim=1).cumsum(dim=0)
        transition_counts = (
            torch.arange(
                1,
                features.shape[0],
                device=features.device,
                dtype=torch.float32,
            )
            * features.shape[1]
        )
        scale[1:] = transition_sums / transition_counts
    normalized_novelty = temporal_energy / scale.clamp_min(
        torch.finfo(torch.float32).eps
    ).unsqueeze(1)
    return residual_energy * (
        1.0 + temporal_novelty_weight * normalized_novelty
    )


def _select_budgeted_indices(
    scores: torch.Tensor,
    *,
    budget: int,
    minimum_per_frame: int,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("residual scores must have shape [frames, tokens]")
    frames, tokens = scores.shape
    if budget <= 0 or budget > frames * tokens:
        raise ValueError("residual token budget is outside the valid range")
    if minimum_per_frame < 0 or minimum_per_frame > tokens:
        raise ValueError("minimum_per_frame is outside the valid range")
    mandatory = frames * minimum_per_frame
    if mandatory > budget:
        raise ValueError("minimum per-frame allocation exceeds the budget")

    flattened_scores = scores.reshape(-1)
    selected = torch.zeros_like(flattened_scores, dtype=torch.bool)
    if minimum_per_frame:
        local = torch.topk(
            scores,
            k=minimum_per_frame,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        offsets = (
            torch.arange(frames, device=scores.device).unsqueeze(1) * tokens
        )
        selected[(local + offsets).reshape(-1)] = True

    remaining = budget - int(selected.sum().item())
    if remaining:
        candidates = flattened_scores.masked_fill(selected, float("-inf"))
        global_indices = torch.topk(
            candidates,
            k=remaining,
            largest=True,
            sorted=False,
        ).indices
        selected[global_indices] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def encode_budgeted_feature_memory(
    features: torch.Tensor,
    codec: LowRankFeatureCodec,
    *,
    residual_token_budget: int,
    allocation: str = "global_energy",
    minimum_per_frame: int = 0,
    temporal_novelty_weight: float = 1.0,
    storage_dtype: torch.dtype = torch.float16,
    index_dtype: torch.dtype = torch.int16,
) -> BudgetedResidualFeatureMemory:
    """Encode an equal-budget residual archive without fixed frame quotas."""

    _validate_features(features, hidden_size=codec.hidden_size)
    frames, tokens_per_frame, hidden_size = features.shape
    addressable_tokens = frames * tokens_per_frame
    if index_dtype == torch.int16 and addressable_tokens > 32767:
        raise ValueError("int16 indices cannot address the flattened memory")

    latents = codec.encode(features).to(storage_dtype)
    reconstruction = codec.decode(latents)
    residual = features.float() - reconstruction
    scores = _budgeted_residual_scores(
        features,
        residual,
        allocation=allocation,
        temporal_novelty_weight=temporal_novelty_weight,
    )
    flat_indices = _select_budgeted_indices(
        scores,
        budget=residual_token_budget,
        minimum_per_frame=minimum_per_frame,
    )
    residual_values = residual.reshape(-1, hidden_size).index_select(
        0,
        flat_indices,
    )
    return BudgetedResidualFeatureMemory(
        latents=latents,
        residual_flat_indices=flat_indices.to(index_dtype),
        residual_values=residual_values.to(storage_dtype),
        tokens_per_frame=tokens_per_frame,
        hidden_size=hidden_size,
        allocation=allocation,
    )


def encode_pooled_sparse_feature_memory(
    features: torch.Tensor,
    codec: LowRankFeatureCodec,
    *,
    residual_vectors_per_frame: int,
    storage_dtype: torch.dtype = torch.float16,
    index_dtype: torch.dtype = torch.int16,
) -> PooledSparseResidualFeatureMemory:
    """Encode one broadcast residual plus sparse local details per frame."""

    _validate_features(features, hidden_size=codec.hidden_size)
    frames, tokens_per_frame, hidden_size = features.shape
    if not 1 <= residual_vectors_per_frame <= tokens_per_frame + 1:
        raise ValueError("residual vector count is outside the valid range")
    if index_dtype == torch.int16 and tokens_per_frame > 32767:
        raise ValueError("int16 indices cannot address all frame tokens")

    sparse_tokens = residual_vectors_per_frame - 1
    latents = codec.encode(features).to(storage_dtype)
    reconstruction = codec.decode(latents)
    residual = features.float() - reconstruction
    pooled_residual = residual.mean(dim=1)
    detail_residual = residual - pooled_residual.unsqueeze(1)
    if sparse_tokens:
        detail_energy = torch.sum(detail_residual.square(), dim=-1)
        indices = torch.topk(
            detail_energy,
            k=sparse_tokens,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        gather_indices = indices.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_size,
        )
        residual_values = torch.gather(
            detail_residual,
            dim=1,
            index=gather_indices,
        ).to(storage_dtype)
        residual_indices = indices.to(index_dtype)
    else:
        residual_indices = torch.empty(
            (frames, 0),
            device=features.device,
            dtype=index_dtype,
        )
        residual_values = torch.empty(
            (frames, 0, hidden_size),
            device=features.device,
            dtype=storage_dtype,
        )
    return PooledSparseResidualFeatureMemory(
        latents=latents,
        pooled_residual_values=pooled_residual.to(storage_dtype),
        residual_indices=residual_indices,
        residual_values=residual_values,
        tokens_per_frame=tokens_per_frame,
        hidden_size=hidden_size,
    )


def encode_spatial_grid_feature_memory(
    features: torch.Tensor,
    codec: LowRankFeatureCodec,
    *,
    residual_grid_size: int,
    storage_dtype: torch.dtype = torch.float16,
) -> SpatialGridResidualFeatureMemory:
    """Encode a block-constant low-frequency spatial residual field."""

    _validate_features(features, hidden_size=codec.hidden_size)
    frames, tokens_per_frame, hidden_size = features.shape
    token_grid_size = isqrt(tokens_per_frame)
    if token_grid_size**2 != tokens_per_frame:
        raise ValueError("spatial residuals require a square token grid")
    if (
        residual_grid_size <= 0
        or residual_grid_size > token_grid_size
        or token_grid_size % residual_grid_size
    ):
        raise ValueError("residual grid must divide the token grid")

    latents = codec.encode(features).to(storage_dtype)
    reconstruction = codec.decode(latents)
    residual = features.float() - reconstruction
    block_size = token_grid_size // residual_grid_size
    grid_residual = residual.reshape(
        frames,
        residual_grid_size,
        block_size,
        residual_grid_size,
        block_size,
        hidden_size,
    ).mean(dim=(2, 4))
    return SpatialGridResidualFeatureMemory(
        latents=latents,
        grid_residual_values=grid_residual.to(storage_dtype),
        token_grid_size=token_grid_size,
        residual_grid_size=residual_grid_size,
        hidden_size=hidden_size,
    )


def encode_routed_spatial_sparse_feature_memory(
    features: torch.Tensor,
    codec: LowRankFeatureCodec,
    *,
    residual_grid_size: int,
    grid_error_ratio: float = 1.0,
    storage_dtype: torch.dtype = torch.float16,
) -> RoutedSpatialSparseFeatureMemory:
    """Choose the lower-error structured or sparse code for each frame."""

    _validate_features(features, hidden_size=codec.hidden_size)
    frames, tokens_per_frame, hidden_size = features.shape
    token_grid_size = isqrt(tokens_per_frame)
    if token_grid_size**2 != tokens_per_frame:
        raise ValueError("routed residuals require a square token grid")
    if (
        residual_grid_size <= 0
        or residual_grid_size > token_grid_size
        or token_grid_size % residual_grid_size
    ):
        raise ValueError("residual grid must divide the token grid")
    if not isfinite(grid_error_ratio) or grid_error_ratio <= 0.0:
        raise ValueError("grid error ratio must be finite and positive")
    if tokens_per_frame > 256:
        raise ValueError("uint8 routed indices cannot address all tokens")

    vectors = residual_grid_size**2
    latents = codec.encode(features).to(storage_dtype)
    reconstruction = codec.decode(latents)
    residual = features.float() - reconstruction
    block_size = token_grid_size // residual_grid_size
    grid_values = residual.reshape(
        frames,
        residual_grid_size,
        block_size,
        residual_grid_size,
        block_size,
        hidden_size,
    ).mean(dim=(2, 4))
    stored_grid_values = grid_values.to(storage_dtype)
    grid_broadcast = (
        stored_grid_values.float()
        .unsqueeze(2)
        .unsqueeze(4)
        .expand(
            -1,
            -1,
            block_size,
            -1,
            block_size,
            -1,
        )
        .reshape(frames, tokens_per_frame, hidden_size)
    )
    stored_grid_reconstruction = (reconstruction + grid_broadcast).to(
        storage_dtype
    )
    grid_error = torch.sum(
        (features.float() - stored_grid_reconstruction.float()).square(),
        dim=(1, 2),
    )

    residual_energy = torch.sum(residual.square(), dim=-1)
    sparse_indices = torch.topk(
        residual_energy,
        k=vectors,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    sparse_values = torch.gather(
        residual,
        dim=1,
        index=sparse_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    stored_sparse_values = sparse_values.to(storage_dtype)
    sparse_candidate = torch.zeros_like(residual)
    sparse_candidate.scatter_add_(
        1,
        sparse_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        stored_sparse_values.float(),
    )
    stored_sparse_reconstruction = (reconstruction + sparse_candidate).to(
        storage_dtype
    )
    sparse_error = torch.sum(
        (features.float() - stored_sparse_reconstruction.float()).square(),
        dim=(1, 2),
    )

    grid_mode = grid_error <= grid_error_ratio * sparse_error
    residual_values = torch.where(
        grid_mode[:, None, None],
        stored_grid_values.reshape(frames, vectors, hidden_size),
        stored_sparse_values,
    )
    residual_indices = torch.where(
        grid_mode[:, None],
        torch.zeros_like(sparse_indices),
        sparse_indices,
    ).to(torch.uint8)
    return RoutedSpatialSparseFeatureMemory(
        latents=latents,
        grid_mode=grid_mode,
        residual_indices=residual_indices,
        residual_values=residual_values,
        token_grid_size=token_grid_size,
        residual_grid_size=residual_grid_size,
        hidden_size=hidden_size,
    )


def reconstruct_feature_memory(
    state: (
        EncodedFeatureMemory
        | BudgetedResidualFeatureMemory
        | PooledSparseResidualFeatureMemory
        | SpatialGridResidualFeatureMemory
        | RoutedSpatialSparseFeatureMemory
    ),
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if state.hidden_size != codec.hidden_size:
        raise ValueError("state and codec hidden sizes differ")
    if state.rank != codec.rank:
        raise ValueError("state and codec ranks differ")
    if isinstance(state, BudgetedResidualFeatureMemory):
        return _reconstruct_budgeted_feature_memory(
            state,
            codec,
            frame_positions=frame_positions,
            output_dtype=output_dtype,
        )
    if isinstance(state, PooledSparseResidualFeatureMemory):
        return _reconstruct_pooled_sparse_feature_memory(
            state,
            codec,
            frame_positions=frame_positions,
            output_dtype=output_dtype,
        )
    if isinstance(state, SpatialGridResidualFeatureMemory):
        return _reconstruct_spatial_grid_feature_memory(
            state,
            codec,
            frame_positions=frame_positions,
            output_dtype=output_dtype,
        )
    if isinstance(state, RoutedSpatialSparseFeatureMemory):
        return _reconstruct_routed_spatial_sparse_feature_memory(
            state,
            codec,
            frame_positions=frame_positions,
            output_dtype=output_dtype,
        )
    if frame_positions is None:
        latents = state.latents
        residual_indices = state.residual_indices
        residual_values = state.residual_values
    else:
        positions = torch.as_tensor(
            frame_positions,
            device=state.latents.device,
            dtype=torch.long,
        )
        if positions.ndim != 1:
            raise ValueError("frame positions must be one-dimensional")
        if len(positions) and (
            int(positions.min()) < 0
            or int(positions.max()) >= state.frames
        ):
            raise ValueError("frame positions are outside encoded state")
        latents = state.latents.index_select(0, positions)
        residual_indices = state.residual_indices.index_select(0, positions)
        residual_values = state.residual_values.index_select(0, positions)

    reconstruction = codec.decode(latents)
    if residual_indices.shape[1]:
        scatter_indices = residual_indices.long().unsqueeze(-1).expand(
            -1,
            -1,
            state.hidden_size,
        )
        reconstruction.scatter_add_(
            1,
            scatter_indices,
            residual_values.float(),
        )
    return reconstruction.to(output_dtype)


def _reconstruct_budgeted_feature_memory(
    state: BudgetedResidualFeatureMemory,
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if state.hidden_size != codec.hidden_size:
        raise ValueError("state and codec hidden sizes differ")
    if state.rank != codec.rank:
        raise ValueError("state and codec ranks differ")

    if frame_positions is None:
        positions = torch.arange(
            state.frames,
            device=state.latents.device,
            dtype=torch.long,
        )
    else:
        positions = torch.as_tensor(
            frame_positions,
            device=state.latents.device,
            dtype=torch.long,
        )
        if positions.ndim != 1:
            raise ValueError("frame positions must be one-dimensional")
        if len(positions) and (
            int(positions.min()) < 0
            or int(positions.max()) >= state.frames
        ):
            raise ValueError("frame positions are outside encoded state")
        if len(torch.unique(positions)) != len(positions):
            raise ValueError("frame positions must be unique")

    reconstruction = codec.decode(state.latents.index_select(0, positions))
    if not state.residual_token_budget:
        return reconstruction.to(output_dtype)

    original_frames = (
        state.residual_flat_indices.long() // state.tokens_per_frame
    )
    token_indices = (
        state.residual_flat_indices.long() % state.tokens_per_frame
    )
    output_position = torch.full(
        (state.frames,),
        -1,
        device=state.latents.device,
        dtype=torch.long,
    )
    output_position[positions] = torch.arange(
        len(positions),
        device=state.latents.device,
    )
    remapped_frames = output_position.index_select(0, original_frames)
    retained = remapped_frames >= 0
    if bool(retained.any()):
        output_flat_indices = (
            remapped_frames[retained] * state.tokens_per_frame
            + token_indices[retained]
        )
        reconstruction.reshape(-1, state.hidden_size).index_add_(
            0,
            output_flat_indices,
            state.residual_values[retained].float(),
        )
    return reconstruction.to(output_dtype)


def _reconstruct_pooled_sparse_feature_memory(
    state: PooledSparseResidualFeatureMemory,
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if frame_positions is None:
        positions = torch.arange(
            state.frames,
            device=state.latents.device,
            dtype=torch.long,
        )
    else:
        positions = torch.as_tensor(
            frame_positions,
            device=state.latents.device,
            dtype=torch.long,
        )
        if positions.ndim != 1:
            raise ValueError("frame positions must be one-dimensional")
        if len(positions) and (
            int(positions.min()) < 0
            or int(positions.max()) >= state.frames
        ):
            raise ValueError("frame positions are outside encoded state")

    latents = state.latents.index_select(0, positions)
    pooled = state.pooled_residual_values.index_select(0, positions)
    residual_indices = state.residual_indices.index_select(0, positions)
    residual_values = state.residual_values.index_select(0, positions)
    reconstruction = codec.decode(latents) + pooled.float().unsqueeze(1)
    if residual_indices.shape[1]:
        scatter_indices = residual_indices.long().unsqueeze(-1).expand(
            -1,
            -1,
            state.hidden_size,
        )
        reconstruction.scatter_add_(
            1,
            scatter_indices,
            residual_values.float(),
        )
    return reconstruction.to(output_dtype)


def _reconstruct_spatial_grid_feature_memory(
    state: SpatialGridResidualFeatureMemory,
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if frame_positions is None:
        positions = torch.arange(
            state.frames,
            device=state.latents.device,
            dtype=torch.long,
        )
    else:
        positions = torch.as_tensor(
            frame_positions,
            device=state.latents.device,
            dtype=torch.long,
        )
        if positions.ndim != 1:
            raise ValueError("frame positions must be one-dimensional")
        if len(positions) and (
            int(positions.min()) < 0
            or int(positions.max()) >= state.frames
        ):
            raise ValueError("frame positions are outside encoded state")

    latents = state.latents.index_select(0, positions)
    grid = state.grid_residual_values.index_select(0, positions).float()
    block_size = state.token_grid_size // state.residual_grid_size
    broadcast_residual = (
        grid.unsqueeze(2)
        .unsqueeze(4)
        .expand(
            -1,
            -1,
            block_size,
            -1,
            block_size,
            -1,
        )
        .reshape(len(positions), state.tokens_per_frame, state.hidden_size)
    )
    reconstruction = codec.decode(latents) + broadcast_residual
    return reconstruction.to(output_dtype)


def _reconstruct_routed_spatial_sparse_feature_memory(
    state: RoutedSpatialSparseFeatureMemory,
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if frame_positions is None:
        positions = torch.arange(
            state.frames,
            device=state.latents.device,
            dtype=torch.long,
        )
    else:
        positions = torch.as_tensor(
            frame_positions,
            device=state.latents.device,
            dtype=torch.long,
        )
        if positions.ndim != 1:
            raise ValueError("frame positions must be one-dimensional")
        if len(positions) and (
            int(positions.min()) < 0
            or int(positions.max()) >= state.frames
        ):
            raise ValueError("frame positions are outside encoded state")

    latents = state.latents.index_select(0, positions)
    modes = state.grid_mode.index_select(0, positions)
    indices = state.residual_indices.index_select(0, positions).long()
    values = state.residual_values.index_select(0, positions).float()
    reconstruction = codec.decode(latents)

    block_size = state.token_grid_size // state.residual_grid_size
    grid_broadcast = (
        values.reshape(
            len(positions),
            state.residual_grid_size,
            state.residual_grid_size,
            state.hidden_size,
        )
        .unsqueeze(2)
        .unsqueeze(4)
        .expand(
            -1,
            -1,
            block_size,
            -1,
            block_size,
            -1,
        )
        .reshape(len(positions), state.tokens_per_frame, state.hidden_size)
    )
    reconstruction += grid_broadcast * modes[:, None, None]
    sparse_values = values * (~modes)[:, None, None]
    reconstruction.scatter_add_(
        1,
        indices.unsqueeze(-1).expand(-1, -1, state.hidden_size),
        sparse_values,
    )
    return reconstruction.to(output_dtype)


def relative_reconstruction_error(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    if reference.shape != reconstruction.shape:
        raise ValueError("reference and reconstruction shapes differ")
    denominator = torch.linalg.vector_norm(reference.float())
    if float(denominator.item()) == 0.0:
        return 0.0
    numerator = torch.linalg.vector_norm(
        reference.float() - reconstruction.float()
    )
    return float((numerator / denominator).item())


def save_codec(
    codec: LowRankFeatureCodec,
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "mean": codec.mean.detach().cpu(),
            "basis": codec.basis.detach().cpu(),
            "metadata": metadata or {},
        },
        path,
    )


def save_routed_feature_memory(
    state: RoutedSpatialSparseFeatureMemory,
    path: Path,
) -> None:
    """Persist a versioned routed state; archive bytes include container metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": ROUTED_STATE_FORMAT_VERSION,
            "state_type": "routed_spatial_sparse",
            "token_layout": ROUTED_TOKEN_LAYOUT,
            "latents": state.latents.detach().cpu(),
            "grid_mode": state.grid_mode.detach().cpu(),
            "residual_indices": state.residual_indices.detach().cpu(),
            "residual_values": state.residual_values.detach().cpu(),
            "token_grid_size": state.token_grid_size,
            "residual_grid_size": state.residual_grid_size,
            "hidden_size": state.hidden_size,
            "tensor_payload_bytes": state.stream_state_bytes,
        },
        path,
    )


def load_routed_feature_memory(
    path: Path,
    *,
    device: torch.device | str | None = None,
) -> RoutedSpatialSparseFeatureMemory:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", 0)) != ROUTED_STATE_FORMAT_VERSION:
        raise ValueError("unsupported routed feature-memory format")
    if payload.get("state_type") != "routed_spatial_sparse":
        raise ValueError("routed feature-memory state type is invalid")
    if payload.get("token_layout") != ROUTED_TOKEN_LAYOUT:
        raise ValueError("routed feature-memory token layout is invalid")
    state = RoutedSpatialSparseFeatureMemory(
        latents=payload["latents"].to(device=device),
        grid_mode=payload["grid_mode"].to(device=device),
        residual_indices=payload["residual_indices"].to(device=device),
        residual_values=payload["residual_values"].to(device=device),
        token_grid_size=int(payload["token_grid_size"]),
        residual_grid_size=int(payload["residual_grid_size"]),
        hidden_size=int(payload["hidden_size"]),
    )
    if int(payload.get("tensor_payload_bytes", -1)) != state.stream_state_bytes:
        raise ValueError("routed feature-memory payload byte count is inconsistent")
    return state


def load_codec(
    path: Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[LowRankFeatureCodec, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported feature codec format")
    codec = LowRankFeatureCodec(
        mean=payload["mean"],
        basis=payload["basis"],
    ).to(device=device, dtype=dtype)
    metadata = dict(payload.get("metadata", {}))
    return codec, metadata
