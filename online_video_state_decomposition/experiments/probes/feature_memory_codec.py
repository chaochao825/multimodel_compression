from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


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


def reconstruct_feature_memory(
    state: EncodedFeatureMemory,
    codec: LowRankFeatureCodec,
    *,
    frame_positions: torch.Tensor | list[int] | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if state.hidden_size != codec.hidden_size:
        raise ValueError("state and codec hidden sizes differ")
    if state.rank != codec.rank:
        raise ValueError("state and codec ranks differ")
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
