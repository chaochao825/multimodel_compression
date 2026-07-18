"""Vector-quantization primitives for the TileLogic-RVQ experiment.

The codec separates vector shape from a low-bit scale state.  Residual VQ is
trained sequentially and reserves codeword zero at every stage, so a deeper
decode always has the option to leave the previous reconstruction unchanged.
Metric weights are diagonal empirical-Fisher/Hessian proxies; they affect
encoding and codebook fitting but not decoder arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


def _floating_storage_dtype(bits: int) -> torch.dtype:
    if bits == 16:
        return torch.float16
    if bits == 32:
        return torch.float32
    raise ValueError("floating storage supports exactly 16 or 32 bits")


def _tensor_storage_bits(value: torch.Tensor) -> int:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("stored numeric payload must be a floating tensor")
    bits = value.element_size() * 8
    _floating_storage_dtype(bits)
    return bits


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N,D] tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _metric_weights(
    weights: torch.Tensor | None,
    dimension: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if weights.ndim != 1 or weights.numel() != dimension:
        raise ValueError(
            f"metric weights must have shape [{dimension}], got {tuple(weights.shape)}"
        )
    result = weights.to(device=device, dtype=torch.float32)
    if not torch.isfinite(result).all() or bool((result <= 0).any()):
        raise ValueError("metric weights must be finite and strictly positive")
    return result


def _seeded_indices(count: int, take: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator)[:take]


def nearest_code_indices(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    *,
    metric_weights: torch.Tensor | None = None,
    batch_size: int = 1024,
    return_distance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return exact nearest-code indices under a shared diagonal metric."""

    vectors = _matrix("vectors", vectors)
    codebook = _matrix("codebook", codebook)
    if vectors.shape[1] != codebook.shape[1]:
        raise ValueError("vector and codebook dimensions differ")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = vectors.device
    centers = codebook.to(device=device, dtype=torch.float32)
    weights = _metric_weights(metric_weights, vectors.shape[1], device=device)
    if weights is not None:
        root = weights.sqrt()
        centers_for_distance = centers * root
    else:
        root = None
        centers_for_distance = centers
    center_norm = centers_for_distance.square().sum(dim=1)
    indices: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, vectors.shape[0], batch_size):
        chunk = vectors[start : start + batch_size].float()
        if root is not None:
            chunk = chunk * root
        distance = (
            chunk.square().sum(dim=1, keepdim=True)
            + center_norm.unsqueeze(0)
            - 2.0 * (chunk @ centers_for_distance.t())
        ).clamp_min_(0.0)
        chunk_distance, chunk_indices = distance.min(dim=1)
        indices.append(chunk_indices)
        if return_distance:
            distances.append(chunk_distance)
    all_indices = torch.cat(indices)
    if return_distance:
        return all_indices, torch.cat(distances)
    return all_indices


def _fit_kmeans(
    vectors: torch.Tensor,
    num_codes: int,
    *,
    metric_weights: torch.Tensor | None,
    iterations: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    vectors = _matrix("vectors", vectors).float()
    if not 1 <= num_codes <= vectors.shape[0]:
        raise ValueError("num_codes must be in [1, number of vectors]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    initial = _seeded_indices(vectors.shape[0], num_codes, seed).to(vectors.device)
    centers = vectors[initial].clone()
    weights = _metric_weights(metric_weights, vectors.shape[1], device=vectors.device)
    for iteration in range(iterations):
        assignments, distances = nearest_code_indices(
            vectors,
            centers,
            metric_weights=weights,
            batch_size=batch_size,
            return_distance=True,
        )
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, vectors)
        counts = torch.bincount(assignments, minlength=num_codes).to(vectors.dtype)
        updated = sums / counts.clamp_min(1).unsqueeze(1)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            farthest = torch.topk(distances, k=int(empty.numel())).indices
            updated[empty] = vectors[farthest]
        denominator = centers.square().mean().sqrt().clamp_min(1e-12)
        relative_shift = (updated - centers).square().mean().sqrt() / denominator
        centers = updated
        if iteration > 0 and float(relative_shift.item()) < 1e-5:
            break
    return centers


def _fit_scalar_levels(
    values: torch.Tensor,
    num_levels: int,
    *,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("scale values must be a non-empty vector")
    if bool((values <= 0).any()):
        raise ValueError("scale values must be positive")
    log_values = values.float().log2().unsqueeze(1)
    centers = _fit_kmeans(
        log_values,
        num_levels,
        metric_weights=None,
        iterations=iterations,
        seed=seed,
        batch_size=max(1, min(4096, values.numel())),
    ).squeeze(1)
    return centers.sort().values.exp2()


def _scale_indices(scales: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    log_scales = scales.float().clamp_min(1e-12).log2().unsqueeze(1)
    log_levels = levels.to(scales.device, torch.float32).log2().unsqueeze(0)
    return (log_scales - log_levels).abs().argmin(dim=1)


@dataclass(frozen=True)
class ScaledCodebook:
    """One-stage shape codebook with a scalar scale table."""

    codewords: torch.Tensor
    scale_levels: torch.Tensor
    metric_weights: torch.Tensor
    codeword_storage_bits: int = 16
    scale_storage_bits: int = 16
    metric_weight_storage_bits: int = 32

    def __post_init__(self) -> None:
        _matrix("codewords", self.codewords)
        if self.scale_levels.ndim != 1 or self.scale_levels.numel() < 1:
            raise ValueError("scale_levels must be a non-empty vector")
        _metric_weights(
            self.metric_weights,
            self.codewords.shape[1],
            device=self.metric_weights.device,
        )
        for bits in (
            self.codeword_storage_bits,
            self.scale_storage_bits,
            self.metric_weight_storage_bits,
        ):
            _floating_storage_dtype(bits)

    @property
    def dimension(self) -> int:
        return int(self.codewords.shape[1])

    @property
    def num_codes(self) -> int:
        return int(self.codewords.shape[0])

    @property
    def index_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.num_codes)))

    @property
    def scale_bits(self) -> int:
        return max(1, math.ceil(math.log2(int(self.scale_levels.numel()))))

    def to(self, device: torch.device | str) -> "ScaledCodebook":
        return ScaledCodebook(
            self.codewords.to(device),
            self.scale_levels.to(device),
            self.metric_weights.to(device),
            self.codeword_storage_bits,
            self.scale_storage_bits,
            self.metric_weight_storage_bits,
        )

    def encode(
        self, vectors: torch.Tensor, *, batch_size: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = _matrix("vectors", vectors)
        if vectors.shape[1] != self.dimension:
            raise ValueError("vector dimension does not match codebook")
        rms = vectors.float().square().mean(dim=1).sqrt().clamp_min(1e-12)
        normalized = vectors.float() / rms.unsqueeze(1)
        indices = nearest_code_indices(
            normalized,
            self.codewords,
            metric_weights=self.metric_weights,
            batch_size=batch_size,
        )
        return indices, _scale_indices(rms, self.scale_levels)

    def decode(
        self, indices: torch.Tensor, scale_indices: torch.Tensor
    ) -> torch.Tensor:
        if indices.ndim != 1 or scale_indices.shape != indices.shape:
            raise ValueError("indices and scale_indices must be equal-length vectors")
        words = self.codewords.to(indices.device)[indices]
        scales = self.scale_levels.to(indices.device)[scale_indices]
        return words * scales.unsqueeze(1)

    def reconstruct(
        self, vectors: torch.Tensor, *, batch_size: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, scale_indices = self.encode(vectors, batch_size=batch_size)
        return self.decode(indices, scale_indices), indices, scale_indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "tilespec_scaled_codebook_v1",
            "codewords": self.codewords.detach()
            .cpu()
            .to(_floating_storage_dtype(self.codeword_storage_bits)),
            "scale_levels": self.scale_levels.detach()
            .cpu()
            .to(_floating_storage_dtype(self.scale_storage_bits)),
            "metric_weights": self.metric_weights.detach()
            .cpu()
            .to(_floating_storage_dtype(self.metric_weight_storage_bits)),
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "ScaledCodebook":
        if state.get("format") != "tilespec_scaled_codebook_v1":
            raise ValueError("unsupported scaled-codebook format")
        return ScaledCodebook(
            state["codewords"].float(),
            state["scale_levels"].float(),
            state["metric_weights"].float(),
            _tensor_storage_bits(state["codewords"]),
            _tensor_storage_bits(state["scale_levels"]),
            _tensor_storage_bits(state["metric_weights"]),
        )


@dataclass(frozen=True)
class ResidualVQCodebook:
    """Sequential residual codebooks with a shared scale state."""

    codewords: torch.Tensor
    scale_levels: torch.Tensor
    metric_weights: torch.Tensor
    codeword_storage_bits: int = 16
    scale_storage_bits: int = 16
    metric_weight_storage_bits: int = 32

    def __post_init__(self) -> None:
        if self.codewords.ndim != 3 or min(self.codewords.shape) <= 0:
            raise ValueError("codewords must have shape [stages,K,D]")
        if self.scale_levels.ndim != 1 or self.scale_levels.numel() < 1:
            raise ValueError("scale_levels must be a non-empty vector")
        _metric_weights(
            self.metric_weights,
            self.codewords.shape[2],
            device=self.metric_weights.device,
        )
        for bits in (
            self.codeword_storage_bits,
            self.scale_storage_bits,
            self.metric_weight_storage_bits,
        ):
            _floating_storage_dtype(bits)
        if not torch.equal(
            self.codewords[:, 0], torch.zeros_like(self.codewords[:, 0])
        ):
            raise ValueError("residual codeword zero must be reserved at every stage")

    @property
    def stages(self) -> int:
        return int(self.codewords.shape[0])

    @property
    def num_codes(self) -> int:
        return int(self.codewords.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.codewords.shape[2])

    @property
    def index_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.num_codes)))

    @property
    def scale_bits(self) -> int:
        return max(1, math.ceil(math.log2(int(self.scale_levels.numel()))))

    def to(self, device: torch.device | str) -> "ResidualVQCodebook":
        return ResidualVQCodebook(
            self.codewords.to(device),
            self.scale_levels.to(device),
            self.metric_weights.to(device),
            self.codeword_storage_bits,
            self.scale_storage_bits,
            self.metric_weight_storage_bits,
        )

    def encode(
        self, vectors: torch.Tensor, *, batch_size: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = _matrix("vectors", vectors)
        if vectors.shape[1] != self.dimension:
            raise ValueError("vector dimension does not match residual codebook")
        rms = vectors.float().square().mean(dim=1).sqrt().clamp_min(1e-12)
        residual = vectors.float() / rms.unsqueeze(1)
        stage_indices: list[torch.Tensor] = []
        for stage in range(self.stages):
            indices = nearest_code_indices(
                residual,
                self.codewords[stage],
                metric_weights=self.metric_weights,
                batch_size=batch_size,
            )
            stage_indices.append(indices)
            residual = residual - self.codewords[stage].to(residual.device)[indices]
        return torch.stack(stage_indices, dim=1), _scale_indices(rms, self.scale_levels)

    def decode(
        self,
        stage_indices: torch.Tensor,
        scale_indices: torch.Tensor,
        *,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stage_indices.ndim != 2 or stage_indices.shape[1] != self.stages:
            raise ValueError("stage_indices must have shape [N,stages]")
        if scale_indices.ndim != 1 or scale_indices.shape[0] != stage_indices.shape[0]:
            raise ValueError("scale_indices must have shape [N]")
        if depths is None:
            depths = torch.full(
                (stage_indices.shape[0],),
                self.stages,
                device=stage_indices.device,
                dtype=torch.long,
            )
        if depths.shape != scale_indices.shape:
            raise ValueError("depths must have shape [N]")
        if bool(((depths < 0) | (depths > self.stages)).any()):
            raise ValueError("depth must be in [0, stages]")
        output = torch.zeros(
            (stage_indices.shape[0], self.dimension),
            device=stage_indices.device,
            dtype=self.codewords.dtype,
        )
        words = self.codewords.to(stage_indices.device)
        for stage in range(self.stages):
            active = depths > stage
            if bool(active.any()):
                output[active] += words[stage, stage_indices[active, stage]]
        scales = self.scale_levels.to(stage_indices.device)[scale_indices]
        return output * scales.unsqueeze(1)

    def reconstructions_by_depth(
        self, vectors: torch.Tensor, *, batch_size: int = 512
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        stage_indices, scale_indices = self.encode(vectors, batch_size=batch_size)
        outputs = [torch.zeros_like(vectors, dtype=self.codewords.dtype)]
        for depth in range(1, self.stages + 1):
            depths = torch.full_like(scale_indices, depth)
            outputs.append(self.decode(stage_indices, scale_indices, depths=depths))
        return outputs, stage_indices, scale_indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "tilespec_residual_vq_v1",
            "codewords": self.codewords.detach()
            .cpu()
            .to(_floating_storage_dtype(self.codeword_storage_bits)),
            "scale_levels": self.scale_levels.detach()
            .cpu()
            .to(_floating_storage_dtype(self.scale_storage_bits)),
            "metric_weights": self.metric_weights.detach()
            .cpu()
            .to(_floating_storage_dtype(self.metric_weight_storage_bits)),
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "ResidualVQCodebook":
        if state.get("format") != "tilespec_residual_vq_v1":
            raise ValueError("unsupported residual-codebook format")
        return ResidualVQCodebook(
            state["codewords"].float(),
            state["scale_levels"].float(),
            state["metric_weights"].float(),
            _tensor_storage_bits(state["codewords"]),
            _tensor_storage_bits(state["scale_levels"]),
            _tensor_storage_bits(state["metric_weights"]),
        )


def fit_scaled_codebook(
    vectors: torch.Tensor,
    *,
    num_codes: int = 256,
    num_scale_levels: int = 16,
    metric_weights: torch.Tensor | None = None,
    iterations: int = 16,
    seed: int = 20260718,
    batch_size: int = 1024,
) -> ScaledCodebook:
    vectors = _matrix("vectors", vectors).float()
    weights = _metric_weights(metric_weights, vectors.shape[1], device=vectors.device)
    if weights is None:
        weights = torch.ones(vectors.shape[1], device=vectors.device)
    rms = vectors.square().mean(dim=1).sqrt().clamp_min(1e-12)
    normalized = vectors / rms.unsqueeze(1)
    codewords = _fit_kmeans(
        normalized,
        num_codes,
        metric_weights=weights,
        iterations=iterations,
        seed=seed,
        batch_size=batch_size,
    )
    scales = _fit_scalar_levels(
        rms,
        num_scale_levels,
        iterations=iterations,
        seed=seed + 1,
    )
    return ScaledCodebook(codewords, scales, weights)


def fit_residual_vq_codebook(
    vectors: torch.Tensor,
    *,
    stages: int = 2,
    num_codes: int = 256,
    num_scale_levels: int = 16,
    metric_weights: torch.Tensor | None = None,
    iterations: int = 16,
    seed: int = 20260718,
    batch_size: int = 512,
) -> ResidualVQCodebook:
    vectors = _matrix("vectors", vectors).float()
    if stages <= 0:
        raise ValueError("stages must be positive")
    if num_codes < 2:
        raise ValueError("residual VQ needs zero plus at least one learned codeword")
    weights = _metric_weights(metric_weights, vectors.shape[1], device=vectors.device)
    if weights is None:
        weights = torch.ones(vectors.shape[1], device=vectors.device)
    rms = vectors.square().mean(dim=1).sqrt().clamp_min(1e-12)
    residual = vectors / rms.unsqueeze(1)
    stage_codewords: list[torch.Tensor] = []
    for stage in range(stages):
        learned = _fit_kmeans(
            residual,
            num_codes - 1,
            metric_weights=weights,
            iterations=iterations,
            seed=seed + 17 * stage,
            batch_size=batch_size,
        )
        codewords = torch.cat((torch.zeros_like(learned[:1]), learned), dim=0)
        indices = nearest_code_indices(
            residual,
            codewords,
            metric_weights=weights,
            batch_size=batch_size,
        )
        residual = residual - codewords[indices]
        stage_codewords.append(codewords)
    scales = _fit_scalar_levels(
        rms,
        num_scale_levels,
        iterations=iterations,
        seed=seed + 991,
    )
    return ResidualVQCodebook(torch.stack(stage_codewords), scales, weights)


@dataclass(frozen=True)
class ScalarQuantizationResult:
    reconstructed: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    bits: int
    scale_storage_bits: int

    @property
    def stream_bits(self) -> int:
        return int(self.codes.numel() * self.bits + self.scales.numel() * self.scale_storage_bits)


def symmetric_vector_quantize(
    vectors: torch.Tensor,
    *,
    bits: int = 4,
    scale_storage_bits: int = 32,
) -> ScalarQuantizationResult:
    """Per-vector symmetric scalar quantization with an explicit scale payload."""

    vectors = _matrix("vectors", vectors)
    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2,8]")
    scale_dtype = _floating_storage_dtype(scale_storage_bits)
    maximum = 2 ** (bits - 1) - 1
    scale = vectors.float().abs().amax(dim=1).clamp_min(1e-12) / maximum
    stored_scale = scale.to(scale_dtype)
    decoded_scale = stored_scale.float()
    codes = torch.round(vectors.float() / decoded_scale.unsqueeze(1)).clamp(
        -maximum, maximum
    )
    storage_dtype = torch.int8
    integer_codes = codes.to(storage_dtype)
    reconstructed = integer_codes.float() * decoded_scale.unsqueeze(1)
    return ScalarQuantizationResult(
        reconstructed=reconstructed,
        codes=integer_codes,
        scales=stored_scale,
        bits=bits,
        scale_storage_bits=scale_storage_bits,
    )


def weighted_squared_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    metric_weights: torch.Tensor,
) -> torch.Tensor:
    reference = _matrix("reference", reference)
    estimate = _matrix("estimate", estimate)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate shapes differ")
    weights = _metric_weights(
        metric_weights, reference.shape[1], device=reference.device
    )
    assert weights is not None
    return ((reference.float() - estimate.float()).square() * weights).sum(dim=1)


def codebook_storage_bits(
    codeword_shape: Sequence[int],
    scale_count: int,
    *,
    codeword_bits: int = 16,
    scale_bits: int = 16,
    metric_weights: int = 0,
    metric_weight_bits: int = 16,
) -> int:
    if any(int(item) <= 0 for item in codeword_shape) or scale_count <= 0:
        raise ValueError("storage shapes must be positive")
    elements = math.prod(int(item) for item in codeword_shape)
    return int(
        elements * codeword_bits
        + scale_count * scale_bits
        + metric_weights * metric_weight_bits
    )
