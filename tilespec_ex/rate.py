"""Exact stream and shared-overhead accounting for TileLogic-RVQ."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .routing import (
    FeatureNormalizer,
    LogicRouter,
    RouterMLP,
    logic_router_storage_bits,
)
from .vq import ResidualVQCodebook, ScaledCodebook, codebook_storage_bits


RATE_STORAGE_POLICY = {
    "base_scalar_scale_bits": 32,
    "vq_codeword_bits": 16,
    "vq_scale_table_bits": 16,
    "vq_metric_weight_bits": 32,
    "mlp_parameter_bits": 32,
    "mlp_normalizer_bits": 32,
    "logic_threshold_bits": 16,
    "logic_leaf_bits": 32,
    "curvature_prior_bits": 32,
    "exact_fallback_value_bits": 16,
}


@dataclass(frozen=True)
class BitComponent:
    name: str
    bits: int
    scope: str

    def __post_init__(self) -> None:
        if self.bits < 0:
            raise ValueError("bit component cannot be negative")
        if self.scope not in {"stream", "shared"}:
            raise ValueError("scope must be stream or shared")


@dataclass
class RateLedger:
    components: list[BitComponent] = field(default_factory=list)

    def add_stream(self, name: str, bits: int) -> None:
        self.components.append(BitComponent(name, int(bits), "stream"))

    def add_shared(self, name: str, bits: int) -> None:
        self.components.append(BitComponent(name, int(bits), "shared"))

    @property
    def stream_bits(self) -> int:
        return sum(item.bits for item in self.components if item.scope == "stream")

    @property
    def shared_bits(self) -> int:
        return sum(item.bits for item in self.components if item.scope == "shared")

    def effective_bits(self, amortization_count: int) -> float:
        if amortization_count <= 0:
            raise ValueError("amortization_count must be positive")
        return self.stream_bits + self.shared_bits / amortization_count

    def metrics(
        self,
        *,
        original_vectors: int,
        vector_dimension: int,
        amortization_count: int,
        raw_value_bits: int = 16,
    ) -> dict[str, float | int | None]:
        if min(original_vectors, vector_dimension, raw_value_bits) <= 0:
            raise ValueError("rate denominators must be positive")
        original_values = original_vectors * vector_dimension
        effective = self.effective_bits(amortization_count)
        raw_bits = original_values * raw_value_bits
        break_even: int | None = None
        if self.stream_bits < raw_bits:
            break_even = (
                0
                if self.shared_bits == 0
                else math.ceil(self.shared_bits / (raw_bits - self.stream_bits))
            )
        return {
            "stream_bits": self.stream_bits,
            "shared_bits": self.shared_bits,
            "effective_bits": effective,
            "stream_bits_per_original_vector": self.stream_bits / original_vectors,
            "stream_bits_per_original_value": self.stream_bits / original_values,
            "effective_bits_per_original_vector": effective / original_vectors,
            "effective_bits_per_original_value": effective / original_values,
            "raw_fp_bits": raw_bits,
            "stream_compression_ratio": raw_bits / max(1, self.stream_bits),
            "effective_compression_ratio": raw_bits / max(1.0, effective),
            "break_even_samples": break_even,
            "amortization_count": amortization_count,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "stream_bits": self.stream_bits,
            "shared_bits": self.shared_bits,
            "components": [item.__dict__ for item in self.components],
        }


def base_scaled_vq_bits(
    ledger: RateLedger,
    codebook: ScaledCodebook,
    vector_count: int,
    *,
    include_shared: bool = True,
    codeword_bits: int | None = None,
    scale_table_bits: int | None = None,
    metric_weight_bits: int | None = None,
) -> None:
    if vector_count < 0:
        raise ValueError("vector_count cannot be negative")
    ledger.add_stream("base_code_indices", vector_count * codebook.index_bits)
    ledger.add_stream("base_scale_indices", vector_count * codebook.scale_bits)
    if include_shared:
        ledger.add_shared(
            "base_codebook",
            codebook_storage_bits(
                codebook.codewords.shape,
                int(codebook.scale_levels.numel()),
                codeword_bits=(
                    codebook.codeword_storage_bits
                    if codeword_bits is None
                    else codeword_bits
                ),
                scale_bits=(
                    codebook.scale_storage_bits
                    if scale_table_bits is None
                    else scale_table_bits
                ),
                metric_weights=codebook.dimension,
                metric_weight_bits=(
                    codebook.metric_weight_storage_bits
                    if metric_weight_bits is None
                    else metric_weight_bits
                ),
            ),
        )


def base_scalar_bits(
    ledger: RateLedger,
    vector_count: int,
    vector_dimension: int,
    *,
    value_bits: int = 4,
    per_vector_scale_bits: int = RATE_STORAGE_POLICY["base_scalar_scale_bits"],
) -> None:
    if min(vector_count, vector_dimension, value_bits, per_vector_scale_bits) <= 0:
        raise ValueError("scalar quantization rate arguments must be positive")
    ledger.add_stream(
        "base_scalar_codes", vector_count * vector_dimension * value_bits
    )
    ledger.add_stream("base_scalar_scales", vector_count * per_vector_scale_bits)


def residual_incremental_costs(
    codebook: ResidualVQCodebook,
    block_count: int,
    *,
    dynamic_positions: bool,
    total_positions: int,
    exact_value_bits: int = 16,
    mode_bits: int = 2,
) -> torch.Tensor:
    """Return per-block costs for RVQ1, RVQ2, ..., exact upgrades."""

    if block_count <= 0 or total_positions <= 0:
        raise ValueError("block and position counts must be positive")
    position_bits = max(1, math.ceil(math.log2(total_positions)))
    first = codebook.scale_bits + codebook.index_bits
    if dynamic_positions:
        first += position_bits + mode_bits
    stage_costs = [first] + [codebook.index_bits] * (codebook.stages - 1)
    exact_payload = codebook.dimension * exact_value_bits
    replaced_payload = codebook.scale_bits + codebook.stages * codebook.index_bits
    exact_upgrade = exact_payload - replaced_payload
    if exact_upgrade <= 0:
        raise ValueError("exact fallback must cost more than RVQ payload")
    stage_costs.append(exact_upgrade)
    return torch.tensor(stage_costs, dtype=torch.long).repeat(block_count, 1)


def residual_stream_bits(
    ledger: RateLedger,
    codebook: ResidualVQCodebook,
    modes: torch.Tensor,
    *,
    dynamic_positions: bool,
    total_positions: int,
    fixed_slot_count: int | None = None,
    exact_value_bits: int = 16,
    mode_bits: int = 2,
    include_shared: bool = True,
    codeword_bits: int | None = None,
    scale_table_bits: int | None = None,
    metric_weight_bits: int | None = None,
) -> None:
    """Account modes 0=drop, 1..S=RVQ depth, S+1=exact FP payload."""

    if modes.ndim != 1:
        raise ValueError("modes must have shape [blocks]")
    if bool(((modes < 0) | (modes > codebook.stages + 1)).any()):
        raise ValueError("residual mode is out of range")
    position_bits = max(1, math.ceil(math.log2(total_positions)))
    active = modes > 0
    exact = modes == codebook.stages + 1
    rvq = active & ~exact
    if dynamic_positions:
        count_bits = max(1, math.ceil(math.log2(total_positions + 1)))
        ledger.add_stream("residual_count_header", count_bits)
        ledger.add_stream("residual_positions", int(active.sum()) * position_bits)
        ledger.add_stream("residual_modes", int(active.sum()) * mode_bits)
    else:
        if fixed_slot_count is None or fixed_slot_count <= 0:
            raise ValueError("fixed-slot accounting needs fixed_slot_count")
        if modes.numel() != fixed_slot_count:
            raise ValueError("fixed-slot modes must enumerate exactly the slots")
        ledger.add_stream("fixed_slot_valid_mask", fixed_slot_count)
        ledger.add_stream("fixed_slot_modes", fixed_slot_count * mode_bits)
    ledger.add_stream("residual_scale_indices", int(rvq.sum()) * codebook.scale_bits)
    rvq_indices = sum(
        int(((modes >= depth) & rvq).sum())
        for depth in range(1, codebook.stages + 1)
    )
    ledger.add_stream("residual_code_indices", rvq_indices * codebook.index_bits)
    ledger.add_stream(
        "exact_fallback_payload", int(exact.sum()) * codebook.dimension * exact_value_bits
    )
    if include_shared:
        ledger.add_shared(
            "residual_codebooks",
            codebook_storage_bits(
                codebook.codewords.shape,
                int(codebook.scale_levels.numel()),
                codeword_bits=(
                    codebook.codeword_storage_bits
                    if codeword_bits is None
                    else codeword_bits
                ),
                scale_bits=(
                    codebook.scale_storage_bits
                    if scale_table_bits is None
                    else scale_table_bits
                ),
                metric_weights=codebook.dimension,
                metric_weight_bits=(
                    codebook.metric_weight_storage_bits
                    if metric_weight_bits is None
                    else metric_weight_bits
                ),
            ),
        )


def fixed_slot_shared_bits(
    ledger: RateLedger,
    *,
    slot_count: int,
    total_positions: int,
) -> None:
    if slot_count <= 0 or total_positions <= 0:
        raise ValueError("fixed-slot metadata arguments must be positive")
    position_bits = max(1, math.ceil(math.log2(total_positions)))
    ledger.add_shared("fixed_slot_positions", slot_count * position_bits)


def mlp_router_shared_bits(
    ledger: RateLedger,
    router: RouterMLP,
    normalizer: FeatureNormalizer,
    *,
    weight_bits: int | None = None,
    normalizer_bits: int | None = None,
) -> None:
    parameters = tuple(router.parameters())
    if weight_bits is None:
        parameter_bits = sum(
            parameter.numel() * parameter.element_size() * 8
            for parameter in parameters
        )
    else:
        if weight_bits <= 0:
            raise ValueError("router weight storage bits must be positive")
        parameter_bits = sum(parameter.numel() for parameter in parameters) * weight_bits
    normalizer_tensors = (normalizer.mean, normalizer.scale)
    if normalizer_bits is None:
        stored_normalizer_bits = sum(
            value.numel() * value.element_size() * 8 for value in normalizer_tensors
        )
    else:
        if normalizer_bits <= 0:
            raise ValueError("normalizer storage bits must be positive")
        stored_normalizer_bits = (
            sum(value.numel() for value in normalizer_tensors) * normalizer_bits
        )
    ledger.add_shared("mlp_router_parameters", parameter_bits)
    ledger.add_shared("mlp_router_normalizer", stored_normalizer_bits)


def logic_router_shared_bits_entry(
    ledger: RateLedger,
    router: LogicRouter,
    *,
    threshold_bits: int | None = None,
    leaf_value_bits: int | None = None,
) -> None:
    ledger.add_shared(
        "logic_router",
        logic_router_storage_bits(
            router,
            threshold_bits=threshold_bits,
            leaf_value_bits=leaf_value_bits,
        ),
    )


def router_curvature_shared_bits(
    ledger: RateLedger,
    curvature_prior: torch.Tensor,
    *,
    value_bits: int | None = None,
) -> None:
    if curvature_prior.ndim != 1 or not curvature_prior.is_floating_point():
        raise ValueError("curvature prior must be a floating vector")
    if value_bits is None:
        value_bits = curvature_prior.element_size() * 8
    if value_bits <= 0:
        raise ValueError("curvature-prior storage bits must be positive")
    ledger.add_shared(
        "router_curvature_prior", curvature_prior.numel() * value_bits
    )
