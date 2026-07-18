"""Method construction and exact rate ledgers for TileLogic-RVQ evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from .core import (
    RETENTION_RATES,
    compress_crop_tiles,
    enumerate_blocks,
)
from .rate import (
    RateLedger,
    base_scaled_vq_bits,
    base_scalar_bits,
    fixed_slot_shared_bits,
    logic_router_shared_bits_entry,
    mlp_router_shared_bits,
    residual_incremental_costs,
    residual_stream_bits,
    router_curvature_shared_bits,
)
from .routing import (
    FeatureNormalizer,
    LogicRouter,
    RouterMLP,
    allocate_variable_depth,
    block_router_features,
)
from .tilelogic_codec import (
    encode_base_scalar,
    encode_base_vq,
    encode_residual_modes,
)
from .vq import ResidualVQCodebook, ScaledCodebook


ORIGINAL_CROP_TOKENS = 4 * 16 * 16
MAIN_TILELOGIC_METHODS = (
    "tile_lowpass",
    "tile_energy_exception",
    "tile_risk_exception",
    "base_scalar_quant",
    "base_vq",
    "base_vq_residual_rvq",
    "base_vq_mlp_router",
    "base_vq_logic_router",
    "logic_router_fixed_slots",
    "logic_router_fixed_slots_exact_fallback",
)
ABLATION_METHODS = ("base_vq_residual_rvq_unweighted",)


def _rate_key(rate: float) -> str:
    return f"rate_{rate:.3f}".replace(".", "p")


def _exception_budget(rate: float) -> tuple[int, int, int]:
    retained = round(ORIGINAL_CROP_TOKENS * rate)
    exception_tokens = round(retained * 0.25 / 4) * 4
    return retained, retained - exception_tokens, exception_tokens


@dataclass(frozen=True)
class RouterBundle:
    rate: float
    normalizer: FeatureNormalizer
    mlp: RouterMLP
    logic: LogicRouter
    curvature_prior: torch.Tensor
    fixed_slot_mask: torch.Tensor
    locations: tuple[tuple[int, int, int], ...]

    def to(self, device: torch.device | str) -> "RouterBundle":
        self.mlp.to(device)
        return RouterBundle(
            self.rate,
            FeatureNormalizer(
                self.normalizer.mean.to(device), self.normalizer.scale.to(device)
            ),
            self.mlp,
            self.logic,
            self.curvature_prior.to(device),
            self.fixed_slot_mask.to(device),
            self.locations,
        )


@dataclass(frozen=True)
class TileLogicArtifacts:
    base: ScaledCodebook
    residual_fisher: ResidualVQCodebook
    residual_unweighted: ResidualVQCodebook
    routers: dict[float, RouterBundle]

    def to(self, device: torch.device | str) -> "TileLogicArtifacts":
        return TileLogicArtifacts(
            self.base.to(device),
            self.residual_fisher.to(device),
            self.residual_unweighted.to(device),
            {rate: bundle.to(device) for rate, bundle in self.routers.items()},
        )


@dataclass(frozen=True)
class TileLogicVariant:
    method: str
    rate: float | None
    scope: str
    reconstructed: torch.Tensor
    ledger: RateLedger
    base_tokens: int
    residual_modes: torch.Tensor | None
    residual_budget_bits: int
    residual_spent_bits: int
    router_marginal_scores: torch.Tensor | None = None


def _load_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_tilelogic_artifacts(training_dir: Path) -> TileLogicArtifacts:
    training_dir = training_dir.resolve()
    base = ScaledCodebook.from_state_dict(_load_state(training_dir / "base_codebook.pt"))
    fisher = ResidualVQCodebook.from_state_dict(
        _load_state(training_dir / "residual_rvq_fisher.pt")
    )
    unweighted = ResidualVQCodebook.from_state_dict(
        _load_state(training_dir / "residual_rvq_unweighted.pt")
    )
    routers: dict[float, RouterBundle] = {}
    for rate in RETENTION_RATES:
        state = _load_state(training_dir / f"router_{_rate_key(rate)}.pt")
        if state.get("format") != "tilelogic_router_bundle_v1":
            raise ValueError("unsupported router-bundle format")
        loaded_rate = float(state["rate"])
        if abs(loaded_rate - rate) > 1e-12:
            raise ValueError("router rate differs from requested artifact")
        locations = tuple(tuple(int(value) for value in item) for item in state["locations"])
        routers[rate] = RouterBundle(
            rate=rate,
            normalizer=FeatureNormalizer.from_state_dict(state["normalizer"]),
            mlp=RouterMLP.from_export(state["mlp"]).eval(),
            logic=LogicRouter.from_state_dict(state["logic"]),
            curvature_prior=state["curvature_prior"].float(),
            fixed_slot_mask=state["fixed_slot_mask"].bool(),
            locations=locations,
        )
    if base.dimension * 4 != fisher.dimension or fisher.dimension != unweighted.dimension:
        raise ValueError("base and residual artifact dimensions are inconsistent")
    return TileLogicArtifacts(base, fisher, unweighted, routers)


def _raw_lowpass_ledger(retained: int, channels: int) -> RateLedger:
    ledger = RateLedger()
    ledger.add_stream("fp16_base_coefficients", retained * channels * 16)
    return ledger


def _raw_exception_ledger(
    base_tokens: int,
    exception_tokens: int,
    channels: int,
    *,
    total_positions: int = 256,
) -> RateLedger:
    ledger = RateLedger()
    block_count = exception_tokens // 4
    position_bits = max(1, math.ceil(math.log2(total_positions)))
    ledger.add_stream("fp16_base_coefficients", base_tokens * channels * 16)
    ledger.add_stream("residual_count_header", math.ceil(math.log2(total_positions + 1)))
    ledger.add_stream("residual_positions", block_count * position_bits)
    ledger.add_stream("fp16_exact_residual", exception_tokens * channels * 16)
    return ledger


def _base_vq_ledger(base: ScaledCodebook, count: int) -> RateLedger:
    ledger = RateLedger()
    base_scaled_vq_bits(ledger, base, count)
    return ledger


def _residual_budget(
    codebook: ResidualVQCodebook,
    energy: torch.Tensor,
    block_count: int,
) -> tuple[int, torch.Tensor]:
    selected = torch.topk(energy, k=block_count).indices
    modes = torch.zeros_like(energy, dtype=torch.long)
    modes[selected] = codebook.stages
    ledger = RateLedger()
    residual_stream_bits(
        ledger,
        codebook,
        modes,
        dynamic_positions=True,
        total_positions=energy.numel(),
        include_shared=False,
    )
    return ledger.stream_bits, modes


def _count_header_bits(total_positions: int) -> int:
    return max(1, math.ceil(math.log2(total_positions + 1)))


def _router_modes(
    scores: torch.Tensor,
    codebook: ResidualVQCodebook,
    *,
    total_budget_bits: int,
    dynamic_positions: bool,
    candidate_mask: torch.Tensor | None = None,
    fixed_header_bits: int = 0,
) -> tuple[torch.Tensor, int]:
    costs = residual_incremental_costs(
        codebook,
        block_count=scores.shape[0],
        dynamic_positions=dynamic_positions,
        total_positions=scores.shape[0],
    ).to(scores.device)
    available = total_budget_bits - fixed_header_bits
    if dynamic_positions:
        available -= _count_header_bits(scores.shape[0])
    if available < 0:
        raise ValueError("residual budget is smaller than mandatory header")
    return allocate_variable_depth(
        scores,
        costs,
        available,
        candidate_mask=candidate_mask,
    )


def _append_residual_shared(
    ledger: RateLedger,
    base: ScaledCodebook,
    residual: ResidualVQCodebook,
    base_count: int,
) -> None:
    base_scaled_vq_bits(ledger, base, base_count)
    # Add residual shared storage using an all-drop stream; stream contribution is
    # only the count header and is removed immediately below.
    temporary = RateLedger()
    residual_stream_bits(
        temporary,
        residual,
        torch.zeros(256, dtype=torch.long),
        dynamic_positions=True,
        total_positions=256,
        include_shared=True,
    )
    ledger.add_shared("residual_codebooks", temporary.shared_bits)


def build_dynamic_logic_variant(
    thumbnail: torch.Tensor,
    crops: torch.Tensor,
    query: torch.Tensor,
    artifacts: TileLogicArtifacts,
    rate: float,
) -> TileLogicVariant:
    """Build the dynamic-position logic-router variant in isolation."""

    if rate not in RETENTION_RATES:
        raise ValueError(f"unsupported retention rate: {rate}")
    device = crops.device
    artifacts = artifacts.to(device)
    _, base_tokens, exception_tokens = _exception_budget(rate)
    block_count = exception_tokens // 4
    base = encode_base_vq(crops, base_tokens, artifacts.base)
    residual = crops - base.reconstructed
    _, locations, energy = enumerate_blocks(residual)
    bundle = artifacts.routers[rate]
    if locations != bundle.locations:
        raise RuntimeError("artifact and runtime residual layouts differ")
    dynamic_budget, _ = _residual_budget(
        artifacts.residual_fisher, energy, block_count
    )
    features, feature_locations = block_router_features(
        crops,
        residual,
        query,
        thumbnail,
        curvature_prior=bundle.curvature_prior,
    )
    if feature_locations != locations:
        raise AssertionError("router and residual locations differ")
    logic_scores = bundle.logic.predict(features).to(device).clamp_min(0)
    modes, spent = _router_modes(
        logic_scores,
        artifacts.residual_fisher,
        total_budget_bits=dynamic_budget,
        dynamic_positions=True,
    )
    encoded = encode_residual_modes(
        base.reconstructed, crops, artifacts.residual_fisher, modes
    )
    ledger = RateLedger()
    _append_residual_shared(
        ledger, artifacts.base, artifacts.residual_fisher, base_tokens
    )
    logic_router_shared_bits_entry(ledger, bundle.logic)
    router_curvature_shared_bits(ledger, bundle.curvature_prior)
    residual_stream_bits(
        ledger,
        artifacts.residual_fisher,
        modes,
        dynamic_positions=True,
        total_positions=len(locations),
        include_shared=False,
    )
    return TileLogicVariant(
        "base_vq_logic_router",
        rate,
        "main",
        encoded.reconstructed,
        ledger,
        base_tokens,
        modes,
        dynamic_budget,
        spent + _count_header_bits(len(locations)),
        logic_scores,
    )


def build_fixed_logic_variant(
    thumbnail: torch.Tensor,
    crops: torch.Tensor,
    query: torch.Tensor,
    artifacts: TileLogicArtifacts,
    rate: float,
    *,
    exact_fallback: bool = False,
) -> TileLogicVariant:
    """Build one targeted fixed-slot variant without constructing other methods."""

    if rate not in RETENTION_RATES:
        raise ValueError(f"unsupported retention rate: {rate}")
    device = crops.device
    artifacts = artifacts.to(device)
    _, base_tokens, exception_tokens = _exception_budget(rate)
    block_count = exception_tokens // 4
    base = encode_base_vq(crops, base_tokens, artifacts.base)
    residual = crops - base.reconstructed
    _, locations, energy = enumerate_blocks(residual)
    bundle = artifacts.routers[rate]
    if locations != bundle.locations:
        raise RuntimeError("artifact and runtime residual layouts differ")
    dynamic_budget, _ = _residual_budget(
        artifacts.residual_fisher, energy, block_count
    )
    features, feature_locations = block_router_features(
        crops,
        residual,
        query,
        thumbnail,
        curvature_prior=bundle.curvature_prior,
    )
    if feature_locations != locations:
        raise AssertionError("router and residual locations differ")
    logic_scores = bundle.logic.predict(features).to(device).clamp_min(0)
    slot_mask = bundle.fixed_slot_mask
    slot_count = int(slot_mask.sum())
    fixed_header = slot_count * 3
    budget = dynamic_budget
    if exact_fallback:
        exact_upgrade = (
            artifacts.residual_fisher.dimension * 16
            - artifacts.residual_fisher.scale_bits
            - artifacts.residual_fisher.stages
            * artifacts.residual_fisher.index_bits
        )
        budget += 4 * exact_upgrade
    modes, spent = _router_modes(
        logic_scores,
        artifacts.residual_fisher,
        total_budget_bits=budget,
        dynamic_positions=False,
        candidate_mask=slot_mask,
        fixed_header_bits=fixed_header,
    )
    encoded = encode_residual_modes(
        base.reconstructed, crops, artifacts.residual_fisher, modes
    )
    ledger = RateLedger()
    _append_residual_shared(
        ledger, artifacts.base, artifacts.residual_fisher, base_tokens
    )
    logic_router_shared_bits_entry(ledger, bundle.logic)
    router_curvature_shared_bits(ledger, bundle.curvature_prior)
    fixed_slot_shared_bits(
        ledger, slot_count=slot_count, total_positions=len(locations)
    )
    residual_stream_bits(
        ledger,
        artifacts.residual_fisher,
        modes[slot_mask],
        dynamic_positions=False,
        total_positions=len(locations),
        fixed_slot_count=slot_count,
        include_shared=False,
    )
    return TileLogicVariant(
        (
            "logic_router_fixed_slots_exact_fallback"
            if exact_fallback
            else "logic_router_fixed_slots"
        ),
        rate,
        "main",
        encoded.reconstructed,
        ledger,
        base_tokens,
        modes,
        budget,
        spent + fixed_header,
        logic_scores,
    )


def build_tilelogic_variants(
    thumbnail: torch.Tensor,
    crops: torch.Tensor,
    query: torch.Tensor,
    artifacts: TileLogicArtifacts,
) -> list[TileLogicVariant]:
    """Construct the full method matrix for one cached sample."""

    device = crops.device
    artifacts = artifacts.to(device)
    channels = crops.shape[-1]
    variants: list[TileLogicVariant] = []
    raw = RateLedger()
    raw.add_stream("fp16_crop_tokens", crops.numel() * 16)
    variants.append(
        TileLogicVariant("none", None, "reference", crops.clone(), raw, ORIGINAL_CROP_TOKENS, None, 0, 0)
    )

    for rate in RETENTION_RATES:
        retained, base_tokens, exception_tokens = _exception_budget(rate)
        block_count = exception_tokens // 4
        for method in ("tile_lowpass", "tile_energy_exception", "tile_risk_exception"):
            result = compress_crop_tiles(
                crops,
                method,
                rate,
                query_embedding=query if method == "tile_risk_exception" else None,
            )
            ledger = (
                _raw_lowpass_ledger(retained, channels)
                if method == "tile_lowpass"
                else _raw_exception_ledger(base_tokens, exception_tokens, channels)
            )
            variants.append(
                TileLogicVariant(method, rate, "baseline", result.reconstructed, ledger, result.base_tokens, None, 0, 0)
            )

        scalar = encode_base_scalar(crops, retained, bits=4, scale_storage_bits=32)
        scalar_ledger = RateLedger()
        base_scalar_bits(
            scalar_ledger,
            retained,
            channels,
            value_bits=4,
            per_vector_scale_bits=scalar.quantization.scale_storage_bits,
        )
        variants.append(
            TileLogicVariant("base_scalar_quant", rate, "main", scalar.reconstructed, scalar_ledger, retained, None, 0, 0)
        )

        base_only = encode_base_vq(crops, retained, artifacts.base)
        variants.append(
            TileLogicVariant(
                "base_vq",
                rate,
                "main",
                base_only.reconstructed,
                _base_vq_ledger(artifacts.base, retained),
                retained,
                None,
                0,
                0,
            )
        )

        base = encode_base_vq(crops, base_tokens, artifacts.base)
        residual = crops - base.reconstructed
        blocks, locations, energy = enumerate_blocks(residual)
        if locations != artifacts.routers[rate].locations:
            raise RuntimeError("artifact and runtime residual layouts differ")
        dynamic_budget, energy_modes = _residual_budget(
            artifacts.residual_fisher, energy, block_count
        )

        for name, codebook, scope in (
            ("base_vq_residual_rvq", artifacts.residual_fisher, "main"),
            (
                "base_vq_residual_rvq_unweighted",
                artifacts.residual_unweighted,
                "ablation",
            ),
        ):
            encoded = encode_residual_modes(
                base.reconstructed, crops, codebook, energy_modes
            )
            ledger = RateLedger()
            _append_residual_shared(ledger, artifacts.base, codebook, base_tokens)
            residual_stream_bits(
                ledger,
                codebook,
                energy_modes,
                dynamic_positions=True,
                total_positions=len(locations),
                include_shared=False,
            )
            variants.append(
                TileLogicVariant(
                    name,
                    rate,
                    scope,
                    encoded.reconstructed,
                    ledger,
                    base_tokens,
                    energy_modes,
                    dynamic_budget,
                    dynamic_budget,
                )
            )

        bundle = artifacts.routers[rate]
        features, runtime_locations = block_router_features(
            crops,
            residual,
            query,
            thumbnail,
            curvature_prior=bundle.curvature_prior,
        )
        if runtime_locations != locations:
            raise AssertionError("router and residual locations differ")
        with torch.inference_mode():
            mlp_scores = bundle.mlp(bundle.normalizer.transform(features)).clamp_min(0)
        modes, spent = _router_modes(
            mlp_scores,
            artifacts.residual_fisher,
            total_budget_bits=dynamic_budget,
            dynamic_positions=True,
        )
        encoded = encode_residual_modes(
            base.reconstructed, crops, artifacts.residual_fisher, modes
        )
        ledger = RateLedger()
        _append_residual_shared(
            ledger, artifacts.base, artifacts.residual_fisher, base_tokens
        )
        mlp_router_shared_bits(
            ledger,
            bundle.mlp,
            bundle.normalizer,
        )
        router_curvature_shared_bits(ledger, bundle.curvature_prior)
        residual_stream_bits(
            ledger,
            artifacts.residual_fisher,
            modes,
            dynamic_positions=True,
            total_positions=len(locations),
            include_shared=False,
        )
        variants.append(
            TileLogicVariant(
                "base_vq_mlp_router",
                rate,
                "main",
                encoded.reconstructed,
                ledger,
                base_tokens,
                modes,
                dynamic_budget,
                spent + _count_header_bits(len(locations)),
                mlp_scores,
            )
        )

        variants.append(
            build_dynamic_logic_variant(thumbnail, crops, query, artifacts, rate)
        )

        variants.append(
            build_fixed_logic_variant(
                thumbnail, crops, query, artifacts, rate, exact_fallback=False
            )
        )
        variants.append(
            build_fixed_logic_variant(
                thumbnail, crops, query, artifacts, rate, exact_fallback=True
            )
        )

    expected = 1 + len(RETENTION_RATES) * (
        len(MAIN_TILELOGIC_METHODS) + len(ABLATION_METHODS)
    )
    if len(variants) != expected:
        raise AssertionError(f"method matrix has {len(variants)} variants, expected {expected}")
    return variants
