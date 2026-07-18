from __future__ import annotations

import torch

from tilespec_ex.core import RETENTION_RATES
from tilespec_ex.rate import RATE_STORAGE_POLICY
from tilespec_ex.routing import (
    FeatureBinarizer,
    FeatureNormalizer,
    LogicRegressionTree,
    LogicRouter,
    LogicTreeNode,
    RouterMLP,
    logic_router_storage_bits,
)
from tilespec_ex.tilelogic_methods import (
    ABLATION_METHODS,
    MAIN_TILELOGIC_METHODS,
    RouterBundle,
    TileLogicArtifacts,
    build_dynamic_logic_variant,
    build_fixed_logic_variant,
    build_tilelogic_variants,
)
from tilespec_ex.vq import ResidualVQCodebook, ScaledCodebook


def _locations() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (tile, row, col)
        for tile in range(4)
        for row in range(0, 16, 2)
        for col in range(0, 16, 2)
    )


def _artifacts() -> TileLogicArtifacts:
    generator = torch.Generator().manual_seed(20260718)
    base = ScaledCodebook(
        torch.randn(4, 2, generator=generator), torch.tensor([0.5, 1.0]), torch.ones(2)
    )
    residual_words = torch.zeros(2, 4, 8)
    residual_words[:, 1:] = torch.randn(2, 3, 8, generator=generator)
    residual = ResidualVQCodebook(
        residual_words, torch.tensor([0.5, 1.0]), torch.ones(8)
    )
    locations = _locations()
    routers = {}
    for rate, slots_per_tile in ((0.125, 2), (0.25, 4)):
        normalizer = FeatureNormalizer(torch.zeros(13), torch.ones(13))
        mlp = RouterMLP(13, output_dim=3, hidden_dim=4).eval()
        binarizer = FeatureBinarizer(torch.zeros(13, 1))
        tree = LogicRegressionTree((LogicTreeNode(-1, -1, -1, 1.0),), 13)
        mask = torch.zeros(256, dtype=torch.bool)
        for tile in range(4):
            mask[tile * 64 : tile * 64 + slots_per_tile] = True
        routers[rate] = RouterBundle(
            rate,
            normalizer,
            mlp,
            LogicRouter(binarizer, (tree, tree, tree)),
            torch.ones(256),
            mask,
            locations,
        )
    return TileLogicArtifacts(base, residual, residual, routers)


def test_full_method_matrix_has_exact_rate_ledgers() -> None:
    generator = torch.Generator().manual_seed(5)
    crops = torch.randn(4, 16, 16, 2, generator=generator)
    thumbnail = torch.randn(16, 16, 2, generator=generator)
    query = torch.randn(2, generator=generator)
    artifacts = _artifacts()
    variants = build_tilelogic_variants(thumbnail, crops, query, artifacts)
    assert len(variants) == 1 + len(RETENTION_RATES) * (
        len(MAIN_TILELOGIC_METHODS) + len(ABLATION_METHODS)
    )
    names_by_rate = {
        rate: {variant.method for variant in variants if variant.rate == rate}
        for rate in RETENTION_RATES
    }
    for rate in RETENTION_RATES:
        assert names_by_rate[rate] == set(MAIN_TILELOGIC_METHODS) | set(
            ABLATION_METHODS
        )
    for variant in variants:
        assert variant.reconstructed.shape == crops.shape
        assert variant.ledger.stream_bits > 0
        if variant.residual_modes is not None:
            assert variant.residual_spent_bits <= variant.residual_budget_bits
        component_bits = {
            item.name: item.bits for item in variant.ledger.components
        }
        if variant.method == "base_scalar_quant":
            assert component_bits["base_scalar_scales"] == (
                variant.base_tokens * RATE_STORAGE_POLICY["base_scalar_scale_bits"]
            )
        if variant.method in {
            "base_vq_mlp_router",
            "base_vq_logic_router",
            "logic_router_fixed_slots",
            "logic_router_fixed_slots_exact_fallback",
        }:
            assert component_bits["router_curvature_prior"] == 256 * 32
        if variant.method == "base_vq_mlp_router":
            bundle = artifacts.routers[float(variant.rate)]
            assert component_bits["mlp_router_parameters"] == sum(
                value.numel() * value.element_size() * 8
                for value in bundle.mlp.parameters()
            )
            assert component_bits["mlp_router_normalizer"] == 2 * 13 * 32
        if variant.method in {
            "base_vq_logic_router",
            "logic_router_fixed_slots",
            "logic_router_fixed_slots_exact_fallback",
        }:
            bundle = artifacts.routers[float(variant.rate)]
            assert component_bits["logic_router"] == logic_router_storage_bits(
                bundle.logic
            )


def test_targeted_logic_builders_match_full_matrix() -> None:
    generator = torch.Generator().manual_seed(19)
    crops = torch.randn(4, 16, 16, 2, generator=generator)
    thumbnail = torch.randn(16, 16, 2, generator=generator)
    query = torch.randn(2, generator=generator)
    artifacts = _artifacts()
    full = build_tilelogic_variants(thumbnail, crops, query, artifacts)
    for rate in RETENTION_RATES:
        targeted_dynamic = build_dynamic_logic_variant(
            thumbnail, crops, query, artifacts, rate
        )
        matrix_dynamic = next(
            variant
            for variant in full
            if variant.rate == rate and variant.method == targeted_dynamic.method
        )
        torch.testing.assert_close(
            targeted_dynamic.reconstructed, matrix_dynamic.reconstructed
        )
        torch.testing.assert_close(
            targeted_dynamic.router_marginal_scores,
            matrix_dynamic.router_marginal_scores,
        )
        assert torch.equal(
            targeted_dynamic.residual_modes, matrix_dynamic.residual_modes
        )
        assert targeted_dynamic.ledger.as_dict() == matrix_dynamic.ledger.as_dict()
        assert (
            targeted_dynamic.residual_spent_bits
            == matrix_dynamic.residual_spent_bits
        )
        for exact in (False, True):
            targeted = build_fixed_logic_variant(
                thumbnail, crops, query, artifacts, rate, exact_fallback=exact
            )
            matrix = next(
                variant
                for variant in full
                if variant.rate == rate and variant.method == targeted.method
            )
            torch.testing.assert_close(targeted.reconstructed, matrix.reconstructed)
            assert targeted.ledger.as_dict() == matrix.ledger.as_dict()
            assert targeted.residual_spent_bits == matrix.residual_spent_bits
