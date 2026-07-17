from __future__ import annotations

import pytest
import torch

from scripts.benchmark_tilespec_ex_latency import (
    _fixed_per_tile_indices,
    compression_budget,
)
from tilespec_ex.core import (
    METHODS,
    RETENTION_RATES,
    compress_crop_tiles,
    compress_risk_structure_variant,
    dct2,
    idct2,
    normalized_mse,
    stitch_crop_tiles,
    unstack_crop_grid,
    zigzag_coordinates,
)


def _features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260717)
    return torch.randn(4, 16, 16, 12, generator=generator)


def test_dct_round_trip_is_orthonormal() -> None:
    tensor = _features()[0]
    restored = idct2(dct2(tensor))
    torch.testing.assert_close(restored, tensor, atol=2e-5, rtol=2e-5)


def test_zigzag_covers_each_coordinate_once() -> None:
    coordinates = zigzag_coordinates(5, 7)
    assert len(coordinates) == 35
    assert len(set(coordinates)) == 35
    assert coordinates[0] == (0, 0)


def test_stitch_and_unstack_are_inverse() -> None:
    features = _features()
    torch.testing.assert_close(unstack_crop_grid(stitch_crop_tiles(features)), features)


@pytest.mark.parametrize("rate", RETENTION_RATES)
@pytest.mark.parametrize("method", METHODS)
def test_all_methods_honor_declared_budget(method: str, rate: float) -> None:
    features = _features()
    query = torch.randn(features.shape[-1], generator=torch.Generator().manual_seed(9))
    result = compress_crop_tiles(
        features,
        method,
        rate,
        query_embedding=query if method == "tile_risk_exception" else None,
    )
    expected = features.shape[0] * features.shape[1] * features.shape[2]
    if method != "none":
        expected = round(expected * rate)
    assert result.retained_tokens == expected
    assert result.compact.shape == (expected, features.shape[-1])
    assert result.reconstructed.shape == features.shape
    assert result.base_tokens + result.exception_tokens == expected


@pytest.mark.parametrize("method", ["tile_energy_exception", "tile_risk_exception"])
def test_exception_methods_exactly_restore_selected_blocks(method: str) -> None:
    features = _features()
    query = torch.randn(features.shape[-1], generator=torch.Generator().manual_seed(11))
    result = compress_crop_tiles(
        features,
        method,
        0.25,
        query_embedding=query if method == "tile_risk_exception" else None,
    )
    assert result.selected_blocks
    for tile, row, col in result.selected_blocks:
        torch.testing.assert_close(
            result.reconstructed[tile, row : row + 2, col : col + 2],
            features[tile, row : row + 2, col : col + 2],
            atol=2e-5,
            rtol=2e-5,
        )


def test_energy_exceptions_improve_their_own_lowpass_base() -> None:
    features = _features()
    result = compress_crop_tiles(features, "tile_energy_exception", 0.25)
    base_rate = result.base_tokens / features.shape[0] / features.shape[1] / features.shape[2]
    base = compress_crop_tiles(features, "tile_lowpass", base_rate)
    assert normalized_mse(features, result.reconstructed) <= normalized_mse(
        features, base.reconstructed
    )


def test_risk_requires_query_embedding() -> None:
    with pytest.raises(ValueError, match="query embedding"):
        compress_crop_tiles(_features(), "tile_risk_exception", 0.25)


@pytest.mark.parametrize(
    "variant",
    [
        "risk_token_unstructured",
        "risk_block_dynamic",
        "risk_block_fixed_slots",
    ],
)
@pytest.mark.parametrize("rate", RETENTION_RATES)
def test_structure_variants_use_identical_budgets(variant: str, rate: float) -> None:
    features = _features()
    query = torch.randn(features.shape[-1], generator=torch.Generator().manual_seed(13))
    result = compress_risk_structure_variant(
        features, variant, rate, query_embedding=query
    )
    expected = round(features.shape[0] * features.shape[1] * features.shape[2] * rate)
    assert result.retained_tokens == expected
    assert result.compact.shape == (expected, features.shape[-1])
    assert result.reconstructed.shape == features.shape


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(0.125, (128, 96, 32, 8)), (0.25, (256, 192, 64, 16))],
)
def test_latency_budget_matches_quality_representation(
    rate: float, expected: tuple[int, int, int, int]
) -> None:
    assert compression_budget(rate) == expected


def test_fixed_slot_selection_consumes_equal_per_tile_indices() -> None:
    scores = torch.arange(256, dtype=torch.float32).repeat(2, 1)
    indices = _fixed_per_tile_indices(scores, 16)
    assert indices.shape == (2, 16)
    for batch_indices in indices:
        tile_counts = torch.bincount(batch_indices // 64, minlength=4)
        torch.testing.assert_close(tile_counts, torch.full((4,), 4))


def test_dynamic_block_variant_matches_headline_risk_method() -> None:
    features = _features()
    query = torch.randn(features.shape[-1], generator=torch.Generator().manual_seed(17))
    dynamic = compress_risk_structure_variant(
        features, "risk_block_dynamic", 0.25, query_embedding=query
    )
    headline = compress_crop_tiles(
        features, "tile_risk_exception", 0.25, query_embedding=query
    )
    assert dynamic.selected_blocks == headline.selected_blocks
    torch.testing.assert_close(dynamic.compact, headline.compact)
    torch.testing.assert_close(dynamic.reconstructed, headline.reconstructed)
