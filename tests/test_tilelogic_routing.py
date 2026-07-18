from __future__ import annotations

import torch

from tilespec_ex.routing import (
    BLOCK_FEATURE_NAMES,
    FeatureBinarizer,
    FeatureNormalizer,
    LogicRouter,
    RouterMLP,
    allocate_variable_depth,
    block_router_features,
    fit_logic_router,
    fixed_slot_mask,
)


def _tile_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260718)
    crops = torch.randn(4, 16, 16, 8, generator=generator)
    residual = crops * 0.2
    thumbnail = torch.randn(16, 16, 8, generator=generator)
    return crops, residual, thumbnail


def test_block_router_features_have_stable_schema() -> None:
    crops, residual, thumbnail = _tile_tensors()
    query = torch.randn(8, generator=torch.Generator().manual_seed(2))
    curvature = torch.linspace(0, 1, 256)
    features, locations = block_router_features(
        crops,
        residual,
        query,
        thumbnail,
        curvature_prior=curvature,
    )
    assert features.shape == (256, len(BLOCK_FEATURE_NAMES))
    assert len(locations) == 256
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[:, -1], curvature)


def test_block_router_features_accept_flattened_thumbnail_cache_layout() -> None:
    crops, residual, thumbnail = _tile_tensors()
    query = torch.randn(8, generator=torch.Generator().manual_seed(21))
    grid_features, grid_locations = block_router_features(
        crops, residual, query, thumbnail
    )
    flat_features, flat_locations = block_router_features(
        crops, residual, query, thumbnail.reshape(256, 8)
    )
    assert flat_locations == grid_locations
    torch.testing.assert_close(flat_features, grid_features)


def test_feature_normalizer_and_binarizer_are_calibration_only_state() -> None:
    features = torch.randn(100, 5, generator=torch.Generator().manual_seed(3))
    normalizer = FeatureNormalizer.fit(features)
    normalized = normalizer.transform(features)
    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(5), atol=1e-6, rtol=0)
    binarizer = FeatureBinarizer.fit(features)
    bits = binarizer.transform(features)
    assert bits.shape == (100, 15)
    assert bits.dtype == torch.bool


def test_logic_router_distills_multiple_marginal_actions() -> None:
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(512, 4, generator=generator)
    targets = torch.stack(
        (
            (features[:, 0] > 0).float(),
            (features[:, 1] > 0.5).float() * 2,
            (features[:, 2] + features[:, 3]).relu(),
        ),
        dim=1,
    )
    router = fit_logic_router(features, targets, max_depth=5, min_leaf=8)
    prediction = router.predict(features)
    assert prediction.shape == targets.shape
    assert torch.isfinite(prediction).all()
    state = router.state_dict()
    for tree_state in state["trees"]:
        assert tree_state["format"] == "tilespec_logic_regression_tree_v2"
        assert tree_state["values"].dtype == torch.float32
    restored = LogicRouter.from_state_dict(state)
    torch.testing.assert_close(restored.predict(features), prediction)


def test_variable_depth_allocator_respects_precedence_and_budget() -> None:
    benefits = torch.tensor(
        [
            [10.0, 1.0, 100.0],
            [8.0, 7.0, 0.0],
            [1.0, 9.0, 0.0],
        ]
    )
    costs = torch.tensor([[4, 2, 20], [4, 2, 20], [4, 2, 20]])
    modes, spent = allocate_variable_depth(benefits, costs, budget_bits=10)
    assert spent <= 10
    assert torch.equal(modes, torch.tensor([1, 2, 0]))
    assert bool((modes >= 0).all())


def test_fixed_slots_are_selected_independently_per_tile() -> None:
    importance = torch.arange(256, dtype=torch.float32)
    locations = tuple(
        (tile, row, col)
        for tile in range(4)
        for row in range(0, 16, 2)
        for col in range(0, 16, 2)
    )
    mask = fixed_slot_mask(importance, locations, slots_per_tile=3)
    assert int(mask.sum()) == 12
    for tile in range(4):
        tile_mask = torch.tensor([location[0] == tile for location in locations])
        assert int((mask & tile_mask).sum()) == 3


def test_router_mlp_export_round_trip() -> None:
    model = RouterMLP(5, output_dim=3, hidden_dim=7).eval()
    features = torch.randn(4, 5, generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        expected = model(features)
    restored = RouterMLP.from_export(model.export_state()).eval()
    with torch.inference_mode():
        actual = restored(features)
    torch.testing.assert_close(actual, expected)
