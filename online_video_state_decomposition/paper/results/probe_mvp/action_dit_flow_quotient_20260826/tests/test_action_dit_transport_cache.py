import numpy as np

from action_dit_transport_cache import (
    DepthwiseTemporalRegressor,
    FrozenBasis,
    RidgeMap,
    captured_energy,
    horizon_shift,
    overlap_mask,
    reuse_with_exact_tail,
)


def test_horizon_shift_and_equal_budget_reuse():
    values = np.arange(10, dtype=np.float32).reshape(1, 5, 2)
    current = values + 100
    shifted = horizon_shift(values, 2)
    np.testing.assert_array_equal(shifted[0, :3], values[0, 2:])
    np.testing.assert_array_equal(shifted[0, 3:], 0)
    aligned = reuse_with_exact_tail(values, current, 2, aligned=True)
    raw = reuse_with_exact_tail(values, current, 2, aligned=False)
    np.testing.assert_array_equal(aligned[0, :3], values[0, 2:])
    np.testing.assert_array_equal(raw[0, :3], values[0, :3])
    np.testing.assert_array_equal(aligned[0, 3:], current[0, 3:])
    np.testing.assert_array_equal(overlap_mask(5, 2), [True, True, True, False, False])


def test_depthwise_nonperiodic_temporal_recovery():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(128, 7, 3)).astype(np.float32)
    true_model = DepthwiseTemporalRegressor(1, False, 1e-8)
    true_model.weight = np.array(
        [[0.25, 1.0, -0.5], [0.1, -0.2, 0.3], [-0.4, 0.7, 0.2]],
        dtype=np.float32,
    )
    true_model.bias = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    target = true_model.predict(source)
    fitted = DepthwiseTemporalRegressor(1, False, 1e-6).fit(
        source, target, np.ones(7, dtype=bool)
    )
    assert captured_energy(target, fitted.predict(source)) > 0.99999


def test_frozen_basis_and_ridge_map_recover_low_dimensional_signal():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(160, 6)).astype(np.float32)
    coefficient_map = rng.normal(size=(6, 3)).astype(np.float32)
    coefficients = features @ coefficient_map
    basis_rows, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    values = (coefficients @ basis_rows.T).reshape(160, 5, 4).astype(np.float32)
    basis = FrozenBasis.fit(values[:100], rank=3)
    projected = basis.project(values[100:])
    assert captured_energy(values[100:], projected) > 0.99999
    train_coefficients = basis.coefficients(values[:100])
    predictor = RidgeMap(1e-6).fit(features[:100], train_coefficients)
    predicted = basis.reconstruct(predictor.predict(features[100:]))
    assert captured_energy(values[100:], predicted) > 0.9999
