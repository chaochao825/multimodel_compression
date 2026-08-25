import numpy as np
import pytest

from action_dit_transport_cache import (
    FrozenBasis,
    RidgeMap,
    coefficient_r2,
    flatten_feature_groups,
    oracle_gap_recovery,
    transfer_basis_coefficients,
)


def test_flatten_feature_groups_preserves_samples():
    first = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    second = np.arange(6, dtype=np.float32).reshape(3, 2)
    features = flatten_feature_groups(first, second)
    assert features.shape == (3, 10)
    np.testing.assert_array_equal(features[:, :8], first.reshape(3, -1))


def test_flatten_feature_groups_rejects_mismatched_samples():
    with pytest.raises(ValueError):
        flatten_feature_groups(np.zeros((3, 2)), np.zeros((4, 2)))


def test_oracle_gap_recovery_has_expected_endpoints():
    assert oracle_gap_recovery(0.4, 0.4, 0.1) == pytest.approx(0.0)
    assert oracle_gap_recovery(0.4, 0.1, 0.1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        oracle_gap_recovery(0.1, 0.1, 0.2)


def test_linear_noise_bridge_recovers_heldout_coefficients():
    rng = np.random.default_rng(7)
    train_noise = rng.normal(size=(128, 10)).astype(np.float32)
    test_noise = rng.normal(size=(32, 10)).astype(np.float32)
    weight = rng.normal(size=(10, 4)).astype(np.float32)
    train_coefficients = train_noise @ weight
    test_coefficients = test_noise @ weight
    model = RidgeMap(1e-6).fit(train_noise, train_coefficients)
    prediction = model.predict(test_noise)
    assert coefficient_r2(prediction, test_coefficients) > 0.999


def test_frozen_basis_and_noise_bridge_reconstruct_synthetic_innovation():
    rng = np.random.default_rng(11)
    train_noise = rng.normal(size=(128, 6)).astype(np.float32)
    test_noise = rng.normal(size=(32, 6)).astype(np.float32)
    weight = rng.normal(size=(6, 3)).astype(np.float32)
    atoms = rng.normal(size=(3, 2, 8)).astype(np.float32)
    train = np.einsum("nr,rhd->nhd", train_noise @ weight, atoms)
    test = np.einsum("nr,rhd->nhd", test_noise @ weight, atoms)
    basis = FrozenBasis.fit(train, rank=3)
    bridge = RidgeMap(1e-6).fit(train_noise, basis.coefficients(train))
    prediction = basis.reconstruct(bridge.predict(test_noise))
    error = np.linalg.norm(prediction - test) / np.linalg.norm(test)
    assert error < 1e-4


def test_transfer_basis_coefficients_matches_dense_reprojection():
    rng = np.random.default_rng(17)
    source_values = rng.normal(size=(32, 3, 5)).astype(np.float32)
    target_values = rng.normal(size=(32, 3, 5)).astype(np.float32)
    source = FrozenBasis.fit(source_values, rank=4)
    target = FrozenBasis.fit(target_values, rank=4)
    coefficients = rng.normal(size=(7, 4)).astype(np.float32)
    expected = target.coefficients(source.reconstruct(coefficients))
    actual = transfer_basis_coefficients(coefficients, source, target)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
