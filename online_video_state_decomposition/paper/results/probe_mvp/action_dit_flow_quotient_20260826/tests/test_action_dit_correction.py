from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from action_dit_correction import (  # noqa: E402
    BucketReducedRankRegressor,
    TemporalKernelRegressor,
    TemporalLowRankRegressor,
    bucket_ids,
    observed_step_count,
)


def synthetic_payload(seed: int = 7):
    rng = np.random.default_rng(seed)
    count, horizon, action_dim = 256, 10, 2
    noisy = rng.normal(size=(count, horizon, action_dim))
    quantized = rng.normal(size=(count, horizon, action_dim))
    condition = rng.normal(size=(count, 6))
    buckets = np.repeat(np.arange(2), count // 2)

    local = np.zeros_like(noisy)
    local[:, 1:] += 0.4 * noisy[:, :-1]
    local[:, :-1] -= 0.2 * quantized[:, 1:]
    global_direction = np.linspace(-1.0, 1.0, horizon * action_dim).reshape(
        horizon, action_dim
    )
    coefficient = condition[:, :1] * 0.3
    defect = local + coefficient[:, None] * global_direction
    return noisy, quantized, condition, defect, buckets


def test_bucket_ids_cover_registered_range():
    steps = np.arange(100)
    buckets = bucket_ids(steps, 100, 10)
    assert buckets.min() == 0
    assert buckets.max() == 9
    assert np.bincount(buckets).tolist() == [10] * 10


def test_observed_step_count_uses_executed_schedule_length():
    scheduler_steps = np.repeat(np.arange(101), 4)
    assert observed_step_count(scheduler_steps) == 101


def test_nonperiodic_kernel_recovers_boundary_sensitive_relation():
    noisy, quantized, condition, defect, buckets = synthetic_payload()
    train = np.arange(192)
    test = np.arange(192, 256)
    toeplitz = TemporalKernelRegressor(1, False, 1e-6).fit(
        noisy[train], quantized[train], condition[train], defect[train], buckets[train]
    )
    circular = TemporalKernelRegressor(1, True, 1e-6).fit(
        noisy[train], quantized[train], condition[train], defect[train], buckets[train]
    )
    toeplitz_error = np.linalg.norm(
        toeplitz.predict(noisy[test], quantized[test], condition[test], buckets[test])
        - defect[test]
    )
    circular_error = np.linalg.norm(
        circular.predict(noisy[test], quantized[test], condition[test], buckets[test])
        - defect[test]
    )
    assert toeplitz_error < circular_error


def test_temporal_low_rank_recovers_local_and_global_defect():
    noisy, quantized, condition, defect, buckets = synthetic_payload()
    train = np.arange(192)
    test = np.arange(192, 256)
    temporal = TemporalKernelRegressor(1, False, 1e-4).fit(
        noisy[train], quantized[train], condition[train], defect[train], buckets[train]
    )
    hybrid = TemporalLowRankRegressor(1, 2, 1e-4).fit(
        noisy[train], quantized[train], condition[train], defect[train], buckets[train]
    )
    temporal_error = np.linalg.norm(
        temporal.predict(noisy[test], quantized[test], condition[test], buckets[test])
        - defect[test]
    )
    hybrid_error = np.linalg.norm(
        hybrid.predict(noisy[test], quantized[test], condition[test], buckets[test])
        - defect[test]
    )
    assert hybrid_error < 0.1 * temporal_error


def test_reduced_rank_predictor_has_bounded_payload():
    noisy, quantized, condition, defect, buckets = synthetic_payload()
    model = BucketReducedRankRegressor(2, 1e-4).fit(
        noisy, quantized, condition, defect, buckets
    )
    prediction = model.predict(noisy, quantized, condition, buckets)
    assert prediction.shape == defect.shape
    assert model.parameter_count < 1000
    assert model.macs_per_sample < 500
