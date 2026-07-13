from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hessian_orthogonal_compression_probe as probe


def make_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(
        [[0.42, 0.24, 0.15, 0.10, 0.06, 0.03]], dtype=np.float64
    )
    logits, probability = probe.centered_fp16_logits(probability)
    damping = probe.fisher_damping(probability)
    return logits, probability, damping


def test_softmax_fisher_is_psd_and_has_gauge_null_direction() -> None:
    _, probability, _ = make_problem()
    delta = np.asarray([[0.7, -0.4, 0.2, -0.1, 0.3, -0.7]])
    assert probe.fisher_quadratic(probability, delta, 0.0) >= 0.0
    gauge = np.ones_like(probability)
    assert abs(probe.fisher_quadratic(probability, gauge, 0.0)) < 1e-14


def test_obs_pruning_is_h_orthogonal_to_any_retained_perturbation() -> None:
    logits, probability, damping = make_problem()
    pruned = np.asarray([[False, True, False, False, True, False]])
    delta = probe.obs_compensated_pruning(
        logits, probability, pruned, damping
    )
    retained_perturbation = np.asarray(
        [[0.3, 0.0, -0.2, 0.4, 0.0, -0.1]], dtype=np.float64
    )
    cross = probe.fisher_inner(
        probability, delta, retained_perturbation, damping
    )
    assert abs(cross) < 1e-12


def test_one_scale_compensation_removes_its_scale_direction_component() -> None:
    logits, probability, damping = make_problem()
    pruned = np.asarray([[False, True, False, False, True, False]])
    naive = probe.naive_pruning(logits, pruned)
    corrected, _ = probe.one_scale_compensated_pruning(
        logits, probability, pruned, damping
    )
    direction = np.where(~pruned, logits, 0.0)
    before = abs(probe.fisher_inner(probability, naive, direction, damping))
    after = abs(probe.fisher_inner(probability, corrected, direction, damping))
    assert after < before
    assert after <= 1e-3 * max(before, 1e-30)


def test_hessian_quadratic_decomposes_into_self_and_cross_terms() -> None:
    _, probability, damping = make_problem()
    left = np.asarray([[0.2, -0.1, 0.0, 0.1, -0.2, 0.0]])
    right = np.asarray([[-0.1, 0.0, 0.2, -0.2, 0.0, 0.1]])
    total = probe.fisher_quadratic(probability, left + right, damping)
    expected = (
        probe.fisher_quadratic(probability, left, damping)
        + probe.fisher_quadratic(probability, right, damping)
        + probe.fisher_inner(probability, left, right, damping)
    )
    assert np.isclose(total, expected, atol=1e-14)


def test_bounded_cross_null_folded_scale_reduces_prune_quant_cross_term() -> None:
    logits, probability, damping = make_problem()
    pruned = np.asarray([[False, True, False, False, True, False]])
    retained = ~pruned
    prune_delta = probe.naive_pruning(logits, pruned)
    max_delta, _, _ = probe.folded_scale_quantization(
        logits,
        probability,
        prune_delta,
        retained,
        4,
        damping,
        "max_scale",
    )
    cross_delta, _, _ = probe.folded_scale_quantization(
        logits,
        probability,
        prune_delta,
        retained,
        4,
        damping,
        "bounded_cross_null_scale",
    )
    max_cross = abs(
        probe.fisher_inner(probability, prune_delta, max_delta, damping)
    )
    nulled_cross = abs(
        probe.fisher_inner(probability, prune_delta, cross_delta, damping)
    )
    assert nulled_cross <= max_cross + 1e-12


def test_bounded_cross_null_is_exact_when_solution_is_interior() -> None:
    logits, probability, damping = make_problem()
    pruned = np.asarray([[False, True, False, False, True, False]])
    retained = ~pruned
    prune_delta = probe.naive_pruning(logits, pruned)
    quant_delta, _, boundary_hits = probe.folded_scale_quantization(
        logits,
        probability,
        prune_delta,
        retained,
        4,
        damping,
        "bounded_cross_null_scale",
    )
    if boundary_hits == 0:
        # The stored scale is FP16, so the remaining inner product is a
        # finite-precision residual rather than a symbolic zero.
        cross = abs(
            probe.fisher_inner(probability, prune_delta, quant_delta, damping)
        )
        norm = np.sqrt(
            probe.fisher_inner(probability, prune_delta, prune_delta, damping)
            * probe.fisher_inner(probability, quant_delta, quant_delta, damping)
        )
        assert cross / max(norm, 1e-30) < 2e-3


def test_support_payload_uses_the_smaller_bitmap_or_fixed_row_coo() -> None:
    bits, encoding, index_width = probe.support_payload_bits(64, 50)
    assert encoding == "bitmap"
    assert bits == 64 * 64
    assert index_width == 6

    bits, encoding, _ = probe.support_payload_bits(64, 3)
    assert encoding == "fixed_row_coo"
    assert bits == 64 * 3 * 6


def test_actual_kl_matches_fisher_quadratic_for_small_logit_perturbation() -> None:
    logits, probability, damping = make_problem()
    prune_delta = np.zeros_like(logits)
    quant_delta = 1e-3 * np.asarray(
        [[0.3, -0.2, 0.1, -0.1, 0.2, -0.3]], dtype=np.float64
    )
    metrics = probe.perturbation_metrics(
        probability, prune_delta, quant_delta, damping
    )
    assert 0.99 <= metrics["taylor_actual_to_quadratic_ratio"] <= 1.01


def test_matched_rate_single_never_uses_more_bits_than_combo() -> None:
    base = {
        "key": "x",
        "source_input_id": "s0",
        "dense_fp16_bits": 1000,
        "mean_actual_kl": 1.0,
        "mean_damped_hessian_quadratic": 1.0,
        "hessian_correlation": 0.0,
        "inside_taylor_comfort_zone": True,
    }
    rows = [
        {
            **base,
            "category": "single_quantization",
            "method": "single-low",
            "parameter_bits": 200,
        },
        {
            **base,
            "category": "single_pruning",
            "method": "single-close",
            "parameter_bits": 298,
            "mean_actual_kl": 0.1,
            "mean_damped_hessian_quadratic": 0.1,
        },
        {
            **base,
            "category": "combined_prune_quant",
            "method": "combo",
            "parameter_bits": 300,
            "mean_actual_kl": 0.4,
            "mean_damped_hessian_quadratic": 0.4,
        },
    ]
    matched = probe.matched_rate_rows(rows)
    assert matched
    assert all(
        row["single_parameter_bits"] <= row["combo_parameter_bits"]
        for row in matched
    )
    assert all(row["single_rate_coverage"] >= 0.99 for row in matched)
    assert all(row["strict_rate_coverage_pass"] for row in matched)


def test_matched_rate_rejects_underfilled_single_method_grid() -> None:
    base = {
        "key": "x",
        "source_input_id": "s0",
        "dense_fp16_bits": 1000,
        "mean_actual_kl": 1.0,
        "mean_damped_hessian_quadratic": 1.0,
        "hessian_correlation": 0.0,
        "inside_taylor_comfort_zone": True,
    }
    rows = [
        {
            **base,
            "category": "single_quantization",
            "method": "single-underfilled",
            "parameter_bits": 200,
        },
        {
            **base,
            "category": "combined_prune_quant",
            "method": "combo",
            "parameter_bits": 300,
            "mean_actual_kl": 0.4,
            "mean_damped_hessian_quadratic": 0.4,
        },
    ]
    assert probe.matched_rate_rows(rows) == []
