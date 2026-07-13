from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparsity_repair_probe as probe


def test_row_topk_mask_has_fixed_count() -> None:
    matrix = np.arange(48, dtype=np.float64).reshape(6, 8)
    mask = probe.row_topk_mask(matrix, 3)
    assert mask.shape == matrix.shape
    assert np.all(mask.sum(axis=1) == 3)
    assert np.all(mask[:, -3:])


def test_mass_conserving_tail_repairs_pruning_without_stored_scale() -> None:
    target = np.asarray(
        [
            [0.55, 0.25, 0.10, 0.10],
            [0.40, 0.30, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    mask = probe.row_topk_mask(target, 2)
    kept = target * mask
    template = probe.uniform_complement_template(mask)
    repaired = kept + (1.0 - kept.sum(axis=1, keepdims=True)) * template
    base_loss = probe.core.evaluate_candidate(target, kept)["normalized_mse"]
    repaired_loss = probe.core.evaluate_candidate(target, repaired)["normalized_mse"]
    assert np.allclose(repaired.sum(axis=1), 1.0)
    assert repaired_loss < base_loss


def test_quantized_shape_mass_scale_protects_raw_amplitude_and_tail() -> None:
    target = np.asarray(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.45, 0.25, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    mask = probe.row_topk_mask(target, 2)
    shape, mass, _ = probe.quantized_kept_shape(target, mask, 3)
    mass_only = mass * shape
    uniform = probe.uniform_complement_template(mask)
    repaired = mass_only + (1.0 - mass) * uniform
    assert probe.core.relative_error(target, mass_only) < probe.core.relative_error(
        target, shape
    )
    assert probe.core.evaluate_candidate(target, repaired)["normalized_mse"] < (
        probe.core.evaluate_candidate(target, shape)["normalized_mse"]
    )
    # A common row mass is intentionally removed if the sparse row is normalized alone.
    assert np.isclose(
        probe.core.evaluate_candidate(target, mass_only)["normalized_mse"],
        probe.core.evaluate_candidate(target, shape)["normalized_mse"],
    )


def test_common_row_scale_is_a_renormalization_noop() -> None:
    raw = np.asarray(
        [[0.4, 0.1, 0.0], [0.2, 0.3, 0.0]], dtype=np.float64
    )
    scales = np.asarray([[0.3], [7.0]], dtype=np.float64)
    assert np.allclose(
        probe.core.row_normalize_nonnegative(raw),
        probe.core.row_normalize_nonnegative(raw * scales),
        atol=1e-14,
    )


def test_column_prior_tail_beats_equal_bit_sparse_on_diffuse_tail() -> None:
    target = np.asarray(
        [
            [0.50, 0.20, 0.15, 0.10, 0.05],
            [0.40, 0.25, 0.15, 0.10, 0.10],
            [0.45, 0.20, 0.15, 0.10, 0.10],
            [0.35, 0.25, 0.20, 0.10, 0.10],
            [0.30, 0.25, 0.20, 0.15, 0.10],
        ],
        dtype=np.float64,
    )
    mask = probe.row_topk_mask(target, 1)
    kept = target * mask
    tail_mass = 1.0 - kept.sum(axis=1, keepdims=True)
    prior_tail, _ = probe.column_prior_template(target, mask)
    prior_repair = kept + tail_mass * prior_tail
    index_width = 3
    sparse_repair, _, _ = probe.add_extra_sparse_with_budget(
        target, mask, target.shape[1] * probe.VALUE_BITS, index_width
    )
    assert probe.core.evaluate_candidate(target, prior_repair)["normalized_mse"] < (
        probe.core.evaluate_candidate(target, sparse_repair)["normalized_mse"]
    )


def test_block_mass_repair_improves_block_pruning() -> None:
    rng = np.random.default_rng(3)
    target = rng.random((8, 8))
    target /= target.sum(axis=1, keepdims=True)
    mask, block_mask, _ = probe.block_topk_mask(target, 4, 0.5)
    kept = target * mask
    repaired = probe.block_mass_repair(target, block_mask, 4, per_row=False)
    assert probe.core.evaluate_candidate(target, repaired)["normalized_mse"] < (
        probe.core.evaluate_candidate(target, kept)["normalized_mse"]
    )


def test_per_row_sparse_scale_handles_amplitude_heterogeneity() -> None:
    sparse = np.asarray(
        [
            [0.001, 0.002, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0],
            [0.1, 0.2, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mask = sparse > 0
    global_q, global_count, _ = probe.quantize_sparse_values(
        sparse, mask, 2, "global"
    )
    row_q, row_count, _ = probe.quantize_sparse_values(sparse, mask, 2, "per_row")
    assert np.linalg.norm(sparse - row_q) <= np.linalg.norm(sparse - global_q)
    assert global_count == 1
    assert row_count == sparse.shape[0]


def test_sparse_orders_have_same_bits_and_respect_cap() -> None:
    examples, _ = probe.core.load_examples()
    target = probe.core.row_normalize_nonnegative(examples[0]["attention"])
    built = {}
    for order in ("backbone_first", "component_first"):
        result = probe.build_sparse_stages(
            target,
            0.25,
            4,
            "block_row",
            order,
            stages=1,
            loss_aware=False,
        )
        assert result is not None
        raw, budget = result
        built[order] = budget
        assert np.all(np.isfinite(raw))
        assert budget["parameter_bits"] <= 0.25 * target.size * probe.VALUE_BITS
    assert built["backbone_first"]["parameter_bits"] == built["component_first"][
        "parameter_bits"
    ]
    assert built["backbone_first"]["k"] == built["component_first"]["k"]


def test_loss_aware_folded_scale_never_worsens_its_fixed_backbone_path() -> None:
    examples, _ = probe.core.load_examples()
    target = probe.core.row_normalize_nonnegative(examples[0]["attention"])
    standard = probe.build_sparse_stages(
        target, 0.25, 4, "per_row", "component_first", 1, False
    )
    aware = probe.build_sparse_stages(
        target, 0.25, 4, "per_row", "component_first", 1, True
    )
    assert standard is not None and aware is not None
    standard_raw, standard_budget = standard
    aware_raw, aware_budget = aware
    assert standard_budget["parameter_bits"] == aware_budget["parameter_bits"]
    assert probe.core.evaluate_candidate(target, aware_raw)["normalized_mse"] <= (
        probe.core.evaluate_candidate(target, standard_raw)["normalized_mse"] + 1e-12
    )


def test_folded_gain_is_representable_by_declared_fp16_payload() -> None:
    target = np.asarray([[0.7, 0.2, 0.1, 0.0]], dtype=np.float64)
    backbone = np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float64)

    exact = probe.fp16_round(np.asarray([[0.23, 0.11, 0.0, 0.0]]))
    exact_adjusted, _ = probe.optimize_folded_group_gain(
        target, backbone, exact, "global", probe.VALUE_BITS
    )
    np.testing.assert_array_equal(exact_adjusted, probe.fp16_round(exact_adjusted))

    base_scale = float(probe.fp16_round(0.037))
    codes = np.asarray([[15.0, 7.0, 2.0, 0.0]])
    quantized = codes * base_scale
    quantized_adjusted, _ = probe.optimize_folded_group_gain(
        target, backbone, quantized, "global", 4
    )
    folded_scale = float(np.max(quantized_adjusted)) / 15.0
    assert folded_scale == float(probe.fp16_round(folded_scale))
    np.testing.assert_allclose(
        quantized_adjusted / max(folded_scale, 1e-15), codes, atol=1e-12
    )


def test_multistage_error_feedback_is_finite_and_budgeted() -> None:
    examples, _ = probe.core.load_examples()
    target = probe.core.row_normalize_nonnegative(examples[0]["attention"])
    result = probe.build_sparse_stages(
        target, 0.25, 4, "block_row", "component_first", 4, True
    )
    assert result is not None
    raw, budget = result
    assert budget["stages"] == 4
    assert budget["parameter_bits"] <= 0.25 * target.size * probe.VALUE_BITS
    assert np.all(np.isfinite(raw))


def test_multistage_loss_aware_updates_do_not_worsen_same_payload() -> None:
    examples, _ = probe.core.load_examples()
    target = probe.core.row_normalize_nonnegative(examples[0]["attention"])
    standard = probe.build_sparse_stages(
        target, 0.25, 4, "block_row", "component_first", 4, False
    )
    aware = probe.build_sparse_stages(
        target, 0.25, 4, "block_row", "component_first", 4, True
    )
    assert standard is not None and aware is not None
    standard_raw, standard_budget = standard
    aware_raw, aware_budget = aware
    assert standard_budget["parameter_bits"] == aware_budget["parameter_bits"]
    assert probe.core.evaluate_candidate(target, aware_raw)["normalized_mse"] <= (
        probe.core.evaluate_candidate(target, standard_raw)["normalized_mse"] + 1e-12
    )


def test_pareto_keeps_zero_bit_endpoint() -> None:
    rows = [
        {"key": "x", "parameter_bits": 0, "normalized_mse": 0.9},
        {"key": "x", "parameter_bits": 10, "normalized_mse": 0.5},
        {"key": "x", "parameter_bits": 20, "normalized_mse": 0.6},
    ]
    pareto = probe.mark_pareto(rows)
    assert rows[0] in pareto
    assert rows[1] in pareto
    assert rows[2] not in pareto
