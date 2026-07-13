from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compression_loss_landscape_probe as probe


def test_independent_block_projection_matches_saved_b4_examples() -> None:
    sources = [
        (
            ROOT / "remote_logs" / "structured_attention_visual_vit_examples_hybrid_20260704.npz",
            ("ex0", "ex1"),
        ),
        (
            ROOT / "remote_logs" / "structured_attention_visual_qwen_examples_hybrid_20260704.npz",
            ("ex0", "ex1"),
        ),
    ]
    for path, keys in sources:
        arrays = np.load(path)
        for key in keys:
            attention = arrays[f"{key}_attention"].astype(np.float64)
            kernels, blocks_per_axis = probe.extract_block_kernels(attention, 4)
            reconstructed = probe.expand_block_kernels(kernels, blocks_per_axis, 4)
            assert np.allclose(
                reconstructed,
                arrays[f"{key}_flat_block_circulant"],
                atol=1e-7,
            )


def test_block_scale_rank1_improves_over_equal_shared_kernel() -> None:
    kernels = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.2, 0.4, 0.6, 0.8],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    shared = np.broadcast_to(kernels.mean(axis=0, keepdims=True), kernels.shape)
    scaled, stats = probe.gamma_scaled_kernels(kernels, 1.0, reoptimize_kernel=True)
    assert np.linalg.norm(kernels - scaled) <= np.linalg.norm(kernels - shared) + 1e-12
    assert np.allclose(kernels, scaled, atol=1e-10)
    assert np.isclose(stats["rank1_kernel_energy_capture"], 1.0)


def test_gamma_paths_start_at_shared_and_scale_fits_do_not_regress() -> None:
    kernels = np.asarray(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.0, 0.1],
            [2.0, 0.1, 0.4, 0.2],
            [0.2, 0.8, 0.1, 0.3],
        ],
        dtype=np.float64,
    )
    shared = np.broadcast_to(kernels.mean(axis=0, keepdims=True), kernels.shape)
    fixed_zero, _ = probe.gamma_scaled_kernels(kernels, 0.0, reoptimize_kernel=False)
    joint_zero, _ = probe.gamma_scaled_kernels(kernels, 0.0, reoptimize_kernel=True)
    fixed_one, _ = probe.gamma_scaled_kernels(kernels, 1.0, reoptimize_kernel=False)
    joint_one, _ = probe.gamma_scaled_kernels(kernels, 1.0, reoptimize_kernel=True)
    assert np.allclose(fixed_zero, shared)
    assert np.allclose(joint_zero, shared)
    assert np.linalg.norm(kernels - fixed_one) <= np.linalg.norm(kernels - shared) + 1e-12
    assert np.linalg.norm(kernels - joint_one) <= np.linalg.norm(kernels - fixed_one) + 1e-12


def test_fixed_kernel_gamma_coefficient_loss_is_quadratic_before_clipping() -> None:
    kernels = np.asarray(
        [
            [0.9, 0.2, 0.1, 0.3],
            [0.4, 0.5, 0.2, 0.1],
            [1.2, 0.1, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    gammas = (0.0, 0.25, 0.5, 0.75, 1.0)
    losses = []
    for gamma in gammas:
        approx, _ = probe.gamma_scaled_kernels(kernels, gamma, reoptimize_kernel=False)
        losses.append(float(np.sum((kernels - approx) ** 2)))
    second_differences = np.diff(losses, n=2)
    assert np.allclose(second_differences, second_differences[0], atol=1e-12)
    assert losses[-1] == min(losses)


def test_rank_max_and_redundant_scale_match_independent_and_bit_formula() -> None:
    arrays = np.load(
        ROOT / "remote_logs" / "structured_attention_visual_vit_examples_hybrid_20260704.npz"
    )
    attention = arrays["ex0_attention"].astype(np.float64)
    kernels, blocks_per_axis = probe.extract_block_kernels(attention, 4)
    rank_max, _ = probe.rank_approximation(kernels, kernels.shape[1])
    independent = probe.expand_block_kernels(kernels, blocks_per_axis, 4)
    reconstructed = probe.expand_block_kernels(rank_max, blocks_per_axis, 4)
    assert np.allclose(reconstructed, independent, atol=1e-12)

    examples, _ = probe.load_examples()
    rows: list[dict] = []
    probe.add_backbone_rows(examples[0], rows)
    base = next(
        row
        for row in rows
        if row["method"] == "independent_block_circulant" and row["block_size"] == 4
    )
    negative = next(
        row
        for row in rows
        if row["method"] == "independent_plus_redundant_scale" and row["block_size"] == 4
    )
    quantized = next(
        row
        for row in rows
        if row["method"] == "quantized_independent_per_block"
        and row["block_size"] == 4
        and row["quant_bits"] == 3
    )
    assert base["normalized_mse"] == negative["normalized_mse"]
    assert base["relative_fro_error"] == negative["relative_fro_error"]
    assert quantized["parameter_bits"] == (
        quantized["quant_code_slots"] * quantized["quant_bits"]
        + quantized["scale_count"] * probe.SCALE_BITS
    )


def test_budget_capped_orders_use_identical_bits_and_respect_cap() -> None:
    examples, _ = probe.load_examples()
    rows: list[dict] = []
    probe.add_compensation_rows(examples[0], rows)
    capped = [row for row in rows if row["category"] == "budget_capped_compensation"]
    assert len(capped) == 6
    for row in capped:
        assert row["parameter_bits"] <= row["budget_cap_bits"]
        assert 0.0 < row["budget_utilization"] <= 1.0
    for method in ("sink", "lowrank", "sparse"):
        pair = [row for row in capped if f"_{method}_" in row["method"]]
        assert len(pair) == 2
        assert pair[0]["parameter_bits"] == pair[1]["parameter_bits"]
        assert pair[0]["compensation_amount"] == pair[1]["compensation_amount"]


def test_per_block_quantization_scale_handles_amplitude_heterogeneity() -> None:
    kernels = np.asarray(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float64,
    )
    global_q, global_scales = probe.quantize_kernels(kernels, 2, "global", 2)
    block_q, block_scales = probe.quantize_kernels(kernels, 2, "per_block", 2)
    assert np.linalg.norm(kernels - block_q) <= np.linalg.norm(kernels - global_q)
    assert global_scales == 1
    assert block_scales == kernels.shape[0]


def test_fp16_scale_underflow_decodes_group_to_zero_without_nan() -> None:
    kernels = np.full((4, 4), 1e-12, dtype=np.float64)
    quantized, scale_count = probe.quantize_kernels(kernels, 2, "per_block", 2)
    assert scale_count == 4
    assert np.all(np.isfinite(quantized))
    assert np.count_nonzero(quantized) == 0


def test_mark_pareto_rejects_dominated_point() -> None:
    rows = [
        {"key": "x", "category": "backbone", "parameter_bits": 10, "normalized_mse": 0.5},
        {"key": "x", "category": "backbone", "parameter_bits": 20, "normalized_mse": 0.4},
        {"key": "x", "category": "backbone", "parameter_bits": 30, "normalized_mse": 0.6},
    ]
    pareto = probe.mark_pareto(rows)
    assert rows[0] in pareto
    assert rows[1] in pareto
    assert rows[2] not in pareto
    assert rows[2]["pareto_nrmse"] is False


def test_probability_divergences_are_normalized_and_nonnegative() -> None:
    target = np.asarray([[0.7, 0.3], [0.0, 1.0]], dtype=np.float64)
    same = target * np.asarray([[1.01], [0.99]])
    shifted = np.asarray([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
    assert np.isclose(probe.mean_row_kl(target, same), 0.0, atol=1e-12)
    assert probe.mean_row_kl(target, shifted) >= 0.0
    assert probe.mean_row_js(target, shifted) >= 0.0
    rescaled_target = target * np.asarray([[3.0], [0.2]])
    rescaled_shifted = shifted * np.asarray([[0.4], [5.0]])
    assert np.isclose(
        probe.mean_row_js(rescaled_target, rescaled_shifted),
        probe.mean_row_js(target, shifted),
        atol=1e-12,
    )
    metrics = probe.evaluate_candidate(target, target)
    assert metrics["mean_row_kl"] >= 0.0
    assert metrics["mean_row_js"] >= 0.0


def test_zero_payload_baseline_is_a_valid_pareto_endpoint() -> None:
    fields = probe.parameter_fields(4, 0, 0)
    assert fields["parameter_fraction_of_dense_fp16"] == 0.0
    assert fields["compression_ratio_vs_dense_fp16"] is None
    rows = [
        {"key": "x", "category": "baseline", "parameter_bits": 0, "normalized_mse": 0.9},
        {"key": "x", "category": "backbone", "parameter_bits": 10, "normalized_mse": 0.5},
    ]
    pareto = probe.mark_pareto(rows)
    assert rows[0] in pareto
    assert rows[1] in pareto


def test_source_clustering_and_anchor_selection_metadata_are_preserved() -> None:
    examples, _ = probe.load_examples()
    assert len({example["source_input_id"] for example in examples}) == 4
    anchors = probe.load_true_v_anchors(examples)
    assert anchors
    required = {
        "original_scope",
        "selection_metric",
        "selection_candidates",
        "target_selected_oracle",
        "is_attention_only_rollout",
    }
    assert all(required.issubset(anchor) for anchor in anchors)
    assert any(anchor["selection_candidates"] > 1 for anchor in anchors)
