#!/usr/bin/env python3
"""High-value invariants for the stronger train-free tail oracle screen."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_trainfree_tail_oracles import (
    aggregate_envelope_records,
    candidate_counts,
)
from probe_trainfree_tail_oracles import build_oracle_envelope
from trainfree_tail_oracle_core import (
    cluster_centroids,
    combine_shared_logits,
    covariance_query_work_ratio,
    covariance_tail_output,
    lowrank_covariance_products,
    oracle_mass_block_selection,
    output_leverage_importance,
    polynomial_tail_output,
    prepare_group_moments,
    stable_seed,
    token_coordinates,
    weighted_kmeans_assignments,
)


class TrainFreeTailOracleCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_oracle_mass_mask_is_tile_shared_and_budgeted(self) -> None:
        probabilities = torch.softmax(torch.randn(3, 10), dim=1)
        blocks, keys = oracle_mass_block_selection(probabilities, 2, 0.4)
        self.assertEqual(blocks.shape, (5,))
        self.assertEqual(keys.shape, (10,))
        self.assertEqual(int(blocks.sum()), 2)
        self.assertEqual(int(keys.sum()), 4)

    def test_singleton_tail_groups_recover_dense_attention(self) -> None:
        scores = torch.randn(4, 9)
        values = torch.randn(9, 5)
        selected = torch.zeros(9, dtype=torch.bool)
        selected[:2] = True
        tail_logits = scores[:, ~selected]
        tail_values = values[~selected]
        approximation, _ = combine_shared_logits(
            scores, values, selected, tail_logits, tail_values
        )
        reference = torch.softmax(scores, dim=1) @ values
        torch.testing.assert_close(approximation, reference, atol=1e-6, rtol=1e-6)

    def test_constant_residual_scores_make_taylor_exact(self) -> None:
        scores = torch.tensor(
            [[2.0, 0.25, 0.25, 0.25], [1.5, -0.5, -0.5, -0.5]]
        )
        values = torch.randn(4, 3)
        selected = torch.tensor([True, False, False, False])
        approximation, diagnostics = polynomial_tail_output(
            scores, values, selected, order=1, center_mode="mean"
        )
        reference = torch.softmax(scores, dim=1) @ values
        torch.testing.assert_close(approximation, reference, atol=1e-6, rtol=1e-6)
        self.assertEqual(diagnostics["negative_tail_weight_fraction"], 0.0)

    def test_full_group_rank_reconstructs_covariances(self) -> None:
        keys = torch.randn(4, 3)
        values = torch.randn(4, 2)
        moments = prepare_group_moments(keys, values, 4, 1, 4)
        reconstructed_k, reconstructed_vk = lowrank_covariance_products(moments, 4)
        centered_k = keys - keys.mean(0)
        centered_v = values - values.mean(0)
        expected_k = centered_k.T @ centered_k / 4
        expected_vk = centered_v.T @ centered_k / 4
        torch.testing.assert_close(reconstructed_k[0], expected_k, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            reconstructed_vk[0], expected_vk, atol=1e-5, rtol=1e-5
        )

    def test_singleton_centroid_groups_recover_dense_attention(self) -> None:
        queries = torch.randn(3, 4)
        keys = torch.randn(8, 4)
        values = torch.randn(8, 2)
        scores = queries @ keys.T
        selected_blocks = torch.zeros(2, dtype=torch.bool)
        selected_keys = torch.zeros(8, dtype=torch.bool)
        moments = prepare_group_moments(keys, values, 4, 4, 0)
        approximation, _ = covariance_tail_output(
            queries,
            scores,
            values,
            selected_blocks,
            selected_keys,
            moments,
            variant="centroid",
            rank=0,
            scale=1.0,
        )
        reference = torch.softmax(scores, dim=1) @ values
        torch.testing.assert_close(approximation, reference, atol=1e-6, rtol=1e-6)

    def test_value_leverage_is_positive_and_finite(self) -> None:
        probabilities = torch.softmax(torch.randn(5, 7), dim=1)
        values = torch.randn(7, 3)
        reference = probabilities @ values
        importance = output_leverage_importance(probabilities, values, reference)
        self.assertTrue(bool(torch.isfinite(importance).all()))
        self.assertTrue(bool((importance > 0).all()))

    def test_weighted_kmeans_is_deterministic(self) -> None:
        features = torch.randn(32, 6)
        weights = torch.rand(32)
        first = weighted_kmeans_assignments(
            features, weights, 5, iterations=3, fit_tokens=24, seed=91
        )
        second = weighted_kmeans_assignments(
            features, weights, 5, iterations=3, fit_tokens=24, seed=91
        )
        torch.testing.assert_close(first, second)
        counts, _, _ = cluster_centroids(features, features, first, 5)
        self.assertGreaterEqual(counts.numel(), 1)

    def test_grid_coordinates_follow_thw_product(self) -> None:
        coordinates = token_coordinates(
            (2, 3, 4), device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(coordinates.shape, (24, 3))
        torch.testing.assert_close(coordinates[0], torch.tensor([-1.0, -1.0, -1.0]))
        torch.testing.assert_close(coordinates[-1], torch.tensor([1.0, 1.0, 1.0]))

    def test_covariance_work_accounting(self) -> None:
        ratio = covariance_query_work_ratio(25, 100, 3, "lowrank_gaussian", 4)
        self.assertAlmostEqual(ratio, 0.4)
        self.assertTrue(math.isfinite(ratio))

    def test_stable_seed_does_not_use_process_hash_randomization(self) -> None:
        self.assertEqual(stable_seed("sample", 3, 0.25), stable_seed("sample", 3, 0.25))
        self.assertNotEqual(stable_seed("sample", 3, 0.25), stable_seed("sample", 4, 0.25))


class TrainFreeTailOracleEnvelopeTests(unittest.TestCase):
    def test_envelope_is_explicitly_posthoc(self) -> None:
        rows = []
        for family in ("a", "b"):
            for sample in ("s0", "s1"):
                for head in (0, 1):
                    for error in (0.02, 0.01):
                        rows.append(
                            {
                                "family": family,
                                "sample_id": sample,
                                "head": head,
                                "residual_sq": error * error,
                                "reference_sq": 1.0,
                                "aggregate_output_relative_l2": error,
                                "projected_query_work_ratio_mean": 0.4,
                                "selected_attention_mass_mean": 0.8,
                                "negative_tail_weight_fraction_mean": float("nan"),
                                "tail_score_range_mean": float("nan"),
                            }
                        )
        selected, summary = build_oracle_envelope(
            rows,
            {
                "oracle_aggregate_output_relative_l2": 0.005,
                "oracle_worst_record_output_relative_l2": 0.01,
            },
        )
        self.assertTrue(
            all(
                row["oracle_selection"] == "posthoc_per_record_minimum_output_error"
                for row in selected
            )
        )
        self.assertTrue(all(row["oracle_quality_gate"] == "FAIL" for row in summary))

    def test_analysis_aggregation_is_energy_weighted_without_reselection(self) -> None:
        rows = [
            {
                "family": "value_aware_coreset",
                "sample_id": "s0",
                "head": "3",
                "residual_sq": "0.0001",
                "reference_sq": "1.0",
                "aggregate_output_relative_l2": "0.01",
                "selected_attention_mass_mean": "0.8",
            },
            {
                "family": "value_aware_coreset",
                "sample_id": "s1",
                "head": "3",
                "residual_sq": "0.09",
                "reference_sq": "9.0",
                "aggregate_output_relative_l2": "0.1",
                "selected_attention_mass_mean": "0.9",
            },
        ]
        aggregated = aggregate_envelope_records(rows, ("family", "head"))
        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(
            float(aggregated[0]["aggregate_output_relative_l2"]),
            math.sqrt(0.0901 / 10.0),
        )
        self.assertEqual(float(aggregated[0]["record_error_max"]), 0.1)

    def test_candidate_counts_preserve_posthoc_choice_frequency(self) -> None:
        base = {
            "family": "value_aware_coreset",
            "variant": "value_aware_kv_thw",
            "density": "0.25",
            "landmarks": "128",
            "order": "0",
            "rank": "0",
            "components": "0",
            "restart": "0",
        }
        counts = candidate_counts([base, dict(base)])
        self.assertEqual(counts[0]["records"], 2)
        self.assertIn("m=128", str(counts[0]["configuration"]))


if __name__ == "__main__":
    unittest.main()
