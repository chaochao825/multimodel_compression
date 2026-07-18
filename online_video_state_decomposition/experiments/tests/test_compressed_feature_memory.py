from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from aggregate_compressed_feature_memory import (
    clopper_pearson_upper,
    paired_selectors_by_variant,
    paired_vs_full,
    summarize_variants,
    task_deltas_vs_full,
)


def row(
    sample_id: str,
    *,
    variant: str,
    correct: int,
    state_bytes: int,
    selector: str = "exact_recent",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task": "demo",
        "policy": f"{selector}__{variant}",
        "selection_policy": selector,
        "memory_variant": variant,
        "parsed": 1,
        "correct": correct,
        "predicted_index": correct,
        "selection_state_proxy_bytes": state_bytes,
        "native_feature_state_bytes": state_bytes - 16,
        "codec_parameter_bytes": 100,
        "codec_rank": 4,
        "residual_tokens_per_frame": 0 if variant == "full" else 1,
        "feature_state_compression_ratio": 4.0,
        "total_state_compression_ratio": 3.5,
        "pool_reconstruction_relative_error": (
            0.0 if variant == "full" else 0.2
        ),
        "selected_reconstruction_relative_error": (
            0.0 if variant == "full" else 0.25
        ),
        "compression_seconds": 0.01,
        "reconstruction_seconds": 0.02,
        "inference_seconds": 0.08,
        "policy_seconds": 1.0,
    }


class CompressedFeatureMemoryAggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            row("a", variant="full", correct=1, state_bytes=1000),
            row("b", variant="full", correct=1, state_bytes=1000),
            row("a", variant="pca_r4_s1", correct=1, state_bytes=250),
            row("b", variant="pca_r4_s1", correct=0, state_bytes=250),
        ]

    def test_variant_summary_preserves_accuracy_and_state(self) -> None:
        summary = summarize_variants(self.rows)
        compressed = next(
            value
            for value in summary
            if value["memory_variant"] == "pca_r4_s1"
        )
        self.assertAlmostEqual(float(compressed["accuracy"]), 0.5)
        self.assertEqual(
            float(compressed["mean_total_state_bytes"]),
            250.0,
        )
        self.assertEqual(
            float(compressed["cold_start_total_state_bytes"]),
            350.0,
        )
        self.assertAlmostEqual(
            float(compressed["mean_selected_reconstruction_error"]),
            0.25,
        )

    def test_routed_summary_is_invariant_to_sample_order(self) -> None:
        rows = [
            row(
                "a",
                variant="pca_r4_route_grid2_s4",
                correct=1,
                state_bytes=250,
            ),
            row(
                "b",
                variant="pca_r4_route_grid2_s4",
                correct=0,
                state_bytes=250,
            ),
        ]
        rows[0].update(
            {
                "sparse_residual_tokens_per_frame": 0,
                "sparse_residual_token_capacity": 64,
                "realized_sparse_residual_tokens": 0,
                "pool_frame_indices": list(range(16)),
            }
        )
        rows[1].update(
            {
                "sparse_residual_tokens_per_frame": 4,
                "sparse_residual_token_capacity": 64,
                "realized_sparse_residual_tokens": 64,
                "pool_frame_indices": list(range(16)),
            }
        )
        forward = summarize_variants(rows)[0]
        reverse = summarize_variants(list(reversed(rows)))[0]
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["sparse_residual_tokens_per_frame"], -1)
        self.assertEqual(forward["sparse_residual_token_capacity"], 64)
        self.assertEqual(
            forward["mean_realized_sparse_residual_tokens"],
            32.0,
        )
        self.assertEqual(
            forward["mean_realized_sparse_tokens_per_frame"],
            2.0,
        )

    def test_paired_comparison_uses_full_variant_reference(self) -> None:
        paired = paired_vs_full(self.rows, seed=3)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["memory_variant"], "pca_r4_s1")
        self.assertAlmostEqual(float(paired[0]["accuracy_gain"]), -0.5)
        self.assertEqual(int(paired[0]["better_samples"]), 0)
        self.assertEqual(int(paired[0]["worse_samples"]), 1)
        self.assertAlmostEqual(
            float(paired[0]["prediction_agreement_rate"]),
            0.5,
        )
        self.assertEqual(int(paired[0]["noninferior_at_margin"]), 0)

    def test_exact_upper_bound_does_not_promote_tiny_smoke(self) -> None:
        self.assertAlmostEqual(
            clopper_pearson_upper(0, 5, alpha=0.05),
            1.0 - 0.05 ** (1.0 / 5),
        )
        self.assertGreater(
            clopper_pearson_upper(0, 5, alpha=0.05),
            0.02,
        )
        self.assertLess(
            clopper_pearson_upper(0, 200, alpha=0.05),
            0.02,
        )

    def test_selector_gain_is_paired_within_memory_variant(self) -> None:
        rows = [
            row(
                "a",
                variant="pca_r4_s1",
                correct=0,
                state_bytes=250,
            ),
            row(
                "b",
                variant="pca_r4_s1",
                correct=1,
                state_bytes=250,
            ),
            row(
                "a",
                variant="pca_r4_s1",
                correct=1,
                state_bytes=250,
                selector="learned_recent_query_topk",
            ),
            row(
                "b",
                variant="pca_r4_s1",
                correct=1,
                state_bytes=250,
                selector="learned_recent_query_topk",
            ),
        ]

        paired = paired_selectors_by_variant(rows, seed=3)

        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["memory_variant"], "pca_r4_s1")
        self.assertAlmostEqual(float(paired[0]["accuracy_gain"]), 0.5)
        self.assertEqual(int(paired[0]["better_samples"]), 1)
        self.assertEqual(int(paired[0]["worse_samples"]), 0)

    def test_task_delta_uses_matching_policy_full_cache(self) -> None:
        task_rows = [
            {
                "task": "demo",
                "selection_policy": "exact_recent",
                "memory_variant": "full",
                "samples": 5,
                "accuracy": 0.8,
            },
            {
                "task": "demo",
                "selection_policy": "exact_recent",
                "memory_variant": "pca_r4_s0",
                "samples": 5,
                "accuracy": 0.6,
            },
        ]
        delta = task_deltas_vs_full(task_rows)
        self.assertEqual(len(delta), 1)
        self.assertAlmostEqual(
            float(delta[0]["accuracy_delta_vs_full"]),
            -0.2,
        )


if __name__ == "__main__":
    unittest.main()
