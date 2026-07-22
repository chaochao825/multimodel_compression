from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from build_mvbench_query_split import build_split, validate_split
from build_mvbench_query_confirmation import (
    build_confirmation,
    build_payload,
    validate_confirmation,
)
from aggregate_mvbench_llava import load_rows, summarize_overall
from compare_mvbench_query_memory_runs import paired_deltas, paired_summary
from report_mvbench_llava_anchor import (
    comparison_is_significant,
    transfer_gate,
)
from mvbench_utils import (
    MVBenchSample,
    clip_question_prompt,
    load_mvbench_samples_by_indices,
)
from query_memory import (
    LearnedFeatureRanker,
    apply_calibrated_feature_policy,
    apply_learned_feature_policy,
    apply_query_policy,
    diverse_recent_pool_indices,
    fit_learned_feature_ranker,
    query_select_indices,
    reservoir_recent_pool_indices,
)

if importlib.util.find_spec("torch") is not None:
    from mvbench_llava_anchor import select_indices
else:
    select_indices = None


class QueryMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.vectors = rng.normal(size=(32, 12))
        self.query = rng.normal(size=12)

    def test_two_tier_pools_are_bounded_and_keep_recent(self) -> None:
        reservoir = reservoir_recent_pool_indices(
            32,
            capacity=16,
            recent_capacity=3,
            seed=9,
        )
        diverse = diverse_recent_pool_indices(
            self.vectors,
            capacity=16,
            recent_capacity=3,
        )
        for pool in (reservoir, diverse):
            self.assertEqual(len(pool), 16)
            self.assertEqual(pool[-3:], [29, 30, 31])
            self.assertEqual(pool, sorted(set(pool)))

    def test_mmr_can_trade_relevance_for_diversity(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        selected = query_select_indices(
            vectors,
            np.asarray([1.0, 0.0]),
            pool_indices=[0, 1, 2, 3],
            budget=2,
            recent_anchors=1,
            diversity_weight=2.0,
            temporal_weight=0.0,
        )
        self.assertEqual(selected, [2, 3])

    def test_policy_accounting_distinguishes_online_upper_bound(self) -> None:
        bounded = apply_query_policy(
            "diverse_recent_query_mmr",
            self.vectors,
            self.query,
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
            diversity_weight=0.25,
            temporal_weight=0.1,
            seed=1,
        )
        offline = apply_query_policy(
            "offline_full_query_mmr",
            self.vectors,
            self.query,
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
            diversity_weight=0.25,
            temporal_weight=0.1,
            seed=1,
        )
        self.assertTrue(bounded.online_bounded)
        self.assertFalse(offline.online_bounded)
        self.assertEqual(bounded.persistent_vectors, 16)
        self.assertEqual(offline.persistent_vectors, 32)
        self.assertEqual(len(bounded.selected_indices), 8)
        self.assertGreater(
            offline.total_state_bytes,
            bounded.total_state_bytes,
        )

    def test_calibrated_policy_is_bounded_and_option_order_invariant(
        self,
    ) -> None:
        rng = np.random.default_rng(11)
        candidates = rng.normal(size=(4, self.vectors.shape[1]))
        result = apply_calibrated_feature_policy(
            self.vectors,
            self.query,
            candidates,
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
            diversity_weight=0.25,
            temporal_weight=0.1,
            option_weight=0.5,
            recency_weight=0.1,
            novelty_weight=0.25,
        )
        permuted = apply_calibrated_feature_policy(
            self.vectors,
            self.query,
            candidates[[2, 0, 3, 1]],
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
            diversity_weight=0.25,
            temporal_weight=0.1,
            option_weight=0.5,
            recency_weight=0.1,
            novelty_weight=0.25,
        )
        self.assertTrue(result.online_bounded)
        self.assertTrue(result.option_aware)
        self.assertEqual(result.ranker_parameter_bytes, 12)
        self.assertEqual(result.persistent_vectors, 16)
        self.assertEqual(len(result.selected_indices), 8)
        self.assertEqual(result.selected_indices, permuted.selected_indices)
        self.assertGreater(result.metadata_bytes, 16 * 4 + 16)

    def test_learned_ranker_is_small_bounded_and_option_invariant(
        self,
    ) -> None:
        ranker = LearnedFeatureRanker(
            feature_mean=(0.0, 0.0, 0.0, 0.0),
            feature_scale=(1.0, 1.0, 1.0, 1.0),
            coefficients=(1.0, 0.5, 0.1, 0.2),
            ridge=1.0,
            training_frames=1600,
        )
        rng = np.random.default_rng(17)
        candidates = rng.normal(size=(4, self.vectors.shape[1]))
        result = apply_learned_feature_policy(
            self.vectors,
            self.query,
            candidates,
            ranker,
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
        )
        permuted = apply_learned_feature_policy(
            self.vectors,
            self.query,
            candidates[[3, 1, 0, 2]],
            ranker,
            evidence_budget=8,
            pool_capacity=16,
            recent_anchors=3,
        )
        self.assertTrue(result.online_bounded)
        self.assertTrue(result.option_aware)
        self.assertEqual(result.ranker_parameter_bytes, 48)
        self.assertEqual(result.selected_indices, permuted.selected_indices)

    def test_ridge_ranker_learns_a_positive_feature_direction(self) -> None:
        features = np.asarray(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        )
        targets = np.asarray([-2.0, -1.0, 1.0, 2.0])
        ranker = fit_learned_feature_ranker(
            [features],
            [targets],
            ridge=1.0,
        )
        self.assertGreater(ranker.coefficients[0], 0.0)
        self.assertEqual(ranker.training_frames, 4)

    @unittest.skipIf(select_indices is None, "torch is not installed locally")
    def test_llava_accepts_external_selection_indices(self) -> None:
        selected = select_indices(
            "diverse_recent_query_mmr",
            total_frames=100,
            frame_budget=4,
            recent_frames=2,
            external_indices=[2, 10, 80, 99],
        )
        self.assertEqual(selected, [2, 10, 80, 99])
        with self.assertRaises(ValueError):
            select_indices(
                "diverse_recent_query_mmr",
                total_frames=100,
                frame_budget=4,
                recent_frames=2,
                external_indices=[2, 10, 99],
            )

    def test_llava_summary_uses_configured_reference(self) -> None:
        rows = [
            {
                "task": "a",
                "policy": "exact_recent",
                "samples": 2,
                "parsed": 2,
                "correct": 1,
                "accuracy": 0.5,
                "mean_inference_seconds": 1.0,
            },
            {
                "task": "a",
                "policy": "query",
                "samples": 2,
                "parsed": 2,
                "correct": 2,
                "accuracy": 1.0,
                "mean_inference_seconds": 1.5,
            },
        ]
        summary = summarize_overall(rows, reference="exact_recent")
        query = next(row for row in summary if row["policy"] == "query")
        self.assertEqual(query["reference_policy"], "exact_recent")
        self.assertAlmostEqual(query["macro_gain_vs_reference"], 0.5)
        self.assertAlmostEqual(query["mean_policy_seconds"], 1.5)

    def test_llava_loader_recovers_end_to_end_seconds_from_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "logs").mkdir()
            checkpoint = {
                "configuration_fingerprint": "demo",
                "rows": [
                    {
                        "sample_id": "sample_a",
                        "policy": "exact_recent",
                    }
                ],
            }
            (run_dir / "checkpoints" / "sample_a.json").write_text(
                json.dumps(checkpoint),
                encoding="utf-8",
            )
            event = {
                "event": "policy_ok",
                "sample": "sample_a",
                "policy": "exact_recent",
                "seconds": 4.25,
            }
            (run_dir / "logs" / "shard_0.log").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            rows, fingerprints = load_rows(run_dir)
            self.assertEqual(fingerprints, ["demo"])
            self.assertAlmostEqual(float(rows[0]["policy_seconds"]), 4.25)

    def test_question_prompt_does_not_include_answers(self) -> None:
        sample = MVBenchSample(
            task="demo",
            index=0,
            video_path=Path("demo.mp4"),
            question="What changed?",
            candidates=("red", "green"),
            answer="green",
        )
        prompt = clip_question_prompt(sample)
        self.assertEqual(prompt, "Question: What changed?")
        self.assertNotIn("red", prompt)
        self.assertNotIn("green", prompt)

    def test_explicit_index_loader_preserves_requested_membership(self) -> None:
        manifest = {
            "root": {"demo": "videos"},
            "meta": {
                "demo": [
                    {
                        "video": f"{index}.mp4",
                        "question": f"q{index}",
                        "candidates": ["yes", "no"],
                        "answer": "yes",
                    }
                    for index in range(5)
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            samples = load_mvbench_samples_by_indices(
                root,
                indices_by_task={"demo": [4, 1]},
            )
            self.assertEqual([sample.index for sample in samples], [4, 1])

    def test_split_builder_is_disjoint_and_excludes_prior_samples(self) -> None:
        excluded = {"demo": {0, 1, 2}}
        split = build_split(
            task_sizes={"demo": 20},
            excluded=excluded,
            calibration_per_task=4,
            evaluation_per_task=6,
            seed=5,
        )
        validate_split(split, excluded)
        calibration = set(split["calibration"]["demo"])
        evaluation = set(split["evaluation"]["demo"])
        self.assertFalse(calibration & evaluation)
        self.assertFalse((calibration | evaluation) & excluded["demo"])

    def test_confirmation_uses_only_untouched_reserve(self) -> None:
        source = {
            "tasks": ["demo"],
            "calibration": {"demo": [0, 1]},
            "evaluation": {"demo": [2, 3]},
            "reserve": {"demo": list(range(4, 20))},
        }
        confirmation = build_confirmation(
            source,
            evaluation_per_task=6,
            seed=13,
        )
        validate_confirmation(source, confirmation)
        self.assertEqual(confirmation["calibration"]["demo"], [])
        self.assertEqual(len(confirmation["evaluation"]["demo"]), 6)
        self.assertFalse(
            set(confirmation["evaluation"]["demo"]) & {0, 1, 2, 3}
        )

    def test_confirmation_payload_records_frozen_stage_and_split_ancestry(
        self,
    ) -> None:
        source = {
            "analysis_stage": "posthoc_reserve_confirmation",
            "source_split": {"name": "base.json", "sha256": "base-hash"},
            "tasks": ["demo"],
            "task_sizes": {"demo": 10},
            "calibration": {"demo": []},
            "evaluation": {"demo": [0, 1]},
            "reserve": {"demo": [2, 3, 4, 5]},
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "confirmation.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            payload = build_payload(
                source,
                source_path=source_path,
                evaluation_per_task=4,
                seed=17,
                primary_policy="learned_recent_query_topk",
                analysis_stage="frozen_independent_replication",
            )
        self.assertEqual(
            payload["analysis_stage"], "frozen_independent_replication"
        )
        self.assertEqual(payload["evaluation"]["demo"], [2, 3, 4, 5])
        self.assertEqual(payload["reserve"]["demo"], [])
        self.assertEqual(
            payload["source_split"]["analysis_stage"],
            "posthoc_reserve_confirmation",
        )
        self.assertEqual(
            payload["source_split"]["parent"]["sha256"], "base-hash"
        )

    def test_cross_split_comparison_keeps_only_paired_samples(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "task": "task_a",
                "policy": "exact_recent",
                "correct": "0",
            },
            {
                "sample_id": "a",
                "task": "task_a",
                "policy": "candidate",
                "correct": "1",
            },
            {
                "sample_id": "b",
                "task": "task_b",
                "policy": "exact_recent",
                "correct": "1",
            },
            {
                "sample_id": "b",
                "task": "task_b",
                "policy": "candidate",
                "correct": "0",
            },
            {
                "sample_id": "unpaired",
                "task": "task_c",
                "policy": "candidate",
                "correct": "1",
            },
        ]
        deltas, tasks = paired_deltas(
            rows,
            policy="candidate",
            reference="exact_recent",
        )
        np.testing.assert_array_equal(deltas, np.asarray([1.0, -1.0]))
        self.assertEqual(tasks, ["task_a", "task_b"])

    def test_cross_split_summary_counts_decision_flips(self) -> None:
        summary = paired_summary(
            np.asarray([1.0, 0.0, -1.0, 1.0]),
            bootstrap_samples=200,
            seed=3,
        )
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["better"], 2)
        self.assertEqual(summary["worse"], 1)
        self.assertEqual(summary["tied"], 1)
        self.assertAlmostEqual(float(summary["gain"]), 0.25)

    def test_llava_transfer_gate_requires_two_positive_tasks(self) -> None:
        self.assertFalse(
            transfer_gate(
                {"task_a": 0.1, "task_b": 0.0, "task_c": 0.0},
                0.02,
            )
        )
        self.assertTrue(
            transfer_gate(
                {"task_a": 0.1, "task_b": 0.05, "task_c": -0.025},
                0.02,
            )
        )
        self.assertFalse(
            transfer_gate(
                {"task_a": 0.1, "task_b": 0.05, "task_c": 0.0},
                0.0,
            )
        )

    def test_llava_significance_uses_actual_mcnemar_p_value(self) -> None:
        self.assertTrue(
            comparison_is_significant({"mcnemar_exact_p": "0.0386"})
        )
        self.assertFalse(
            comparison_is_significant({"mcnemar_exact_p": "0.0923"})
        )


if __name__ == "__main__":
    unittest.main()
