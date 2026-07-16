from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from probes.mvbench_utils import (
    MVBenchSample,
    hybrid_frame_indices,
    load_mvbench_samples,
    parse_choice_output,
    recent_frame_indices,
    shard_samples,
    uniform_frame_indices,
)
from probes.aggregate_mvbench_tasks import (
    exact_binomial_two_sided,
    paired_comparisons,
)
from probes.aggregate_mvbench_llava import paired_policy_comparisons
from probes.task_memory import (
    METHODS,
    make_memory,
    normalize_rows,
    state_accounting,
)


class MVBenchUtilsTest(unittest.TestCase):
    def test_frame_policies_are_bounded_and_ordered(self) -> None:
        self.assertEqual(uniform_frame_indices(10, 4), [1, 3, 6, 8])
        self.assertEqual(recent_frame_indices(10, 4), [6, 7, 8, 9])
        hybrid = hybrid_frame_indices(10, 4, recent_count=2)
        self.assertEqual(hybrid[-2:], [8, 9])
        self.assertEqual(len(hybrid), 4)
        self.assertEqual(hybrid, sorted(set(hybrid)))

    def test_choice_parser_handles_letters_and_text(self) -> None:
        candidates = ("red ball", "green cube", "blue sphere")
        self.assertEqual(parse_choice_output("Option B.", candidates), 1)
        self.assertEqual(parse_choice_output("green cube", candidates), 1)
        self.assertIsNone(parse_choice_output("uncertain", candidates))

    def test_manifest_loading_and_sharding_are_deterministic(self) -> None:
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
                    for index in range(8)
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test.json").write_text(
                __import__("json").dumps(manifest),
                encoding="utf-8",
            )
            left = load_mvbench_samples(
                root,
                tasks=["demo"],
                samples_per_task=4,
                selection_seed=9,
            )
            right = load_mvbench_samples(
                root,
                tasks=["demo"],
                samples_per_task=4,
                selection_seed=9,
            )
            self.assertEqual(left, right)
            shards = [
                shard_samples(left, shard_index=index, shard_count=2)
                for index in range(2)
            ]
            self.assertEqual(sum(map(len, shards)), len(left))

    def test_answer_index_requires_exact_candidate(self) -> None:
        sample = MVBenchSample(
            task="demo",
            index=0,
            video_path=Path("video.mp4"),
            question="q",
            candidates=("yes", "no"),
            answer="Yes.",
        )
        self.assertEqual(sample.answer_index, 0)


class TaskMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)
        self.hidden_dim = 24
        self.values = normalize_rows(
            self.rng.normal(size=(64, self.hidden_dim))
        )

    def test_all_methods_produce_finite_option_scores(self) -> None:
        text = normalize_rows(self.rng.normal(size=(4, self.hidden_dim)))
        for method in METHODS:
            memory = make_memory(
                method,
                capacity=8,
                hidden_dim=self.hidden_dim,
                seed=7,
                instant_capacity=3,
                oja_lr=0.5,
                prototype_scale=0.5,
                slot_min_lr=0.05,
                slot_replace_similarity=0.75,
            )
            for value in self.values:
                memory.update(value[None])
            scores = memory.score(text, temperature=10.0)
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(np.all(np.isfinite(scores)), method)

    def test_oja_prefers_a_query_in_the_learned_subspace(self) -> None:
        basis = np.linalg.qr(
            self.rng.normal(size=(self.hidden_dim, 3))
        )[0]
        stream = normalize_rows(
            self.rng.normal(size=(200, 3)) @ basis.T
        )
        memory = make_memory(
            "oja_subspace",
            capacity=4,
            hidden_dim=self.hidden_dim,
            seed=0,
            instant_capacity=0,
            oja_lr=0.8,
            prototype_scale=0.75,
            slot_min_lr=0.05,
            slot_replace_similarity=0.75,
        )
        for value in stream:
            memory.update(value[None])
        in_span = normalize_rows(
            self.rng.normal(size=(1, 3)) @ basis.T
        )
        orthogonal = self.rng.normal(size=(1, self.hidden_dim))
        orthogonal -= (orthogonal @ basis) @ basis.T
        orthogonal = normalize_rows(orthogonal)
        scores = memory.score(
            np.concatenate((in_span, orthogonal)),
            temperature=20.0,
        )
        self.assertGreater(scores[0], scores[1])

    def test_payload_is_matched_across_methods(self) -> None:
        payloads = {
            state_accounting(
                method,
                capacity=8,
                hidden_dim=768,
                storage_bits=16,
                instant_capacity=3,
            ).payload_bytes
            for method in METHODS
        }
        self.assertEqual(payloads, {8 * 768 * 2})

    def test_paired_comparison_uses_matched_samples(self) -> None:
        rows = []
        recent = [0, 0, 1, 1]
        proposed = [1, 0, 1, 0]
        for index, (baseline, candidate) in enumerate(
            zip(recent, proposed, strict=True)
        ):
            rows.extend(
                [
                    {
                        "sample_id": f"s{index}",
                        "method": "recent_window",
                        "capacity": 8,
                        "correct": baseline,
                    },
                    {
                        "sample_id": f"s{index}",
                        "method": "instant_oja",
                        "capacity": 8,
                        "correct": candidate,
                    },
                ]
            )
        comparison = paired_comparisons(
            rows,
            seed=1,
            bootstrap_samples=200,
        )[0]
        self.assertEqual(comparison["paired_samples"], 4)
        self.assertEqual(comparison["better_samples"], 1)
        self.assertEqual(comparison["worse_samples"], 1)
        self.assertAlmostEqual(comparison["accuracy_gain_vs_recent"], 0.0)
        self.assertAlmostEqual(exact_binomial_two_sided(0, 4), 0.125)

    def test_llava_policy_comparison_is_paired(self) -> None:
        rows = []
        for index, (uniform, hybrid) in enumerate(
            zip([0, 1, 0], [1, 1, 0], strict=True)
        ):
            rows.extend(
                [
                    {
                        "sample_id": f"s{index}",
                        "policy": "uniform",
                        "correct": uniform,
                    },
                    {
                        "sample_id": f"s{index}",
                        "policy": "hybrid",
                        "correct": hybrid,
                    },
                ]
            )
        comparison = paired_policy_comparisons(
            rows,
            bootstrap_samples=200,
        )[0]
        self.assertEqual(comparison["paired_samples"], 3)
        self.assertEqual(comparison["better_samples"], 1)
        self.assertEqual(comparison["worse_samples"], 0)
        self.assertAlmostEqual(comparison["accuracy_gain"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
