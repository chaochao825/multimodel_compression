from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_attention_head_factorial import build_comparisons, pair_type, summarize_types


def metadata(label: str, prompt: int, seed: int, step: int, branch: str) -> dict[str, object]:
    return {
        "label": label,
        "sample_id": f"p{prompt}s{seed}",
        "prompt_index": prompt,
        "seed": seed,
        "sampling_step": step,
        "branch": branch,
        "layer": 0,
    }


def heads(offset: float = 0.0) -> dict[int, dict[str, float]]:
    return {
        0: {
            "actual_normalized_entropy_mean": 0.4 + offset,
            "geometry_mass_mean": 0.9 - offset,
            "actual_top64_mass_mean": 0.8 - offset,
            "actual_participation_support_fraction_mean": 0.1 + offset,
        },
        1: {
            "actual_normalized_entropy_mean": 0.9 + offset,
            "geometry_mass_mean": 0.3 - offset,
            "actual_top64_mass_mean": 0.2 - offset,
            "actual_participation_support_fraction_mean": 0.8 + offset,
        },
    }


class AttentionHeadFactorialTests(unittest.TestCase):
    def test_pair_type(self) -> None:
        base = metadata("a", 0, 10, 0, "cond")
        self.assertEqual(pair_type(base, metadata("b", 0, 11, 0, "cond")), "seed")
        self.assertEqual(pair_type(base, metadata("c", 1, 10, 0, "cond")), "prompt")
        self.assertEqual(pair_type(base, metadata("d", 1, 11, 0, "cond")), "mixed_sample")

    def test_factorial_comparison_counts(self) -> None:
        rows = []
        runs = {}
        for prompt in (0, 1):
            for seed in (10, 11):
                for step in (0, 9):
                    for branch in ("cond", "uncond"):
                        label = f"p{prompt}s{seed}t{step}{branch}"
                        rows.append(metadata(label, prompt, seed, step, branch))
                        runs[label] = heads()
        gates = {"entropy": 0.9, "geometry": 0.9, "class": 0.75, "jaccard": 0.5}
        comparisons = build_comparisons(rows, runs, gates)
        counts = {row["comparison_type"]: 0 for row in comparisons}
        for row in comparisons:
            counts[row["comparison_type"]] += 1
        self.assertEqual(counts["seed"], 8)
        self.assertEqual(counts["prompt"], 8)
        self.assertEqual(counts["mixed_sample"], 8)
        self.assertEqual(counts["step"], 8)
        self.assertEqual(counts["branch"], 8)
        self.assertTrue(all(row["router_class_pilot_go"] for row in comparisons))
        summary = summarize_types(comparisons)
        self.assertTrue(all(row["go_fraction"] == 1.0 for row in summary))

    def test_runner_contract(self) -> None:
        runner = (SCRIPT_DIR / "run_phase2_head_role_factorial_v1.sh").read_text(encoding="utf-8")
        self.assertIn("0:20260740,1:20260740,0:20260741,1:20260741", runner)
        self.assertIn("--capture-steps 0,9,19", runner)
        self.assertIn("--capture-layers 0,14,29", runner)
        self.assertIn("flock 9", runner)
        self.assertIn("flock -u 9", runner)
        self.assertIn("GPU_CANDIDATES=\"${GPU_CANDIDATES:-2,3}\"", runner)


if __name__ == "__main__":
    unittest.main()
