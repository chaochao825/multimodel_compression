from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_wan_rcm_baseline as evaluator  # noqa: E402


class WanRcmEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "wan_rcm_baseline_exp047_v1.json"
        )
        cls.config = evaluator.load_config(config_path)

    def test_config_keeps_eight_registered_dimensions(self) -> None:
        self.assertEqual(
            tuple(self.config["metrics"]["vbench_dimensions"]), evaluator.DIMENSIONS
        )

    def test_evaluation_filename_round_trip(self) -> None:
        identity = evaluator.parse_evaluation_name(
            "/tmp/rcm4_p03_s2026082702.mp4"
        )
        self.assertEqual(identity, ("rcm4", 3, 2026082702))
        with self.assertRaisesRegex(ValueError, "unexpected VBench"):
            evaluator.parse_evaluation_name("rcm4_unknown.mp4")

    def test_vbench_summary_uses_method_means_and_teacher_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            result_dir = work_dir / "vbench_results"
            result_dir.mkdir()
            for dimension_index, dimension in enumerate(evaluator.DIMENSIONS):
                details = []
                teacher = 1.0 + dimension_index
                for method, scale in (("teacher20", 1.0), ("native4", 0.8), ("rcm4", 0.9)):
                    for prompt_index in range(4):
                        for seed in (2026082701, 2026082702):
                            details.append(
                                {
                                    "video_path": (
                                        f"/tmp/{method}_p{prompt_index:02d}_s{seed}.mp4"
                                    ),
                                    "video_results": teacher * scale,
                                }
                            )
                payload = {dimension: [0.0, details]}
                (result_dir / f"exp047_{dimension}_eval_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            summary = evaluator.summarize_vbench(work_dir)
            self.assertAlmostEqual(
                summary["aggregate"]["rcm4"]["mean_teacher_normalized"], 0.9
            )
            self.assertTrue(summary["aggregate"]["rcm4"]["passes_mean_0_90"])
            self.assertAlmostEqual(
                summary["dimensions"]["dynamic_degree"]["teacher_normalized"]["native4"],
                0.8,
            )

    def test_diversity_gate_uses_three_registered_distances(self) -> None:
        rows = []
        for method, scale in (("teacher20", 1.0), ("native4", 0.6), ("rcm4", 0.8)):
            for prompt_index in range(4):
                rows.append(
                    {
                        "method": method,
                        "prompt_index": prompt_index,
                        "video_embedding_distance": 0.1 * scale,
                        "frame_lpips": 0.2 * scale,
                        "frame_l1": 0.3 * scale,
                    }
                )
        summary = evaluator.summarize_diversity(rows)
        self.assertAlmostEqual(summary["rcm4"]["minimum_prompt_ratio"], 0.8)
        self.assertTrue(summary["rcm4"]["passes_three_of_four_0_70"])
        self.assertFalse(summary["native4"]["passes_three_of_four_0_70"])


if __name__ == "__main__":
    unittest.main()
