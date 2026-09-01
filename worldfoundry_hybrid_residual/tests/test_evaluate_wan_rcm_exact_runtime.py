from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_wan_rcm_exact_runtime as evaluator  # noqa: E402


class ExactRuntimeEvaluatorTest(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_fixture(self, root: Path) -> None:
        text_root = root / "text-screen" / "attempt02"
        self.write_json(text_root / "SUCCESS.json", {"status": "complete"})
        self.write_json(
            text_root / "manifest.json",
            {
                "result": {
                    "exact": True,
                    "advance": True,
                    "minimum_median_saving_seconds": 15.0,
                    "required_saving_seconds": 5.5,
                }
            },
        )
        request_times = {"teacher20": 40.0, "native4": 10.0, "rcm4": 10.0}
        denoiser_times = {"teacher20": 32.0, "native4": 6.4, "rcm4": 3.2}
        for method in evaluator.METHODS:
            f17_root = root / "f17-exact" / method
            self.write_json(f17_root / "SUCCESS.json", {"status": "complete"})
            self.write_json(
                f17_root / "manifest.json",
                {"result": {"video_equal": True, "network_calls_equal": True}},
            )
            rows = [
                {
                    "status": "ok",
                    "prompt_index": index,
                    "request_seconds": request_times[method],
                    "text_seconds": 0.1,
                    "denoiser_seconds": denoiser_times[method],
                    "vae_seconds": 4.0,
                    "cpu_transfer_seconds": 0.2,
                    "serialization_seconds": 1.7,
                    "network_forward_calls": evaluator.EXPECTED_FORWARD_CALLS[method],
                    "peak_reserved_mib": 40000.0,
                }
                for index in range(4)
            ]
            f81_root = root / "f81-resident" / method
            self.write_json(f81_root / "SUCCESS.json", {"status": "complete"})
            self.write_json(
                f81_root / "manifest.json",
                {
                    "environment": {"gpu_total_memory_bytes": 150_000_000_000},
                    "result": {
                        "rows": rows,
                        "summary": {
                            "median_request_seconds": request_times[method],
                            "median_text_seconds": 0.1,
                            "median_denoiser_seconds": denoiser_times[method],
                            "median_vae_seconds": 4.0,
                            "median_cpu_transfer_seconds": 0.2,
                            "median_serialization_seconds": 1.7,
                            "median_peak_reserved_mib": 40000.0,
                        },
                        "text_policy": {
                            "positive_cache_hits": 0,
                            "positive_model_calls": 4,
                        },
                    },
                },
            )

    def test_passing_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            result = evaluator.evaluate(root, 2.5)
            self.assertEqual(result["outcome"], "pass")
            self.assertEqual(result["primary"]["resident_warm_speedup"], 4.0)
            self.assertTrue(result["guards"]["positive_cache_guard"])


if __name__ == "__main__":
    unittest.main()
