from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from benchmark_official_streaming_kernels import (  # noqa: E402
    REQUIRED_PATHS,
    base_version_equals,
    higher_quantile,
    resolve_frames,
    retrieval_group_budget,
    summarize_distribution,
    validate_checkout,
    version_at_least,
)


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class OfficialStreamingKernelBenchmarkTest(unittest.TestCase):
    def test_method_defaults_exercise_cross_batch_state_and_oqm_topk(self) -> None:
        self.assertEqual(resolve_frames("streamingtom_ctr", 0), 64)
        self.assertEqual(resolve_frames("streamingtom_oqm_write", 0), 64)
        self.assertEqual(resolve_frames("streamingtom_oqm_select", 0), 256)
        self.assertEqual(resolve_frames("streamingtom_oqm_select", 300), 300)

    def test_oqm_budget_uses_official_ceiling_rule(self) -> None:
        self.assertEqual(retrieval_group_budget(12544, 50, 256), 251)
        self.assertEqual(retrieval_group_budget(12544, 50, 200), 200)

    def test_runtime_versions_do_not_accept_prefix_collisions(self) -> None:
        self.assertTrue(base_version_equals("2.5.1+cu121", "2.5.1"))
        self.assertFalse(base_version_equals("2.5.10", "2.5.1"))
        self.assertTrue(version_at_least("2.6.0+cu124", "2.1"))
        self.assertFalse(version_at_least("2.0.1", "2.1"))

    def test_higher_quantile_uses_observed_samples(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        self.assertEqual(higher_quantile(values, 0.50), 3.0)
        self.assertEqual(higher_quantile(values, 0.95), 100.0)
        summary = summarize_distribution(values)
        self.assertEqual(summary["p99"], 100.0)
        self.assertEqual(summary["count"], 5.0)

    def test_pinned_checkout_allows_only_python_cache_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "StreamingTOM"
            checkout.mkdir()
            _git("init", cwd=checkout)
            _git("config", "user.email", "benchmark@example.invalid", cwd=checkout)
            _git("config", "user.name", "Benchmark Test", cwd=checkout)
            for relative in REQUIRED_PATHS["streamingtom"]:
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n", encoding="utf-8")
            _git("add", ".", cwd=checkout)
            _git("commit", "-m", "fixture", cwd=checkout)
            commit = _git("rev-parse", "HEAD", cwd=checkout)
            cache = checkout / "streamingtom/__pycache__/main.cpython-310.pyc"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"cache")

            from benchmark_official_streaming_kernels import SOURCE_COMMITS

            original = SOURCE_COMMITS["streamingtom"]
            SOURCE_COMMITS["streamingtom"] = commit
            try:
                result = validate_checkout(checkout, source_name="streamingtom")
            finally:
                SOURCE_COMMITS["streamingtom"] = original

            self.assertTrue(result["code_clean"])
            self.assertEqual(result["commit"], commit)


if __name__ == "__main__":
    unittest.main()
