from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_wan_ffn_exact_paths import aggregate_paths, evaluate_rows


def fixture_row(
    path: str,
    *,
    median: float,
    p95: float,
    setup: float,
    bitwise: bool,
) -> dict[str, str]:
    amortization_calls = 40
    return {
        "case": "F17",
        "layer": "0",
        "path": path,
        "status": "ok",
        "latency_ms_median": str(median),
        "latency_ms_p95": str(p95),
        "setup_ms": str(setup),
        "amortization_calls": str(amortization_calls),
        "amortized_latency_ms": str(median + setup / amortization_calls),
        "bitwise_equal": str(bitwise),
        "max_abs": "0.0" if bitwise else "0.0078125",
        "relative_l2": "0.0" if bitwise else "1e-5",
        "incremental_peak_allocated_bytes": str(128 * 2**20),
    }


class WanFFNExactSummaryTests(unittest.TestCase):
    def test_exact_fast_amortized_candidate_passes(self) -> None:
        rows = [
            fixture_row("eager", median=10.0, p95=12.0, setup=0.0, bitwise=True),
            fixture_row(
                "cuda_graph_eager_static",
                median=8.0,
                p95=9.0,
                setup=10.0,
                bitwise=True,
            ),
        ]
        evaluated = evaluate_rows(
            rows,
            min_median_speedup=1.03,
            min_p95_speedup=1.0,
            min_amortized_speedup=1.0,
            max_incremental_memory_gib=4.0,
        )
        self.assertEqual(evaluated[0]["decision"], "REFERENCE")
        self.assertEqual(evaluated[1]["decision"], "GO")
        self.assertGreater(evaluated[1]["median_speedup"], 1.2)
        self.assertEqual(evaluated[1]["break_even_calls_vs_eager"], 5.0)
        aggregate = aggregate_paths(evaluated)
        self.assertEqual(aggregate[0]["decision"], "GO")
        self.assertTrue(aggregate[0]["bitwise_exact_all"])

    def test_non_bitwise_compile_path_is_no_go(self) -> None:
        rows = [
            fixture_row("eager", median=10.0, p95=12.0, setup=0.0, bitwise=True),
            fixture_row(
                "compile_default",
                median=7.0,
                p95=8.0,
                setup=100.0,
                bitwise=False,
            ),
        ]
        evaluated = evaluate_rows(
            rows,
            min_median_speedup=1.03,
            min_p95_speedup=1.0,
            min_amortized_speedup=1.0,
            max_incremental_memory_gib=4.0,
        )
        candidate = evaluated[1]
        self.assertEqual(candidate["decision"], "NO-GO")
        self.assertIn("not_bitwise_exact", candidate["decision_reason"])

    def test_source_contract_uses_real_complete_ffn_and_shared_lock(self) -> None:
        benchmark = (SCRIPT_DIR / "benchmark_wan_ffn_exact_paths.py").read_text(
            encoding="utf-8"
        )
        runner = (SCRIPT_DIR / "run_phase2_ffn_exact_v1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("WanModel.from_pretrained", benchmark)
        self.assertIn("function=lambda: ffn(value)", benchmark)
        self.assertIn("static_output = ffn(static_value)", benchmark)
        self.assertIn("no GEMM-epilogue fusion is claimed", benchmark)
        self.assertIn("GPU=\"${GPU:-2}\"", runner)
        self.assertIn("GPU_CANDIDATES=\"${GPU_CANDIDATES:-2,3}\"", runner)
        self.assertIn("selected first-idle GPU", runner)
        self.assertIn("/tmp/codex_phase2_strict_h200_v1.lock", runner)
        self.assertIn("flock 9", runner)


if __name__ == "__main__":
    unittest.main()
