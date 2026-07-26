from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_ffn_fusion_ceiling import amdahl_speedup, analyze_cases


def benchmark_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for token_rows in (7800, 32760):
        for operation, latency in (
            ("ffn_full_eager", 1.0),
            ("ffn_up_linear_no_bias", 0.4),
            ("ffn_up_linear_bias_gelu_eager", 0.5),
            ("ffn_up_linear_triton_bias_gelu", 0.55),
            ("copy", 0.1),
        ):
            rows.append(
                {
                    "rows": str(token_rows),
                    "width": "8960",
                    "operation": operation,
                    "latency_ms_median": str(latency),
                    "status": "ok",
                    "relative_l2_vs_torch": (
                        "0.002" if operation == "ffn_up_linear_triton_bias_gelu" else ""
                    ),
                    "max_abs_vs_torch": (
                        "0.125" if operation == "ffn_up_linear_triton_bias_gelu" else ""
                    ),
                }
            )
    return rows


def profile_rows(total_ms: float = 100.0) -> list[dict[str, str]]:
    components = {
        "self_attention_core": 30.0,
        "elementwise_memory": 40.0,
        "linear_gemm": 20.0,
        "normalization": 5.0,
        "cross_attention_core": 4.0,
        "other": 1.0,
    }
    return [
        {
            "case": case,
            "component": component,
            "per_denoise_step_ms": str(value * total_ms / 100.0),
        }
        for case in ("F17", "F81")
        for component, value in components.items()
    ]


class FFNFusionCeilingTests(unittest.TestCase):
    def test_amdahl_handles_finite_and_removed_paths(self) -> None:
        self.assertAlmostEqual(amdahl_speedup(0.2, 2.0), 1.0 / 0.9)
        self.assertAlmostEqual(amdahl_speedup(0.2, math.inf), 1.25)

    def test_source_bound_ceiling_rejects_approximate_slower_triton(self) -> None:
        rows = analyze_cases(
            benchmark_rows(), profile_rows(), layers=30, cfg_branches=2
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["ideal_epilogue_local_speedup"], 1.0 / 0.9)
            self.assertAlmostEqual(
                row["ideal_intermediate_traffic_local_speedup"], 1.0 / 0.9
            )
            self.assertLess(row["standalone_triton_projected_local_speedup"], 1.0)
            self.assertFalse(row["triton_bitwise_exact_proxy"])
            self.assertEqual(row["standalone_triton_decision"], "NO-GO")
            self.assertAlmostEqual(row["estimated_ffn_share"], 0.6)

    def test_impossible_profile_share_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "estimated FFN share"):
            analyze_cases(
                benchmark_rows(), profile_rows(total_ms=10.0), layers=30, cfg_branches=2
            )


if __name__ == "__main__":
    unittest.main()
