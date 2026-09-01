from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_wan_rcm_successor_plan066 as analysis  # noqa: E402


class AmdahlProjectionTest(unittest.TestCase):
    def test_required_local_speedup_reaches_request_target(self) -> None:
        total = 9.637995031895116
        vae = 4.308299851138145
        local_speedup = analysis.required_local_speedup(total, vae, 1.10)
        self.assertAlmostEqual(local_speedup, 1.2552887875132517)
        self.assertAlmostEqual(
            analysis.projected_request_speedup(total, vae, local_speedup),
            1.10,
        )

    def test_measured_attention_ceiling_matches_exp054(self) -> None:
        total = 9.637995031895116
        attention = 3.20536530460231 * 0.5388086760322437
        speedup = analysis.projected_request_speedup(
            total,
            attention,
            1.586376845918112,
        )
        self.assertAlmostEqual(speedup, 1.0709347207832607)

    def test_unreachable_target_returns_infinite_requirement(self) -> None:
        requirement = analysis.required_local_speedup(
            9.637995031895116,
            0.2537410487420857,
            1.05,
        )
        self.assertTrue(math.isinf(requirement))


if __name__ == "__main__":
    unittest.main()
