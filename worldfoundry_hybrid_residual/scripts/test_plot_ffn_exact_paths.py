from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_ffn_exact_paths import PATH_ORDER, build_plot_rows, harmonic_mean


def row(case: str, path: str, layer: int, *, speedup: float, exact: bool) -> dict[str, str]:
    return {
        "case": case,
        "path": path,
        "layer": str(layer),
        "status": "ok",
        "median_speedup": str(speedup),
        "p95_speedup": str(speedup - 0.01),
        "amortized_speedup": str(speedup - 0.1),
        "setup_ms": "10",
        "relative_l2": "0" if exact else "0.002",
        "bitwise_equal": str(exact),
    }


class FFNExactPlotTests(unittest.TestCase):
    def test_harmonic_mean(self) -> None:
        self.assertAlmostEqual(harmonic_mean([1.0, 2.0]), 4.0 / 3.0)
        self.assertTrue(math.isnan(harmonic_mean([])))

    def test_build_plot_rows_requires_and_aggregates_all_cells(self) -> None:
        rows = []
        for case in ("F17", "F81"):
            for path in PATH_ORDER:
                for layer in (0, 14, 29):
                    rows.append(
                        row(
                            case,
                            path,
                            layer,
                            speedup=1.02 + layer / 10000,
                            exact=path == "cuda_graph_eager_static",
                        )
                    )
        output = build_plot_rows(rows)
        self.assertEqual(len(output), 8)
        graph_f17 = next(
            item
            for item in output
            if item["case"] == "F17" and item["path"] == "cuda_graph_eager_static"
        )
        self.assertEqual(graph_f17["layers"], 3)
        self.assertTrue(graph_f17["bitwise_exact_all"])
        self.assertEqual(graph_f17["relative_l2_max"], 0.0)

    def test_missing_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing result group"):
            build_plot_rows([])


if __name__ == "__main__":
    unittest.main()
