from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from analyze_compressed_memory_losses import (  # noqa: E402
    format_optional,
    rank_auc,
    unique_in_order,
)


class AnalyzeCompressedMemoryLossesTest(unittest.TestCase):
    def test_auc_without_positive_examples_is_reported_as_na(self) -> None:
        value = rank_auc([0.1, 0.2, 0.3], [0, 0, 0])
        self.assertIsNone(value)
        self.assertEqual(format_optional(value), "N/A")

    def test_auc_with_both_classes_is_formatted(self) -> None:
        value = rank_auc([0.1, 0.9], [0, 1])
        self.assertEqual(value, 1.0)
        self.assertEqual(format_optional(value), "1.000")

    def test_identical_selectors_are_analyzed_once(self) -> None:
        self.assertEqual(unique_in_order(("learned", "learned")), ["learned"])


if __name__ == "__main__":
    unittest.main()
