import unittest

import torch

from ar_video_residual_width_core import (
    enumerate_summary_tiles,
    event_mask_from_indices,
    indices_from_event_mask,
    jaccard_similarity,
    normalized_tail_objective,
    select_top_indices,
    selection_budget,
    tail_energy_by_head,
)


class ResidualWidthCoreTest(unittest.TestCase):
    def test_tile_enumeration_and_mask_round_trip(self) -> None:
        candidates = enumerate_summary_tiles(((2, 4),), spatial_tokens=5, tile_size=3)
        self.assertEqual(
            [(item.frame, item.start, item.end) for item in candidates],
            [(2, 0, 3), (2, 3, 5), (4, 0, 3), (4, 3, 5)],
        )
        mask = event_mask_from_indices(candidates, [1, 2], 5, 5, "cpu")
        self.assertEqual(indices_from_event_mask(candidates, mask), [1, 2])

    def test_budget_uses_ceiling_and_top_selection_is_stable(self) -> None:
        self.assertEqual(selection_budget(9, 0.1), 1)
        self.assertEqual(selection_budget(10, 0.11), 2)
        scores = torch.tensor([1.0, 2.0, 2.0, -1.0])
        self.assertEqual(select_top_indices(scores, 3), [1, 2, 0])

    def test_tail_energy_matches_known_singular_values(self) -> None:
        defect = torch.zeros(2, 1, 2)
        defect[:, 0] = torch.diag(torch.tensor([3.0, 2.0]))
        self.assertAlmostEqual(float(tail_energy_by_head(defect, 1)[0]), 4.0)
        reference = torch.ones_like(defect)
        self.assertAlmostEqual(
            float(normalized_tail_objective(defect, reference, 1)), 1.0
        )

    def test_jaccard_handles_empty_and_partial_overlap(self) -> None:
        self.assertEqual(jaccard_similarity([], []), 1.0)
        self.assertAlmostEqual(jaccard_similarity([1, 2], [2, 3]), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
