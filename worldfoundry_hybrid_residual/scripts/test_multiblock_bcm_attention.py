#!/usr/bin/env python3
"""Unit tests for the calibration-only multi-block BCM attention probe."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("multi-block BCM tests require torch") from error

from probe_multiblock_bcm_attention import (
    ModelSpec,
    bucket_count,
    delta_bucket_indices,
    fit_attention_model,
    index_capture_rows,
    parameter_count,
    query_group_indices,
    validate_split_ids,
)


class MultiBlockBCMAttentionTests(unittest.TestCase):
    def test_physical_and_modulo_bucket_indices_are_distinct_at_boundary(self) -> None:
        shape = (1, 1, 4)
        queries = torch.tensor([0, 3])
        physical, physical_wrap = delta_bucket_indices(
            queries, shape, (1, 1, 1), periodic=False
        )
        modulo, modulo_wrap = delta_bucket_indices(
            queries, shape, (1, 1, 1), periodic=True
        )
        self.assertFalse(physical_wrap.any())
        self.assertTrue(modulo_wrap.any())
        self.assertNotEqual(int(physical[0, 3]), int(physical[1, 0]))
        self.assertEqual(int(modulo[0, 3]), int(modulo[1, 2]))

    def test_query_group_grid_partitions_physical_queries(self) -> None:
        groups = query_group_indices(
            torch.tensor([0, 3, 4, 7]), (1, 2, 4), (1, 2, 2)
        )
        self.assertEqual(groups.tolist(), [0, 1, 2, 3])

    def test_fit_is_nonnegative_normalized_and_recovers_bucket_target(self) -> None:
        shape = (1, 1, 4)
        queries = torch.arange(4)
        spec = ModelSpec("global_coarse_bccb", (1, 1, 1))
        bucket_index, _ = delta_bucket_indices(queries, shape, (1, 1, 1), True)
        generator = torch.tensor([0.55, 0.25, 0.15, 0.05])
        target = generator.take(bucket_index)
        target = target / target.sum(dim=1, keepdim=True)
        fitted = fit_attention_model([target, target], queries, shape, spec)
        prediction = fitted.predict()
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.allclose(prediction.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.allclose(prediction, target, atol=1e-6))

    def test_query_conditioned_and_hierarchical_fits_stay_normalized(self) -> None:
        shape = (1, 2, 4)
        queries = torch.arange(8)
        target = torch.eye(8) * 0.8 + torch.ones(8, 8) * 0.025
        specs = (
            ModelSpec("query_block_multi_bcm", (1, 1, 1), (1, 2, 4)),
            ModelSpec(
                "coarse_tile_local_residual",
                (1, 1, 2),
                (1, 2, 2),
                (1, 1, 1),
                (1, 1, 1),
            ),
        )
        for spec in specs:
            prediction = fit_attention_model([target], queries, shape, spec).predict()
            self.assertTrue(torch.all(prediction >= 0))
            self.assertTrue(torch.allclose(prediction.sum(dim=1), torch.ones(8)))
        multi_prediction = fit_attention_model(
            [target], queries, shape, specs[0]
        ).predict()
        self.assertLess(float((multi_prediction - target).norm()), 1e-6)

    def test_cross_replay_split_rejects_ids_and_shared_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_split_ids(("sample-a",), ("sample-a",))

        replay = Path("/synthetic/shared.pt")
        index = Path("/synthetic/capture_index.csv")
        rows = [
            {"sample_id": "cal", "layer": "0", "sampling_step": "0", "branch": "cond", "path": str(replay)},
            {"sample_id": "test", "layer": "0", "sampling_step": "0", "branch": "cond", "path": str(replay)},
        ]
        with patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "shared replay path"):
                index_capture_rows(
                    rows,
                    {"calibration": ("cal",), "heldout": ("test",)},
                    index,
                )

    def test_parameter_count_includes_heads_groups_and_residual_tables(self) -> None:
        shape = (2, 3, 4)
        coarse = (1, 2, 2)
        grid = (1, 2, 2)
        global_spec = ModelSpec("global_coarse_bccb", coarse)
        multi_spec = ModelSpec("query_block_multi_bcm", coarse, grid)
        hier_spec = ModelSpec(
            "coarse_tile_local_residual", coarse, grid, (1, 1, 1), (1, 1, 1)
        )
        periodic_buckets = bucket_count(shape, coarse, True)
        physical_buckets = bucket_count(shape, coarse, False)
        self.assertEqual(parameter_count(global_spec, shape, 3), periodic_buckets * 3)
        self.assertEqual(parameter_count(multi_spec, shape, 3), physical_buckets * 4 * 3)
        expected_per_head = physical_buckets + 4 * physical_buckets + 4 * 27
        self.assertEqual(parameter_count(hier_spec, shape, 3), expected_per_head * 3)


if __name__ == "__main__":
    unittest.main()
