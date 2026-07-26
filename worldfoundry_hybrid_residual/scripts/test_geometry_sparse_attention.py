#!/usr/bin/env python3
"""Unit tests for content-independent THW geometry masks."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("geometry mask tests require torch") from error
from probe_geometry_sparse_attention import (
    GeometrySpec,
    geometry_mask,
    grid_from_metadata,
    infer_grid,
)


def test_infer_grid_for_wan_f17_and_f81() -> None:
    assert infer_grid(17, 7800, 30, 52) == (5, 30, 52)
    assert infer_grid(81, 32760, 30, 52) == (21, 30, 52)


def test_replay_grid_metadata_overrides_ambiguous_cli_geometry() -> None:
    metadata = {"frame_num": 81, "grid_size": [21, 30, 52]}
    assert grid_from_metadata(metadata, 32760, 28, 60) == (21, 30, 52)


def test_geometry_mask_is_content_independent_and_contains_self() -> None:
    shape = (5, 4, 6)
    queries = torch.tensor([0, 17, 119])
    spec = GeometrySpec("test", 1, 0, True, 2)
    first, first_info = geometry_mask(queries, shape, spec, 2, 3)
    second, second_info = geometry_mask(queries, shape, spec, 2, 3)
    assert torch.equal(first, second)
    assert first_info == second_info
    assert first[torch.arange(queries.numel()), queries].all()


def test_larger_spatial_tile_window_never_reduces_coverage() -> None:
    shape = (5, 4, 6)
    queries = torch.arange(0, 120, 11)
    sparse, _ = geometry_mask(
        queries, shape, GeometrySpec("sparse", 0, 0, False, 0), 2, 3
    )
    dense, _ = geometry_mask(
        queries, shape, GeometrySpec("dense", 1, 0, False, 0), 2, 3
    )
    assert torch.logical_or(~sparse, dense).all()
