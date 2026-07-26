#!/usr/bin/env python3
"""Regression tests for leakage-safe geometry generalization inputs."""

from __future__ import annotations

import unittest

from capture_wan_qkv_trajectory import parse_sample_plan
from summarize_geometry_generalization import index_unique_rows, validate_head_groups


def make_row(head: int = 0) -> dict[str, str]:
    return {
        "layer": "0",
        "sampling_step": "0",
        "timestep": "1000",
        "branch": "cond",
        "mask": "s3",
        "head": str(head),
    }


def test_sample_plan_can_hold_noise_fixed_across_prompts() -> None:
    plan = parse_sample_plan("0:41,1:41,0:42,2:42")
    assert plan == ((0, 41), (1, 41), (0, 42), (2, 42))


def test_duplicate_policy_key_is_rejected() -> None:
    row = make_row()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "duplicate policy keys"):
        index_unique_rows([row, dict(row)], "sample")


def test_incomplete_head_group_is_rejected() -> None:
    rows = [make_row(head) for head in range(11)]
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "incomplete head groups"):
        validate_head_groups(rows, "sample", expected_heads=12)
