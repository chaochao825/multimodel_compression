from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

torch = pytest.importorskip("torch")

from feature_memory_codec import LowRankFeatureCodec  # noqa: E402
from mvbench_onevision_rank_support_allocation import (  # noqa: E402
    DEFAULT_ALLOCATIONS,
    allocation_variants,
    parse_allocations,
    prefix_codec,
)
from mvbench_onevision_utils import expected_feature_state_bytes  # noqa: E402


def test_default_allocation_payloads_are_within_frozen_budget() -> None:
    payloads = [
        expected_feature_state_bytes(
            frames=16,
            tokens_per_frame=196,
            hidden_size=3584,
            rank=rank,
            residual_tokens_per_frame=support,
        )["compressed_state_bytes"]
        for rank, support in DEFAULT_ALLOCATIONS
    ]
    assert payloads == [2_867_328, 2_865_504, 2_863_680, 2_861_856, 2_860_032]
    assert max(payloads) - min(payloads) == 7_296


def test_allocation_variants_omit_reader_support_for_zero_support() -> None:
    variants = allocation_variants(parse_allocations("384:4,456:0"))
    assert variants == (
        "euclidean_r384_s4",
        "fisher_r384_s4",
        "mixed_r384_s4",
        "euclidean_r456_s0",
    )


def test_prefix_codec_reuses_ordered_basis_prefix() -> None:
    codec = LowRankFeatureCodec(
        mean=torch.zeros(5),
        basis=torch.arange(30, dtype=torch.float32).reshape(5, 6),
    )
    prefix = prefix_codec(codec, 4)
    assert prefix.rank == 4
    assert torch.equal(prefix.basis, codec.basis[:, :4])
