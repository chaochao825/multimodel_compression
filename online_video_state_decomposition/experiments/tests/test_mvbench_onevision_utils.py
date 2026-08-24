from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

torch = pytest.importorskip("torch")

from mvbench_onevision_utils import (  # noqa: E402
    expected_feature_state_bytes,
    pool_and_recent_positions,
    video_features_with_newline,
)


def test_pool_and_recent_positions_are_nested() -> None:
    pool, positions = pool_and_recent_positions(
        100,
        sampled_frames=32,
        feature_pool_frames=16,
        frame_budget=8,
    )
    assert len(pool) == 16
    assert positions == list(range(8, 16))


def test_onevision_rank384_state_matches_target_ratio() -> None:
    accounting = expected_feature_state_bytes(
        frames=16,
        tokens_per_frame=196,
        hidden_size=3584,
        rank=384,
        residual_tokens_per_frame=4,
    )
    assert accounting["compressed_state_bytes"] == 2_867_328
    assert math.isclose(accounting["compression_ratio"], 7.8396500156)


def test_video_newline_is_appended_once() -> None:
    class Inner:
        image_newline = torch.tensor([7.0, 8.0])

    class Model:
        model = Inner()

    features = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    result = video_features_with_newline(Model(), features)
    assert result.shape == (7, 2)
    assert torch.equal(result[-1], torch.tensor([7.0, 8.0]))
