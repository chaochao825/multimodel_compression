from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_group_compaction_geometry import (  # noqa: E402
    compact_group_tokens_and_offsets,
    diagnostic_decision,
    first_token_logits_from_positioned_video_tokens,
    fixed_length_group_tokens,
)


class FakeReader(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(video_token_index=99)
        self.model = SimpleNamespace(image_newline=torch.zeros(2))
        self.embedding = torch.nn.Embedding(128, 2)
        self.position_ids = None

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **kwargs: object) -> object:
        self.position_ids = kwargs["position_ids"]
        logits = torch.zeros(
            (1, inputs_embeds.shape[1], 4),
            dtype=inputs_embeds.dtype,
        )
        return SimpleNamespace(logits=logits)


def test_group_layouts_preserve_values_and_original_offsets() -> None:
    exact = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    means = exact.mean(dim=1)
    selected = torch.tensor([1])
    fixed = fixed_length_group_tokens(exact, means, selected)
    assert fixed.shape == (6, 2)
    assert torch.equal(fixed[2:4], exact[1])
    assert torch.equal(fixed[:2], means[0].expand(2, -1))

    compact, offsets = compact_group_tokens_and_offsets(
        exact,
        means,
        selected,
        representative_offset=0,
    )
    assert compact.shape == (4, 2)
    assert offsets.tolist() == [0, 2, 3, 4]
    assert torch.equal(compact[1:3], exact[1])


def test_positioned_readout_preserves_original_suffix_position() -> None:
    model = FakeReader()
    logits = first_token_logits_from_positioned_video_tokens(
        model=model,
        input_ids=torch.tensor([[10, 99, 99, 99, 20]]),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        video_tokens=torch.ones((1, 2)),
        video_position_offsets=torch.tensor([1]),
        full_video_token_count=2,
    )
    assert logits.shape == (4,)
    assert model.position_ids.tolist() == [[0, 2, 3, 4]]


def test_diagnostic_decision_prefers_position_recovery() -> None:
    failed = {
        "mismatch_count": 1,
        "harmful_count": 0,
        "candidate_kl_mean": 0.02,
        "candidate_kl_p95": 0.04,
    }
    passed = {
        "mismatch_count": 0,
        "harmful_count": 0,
        "candidate_kl_mean": 0.009,
        "candidate_kl_p95": 0.019,
    }
    summaries = {
        "compact_contiguous": {98: failed},
        "compact_original_position": {98: passed},
        "fixed_repeated": {98: passed},
    }
    assert diagnostic_decision(summaries) == ("POSITION_GEOMETRY_RECOVERY", 98)

    summaries["compact_original_position"] = {98: failed}
    assert diagnostic_decision(summaries) == ("FIXED_MULTIPLICITY_ONLY", 98)

    summaries["fixed_repeated"] = {98: failed}
    assert diagnostic_decision(summaries) == ("NO_GEOMETRY_RECOVERY", None)
