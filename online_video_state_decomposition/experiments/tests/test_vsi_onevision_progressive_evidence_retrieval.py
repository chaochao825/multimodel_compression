from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from mvbench_onevision_utils import (  # noqa: E402
    first_token_logits_from_features,
    first_token_logits_from_variable_video_tokens,
)
from probe_vsi_onevision_progressive_evidence_retrieval import (  # noqa: E402
    evidence_combinations,
    question_conditioned_frame_scores,
    spatial_pool_frames,
)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(video_token_index=99)
        self.embedding = torch.nn.Embedding(128, 4)
        self.model = SimpleNamespace(image_newline=torch.nn.Parameter(torch.ones(4)))
        self.readout = torch.nn.Linear(4, 7, bias=False)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert not use_cache and return_dict and logits_to_keep == 1
        pooled = (
            inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        ).sum(dim=1)
        return SimpleNamespace(logits=self.readout(pooled).unsqueeze(1))


def test_variable_video_path_matches_fixed_path_at_full_token_count() -> None:
    torch.manual_seed(3)
    model = _DummyModel()
    input_ids = torch.tensor([[1, 99, 99, 99, 2]])
    attention_mask = torch.ones_like(input_ids)
    features = torch.randn(1, 2, 4)
    fixed = first_token_logits_from_features(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        features=features,
    )
    variable = first_token_logits_from_variable_video_tokens(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        video_tokens=features.reshape(-1, 4),
    )
    assert torch.equal(fixed, variable)


def test_spatial_pool_frames_preserves_block_means() -> None:
    features = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    pooled = spatial_pool_frames(features, pooled_side=2)
    assert pooled.shape == (1, 4, 1)
    assert torch.allclose(
        pooled.flatten(),
        torch.tensor([2.5, 4.5, 10.5, 12.5]),
    )


def test_evidence_combinations_are_regular_and_bounded() -> None:
    assert evidence_combinations(4, 2) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    with pytest.raises(ValueError):
        evidence_combinations(4, 5)


def test_question_conditioned_score_uses_post_video_text() -> None:
    model = _DummyModel()
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[2, 0] = 1.0
    scores = question_conditioned_frame_scores(
        model=model,
        input_ids=torch.tensor([[1, 99, 99, 2]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        pooled_frames=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0]],
            ]
        ),
    )
    assert scores.tolist() == [1.0, 0.0]
