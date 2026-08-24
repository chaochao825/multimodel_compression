from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBES = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBES))

from mvbench_reader_quotient_static_prior import choose_samples  # noqa: E402


class Sample:
    def __init__(self, sample_id: str, task: str) -> None:
        self.sample_id = sample_id
        self.task = task


def test_choose_samples_excludes_previous_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        Sample("a0", "a"),
        Sample("a1", "a"),
        Sample("a2", "a"),
        Sample("b0", "b"),
        Sample("b1", "b"),
        Sample("b2", "b"),
    ]
    monkeypatch.setattr(
        "mvbench_reader_quotient_static_prior.load_manifest_samples",
        lambda *_args, **_kwargs: candidates,
    )
    selected = choose_samples(
        dataset_root=Path("."),
        selection_manifest={},
        tasks=["a", "b"],
        manifest_samples_per_task=3,
        take_per_task=2,
        selection_seed=1,
        excluded={"a0", "b1"},
    )
    assert [sample.sample_id for sample in selected] == ["a1", "a2", "b0", "b2"]
