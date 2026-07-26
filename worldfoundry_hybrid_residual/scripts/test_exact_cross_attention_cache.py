#!/usr/bin/env python3
"""Unit tests for branch-local, block-local exact cross-attention cache wiring."""

from __future__ import annotations

from exact_cross_attention_cache import ExactCrossAttentionKVCache


class FakeCrossAttention:
    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self, value: int, crossattn_cache: dict | None = None) -> int:
        self.forward_calls += 1
        if crossattn_cache is None:
            return value * 10
        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            crossattn_cache["k"] = value
            crossattn_cache["v"] = value
        return int(crossattn_cache["k"]) * 10


class FakeBlock:
    def __init__(self) -> None:
        self.cross_attn = FakeCrossAttention()


class FakeModel:
    def __init__(self, blocks: int = 2) -> None:
        self.blocks = [FakeBlock() for _ in range(blocks)]


def test_cache_is_disjoint_across_cfg_branches_and_blocks() -> None:
    model = FakeModel()
    controller = ExactCrossAttentionKVCache(model)
    controller.install()
    controller.activate("conditional")
    assert model.blocks[0].cross_attn.forward(3) == 30
    assert model.blocks[0].cross_attn.forward(99) == 30
    assert model.blocks[1].cross_attn.forward(4) == 40
    controller.activate("unconditional")
    assert model.blocks[0].cross_attn.forward(7) == 70
    assert controller.stats()["crossattn_cache_misses"] == 3
    assert controller.stats()["crossattn_cache_hits"] == 1
    controller.restore()


def test_restore_returns_uncached_forward() -> None:
    model = FakeModel(1)
    controller = ExactCrossAttentionKVCache(model)
    controller.install()
    controller.activate("conditional")
    assert model.blocks[0].cross_attn.forward(2) == 20
    controller.restore()
    assert model.blocks[0].cross_attn.forward(5) == 50
