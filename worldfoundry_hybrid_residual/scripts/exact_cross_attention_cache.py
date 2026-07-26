#!/usr/bin/env python3
"""Generation-local exact text K/V cache wiring for Wan cross-attention probes."""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from typing import Any


class ExactCrossAttentionKVCache:
    """Temporarily connect Wan's existing per-block cross-attention cache API.

    The active slot must be selected before every model call. Slots are intended
    to represent CFG branches within one generation and must never be reused
    across prompts or weight changes.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.modules = [block.cross_attn for block in model.blocks]
        self.original_forwards: list[Any] = []
        self.slots: dict[Hashable, list[dict[str, Any]]] = {}
        self.active_slot: Hashable | None = None
        self.installed = False
        self.hits = 0
        self.misses = 0

    def _new_slot(self) -> list[dict[str, Any]]:
        return [{"is_init": False} for _ in self.modules]

    def activate(self, slot: Hashable) -> None:
        if not self.installed:
            raise RuntimeError("cache controller must be installed before activation")
        if slot not in self.slots:
            self.slots[slot] = self._new_slot()
        self.active_slot = slot

    def deactivate(self) -> None:
        self.active_slot = None

    def install(self) -> None:
        if self.installed:
            return
        self.original_forwards = [module.forward for module in self.modules]
        for module, original, index in zip(
            self.modules, self.original_forwards, range(len(self.modules))
        ):
            if "crossattn_cache" not in inspect.signature(original).parameters:
                raise TypeError(f"block {index} cross-attention has no cache argument")

            def cached_forward(*args: Any, _original: Any = original, _index: int = index, **kwargs: Any) -> Any:
                if self.active_slot is None:
                    return _original(*args, **kwargs)
                if kwargs.get("crossattn_cache") is not None:
                    raise RuntimeError("two cross-attention cache owners were supplied")
                cache = self.slots[self.active_slot][_index]
                if cache["is_init"]:
                    self.hits += 1
                else:
                    self.misses += 1
                kwargs["crossattn_cache"] = cache
                return _original(*args, **kwargs)

            module.forward = cached_forward
        self.installed = True

    def restore(self) -> None:
        if self.installed:
            for module, original in zip(self.modules, self.original_forwards):
                module.forward = original
        self.installed = False
        self.active_slot = None
        self.original_forwards.clear()

    def clear(self) -> None:
        self.active_slot = None
        self.slots.clear()

    def stats(self) -> dict[str, int]:
        initialized = sum(
            int(cache["is_init"])
            for slot in self.slots.values()
            for cache in slot
        )
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in self.slots.values()
            for cache in slot
            for name in ("k", "v")
            if (tensor := cache.get(name)) is not None
            and hasattr(tensor, "numel")
            and hasattr(tensor, "element_size")
        )
        return {
            "crossattn_cache_hits": self.hits,
            "crossattn_cache_misses": self.misses,
            "crossattn_cache_initialized_blocks": initialized,
            "crossattn_cache_bytes": tensor_bytes,
        }

    def __enter__(self) -> "ExactCrossAttentionKVCache":
        self.install()
        return self

    def __exit__(self, *_: object) -> None:
        self.restore()
        self.clear()
