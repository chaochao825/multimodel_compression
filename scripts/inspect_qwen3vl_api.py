#!/usr/bin/env python3
"""Inspect Qwen3-VL-MoE vision API signatures."""

from __future__ import annotations

import inspect

from introspect_qwen3vl_patched import install_register_fake_shim


def main() -> int:
    install_register_fake_shim()
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as m

    targets = [
        m.Qwen3VLMoeVisionModel,
        m.Qwen3VLMoeVisionBlock,
        m.Qwen3VLMoeVisionAttention,
        m.Qwen3VLMoeVisionPatchEmbed,
    ]
    for cls in targets:
        print(f"## {cls.__name__}")
        print("init:", inspect.signature(cls.__init__))
        print("forward:", inspect.signature(cls.forward))
        print(inspect.getsource(cls.forward)[:2400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
