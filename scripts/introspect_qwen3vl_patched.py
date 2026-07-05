#!/usr/bin/env python3
"""Import Qwen3-VL modeling after a narrow torch.register_fake compatibility shim."""

from __future__ import annotations

import importlib
import json


def install_register_fake_shim() -> None:
    import torch

    if hasattr(torch.library, "register_fake"):
        return

    def register_fake(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    torch.library.register_fake = register_fake  # type: ignore[attr-defined]


def main() -> int:
    install_register_fake_shim()
    mods = [
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    ]
    out = {}
    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
            names = [n for n in dir(mod) if "Qwen3" in n or "Vision" in n or "VL" in n]
            out[mod_name] = names
        except Exception as exc:
            out[mod_name] = f"{exc.__class__.__name__}: {exc}"
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
