#!/usr/bin/env python3
"""List Qwen3-VL classes available in the current transformers install."""

from __future__ import annotations

import importlib
import json


def main() -> int:
    mods = [
        "transformers.models.qwen3_vl",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers",
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
