#!/usr/bin/env python3
"""Print basic Python/ML environment facts without shell quoting tricks."""

from __future__ import annotations

import importlib
import json


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT_ERROR: {exc.__class__.__name__}: {exc}"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    info: dict[str, object] = {
        "torch": module_version("torch"),
        "transformers": module_version("transformers"),
        "diffusers": module_version("diffusers"),
        "safetensors": module_version("safetensors"),
        "numpy": module_version("numpy"),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        info["cuda_error"] = f"{exc.__class__.__name__}: {exc}"
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
