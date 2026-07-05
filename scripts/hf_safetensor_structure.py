#!/usr/bin/env python3
"""Summarize attention-related safetensor keys without loading full tensors."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--max-keys", type=int, default=200)
    return parser.parse_args()


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    weight_map: dict[str, str] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        weight_map[f"__file__::{path.name}"] = path.name
    return weight_map


def safe_shape(path: Path, key: str) -> list[int] | None:
    try:
        from safetensors import safe_open
    except Exception:
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as sf:
            if key.startswith("__file__::"):
                return None
            return list(sf.get_slice(key).get_shape())
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    weight_map = load_index(model_dir)
    default_patterns = ("attn", "attention", "qkv", ".q.", ".k.", ".v.")
    patterns = tuple(p.lower() for p in (args.pattern or default_patterns))
    matched = [k for k in weight_map if any(p in k.lower() for p in patterns)]
    shard_counts = Counter(weight_map[k] for k in matched)
    layer_counts: dict[str, int] = defaultdict(int)
    layer_re = re.compile(r"(?:layers|blocks)\.(\d+)")
    for key in matched:
        m = layer_re.search(key)
        if m:
            layer_counts[m.group(1)] += 1
    rows = []
    for key in matched[: args.max_keys]:
        shard = weight_map[key]
        shape = safe_shape(model_dir / shard, key)
        rows.append({"key": key, "shard": shard, "shape": shape})
    out = {
        "model_dir": str(model_dir),
        "total_keys": len(weight_map),
        "matched_keys": len(matched),
        "matched_shards": dict(shard_counts),
        "layer_key_counts": dict(sorted(layer_counts.items(), key=lambda kv: int(kv[0]))),
        "sample": rows,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
