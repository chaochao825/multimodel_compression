#!/usr/bin/env python3
"""Bounded read-only scan for existing video model artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path


DEFAULT_KEYWORDS = (
    "wan2",
    "wan2.2",
    "video",
    "qwen",
    "qwen3-vl",
    "qwen2.5-omni",
    "cambrian",
    "llava",
    "cosmos",
    "videomme",
    "mvbench",
    "hunyuan",
    "cogvideo",
)

SKIP_DIR_NAMES = {
    ".cache",
    ".config",
    ".git",
    ".huggingface",
    ".local",
    ".ssh",
    "__pycache__",
    "wandb",
}

SKIP_FILE_GLOBS = (
    "*.lock",
    "*.metadata",
    "*credential*",
    "*password*",
    "*secret*",
    "*token*",
    ".netrc",
    "id_*",
)

CONFIG_NAMES = {
    "config.json",
    "model_index.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
}

INTERESTING_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".sh",
    ".md",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=250)
    parser.add_argument("--jsonl", action="store_true")
    return parser.parse_args()


def within_depth(path: Path, root: Path, max_depth: int) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) <= max_depth


def is_sensitive_like(path: Path) -> bool:
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_FILE_GLOBS)


def match_text(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    # Do not match against absolute user/home prefixes such as /home/wangmeiqi.
    return "/".join(part.lower() for part in rel.parts)


def is_interesting(path: Path, root: Path, keywords: tuple[str, ...]) -> bool:
    name = path.name.lower()
    rel = match_text(path, root)
    if any(k in rel for k in keywords):
        return True
    if name in CONFIG_NAMES:
        return True
    return path.suffix.lower() in INTERESTING_SUFFIXES and any(k in name for k in keywords)


def describe(path: Path) -> dict[str, object]:
    item: dict[str, object] = {"path": str(path), "kind": "dir" if path.is_dir() else "file"}
    try:
        stat = path.stat()
        item["size"] = stat.st_size
        item["mtime"] = int(stat.st_mtime)
    except OSError as exc:
        item["stat_error"] = str(exc)
    if path.is_dir():
        try:
            children = list(path.iterdir())
            item["child_count"] = len(children)
            item["sample_children"] = [child.name for child in children[:12]]
        except OSError as exc:
            item["list_error"] = str(exc)
    elif path.name.lower() in CONFIG_NAMES or path.suffix.lower() in {".json", ".yaml", ".yml", ".md"}:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            item["head"] = text[:1200]
        except OSError as exc:
            item["read_error"] = str(exc)
    return item


def main() -> int:
    args = parse_args()
    keywords = tuple(k.lower() for k in (args.keyword or DEFAULT_KEYWORDS))
    emitted = 0
    seen: set[str] = set()

    for root_text in args.root:
        root = Path(root_text).expanduser()
        if not root.exists():
            item = {"path": str(root), "kind": "missing"}
            print(json.dumps(item, ensure_ascii=False) if args.jsonl else item)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if name.lower() not in SKIP_DIR_NAMES and not name.startswith(".")
            ]
            if not within_depth(current, root, args.max_depth):
                dirnames[:] = []
                continue
            candidates = [current] + [
                current / name for name in filenames if not is_sensitive_like(current / name)
            ]
            for path in candidates:
                key = str(path)
                if key in seen or not is_interesting(path, root, keywords):
                    continue
                seen.add(key)
                item = describe(path)
                print(json.dumps(item, ensure_ascii=False) if args.jsonl else item)
                emitted += 1
                if emitted >= args.max_items:
                    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
