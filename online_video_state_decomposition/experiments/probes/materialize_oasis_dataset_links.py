from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from prepare_oasis_streamingbench_subset import crc32_file, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize audited OASIS dataset paths as symbolic links"
    )
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"invalid POSIX relative path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*pure.parts)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
        raise ValueError(f"invalid OASIS mapping manifest: {path}")
    return payload


def build_link_plan(
    mapping_manifest: Path,
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = mapping_manifest.resolve()
    root = dataset_root.resolve()
    payload = _load_manifest(manifest_path)
    records = payload["videos"]
    if len(records) != payload.get("video_count"):
        raise ValueError("mapping manifest video_count does not match videos")
    plan = []
    seen_targets: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("mapping manifest contains a non-object video record")
        relative = _safe_relative_path(record.get("relative_path"))
        target = root / relative
        resolved_parent = target.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(root)
        except ValueError as error:
            raise ValueError(f"target escapes dataset root: {target}") from error
        if target in seen_targets:
            raise ValueError(f"duplicate materialization target: {target}")
        seen_targets.add(target)
        source = Path(str(record.get("source_path", ""))).resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"source video is missing or empty: {source}")
        if source.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"source size differs from manifest: {source}")
        expected_crc = str(record.get("crc32", "")).lower()
        if len(expected_crc) != 8 or crc32_file(source) != expected_crc:
            raise ValueError(f"source CRC32 differs from manifest: {source}")
        plan.append(
            {
                "video_id": record.get("video_id"),
                "relative_path": relative.as_posix(),
                "source_path": str(source),
                "target_path": str(target),
                "size_bytes": source.stat().st_size,
                "crc32": expected_crc,
            }
        )
    return payload, plan


def _target_matches(target: Path, source: Path) -> bool:
    if not target.is_symlink():
        return False
    try:
        return target.resolve(strict=True) == source
    except FileNotFoundError:
        return False


def materialize_links(
    *,
    mapping_manifest: Path,
    dataset_root: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    source_manifest, plan = build_link_plan(mapping_manifest, dataset_root)
    root = dataset_root.resolve()
    output_path = output_manifest.resolve()
    if output_path == mapping_manifest.resolve():
        raise ValueError("output manifest must not overwrite the mapping manifest")
    created = 0
    reused = 0
    for record in plan:
        source = Path(record["source_path"])
        target = Path(record["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not _target_matches(target, source):
                raise FileExistsError(
                    f"materialization target exists with different content: {target}"
                )
            reused += 1
            continue
        # Creating the final symlink directly is atomic and refuses races.
        os.symlink(source, target)
        created += 1
    result = {
        "format_version": 1,
        "mapping_manifest_path": str(mapping_manifest.resolve()),
        "mapping_manifest_sha256": sha256_file(mapping_manifest.resolve()),
        "source_subset_metadata_sha256": source_manifest.get(
            "subset_metadata_sha256"
        ),
        "dataset_root": str(root),
        "video_count": len(plan),
        "created_links": created,
        "reused_links": reused,
        "links": plan,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.name}.part-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, output_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_links(
        mapping_manifest=args.mapping_manifest,
        dataset_root=args.dataset_root,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
