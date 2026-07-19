from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import re
import uuid
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


QUESTION_ID_PATTERN = re.compile(
    r"^Real-Time Visual Understanding_sample_(\d+)_(\d+)$"
)
VIDEO_PATH_PATTERN = re.compile(
    r"^StreamingBench/Real-Time_Visual_Understanding/"
    r"sample_(\d+)/video\.mp4$"
)
REQUIRED_CSV_COLUMNS = {
    "question_id",
    "task_type",
    "question",
    "time_stamp",
    "answer",
    "options",
}
QUESTIONS_PER_VIDEO = 5
FORMAL_FIRST_VIDEO = 1
FORMAL_LAST_VIDEO = 50


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an audited OASIS StreamingBench RTU subset without copying "
            "videos."
        )
    )
    parser.add_argument(
        "--oasis-unified-json",
        "--source-metadata",
        dest="oasis_unified_json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--streamingbench-csv",
        "--metadata-csv",
        dest="streamingbench_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prepared-video-root",
        "--video-root",
        dest="prepared_video_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--upstream-manifest",
        type=Path,
        help="Optional audited manifest that produced the verified CSV/videos.",
    )
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument(
        "--mapping-manifest",
        "--manifest",
        dest="mapping_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--first-video", type=int, default=FORMAL_FIRST_VIDEO)
    parser.add_argument("--last-video", type=int, default=FORMAL_LAST_VIDEO)
    return parser.parse_args(argv)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def parse_question_id(question_id: str) -> tuple[int, int]:
    if not isinstance(question_id, str):
        raise ValueError("question_id must be a string")
    match = QUESTION_ID_PATTERN.fullmatch(question_id)
    if match is None:
        raise ValueError(f"unrecognized StreamingBench question_id: {question_id!r}")
    video_id, question_index = (int(part) for part in match.groups())
    if video_id < 1 or question_index < 1:
        raise ValueError(f"invalid StreamingBench question_id: {question_id!r}")
    return video_id, question_index


def normalize_csv_timestamp(value: str) -> tuple[str, int, bool]:
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be a string: {value!r}")
    original = value
    stripped = value.strip()
    parts = stripped.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    normalized = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return normalized, total_seconds, normalized != original


def parse_csv_options(value: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"CSV options must be a string: {value!r}")
    try:
        options = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid CSV options literal: {value!r}") from exc
    if not isinstance(options, list) or not options:
        raise ValueError("CSV options must decode to a non-empty list")
    if any(not isinstance(option, str) for option in options):
        raise ValueError("every CSV option must be a string")
    return options


def read_unified_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid OASIS unified JSON: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("OASIS unified JSON must contain a non-empty list")
    return payload


def read_verified_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"StreamingBench CSV is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("StreamingBench CSV contains no questions")
    return rows


def _validate_video_relative_path(value: Any, video_id: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample_{video_id} has no OASIS info.video_path")
    if "\\" in value:
        raise ValueError(
            f"sample_{video_id} info.video_path must use POSIX separators: {value!r}"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(
            f"sample_{video_id} info.video_path is not a safe relative path: "
            f"{value!r}"
        )
    match = VIDEO_PATH_PATTERN.fullmatch(path.as_posix())
    if match is None or int(match.group(1)) != video_id:
        raise ValueError(
            f"sample_{video_id} info.video_path is not the official OASIS "
            f"dataset_root path: {value!r}"
        )
    return path.as_posix()


def _index_csv_rows(
    rows: Iterable[Mapping[str, str]],
) -> tuple[dict[str, Mapping[str, str]], dict[int, list[str]]]:
    by_question: dict[str, Mapping[str, str]] = {}
    by_video: dict[int, list[str]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        missing = REQUIRED_CSV_COLUMNS - set(row)
        if missing:
            raise ValueError(
                f"StreamingBench CSV row {row_number} is missing fields: "
                f"{sorted(missing)}"
            )
        question_id = row["question_id"]
        video_id, _ = parse_question_id(question_id)
        if question_id in by_question:
            raise ValueError(
                f"duplicate question_id in StreamingBench CSV: {question_id}"
            )
        by_question[question_id] = row
        by_video[video_id].append(question_id)
    if not by_question:
        raise ValueError("StreamingBench CSV contains no questions")
    return by_question, by_video


def _index_unified_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], set[str]]:
    by_video: dict[int, Mapping[str, Any]] = {}
    seen_question_ids: set[str] = set()
    for item_number, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"OASIS item {item_number} is not an object")
        info = item.get("info")
        breakpoints = item.get("breakpoint")
        if not isinstance(info, Mapping):
            raise ValueError(f"OASIS item {item_number} has invalid info")
        if not isinstance(breakpoints, list) or not breakpoints:
            raise ValueError(f"OASIS item {item_number} has no breakpoints")

        item_video_ids: set[int] = set()
        for breakpoint_number, breakpoint in enumerate(breakpoints, start=1):
            if not isinstance(breakpoint, Mapping):
                raise ValueError(
                    f"OASIS item {item_number} breakpoint {breakpoint_number} "
                    "is not an object"
                )
            question_id = breakpoint.get("question_id")
            video_id, _ = parse_question_id(question_id)
            if question_id in seen_question_ids:
                raise ValueError(
                    f"duplicate question_id in OASIS metadata: {question_id}"
                )
            seen_question_ids.add(question_id)
            item_video_ids.add(video_id)

        if len(item_video_ids) != 1:
            raise ValueError(
                f"OASIS item {item_number} mixes video IDs: "
                f"{sorted(item_video_ids)}"
            )
        video_id = next(iter(item_video_ids))
        _validate_video_relative_path(info.get("video_path"), video_id)
        if video_id in by_video:
            raise ValueError(f"duplicate OASIS item for sample_{video_id}")
        by_video[video_id] = item
    return by_video, seen_question_ids


def _require_equal(
    question_id: str,
    field: str,
    oasis_value: Any,
    csv_value: Any,
) -> None:
    if oasis_value != csv_value:
        raise ValueError(
            f"{question_id}: {field} mismatch between OASIS metadata and "
            f"StreamingBench CSV; OASIS={oasis_value!r}, CSV={csv_value!r}"
        )


def is_formal_contract(video_ids: Sequence[int], question_ids: Sequence[str]) -> bool:
    expected_video_ids = list(range(FORMAL_FIRST_VIDEO, FORMAL_LAST_VIDEO + 1))
    expected_question_ids = {
        f"Real-Time Visual Understanding_sample_{video_id}_{question_index}"
        for video_id in expected_video_ids
        for question_index in range(1, QUESTIONS_PER_VIDEO + 1)
    }
    return (
        list(video_ids) == expected_video_ids
        and len(question_ids) == len(expected_question_ids)
        and set(question_ids) == expected_question_ids
    )


def build_subset(
    unified_items: Sequence[Mapping[str, Any]],
    csv_rows: Iterable[Mapping[str, str]],
    *,
    first_video: int = FORMAL_FIRST_VIDEO,
    last_video: int = FORMAL_LAST_VIDEO,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate both metadata sources and return a selected subset plus audit data."""
    if first_video < 1 or last_video < first_video:
        raise ValueError("invalid video range")

    csv_by_question, csv_by_video = _index_csv_rows(csv_rows)
    oasis_by_video, _ = _index_unified_items(unified_items)
    video_ids = list(range(first_video, last_video + 1))
    subset: list[dict[str, Any]] = []
    question_ids: list[str] = []
    video_paths: list[dict[str, Any]] = []
    timestamp_normalizations: list[dict[str, Any]] = []

    for video_id in video_ids:
        if video_id not in oasis_by_video:
            raise ValueError(f"OASIS metadata is missing sample_{video_id}")
        item = oasis_by_video[video_id]
        breakpoints = item["breakpoint"]
        expected_ids = {
            f"Real-Time Visual Understanding_sample_{video_id}_{question_index}"
            for question_index in range(1, QUESTIONS_PER_VIDEO + 1)
        }
        oasis_ids = [breakpoint["question_id"] for breakpoint in breakpoints]
        if len(oasis_ids) != QUESTIONS_PER_VIDEO or set(oasis_ids) != expected_ids:
            raise ValueError(
                f"sample_{video_id} must have exactly question IDs 1 through "
                f"{QUESTIONS_PER_VIDEO} in OASIS metadata; found={oasis_ids}"
            )
        csv_ids = csv_by_video.get(video_id, [])
        if len(csv_ids) != QUESTIONS_PER_VIDEO or set(csv_ids) != expected_ids:
            raise ValueError(
                f"sample_{video_id} must have exactly question IDs 1 through "
                f"{QUESTIONS_PER_VIDEO} in StreamingBench CSV; found={csv_ids}"
            )

        for breakpoint in breakpoints:
            question_id = breakpoint["question_id"]
            row = csv_by_question[question_id]
            _require_equal(question_id, "question", breakpoint.get("question"), row["question"])
            _require_equal(
                question_id,
                "options",
                breakpoint.get("options"),
                parse_csv_options(row["options"]),
            )
            _require_equal(question_id, "gt", breakpoint.get("gt"), row["answer"])
            _require_equal(
                question_id, "answer", breakpoint.get("answer"), row["answer"]
            )
            _require_equal(question_id, "task", breakpoint.get("task"), row["task_type"])

            normalized, seconds, changed = normalize_csv_timestamp(row["time_stamp"])
            oasis_time = breakpoint.get("time")
            if type(oasis_time) is not int or oasis_time < 0:
                raise ValueError(
                    f"{question_id}: OASIS time must be a non-negative integer; "
                    f"found={oasis_time!r}"
                )
            _require_equal(question_id, "time", oasis_time, seconds)
            if changed:
                timestamp_normalizations.append(
                    {
                        "question_id": question_id,
                        "original": row["time_stamp"],
                        "normalized": normalized,
                        "seconds": seconds,
                    }
                )
            question_ids.append(question_id)

        relative_path = _validate_video_relative_path(
            item["info"].get("video_path"), video_id
        )
        video_paths.append(
            {"video_id": video_id, "relative_path": relative_path}
        )
        subset.append(copy.deepcopy(dict(item)))

    audit = {
        "formal_contract": is_formal_contract(video_ids, question_ids),
        "video_count": len(video_ids),
        "question_count": len(question_ids),
        "video_ids": video_ids,
        "question_ids": question_ids,
        "video_paths": video_paths,
        "timestamp_normalizations": timestamp_normalizations,
    }
    return subset, audit


def inspect_prepared_videos(
    prepared_video_root: Path,
    video_paths: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = prepared_video_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"prepared video root not found: {root}")
    records: list[dict[str, Any]] = []
    for mapping in video_paths:
        video_id = int(mapping["video_id"])
        source = root / f"sample_{video_id}" / "video.mp4"
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"prepared video is missing or empty: {source}")
        source = source.resolve()
        records.append(
            {
                "video_id": video_id,
                "relative_path": str(mapping["relative_path"]),
                "source_path": str(source),
                "size_bytes": source.stat().st_size,
                "crc32": crc32_file(source),
            }
        )
    return records


def validate_upstream_manifest(
    upstream_manifest: Path,
    *,
    verified_csv: Path,
    prepared_video_root: Path,
) -> dict[str, Any]:
    path = upstream_manifest.resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid upstream manifest JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"upstream manifest must be an object: {path}")
    expected_csv_sha = sha256_file(verified_csv)
    if payload.get("subset_sha256") != expected_csv_sha:
        raise ValueError("upstream manifest does not bind the verified CSV")
    if Path(str(payload.get("subset_csv", ""))).resolve() != verified_csv.resolve():
        raise ValueError("upstream manifest subset_csv path differs from input")
    if Path(str(payload.get("video_dir", ""))).resolve() != prepared_video_root.resolve():
        raise ValueError("upstream manifest video_dir differs from input")
    archive = Path(str(payload.get("archive", "")))
    if not archive.is_file() or archive.stat().st_size != payload.get("archive_bytes"):
        raise ValueError("upstream archive is missing or has a different size")
    archive_sha = str(payload.get("archive_sha256", ""))
    metadata_sha = str(payload.get("metadata_sha256", ""))
    if len(archive_sha) != 64 or len(metadata_sha) != 64:
        raise ValueError("upstream manifest lacks archive or metadata SHA256")
    normalizations = payload.get("timestamp_normalizations")
    if not isinstance(normalizations, list) or payload.get(
        "timestamp_normalization_count"
    ) != len(normalizations):
        raise ValueError("upstream timestamp normalization audit is inconsistent")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "metadata_sha256": metadata_sha,
        "subset_sha256": expected_csv_sha,
        "video_count": payload.get("video_count"),
        "question_count": payload.get("question_count"),
        "timestamp_normalization_count": len(normalizations),
        "timestamp_normalizations": normalizations,
    }


def render_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.part-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_subset(
    *,
    oasis_unified_json: Path,
    streamingbench_csv: Path,
    prepared_video_root: Path,
    subset_json: Path,
    mapping_manifest: Path,
    upstream_manifest: Path | None = None,
    first_video: int = FORMAL_FIRST_VIDEO,
    last_video: int = FORMAL_LAST_VIDEO,
) -> dict[str, Any]:
    source_path = oasis_unified_json.resolve()
    csv_path = streamingbench_csv.resolve()
    video_root = prepared_video_root.resolve()
    subset_path = subset_json.resolve()
    manifest_path = mapping_manifest.resolve()
    if subset_path == manifest_path:
        raise ValueError("subset JSON and mapping manifest must use different paths")
    if subset_path in {source_path, csv_path} or manifest_path in {
        source_path,
        csv_path,
    }:
        raise ValueError("output paths must not overwrite source metadata")

    unified_items = read_unified_metadata(source_path)
    csv_rows = read_verified_csv(csv_path)
    subset, audit = build_subset(
        unified_items,
        csv_rows,
        first_video=first_video,
        last_video=last_video,
    )
    videos = inspect_prepared_videos(video_root, audit["video_paths"])
    upstream = (
        validate_upstream_manifest(
            upstream_manifest,
            verified_csv=csv_path,
            prepared_video_root=video_root,
        )
        if upstream_manifest is not None
        else None
    )
    subset_payload = render_json(subset)
    manifest = {
        "format_version": 1,
        "formal_contract": audit["formal_contract"],
        "source_metadata_sha256": sha256_file(source_path),
        "subset_metadata_sha256": hashlib.sha256(subset_payload).hexdigest(),
        "video_count": audit["video_count"],
        "question_count": audit["question_count"],
        "video_ids": audit["video_ids"],
        "question_ids": audit["question_ids"],
        "videos": videos,
        "verified_csv_sha256": sha256_file(csv_path),
        "source_metadata_path": str(source_path),
        "verified_csv_path": str(csv_path),
        "subset_metadata_path": str(subset_path),
        "prepared_video_root": str(video_root),
        "selection": {
            "first_video": first_video,
            "last_video": last_video,
            "questions_per_video": QUESTIONS_PER_VIDEO,
        },
        "timestamp_normalization_count": len(
            audit["timestamp_normalizations"]
        ),
        "timestamp_normalizations": audit["timestamp_normalizations"],
        "video_materialization": "none_manifest_mapping_only",
        "upstream_dataset_manifest": upstream,
    }
    manifest_payload = render_json(manifest)

    # Both payloads are complete and validated before either destination changes.
    atomic_write(subset_path, subset_payload)
    atomic_write(manifest_path, manifest_payload)
    return manifest


prepare_oasis_streamingbench_subset = prepare_subset


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_subset(
        oasis_unified_json=args.oasis_unified_json,
        streamingbench_csv=args.streamingbench_csv,
        prepared_video_root=args.prepared_video_root,
        subset_json=args.subset_json,
        mapping_manifest=args.mapping_manifest,
        upstream_manifest=args.upstream_manifest,
        first_video=args.first_video,
        last_video=args.last_video,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
