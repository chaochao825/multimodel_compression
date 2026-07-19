from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable


QUESTION_ID_PATTERN = re.compile(r"_sample_(\d+)_(\d+)$")
VIDEO_MEMBER_PATTERN = re.compile(r"(?:^|/)sample_(\d+)/video\.mp4$")
REQUIRED_COLUMNS = {
    "question_id",
    "task_type",
    "question",
    "time_stamp",
    "answer",
    "options",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an auditable StreamingBench real-time subset."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--subset-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--first-video", type=int, default=1)
    parser.add_argument("--last-video", type=int, default=50)
    parser.add_argument(
        "--questions-per-video",
        type=int,
        default=0,
        help="Zero preserves every official question for each selected video.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help=(
            "Select an exact question ID. Repeat for multiple IDs; when used, "
            "the video range and per-video cap are ignored."
        ),
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-archive-sha256", action="store_true")
    return parser.parse_args()


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
    match = QUESTION_ID_PATTERN.search(question_id)
    if match is None:
        raise ValueError(f"unrecognized StreamingBench question id: {question_id}")
    return int(match.group(1)), int(match.group(2))


def normalize_timestamp(value: str) -> tuple[str, bool]:
    original = value.strip()
    parts = original.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    normalized = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return normalized, normalized != original


def normalize_timestamps(
    rows: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    normalized_rows = []
    changes = []
    for source in rows:
        row = dict(source)
        original = row["time_stamp"]
        normalized, changed = normalize_timestamp(original)
        row["time_stamp"] = normalized
        normalized_rows.append(row)
        if changed:
            changes.append(
                {
                    "question_id": row["question_id"],
                    "original": original,
                    "normalized": normalized,
                    "reason": "official evaluator requires HH:MM:SS",
                }
            )
    return normalized_rows, changes


def read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("metadata contains no questions")
    return fieldnames, rows


def select_rows(
    rows: Iterable[dict[str, str]],
    *,
    first_video: int,
    last_video: int,
    questions_per_video: int = 0,
    exact_question_ids: Iterable[str] = (),
) -> list[dict[str, str]]:
    if first_video < 1 or last_video < first_video:
        raise ValueError("invalid video range")
    if questions_per_video < 0:
        raise ValueError("questions_per_video must be non-negative")

    requested = list(exact_question_ids)
    if requested:
        if len(requested) != len(set(requested)):
            raise ValueError("exact question IDs contain duplicates")
        requested_set = set(requested)
        by_question: dict[str, dict[str, str]] = {}
        for row in rows:
            question_id = row["question_id"]
            if question_id not in requested_set:
                continue
            parse_question_id(question_id)
            if question_id in by_question:
                raise ValueError(f"duplicate question id: {question_id}")
            by_question[question_id] = row
        missing = [
            question_id for question_id in requested if question_id not in by_question
        ]
        if missing:
            raise ValueError(f"metadata is missing exact question IDs: {missing}")
        return [by_question[question_id] for question_id in requested]

    selected: list[dict[str, str]] = []
    counts: Counter[int] = Counter()
    seen_questions: set[str] = set()
    for row in rows:
        question_id = row["question_id"]
        video_id, _ = parse_question_id(question_id)
        if not first_video <= video_id <= last_video:
            continue
        if question_id in seen_questions:
            raise ValueError(f"duplicate question id: {question_id}")
        if questions_per_video and counts[video_id] >= questions_per_video:
            continue
        selected.append(row)
        counts[video_id] += 1
        seen_questions.add(question_id)

    expected_videos = set(range(first_video, last_video + 1))
    missing_videos = sorted(expected_videos - set(counts))
    if missing_videos:
        raise ValueError(f"metadata has no questions for videos: {missing_videos}")
    if questions_per_video:
        short = {
            video_id: counts[video_id]
            for video_id in sorted(expected_videos)
            if counts[video_id] != questions_per_video
        }
        if short:
            raise ValueError(f"insufficient questions for requested cap: {short}")
    return selected


def render_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_if_absent_or_identical(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different file: {path}")
        return
    path.write_bytes(payload)


def find_video_members(
    archive: zipfile.ZipFile,
    video_ids: Iterable[int],
) -> dict[int, zipfile.ZipInfo]:
    requested = set(video_ids)
    matches: dict[int, zipfile.ZipInfo] = {}
    for member in archive.infolist():
        normalized = member.filename.replace("\\", "/")
        match = VIDEO_MEMBER_PATTERN.search(normalized)
        if match is None:
            continue
        video_id = int(match.group(1))
        if video_id not in requested:
            continue
        if video_id in matches:
            raise ValueError(
                f"archive has duplicate video member for sample_{video_id}"
            )
        matches[video_id] = member
    missing = sorted(requested - set(matches))
    if missing:
        raise ValueError(f"archive is missing selected videos: {missing}")
    return matches


def extract_videos(
    archive_path: Path,
    video_dir: Path,
    video_ids: Iterable[int],
) -> list[dict[str, object]]:
    video_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = find_video_members(archive, video_ids)
        records: list[dict[str, object]] = []
        for video_id in sorted(members):
            member = members[video_id]
            target = video_dir / f"sample_{video_id}" / "video.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            status = "existing"
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise FileExistsError(
                        f"existing video has the wrong size and was preserved: {target}"
                    )
            else:
                temporary = target.with_name(f"video.mp4.part-{os.getpid()}")
                with (
                    archive.open(member) as source,
                    temporary.open("xb") as destination,
                ):
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                if temporary.stat().st_size != member.file_size:
                    raise OSError(
                        f"incomplete extracted video preserved at: {temporary}"
                    )
                os.replace(temporary, target)
                status = "extracted"
            actual_crc32 = crc32_file(target)
            expected_crc32 = f"{member.CRC:08x}"
            if actual_crc32 != expected_crc32:
                raise OSError(
                    f"video CRC32 mismatch for sample_{video_id}: "
                    f"expected {expected_crc32}, found {actual_crc32}"
                )
            records.append(
                {
                    "video_id": video_id,
                    "archive_member": member.filename,
                    "target": str(target),
                    "bytes": member.file_size,
                    "crc32": actual_crc32,
                    "status": status,
                }
            )
    return records


def verify_extracted_videos(
    video_dir: Path,
    video_ids: Iterable[int],
) -> list[dict[str, object]]:
    records = []
    for video_id in sorted(video_ids):
        path = video_dir / f"sample_{video_id}" / "video.mp4"
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"prepared video is missing or empty: {path}")
        records.append(
            {
                "video_id": video_id,
                "target": str(path),
                "bytes": path.stat().st_size,
                "crc32": crc32_file(path),
                "status": "verified_existing",
            }
        )
    return records


def prepare_subset(
    *,
    archive_path: Path,
    metadata_csv: Path,
    video_dir: Path,
    subset_csv: Path,
    first_video: int,
    last_video: int,
    questions_per_video: int,
    extract: bool,
    hash_archive: bool,
    exact_question_ids: Iterable[str] = (),
) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)

    requested_question_ids = tuple(exact_question_ids)
    fieldnames, rows = read_metadata(metadata_csv)
    selected = select_rows(
        rows,
        first_video=first_video,
        last_video=last_video,
        questions_per_video=questions_per_video,
        exact_question_ids=requested_question_ids,
    )
    selected, timestamp_normalizations = normalize_timestamps(selected)
    subset_payload = render_csv(fieldnames, selected)
    write_if_absent_or_identical(subset_csv, subset_payload)
    selected_question_ids = [row["question_id"] for row in selected]
    video_ids = sorted(
        {parse_question_id(question_id)[0] for question_id in selected_question_ids}
    )
    if extract:
        videos = extract_videos(archive_path, video_dir, video_ids)
    else:
        videos = verify_extracted_videos(video_dir, video_ids)

    counts = Counter(parse_question_id(row["question_id"])[0] for row in selected)
    return {
        "format_version": 1,
        "evidence_scope": "official_streamingbench_rt_subset",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path) if hash_archive else None,
        "metadata_csv": str(metadata_csv),
        "metadata_sha256": sha256_file(metadata_csv),
        "subset_csv": str(subset_csv),
        "subset_sha256": hashlib.sha256(subset_payload).hexdigest(),
        "video_dir": str(video_dir),
        "selection_mode": (
            "exact_question_ids" if requested_question_ids else "video_range"
        ),
        "selected_question_ids": (
            selected_question_ids if requested_question_ids else []
        ),
        "first_video": min(video_ids),
        "last_video": max(video_ids),
        "video_count": len(video_ids),
        "question_count": len(selected),
        "timestamp_normalization_count": len(timestamp_normalizations),
        "timestamp_normalizations": timestamp_normalizations,
        "questions_per_video": dict(sorted(counts.items())),
        "videos": videos,
    }


def main() -> int:
    args = parse_args()
    payload = prepare_subset(
        archive_path=args.archive.resolve(),
        metadata_csv=args.metadata_csv.resolve(),
        video_dir=args.video_dir.resolve(),
        subset_csv=args.subset_csv.resolve(),
        first_video=args.first_video,
        last_video=args.last_video,
        questions_per_video=args.questions_per_video,
        extract=not args.skip_extract,
        hash_archive=not args.skip_archive_sha256,
        exact_question_ids=args.question_id,
    )
    manifest_payload = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_if_absent_or_identical(args.manifest.resolve(), manifest_payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
