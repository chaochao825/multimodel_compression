from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from prepare_oasis_streamingbench_subset import (  # noqa: E402
    build_subset,
    crc32_file,
    is_formal_contract,
    normalize_csv_timestamp,
    prepare_subset,
)


FIELDNAMES = [
    "question_id",
    "task_type",
    "question",
    "time_stamp",
    "answer",
    "options",
]


def _question_id(video_id: int, question_index: int) -> str:
    return (
        f"Real-Time Visual Understanding_sample_{video_id}_{question_index}"
    )


def _fixture(
    video_ids: list[int],
    *,
    timestamp_overrides: dict[tuple[int, int], tuple[str, int]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    overrides = timestamp_overrides or {}
    unified = []
    rows = []
    for video_id in video_ids:
        breakpoints = []
        for question_index in range(1, 6):
            question_id = _question_id(video_id, question_index)
            question = f"Question {video_id}-{question_index}?"
            options = ["A. One.", "B. Two.", "C. Three.", "D. Four."]
            timestamp, seconds = overrides.get(
                (video_id, question_index),
                (f"00:00:{question_index:02d}", question_index),
            )
            rows.append(
                {
                    "question_id": question_id,
                    "task_type": "Object Perception",
                    "question": question,
                    "time_stamp": timestamp,
                    "answer": "D",
                    "options": repr(options),
                }
            )
            breakpoints.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "options": options,
                    "gt": "D",
                    "answer": "D",
                    "task": "Object Perception",
                    "type": "multiple_choice",
                    "time": seconds,
                }
            )
        unified.append(
            {
                "info": {
                    "video_path": (
                        "StreamingBench/Real-Time_Visual_Understanding/"
                        f"sample_{video_id}/video.mp4"
                    ),
                    "dataset": "streamingbench",
                },
                "breakpoint": breakpoints,
            }
        )
    return unified, rows


def _write_inputs(
    root: Path,
    unified: list[dict[str, object]],
    rows: list[dict[str, str]],
) -> tuple[Path, Path]:
    source = root / "oasis_unified.json"
    metadata_csv = root / "verified.csv"
    source.write_text(json.dumps(unified), encoding="utf-8")
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return source, metadata_csv


def _write_videos(root: Path, video_ids: list[int]) -> Path:
    prepared = root / "prepared"
    for video_id in video_ids:
        path = prepared / f"sample_{video_id}" / "video.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"video-{video_id}".encode())
    return prepared


class OasisStreamingBenchPreparationTest(unittest.TestCase):
    def test_prepare_subset_writes_stable_mapping_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, rows = _fixture([1, 2])
            source, metadata_csv = _write_inputs(root, unified, rows)
            prepared = _write_videos(root, [1, 2])
            subset_path = root / "output" / "subset.json"
            manifest_path = root / "output" / "mapping.json"

            manifest = prepare_subset(
                oasis_unified_json=source,
                streamingbench_csv=metadata_csv,
                prepared_video_root=prepared,
                subset_json=subset_path,
                mapping_manifest=manifest_path,
                first_video=1,
                last_video=2,
            )

            required = {
                "format_version",
                "formal_contract",
                "source_metadata_sha256",
                "subset_metadata_sha256",
                "video_count",
                "question_count",
                "video_ids",
                "question_ids",
                "videos",
            }
            self.assertTrue(required <= set(manifest))
            self.assertFalse(manifest["formal_contract"])
            self.assertEqual(manifest["video_count"], 2)
            self.assertEqual(manifest["question_count"], 10)
            self.assertEqual(manifest["video_ids"], [1, 2])
            self.assertEqual(len(manifest["question_ids"]), 10)
            self.assertEqual(
                manifest["subset_metadata_sha256"],
                hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            )
            first_video = manifest["videos"][0]
            self.assertEqual(
                set(first_video),
                {
                    "video_id",
                    "relative_path",
                    "source_path",
                    "size_bytes",
                    "crc32",
                },
            )
            self.assertEqual(first_video["video_id"], 1)
            self.assertEqual(
                first_video["relative_path"],
                "StreamingBench/Real-Time_Visual_Understanding/"
                "sample_1/video.mp4",
            )
            video_path = prepared / "sample_1" / "video.mp4"
            self.assertEqual(first_video["source_path"], str(video_path.resolve()))
            self.assertEqual(first_video["size_bytes"], video_path.stat().st_size)
            self.assertEqual(first_video["crc32"], crc32_file(video_path))
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")), manifest
            )
            self.assertEqual(json.loads(subset_path.read_text()), unified)

    def test_build_subset_rejects_field_mismatch(self) -> None:
        unified, rows = _fixture([1])
        mismatches = {
            "question": "Different question?",
            "options": ["A. Different.", "B. Two.", "C. Three.", "D. Four."],
            "gt": "A",
            "answer": "A",
            "task": "Different Task",
            "time": 99,
        }
        for field, wrong_value in mismatches.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(unified)
                changed[0]["breakpoint"][0][field] = wrong_value
                with self.assertRaisesRegex(ValueError, f"{field} mismatch"):
                    build_subset(changed, rows, first_video=1, last_video=1)

    def test_prepare_subset_rejects_missing_video_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, rows = _fixture([1, 2])
            source, metadata_csv = _write_inputs(root, unified, rows)
            prepared = _write_videos(root, [1])
            subset_path = root / "subset.json"
            manifest_path = root / "mapping.json"

            with self.assertRaisesRegex(FileNotFoundError, "sample_2"):
                prepare_subset(
                    oasis_unified_json=source,
                    streamingbench_csv=metadata_csv,
                    prepared_video_root=prepared,
                    subset_json=subset_path,
                    mapping_manifest=manifest_path,
                    first_video=1,
                    last_video=2,
                )

            self.assertFalse(subset_path.exists())
            self.assertFalse(manifest_path.exists())

    def test_build_subset_rejects_duplicate_question_id(self) -> None:
        unified, rows = _fixture([1])
        duplicate = unified[0]["breakpoint"][0]["question_id"]
        unified[0]["breakpoint"][1]["question_id"] = duplicate

        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            build_subset(unified, rows, first_video=1, last_video=1)

    def test_mm_ss_timestamp_normalizes_to_seconds_and_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, rows = _fixture(
                [40], timestamp_overrides={(40, 3): ("08:01", 481)}
            )
            source, metadata_csv = _write_inputs(root, unified, rows)
            prepared = _write_videos(root, [40])

            manifest = prepare_subset(
                oasis_unified_json=source,
                streamingbench_csv=metadata_csv,
                prepared_video_root=prepared,
                subset_json=root / "subset.json",
                mapping_manifest=root / "mapping.json",
                first_video=40,
                last_video=40,
            )

            self.assertEqual(
                normalize_csv_timestamp("08:01"), ("00:08:01", 481, True)
            )
            self.assertEqual(manifest["timestamp_normalization_count"], 1)
            self.assertEqual(
                manifest["timestamp_normalizations"],
                [
                    {
                        "question_id": _question_id(40, 3),
                        "original": "08:01",
                        "normalized": "00:08:01",
                        "seconds": 481,
                    }
                ],
            )

    def test_prepare_subset_binds_upstream_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, rows = _fixture([1])
            source, metadata_csv = _write_inputs(root, unified, rows)
            prepared = _write_videos(root, [1])
            archive = root / "videos.zip"
            archive.write_bytes(b"archive fixture")
            upstream_path = root / "upstream.json"
            upstream = {
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
                "subset_csv": str(metadata_csv),
                "subset_sha256": hashlib.sha256(metadata_csv.read_bytes()).hexdigest(),
                "video_dir": str(prepared),
                "video_count": 1,
                "question_count": 5,
                "timestamp_normalization_count": 1,
                "timestamp_normalizations": [
                    {
                        "question_id": _question_id(1, 1),
                        "original": "00:01",
                        "normalized": "00:00:01",
                    }
                ],
            }
            upstream_path.write_text(json.dumps(upstream), encoding="utf-8")

            manifest = prepare_subset(
                oasis_unified_json=source,
                streamingbench_csv=metadata_csv,
                prepared_video_root=prepared,
                subset_json=root / "subset.json",
                mapping_manifest=root / "mapping.json",
                upstream_manifest=upstream_path,
                first_video=1,
                last_video=1,
            )
            bound = manifest["upstream_dataset_manifest"]
            self.assertEqual(bound["sha256"], hashlib.sha256(upstream_path.read_bytes()).hexdigest())
            self.assertEqual(bound["timestamp_normalization_count"], 1)
            self.assertEqual(bound["archive_sha256"], "a" * 64)

    def test_formal_contract_requires_exact_50_by_5_grid(self) -> None:
        videos = list(range(1, 51))
        questions = [
            _question_id(video_id, question_index)
            for video_id in videos
            for question_index in range(1, 6)
        ]
        self.assertTrue(is_formal_contract(videos, questions))
        self.assertFalse(is_formal_contract(videos[:-1], questions[:-5]))
        self.assertFalse(is_formal_contract(videos, questions[:-1]))


if __name__ == "__main__":
    unittest.main()
