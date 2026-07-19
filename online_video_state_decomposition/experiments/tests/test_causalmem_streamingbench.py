from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from prepare_streamingbench_subset import (  # noqa: E402
    crc32_file,
    normalize_timestamp,
    prepare_subset,
    select_rows,
)
from run_causalmem_streamingbench import (  # noqa: E402
    OFFICIAL_ARCHIVE_SHA256,
    build_preflight,
    require_resumable_predictions,
    summarize_predictions,
    validate_dataset,
    validate_dataset_manifest,
    validate_model,
    validate_source,
)


FIELDNAMES = [
    "question_id",
    "task_type",
    "question",
    "time_stamp",
    "answer",
    "options",
    "frames_required",
    "temporal_clue_type",
]


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _row(video_id: int, question_id: int) -> dict[str, str]:
    return {
        "question_id": (
            f"Real-Time Visual Understanding_sample_{video_id}_{question_id}"
        ),
        "task_type": "test",
        "question": "What happened?",
        "time_stamp": "00:00:01",
        "answer": "A",
        "options": "['yes', 'no', 'later', 'unknown']",
        "frames_required": "[]",
        "temporal_clue_type": "test",
    }


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _create_source_checkout(path: Path) -> str:
    path.mkdir()
    _git("init", cwd=path)
    _git("config", "user.email", "causalmem@example.invalid", cwd=path)
    _git("config", "user.name", "CausalMem Test", cwd=path)
    required = [
        "llava/eval/modeling_streamingbench.py",
        "llava/model/builder.py",
        "llava/model/llava_arch_v3.py",
    ]
    for relative in required:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", ".", cwd=path)
    _git("commit", "-m", "fixture", cwd=path)
    return _git("rev-parse", "HEAD", cwd=path)


def _create_model(path: Path, model_type: str = "llava") -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": ["LlavaQwenForCausalLM"],
                "mm_vision_tower": "google/siglip-so400m-patch14-384",
            }
        ),
        encoding="utf-8",
    )
    shard_name = "model-00001-of-00001.safetensors"
    (path / shard_name).write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"model.embed_tokens.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )


def _create_hf_cache(path: Path) -> None:
    storage = path / "models--google--siglip-so400m-patch14-384"
    revision = "1" * 40
    (storage / "refs").mkdir(parents=True)
    (storage / "refs/main").write_text(revision, encoding="utf-8")
    snapshot = storage / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"vision-weights")


class StreamingBenchPreparationTest(unittest.TestCase):
    def test_timestamp_normalization_repairs_official_mm_ss_shape(self) -> None:
        self.assertEqual(normalize_timestamp("08:01"), ("00:08:01", True))
        self.assertEqual(normalize_timestamp("00:08:01"), ("00:08:01", False))
        with self.assertRaises(ValueError):
            normalize_timestamp("00:61:00")

    def test_select_rows_preserves_official_order_and_caps_per_video(self) -> None:
        rows = [_row(video, question) for video in (1, 2) for question in (1, 2, 3)]
        selected = select_rows(
            rows,
            first_video=1,
            last_video=2,
            questions_per_video=2,
        )
        self.assertEqual(
            [row["question_id"] for row in selected],
            [
                "Real-Time Visual Understanding_sample_1_1",
                "Real-Time Visual Understanding_sample_1_2",
                "Real-Time Visual Understanding_sample_2_1",
                "Real-Time Visual Understanding_sample_2_2",
            ],
        )

    def test_select_rows_supports_exact_smoke_question_ids(self) -> None:
        rows = [_row(video, question) for video in (1, 36) for question in (1, 5)]
        requested = [
            "Real-Time Visual Understanding_sample_36_5",
            "Real-Time Visual Understanding_sample_1_1",
        ]

        selected = select_rows(
            rows,
            first_video=1,
            last_video=50,
            exact_question_ids=requested,
        )

        self.assertEqual(
            [row["question_id"] for row in selected],
            requested,
        )

    def test_prepare_subset_maps_nested_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            archive = root / "videos.zip"
            prepared = root / "prepared"
            subset = prepared / "subset.csv"
            _write_metadata(metadata, [_row(1, 1), _row(2, 1)])
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("upstream/realtime/sample_1/video.mp4", b"video-one")
                bundle.writestr("upstream/realtime/sample_2/video.mp4", b"video-two")

            result = prepare_subset(
                archive_path=archive,
                metadata_csv=metadata,
                video_dir=prepared / "realtime_video",
                subset_csv=subset,
                first_video=1,
                last_video=2,
                questions_per_video=0,
                extract=True,
                hash_archive=False,
            )

            self.assertEqual(result["video_count"], 2)
            self.assertEqual(result["question_count"], 2)
            self.assertEqual(result["timestamp_normalization_count"], 0)
            self.assertEqual(
                result["videos"][0]["crc32"],
                crc32_file(prepared / "realtime_video/sample_1/video.mp4"),
            )
            self.assertEqual(
                (prepared / "realtime_video/sample_1/video.mp4").read_bytes(),
                b"video-one",
            )
            self.assertTrue(subset.is_file())

    def test_prepare_subset_rejects_same_size_video_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            archive = root / "videos.zip"
            prepared = root / "prepared"
            subset = prepared / "subset.csv"
            _write_metadata(metadata, [_row(1, 1)])
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("realtime/sample_1/video.mp4", b"video-one")
            prepare_subset(
                archive_path=archive,
                metadata_csv=metadata,
                video_dir=prepared / "realtime_video",
                subset_csv=subset,
                first_video=1,
                last_video=1,
                questions_per_video=0,
                extract=True,
                hash_archive=False,
            )
            video = prepared / "realtime_video/sample_1/video.mp4"
            video.write_bytes(b"video-NO!")

            with self.assertRaisesRegex(OSError, "CRC32 mismatch"):
                prepare_subset(
                    archive_path=archive,
                    metadata_csv=metadata,
                    video_dir=prepared / "realtime_video",
                    subset_csv=subset,
                    first_video=1,
                    last_video=1,
                    questions_per_video=0,
                    extract=True,
                    hash_archive=False,
                )


class CausalMemRunnerTest(unittest.TestCase):
    def test_validate_model_requires_shard_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llava",
                        "architectures": ["LlavaQwenForCausalLM"],
                        "mm_vision_tower": (
                            "google/siglip-so400m-patch14-384"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (model / "model.safetensors").write_bytes(b"weights")

            with self.assertRaisesRegex(FileNotFoundError, "shard index"):
                validate_model(model)

    def test_validate_model_rejects_transformers_native_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _create_model(model, model_type="llava_onevision")
            with self.assertRaisesRegex(ValueError, "LlavaQwenForCausalLM"):
                validate_model(model)

    def test_source_cache_artifacts_are_recorded_but_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "CausalMem"
            commit = _create_source_checkout(source)
            cache = source / "llava/model/__pycache__/builder.cpython-310.pyc"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"cache")

            result = validate_source(source, commit)

            self.assertTrue(result["code_clean"])
            self.assertTrue(result["ignored_cache_status"])

    def test_preflight_builds_single_chunk_official_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "CausalMem"
            commit = _create_source_checkout(source)
            model = root / "model"
            _create_model(model)
            hf_hub_cache = root / "hf-cache"
            _create_hf_cache(hf_hub_cache)
            videos = root / "videos"
            (videos / "sample_1").mkdir(parents=True)
            (videos / "sample_1/video.mp4").write_bytes(b"video")
            video_crc32 = crc32_file(videos / "sample_1/video.mp4")
            gt_file = root / "gt.csv"
            _write_metadata(gt_file, [_row(1, 1)])
            gt_sha256 = hashlib.sha256(gt_file.read_bytes()).hexdigest()
            dataset_manifest = root / "dataset_manifest.json"
            dataset_manifest.write_text(
                json.dumps(
                    {
                        "evidence_scope": "official_streamingbench_rt_subset",
                        "subset_sha256": gt_sha256,
                        "question_count": 1,
                        "video_count": 1,
                        "archive": "fixture.zip",
                        "archive_bytes": 1,
                        "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
                        "metadata_sha256": "b" * 64,
                        "selection_mode": "exact_question_ids",
                        "selected_question_ids": [
                            "Real-Time Visual Understanding_sample_1_1"
                        ],
                        "timestamp_normalization_count": 0,
                        "timestamp_normalizations": [],
                        "videos": [
                            {"video_id": 1, "bytes": 5, "crc32": video_crc32}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                source_root=source,
                expected_source_commit=commit,
                model_path=model,
                hf_hub_cache=hf_hub_cache,
                video_dir=videos,
                gt_file=gt_file,
                dataset_manifest=dataset_manifest,
                output_dir=root / "out",
                python_bin=sys.executable,
                method="causal_mem",
                output_name="pred",
                conv_mode="qwen_1_5",
                gpu_index=2,
                foss_budget=12000,
                foss_decay=0.9,
                foss_k_max=64,
                foss_max_new_basis=8,
                foss_time_weight=0.8,
                foss_update_ratio=0.1,
                foss_time_power=1.0,
                allow_smoke_subset=True,
            )

            preflight, command, environment = build_preflight(args)

            self.assertEqual(
                preflight["evidence_tier"], "official_model_level_smoke"
            )
            self.assertEqual(environment["METHOD"], "causal_mem")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(environment["FOSS_UPDATE_RATIO"], "0.1")
            self.assertEqual(environment["FOSS_TIME_POWER"], "1.0")
            self.assertNotIn("WRAPPER", environment)
            self.assertIn("--num-chunks", command)
            self.assertEqual(command[command.index("--num-chunks") + 1], "1")
            self.assertFalse(
                preflight["official_evaluator_limitations"]["tail_latency_available"]
            )

    def test_formal_dataset_contract_rejects_smoke_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            (videos / "sample_1").mkdir(parents=True)
            video = videos / "sample_1/video.mp4"
            video.write_bytes(b"video")
            gt_file = root / "gt.csv"
            _write_metadata(gt_file, [_row(1, 1)])
            dataset = validate_dataset(gt_file, videos)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "evidence_scope": "official_streamingbench_rt_subset",
                        "subset_sha256": dataset["gt_sha256"],
                        "question_count": 1,
                        "video_count": 1,
                        "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
                        "selection_mode": "exact_question_ids",
                        "selected_question_ids": dataset[
                            "question_ids_in_order"
                        ],
                        "timestamp_normalization_count": 0,
                        "timestamp_normalizations": [],
                        "videos": [
                            {
                                "video_id": 1,
                                "bytes": video.stat().st_size,
                                "crc32": crc32_file(video),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "formal .* contract failed"):
                validate_dataset_manifest(
                    manifest,
                    gt_file=gt_file,
                    video_dir=videos,
                    dataset=dataset,
                )

    def test_prediction_summary_detects_missing_and_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "q1",
                                "acc": "True",
                                "pred": "A",
                                "answer_id": "A",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "q1",
                                "acc": "False",
                                "pred": "B",
                                "answer_id": "A",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "q2",
                                "acc": "True",
                                "pred": "A",
                                "answer_id": "A",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_predictions(path, {"q1", "q2", "q3"})

            self.assertEqual(summary["completed_questions"], 2)
            self.assertEqual(summary["missing_question_ids"], ["q3"])
            self.assertEqual(summary["duplicate_ids"], ["q1"])
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["invalid_records"], [])

    def test_prediction_resume_rejects_partial_trailing_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "acc": "True",
                        "pred": "A",
                        "answer_id": "A",
                    }
                )
                + "\n{\"id\": \"q2\"",
                encoding="utf-8",
            )

            summary = summarize_predictions(path, {"q1", "q2"})

            with self.assertRaisesRegex(ValueError, "unsafe to resume"):
                require_resumable_predictions(summary)


if __name__ == "__main__":
    unittest.main()
