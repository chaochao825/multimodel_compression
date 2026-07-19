from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import run_oasis_streamingbench as runner  # noqa: E402


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_source(root: Path) -> str:
    root.mkdir()
    _git("init", cwd=root)
    _git("config", "user.email", "fixture@example.invalid", cwd=root)
    _git("config", "user.name", "Fixture", cwd=root)
    for relative in runner.REQUIRED_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "fixture", cwd=root)
    return _git("rev-parse", "HEAD", cwd=root)


def _write_mllm(root: Path) -> None:
    root.mkdir()
    config = {
        "architectures": [runner.EXPECTED_MLLM_ARCHITECTURE],
        "model_type": runner.EXPECTED_MLLM_TYPE,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"mllm fixture")
    index = {"weight_map": {"weight": shard.name}}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _write_embedding(root: Path) -> None:
    root.mkdir()
    config = {
        "architectures": [runner.EXPECTED_EMBEDDING_ARCHITECTURE],
        "model_type": runner.EXPECTED_EMBEDDING_TYPE,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (
        "config_sentence_transformers.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"embedding fixture")


def _metadata(video_id: int = 1) -> list[dict]:
    breakpoints = []
    for question_index in range(1, 6):
        breakpoints.append(
            {
                "question": f"Question {question_index}?",
                "answer": "A",
                "options": ["A. yes", "B. no"],
                "gt": "A",
                "type": "multiple_choice",
                "time": question_index,
                "task": "Fixture Task",
                "question_id": (
                    "Real-Time Visual Understanding_"
                    f"sample_{video_id}_{question_index}"
                ),
            }
        )
    return [
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
    ]


def _write_dataset(root: Path, metadata_path: Path, manifest_path: Path) -> Path:
    relative = Path(
        "StreamingBench/Real-Time_Visual_Understanding/sample_1/video.mp4"
    )
    video = root / relative
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video fixture")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "formal_contract": False,
        "source_metadata_sha256": runner.EXPECTED_OASIS_METADATA_SHA256,
        "subset_metadata_sha256": runner.sha256_file(metadata_path),
        "video_count": 1,
        "question_count": 5,
        "video_ids": [1],
        "question_ids": [bp["question_id"] for bp in metadata[0]["breakpoint"]],
        "videos": [
            {
                "video_id": 1,
                "relative_path": relative.as_posix(),
                "source_path": str(video),
                "size_bytes": video.stat().st_size,
                "crc32": runner.crc32_file(video),
            }
        ],
        "upstream_dataset_manifest": {
            "sha256": runner.EXPECTED_UPSTREAM_MANIFEST_SHA256,
            "archive_sha256": runner.EXPECTED_STREAMINGBENCH_ARCHIVE_SHA256,
            "metadata_sha256": runner.EXPECTED_STREAMINGBENCH_METADATA_SHA256,
            "timestamp_normalization_count": 1,
            "timestamp_normalizations": [
                {
                    "question_id": (
                        "Real-Time Visual Understanding_sample_40_3"
                    ),
                    "original": "08:01",
                    "normalized": "00:08:01",
                    "reason": "official evaluator requires HH:MM:SS",
                }
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return video


def _official_output(metadata: list[dict], *, error: bool = False) -> dict:
    result = copy.deepcopy(metadata[0])
    for index, breakpoint in enumerate(result["breakpoint"]):
        if error and index == 0:
            breakpoint["prediction"] = "Error"
            breakpoint["error"] = "fixture failure"
            continue
        breakpoint["response"] = "<answer>A</answer>"
        breakpoint["prediction"] = "A"
        breakpoint["correct"] = True
    scored = 4 if error else 5
    return {
        "results": [result],
        "total_videos": 1,
        "mc_total": scored,
        "mc_correct": scored,
        "mc_accuracy": 1.0,
    }


class OasisStreamingBenchTests(unittest.TestCase):
    def test_source_and_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            commit = _write_source(source)
            validated_source = runner.validate_source(source, commit)
            self.assertTrue(validated_source["code_clean"])

            mllm = root / "mllm"
            embedding = root / "embedding"
            _write_mllm(mllm)
            _write_embedding(embedding)
            self.assertEqual(
                runner.validate_mllm(mllm, enforce_pinned=False)["weight_bytes"],
                12,
            )
            self.assertEqual(
                runner.validate_embedding_model(
                    embedding, enforce_pinned=False
                )["weight_bytes"],
                17,
            )
            with self.assertRaisesRegex(ValueError, "pinned asset"):
                runner.validate_mllm(mllm)

            (source / "dirty.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-cache changes"):
                runner.validate_source(source, commit)

    def test_smoke_metadata_and_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "subset.json"
            metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
            dataset_root = root / "dataset"
            manifest_path = root / "manifest.json"
            video = _write_dataset(dataset_root, metadata_path, manifest_path)

            metadata = runner.validate_metadata(
                metadata_path, dataset_root, allow_smoke_subset=True
            )
            self.assertEqual(metadata["question_count"], 5)
            self.assertFalse(metadata["formal_contract"])
            manifest = runner.validate_dataset_manifest(
                manifest_path, metadata=metadata
            )
            self.assertEqual(manifest["video_count"], 1)

            original_manifest = manifest_path.read_text(encoding="utf-8")
            tampered = json.loads(original_manifest)
            tampered["upstream_dataset_manifest"]["archive_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned upstream provenance"):
                runner.validate_dataset_manifest(manifest_path, metadata=metadata)
            manifest_path.write_text(original_manifest, encoding="utf-8")

            video.write_bytes(b"corrupt video")
            with self.assertRaisesRegex(ValueError, "CRC32 mismatch"):
                runner.validate_dataset_manifest(manifest_path, metadata=metadata)

    def test_metadata_rejects_zero_time_and_duplicate_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            relative = Path(
                "StreamingBench/Real-Time_Visual_Understanding/sample_1/video.mp4"
            )
            video = dataset_root / relative
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")

            payload = _metadata()
            payload[0]["breakpoint"][0]["time"] = 0
            path = root / "zero.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-positive breakpoint"):
                runner.validate_metadata(path, dataset_root, allow_smoke_subset=True)

            payload = _metadata()
            payload[0]["breakpoint"][1]["question_id"] = payload[0]["breakpoint"][0][
                "question_id"
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate question ID"):
                runner.validate_metadata(path, dataset_root, allow_smoke_subset=True)

    def test_official_output_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = _metadata()
            metadata_path = root / "subset.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            output_path = root / "output.json"
            output_path.write_text(
                json.dumps(_official_output(metadata)), encoding="utf-8"
            )
            summary = runner.summarize_official_output(
                output_path, metadata_path=metadata_path
            )
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["scored_questions"], 5)

            output_path.write_text(
                json.dumps(_official_output(metadata, error=True)), encoding="utf-8"
            )
            summary = runner.summarize_official_output(
                output_path, metadata_path=metadata_path
            )
            self.assertEqual(summary["scored_questions"], 4)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertIsNone(summary["accuracy"])
            self.assertEqual(summary["accuracy_on_scored"], 1.0)
            self.assertEqual(summary["scored_coverage"], 0.8)

    def test_official_output_rejects_non_prefix_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = _metadata()
            metadata_path = root / "subset.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            payload = _official_output(metadata)
            payload["results"][0]["info"]["video_path"] = "wrong.mp4"
            output_path = root / "output.json"
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume prefix mismatch"):
                runner.summarize_official_output(
                    output_path, metadata_path=metadata_path
                )

    def test_build_command_uses_official_arguments(self) -> None:
        command = runner.build_command(
            python_bin=Path("/env/bin/python"),
            source_root=Path("/source"),
            metadata=Path("/data/subset.json"),
            dataset_root=Path("/data"),
            output_dir=Path("/output"),
            mllm_path=Path("/models/mllm"),
            embedding_path=Path("/models/embedding"),
            config=runner.OFFICIAL_CONFIG,
            seed=runner.DEFAULT_SEED,
        )
        self.assertEqual(
            command,
            [
                str(Path("/env/bin/python")),
                "-c",
                runner.SEEDED_LAUNCH_CODE,
                str(runner.DEFAULT_SEED),
                str(Path("/source") / runner.EVALUATOR_PATH),
                "--metadata",
                str(Path("/data/subset.json")),
                "--dataset_root",
                str(Path("/data")),
                "--output_dir",
                str(Path("/output")),
                "--fps",
                "2.0",
                "--pace",
                "0.0",
                "--shortmem_frames_limit",
                "32",
                "--now_window_frames_limit",
                "16",
                "--buffer_fps",
                "1.0",
                "--frames_per_node",
                "16",
                "--tokens_per_frame",
                "256",
                "--root_cnt_limit",
                "4",
                "--asr",
                "none",
                "--rag_event_retrieve_limit",
                "2",
                "--rag_qa_retrieve_limit",
                "1",
                "--mllm_path",
                str(Path("/models/mllm")),
                "--embedding_path",
                str(Path("/models/embedding")),
            ],
        )

    def test_audited_policy_overrides_are_rejected_before_io(self) -> None:
        base = {
            "gpu_index": 0,
            "allow_busy_gpu": False,
            "expected_source_commit": runner.DEFAULT_SOURCE_COMMIT,
            "seed": runner.DEFAULT_SEED,
            "monitor_interval_seconds": 0.25,
        }
        with self.assertRaisesRegex(ValueError, "busy-GPU bypass"):
            runner.build_preflight(SimpleNamespace(**{**base, "allow_busy_gpu": True}))
        with self.assertRaisesRegex(ValueError, "source commit cannot be overridden"):
            runner.build_preflight(
                SimpleNamespace(**{**base, "expected_source_commit": "0" * 40})
            )
        with self.assertRaisesRegex(ValueError, "preregistered OASIS seed"):
            runner.build_preflight(SimpleNamespace(**{**base, "seed": 7}))

    def test_runtime_result_is_validated_and_fingerprinted(self) -> None:
        runtime = {
            "python": "3.12.13",
            "libc": ["glibc", "2.31"],
            "packages": {
                **runner.EXPECTED_RUNTIME_VERSIONS,
                "numpy": "2.5.1",
                "opencv_python": "5.0.0.93",
            },
            "torch_cuda_version": "12.4",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_capability": [8, 0],
            "cuda_device_name": "NVIDIA A800 80GB PCIe",
            "bf16_supported": True,
            "flash_attn_kernel": {
                "dtype": "bfloat16",
                "shape": [1, 8, 2, 64],
                "finite": True,
            },
            "torch_cxx11_abi": False,
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=runner.RUNTIME_RESULT_PREFIX + json.dumps(runtime) + "\n",
            stderr="",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            observed = runner.validate_runtime(
                Path("/env/bin/python"),
                source_root=Path("/source"),
                environment={"CUDA_VISIBLE_DEVICES": "0"},
            )
        self.assertEqual(observed, runtime)

        preflight = {
            "runtime": None,
            "fingerprint_inputs": {"runtime": None, "scope": "fixture"},
            "run_fingerprint": "static",
        }
        updated = runner.attach_runtime(preflight, runtime)
        self.assertIsNone(preflight["runtime"])
        self.assertEqual(updated["runtime"], runtime)
        self.assertNotEqual(updated["run_fingerprint"], "static")
        self.assertEqual(
            updated["run_fingerprint"],
            runner.stable_fingerprint(updated["fingerprint_inputs"]),
        )

    def test_resume_snapshot_and_gpu_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metadata_path = output_dir / "subset.json"
            metadata_path.write_text("[]", encoding="utf-8")
            official_path = runner._official_output_path(metadata_path, output_dir)
            official_path.write_bytes(b'{"results": []}')
            summary = {
                "complete": False,
                "completed_videos": 0,
                "output_sha256": runner.sha256_file(official_path),
            }
            snapshot = runner.snapshot_resume_output(
                output_dir=output_dir,
                metadata_path=metadata_path,
                summary=summary,
            )
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.read_bytes(), official_path.read_bytes())
            self.assertEqual(
                runner.snapshot_resume_output(
                    output_dir=output_dir,
                    metadata_path=metadata_path,
                    summary=summary,
                ),
                snapshot,
            )
            snapshot.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "snapshot hash mismatch"):
                runner.snapshot_resume_output(
                    output_dir=output_dir,
                    metadata_path=metadata_path,
                    summary=summary,
                )

        idle = {
            "memory_total_mib": 81_920,
            "memory_used_mib": 4_096,
            "utilization_percent": 20,
        }
        runner.require_idle_gpu(
            idle,
            max_memory_mib=4_096,
            max_utilization=20,
            phase="fixture",
        )
        with self.assertRaisesRegex(RuntimeError, "idle gate failed"):
            runner.require_idle_gpu(
                {**idle, "memory_used_mib": 4_097},
                max_memory_mib=4_096,
                max_utilization=20,
                phase="fixture",
            )
        with self.assertRaisesRegex(RuntimeError, "insufficient total memory"):
            runner.require_idle_gpu(
                {**idle, "memory_total_mib": 65_536},
                max_memory_mib=4_096,
                max_utilization=20,
                phase="fixture",
            )

    def test_static_dry_run_does_not_query_or_mutate_gpu_state(self) -> None:
        arguments = SimpleNamespace(dry_run=True, check_runtime=False)
        preflight = {"run_fingerprint": "static"}
        with (
            mock.patch.object(runner, "parse_args", return_value=arguments),
            mock.patch.object(
                runner,
                "build_preflight",
                return_value=(preflight, ["official"], {"CUDA_VISIBLE_DEVICES": "0"}),
            ),
            mock.patch.object(runner, "query_gpu_state") as query_gpu,
        ):
            self.assertEqual(runner.main(), 0)
        query_gpu.assert_not_called()

    def test_preflight_binds_source_models_dataset_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            commit = _write_source(source)
            mllm = root / "mllm"
            embedding = root / "embedding"
            _write_mllm(mllm)
            _write_embedding(embedding)
            metadata_path = root / "subset.json"
            metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
            dataset_root = root / "dataset"
            manifest_path = root / "manifest.json"
            _write_dataset(dataset_root, metadata_path, manifest_path)
            arguments = {
                "source_root": source,
                "metadata": metadata_path,
                "dataset_root": dataset_root,
                "dataset_manifest": manifest_path,
                "mllm_path": mllm,
                "embedding_path": embedding,
                "output_dir": root / "output",
                "python_bin": Path(sys.executable),
                "expected_source_commit": runner.DEFAULT_SOURCE_COMMIT,
                "gpu_index": 0,
                "seed": runner.DEFAULT_SEED,
                "monitor_interval_seconds": 0.25,
                "allow_smoke_subset": True,
                "allow_busy_gpu": False,
                "max_idle_memory_mib": 4096,
                "max_idle_utilization": 20,
            }
            arguments.update(runner.OFFICIAL_CONFIG)
            original_mllm = runner.validate_mllm
            original_embedding = runner.validate_embedding_model
            original_source = runner.validate_source
            sampling = {
                "target_fps": 2.0,
                "video_count": 1,
                "minimum_sampled_frames": 1,
                "records": [],
            }
            with (
                mock.patch.object(
                    runner,
                    "validate_source",
                    side_effect=lambda path, _expected: original_source(path, commit),
                ),
                mock.patch.object(
                    runner,
                    "validate_mllm",
                    side_effect=lambda path: original_mllm(
                        path, enforce_pinned=False
                    ),
                ),
                mock.patch.object(
                    runner,
                    "validate_embedding_model",
                    side_effect=lambda path: original_embedding(
                        path, enforce_pinned=False
                    ),
                ),
                mock.patch.object(
                    runner, "validate_video_sampling", return_value=sampling
                ),
            ):
                preflight, command, environment = runner.build_preflight(
                    SimpleNamespace(**arguments)
                )
            self.assertEqual(preflight["source"]["commit"], commit)
            self.assertEqual(preflight["metadata"]["video_count"], 1)
            self.assertEqual(preflight["official_config"], runner.OFFICIAL_CONFIG)
            self.assertEqual(preflight["fingerprint_inputs"]["runtime"], None)
            self.assertEqual(command, preflight["command"])
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
