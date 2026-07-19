from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from prepare_streamingbench_subset import crc32_file
from run_stc_rekv_official import (
    GPUMonitor,
    gpu_lock,
    query_gpu_state,
    sha256_file,
    stable_fingerprint,
    utc_now,
    write_gpu_samples,
    write_json,
)


DEFAULT_SOURCE_COMMIT = "dbd342c79a1b9b03327d4ec5daa87488737db988"
DEFAULT_SEED = 20260719
MINIMUM_GPU_TOTAL_MEMORY_MIB = 70_000
EVALUATOR_PATH = Path("src/scripts/eval.py")
REQUIRED_SOURCE_PATHS = (
    EVALUATOR_PATH,
    Path("src/oasis/config.py"),
    Path("src/oasis/model.py"),
    Path("src/oasis/event/forest.py"),
    Path("src/oasis/event/segmenter.py"),
    Path("src/oasis/io/stream.py"),
)
EXPECTED_MLLM_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
EXPECTED_MLLM_TYPE = "qwen3_vl"
EXPECTED_EMBEDDING_ARCHITECTURE = "Qwen3ForCausalLM"
EXPECTED_EMBEDDING_TYPE = "qwen3"
EXPECTED_OASIS_METADATA_SHA256 = (
    "5ec95ccacdd5000d40884f2a50a72d7634cbc068d5c104798517ae81fabb0493"
)
EXPECTED_UPSTREAM_MANIFEST_SHA256 = (
    "fc4c18de257107abd9c519d7b92ea22220041f384913b7961f1e0416e7ccbd7a"
)
EXPECTED_STREAMINGBENCH_ARCHIVE_SHA256 = (
    "39f8e42130424bddfa8c298be882b21fa3e818318e9782e28ef705851c0c82c5"
)
EXPECTED_STREAMINGBENCH_METADATA_SHA256 = (
    "0121417a71e1dcf367222ae97e9512e3e99725d72e60d50760a050816ccd58da"
)
EXPECTED_MLLM_ASSET = {
    "config_sha256": "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
    "index_sha256": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    "support_file_sha256": {
        "chat_template.json": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
        "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    },
    "weight_files": [
        {
            "name": "model-00001-of-00004.safetensors",
            "bytes": 4_902_275_944,
            "sha256": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
        },
        {
            "name": "model-00002-of-00004.safetensors",
            "bytes": 4_915_962_496,
            "sha256": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
        },
        {
            "name": "model-00003-of-00004.safetensors",
            "bytes": 4_999_831_048,
            "sha256": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
        },
        {
            "name": "model-00004-of-00004.safetensors",
            "bytes": 2_716_270_024,
            "sha256": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
        },
    ],
}
EXPECTED_EMBEDDING_ASSET = {
    "config_sha256": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "support_file_sha256": {
        "config_sentence_transformers.json": "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
        "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
        "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
    "weight_files": [
        {
            "name": "model.safetensors",
            "bytes": 1_191_586_416,
            "sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        }
    ],
}
RUNTIME_RESULT_PREFIX = "OASIS_RUNTIME_RESULT="
VIDEO_PATH_RE = re.compile(
    r"^StreamingBench/Real-Time_Visual_Understanding/sample_(\d+)/video\.mp4$"
)
QUESTION_ID_RE = re.compile(
    r"^Real-Time Visual Understanding_sample_(\d+)_(\d+)$"
)
OFFICIAL_CONFIG: dict[str, float | int | str] = {
    "fps": 2.0,
    "pace": 0.0,
    "shortmem_frames_limit": 32,
    "now_window_frames_limit": 16,
    "buffer_fps": 1.0,
    "frames_per_node": 16,
    "tokens_per_frame": 256,
    "root_cnt_limit": 4,
    "asr": "none",
    "rag_event_retrieve_limit": 2,
    "rag_qa_retrieve_limit": 1,
}
EXPECTED_RUNTIME_VERSIONS = {
    "accelerate": "1.10.1",
    "decord": "0.6.0",
    "flash_attn": "2.8.3",
    "pydantic": "2.13.4",
    "qwen_vl_utils": "0.0.14",
    "sentence_transformers": "5.6.0",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.57.6",
}
SEEDED_LAUNCH_CODE = """\
import random
import runpy
import sys

import numpy as np
import torch
from transformers import set_seed

seed = int(sys.argv[1])
script = sys.argv[2]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audited wrapper for OASIS on StreamingBench RTU"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--mllm-path", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-idle-memory-mib", type=int, default=4096)
    parser.add_argument("--max-idle-utilization", type=int, default=20)
    parser.add_argument("--monitor-interval-seconds", type=float, default=0.25)
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--allow-smoke-subset",
        action="store_true",
        help="Allow an exact-ID subset; it is marked smoke-only, not formal quality evidence.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    for name, default in OFFICIAL_CONFIG.items():
        argument = "--" + name.replace("_", "-")
        parser.add_argument(argument, type=type(default), default=default)
    return parser.parse_args()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")
    return completed.stdout.strip()


def _cache_only_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "/__pycache__/" in f"/{normalized}" or normalized.endswith(
        (".pyc", ".pyo")
    )


def validate_source(source_root: Path, expected_commit: str) -> dict[str, Any]:
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(f"OASIS git checkout not found: {source_root}")
    missing = [
        str(path) for path in REQUIRED_SOURCE_PATHS if not (source_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"OASIS checkout is missing required files: {missing}")
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
    if commit != expected_commit:
        raise ValueError(
            f"OASIS commit mismatch: expected {expected_commit}, found {commit}"
        )
    status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_root,
    ).splitlines()
    cache_status = []
    code_status = []
    for line in status:
        paths = line[3:].split(" -> ")
        if all(_cache_only_path(path) for path in paths):
            cache_status.append(line)
        else:
            code_status.append(line)
    if code_status:
        raise ValueError(f"OASIS source has non-cache changes: {code_status}")
    return {
        "path": str(source_root),
        "commit": commit,
        "code_clean": True,
        "ignored_cache_status": cache_status,
        "required_file_sha256": {
            str(path): sha256_file(source_root / path) for path in REQUIRED_SOURCE_PATHS
        },
    }


def _safe_indexed_shards(model_path: Path) -> tuple[Path, list[Path], int]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"model shard index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model shard index has no weight_map: {index_path}")
    names = sorted({str(value) for value in weight_map.values()})
    unsafe = [
        name
        for name in names
        if Path(name).name != name or not name.endswith(".safetensors")
    ]
    if unsafe:
        raise ValueError(f"model shard index has unsafe paths: {unsafe}")
    shards = [model_path / name for name in names]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model shard index references missing files: {missing}")
    actual = sorted(path.name for path in model_path.glob("*.safetensors"))
    if actual != names:
        raise ValueError(
            "model safetensor set differs from index: "
            f"referenced={names}, actual={actual}"
        )
    return index_path, shards, len(weight_map)


def _require_pinned_asset(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} differs from the pinned asset: {mismatches}")


def validate_mllm(
    model_path: Path,
    *,
    enforce_pinned: bool = True,
) -> dict[str, Any]:
    required = (
        "config.json",
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3-VL model files are missing: {missing}")
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_MLLM_TYPE or (
        EXPECTED_MLLM_ARCHITECTURE not in (config.get("architectures") or [])
    ):
        raise ValueError(
            "OASIS requires Qwen3-VL-8B-Instruct; found "
            f"model_type={config.get('model_type')!r}, "
            f"architectures={config.get('architectures')!r}"
        )
    index_path, shards, tensor_count = _safe_indexed_shards(model_path)
    result = {
        "path": str(model_path),
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "indexed_tensor_count": tensor_count,
        "support_file_sha256": {
            name: sha256_file(model_path / name) for name in required[1:]
        },
        "weight_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in shards
        ],
        "weight_bytes": sum(path.stat().st_size for path in shards),
    }
    if enforce_pinned:
        _require_pinned_asset(result, EXPECTED_MLLM_ASSET, label="Qwen3-VL")
    return result


def validate_embedding_model(
    model_path: Path,
    *,
    enforce_pinned: bool = True,
) -> dict[str, Any]:
    required = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3 embedding model files are missing: {missing}")
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_EMBEDDING_TYPE or (
        EXPECTED_EMBEDDING_ARCHITECTURE not in (config.get("architectures") or [])
    ):
        raise ValueError(
            "OASIS requires Qwen3-Embedding-0.6B; found "
            f"model_type={config.get('model_type')!r}, "
            f"architectures={config.get('architectures')!r}"
        )
    weight_path = model_path / "model.safetensors"
    result = {
        "path": str(model_path),
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "config_sha256": sha256_file(config_path),
        "support_file_sha256": {
            name: sha256_file(model_path / name) for name in required[1:-1]
        },
        "weight_files": [
            {
                "name": weight_path.name,
                "bytes": weight_path.stat().st_size,
                "sha256": sha256_file(weight_path),
            }
        ],
        "weight_bytes": weight_path.stat().st_size,
    }
    if enforce_pinned:
        _require_pinned_asset(
            result,
            EXPECTED_EMBEDDING_ASSET,
            label="Qwen3 embedding",
        )
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _validate_breakpoint(
    breakpoint: Any,
    *,
    video_id: int,
    seen_question_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(breakpoint, dict):
        raise ValueError(f"sample_{video_id} breakpoint is not an object")
    question_id = breakpoint.get("question_id")
    if not isinstance(question_id, str):
        raise ValueError(f"sample_{video_id} breakpoint has no question_id")
    match = QUESTION_ID_RE.fullmatch(question_id)
    if match is None or int(match.group(1)) != video_id:
        raise ValueError(f"question ID does not match sample_{video_id}: {question_id}")
    if question_id in seen_question_ids:
        raise ValueError(f"duplicate question ID: {question_id}")
    seen_question_ids.add(question_id)
    if not isinstance(breakpoint.get("question"), str) or not breakpoint["question"]:
        raise ValueError(f"question text is empty: {question_id}")
    options = breakpoint.get("options")
    if not isinstance(options, list) or len(options) < 2 or not all(
        isinstance(option, str) and option for option in options
    ):
        raise ValueError(f"invalid options: {question_id}")
    if breakpoint.get("type") != "multiple_choice":
        raise ValueError(f"non-multiple-choice breakpoint: {question_id}")
    gt = breakpoint.get("gt")
    if not isinstance(gt, str) or len(gt.strip()) != 1:
        raise ValueError(f"invalid ground-truth answer: {question_id}")
    gt = gt.strip().upper()
    answer = str(breakpoint.get("answer", "")).strip().upper()
    if answer != gt:
        raise ValueError(f"answer and gt differ: {question_id}")
    seconds = breakpoint.get("time")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"breakpoint time is not numeric: {question_id}")
    if not 0 < float(seconds):
        raise ValueError(
            "OASIS queries before pushing the first frame at t=0; "
            f"non-positive breakpoint is unsafe: {question_id}"
        )
    if not isinstance(breakpoint.get("task"), str) or not breakpoint["task"]:
        raise ValueError(f"breakpoint task is empty: {question_id}")
    return {
        "question_id": question_id,
        "time": float(seconds),
        "task": breakpoint["task"],
        "gt": gt,
    }


def validate_metadata(
    metadata_path: Path,
    dataset_root: Path,
    *,
    allow_smoke_subset: bool,
) -> dict[str, Any]:
    payload = _load_json(metadata_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"OASIS metadata must be a non-empty list: {metadata_path}")
    seen_video_ids: set[int] = set()
    seen_question_ids: set[str] = set()
    videos = []
    questions = []
    for position, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("info"), dict):
            raise ValueError(f"metadata item {position} has no info object")
        info = item["info"]
        relative_path = info.get("video_path")
        if not isinstance(relative_path, str):
            raise ValueError(f"metadata item {position} has no video_path")
        match = VIDEO_PATH_RE.fullmatch(relative_path)
        if match is None:
            raise ValueError(f"unexpected OASIS video path: {relative_path}")
        video_id = int(match.group(1))
        if video_id in seen_video_ids:
            raise ValueError(f"duplicate video ID: sample_{video_id}")
        seen_video_ids.add(video_id)
        if info.get("dataset") != "streamingbench":
            raise ValueError(f"unexpected dataset for sample_{video_id}: {info}")
        video_path = dataset_root / Path(relative_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"OASIS video not found: {video_path}")
        breakpoints = item.get("breakpoint")
        if not isinstance(breakpoints, list) or len(breakpoints) != 5:
            raise ValueError(f"sample_{video_id} must contain exactly five questions")
        validated = [
            _validate_breakpoint(
                breakpoint,
                video_id=video_id,
                seen_question_ids=seen_question_ids,
            )
            for breakpoint in breakpoints
        ]
        if [row["time"] for row in validated] != sorted(
            row["time"] for row in validated
        ):
            raise ValueError(f"sample_{video_id} breakpoints are not time-sorted")
        videos.append(
            {
                "video_id": video_id,
                "relative_path": relative_path,
                "path": str(video_path),
                "size_bytes": video_path.stat().st_size,
            }
        )
        questions.extend(validated)
    video_ids = [row["video_id"] for row in videos]
    formal_contract = video_ids == list(range(1, 51)) and len(questions) == 250
    if not allow_smoke_subset and not formal_contract:
        raise ValueError(
            "formal OASIS run requires exact ordered samples 1-50 and 250 questions; "
            f"found video_ids={video_ids}, questions={len(questions)}"
        )
    if allow_smoke_subset and any(video_id not in range(1, 51) for video_id in video_ids):
        raise ValueError(f"smoke subset contains IDs outside samples 1-50: {video_ids}")
    return {
        "path": str(metadata_path),
        "sha256": sha256_file(metadata_path),
        "dataset_root": str(dataset_root),
        "video_count": len(videos),
        "question_count": len(questions),
        "video_ids": video_ids,
        "question_ids": [row["question_id"] for row in questions],
        "minimum_breakpoint_time_seconds": min(row["time"] for row in questions),
        "formal_contract": formal_contract and not allow_smoke_subset,
        "videos": videos,
        "task_counts": dict(sorted(Counter(row["task"] for row in questions).items())),
    }


def validate_video_sampling(
    metadata: dict[str, Any],
    *,
    fps: float,
) -> dict[str, Any]:
    import decord

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"sampling FPS must be positive and finite: {fps}")
    records = []
    for video in metadata["videos"]:
        path = Path(video["path"])
        try:
            reader = decord.VideoReader(str(path))
            total_frames = len(reader)
            source_fps = float(reader.get_avg_fps())
            if total_frames <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(
                    f"invalid frame count/FPS: frames={total_frames}, fps={source_fps}"
                )
            sampled_frames = int(total_frames / source_fps * fps)
            if sampled_frames < 1:
                raise ValueError(
                    "official OASIS sampler would produce no frames: "
                    f"frames={total_frames}, source_fps={source_fps}, target_fps={fps}"
                )
            first_frame_shape = list(reader[0].shape)
        except Exception as error:
            raise RuntimeError(
                f"video decode/sampling check failed for sample_{video['video_id']}: {error}"
            ) from error
        records.append(
            {
                "video_id": video["video_id"],
                "total_frames": total_frames,
                "source_fps": source_fps,
                "sampled_frames": sampled_frames,
                "first_frame_shape": first_frame_shape,
            }
        )
    return {
        "target_fps": fps,
        "video_count": len(records),
        "minimum_sampled_frames": min(row["sampled_frames"] for row in records),
        "records": records,
    }


def validate_dataset_manifest(
    manifest_path: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must be an object: {manifest_path}")
    if payload.get("source_metadata_sha256") != EXPECTED_OASIS_METADATA_SHA256:
        raise ValueError("dataset manifest does not bind pinned OASIS metadata")
    upstream = payload.get("upstream_dataset_manifest")
    if not isinstance(upstream, dict):
        raise ValueError("dataset manifest lacks upstream preparation provenance")
    pinned_upstream = {
        "sha256": EXPECTED_UPSTREAM_MANIFEST_SHA256,
        "archive_sha256": EXPECTED_STREAMINGBENCH_ARCHIVE_SHA256,
        "metadata_sha256": EXPECTED_STREAMINGBENCH_METADATA_SHA256,
        "timestamp_normalization_count": 1,
    }
    upstream_mismatches = {
        key: {"expected": value, "observed": upstream.get(key)}
        for key, value in pinned_upstream.items()
        if upstream.get(key) != value
    }
    expected_normalization = {
        "question_id": "Real-Time Visual Understanding_sample_40_3",
        "original": "08:01",
        "normalized": "00:08:01",
        "reason": "official evaluator requires HH:MM:SS",
    }
    if upstream.get("timestamp_normalizations") != [expected_normalization]:
        upstream_mismatches["timestamp_normalizations"] = {
            "expected": [expected_normalization],
            "observed": upstream.get("timestamp_normalizations"),
        }
    if upstream_mismatches:
        raise ValueError(
            "dataset manifest differs from pinned upstream provenance: "
            f"{upstream_mismatches}"
        )
    expected = {
        "subset_metadata_sha256": metadata["sha256"],
        "video_count": metadata["video_count"],
        "question_count": metadata["question_count"],
        "video_ids": metadata["video_ids"],
        "question_ids": metadata["question_ids"],
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dataset manifest differs from metadata: {mismatches}")
    records = payload.get("videos")
    if not isinstance(records, list) or len(records) != metadata["video_count"]:
        raise ValueError("dataset manifest has an invalid videos list")
    records_by_id = {record.get("video_id"): record for record in records if isinstance(record, dict)}
    if len(records_by_id) != len(records):
        raise ValueError("dataset manifest contains duplicate or invalid video records")
    if sorted(records_by_id) != sorted(metadata["video_ids"]):
        raise ValueError("dataset manifest video IDs do not match metadata")
    for video in metadata["videos"]:
        record = records_by_id[video["video_id"]]
        for key in ("relative_path", "size_bytes"):
            if record.get(key) != video[key]:
                raise ValueError(
                    f"dataset manifest {key} mismatch for sample_{video['video_id']}"
                )
        expected_crc = str(record.get("crc32", "")).lower()
        if len(expected_crc) != 8:
            raise ValueError(
                f"dataset manifest lacks CRC32 for sample_{video['video_id']}"
            )
        actual_crc = crc32_file(Path(video["path"]))
        if actual_crc != expected_crc:
            raise ValueError(
                f"video CRC32 mismatch for sample_{video['video_id']}: "
                f"expected={expected_crc}, actual={actual_crc}"
            )
    if bool(payload.get("formal_contract")) != bool(metadata["formal_contract"]):
        raise ValueError("dataset manifest formal_contract does not match run scope")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "source_metadata_sha256": payload.get("source_metadata_sha256"),
        "subset_metadata_sha256": payload["subset_metadata_sha256"],
        "video_count": payload["video_count"],
        "question_count": payload["question_count"],
        "formal_contract": payload["formal_contract"],
        "upstream_dataset_manifest": upstream,
        "video_crc32": {
            str(video_id): records_by_id[video_id]["crc32"]
            for video_id in metadata["video_ids"]
        },
    }


def build_environment(
    *,
    source_root: Path,
    gpu_index: int,
    seed: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": os.pathsep.join([str(source_root), python_path]).rstrip(
                os.pathsep
            ),
        }
    )
    environment.pop("PREFIX", None)
    return environment


def build_command(
    *,
    python_bin: Path,
    source_root: Path,
    metadata: Path,
    dataset_root: Path,
    output_dir: Path,
    mllm_path: Path,
    embedding_path: Path,
    config: dict[str, float | int | str],
    seed: int,
) -> list[str]:
    return [
        str(python_bin),
        "-c",
        SEEDED_LAUNCH_CODE,
        str(seed),
        str(source_root / EVALUATOR_PATH),
        "--metadata",
        str(metadata),
        "--dataset_root",
        str(dataset_root),
        "--output_dir",
        str(output_dir),
        "--fps",
        str(config["fps"]),
        "--pace",
        str(config["pace"]),
        "--shortmem_frames_limit",
        str(config["shortmem_frames_limit"]),
        "--now_window_frames_limit",
        str(config["now_window_frames_limit"]),
        "--buffer_fps",
        str(config["buffer_fps"]),
        "--frames_per_node",
        str(config["frames_per_node"]),
        "--tokens_per_frame",
        str(config["tokens_per_frame"]),
        "--root_cnt_limit",
        str(config["root_cnt_limit"]),
        "--asr",
        str(config["asr"]),
        "--rag_event_retrieve_limit",
        str(config["rag_event_retrieve_limit"]),
        "--rag_qa_retrieve_limit",
        str(config["rag_qa_retrieve_limit"]),
        "--mllm_path",
        str(mllm_path),
        "--embedding_path",
        str(embedding_path),
    ]


def validate_runtime(
    python_bin: Path,
    *,
    source_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    code = f"""\
import importlib
import json
import platform
from importlib.metadata import version

modules = {{
    "accelerate": "accelerate",
    "decord": "decord",
    "flash_attn": "flash_attn",
    "numpy": "numpy",
    "opencv_python": "cv2",
    "pydantic": "pydantic",
    "qwen_vl_utils": "qwen_vl_utils",
    "sentence_transformers": "sentence_transformers",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
}}
for module in modules.values():
    importlib.import_module(module)
import torch
from flash_attn import flash_attn_func
importlib.import_module("src.oasis.model")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("selected GPU does not support BF16")
q = torch.randn((1, 8, 2, 64), device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
kernel_output = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
if kernel_output.shape != q.shape or not torch.isfinite(kernel_output).all():
    raise RuntimeError("FlashAttention BF16 kernel smoke failed")
payload = {{
    "python": platform.python_version(),
    "libc": platform.libc_ver(),
    "packages": {{
        "accelerate": version("accelerate"),
        "decord": version("decord"),
        "flash_attn": version("flash-attn"),
        "numpy": version("numpy"),
        "opencv_python": version("opencv-python"),
        "pydantic": version("pydantic"),
        "qwen_vl_utils": version("qwen-vl-utils"),
        "sentence_transformers": version("sentence-transformers"),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "transformers": version("transformers"),
    }},
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "flash_attn_kernel": {{
        "dtype": "bfloat16",
        "shape": list(kernel_output.shape),
        "finite": bool(torch.isfinite(kernel_output).all().item()),
    }},
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
}}
print({RUNTIME_RESULT_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr)[-12000:]
        raise RuntimeError(f"OASIS runtime import check failed:\n{output}")
    runtime = None
    for line in completed.stdout.splitlines():
        if line.startswith(RUNTIME_RESULT_PREFIX):
            runtime = json.loads(line[len(RUNTIME_RESULT_PREFIX) :])
    if runtime is None:
        raise RuntimeError("OASIS runtime import check produced no structured result")
    mismatches = {}
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        observed = str(runtime["packages"].get(package, ""))
        try:
            matches = Version(observed).base_version == Version(expected).base_version
        except InvalidVersion:
            matches = False
        if not matches:
            mismatches[package] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(f"OASIS runtime version mismatch: {mismatches}")
    if not runtime["cuda_available"] or runtime["cuda_device_count"] != 1:
        raise RuntimeError(f"OASIS runtime does not see exactly one CUDA GPU: {runtime}")
    if runtime.get("cuda_capability") != [8, 0]:
        raise RuntimeError(f"OASIS runtime requires an sm_80 GPU: {runtime}")
    if runtime.get("torch_cxx11_abi") is not False:
        raise RuntimeError(f"unexpected Torch C++11 ABI for OASIS: {runtime}")
    kernel = runtime.get("flash_attn_kernel") or {}
    if not runtime.get("bf16_supported") or kernel.get("finite") is not True:
        raise RuntimeError(f"OASIS BF16 FlashAttention kernel check failed: {runtime}")
    return runtime


def attach_runtime(
    preflight: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(preflight)
    updated["runtime"] = runtime
    updated["fingerprint_inputs"]["runtime"] = runtime
    updated["run_fingerprint"] = stable_fingerprint(updated["fingerprint_inputs"])
    return updated


def _base_breakpoint_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    keys = ("question_id", "question", "options", "gt", "answer", "type", "time", "task")
    return all(observed.get(key) == expected.get(key) for key in keys)


def summarize_official_output(
    output_path: Path,
    *,
    metadata_path: Path,
) -> dict[str, Any]:
    expected_items = _load_json(metadata_path)
    payload = _load_json(output_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"invalid OASIS output structure: {output_path}")
    results = payload["results"]
    if len(results) > len(expected_items):
        raise ValueError("OASIS output has more videos than metadata")
    expected_question_count = sum(len(item["breakpoint"]) for item in expected_items)
    observed_ids: set[str] = set()
    errors = []
    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    task_errors: Counter[str] = Counter()
    correct = 0
    scored_questions = 0
    for index, result in enumerate(results):
        expected = expected_items[index]
        if not isinstance(result, dict) or result.get("info") != expected.get("info"):
            raise ValueError(f"OASIS resume prefix mismatch at video index {index}")
        expected_breakpoints = sorted(expected["breakpoint"], key=lambda row: row["time"])
        observed_breakpoints = result.get("breakpoint")
        if not isinstance(observed_breakpoints, list) or len(observed_breakpoints) != len(
            expected_breakpoints
        ):
            raise ValueError(f"incomplete breakpoint set at video index {index}")
        for bp_index, (expected_bp, observed_bp) in enumerate(
            zip(expected_breakpoints, observed_breakpoints, strict=True)
        ):
            if not isinstance(observed_bp, dict) or not _base_breakpoint_matches(
                expected_bp, observed_bp
            ):
                raise ValueError(
                    f"breakpoint mismatch at video index {index}, question {bp_index}"
                )
            question_id = expected_bp["question_id"]
            if question_id in observed_ids:
                raise ValueError(f"duplicate question in OASIS output: {question_id}")
            observed_ids.add(question_id)
            task = expected_bp["task"]
            task_totals[task] += 1
            if "error" in observed_bp or observed_bp.get("prediction") == "Error":
                errors.append(
                    {"question_id": question_id, "error": observed_bp.get("error")}
                )
                task_errors[task] += 1
                continue
            if not isinstance(observed_bp.get("response"), str) or not isinstance(
                observed_bp.get("prediction"), str
            ):
                raise ValueError(f"missing OASIS response/prediction: {question_id}")
            observed_correct = observed_bp.get("correct")
            expected_correct = (
                observed_bp["prediction"].strip().upper()
                == expected_bp["gt"].strip().upper()
            )
            if not isinstance(observed_correct, bool) or observed_correct != expected_correct:
                raise ValueError(f"invalid correctness flag: {question_id}")
            scored_questions += 1
            if observed_correct:
                correct += 1
                task_correct[task] += 1
    completed_questions = len(observed_ids)
    reported = {
        "total_videos": payload.get("total_videos"),
        "mc_total": payload.get("mc_total"),
        "mc_correct": payload.get("mc_correct"),
        "mc_accuracy": payload.get("mc_accuracy"),
    }
    expected_accuracy = correct / scored_questions if scored_questions else 0.0
    if reported["total_videos"] != len(results):
        raise ValueError("OASIS output total_videos is inconsistent")
    if reported["mc_total"] != scored_questions:
        raise ValueError("OASIS output mc_total is inconsistent")
    if reported["mc_correct"] != correct:
        raise ValueError("OASIS output mc_correct is inconsistent")
    if not isinstance(reported["mc_accuracy"], (int, float)) or abs(
        float(reported["mc_accuracy"]) - expected_accuracy
    ) > 1e-12:
        raise ValueError("OASIS output mc_accuracy is inconsistent")
    complete = len(results) == len(expected_items)
    accuracy_on_scored = correct / scored_questions if scored_questions else None
    coverage = scored_questions / expected_question_count
    return {
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "completed_videos": len(results),
        "expected_videos": len(expected_items),
        "completed_questions": completed_questions,
        "scored_questions": scored_questions,
        "expected_questions": expected_question_count,
        "correct": correct,
        "accuracy": accuracy_on_scored if complete and not errors else None,
        "accuracy_on_scored": accuracy_on_scored,
        "accuracy_all_expected": correct / expected_question_count,
        "scored_coverage": coverage,
        "errors": errors,
        "complete": complete,
        "safe_resume_prefix": not errors and not complete,
        "task_metrics": {
            task: {
                "correct": task_correct[task],
                "total": total,
                "scored": total - task_errors[task],
                "errors": task_errors[task],
                "accuracy": (
                    task_correct[task] / (total - task_errors[task])
                    if total > task_errors[task]
                    else None
                ),
                "coverage": (total - task_errors[task]) / total,
            }
            for task, total in sorted(task_totals.items())
        },
    }


def _official_output_path(metadata_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{metadata_path.stem}_output.json"


def _config_from_args(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {key: getattr(args, key) for key in OFFICIAL_CONFIG}


def build_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    if args.gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    if args.allow_busy_gpu:
        raise ValueError("busy-GPU bypass is forbidden for audited OASIS runs")
    if args.expected_source_commit != DEFAULT_SOURCE_COMMIT:
        raise ValueError("the pinned OASIS source commit cannot be overridden")
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"the preregistered OASIS seed must be {DEFAULT_SEED}")
    if args.monitor_interval_seconds <= 0:
        raise ValueError("monitor-interval-seconds must be positive")
    source_root = args.source_root.resolve()
    metadata_path = args.metadata.resolve()
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.dataset_manifest.resolve()
    mllm_path = args.mllm_path.resolve()
    embedding_path = args.embedding_path.resolve()
    output_dir = args.output_dir.resolve()
    python_bin = args.python_bin.resolve()
    config = _config_from_args(args)
    if config != OFFICIAL_CONFIG:
        raise ValueError(
            "paper-comparable OASIS runs must use the pinned official configuration; "
            f"expected={OFFICIAL_CONFIG}, observed={config}"
        )
    source = validate_source(source_root, args.expected_source_commit)
    mllm = validate_mllm(mllm_path)
    embedding = validate_embedding_model(embedding_path)
    metadata = validate_metadata(
        metadata_path,
        dataset_root,
        allow_smoke_subset=args.allow_smoke_subset,
    )
    video_sampling = validate_video_sampling(metadata, fps=float(config["fps"]))
    dataset_manifest = validate_dataset_manifest(manifest_path, metadata=metadata)
    environment = build_environment(
        source_root=source_root,
        gpu_index=args.gpu_index,
        seed=args.seed,
    )
    command = build_command(
        python_bin=python_bin,
        source_root=source_root,
        metadata=metadata_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        mllm_path=mllm_path,
        embedding_path=embedding_path,
        config=config,
        seed=args.seed,
    )
    # CUDA imports and the kernel smoke must only run after the GPU lock is held.
    runtime = None
    runner_path = Path(__file__).resolve()
    helper_path = Path(__file__).with_name("run_stc_rekv_official.py")
    fingerprint_inputs = {
        "format_version": 1,
        "runner_sha256": sha256_file(runner_path),
        "shared_helper_sha256": sha256_file(helper_path),
        "source": source,
        "mllm": mllm,
        "embedding": embedding,
        "metadata_sha256": metadata["sha256"],
        "dataset_manifest_sha256": dataset_manifest["sha256"],
        "video_crc32": dataset_manifest["video_crc32"],
        "video_sampling": video_sampling,
        "runtime": runtime,
        "config": config,
        "seed": args.seed,
        "allow_smoke_subset": args.allow_smoke_subset,
        "gpu_index": args.gpu_index,
        "max_idle_memory_mib": args.max_idle_memory_mib,
        "max_idle_utilization": args.max_idle_utilization,
        "minimum_gpu_total_memory_mib": MINIMUM_GPU_TOTAL_MEMORY_MIB,
        "command": command,
    }
    preflight = {
        "format_version": 1,
        "created_at": utc_now(),
        "benchmark": "StreamingBench Real-Time Visual Understanding",
        "method": "OASIS",
        "evidence_tier": (
            "official_model_level_smoke"
            if args.allow_smoke_subset
            else "official_model_level_rt_1_50"
        ),
        "evidence_scope": (
            "official OASIS quality evaluation; elapsed wall time is an offline, "
            "pace=0 whole-run measurement, not request TTFT or SLO latency"
        ),
        "source": source,
        "mllm": mllm,
        "embedding": embedding,
        "metadata": metadata,
        "video_sampling": video_sampling,
        "dataset_manifest": dataset_manifest,
        "runtime": runtime,
        "official_config": config,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONPATH",
            )
        },
        "fingerprint_inputs": fingerprint_inputs,
        "run_fingerprint": stable_fingerprint(fingerprint_inputs),
    }
    return preflight, command, environment


def _prepare_output_state(
    *,
    output_dir: Path,
    metadata_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any] | None:
    preflight_path = output_dir / "preflight.json"
    official_path = _official_output_path(metadata_path, output_dir)
    result_path = output_dir / "result.json"
    if preflight_path.exists():
        existing = _load_json(preflight_path)
        if existing.get("run_fingerprint") != preflight["run_fingerprint"]:
            raise RuntimeError(
                "output directory has a different run fingerprint and was preserved: "
                f"{output_dir}"
            )
    elif official_path.exists() or result_path.exists():
        raise RuntimeError(
            "untracked OASIS output exists without preflight and was preserved: "
            f"{output_dir}"
        )
    write_json(preflight_path, preflight)
    if not official_path.exists():
        if result_path.exists():
            raise RuntimeError(f"result exists without official output: {result_path}")
        return None
    summary = summarize_official_output(official_path, metadata_path=metadata_path)
    if summary["errors"]:
        raise RuntimeError(
            "existing OASIS output contains failed questions and is unsafe to resume; "
            f"it was preserved: {summary['errors'][:8]}"
        )
    if not summary["complete"]:
        if result_path.exists():
            raise RuntimeError("partial OASIS output unexpectedly has result.json")
        return summary
    recovered = {
        "format_version": 1,
        "status": "recovered_from_valid_official_output",
        "run_fingerprint": preflight["run_fingerprint"],
        "evidence_scope": preflight["evidence_scope"],
        "metrics": summary,
        "official_output_path": str(official_path),
        "preflight_path": str(preflight_path),
    }
    if result_path.exists():
        prior = _load_json(result_path)
        if prior.get("run_fingerprint") != preflight["run_fingerprint"]:
            raise RuntimeError(f"result fingerprint mismatch: {result_path}")
    else:
        write_json(result_path, recovered)
    return summary


def require_idle_gpu(
    state: dict[str, Any],
    *,
    max_memory_mib: int,
    max_utilization: int,
    phase: str,
) -> None:
    if state["memory_total_mib"] < MINIMUM_GPU_TOTAL_MEMORY_MIB:
        raise RuntimeError(
            f"{phase}: GPU has insufficient total memory for no-offload OASIS: {state}"
        )
    if (
        state["memory_used_mib"] > max_memory_mib
        or state["utilization_percent"] > max_utilization
    ):
        raise RuntimeError(
            f"{phase}: GPU idle gate failed: memory={state['memory_used_mib']} MiB, "
            f"utilization={state['utilization_percent']}%"
        )


def snapshot_resume_output(
    *,
    output_dir: Path,
    metadata_path: Path,
    summary: dict[str, Any] | None,
) -> Path | None:
    if summary is None or summary["complete"]:
        return None
    official_path = _official_output_path(metadata_path, output_dir)
    snapshot_dir = output_dir / "resume_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"videos_{summary['completed_videos']:04d}_"
        f"{summary['output_sha256'][:16]}.json"
    )
    destination = snapshot_dir / name
    payload = official_path.read_bytes()
    if destination.exists():
        if sha256_file(destination) != summary["output_sha256"]:
            raise RuntimeError(f"resume snapshot hash mismatch: {destination}")
        return destination
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if sha256_file(destination) != summary["output_sha256"]:
        raise RuntimeError(f"resume snapshot write verification failed: {destination}")
    return destination


def terminate_owned_process_group(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 30.0,
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout_seconds)


def main() -> int:
    args = parse_args()
    preflight, command, environment = build_preflight(args)
    if args.dry_run and not args.check_runtime:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    gpu_before = query_gpu_state(args.gpu_index)
    require_idle_gpu(
        gpu_before,
        max_memory_mib=args.max_idle_memory_mib,
        max_utilization=args.max_idle_utilization,
        phase="before lock",
    )
    lock_path = args.gpu_lock_path or Path(
        f"/tmp/online-video-state-gpu-{args.gpu_index}.lock"
    )
    output_dir = args.output_dir.resolve()
    metadata_path = args.metadata.resolve()
    caught_error: BaseException | None = None
    with gpu_lock(lock_path):
        gpu_locked = query_gpu_state(args.gpu_index)
        require_idle_gpu(
            gpu_locked,
            max_memory_mib=args.max_idle_memory_mib,
            max_utilization=args.max_idle_utilization,
            phase="after lock",
        )
        runtime = validate_runtime(
            args.python_bin.resolve(),
            source_root=args.source_root.resolve(),
            environment=environment,
        )
        preflight = attach_runtime(preflight, runtime)
        if args.dry_run:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = _prepare_output_state(
            output_dir=output_dir,
            metadata_path=metadata_path,
            preflight=preflight,
        )
        if existing is not None and existing["complete"]:
            print(f"validated existing OASIS result: {output_dir / 'result.json'}")
            return 0
        resume_snapshot = snapshot_resume_output(
            output_dir=output_dir,
            metadata_path=metadata_path,
            summary=existing,
        )
        attempt_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        log_path = output_dir / f"launch_{attempt_id}.log"
        started_at = utc_now()
        started_monotonic = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=args.source_root.resolve(),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            monitor = GPUMonitor(
                gpu_index=args.gpu_index,
                gpu_uuid=str(gpu_locked["uuid"]),
                pid=process.pid,
                baseline_total_mib=int(gpu_locked["memory_used_mib"]),
                interval_seconds=args.monitor_interval_seconds,
            )
            monitor.start()
            try:
                return_code = process.wait()
            except BaseException as error:
                caught_error = error
                terminate_owned_process_group(process)
                return_code = process.returncode if process.returncode is not None else -1
            finally:
                monitor.stop()
        gpu_after_locked = query_gpu_state(args.gpu_index)
    finished_at = utc_now()
    elapsed = time.monotonic() - started_monotonic
    gpu_samples_path = output_dir / f"gpu_samples_{attempt_id}.csv"
    write_gpu_samples(gpu_samples_path, monitor.samples)
    official_path = _official_output_path(metadata_path, output_dir)
    partial_metrics = None
    partial_validation_error = None
    if official_path.exists():
        try:
            partial_metrics = summarize_official_output(
                official_path,
                metadata_path=metadata_path,
            )
        except (OSError, ValueError) as error:
            partial_validation_error = f"{type(error).__name__}: {error}"
    run_record = {
        "format_version": 1,
        "attempt_id": attempt_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_wall_seconds": elapsed,
        "elapsed_semantics": "whole official pace=0 evaluation; not request latency",
        "return_code": return_code,
        "run_fingerprint": preflight["run_fingerprint"],
        "gpu_before": gpu_before,
        "gpu_locked": gpu_locked,
        "gpu_after_locked": gpu_after_locked,
        "gpu_monitor": monitor.summary(),
        "gpu_samples_path": str(gpu_samples_path),
        "log_path": str(log_path),
        "resume_snapshot_path": str(resume_snapshot) if resume_snapshot else None,
        "partial_metrics": partial_metrics,
        "partial_validation_error": partial_validation_error,
        "interrupted": caught_error is not None,
    }
    write_json(output_dir / f"run_record_{attempt_id}.json", run_record)
    if caught_error is not None:
        raise RuntimeError(
            f"OASIS run was interrupted; owned process group was stopped: {log_path}"
        ) from caught_error
    if not monitor.samples:
        raise RuntimeError("OASIS GPU monitor produced no samples")
    if return_code != 0:
        raise RuntimeError(
            f"official OASIS evaluator failed with exit code {return_code}; see {log_path}"
        )
    if partial_validation_error is not None:
        raise RuntimeError(
            "official OASIS output failed validation: " + partial_validation_error
        )
    if partial_metrics is None or not partial_metrics["complete"]:
        raise RuntimeError("official OASIS evaluator exited without a complete output")
    if partial_metrics["errors"]:
        raise RuntimeError(
            f"official OASIS output contains failed questions: {partial_metrics['errors'][:8]}"
        )
    result = {
        "format_version": 1,
        "status": "complete",
        "run_fingerprint": preflight["run_fingerprint"],
        "evidence_scope": preflight["evidence_scope"],
        "metrics": partial_metrics,
        "run_record": run_record,
        "official_output_path": str(official_path),
        "preflight_path": str(output_dir / "preflight.json"),
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
