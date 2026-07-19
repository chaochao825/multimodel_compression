from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from prepare_streamingbench_subset import (
    crc32_file,
    normalize_timestamp,
    parse_question_id,
    sha256_file,
)


DEFAULT_SOURCE_COMMIT = "640104b3786125c4918924f9b666ff7fe04d81de"
EVALUATOR_PATH = Path("llava/eval/modeling_streamingbench.py")
RUNTIME_RESULT_PREFIX = "CAUSALMEM_RUNTIME_RESULT="
EXPECTED_VISION_TOWER = "google/siglip-so400m-patch14-384"
OFFICIAL_ARCHIVE_SHA256 = (
    "39f8e42130424bddfa8c298be882b21fa3e818318e9782e28ef705851c0c82c5"
)
OFFICIAL_VIDEO_IDS = tuple(range(1, 51))
OFFICIAL_QUESTION_COUNT = 250
OFFICIAL_QUESTIONS_PER_VIDEO = 5
OFFICIAL_TIMESTAMP_NORMALIZATION = {
    "question_id": "Real-Time Visual Understanding_sample_40_3",
    "original": "08:01",
    "normalized": "00:08:01",
    "reason": "official evaluator requires HH:MM:SS",
}
REQUIRED_SOURCE_PATHS = (
    EVALUATOR_PATH,
    Path("llava/model/builder.py"),
    Path("llava/model/llava_arch_v3.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned official CausalMem StreamingBench evaluator."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--method", choices=("causal_mem",), required=True)
    parser.add_argument("--output-name", default="pred")
    parser.add_argument("--conv-mode", default="qwen_1_5")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--foss-budget", type=int, default=12000)
    parser.add_argument("--foss-decay", type=float, default=0.9)
    parser.add_argument("--foss-k-max", type=int, default=64)
    parser.add_argument("--foss-max-new-basis", type=int, default=8)
    parser.add_argument("--foss-time-weight", type=float, default=0.8)
    parser.add_argument("--foss-update-ratio", type=float, default=0.1)
    parser.add_argument("--foss-time-power", type=float, default=1.0)
    parser.add_argument("--max-idle-memory-mib", type=int, default=4096)
    parser.add_argument("--max-idle-utilization", type=int, default=20)
    parser.add_argument("--monitor-interval-seconds", type=float, default=0.2)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--allow-smoke-subset",
        action="store_true",
        help=(
            "Allow an exact-ID subset for model-load smoke tests. Such runs are "
            "marked smoke-only and cannot satisfy the formal 50-video contract."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="With --dry-run, also import the pinned CUDA runtime.",
    )
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
    return "__pycache__/" in normalized or normalized.endswith(".pyc")


def validate_source(source_root: Path, expected_commit: str) -> dict[str, Any]:
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(f"CausalMem git checkout not found: {source_root}")
    missing = [
        str(path)
        for path in REQUIRED_SOURCE_PATHS
        if not (source_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"CausalMem checkout is missing required files: {missing}"
        )
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
    if commit != expected_commit:
        raise ValueError(
            f"CausalMem commit mismatch: expected {expected_commit}, found {commit}"
        )
    status_lines = [
        line
        for line in _run_text(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
        ).splitlines()
        if line
    ]
    code_changes = []
    cache_changes = []
    for line in status_lines:
        paths = line[3:].split(" -> ")
        if all(_cache_only_path(path) for path in paths):
            cache_changes.append(line)
        else:
            code_changes.append(line)
    if code_changes:
        raise ValueError(f"CausalMem source has non-cache changes: {code_changes}")
    return {
        "path": str(source_root),
        "commit": commit,
        "required_paths": [str(path) for path in REQUIRED_SOURCE_PATHS],
        "code_clean": True,
        "ignored_cache_status": cache_changes,
    }


def validate_model(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    expected_architecture = "LlavaQwenForCausalLM"
    if (
        model_type not in {"llava", "llava_qwen"}
        or expected_architecture not in architectures
    ):
        raise ValueError(
            "CausalMem requires the original lmms-lab LLaVA-NeXT checkpoint "
            f"with architecture {expected_architecture!r}; found "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )
    vision_tower = config.get("mm_vision_tower")
    if vision_tower != EXPECTED_VISION_TOWER:
        raise ValueError(
            "CausalMem requires the pinned SigLIP vision tower; "
            f"expected {EXPECTED_VISION_TOWER!r}, found {vision_tower!r}"
        )
    incomplete = sorted(str(path) for path in model_path.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"model download is incomplete: {incomplete[:8]}")
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"model shard index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model shard index has no weight_map: {index_path}")
    referenced_names = sorted({str(value) for value in weight_map.values()})
    unsafe_names = [
        name
        for name in referenced_names
        if Path(name).name != name or not name.endswith(".safetensors")
    ]
    if unsafe_names:
        raise ValueError(f"model shard index has unsafe weight paths: {unsafe_names}")
    shards = [model_path / name for name in referenced_names]
    missing_shards = [str(path) for path in shards if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"model shard index references missing files: {missing_shards}"
        )
    actual_names = sorted(path.name for path in model_path.glob("*.safetensors"))
    if actual_names != referenced_names:
        raise ValueError(
            "model safetensor set differs from the shard index: "
            f"referenced={referenced_names}, actual={actual_names}"
        )
    empty = [str(path) for path in shards if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty model shards: {empty}")
    return {
        "path": str(model_path),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "indexed_tensor_count": len(weight_map),
        "model_type": model_type,
        "architectures": architectures,
        "vision_tower": vision_tower,
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


def validate_hf_cached_repo(cache_dir: Path, repo_id: str) -> dict[str, Any]:
    if repo_id != EXPECTED_VISION_TOWER:
        raise ValueError(
            f"unexpected vision repository: expected {EXPECTED_VISION_TOWER!r}, "
            f"found {repo_id!r}"
        )
    storage = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = storage / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"offline Hugging Face cache is missing: {storage}")
    revision = None
    main_ref = storage / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    snapshot = snapshots / revision if revision else None
    if snapshot is None or not snapshot.is_dir():
        if len(candidates) != 1:
            raise ValueError(
                f"cannot select an offline snapshot for {repo_id}: {candidates}"
            )
        snapshot = candidates[0]
        revision = snapshot.name
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"cached repository has no config.json: {snapshot}")
    weights = sorted(snapshot.glob("*.safetensors")) + sorted(
        snapshot.glob("pytorch_model*.bin")
    )
    if not weights:
        raise FileNotFoundError(f"cached repository has no model weights: {snapshot}")
    incomplete = sorted(str(path) for path in storage.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"cached repository download is incomplete: {incomplete[:8]}")
    return {
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "snapshot": str(snapshot),
        "revision": revision,
        "config_sha256": sha256_file(config_path),
        "weight_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weights
        ],
        "weight_bytes": sum(path.stat().st_size for path in weights),
    }


def validate_dataset(gt_file: Path, video_dir: Path) -> dict[str, Any]:
    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)
    if not video_dir.is_dir():
        raise FileNotFoundError(video_dir)
    with gt_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ground-truth CSV contains no rows")
    question_ids: set[str] = set()
    question_ids_in_order: list[str] = []
    video_ids: set[int] = set()
    questions_per_video: Counter[int] = Counter()
    missing_videos: list[str] = []
    for row in rows:
        question_id = row.get("question_id", "")
        if question_id in question_ids:
            raise ValueError(f"duplicate question id in ground truth: {question_id}")
        video_id, question_index = parse_question_id(question_id)
        if question_index < 1:
            raise ValueError(f"invalid question index in ground truth: {question_id}")
        timestamp = row.get("time_stamp", "")
        normalized_timestamp, changed = normalize_timestamp(timestamp)
        if changed:
            raise ValueError(
                f"timestamp must be normalized before evaluation: {question_id} "
                f"has {timestamp!r}, expected {normalized_timestamp!r}"
            )
        question_ids.add(question_id)
        question_ids_in_order.append(question_id)
        video_ids.add(video_id)
        questions_per_video[video_id] += 1
        video_path = video_dir / f"sample_{video_id}" / "video.mp4"
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            missing_videos.append(str(video_path))
    if missing_videos:
        unique = sorted(set(missing_videos))
        raise FileNotFoundError(f"missing prepared videos: {unique[:12]}")
    return {
        "gt_file": str(gt_file),
        "gt_sha256": sha256_file(gt_file),
        "video_dir": str(video_dir),
        "question_count": len(rows),
        "video_count": len(video_ids),
        "video_ids": sorted(video_ids),
        "questions_per_video": dict(sorted(questions_per_video.items())),
        "first_video": min(video_ids),
        "last_video": max(video_ids),
        "question_ids": sorted(question_ids),
        "question_ids_in_order": question_ids_in_order,
    }


def validate_dataset_manifest(
    manifest_path: Path,
    *,
    gt_file: Path,
    video_dir: Path,
    dataset: dict[str, Any],
    allow_smoke_subset: bool = False,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("evidence_scope") != "official_streamingbench_rt_subset":
        raise ValueError("dataset manifest has an unexpected evidence scope")
    if payload.get("subset_sha256") != dataset["gt_sha256"]:
        raise ValueError("dataset manifest does not match the ground-truth CSV")
    if int(payload.get("question_count", -1)) != dataset["question_count"]:
        raise ValueError("dataset manifest question count does not match the CSV")
    if int(payload.get("video_count", -1)) != dataset["video_count"]:
        raise ValueError("dataset manifest video count does not match the CSV")
    if payload.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError(
            "dataset manifest archive SHA-256 does not match the pinned official "
            "StreamingBench RT 1-50 archive"
        )

    expected_full_ids = list(OFFICIAL_VIDEO_IDS)
    expected_counts = {
        video_id: OFFICIAL_QUESTIONS_PER_VIDEO for video_id in OFFICIAL_VIDEO_IDS
    }
    if allow_smoke_subset:
        if payload.get("selection_mode") != "exact_question_ids":
            raise ValueError("smoke subsets must use exact_question_ids selection")
        selected_ids = payload.get("selected_question_ids") or []
        if selected_ids != dataset["question_ids_in_order"]:
            raise ValueError("smoke manifest question order does not match the CSV")
    else:
        contract_errors = []
        if dataset["question_count"] != OFFICIAL_QUESTION_COUNT:
            contract_errors.append(
                f"questions={dataset['question_count']} "
                f"expected={OFFICIAL_QUESTION_COUNT}"
            )
        if dataset["video_ids"] != expected_full_ids:
            contract_errors.append(
                "video IDs are not exactly sample_1 through sample_50"
            )
        if dataset["questions_per_video"] != expected_counts:
            contract_errors.append(
                "each official video must have exactly five questions"
            )
        expected_question_ids = {
            f"Real-Time Visual Understanding_sample_{video_id}_{question_index}"
            for video_id in OFFICIAL_VIDEO_IDS
            for question_index in range(1, OFFICIAL_QUESTIONS_PER_VIDEO + 1)
        }
        if set(dataset["question_ids"]) != expected_question_ids:
            contract_errors.append("question IDs do not match the official 50x5 grid")
        if payload.get("selection_mode") != "video_range":
            contract_errors.append("formal dataset must use video_range selection")
        if contract_errors:
            raise ValueError(
                "formal StreamingBench RT 1-50 contract failed: "
                + "; ".join(contract_errors)
            )

    expected_normalizations = []
    if OFFICIAL_TIMESTAMP_NORMALIZATION["question_id"] in dataset["question_ids"]:
        expected_normalizations.append(OFFICIAL_TIMESTAMP_NORMALIZATION)
    observed_normalizations = payload.get("timestamp_normalizations") or []
    if observed_normalizations != expected_normalizations:
        raise ValueError(
            "dataset manifest timestamp normalizations differ from the pinned "
            f"contract: expected={expected_normalizations}, "
            f"observed={observed_normalizations}"
        )
    if int(payload.get("timestamp_normalization_count", -1)) != len(
        expected_normalizations
    ):
        raise ValueError("dataset manifest timestamp normalization count is invalid")

    video_records = payload.get("videos") or []
    record_by_id = {int(record["video_id"]): record for record in video_records}
    if len(record_by_id) != len(video_records):
        raise ValueError("dataset manifest has duplicate video records")
    expected_ids = dataset["video_ids"]
    if sorted(record_by_id) != expected_ids:
        raise ValueError("dataset manifest video IDs do not match the ground-truth CSV")
    for video_id in expected_ids:
        if video_id not in record_by_id:
            raise ValueError(f"dataset manifest is missing sample_{video_id}")
        path = video_dir / f"sample_{video_id}" / "video.mp4"
        expected_bytes = int(record_by_id[video_id].get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"prepared video size differs from manifest: sample_{video_id}"
            )
        expected_crc32 = str(record_by_id[video_id].get("crc32", "")).lower()
        if len(expected_crc32) != 8:
            raise ValueError(f"dataset manifest lacks CRC32 for sample_{video_id}")
        actual_crc32 = crc32_file(path)
        if actual_crc32 != expected_crc32:
            raise ValueError(
                f"prepared video CRC32 differs from manifest: sample_{video_id}; "
                f"expected={expected_crc32}, actual={actual_crc32}"
            )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "archive": payload.get("archive"),
        "archive_bytes": payload.get("archive_bytes"),
        "archive_sha256": payload["archive_sha256"],
        "metadata_sha256": payload.get("metadata_sha256"),
        "subset_sha256": payload["subset_sha256"],
        "question_count": payload["question_count"],
        "video_count": payload["video_count"],
        "selection_mode": payload.get("selection_mode"),
        "formal_contract": not allow_smoke_subset,
        "timestamp_normalization_count": payload.get(
            "timestamp_normalization_count", 0
        ),
        "timestamp_normalizations": payload.get("timestamp_normalizations", []),
        "gt_file": str(gt_file),
    }


def build_command(
    *,
    python_bin: str,
    source_root: Path,
    model_path: Path,
    video_dir: Path,
    gt_file: Path,
    output_dir: Path,
    output_name: str,
    conv_mode: str,
) -> list[str]:
    return [
        python_bin,
        "-W",
        "ignore",
        str(source_root / EVALUATOR_PATH),
        "--model-path",
        str(model_path),
        "--video-dir",
        str(video_dir),
        "--gt-file",
        str(gt_file),
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
        "--num-chunks",
        "1",
        "--chunk-idx",
        "0",
        "--conv-mode",
        conv_mode,
    ]


def build_environment(
    *,
    source_root: Path,
    hf_hub_cache: Path,
    method: str,
    gpu_index: int,
    foss_budget: int,
    foss_decay: float,
    foss_k_max: int,
    foss_max_new_basis: int,
    foss_time_weight: float,
    foss_update_ratio: float,
    foss_time_power: float,
) -> dict[str, str]:
    if method != "causal_mem":
        raise ValueError(
            "the pinned CausalMem checkout only has a complete causal_mem model "
            "path; its baseline import references a missing upstream source file"
        )
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "FOSS_BUDGET": str(foss_budget),
            "FOSS_DECAY": str(foss_decay),
            "FOSS_K_MAX": str(foss_k_max),
            "FOSS_MAX_NEW_BASIS": str(foss_max_new_basis),
            "FOSS_TIME_WEIGHT": str(foss_time_weight),
            "FOSS_UPDATE_RATIO": str(foss_update_ratio),
            "FOSS_TIME_POWER": str(foss_time_power),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_CACHE": str(hf_hub_cache),
            "METHOD": method,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join([str(source_root), python_path]).rstrip(
                os.pathsep
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "TRANSFORMERS_CACHE": str(hf_hub_cache),
        }
    )
    environment.pop("PREFIX", None)
    environment.pop("WRAPPER", None)
    return environment


def validate_runtime(
    python_bin: str,
    *,
    environment: dict[str, str],
    source_root: Path,
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
    "pandas": "pandas",
    "shortuuid": "shortuuid",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
}}
for module in modules.values():
    importlib.import_module(module)
import torch
payload = {{
    "python": platform.python_version(),
    "packages": {{name: version(name.replace("_", "-")) for name in modules}},
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}}
print({RUNTIME_RESULT_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [python_bin, "-c", code],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr)[-8000:]
        raise RuntimeError(f"CausalMem runtime import check failed:\n{output}")
    result = None
    for line in completed.stdout.splitlines():
        if line.startswith(RUNTIME_RESULT_PREFIX):
            result = json.loads(line[len(RUNTIME_RESULT_PREFIX) :])
    if result is None:
        raise RuntimeError("CausalMem runtime import check produced no result")
    expected = {
        "accelerate": "0.26.0",
        "decord": "0.6.0",
        "flash_attn": "2.5.8",
        "numpy": "1.26.1",
        "torch": "2.2.1",
        "torchvision": "0.17.1",
        "transformers": "4.45.1",
    }
    mismatches = {}
    for name, wanted in expected.items():
        observed = str(result["packages"].get(name, ""))
        try:
            matches = Version(observed).base_version == Version(wanted).base_version
        except InvalidVersion:
            matches = False
        if not matches:
            mismatches[name] = {"expected": wanted, "observed": observed}
    if mismatches:
        raise ValueError(f"CausalMem runtime version mismatch: {mismatches}")
    if not result["cuda_available"]:
        raise RuntimeError("CausalMem runtime cannot see CUDA")
    return result


def query_gpu_state(gpu_index: int) -> dict[str, Any]:
    output = _run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {output}")
    return {
        "uuid": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
    }


def query_process_memory_mib(pid: int, gpu_uuid: str) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return 0
    total = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[0] != str(pid) or fields[1] != gpu_uuid:
            continue
        try:
            total += int(fields[2])
        except ValueError:
            continue
    return total


@dataclass
class GPUMonitor:
    gpu_index: int
    gpu_uuid: str
    pid: int
    interval_seconds: float
    baseline_total_mib: int
    total_samples: list[int] = field(default_factory=list)
    process_samples: list[int] = field(default_factory=list)
    elapsed_samples: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _started_at: float = field(default_factory=time.perf_counter)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                state = query_gpu_state(self.gpu_index)
                self.total_samples.append(int(state["memory_used_mib"]))
                self.process_samples.append(
                    query_process_memory_mib(self.pid, self.gpu_uuid)
                )
                self.elapsed_samples.append(time.perf_counter() - self._started_at)
            except Exception as exc:  # Monitoring must not kill the evaluator.
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        peak_total = max(self.total_samples, default=self.baseline_total_mib)
        return {
            "sample_count": len(self.total_samples),
            "baseline_total_mib": self.baseline_total_mib,
            "gpu_peak_total_mib_sampled": peak_total,
            "gpu_peak_total_delta_mib_sampled": max(
                0, peak_total - self.baseline_total_mib
            ),
            "gpu_peak_process_mib_sampled": max(self.process_samples, default=0),
            "monitor_errors": self.errors[-20:],
        }

    def rows(self) -> list[dict[str, float | int]]:
        return [
            {
                "elapsed_seconds": elapsed,
                "gpu_total_memory_mib": total,
                "evaluator_process_memory_mib": process,
            }
            for elapsed, total, process in zip(
                self.elapsed_samples,
                self.total_samples,
                self.process_samples,
                strict=True,
            )
        ]


def summarize_predictions(
    predictions_path: Path,
    expected_question_ids: set[str],
) -> dict[str, Any]:
    records: list[tuple[int, dict[str, Any]]] = []
    parse_error_lines: list[int] = []
    invalid_records: list[dict[str, Any]] = []
    trailing_newline = True
    if predictions_path.is_file():
        payload = predictions_path.read_bytes()
        trailing_newline = not payload or payload.endswith(b"\n")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            return {
                "predictions_path": str(predictions_path),
                "jsonl_rows": 0,
                "parse_errors": 1,
                "parse_error_lines": [exc.start],
                "trailing_newline": trailing_newline,
                "invalid_records": [],
                "duplicate_ids": [],
                "unexpected_ids": [],
                "completed_questions": 0,
                "expected_questions": len(expected_question_ids),
                "missing_question_ids": sorted(expected_question_ids),
                "correct": 0,
                "accuracy": None,
            }
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_error_lines.append(line_number)
                continue
            if not isinstance(record, dict):
                invalid_records.append(
                    {"line": line_number, "reasons": ["record is not an object"]}
                )
                continue
            records.append((line_number, record))

    valid_records: list[tuple[int, dict[str, Any]]] = []
    for line_number, record in records:
        reasons = []
        question_id = record.get("id")
        if not isinstance(question_id, str) or not question_id:
            reasons.append("id must be a non-empty string")
        if record.get("acc") not in {True, False, "True", "False"}:
            reasons.append("acc must be True or False")
        if not isinstance(record.get("pred"), str):
            reasons.append("pred must be a string")
        answer_id = record.get("answer_id")
        if not isinstance(answer_id, str) or not answer_id:
            reasons.append("answer_id must be a non-empty string")
        if reasons:
            invalid_records.append({"line": line_number, "reasons": reasons})
        else:
            valid_records.append((line_number, record))

    by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for _, record in valid_records:
        question_id = str(record["id"])
        if question_id in seen_ids:
            duplicate_ids.append(question_id)
        seen_ids.add(question_id)
        by_id[question_id] = record
    completed_ids = set(by_id) & expected_question_ids
    correct = sum(
        str(by_id[question_id]["acc"]).lower() == "true"
        for question_id in completed_ids
    )
    completed = len(completed_ids)
    return {
        "predictions_path": str(predictions_path),
        "jsonl_rows": len(records),
        "parse_errors": len(parse_error_lines),
        "parse_error_lines": parse_error_lines,
        "trailing_newline": trailing_newline,
        "invalid_records": invalid_records,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "unexpected_ids": sorted(set(by_id) - expected_question_ids),
        "completed_questions": completed,
        "expected_questions": len(expected_question_ids),
        "missing_question_ids": sorted(expected_question_ids - completed_ids),
        "correct": correct,
        "accuracy": correct / completed if completed else None,
    }


def prediction_integrity_issues(summary: dict[str, Any]) -> list[str]:
    issues = []
    if not summary["trailing_newline"]:
        issues.append("prediction JSONL does not end with a newline")
    if summary["parse_errors"]:
        issues.append(f"malformed JSON rows at {summary['parse_error_lines']}")
    if summary["invalid_records"]:
        issues.append(f"invalid prediction records: {summary['invalid_records'][:8]}")
    if summary["duplicate_ids"]:
        issues.append(f"duplicate IDs: {summary['duplicate_ids'][:8]}")
    if summary["unexpected_ids"]:
        issues.append(f"unexpected IDs: {summary['unexpected_ids'][:8]}")
    return issues


def require_resumable_predictions(summary: dict[str, Any]) -> None:
    issues = prediction_integrity_issues(summary)
    if issues:
        raise ValueError(
            "existing prediction JSONL is unsafe to resume and was preserved: "
            + "; ".join(issues)
        )


def stable_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def write_gpu_samples(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "elapsed_seconds",
            "gpu_total_memory_mib",
            "evaluator_process_memory_mib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_preflight(
    args: argparse.Namespace,
    *,
    check_runtime: bool = False,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    allow_smoke_subset = bool(getattr(args, "allow_smoke_subset", False))
    source_root = args.source_root.resolve()
    model_path = args.model_path.resolve()
    hf_hub_cache = args.hf_hub_cache.resolve()
    video_dir = args.video_dir.resolve()
    gt_file = args.gt_file.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    output_dir = args.output_dir.resolve()
    source = validate_source(source_root, args.expected_source_commit)
    model = validate_model(model_path)
    vision_tower = validate_hf_cached_repo(
        hf_hub_cache, str(model["vision_tower"] or "")
    )
    dataset = validate_dataset(gt_file, video_dir)
    dataset_manifest = validate_dataset_manifest(
        dataset_manifest_path,
        gt_file=gt_file,
        video_dir=video_dir,
        dataset=dataset,
        allow_smoke_subset=allow_smoke_subset,
    )
    command = build_command(
        python_bin=args.python_bin,
        source_root=source_root,
        model_path=model_path,
        video_dir=video_dir,
        gt_file=gt_file,
        output_dir=output_dir,
        output_name=args.output_name,
        conv_mode=args.conv_mode,
    )
    environment = build_environment(
        source_root=source_root,
        hf_hub_cache=hf_hub_cache,
        method=args.method,
        gpu_index=args.gpu_index,
        foss_budget=args.foss_budget,
        foss_decay=args.foss_decay,
        foss_k_max=args.foss_k_max,
        foss_max_new_basis=args.foss_max_new_basis,
        foss_time_weight=args.foss_time_weight,
        foss_update_ratio=args.foss_update_ratio,
        foss_time_power=args.foss_time_power,
    )
    fingerprint_inputs = {
        "format_version": 2,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_commit": source["commit"],
        "model_config_sha256": model["config_sha256"],
        "model_index_sha256": model["index_sha256"],
        "model_weight_files": model["weight_files"],
        "vision_tower_revision": vision_tower["revision"],
        "vision_tower_config_sha256": vision_tower["config_sha256"],
        "vision_tower_weight_files": vision_tower["weight_files"],
        "gt_sha256": dataset["gt_sha256"],
        "dataset_manifest_sha256": dataset_manifest["sha256"],
        "method": args.method,
        "output_name": args.output_name,
        "conv_mode": args.conv_mode,
        "foss_budget": args.foss_budget,
        "foss_decay": args.foss_decay,
        "foss_k_max": args.foss_k_max,
        "foss_max_new_basis": args.foss_max_new_basis,
        "foss_time_weight": args.foss_time_weight,
        "foss_update_ratio": args.foss_update_ratio,
        "foss_time_power": args.foss_time_power,
        "allow_smoke_subset": allow_smoke_subset,
    }
    evidence_tier = (
        "official_model_level_smoke"
        if allow_smoke_subset
        else "official_model_level_rt_1_50"
    )
    preflight = {
        "format_version": 2,
        "evidence_tier": evidence_tier,
        "benchmark": "StreamingBench Real-Time Visual Understanding",
        "scope": (
            "explicit exact-ID smoke subset; not quality evidence"
            if allow_smoke_subset
            else (
                "official Real-Time Visual Understanding samples 1-50 "
                "(50 videos, 250 questions)"
            )
        ),
        "method": args.method,
        "source": source,
        "model": model,
        "vision_tower": vision_tower,
        "dataset": {
            key: value
            for key, value in dataset.items()
            if key not in {"question_ids", "question_ids_in_order"}
        },
        "dataset_manifest": dataset_manifest,
        "official_evaluator_limitations": {
            "generation_max_new_tokens": 4096,
            "per_sample_latency_instrumented": False,
            "tail_latency_available": False,
            "video_decode_exceptions_are_swallowed_upstream": True,
            "test_size_argument_is_ignored_upstream": True,
            "baseline_mode_available": False,
            "baseline_blocker": (
                "pinned upstream llava_qwen.py imports missing "
                "llava/model/llava_arch_baseline.py"
            ),
        },
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "FOSS_BUDGET",
                "FOSS_DECAY",
                "FOSS_K_MAX",
                "FOSS_MAX_NEW_BASIS",
                "FOSS_TIME_WEIGHT",
                "FOSS_UPDATE_RATIO",
                "FOSS_TIME_POWER",
                "HF_HUB_OFFLINE",
                "HF_HUB_CACHE",
                "METHOD",
                "TRANSFORMERS_OFFLINE",
                "TRANSFORMERS_CACHE",
            )
        },
        "fingerprint_inputs": fingerprint_inputs,
        "run_fingerprint": stable_fingerprint(fingerprint_inputs),
    }
    if check_runtime:
        runtime = validate_runtime(
            args.python_bin,
            environment=environment,
            source_root=source_root,
        )
        fingerprint_inputs["runtime"] = runtime
        preflight["runtime"] = runtime
        preflight["run_fingerprint"] = stable_fingerprint(fingerprint_inputs)
    return preflight, command, environment


def main() -> int:
    args = parse_args()
    check_runtime = not args.dry_run or args.check_runtime
    preflight, command, environment = build_preflight(
        args, check_runtime=check_runtime
    )
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != preflight["run_fingerprint"]:
            raise FileExistsError(
                "output has a different run fingerprint and was preserved: "
                f"{output_dir}"
            )
    else:
        write_json(manifest_path, preflight)

    predictions_path = output_dir / f"{args.output_name}.json"
    expected_ids = set(
        validate_dataset(args.gt_file.resolve(), args.video_dir.resolve())[
            "question_ids"
        ]
    )
    quality_before = summarize_predictions(predictions_path, expected_ids)
    require_resumable_predictions(quality_before)
    attempt_started = datetime.now(timezone.utc)
    started_at = attempt_started.isoformat()
    attempt_id = attempt_started.strftime("%Y%m%dT%H%M%S.%fZ")
    if (
        quality_before["completed_questions"]
        == quality_before["expected_questions"]
    ):
        completed_metrics = {
            "format_version": 2,
            "evidence_tier": preflight["evidence_tier"],
            "run_fingerprint": preflight["run_fingerprint"],
            "attempt_id": attempt_id,
            "method": args.method,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": 0,
            "success": True,
            "skipped_already_complete": True,
            "wall_seconds": 0.0,
            "resumed": True,
            "completed_questions_before_attempt": quality_before[
                "completed_questions"
            ],
            "completed_questions_this_attempt": 0,
            "completed_questions_per_second": None,
            "quality": quality_before,
        }
        append_jsonl(output_dir / "attempts.jsonl", completed_metrics)
        print(json.dumps(completed_metrics, sort_keys=True))
        return 0

    gpu_before = query_gpu_state(args.gpu_index)
    if not args.allow_busy_gpu and (
        gpu_before["memory_used_mib"] > args.max_idle_memory_mib
        or gpu_before["utilization_percent"] > args.max_idle_utilization
    ):
        raise RuntimeError(
            "GPU idle gate failed: "
            f"memory={gpu_before['memory_used_mib']} MiB, "
            f"utilization={gpu_before['utilization_percent']}%"
        )

    resumed = quality_before["completed_questions"] > 0
    log_path = output_dir / "run.log"
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== run {started_at} fingerprint={preflight['run_fingerprint']} ===\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=args.source_root.resolve(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitor = GPUMonitor(
            gpu_index=args.gpu_index,
            gpu_uuid=str(gpu_before["uuid"]),
            pid=process.pid,
            interval_seconds=args.monitor_interval_seconds,
            baseline_total_mib=int(gpu_before["memory_used_mib"]),
        )
        monitor.start()
        try:
            returncode = process.wait()
        finally:
            monitor.stop()
    wall_seconds = time.perf_counter() - start

    quality = summarize_predictions(predictions_path, expected_ids)
    completed_this_attempt = max(
        0,
        quality["completed_questions"]
        - quality_before["completed_questions"],
    )
    success = (
        returncode == 0
        and quality["completed_questions"] == quality["expected_questions"]
        and not prediction_integrity_issues(quality)
    )
    gpu_samples_path = output_dir / f"gpu_samples_{attempt_id}.csv"
    metrics = {
        "format_version": 2,
        "evidence_tier": preflight["evidence_tier"],
        "run_fingerprint": preflight["run_fingerprint"],
        "attempt_id": attempt_id,
        "method": args.method,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "success": success,
        "wall_seconds": wall_seconds,
        "resumed": resumed,
        "completed_questions_before_attempt": quality_before["completed_questions"],
        "completed_questions_this_attempt": completed_this_attempt,
        "completed_questions_per_second": (
            completed_this_attempt / wall_seconds if wall_seconds else None
        ),
        "gpu_before": gpu_before,
        "gpu_monitor": monitor.summary(),
        "quality": quality,
        "latency_scope": {
            "wall_seconds_is_process_level": True,
            "wall_seconds_comparable_to_uninterrupted_run": not resumed,
            "per_sample_p50_p95_p99_available": False,
            "valid_for_streamingtom_stc_tail_latency_comparison": False,
        },
        "gpu_samples_path": str(gpu_samples_path),
        "log_path": str(log_path),
    }
    write_gpu_samples(gpu_samples_path, monitor.rows())
    write_json(output_dir / "metrics.json", metrics)
    append_jsonl(output_dir / "attempts.jsonl", metrics)
    print(json.dumps(metrics, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
