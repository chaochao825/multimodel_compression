from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE_COMMIT = "cf53f781d8740df5c07d7924756acc429641ffd0"
BENCHMARK_PATH = Path("speed_benchmark/benchmark_rekv.py")
REQUIRED_SOURCE_PATHS = (
    BENCHMARK_PATH,
    Path("models/rekv/model/abstract_rekv.py"),
    Path("models/rekv/model/llava_onevision_rekv.py"),
    Path("models/rekv/model/patch.py"),
    Path("stc/__init__.py"),
    Path("stc/config.py"),
)
REQUIRED_MODEL_PATHS = (
    Path("config.json"),
    Path("model.safetensors.index.json"),
    Path("preprocessor_config.json"),
    Path("processor_config.json"),
    Path("tokenizer_config.json"),
    Path("tokenizer.json"),
)
EXPECTED_ARCHITECTURE = "LlavaOnevisionForConditionalGeneration"
EXPECTED_MODEL_TYPE = "llava_onevision"
EXPECTED_VISION_MODEL_TYPE = "siglip_vision_model"
MODE_CONFIGS: dict[str, dict[str, float | int | bool]] = {
    "rekv": {
        "patch_vision": False,
        "token_per_frame": 196,
        "update_token_ratio": 1.0,
        "cache_interval": 4,
    },
    "rekv_stc": {
        "patch_vision": True,
        "token_per_frame": 64,
        "update_token_ratio": 0.25,
        "cache_interval": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audited wrapper for STC's official ReKV latency benchmark"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(os.sys.executable))
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), required=True)
    parser.add_argument("--expected-source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-idle-memory-mib", type=int, default=4096)
    parser.add_argument("--max-idle-utilization", type=int, default=20)
    parser.add_argument("--monitor-interval-seconds", type=float, default=0.25)
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="validate the CUDA Python runtime during a dry run",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_only_path(path: str) -> bool:
    parts = Path(path).parts
    return "__pycache__" in parts or path.endswith((".pyc", ".pyo"))


def validate_checkout(source_root: Path, expected_commit: str) -> dict[str, Any]:
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(f"STC git checkout not found: {source_root}")
    missing = [str(path) for path in REQUIRED_SOURCE_PATHS if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"STC checkout is missing required files: {missing}")

    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
    if commit != expected_commit:
        raise ValueError(f"STC commit mismatch: expected {expected_commit}, found {commit}")
    status_lines = _run_text(
        ["git", "status", "--short", "--untracked-files=all"], cwd=source_root
    ).splitlines()
    code_changes = []
    for line in status_lines:
        relative = line[3:].strip()
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        if relative and not _cache_only_path(relative):
            code_changes.append(line)
    if code_changes:
        raise ValueError(f"STC checkout has non-cache changes: {code_changes}")

    return {
        "path": str(source_root),
        "commit": commit,
        "code_clean": True,
        "required_file_sha256": {
            str(path): sha256_file(source_root / path) for path in REQUIRED_SOURCE_PATHS
        },
    }


def validate_model(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    missing = [str(path) for path in REQUIRED_MODEL_PATHS if not (model_path / path).is_file()]
    if missing:
        raise FileNotFoundError(f"model directory is missing required files: {missing}")

    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    model_type = config.get("model_type")
    vision_model_type = (config.get("vision_config") or {}).get("model_type")
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(f"unexpected model architecture: {architectures}")
    if model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(f"unexpected model_type: {model_type}")
    if vision_model_type != EXPECTED_VISION_MODEL_TYPE:
        raise ValueError(f"unexpected vision model_type: {vision_model_type}")

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model shard index has no non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name for name in shard_names):
        raise ValueError("model shard index contains unsafe or nested paths")
    actual_shards = sorted(path.name for path in model_path.glob("*.safetensors"))
    if actual_shards != shard_names:
        raise ValueError(
            "model shard set does not match index: "
            f"expected={shard_names}, actual={actual_shards}"
        )

    shard_records = []
    for name in shard_names:
        path = model_path / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty model shard: {path}")
        shard_records.append(
            {"name": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    metadata_paths = sorted(REQUIRED_MODEL_PATHS)
    return {
        "path": str(model_path),
        "architecture": EXPECTED_ARCHITECTURE,
        "model_type": model_type,
        "vision_model_type": vision_model_type,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "metadata_sha256": {
            str(path): sha256_file(model_path / path) for path in metadata_paths
        },
        "shards": shard_records,
        "total_shard_bytes": sum(record["size_bytes"] for record in shard_records),
    }


def mode_environment(mode: str) -> dict[str, str]:
    config = MODE_CONFIGS[mode]
    return {
        "STC_PATCH_VISION": "1" if config["patch_vision"] else "0",
        "STC_TOKEN_PER_FRAME": str(config["token_per_frame"]),
        "STC_UPDATE_TOKEN_RATIO": str(config["update_token_ratio"]),
        "STC_CACHE_INTERVAL": str(config["cache_interval"]),
    }


def build_environment(
    *, source_root: Path, model_path: Path, mode: str, gpu_index: int
) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(source_root / "models/rekv"), str(source_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "REKV_LLAVA_OV_7B_PATH": str(model_path),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29577 + gpu_index),
        }
    )
    environment.update(mode_environment(mode))
    environment.pop("PREFIX", None)
    return environment


def build_command(
    *,
    python_bin: Path,
    source_root: Path,
    mode: str,
    raw_output: Path,
    num_frames: int,
    image_size: int,
    repeats: int,
    warmup: int,
    video: Path | None,
    sample_fps: float,
) -> list[str]:
    command = [
        str(python_bin),
        str(source_root / BENCHMARK_PATH),
        "--model",
        "llava_ov_7b",
        "--num-frames",
        str(num_frames),
        "--image-size",
        str(image_size),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--label",
        mode,
        "--out",
        str(raw_output),
    ]
    if video is not None:
        command.extend(["--video", str(video), "--sample-fps", str(sample_fps)])
    return command


def validate_runtime(
    *, python_bin: Path, environment: dict[str, str], require_decord: bool
) -> dict[str, Any]:
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_bin}")
    script = """
import importlib
import json
import platform

names = ["torch", "transformers", "numpy", "logzero", "accelerate", "safetensors"]
if __import__("os").environ.get("STC_REQUIRE_DECORD") == "1":
    names.append("decord")
versions = {"python": platform.python_version()}
for name in names:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
import torch
versions["cuda_available"] = torch.cuda.is_available()
versions["torch_cuda"] = torch.version.cuda
import stc
from model.llava_onevision_rekv import load_model
from speed_benchmark.benchmark_rekv import build_synthetic_video, resolve_llava_ov_path
sample = build_synthetic_video(2, 32)
versions["official_imports"] = {
    "stc_module": stc.__file__,
    "load_model_callable": callable(load_model),
    "resolved_model": resolve_llava_ov_path("llava_ov_7b"),
    "synthetic_shape": list(sample.shape),
    "synthetic_dtype": str(sample.dtype),
}
print(json.dumps(versions, sort_keys=True))
"""
    runtime_environment = environment.copy()
    runtime_environment["STC_REQUIRE_DECORD"] = "1" if require_decord else "0"
    completed = subprocess.run(
        [str(python_bin), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=runtime_environment,
    )
    runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    if not runtime.get("cuda_available"):
        raise RuntimeError("STC benchmark runtime cannot see CUDA")
    torch_base = str(runtime["torch"]).split("+", 1)[0]
    torch_parts = tuple(int(part) for part in torch_base.split(".")[:2])
    if torch_parts < (2, 1):
        raise RuntimeError(f"STC requires torch>=2.1, found {runtime['torch']}")
    return runtime


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
        raise ValueError(f"unexpected nvidia-smi output: {output}")
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
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == str(pid) and fields[1] == gpu_uuid:
            try:
                return int(fields[2])
            except ValueError:
                return 0
    return 0


class GPUMonitor:
    def __init__(
        self,
        *,
        gpu_index: int,
        gpu_uuid: str,
        pid: int,
        baseline_total_mib: int,
        interval_seconds: float,
    ) -> None:
        self.gpu_index = gpu_index
        self.gpu_uuid = gpu_uuid
        self.pid = pid
        self.baseline_total_mib = baseline_total_mib
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 4))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = query_gpu_state(self.gpu_index)
                self.samples.append(
                    {
                        "monotonic_seconds": time.monotonic(),
                        "gpu_total_memory_mib": state["memory_used_mib"],
                        "gpu_utilization_percent": state["utilization_percent"],
                        "process_memory_mib": query_process_memory_mib(
                            self.pid, self.gpu_uuid
                        ),
                    }
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        peak_total = max(
            (int(row["gpu_total_memory_mib"]) for row in self.samples), default=0
        )
        return {
            "sample_count": len(self.samples),
            "gpu_peak_total_mib_sampled": peak_total,
            "gpu_peak_total_delta_mib_sampled": max(
                0, peak_total - self.baseline_total_mib
            ),
            "gpu_peak_process_mib_sampled": max(
                (int(row["process_memory_mib"]) for row in self.samples), default=0
            ),
            "gpu_peak_utilization_percent_sampled": max(
                (int(row["gpu_utilization_percent"]) for row in self.samples), default=0
            ),
        }


def write_gpu_samples(path: Path, samples: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "monotonic_seconds",
                "gpu_total_memory_mib",
                "gpu_utilization_percent",
                "process_memory_mib",
            ],
        )
        writer.writeheader()
        writer.writerows(samples)


def higher_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], found {quantile}")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * (len(ordered) - 1)))
    return float(ordered[index])


def summarize_samples(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return {
        "count": len(values),
        "min": min(values),
        "p50": higher_quantile(values, 0.50),
        "p95": higher_quantile(values, 0.95),
        "p99": higher_quantile(values, 0.99),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "max": max(values),
    }


def _validate_stage(stage: Any, *, repeats: int, name: str) -> list[float]:
    if not isinstance(stage, dict):
        raise ValueError(f"{name} must be an object")
    raw_samples = stage.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != repeats:
        raise ValueError(f"{name}.samples must contain exactly {repeats} values")
    samples = [float(value) for value in raw_samples]
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise ValueError(f"{name}.samples contains non-finite or non-positive values")
    expected = {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "std": statistics.pstdev(samples),
        "max": max(samples),
    }
    for key, value in expected.items():
        reported = stage.get(key)
        if reported is None or not math.isclose(
            float(reported), value, rel_tol=1e-6, abs_tol=1e-5
        ):
            raise ValueError(
                f"{name}.{key} is inconsistent: reported={reported}, expected={value}"
            )
    return samples


def validate_official_result(
    payload: dict[str, Any], *, mode: str, num_frames: int, repeats: int
) -> dict[str, Any]:
    if payload.get("label") != mode:
        raise ValueError(f"unexpected result label: {payload.get('label')}")
    if payload.get("model") != "llava_ov_7b":
        raise ValueError(f"unexpected result model: {payload.get('model')}")
    if int(payload.get("num_frames", -1)) != num_frames:
        raise ValueError(f"unexpected result num_frames: {payload.get('num_frames')}")
    if int(payload.get("repeats", -1)) != repeats:
        raise ValueError(f"unexpected result repeats: {payload.get('repeats')}")

    expected_config = MODE_CONFIGS[mode]
    actual_config = payload.get("config")
    if not isinstance(actual_config, dict):
        raise ValueError("result config must be an object")
    for key, expected in expected_config.items():
        actual = actual_config.get(key)
        if isinstance(expected, float):
            matches = actual is not None and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"result config mismatch for {key}: expected={expected}, actual={actual}"
            )

    peak_mem_gb = float(payload.get("peak_mem_gb", -1.0))
    if not math.isfinite(peak_mem_gb) or peak_mem_gb <= 0.0:
        raise ValueError(f"invalid peak_mem_gb: {peak_mem_gb}")
    vit_samples = _validate_stage(
        payload.get("vit_encode_ms"), repeats=repeats, name="vit_encode_ms"
    )
    llm_samples = _validate_stage(
        payload.get("llm_prefill_ms"), repeats=repeats, name="llm_prefill_ms"
    )
    total_samples = [vit + llm for vit, llm in zip(vit_samples, llm_samples, strict=True)]
    return {
        "vit_encode_ms": summarize_samples(vit_samples),
        "llm_prefill_ms": summarize_samples(llm_samples),
        "instrumented_stage_sum_ms": summarize_samples(total_samples),
        "peak_mem_gb_official": peak_mem_gb,
    }


@contextmanager
def gpu_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU lock is already held: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    if args.gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    for name in ("num_frames", "image_size", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.monitor_interval_seconds <= 0.0:
        raise ValueError("monitor-interval-seconds must be positive")

    source_root = args.source_root.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    python_bin = args.python_bin.resolve()
    video = args.video.resolve() if args.video else None
    if video is not None and not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")

    source = validate_checkout(source_root, args.expected_source_commit)
    model = validate_model(model_path)
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_path)
    environment = build_environment(
        source_root=source_root,
        model_path=model_path,
        mode=args.mode,
        gpu_index=args.gpu_index,
    )
    raw_output = output_dir / "official_raw.json"
    command = build_command(
        python_bin=python_bin,
        source_root=source_root,
        mode=args.mode,
        raw_output=raw_output,
        num_frames=args.num_frames,
        image_size=args.image_size,
        repeats=args.repeats,
        warmup=args.warmup,
        video=video,
        sample_fps=args.sample_fps,
    )
    runtime = None
    if not args.dry_run or args.check_runtime:
        runtime = validate_runtime(
            python_bin=python_bin,
            environment=environment,
            require_decord=video is not None,
        )

    fingerprint_inputs = {
        "format_version": 1,
        "runner_sha256": runner_sha256,
        "source_commit": source["commit"],
        "source_file_sha256": source["required_file_sha256"],
        "model": model,
        "runtime": runtime,
        "mode": args.mode,
        "mode_environment": mode_environment(args.mode),
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "video": str(video) if video else None,
        "video_sha256": sha256_file(video) if video else None,
        "sample_fps": args.sample_fps,
        "gpu_index": args.gpu_index,
        "command": command,
    }
    preflight = {
        "format_version": 1,
        "created_at": utc_now(),
        "runner": {"path": str(runner_path), "sha256": runner_sha256},
        "evidence_scope": (
            "official STC ReKV synthetic/real-video stage latency; measures ViT encode "
            "and visual-token LLM prefill, not TTFT, decode latency, or task quality"
        ),
        "source": source,
        "model": model,
        "runtime": runtime,
        "mode": args.mode,
        "mode_config": MODE_CONFIGS[args.mode],
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTHONPATH",
                "REKV_LLAVA_OV_7B_PATH",
                "STC_PATCH_VISION",
                "STC_TOKEN_PER_FRAME",
                "STC_UPDATE_TOKEN_RATIO",
                "STC_CACHE_INTERVAL",
            )
        },
        "fingerprint_inputs": fingerprint_inputs,
        "run_fingerprint": stable_fingerprint(fingerprint_inputs),
    }
    return preflight, command, environment


def _recover_or_validate_existing(
    *,
    output_dir: Path,
    preflight: dict[str, Any],
    mode: str,
    num_frames: int,
    repeats: int,
) -> bool:
    preflight_path = output_dir / "preflight.json"
    result_path = output_dir / "result.json"
    raw_path = output_dir / "official_raw.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != preflight["run_fingerprint"]:
            raise RuntimeError(
                "output directory has a different run fingerprint and was preserved: "
                f"{output_dir}"
            )
    write_json(preflight_path, preflight)
    if not raw_path.exists():
        if result_path.exists():
            raise RuntimeError(f"result exists without official raw output: {result_path}")
        return False

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    derived = validate_official_result(
        raw, mode=mode, num_frames=num_frames, repeats=repeats
    )
    existing_result = {
        "format_version": 1,
        "status": "recovered_from_valid_official_raw",
        "run_fingerprint": preflight["run_fingerprint"],
        "mode": mode,
        "evidence_scope": preflight["evidence_scope"],
        "derived": derived,
        "official_raw_path": str(raw_path),
        "preflight_path": str(preflight_path),
    }
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != preflight["run_fingerprint"]:
            raise RuntimeError(f"result fingerprint mismatch: {result_path}")
    else:
        write_json(result_path, existing_result)
    return True


def main() -> int:
    args = parse_args()
    preflight, command, environment = build_preflight(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if _recover_or_validate_existing(
        output_dir=output_dir,
        preflight=preflight,
        mode=args.mode,
        num_frames=args.num_frames,
        repeats=args.repeats,
    ):
        print(f"validated existing STC ReKV result: {output_dir / 'result.json'}")
        return 0
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
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

    lock_path = args.gpu_lock_path or Path(
        f"/tmp/online-video-state-gpu-{args.gpu_index}.lock"
    )
    log_path = output_dir / "launch.log"
    started_at = utc_now()
    started_monotonic = time.monotonic()
    with gpu_lock(lock_path):
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== run {started_at} fingerprint={preflight['run_fingerprint']} ===\n"
            )
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=args.source_root.resolve(),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            monitor = GPUMonitor(
                gpu_index=args.gpu_index,
                gpu_uuid=str(gpu_before["uuid"]),
                pid=process.pid,
                baseline_total_mib=int(gpu_before["memory_used_mib"]),
                interval_seconds=args.monitor_interval_seconds,
            )
            monitor.start()
            return_code = process.wait()
            monitor.stop()

    gpu_after = query_gpu_state(args.gpu_index)
    sample_path = output_dir / "gpu_samples.csv"
    write_gpu_samples(sample_path, monitor.samples)
    run_record = {
        "format_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_wall_seconds": time.monotonic() - started_monotonic,
        "return_code": return_code,
        "run_fingerprint": preflight["run_fingerprint"],
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_monitor": monitor.summary(),
        "gpu_samples_path": str(sample_path),
        "log_path": str(log_path),
    }
    write_json(output_dir / "run_record.json", run_record)
    if return_code != 0:
        raise RuntimeError(
            f"official STC benchmark failed with exit code {return_code}; see {log_path}"
        )

    raw_path = output_dir / "official_raw.json"
    if not raw_path.is_file():
        raise RuntimeError(f"official benchmark did not produce {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    derived = validate_official_result(
        raw, mode=args.mode, num_frames=args.num_frames, repeats=args.repeats
    )
    result = {
        "format_version": 1,
        "status": "complete",
        "run_fingerprint": preflight["run_fingerprint"],
        "mode": args.mode,
        "evidence_scope": preflight["evidence_scope"],
        "derived": derived,
        "official_raw_path": str(raw_path),
        "preflight_path": str(output_dir / "preflight.json"),
        "run_record_path": str(output_dir / "run_record.json"),
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
