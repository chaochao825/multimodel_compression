#!/usr/bin/env python3
"""Run the frozen EXP-052 exact resident-text runtime stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import run_wan_rcm_baseline as baseline
from experiment_artifacts import JsonlEventLog, atomic_write_json, require_fresh_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("text-screen", "f17-exact", "f81-resident")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--method", choices=baseline.METHODS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-052" or config["gate_id"] != "G-031":
        raise ValueError("config is not the frozen EXP-052/G-031 configuration")
    if tuple(config["methods"]) != baseline.METHODS:
        raise ValueError(f"method order must remain frozen as {baseline.METHODS}")
    if config["positive_prompt_cache"] is not False:
        raise ValueError("EXP-052 forbids positive-prompt embedding reuse")
    if config["resident_text_encoder"] is not True:
        raise ValueError("EXP-052 requires a resident text encoder")
    if config["cache_fixed_negative_prompt"] is not True:
        raise ValueError("EXP-052 requires exact fixed-negative reuse")
    if tuple(config["timing_prompt_indices"]) != (0, 1, 2, 3):
        raise ValueError("EXP-052 requires all four frozen prompt indices")

    base_path = (PROJECT_ROOT / config["base_config"]).resolve()
    base_config = baseline.load_config(base_path)
    if config["timing_seed"] not in base_config["generation"]["formal_seeds"]:
        raise ValueError("EXP-052 timing seed must be registered in EXP-047")
    return config, base_config


class TextEncoderPolicy:
    """Control only UMT5 lifetime; positive prompts are never memoized."""

    def __init__(
        self,
        runtime: dict[str, Any],
        checkpoint_path: str,
        device: Any,
        resident: bool,
        cache_negative: bool,
    ) -> None:
        self.runtime = runtime
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.resident = resident
        self.cache_negative = cache_negative
        self.negative_embedding: Any | None = None
        self.positive_calls = 0
        self.negative_model_calls = 0
        self.negative_cache_hits = 0

    def _model_forward(self, prompt: str) -> Any:
        return self.runtime["get_umt5"](
            checkpoint_path=self.checkpoint_path,
            prompts=prompt,
        )

    def warm(self, prompt: str, need_negative: bool) -> None:
        if not self.resident:
            raise ValueError("only a resident policy can be warmed")
        self.runtime["clear_umt5"]()
        self._model_forward(prompt)
        if need_negative:
            self.negative_embedding = self._model_forward(baseline.NEGATIVE_PROMPT)
            self.negative_model_calls += 1
        baseline.synchronize(self.runtime["torch"], self.device)

    def encode_raw(self, prompt: str, need_negative: bool) -> tuple[Any, Any | None]:
        positive = self._model_forward(prompt)
        self.positive_calls += 1
        negative = None
        if need_negative:
            if self.cache_negative and self.negative_embedding is not None:
                negative = self.negative_embedding
                self.negative_cache_hits += 1
            else:
                negative = self._model_forward(baseline.NEGATIVE_PROMPT)
                self.negative_model_calls += 1
                if self.cache_negative:
                    self.negative_embedding = negative
        if not self.resident:
            self.runtime["clear_umt5"]()
        baseline.synchronize(self.runtime["torch"], self.device)
        return positive, negative

    def encode_condition(
        self, prompt: str, need_negative: bool
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        positive, negative = self.encode_raw(prompt, need_negative)
        repeat = self.runtime["repeat"]
        positive = positive.to(device=self.device, dtype=self.runtime["torch"].bfloat16)
        condition = {
            "crossattn_emb": repeat(positive, "b l d -> (k b) l d", k=1)
        }
        uncondition = None
        if need_negative:
            if negative is None:
                raise RuntimeError("native UniPC requires the frozen negative embedding")
            negative = negative.to(
                device=self.device,
                dtype=self.runtime["torch"].bfloat16,
            )
            uncondition = {
                "crossattn_emb": repeat(negative, "b l d -> (k b) l d", k=1)
            }
        return condition, uncondition

    def stats(self) -> dict[str, int | bool]:
        return {
            "resident": self.resident,
            "cache_negative": self.cache_negative,
            "positive_model_calls": self.positive_calls,
            "positive_cache_hits": 0,
            "negative_model_calls": self.negative_model_calls,
            "negative_cache_hits": self.negative_cache_hits,
        }

    def close(self) -> None:
        self.negative_embedding = None
        self.runtime["clear_umt5"]()


def need_negative(base_config: dict[str, Any], method: str) -> bool:
    return base_config["methods"][method]["kind"] == "native_unipc"


def resolve_output_dir(
    config: dict[str, Any], stage: str, method: str | None, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    suffix = method if method is not None else "all"
    return (Path(config["remote_output_root"]) / stage / suffix).resolve()


def runtime_environment(torch: Any, device: Any) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": gpu.name,
        "gpu_total_memory_bytes": gpu.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
    }


def clone_cpu(tensor: Any) -> Any:
    return tensor.detach().cpu().clone()


def compare_raw_embeddings(
    torch: Any,
    candidate: tuple[Any, Any | None],
    reference: tuple[Any, Any | None],
) -> tuple[bool, bool | None]:
    positive_equal = bool(torch.equal(candidate[0].detach().cpu(), reference[0]))
    if reference[1] is None:
        if candidate[1] is not None:
            raise RuntimeError("candidate unexpectedly produced a negative embedding")
        return positive_equal, None
    if candidate[1] is None:
        raise RuntimeError("candidate omitted the required negative embedding")
    negative_equal = bool(torch.equal(candidate[1].detach().cpu(), reference[1]))
    return positive_equal, negative_equal


def median(rows: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(float(row[field]) for row in rows))


def run_text_profile(
    base_config: dict[str, Any],
    prompts: tuple[str, ...],
    runtime: dict[str, Any],
    device: Any,
    profile: str,
) -> dict[str, Any]:
    torch = runtime["torch"]
    native = profile == "native_cfg"
    if profile not in ("positive_only", "native_cfg"):
        raise ValueError(f"unknown text profile: {profile}")

    resident = TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=native,
    )
    reload_policy = TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=False,
        cache_negative=False,
    )
    resident_rows: list[dict[str, Any]] = []
    reload_rows: list[dict[str, Any]] = []
    references: list[tuple[Any, Any | None]] = []
    try:
        resident.warm(base_config["generation"]["smoke_prompt"], native)
        for prompt_index, prompt in enumerate(prompts):
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            candidate = resident.encode_raw(prompt, native)
            seconds = time.perf_counter() - start
            resident_rows.append(
                {
                    "prompt_index": prompt_index,
                    "seconds": seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                    / (1024.0**2),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                    / (1024.0**2),
                }
            )
            references.append(
                (
                    clone_cpu(candidate[0]),
                    clone_cpu(candidate[1]) if candidate[1] is not None else None,
                )
            )
        resident.close()

        exact_rows: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            reference_candidate = reload_policy.encode_raw(prompt, native)
            seconds = time.perf_counter() - start
            positive_equal, negative_equal = compare_raw_embeddings(
                torch,
                reference_candidate,
                references[prompt_index],
            )
            exact_rows.append(
                {
                    "prompt_index": prompt_index,
                    "positive_equal": positive_equal,
                    "negative_equal": negative_equal,
                }
            )
            reload_rows.append(
                {
                    "prompt_index": prompt_index,
                    "seconds": seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                    / (1024.0**2),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                    / (1024.0**2),
                }
            )
    finally:
        resident.close()
        reload_policy.close()

    resident_median = median(resident_rows, "seconds")
    reload_median = median(reload_rows, "seconds")
    exact = all(row["positive_equal"] for row in exact_rows) and all(
        row["negative_equal"] in (None, True) for row in exact_rows
    )
    return {
        "profile": profile,
        "resident_rows": resident_rows,
        "reload_rows": reload_rows,
        "exact_rows": exact_rows,
        "resident_policy": resident.stats(),
        "reload_policy": reload_policy.stats(),
        "median_resident_seconds": resident_median,
        "median_reload_seconds": reload_median,
        "median_saving_seconds": reload_median - resident_median,
        "exact": exact,
    }


def execute_request(
    base_config: dict[str, Any],
    spec: baseline.RunSpec,
    network: Any,
    tokenizer: Any,
    runtime: dict[str, Any],
    device: Any,
    text_policy: TextEncoderPolicy,
    output_path: Path | None,
) -> tuple[dict[str, Any], Any]:
    torch = runtime["torch"]
    baseline.synchronize(torch, device)
    total_start = time.perf_counter()

    text_start = time.perf_counter()
    condition, uncondition = text_policy.encode_condition(
        spec.prompt,
        need_negative(base_config, spec.method),
    )
    text_seconds = time.perf_counter() - text_start

    noise, generator = baseline.make_initial_noise(
        base_config, spec, tokenizer, runtime, device
    )
    baseline.synchronize(torch, device)
    denoiser_start = time.perf_counter()
    if need_negative(base_config, spec.method):
        if uncondition is None:
            raise RuntimeError("native UniPC requires the frozen negative condition")
        samples, forward_calls = baseline.denoise_native(
            base_config,
            spec.method,
            network,
            noise,
            generator,
            condition,
            uncondition,
            runtime,
            device,
        )
    else:
        samples, forward_calls = baseline.denoise_rcm(
            base_config,
            network,
            noise,
            generator,
            condition,
            runtime,
            device,
        )
    denoiser_seconds = time.perf_counter() - denoiser_start

    vae_start = time.perf_counter()
    video = tokenizer.decode(samples)
    baseline.synchronize(torch, device)
    vae_seconds = time.perf_counter() - vae_start

    transfer_start = time.perf_counter()
    video = ((1.0 + video.float().clamp(-1, 1)) / 2.0)[0].cpu()
    cpu_transfer_seconds = time.perf_counter() - transfer_start

    serialization_seconds = 0.0
    if output_path is not None:
        serialization_start = time.perf_counter()
        runtime["save_image_or_video"](
            video,
            str(output_path),
            fps=int(base_config["generation"]["fps"]),
        )
        serialization_seconds = time.perf_counter() - serialization_start

    request_seconds = time.perf_counter() - total_start
    return (
        {
            "text_seconds": text_seconds,
            "denoiser_seconds": denoiser_seconds,
            "vae_seconds": vae_seconds,
            "cpu_transfer_seconds": cpu_transfer_seconds,
            "serialization_seconds": serialization_seconds,
            "request_seconds": request_seconds,
            "sampling_steps": int(base_config["methods"][spec.method]["num_steps"]),
            "network_forward_calls": forward_calls,
            "denoiser_seconds_per_forward": denoiser_seconds / forward_calls,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
            / (1024.0**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
            / (1024.0**2),
        },
        video,
    )


def make_spec(
    base_config: dict[str, Any],
    method: str,
    prompt: str,
    prompt_index: int,
    seed: int,
    num_frames: int,
    output_dir: Path,
) -> baseline.RunSpec:
    return baseline.RunSpec(
        method=method,
        stage="formal" if num_frames == 81 else "f17-smoke",
        prompt=prompt,
        prompt_index=prompt_index,
        seed=seed,
        num_frames=num_frames,
        warmups=0,
        repeats=1,
        output_dir=output_dir,
    )


def setup_runtime(
    base_config: dict[str, Any], device_name: str
) -> tuple[dict[str, Any], Any, str]:
    source_commit = baseline.verify_source(base_config)
    baseline.configure_offline_model_cache(base_config)
    runtime = baseline.import_runtime(base_config)
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-052 requires one visible CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"EXP-052 requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    return runtime, device, source_commit


def run_text_screen(
    config: dict[str, Any],
    base_config: dict[str, Any],
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    prompts = baseline.load_formal_prompts(base_config)
    profiles = [
        run_text_profile(base_config, prompts, runtime, device, "positive_only"),
        run_text_profile(base_config, prompts, runtime, device, "native_cfg"),
    ]
    min_saving = min(float(row["median_saving_seconds"]) for row in profiles)
    exact = all(bool(row["exact"]) for row in profiles)
    return {
        "profiles": profiles,
        "minimum_median_saving_seconds": min_saving,
        "required_saving_seconds": float(config["text_screen_min_saving_seconds"]),
        "exact": exact,
        "advance": exact
        and min_saving >= float(config["text_screen_min_saving_seconds"]),
    }


def run_f17_exact(
    base_config: dict[str, Any],
    method: str,
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(base_config, method)
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, method, runtime, device
    )
    spec = make_spec(
        base_config,
        method,
        base_config["generation"]["smoke_prompt"],
        -1,
        int(base_config["generation"]["smoke_seed"]),
        17,
        output_dir,
    )
    reload_policy = TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=False,
        cache_negative=False,
    )
    resident = TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=need_negative(base_config, method),
    )
    try:
        torch.cuda.reset_peak_memory_stats(device)
        reload_row, reload_video = execute_request(
            base_config,
            spec,
            network,
            tokenizer,
            runtime,
            device,
            reload_policy,
            None,
        )
        resident.warm(
            baseline.load_formal_prompts(base_config)[0],
            need_negative(base_config, method),
        )
        torch.cuda.reset_peak_memory_stats(device)
        resident_row, resident_video = execute_request(
            base_config,
            spec,
            network,
            tokenizer,
            runtime,
            device,
            resident,
            output_dir / f"{method}_resident_f17.mp4",
        )
        video_equal = bool(torch.equal(reload_video, resident_video))
    finally:
        reload_policy.close()
        resident.close()
    return {
        "method": method,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "reload_row": reload_row,
        "resident_row": resident_row,
        "reload_policy": reload_policy.stats(),
        "resident_policy": resident.stats(),
        "video_equal": video_equal,
        "network_calls_equal": reload_row["network_forward_calls"]
        == resident_row["network_forward_calls"],
        "advance": video_equal
        and reload_row["network_forward_calls"]
        == resident_row["network_forward_calls"],
    }


def timing_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "text_seconds",
        "denoiser_seconds",
        "vae_seconds",
        "cpu_transfer_seconds",
        "serialization_seconds",
        "request_seconds",
        "denoiser_seconds_per_forward",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    return {
        f"median_{field}": median(rows, field)
        for field in fields
    }


def run_f81_resident(
    config: dict[str, Any],
    base_config: dict[str, Any],
    method: str,
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(base_config, method)
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, method, runtime, device
    )
    prompts = baseline.load_formal_prompts(base_config)
    resident = TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=need_negative(base_config, method),
    )
    rows: list[dict[str, Any]] = []
    try:
        resident.warm(
            base_config["generation"]["smoke_prompt"],
            need_negative(base_config, method),
        )
        for prompt_index in config["timing_prompt_indices"]:
            prompt = prompts[prompt_index]
            spec = make_spec(
                base_config,
                method,
                prompt,
                int(prompt_index),
                int(config["timing_seed"]),
                int(base_config["generation"]["num_frames"]),
                output_dir,
            )
            torch.manual_seed(spec.seed)
            torch.cuda.manual_seed_all(spec.seed)
            torch.cuda.reset_peak_memory_stats(device)
            filename = f"p{prompt_index:02d}_{method}_resident_seed{spec.seed}.mp4"
            row, _video = execute_request(
                base_config,
                spec,
                network,
                tokenizer,
                runtime,
                device,
                resident,
                output_dir / filename,
            )
            row.update(
                {
                    "method": method,
                    "prompt_index": int(prompt_index),
                    "prompt": prompt,
                    "seed": spec.seed,
                    "video_file": filename,
                    "status": "ok",
                }
            )
            rows.append(row)
            atomic_write_json(output_dir / "rows.partial.json", rows)
            torch.cuda.empty_cache()
    finally:
        resident.close()
    return {
        "method": method,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "rows": rows,
        "summary": timing_summary(rows),
        "text_policy": resident.stats(),
    }


def main() -> None:
    args = parse_args()
    config, base_config = load_configs(args.config.resolve())
    if args.stage == "text-screen" and args.method is not None:
        raise ValueError("text-screen evaluates both registered text profiles")
    if args.stage != "text-screen" and args.method is None:
        raise ValueError(f"{args.stage} requires --method")

    output_dir = resolve_output_dir(config, args.stage, args.method, args.output_dir)
    require_fresh_output_dir(output_dir)
    log = JsonlEventLog(output_dir / "events.jsonl", f"EXP-052-{args.stage}")
    log.emit("run_start", stage=args.stage, method=args.method)

    runtime, device, source_commit = setup_runtime(base_config, args.device)
    if args.stage == "text-screen":
        result = run_text_screen(config, base_config, runtime, device)
    elif args.stage == "f17-exact":
        if args.method is None:
            raise RuntimeError("f17-exact method validation was bypassed")
        result = run_f17_exact(
            base_config, args.method, output_dir, runtime, device
        )
    else:
        if args.method is None:
            raise RuntimeError("f81-resident method validation was bypassed")
        result = run_f81_resident(
            config,
            base_config,
            args.method,
            output_dir,
            runtime,
            device,
        )

    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "stage": args.stage,
        "method": args.method,
        "source_commit": source_commit,
        "positive_prompt_cache": config["positive_prompt_cache"],
        "result": result,
        "environment": runtime_environment(runtime["torch"], device),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "stage": args.stage, "method": args.method},
    )
    log.emit("run_complete", stage=args.stage, method=args.method)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
