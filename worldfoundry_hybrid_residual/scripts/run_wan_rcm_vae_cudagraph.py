#!/usr/bin/env python3
"""Run the frozen EXP-055 exact Wan VAE CUDA Graph stages."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import run_wan_rcm_baseline as baseline
import run_wan_rcm_exact_runtime as exact_runtime
import run_wan_rcm_vae_schedule as vae_schedule
from experiment_artifacts import JsonlEventLog, atomic_write_json, require_fresh_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("f17-screen", "f81-component", "f81-request")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--f17-manifest", type=Path)
    parser.add_argument("--f81-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-055" or config["gate_id"] != "G-034":
        raise ValueError("config is not the frozen EXP-055/G-034 configuration")
    if tuple(config["methods"]) != ("rcm4",):
        raise ValueError("EXP-055 permits only the exact resident rCM4 incumbent")
    if tuple(int(value) for value in config["f17_seeds"]) != (
        2026082700,
        2026082703,
        2026082704,
    ):
        raise ValueError("EXP-055 requires the three frozen F17 seeds")
    if tuple(int(value) for value in config["f17_replay_order"]) != (0, 1, 2, 1, 0):
        raise ValueError("EXP-055 requires replay order 0,1,2,1,0")
    round_orders = tuple(
        tuple(int(index) for index in order) for order in config["f81_round_orders"]
    )
    if round_orders != ((0, 1, 2, 3), (3, 2, 1, 0)):
        raise ValueError("EXP-055 requires two frozen alternating F81 rounds")
    if tuple(int(value) for value in config["timing_prompt_indices"]) != (0, 1, 2, 3):
        raise ValueError("EXP-055 requires all four frozen EXP-052 prompts")
    if int(config["graph_warmups"]) != 2:
        raise ValueError("EXP-055 requires two side-stream graph warmups")

    base_path = (PROJECT_ROOT / config["base_config"]).resolve()
    base_config = baseline.load_config(base_path)
    if int(config["timing_seed"]) not in base_config["generation"]["formal_seeds"]:
        raise ValueError("EXP-055 timing seed must remain inside EXP-047")
    return config, base_config


class ExactVaeCudaGraph:
    """Capture one fixed-shape official VAE decode and return owned outputs."""

    def __init__(
        self,
        tokenizer: Any,
        example: Any,
        torch: Any,
        device: Any,
        warmups: int,
    ) -> None:
        if warmups < 1:
            raise ValueError("CUDA Graph capture requires at least one warmup")
        if example.device.type != "cuda":
            raise ValueError("CUDA Graph input must already reside on CUDA")

        self.torch = torch
        self.device = device
        self.static_input = torch.empty_like(example)
        self.static_input.copy_(example)
        baseline.synchronize(torch, device)

        current_stream = torch.cuda.current_stream(device)
        self.capture_stream = torch.cuda.Stream(device=device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for _ in range(warmups):
                warm_output = tokenizer.decode(self.static_input)
                del warm_output
        self.capture_stream.synchronize()

        # Python cache lists are rebuilt during capture. Replays use only the
        # captured tensor dependencies, so stale-state tests remain mandatory.
        tokenizer.model.model.clear_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.static_output = tokenizer.decode(self.static_input)
        current_stream.wait_stream(self.capture_stream)
        baseline.synchronize(torch, device)
        tokenizer.model.model.clear_cache()

    def decode(self, latent: Any) -> Any:
        if tuple(latent.shape) != tuple(self.static_input.shape):
            raise ValueError(
                f"graph shape mismatch: {tuple(latent.shape)} != "
                f"{tuple(self.static_input.shape)}"
            )
        if latent.dtype != self.static_input.dtype:
            raise ValueError(f"graph dtype mismatch: {latent.dtype} != {self.static_input.dtype}")
        if latent.device != self.static_input.device:
            raise ValueError(
                f"graph device mismatch: {latent.device} != {self.static_input.device}"
            )
        self.static_input.copy_(latent)
        self.graph.replay()
        output = self.static_output.clone()
        if output.data_ptr() == self.static_output.data_ptr():
            raise RuntimeError("CUDA Graph output handoff aliases graph-owned storage")
        return output


def timed_decode(
    torch: Any,
    device: Any,
    decode: Callable[[], Any],
) -> tuple[Any, float, float]:
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = decode()
    end.record()
    end.synchronize()
    return (
        output,
        float(start.elapsed_time(end)) / 1000.0,
        float(torch.cuda.max_memory_reserved(device)) / (1024.0**2),
    )


def normalized_cpu(torch: Any, video: Any) -> Any:
    return ((1.0 + video.float().clamp(-1, 1)) / 2.0)[0].cpu()


def compare_outputs(torch: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    comparison = vae_schedule.tensor_comparison(torch, reference, candidate)
    comparison.update(
        {
            "finite": bool(torch.isfinite(candidate).all().item()),
            "dtype_equal": candidate.dtype == reference.dtype,
            "device_equal": candidate.device == reference.device,
            "cpu_raw_equal": bool(torch.equal(reference.cpu(), candidate.cpu())),
            "cpu_decoded_equal": bool(
                torch.equal(normalized_cpu(torch, reference), normalized_cpu(torch, candidate))
            ),
        }
    )
    return comparison


def comparison_passes(row: dict[str, Any]) -> bool:
    return all(
        bool(row[field])
        for field in (
            "bitwise_equal",
            "finite",
            "dtype_equal",
            "device_equal",
            "cpu_raw_equal",
            "cpu_decoded_equal",
        )
    )


def component_decision(
    exact: bool,
    memory_ok: bool,
    vae_speedup: float,
    projected_speedup: float,
    min_vae_speedup: float,
    min_projected_speedup: float,
) -> tuple[bool, str]:
    if not exact:
        return False, "exactness-null"
    if not memory_ok:
        return False, "invalid-memory"
    if vae_speedup < min_vae_speedup or projected_speedup < min_projected_speedup:
        return False, "performance-null"
    return True, "advance"


def request_decision(
    exact: bool,
    memory_ok: bool,
    median_request_seconds: float,
    max_request_seconds: float,
) -> tuple[bool, str]:
    if not exact:
        return False, "exactness-null"
    if not memory_ok:
        return False, "invalid-memory"
    if median_request_seconds > max_request_seconds:
        return False, "speed-boundary"
    return True, "pass"


def validate_prerequisite(path: Path, required_stage: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["experiment_id"] != "EXP-055" or manifest["gate_id"] != "G-034":
        raise ValueError("prerequisite manifest belongs to another experiment")
    if manifest["stage"] != required_stage:
        raise ValueError(f"expected prerequisite stage {required_stage}")
    if manifest["result"]["advance"] is not True:
        raise ValueError(f"prerequisite stage {required_stage} did not advance")
    return manifest


def load_rcm_latents(
    config: dict[str, Any],
    base_config: dict[str, Any],
    prompts: tuple[str, ...],
    prompt_indices: tuple[int, ...],
    seeds: tuple[int, ...],
    num_frames: int,
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> tuple[Any, list[Any], list[dict[str, int]], dict[str, Any], str, str]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(base_config, "rcm4")
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, "rcm4", runtime, device
    )
    text_policy = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    latents: list[Any] = []
    latent_rows: list[dict[str, int]] = []
    try:
        text_policy.warm(base_config["generation"]["smoke_prompt"], False)
        for prompt, prompt_index, seed in zip(prompts, prompt_indices, seeds):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            latent, latent_info = vae_schedule.make_latent(
                base_config,
                network,
                tokenizer,
                runtime,
                device,
                text_policy,
                prompt,
                prompt_index,
                seed,
                num_frames,
                output_dir,
            )
            latents.append(latent)
            latent_rows.append(latent_info)
    finally:
        text_policy.close()
    del text_policy
    del network
    gc.collect()
    torch.cuda.empty_cache()
    return (
        tokenizer,
        latents,
        latent_rows,
        load_info,
        str(checkpoint_path),
        checkpoint_identity,
    )


def run_f17_screen(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    seeds = tuple(int(seed) for seed in config["f17_seeds"])
    prompts = tuple(base_config["generation"]["smoke_prompt"] for _ in seeds)
    prompt_indices = tuple(-1 for _ in seeds)
    tokenizer, latents, latent_rows, load_info, checkpoint_path, checkpoint_identity = (
        load_rcm_latents(
            config,
            base_config,
            prompts,
            prompt_indices,
            seeds,
            17,
            output_dir,
            runtime,
            device,
        )
    )

    references = []
    for latent in latents:
        reference = tokenizer.decode(latent)
        baseline.synchronize(torch, device)
        references.append(reference)

    torch.cuda.reset_peak_memory_stats(device)
    graph = ExactVaeCudaGraph(
        tokenizer, latents[0], torch, device, int(config["graph_warmups"])
    )
    capture_peak = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
    first_outputs: dict[int, Any] = {}
    replay_rows = []
    for position, latent_index in enumerate(config["f17_replay_order"]):
        candidate, seconds, peak = timed_decode(
            torch, device, lambda index=latent_index: graph.decode(latents[index])
        )
        comparison = compare_outputs(torch, references[latent_index], candidate)
        candidate_cpu = candidate.cpu()
        repeat_equal = (
            True
            if latent_index not in first_outputs
            else bool(torch.equal(first_outputs[latent_index], candidate_cpu))
        )
        if latent_index not in first_outputs:
            first_outputs[latent_index] = candidate_cpu.clone()
        replay_rows.append(
            {
                "position": position,
                "latent_index": int(latent_index),
                "seed": seeds[latent_index],
                "seconds": seconds,
                "peak_reserved_mib": peak,
                "repeat_equal": repeat_equal,
                "comparison": comparison,
            }
        )
        del candidate

    eager_after_rows = []
    for latent_index, latent in enumerate(latents):
        eager_after = tokenizer.decode(latent)
        baseline.synchronize(torch, device)
        eager_after_rows.append(
            {
                "latent_index": latent_index,
                "comparison": compare_outputs(
                    torch, references[latent_index], eager_after
                ),
            }
        )
        del eager_after

    exact = all(comparison_passes(row["comparison"]) for row in replay_rows)
    stale_free = all(bool(row["repeat_equal"]) for row in replay_rows) and all(
        comparison_passes(row["comparison"]) for row in eager_after_rows
    )
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "latents": latent_rows,
        "capture_peak_reserved_mib": capture_peak,
        "replay_rows": replay_rows,
        "eager_after_rows": eager_after_rows,
        "bitwise_exact": exact,
        "stale_state_free": stale_free,
        "outcome": "advance" if exact and stale_free else "exactness-null",
        "advance": exact and stale_free,
    }


def run_f81_component(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    all_prompts = baseline.load_formal_prompts(base_config)
    prompt_indices = tuple(int(index) for index in config["timing_prompt_indices"])
    prompts = tuple(all_prompts[index] for index in prompt_indices)
    seed = int(config["timing_seed"])
    seeds = tuple(seed for _ in prompt_indices)
    tokenizer, latents, latent_rows, load_info, checkpoint_path, checkpoint_identity = (
        load_rcm_latents(
            config,
            base_config,
            prompts,
            prompt_indices,
            seeds,
            int(base_config["generation"]["num_frames"]),
            output_dir,
            runtime,
            device,
        )
    )

    torch.cuda.reset_peak_memory_stats(device)
    graph = ExactVaeCudaGraph(
        tokenizer, latents[0], torch, device, int(config["graph_warmups"])
    )
    capture_peak = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
    first_outputs: dict[int, Any] = {}
    rows = []
    for round_index, order in enumerate(config["f81_round_orders"]):
        for position, latent_index in enumerate(order):
            reference, eager_seconds, eager_peak = timed_decode(
                torch,
                device,
                lambda index=latent_index: tokenizer.decode(latents[index]),
            )
            candidate, graph_seconds, graph_peak = timed_decode(
                torch, device, lambda index=latent_index: graph.decode(latents[index])
            )
            comparison = compare_outputs(torch, reference, candidate)
            candidate_cpu = candidate.cpu()
            repeat_equal = (
                True
                if latent_index not in first_outputs
                else bool(torch.equal(first_outputs[latent_index], candidate_cpu))
            )
            if latent_index not in first_outputs:
                first_outputs[latent_index] = candidate_cpu.clone()
            rows.append(
                {
                    "round": round_index,
                    "position": position,
                    "prompt_index": prompt_indices[latent_index],
                    "latent_index": int(latent_index),
                    "eager_seconds": eager_seconds,
                    "graph_seconds": graph_seconds,
                    "speedup": eager_seconds / graph_seconds,
                    "eager_peak_reserved_mib": eager_peak,
                    "graph_peak_reserved_mib": graph_peak,
                    "repeat_equal": repeat_equal,
                    "comparison": comparison,
                }
            )
            atomic_write_json(output_dir / "rows.partial.json", rows)
            del reference
            del candidate

    exact = all(comparison_passes(row["comparison"]) for row in rows) and all(
        bool(row["repeat_equal"]) for row in rows
    )
    eager_median = float(statistics.median(row["eager_seconds"] for row in rows))
    graph_median = float(statistics.median(row["graph_seconds"] for row in rows))
    vae_speedup = eager_median / graph_median
    projected_seconds = (
        float(config["incumbent_warm_seconds"])
        - float(config["incumbent_vae_seconds"])
        + graph_median
    )
    projected_speedup = float(config["incumbent_warm_seconds"]) / projected_seconds
    peak_reserved = max(
        [capture_peak]
        + [float(row["eager_peak_reserved_mib"]) for row in rows]
        + [float(row["graph_peak_reserved_mib"]) for row in rows]
    )
    memory_ok = peak_reserved <= float(config["max_peak_reserved_mib"])
    advance, outcome = component_decision(
        exact,
        memory_ok,
        vae_speedup,
        projected_speedup,
        float(config["min_vae_speedup"]),
        float(config["min_projected_warm_speedup"]),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "latents": latent_rows,
        "rows": rows,
        "summary": {
            "bitwise_exact": exact,
            "baseline_vae_median_seconds": eager_median,
            "graph_vae_median_seconds": graph_median,
            "vae_speedup": vae_speedup,
            "projected_request_seconds": projected_seconds,
            "projected_request_speedup": projected_speedup,
            "peak_reserved_mib": peak_reserved,
            "memory_ok": memory_ok,
        },
        "outcome": outcome,
        "advance": advance,
    }


def run_f81_request(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(base_config, "rcm4")
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, "rcm4", runtime, device
    )
    prompts = baseline.load_formal_prompts(base_config)
    resident = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    rows = []
    try:
        resident.warm(base_config["generation"]["smoke_prompt"], False)
        capture_seed = int(config["timing_seed"])
        torch.manual_seed(capture_seed)
        torch.cuda.manual_seed_all(capture_seed)
        capture_latent, capture_info = vae_schedule.make_latent(
            base_config,
            network,
            tokenizer,
            runtime,
            device,
            resident,
            prompts[int(config["timing_prompt_indices"][0])],
            int(config["timing_prompt_indices"][0]),
            capture_seed,
            int(base_config["generation"]["num_frames"]),
            output_dir,
        )
        torch.cuda.reset_peak_memory_stats(device)
        graph = ExactVaeCudaGraph(
            tokenizer,
            capture_latent,
            torch,
            device,
            int(config["graph_warmups"]),
        )
        capture_peak = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
        del capture_latent

        original_decode = tokenizer.decode
        for prompt_index in config["timing_prompt_indices"]:
            prompt_index = int(prompt_index)
            spec = exact_runtime.make_spec(
                base_config,
                "rcm4",
                prompts[prompt_index],
                prompt_index,
                int(config["timing_seed"]),
                int(base_config["generation"]["num_frames"]),
                output_dir,
            )
            torch.manual_seed(spec.seed)
            torch.cuda.manual_seed_all(spec.seed)
            torch.cuda.reset_peak_memory_stats(device)
            eager_row, eager_video = exact_runtime.execute_request(
                base_config,
                spec,
                network,
                tokenizer,
                runtime,
                device,
                resident,
                output_dir / f"p{prompt_index:02d}_eager_seed{spec.seed}.mp4",
            )

            torch.manual_seed(spec.seed)
            torch.cuda.manual_seed_all(spec.seed)
            torch.cuda.reset_peak_memory_stats(device)
            tokenizer.decode = graph.decode
            try:
                graph_row, graph_video = exact_runtime.execute_request(
                    base_config,
                    spec,
                    network,
                    tokenizer,
                    runtime,
                    device,
                    resident,
                    output_dir / f"p{prompt_index:02d}_graph_seed{spec.seed}.mp4",
                )
            finally:
                tokenizer.decode = original_decode
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompts[prompt_index],
                    "seed": spec.seed,
                    "cpu_video_equal": bool(torch.equal(eager_video, graph_video)),
                    "network_calls_equal": eager_row["network_forward_calls"]
                    == graph_row["network_forward_calls"]
                    == 4,
                    "eager": eager_row,
                    "graph": graph_row,
                }
            )
            atomic_write_json(output_dir / "rows.partial.json", rows)
    finally:
        resident.close()

    exact = all(bool(row["cpu_video_equal"]) for row in rows) and all(
        bool(row["network_calls_equal"]) for row in rows
    )
    median_request = float(
        statistics.median(row["graph"]["request_seconds"] for row in rows)
    )
    peak_reserved = max(
        [capture_peak]
        + [float(row["graph"]["peak_reserved_mib"]) for row in rows]
        + [float(row["eager"]["peak_reserved_mib"]) for row in rows]
    )
    memory_ok = peak_reserved <= float(config["max_peak_reserved_mib"])
    advance, outcome = request_decision(
        exact,
        memory_ok,
        median_request,
        float(config["max_request_seconds"]),
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "capture_latent": capture_info,
        "capture_peak_reserved_mib": capture_peak,
        "rows": rows,
        "summary": {
            "bitwise_exact": exact,
            "median_graph_request_seconds": median_request,
            "request_speedup_over_incumbent": float(config["incumbent_warm_seconds"])
            / median_request,
            "peak_reserved_mib": peak_reserved,
            "memory_ok": memory_ok,
        },
        "text_policy": resident.stats(),
        "outcome": outcome,
        "advance": advance,
    }


def resolve_output_dir(
    config: dict[str, Any], stage: str, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    return (Path(config["remote_output_root"]) / stage).resolve()


def main() -> None:
    args = parse_args()
    config, base_config = load_configs(args.config.resolve())
    if args.stage == "f17-screen":
        if args.f17_manifest is not None or args.f81_manifest is not None:
            raise ValueError("F17 cannot consume a later-stage manifest")
    elif args.stage == "f81-component":
        if args.f17_manifest is None or args.f81_manifest is not None:
            raise ValueError("F81 component requires only --f17-manifest")
        validate_prerequisite(args.f17_manifest.resolve(), "f17-screen")
    else:
        if args.f17_manifest is not None or args.f81_manifest is None:
            raise ValueError("F81 request requires only --f81-manifest")
        validate_prerequisite(args.f81_manifest.resolve(), "f81-component")

    output_dir = resolve_output_dir(config, args.stage, args.output_dir)
    require_fresh_output_dir(output_dir)
    log = JsonlEventLog(output_dir / "events.jsonl", f"EXP-055-{args.stage}")
    log.emit("run_start", stage=args.stage)
    runtime, device, source_commit = exact_runtime.setup_runtime(base_config, args.device)
    if args.stage == "f17-screen":
        result = run_f17_screen(config, base_config, output_dir, runtime, device)
    elif args.stage == "f81-component":
        result = run_f81_component(config, base_config, output_dir, runtime, device)
    else:
        result = run_f81_request(config, base_config, output_dir, runtime, device)

    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "stage": args.stage,
        "source_commit": source_commit,
        "result": result,
        "environment": exact_runtime.runtime_environment(runtime["torch"], device),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "stage": args.stage, "outcome": result["outcome"]},
    )
    log.emit("run_complete", stage=args.stage, outcome=result["outcome"])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
