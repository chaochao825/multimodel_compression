#!/usr/bin/env python3
"""Capture the EXP-048 teacher/rCM weight-by-trajectory closure cross."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_json,
    file_sha256,
    require_fresh_output_dir,
)
from rcm_state_closure_core import (
    ClosureTrajectory,
    capacity_error_terms,
    orthogonal_error_terms,
    project_trajectory,
    rollout_coordinates,
)
from run_wan_rcm_baseline import (
    NEGATIVE_PROMPT,
    import_runtime,
    normalize_state_dict_keys,
    remove_rcm_training_metadata,
)
from wan_rcm_state_closure_runtime import WanBlockSequenceRecorder


MODELS = ("teacher", "rcm")
TRAJECTORIES = ("native4", "rcm4")
BASIS_SCOPES = ("model_specific", "shared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "selection"), required=True)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--closure-model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    return parser.parse_args()


def parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("sample-indices must be a nonempty unique comma list")
    return indices


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-048" or config["gate_id"] != "G-027":
        raise ValueError("config is not the frozen EXP-048/G-027 configuration")
    if tuple(config["models"]) != MODELS:
        raise ValueError(f"model order must remain frozen as {MODELS}")
    if tuple(config["trajectories"]) != TRAJECTORIES:
        raise ValueError(f"trajectory order must remain frozen as {TRAJECTORIES}")
    return config


def validate_indices(
    config: dict[str, Any],
    split: str,
    indices: tuple[int, ...],
    *,
    engineering_smoke: bool,
) -> list[dict[str, Any]]:
    expected = tuple(int(value) for value in config["splits"][split])
    if engineering_smoke:
        if split != "calibration" or indices != (expected[0],):
            raise ValueError(
                "engineering smoke is restricted to the first calibration identity"
            )
    elif indices != expected:
        raise ValueError(
            f"{split} run must use the frozen ordered identities {expected}, got {indices}"
        )
    identities = config["identities"]
    selected = [identities[index] for index in indices]
    if any(identity["split"] != split for identity in selected):
        raise ValueError("identity split labels disagree with the frozen split")
    return selected


def verify_source_and_checkpoints(config: dict[str, Any]) -> dict[str, str]:
    source_root = Path(config["remote"]["rcm_root"]).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["source"]["commit"]:
        raise RuntimeError(
            f"rCM source mismatch: expected {config['source']['commit']}, got {commit}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"rCM source is dirty:\n{status}")
    identities = {"source_commit": commit}
    for model in MODELS:
        checkpoint = Path(config["remote"][f"{model}_checkpoint"]).resolve()
        actual = file_sha256(checkpoint)
        expected = config["source"][f"{model}_checkpoint_sha256"]
        if actual != expected:
            raise RuntimeError(
                f"{model} checkpoint mismatch: expected {expected}, got {actual}"
            )
        identities[f"{model}_checkpoint_sha256"] = actual
    return identities


def load_network(
    config: dict[str, Any], model_name: str, runtime: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    with runtime["init_weights"]():
        network = runtime["instantiate"](
            runtime["dit_configs"][config["generation"]["model_size"]]
        ).eval()
    checkpoint = config["remote"][f"{model_name}_checkpoint"]
    if model_name == "teacher":
        state_dict = runtime["load_safetensors"](checkpoint, device="cpu")
        state_dict["patch_embedding.weight"] = state_dict[
            "patch_embedding.weight"
        ].flatten(1)
    else:
        state_dict = runtime["load_state_dict"](checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    if model_name == "rcm":
        state_dict = remove_rcm_training_metadata(state_dict)
    incompatible = network.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{model_name} checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    return network.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)


def encode_prompts(
    config: dict[str, Any], identities: list[dict[str, Any]], device: torch.device
) -> tuple[list[torch.Tensor], torch.Tensor]:
    source_root = str(Path(config["remote"]["rcm_root"]).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from rcm.utils.umt5 import UMT5EncoderModel

    encoder = UMT5EncoderModel(
        text_len=int(config["generation"]["text_length"]),
        device=device,
        checkpoint_path=config["remote"]["text_encoder"],
        tokenizer_path=config["remote"]["tokenizer"],
    )
    texts = [identity["prompt"] for identity in identities] + [NEGATIVE_PROMPT]
    with torch.inference_mode():
        embeddings = encoder(texts, device=device).to(
            device="cpu", dtype=torch.bfloat16
        )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return [embeddings[index : index + 1] for index in range(len(identities))], embeddings[-1:]


def initial_noise(
    config: dict[str, Any], seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Generator]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = tuple(int(value) for value in config["generation"]["latent_shape"])
    noise = torch.randn(
        1,
        *shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return noise, generator


@torch.inference_mode()
def native4_trajectory(
    config: dict[str, Any],
    network: torch.nn.Module,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    seed: int,
    runtime: dict[str, Any],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    noise, _ = initial_noise(config, seed, device)
    method = config["methods"]["native4"]
    sigma_max = 5000.0 / 5001.0
    shift = float(method["timestep_shift"])
    unshifted_sigma_max = sigma_max / (shift - (shift - 1.0) * sigma_max)
    sampler = runtime["unipc"](
        num_train_timesteps=1000,
        sigma_max=unshifted_sigma_max,
        sigma_min=0.0,
    )
    sampler.set_timesteps(
        num_inference_steps=int(method["num_steps"]), device=device, shift=shift
    )
    x = noise.to(torch.float64)
    ones = torch.ones(1, 1, device=device, dtype=torch.float64)
    states: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    for timestep in sampler.timesteps:
        model_input = x.to(dtype=torch.bfloat16)
        time_input = (timestep * ones).to(dtype=torch.bfloat16)
        states.append(model_input.clone())
        times.append(time_input.clone())
        conditional = network(
            x_B_C_T_H_W=model_input,
            timesteps_B_T=time_input,
            crossattn_emb=condition,
        ).float()
        unconditional = network(
            x_B_C_T_H_W=model_input,
            timesteps_B_T=time_input,
            crossattn_emb=uncondition,
        ).float()
        prediction = unconditional + float(method["guidance_scale"]) * (
            conditional - unconditional
        )
        x = sampler.step(prediction, timestep, x)
    return states, times


@torch.inference_mode()
def rcm4_trajectory(
    config: dict[str, Any],
    network: torch.nn.Module,
    condition: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    noise, generator = initial_noise(config, seed, device)
    method = config["methods"]["rcm4"]
    steps = int(method["num_steps"])
    middle = [float(value) for value in method["middle_sigmas"]][: steps - 1]
    t_steps = torch.tensor(
        [math.atan(float(method["sigma_max"])), *middle, 0.0],
        dtype=torch.float64,
        device=device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    x = noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(1, 1, device=device, dtype=torch.float64)
    states: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    for current, following in zip(t_steps[:-1], t_steps[1:]):
        model_input = x.to(dtype=torch.bfloat16)
        time_input = (current.float() * ones * 1000).to(dtype=torch.bfloat16)
        states.append(model_input.clone())
        times.append(time_input.clone())
        prediction = network(
            x_B_C_T_H_W=model_input,
            timesteps_B_T=time_input,
            crossattn_emb=condition,
        ).to(torch.float64)
        x = (1 - following) * (x - current * prediction) + following * torch.randn(
            *x.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    return states, times


@torch.inference_mode()
def capture_combination(
    network: torch.nn.Module,
    recorder: WanBlockSequenceRecorder,
    states: list[torch.Tensor],
    times: list[torch.Tensor],
    condition: torch.Tensor,
    *,
    verify_equivalence: bool,
) -> dict[int, ClosureTrajectory]:
    reference = None
    if verify_equivalence:
        reference = network(
            x_B_C_T_H_W=states[0],
            timesteps_B_T=times[0],
            crossattn_emb=condition,
        )
    recorder.begin()
    observed = None
    for index, (state, timestep) in enumerate(zip(states, times)):
        output = network(
            x_B_C_T_H_W=state,
            timesteps_B_T=timestep,
            crossattn_emb=condition,
        )
        if index == 0 and verify_equivalence:
            observed = output
        del output
    if verify_equivalence:
        if reference is None or observed is None:
            raise RuntimeError("equivalence outputs were not produced")
        if not torch.equal(reference, observed):
            relative = float(
                (reference.float() - observed.float()).square().sum().sqrt()
                / reference.float().square().sum().sqrt().clamp_min(1e-30)
            )
            raise RuntimeError(f"capture changed Wan output: relative_l2={relative}")
    return recorder.end()


def serializable_capture(
    captures: dict[int, ClosureTrajectory]
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        block: {
            "block_input": trajectory.block_input,
            "residual": trajectory.residual,
        }
        for block, trajectory in captures.items()
    }


def sliced_transitions(
    transitions: dict[str, dict[int, torch.Tensor]], rank: int
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        method: {
            stage: coefficients[:, :rank]
            for stage, coefficients in stages.items()
        }
        for method, stages in transitions.items()
    }


def metric_row(
    *,
    sample: dict[str, Any],
    model: str,
    trajectory_name: str,
    basis_scope: str,
    block: int,
    rank: int,
    method: str,
    target_stage: int,
    horizon: int,
    error_sq: float,
    target_sq: float,
    output_target_sq: float,
    token_count: int,
    channel_count: int,
) -> dict[str, object]:
    state_bytes = token_count * rank * 2
    basis_bytes = channel_count * rank * 2
    if method == "capacity":
        estimated_macs = token_count * channel_count * rank
    else:
        estimated_macs = (
            (horizon + 1) * token_count * channel_count * rank
            + 5 * horizon * token_count * rank
        )
    return {
        "sample_id": sample["id"],
        "sample_index": sample["index"],
        "seed": sample["seed"],
        "model": model,
        "input_trajectory": trajectory_name,
        "basis_scope": basis_scope,
        "block": block,
        "rank": rank,
        "method": method,
        "target_stage": target_stage,
        "horizon": horizon,
        "relative_l2": (error_sq / max(target_sq, 1e-30)) ** 0.5,
        "output_relative_l2": (error_sq / max(output_target_sq, 1e-30)) ** 0.5,
        "error_sq": error_sq,
        "target_sq": target_sq,
        "output_target_sq": output_target_sq,
        "state_bytes_bf16": state_bytes,
        "basis_bytes_bf16": basis_bytes,
        "estimated_macs": estimated_macs,
    }


def evaluate_selection(
    sample: dict[str, Any],
    model: str,
    trajectory_name: str,
    captures: dict[int, ClosureTrajectory],
    closure_model: dict[str, Any],
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranks = tuple(int(value) for value in closure_model["ranks"])
    max_rank = int(closure_model["max_rank"])
    for block, trajectory in captures.items():
        for basis_scope in BASIS_SCOPES:
            basis = (
                closure_model["bases"]["model_specific"][model][block]
                if basis_scope == "model_specific"
                else closure_model["bases"]["shared"][block]
            )
            input_max, residual_max = project_trajectory(
                trajectory, basis, device=device
            )
            transition_max = closure_model["transitions"][basis_scope][model][
                trajectory_name
            ][block]
            output_target_sq = [
                float(
                    (
                        trajectory.block_input[stage].double()
                        + trajectory.residual[stage].double()
                    )
                    .square()
                    .sum()
                )
                for stage in range(trajectory.step_count)
            ]
            for rank in ranks:
                if rank > max_rank:
                    raise ValueError("closure rank exceeds the fitted maximum")
                input_coordinates = input_max[:, :, :rank]
                residual_coordinates = residual_max[:, :, :rank]
                transitions = sliced_transitions(transition_max, rank)
                for target_stage in range(1, trajectory.step_count):
                    capacity_error, target_sq = capacity_error_terms(
                        trajectory.residual[target_stage],
                        residual_coordinates[target_stage],
                    )
                    rows.append(
                        metric_row(
                            sample=sample,
                            model=model,
                            trajectory_name=trajectory_name,
                            basis_scope=basis_scope,
                            block=block,
                            rank=rank,
                            method="capacity",
                            target_stage=target_stage,
                            horizon=0,
                            error_sq=capacity_error,
                            target_sq=target_sq,
                            output_target_sq=output_target_sq[target_stage],
                            token_count=trajectory.token_count,
                            channel_count=trajectory.channel_count,
                        )
                    )
                    for horizon in range(1, target_stage + 1):
                        for method in ("reuse", "ar1", "drift"):
                            prediction = rollout_coordinates(
                                input_coordinates,
                                residual_coordinates,
                                transitions,
                                method=method,
                                target_stage=target_stage,
                                horizon=horizon,
                            )
                            error_sq, target_sq = orthogonal_error_terms(
                                trajectory.residual[target_stage],
                                residual_coordinates[target_stage],
                                prediction,
                            )
                            rows.append(
                                metric_row(
                                    sample=sample,
                                    model=model,
                                    trajectory_name=trajectory_name,
                                    basis_scope=basis_scope,
                                    block=block,
                                    rank=rank,
                                    method=method,
                                    target_stage=target_stage,
                                    horizon=horizon,
                                    error_sq=error_sq,
                                    target_sq=target_sq,
                                    output_target_sq=output_target_sq[target_stage],
                                    token_count=trajectory.token_count,
                                    channel_count=trajectory.channel_count,
                                )
                            )
                    if target_stage >= 2:
                        prediction = rollout_coordinates(
                            input_coordinates,
                            residual_coordinates,
                            transitions,
                            method="ar2_drift",
                            target_stage=target_stage,
                            horizon=1,
                        )
                        error_sq, target_sq = orthogonal_error_terms(
                            trajectory.residual[target_stage],
                            residual_coordinates[target_stage],
                            prediction,
                        )
                        rows.append(
                            metric_row(
                                sample=sample,
                                model=model,
                                trajectory_name=trajectory_name,
                                basis_scope=basis_scope,
                                block=block,
                                rank=rank,
                                method="ar2_drift",
                                target_stage=target_stage,
                                horizon=1,
                                error_sq=error_sq,
                                target_sq=target_sq,
                                output_target_sq=output_target_sq[target_stage],
                                token_count=trajectory.token_count,
                                channel_count=trajectory.channel_count,
                            )
                        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    indices = parse_indices(args.sample_indices)
    identities = validate_indices(
        config,
        args.split,
        indices,
        engineering_smoke=args.engineering_smoke,
    )
    if args.engineering_smoke and not args.verify_equivalence:
        raise ValueError("engineering smoke must enable --verify-equivalence")
    if args.split == "selection" and args.closure_model is None:
        raise ValueError("selection requires a frozen --closure-model")
    if args.split == "calibration" and args.closure_model is not None:
        raise ValueError("calibration must not read a closure model")
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    log = JsonlEventLog(output_dir / "events.jsonl", f"EXP-048-{args.split}")
    log.emit("run_start", split=args.split, sample_indices=list(indices))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    identities_verified = verify_source_and_checkpoints(config)
    runtime = import_runtime(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("EXP-048 requires exactly one visible CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    prompt_embeddings, negative_embedding = encode_prompts(config, identities, device)
    log.emit("prompts_encoded", count=len(prompt_embeddings))
    networks = {
        model: load_network(config, model, runtime, device) for model in MODELS
    }
    torch.cuda.synchronize(device)
    log.emit("networks_loaded")

    closure_model = None
    if args.closure_model is not None:
        closure_model = torch.load(
            args.closure_model.resolve(), map_location="cpu", weights_only=False
        )
        if closure_model["experiment_id"] != config["experiment_id"]:
            raise ValueError("closure model experiment identity mismatch")
        if tuple(closure_model["sample_indices"]) != tuple(
            config["splits"]["calibration"]
        ):
            raise ValueError("closure model was not fit on the frozen calibration split")

    samples_payload: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    blocks = tuple(int(value) for value in config["capture"]["blocks"])
    sampled_tokens = (
        int(config["capture"]["calibration_sampled_tokens"])
        if args.split == "calibration"
        else None
    )
    started = time.perf_counter()
    for identity, embedding_cpu in zip(identities, prompt_embeddings):
        condition = embedding_cpu.to(device=device, dtype=torch.bfloat16)
        uncondition = negative_embedding.to(device=device, dtype=torch.bfloat16)
        native_states, native_times = native4_trajectory(
            config,
            networks["teacher"],
            condition,
            uncondition,
            int(identity["seed"]),
            runtime,
            device,
        )
        rcm_states, rcm_times = rcm4_trajectory(
            config,
            networks["rcm"],
            condition,
            int(identity["seed"]),
            device,
        )
        trajectory_states = {
            "native4": (native_states, native_times),
            "rcm4": (rcm_states, rcm_times),
        }
        sample_record: dict[str, object] = {
            "id": identity["id"],
            "index": identity["index"],
            "seed": identity["seed"],
            "split": identity["split"],
            "prompt": identity["prompt"],
            "timesteps": {
                name: [float(value.item()) for value in values[1]]
                for name, values in trajectory_states.items()
            },
            "captures": {},
        }
        for model in MODELS:
            sample_record["captures"][model] = {}
            recorder = WanBlockSequenceRecorder(
                networks[model],
                blocks=blocks,
                expected_steps=int(config["methods"]["rcm4"]["num_steps"]),
                sampled_tokens=sampled_tokens,
            )
            try:
                for trajectory_name in TRAJECTORIES:
                    states, times = trajectory_states[trajectory_name]
                    captures = capture_combination(
                        networks[model],
                        recorder,
                        states,
                        times,
                        condition,
                        verify_equivalence=args.verify_equivalence,
                    )
                    if args.split == "calibration":
                        sample_record["captures"][model][trajectory_name] = (
                            serializable_capture(captures)
                        )
                    else:
                        if closure_model is None:
                            raise RuntimeError("selection closure model was not loaded")
                        metric_rows.extend(
                            evaluate_selection(
                                identity,
                                model,
                                trajectory_name,
                                captures,
                                closure_model,
                                device,
                            )
                        )
                        write_csv(output_dir / "cell_metrics.partial.csv", metric_rows)
                    del captures
                    gc.collect()
            finally:
                recorder.restore()
        if args.split == "calibration":
            samples_payload.append(sample_record)
            torch.save(
                {
                    "experiment_id": config["experiment_id"],
                    "gate_id": config["gate_id"],
                    "split": args.split,
                    "sample_indices": indices,
                    "samples": samples_payload,
                },
                output_dir / "calibration_payload.partial.pt",
            )
        log.emit("sample_complete", sample_id=identity["id"])
        del native_states, native_times, rcm_states, rcm_times, trajectory_states
        del condition, uncondition
        gc.collect()
        torch.cuda.empty_cache()

    if args.split == "calibration":
        final_payload = output_dir / "calibration_payload.pt"
        torch.save(
            {
                "experiment_id": config["experiment_id"],
                "gate_id": config["gate_id"],
                "split": args.split,
                "sample_indices": indices,
                "samples": samples_payload,
            },
            final_payload,
        )
    else:
        write_csv(output_dir / "cell_metrics.csv", metric_rows)

    gpu = torch.cuda.get_device_properties(device)
    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "split": args.split,
        "sample_indices": list(indices),
        "sample_ids": [identity["id"] for identity in identities],
        "identity": identities_verified,
        "elapsed_seconds": time.perf_counter() - started,
        "metric_rows": len(metric_rows),
        "calibration_samples": len(samples_payload),
        "verify_equivalence": args.verify_equivalence,
        "engineering_smoke": args.engineering_smoke,
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
    }
    atomic_write_json(output_dir / "capture_manifest.json", manifest)
    atomic_write_json(output_dir / "SUCCESS.json", {"status": "complete"})
    log.emit("run_complete", metric_rows=len(metric_rows))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
