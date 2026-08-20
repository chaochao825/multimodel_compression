#!/usr/bin/env python3
"""Capture and evaluate the prospective EXP-045 Wan observability Gate."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import torch.cuda.amp as amp

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_SCRIPT_DIR = SCRIPT_DIR.parent.parent / "scripts"
if REPOSITORY_SCRIPT_DIR.is_dir():
    sys.path.insert(0, str(REPOSITORY_SCRIPT_DIR))

from current_input_observability_core import (
    CellTrajectory,
    ScalarAR2Params,
    ScalarAR2Statistics,
    relative_l2_terms,
    rollout_predict,
    target_visible_transport_oracle,
)
from generate_wan_cfg_parallel import make_scheduler, sequence_length, target_shape
from generate_wan_h200_v4 import (
    AttentionDispatcher,
    install_grid_compatibility,
    load_backends,
)
from wan_current_input_observability_runtime import (
    WanCurrentInputObservabilityRecorder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-solver", choices=("unipc",), default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-equivalence", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("sample-indices must be a nonempty unique comma list")
    return indices


def method_runtime_cost(
    method: str,
    *,
    tokens: int,
    channels: int,
    qk_rows: int,
    horizon: int,
    effective_secants: int,
    shift_count: int,
) -> dict[str, int]:
    nd = tokens * channels
    qk_projection = 0
    routing = 0
    if method in {"ar2", "online_ar2", "taylor1"}:
        predictor = 3 * nd
    elif method == "diagonal":
        predictor = 4 * nd
    elif method.startswith("broyden"):
        predictor = (3 + 4 * effective_secants) * nd
    elif method.startswith("transport"):
        predictor = (4 + 3 * shift_count) * nd
        if method.endswith("_qk"):
            qk_projection = 2 * nd * qk_rows
            routing = 75 * tokens * (2 * qk_rows)
        else:
            routing = 75 * tokens * min(16, channels)
    elif method.startswith("dplr"):
        rank = int(method.removeprefix("dplr"))
        predictor = (2 + 2 * rank) * nd
    else:
        raise ValueError(f"unsupported method cost model: {method}")
    return {
        "predictor_macs": horizon * predictor,
        "observable_macs": horizon * (qk_projection + routing),
        "qk_projection_macs": horizon * qk_projection,
        "routing_macs": horizon * routing,
        "total_runtime_macs": horizon * (predictor + qk_projection + routing),
    }


def drift_diagnostics(
    trajectory: CellTrajectory, target_step: int
) -> dict[str, float]:
    block_input = trajectory.block_input
    residual = trajectory.residual
    adaln = trajectory.adaln
    qk = trajectory.qk_sketch
    return {
        "input_drift_relative_l2": float(
            (block_input[target_step] - block_input[target_step - 1])
            .double()
            .square()
            .sum()
            .sqrt()
            / block_input[target_step].double().square().sum().sqrt().clamp_min(1e-30)
        ),
        "residual_drift_relative_l2": float(
            (residual[target_step] - residual[target_step - 1])
            .double()
            .square()
            .sum()
            .sqrt()
            / residual[target_step].double().square().sum().sqrt().clamp_min(1e-30)
        ),
        "adaln_drift_relative_l2": float(
            (adaln[target_step] - adaln[target_step - 1])
            .double()
            .square()
            .sum()
            .sqrt()
            / adaln[target_step].double().square().sum().sqrt().clamp_min(1e-30)
        ),
        "qk_sketch_drift_relative_l2": float(
            (qk[target_step] - qk[target_step - 1])
            .double()
            .square()
            .sum()
            .sqrt()
            / qk[target_step].double().square().sum().sqrt().clamp_min(1e-30)
        ),
    }


@torch.inference_mode()
def evaluate_cell(
    trajectory: CellTrajectory,
    *,
    sample_id: str,
    split: str,
    block: int,
    branch: int,
    config: dict[str, object],
    device: torch.device,
    calibrated_ar2: dict[int, ScalarAR2Params] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ridge = float(config["ridge"])
    method_horizons = config["method_horizons"]
    qk_rows = int(config["qk_rows_per_projection"])
    for target_step_value in config["target_steps"]:
        target_step = int(target_step_value)
        diagnostics = drift_diagnostics(trajectory, target_step)
        for method_value in config["methods"]:
            method = str(method_value)
            if method == "ar2" and calibrated_ar2 is None:
                continue
            horizons = tuple(int(value) for value in method_horizons[method])
            for horizon in horizons:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = rollout_predict(
                    trajectory,
                    method=method,
                    target_step=target_step,
                    horizon=horizon,
                    ridge=ridge,
                    calibrated_ar2=calibrated_ar2,
                )
                torch.cuda.synchronize(device)
                elapsed_ms = 1000 * (time.perf_counter() - started)
                relative, error_sq, target_sq = relative_l2_terms(
                    result.prediction, trajectory.residual[target_step]
                )
                output_target_sq = float(
                    (
                        trajectory.block_input[target_step]
                        + trajectory.residual[target_step]
                    )
                    .double()
                    .square()
                    .sum()
                )
                costs = method_runtime_cost(
                    method,
                    tokens=trajectory.token_count,
                    channels=trajectory.channel_count,
                    qk_rows=qk_rows,
                    horizon=horizon,
                    effective_secants=result.effective_secants,
                    shift_count=len(result.shifts),
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "block": block,
                        "target_step": target_step,
                        "branch": branch,
                        "horizon": horizon,
                        "method": method,
                        "relative_l2": relative,
                        "error_sq": error_sq,
                        "target_sq": target_sq,
                        "output_relative_l2": (
                            error_sq / max(output_target_sq, 1e-30)
                        )
                        ** 0.5,
                        "output_target_sq": output_target_sq,
                        "effective_secants": result.effective_secants,
                        "selected_shifts": json.dumps(result.shifts),
                        "fit_and_predict_ms": elapsed_ms,
                        **diagnostics,
                        **costs,
                    }
                )
                del result
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        oracle = target_visible_transport_oracle(
            trajectory, target_step=target_step, ridge=1e-6
        )
        torch.cuda.synchronize(device)
        elapsed_ms = 1000 * (time.perf_counter() - started)
        relative, error_sq, target_sq = relative_l2_terms(
            oracle.prediction, trajectory.residual[target_step]
        )
        output_target_sq = float(
            (
                trajectory.block_input[target_step]
                + trajectory.residual[target_step]
            )
            .double()
            .square()
            .sum()
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "block": block,
                "target_step": target_step,
                "branch": branch,
                "horizon": 1,
                "method": "oracle_transport75_token_ls",
                "relative_l2": relative,
                "error_sq": error_sq,
                "target_sq": target_sq,
                "output_relative_l2": (
                    error_sq / max(output_target_sq, 1e-30)
                )
                ** 0.5,
                "output_target_sq": output_target_sq,
                "effective_secants": 0,
                "selected_shifts": "target-visible 75-shift per-token selection",
                "fit_and_predict_ms": elapsed_ms,
                **diagnostics,
                "predictor_macs": 0,
                "observable_macs": 0,
                "qk_projection_macs": 0,
                "routing_macs": 0,
                "total_runtime_macs": 0,
            }
        )
        del oracle
    return rows


def update_ar2_statistics(
    statistics: dict[tuple[int, int, int], ScalarAR2Statistics],
    trajectory: CellTrajectory,
    *,
    block: int,
    branch: int,
) -> None:
    for step in range(2, trajectory.step_count):
        key = (block, branch, step)
        if key not in statistics:
            statistics[key] = ScalarAR2Statistics()
        statistics[key].update(
            trajectory.residual[step],
            trajectory.residual[step - 1],
            trajectory.residual[step - 2],
        )


def fit_calibrated_ar2(
    statistics: dict[tuple[int, int, int], ScalarAR2Statistics],
    ridge: float,
) -> tuple[
    dict[tuple[int, int, int], ScalarAR2Params],
    list[dict[str, object]],
]:
    if not statistics:
        raise ValueError("calibrated AR(2) statistics are empty")
    parameters: dict[tuple[int, int, int], ScalarAR2Params] = {}
    rows: list[dict[str, object]] = []
    for key in sorted(statistics):
        block, branch, step = key
        fitted = statistics[key].fit(ridge)
        parameters[key] = fitted
        rows.append(
            {
                "block": block,
                "branch": branch,
                "step": step,
                "lag1": fitted.lag1,
                "lag2": fitted.lag2,
                "observations": statistics[key].observations,
            }
        )
    return parameters, rows


@torch.inference_mode()
def run_dense_trajectory(
    pipeline: object,
    dispatcher: AttentionDispatcher,
    recorder: WanCurrentInputObservabilityRecorder | None,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    sample_id: str,
    split: str,
    prompt_index: int,
    sampling_steps: int,
    frame_num: int,
    width: int,
    height: int,
    sample_solver: str,
    shift: float,
    guide_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object] | None, float]:
    size = (width, height)
    shape = target_shape(pipeline, frame_num, size)
    seq_len = sequence_length(pipeline, shape)
    actual_negative = negative_prompt or pipeline.sample_neg_prompt
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([prompt], device)
    context_null = pipeline.text_encoder([actual_negative], device)
    latent = torch.randn(*shape, dtype=torch.float32, device=device, generator=generator)
    scheduler, timesteps = make_scheduler(
        sample_solver,
        pipeline.num_train_timesteps,
        sampling_steps,
        shift,
        device,
    )
    if recorder is not None:
        recorder.begin_run(
            sample_id,
            [float(value) for value in timesteps],
            {
                "prompt_index": prompt_index,
                "seed": seed,
                "split": split,
                "frame_num": frame_num,
            },
        )
    dispatcher.begin("fa3_bf16")

    @contextmanager
    def noop_no_sync():
        yield

    no_sync: Callable[[], object] = getattr(pipeline.model, "no_sync", noop_no_sync)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with amp.autocast(dtype=pipeline.param_dtype), no_sync():
        for timestep in timesteps:
            model_input = latent.unsqueeze(0)
            conditional = pipeline.model(
                model_input,
                t=torch.stack([timestep]),
                context=context,
                seq_len=seq_len,
            )[0]
            unconditional = pipeline.model(
                model_input,
                t=torch.stack([timestep]),
                context=context_null,
                seq_len=seq_len,
            )[0]
            guided = unconditional + guide_scale * (conditional - unconditional)
            next_latent = scheduler.step(
                guided.unsqueeze(0),
                timestep,
                model_input,
                return_dict=False,
                generator=generator,
            )[0]
            latent = next_latent.squeeze(0)
            del model_input, conditional, unconditional, guided, next_latent
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    recorder_summary = None
    if recorder is not None:
        recorder_summary = recorder.end_run()
    del context, context_null
    return latent, recorder_summary, seconds


def debug_payload(
    recorder: WanCurrentInputObservabilityRecorder,
    config: dict[str, object],
) -> dict[str, object]:
    block = int(config["blocks"][0])
    branch = int(config["branches"][0])
    payload: dict[str, object] = {
        "warning": "limited debug rows only; never use for Gate metrics",
        "block": block,
        "branch": branch,
        "rows": 8,
    }
    for target_value in config["target_steps"]:
        target = int(target_value)
        record = recorder.records[(block, target, branch)]
        payload[f"step_{target}"] = {
            "block_input": record["block_input"][:8].clone(),
            "residual": record["residual"][:8].clone(),
            "adaln": record["adaln"].clone(),
            "qk_sketch": record["qk_sketch"][:8].clone(),
        }
    return payload


def main() -> None:
    args = parse_args()
    args.wan_source = args.wan_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.prompt_file = args.prompt_file.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_plan = config["sample_plan"]
    sample_indices = parse_indices(args.sample_indices)
    if sample_indices != tuple(sorted(sample_indices)):
        raise ValueError("sample indices must follow the frozen split order")
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(sample_plan):
            raise ValueError("sample index lies outside the frozen plan")
        item = sample_plan[sample_index]
        if item["split"] not in config["allowed_initial_splits"]:
            raise PermissionError(
                f"split {item['split']} is locked until the router stage"
            )
        if int(item["prompt_index"]) >= len(prompts):
            raise ValueError("sample plan references an unavailable prompt")
    if args.verify_equivalence and len(sample_indices) != 1:
        raise ValueError("equivalence verification accepts exactly one sample")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(args.wan_source))
    os.chdir(args.wan_source)

    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    import wan.modules.model as wan_model_module

    compatibility = install_grid_compatibility(wan_model_module)
    backends = load_backends()
    if "fa3_bf16" not in backends:
        raise RuntimeError("FA3 BF16 backend is required")
    sampling_steps = int(config["sampling_steps"])
    dispatcher = AttentionDispatcher(
        backends, hybrid_layer_count=30, sampling_steps=sampling_steps
    )
    original_attention = wan_model_module.flash_attention
    wan_model_module.flash_attention = dispatcher
    recorder: WanCurrentInputObservabilityRecorder | None = None
    load_started = time.perf_counter()
    all_rows: list[dict[str, object]] = []
    sample_summaries: list[dict[str, object]] = []
    ar2_statistics: dict[tuple[int, int, int], ScalarAR2Statistics] = {}
    calibrated_ar2: dict[tuple[int, int, int], ScalarAR2Params] | None = None
    calibration_samples: set[str] = set()
    try:
        pipeline = WanT2V(
            config=WAN_CONFIGS["t2v-1.3B"],
            checkpoint_dir=str(args.checkpoint),
            device_id=device.index or 0,
            rank=0,
            t5_cpu=False,
        )
        pipeline.model.to(device=device, dtype=WAN_CONFIGS["t2v-1.3B"].param_dtype)
        load_seconds = time.perf_counter() - load_started

        dense_reference: torch.Tensor | None = None
        if args.verify_equivalence:
            item = sample_plan[sample_indices[0]]
            prompt_index = int(item["prompt_index"])
            dense_reference, _, dense_seconds = run_dense_trajectory(
                pipeline,
                dispatcher,
                None,
                prompt=prompts[prompt_index],
                negative_prompt=args.negative_prompt,
                seed=int(item["seed"]),
                sample_id="equivalence_dense",
                split=str(item["split"]),
                prompt_index=prompt_index,
                sampling_steps=sampling_steps,
                frame_num=int(config["frame_num"]),
                width=int(config["width"]),
                height=int(config["height"]),
                sample_solver=args.sample_solver,
                shift=args.shift,
                guide_scale=args.guide_scale,
                device=device,
            )
        else:
            dense_seconds = 0.0

        recorder = WanCurrentInputObservabilityRecorder(
            pipeline.model,
            sampling_steps=sampling_steps,
            capture_steps=tuple(int(value) for value in config["capture_steps"]),
            blocks=tuple(int(value) for value in config["blocks"]),
            branches=tuple(int(value) for value in config["branches"]),
            qk_rows=int(config["qk_rows_per_projection"]),
        )
        for sample_index in sample_indices:
            item = sample_plan[sample_index]
            prompt_index = int(item["prompt_index"])
            seed = int(item["seed"])
            split = str(item["split"])
            sample_id = f"p{prompt_index:02d}_seed{seed}"
            if split == "selection" and calibrated_ar2 is None:
                if len(calibration_samples) != 4:
                    raise RuntimeError(
                        "selection requires all four calibration identities in the same run"
                    )
                calibrated_ar2, calibrated_rows = fit_calibrated_ar2(
                    ar2_statistics, float(config["ridge"])
                )
                write_csv(args.out_dir / "calibrated_ar2.csv", calibrated_rows)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            latent, capture_summary, generation_seconds = run_dense_trajectory(
                pipeline,
                dispatcher,
                recorder,
                prompt=prompts[prompt_index],
                negative_prompt=args.negative_prompt,
                seed=seed,
                sample_id=sample_id,
                split=split,
                prompt_index=prompt_index,
                sampling_steps=sampling_steps,
                frame_num=int(config["frame_num"]),
                width=int(config["width"]),
                height=int(config["height"]),
                sample_solver=args.sample_solver,
                shift=args.shift,
                guide_scale=args.guide_scale,
                device=device,
            )
            if capture_summary is None:
                raise RuntimeError("recorded generation returned no capture summary")
            equivalence_relative_l2 = None
            equivalence_exact = None
            if dense_reference is not None:
                equivalence_relative_l2 = relative_l2_terms(
                    latent, dense_reference
                )[0]
                equivalence_exact = bool(torch.equal(latent, dense_reference))
                if equivalence_relative_l2 > 1e-7:
                    raise RuntimeError(
                        "observability capture changed the dense final latent: "
                        f"{equivalence_relative_l2:.6e}"
                    )

            sample_dir = args.out_dir / sample_id
            sample_dir.mkdir()
            torch.save(debug_payload(recorder, config), sample_dir / "debug_rows.pt")
            sample_rows: list[dict[str, object]] = []
            evaluation_started = time.perf_counter()
            for block_value in config["blocks"]:
                block = int(block_value)
                for branch_value in config["branches"]:
                    branch = int(branch_value)
                    trajectory = recorder.cell_trajectory(
                        block, branch, device=device
                    )
                    if split == "calibration":
                        update_ar2_statistics(
                            ar2_statistics,
                            trajectory,
                            block=block,
                            branch=branch,
                        )
                    cell_ar2 = None
                    if calibrated_ar2 is not None:
                        cell_ar2 = {
                            step: calibrated_ar2[(block, branch, step)]
                            for step in range(2, trajectory.step_count)
                        }
                    sample_rows.extend(
                        evaluate_cell(
                            trajectory,
                            sample_id=sample_id,
                            split=split,
                            block=block,
                            branch=branch,
                            config=config,
                            device=device,
                            calibrated_ar2=cell_ar2,
                        )
                    )
                    del trajectory
                    torch.cuda.empty_cache()
            evaluation_seconds = time.perf_counter() - evaluation_started
            write_csv(sample_dir / "cell_metrics.csv", sample_rows)
            sample_summary = {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "prompt_index": prompt_index,
                "seed": seed,
                "split": split,
                "generation_seconds": generation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "capture": capture_summary,
                "equivalence_relative_l2": equivalence_relative_l2,
                "equivalence_exact": equivalence_exact,
                "latent_norm": float(latent.double().square().sum().sqrt()),
                "metric_rows": len(sample_rows),
            }
            write_json(sample_dir / "sample_manifest.json", sample_summary)
            all_rows.extend(sample_rows)
            sample_summaries.append(sample_summary)
            if split == "calibration":
                calibration_samples.add(sample_id)
            recorder.clear()
            del latent, sample_rows
            dense_reference = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"DONE sample={sample_id} generation={generation_seconds:.2f}s "
                f"evaluation={evaluation_seconds:.2f}s rows={sample_summary['metric_rows']}",
                flush=True,
            )
    finally:
        if recorder is not None:
            recorder.restore()
        wan_model_module.flash_attention = original_attention

    write_csv(args.out_dir / "all_cell_metrics.csv", all_rows)
    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "scope": config["scope"],
        "warning": "component observability evidence only; no rollout or speed claim",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "sample_indices": list(sample_indices),
        "sample_summaries": sample_summaries,
        "load_seconds": load_seconds,
        "equivalence_dense_seconds": dense_seconds,
        "grid_compatibility_installed": compatibility,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(args.out_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
