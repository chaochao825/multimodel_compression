from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_dit_correction import (  # noqa: E402
    BucketChannelAffineRegressor,
    BucketDenseRegressor,
    BucketMeanRegressor,
    BucketReducedRankRegressor,
    TemporalKernelRegressor,
    TemporalLowRankRegressor,
    bucket_ids,
    frozen_basis_projection,
    observed_step_count,
    quantize_action_dit_ffn,
    relative_l2,
    row_relative_l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-policy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--evaluation-samples", type=int, default=64)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_indices(length: int, count: int) -> np.ndarray:
    if count > length:
        raise ValueError(f"requested {count} samples from a split of length {length}")
    return np.linspace(0, length - 1, count, dtype=np.int64)


def load_observations(dataset, indices: np.ndarray) -> torch.Tensor:
    return torch.stack([dataset[int(index)]["obs"] for index in indices])


def load_policy_and_data(args: argparse.Namespace):
    sys.path.insert(0, str(args.diffusion_policy_root))
    os.chdir(args.diffusion_policy_root)
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        pickle_module=dill,
    )
    workspace_class = hydra.utils.get_class(payload["cfg"]._target_)
    workspace = workspace_class(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model
    policy.eval().to(args.device)

    dataset = hydra.utils.instantiate(payload["cfg"].task.dataset)
    validation_dataset = dataset.get_validation_dataset()
    calibration_indices = deterministic_indices(len(dataset), args.calibration_samples)
    evaluation_indices = deterministic_indices(
        len(validation_dataset), args.evaluation_samples
    )
    calibration_obs = load_observations(dataset, calibration_indices)
    evaluation_obs = load_observations(validation_dataset, evaluation_indices)
    return (
        payload["cfg"],
        policy,
        calibration_obs,
        evaluation_obs,
        calibration_indices,
        evaluation_indices,
        len(dataset),
        len(validation_dataset),
    )


def normalized_condition(policy, observations: torch.Tensor, device: str) -> torch.Tensor:
    normalized = policy.normalizer["obs"].normalize(observations.to(device))
    return normalized[:, : policy.n_obs_steps]


@torch.inference_mode()
def collect_teacher_trajectory(
    full_model,
    quantized_model,
    policy,
    observations: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    condition = normalized_condition(policy, observations, device)
    batch_size = len(observations)
    scheduler = copy.deepcopy(policy.noise_scheduler)
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    trajectory = torch.randn(
        (batch_size, policy.horizon, policy.action_dim),
        dtype=policy.dtype,
        device=device,
        generator=generator,
    )

    noisy_actions = []
    quantized_outputs = []
    defects = []
    conditions = []
    step_indices = []
    timestep_values = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        full_output = full_model(trajectory, timestep, condition)
        quantized_output = quantized_model(trajectory, timestep, condition)
        noisy_actions.append(trajectory.float().cpu().numpy())
        quantized_outputs.append(quantized_output.float().cpu().numpy())
        defects.append((full_output - quantized_output).float().cpu().numpy())
        conditions.append(
            condition.reshape(batch_size, -1).float().cpu().numpy()
        )
        step_indices.append(np.full(batch_size, step_index, dtype=np.int64))
        timestep_values.append(
            np.full(batch_size, int(timestep.item()), dtype=np.int64)
        )
        trajectory = scheduler.step(
            full_output,
            timestep,
            trajectory,
            generator=generator,
        ).prev_sample

    return {
        "noisy_action": np.concatenate(noisy_actions),
        "quantized_output": np.concatenate(quantized_outputs),
        "defect": np.concatenate(defects),
        "condition": np.concatenate(conditions),
        "step_index": np.concatenate(step_indices),
        "timestep": np.concatenate(timestep_values),
    }


def make_methods(alpha: float) -> dict[str, object]:
    return {
        "bucket_mean": BucketMeanRegressor(),
        "channel_affine": BucketChannelAffineRegressor(alpha),
        "circulant_r2": TemporalKernelRegressor(2, True, alpha),
        "toeplitz_r2": TemporalKernelRegressor(2, False, alpha),
        "reduced_rank_r4": BucketReducedRankRegressor(4, alpha),
        "toeplitz_r2_rank4": TemporalLowRankRegressor(2, 4, alpha),
        "dense_ridge_ceiling": BucketDenseRegressor(alpha),
    }


def fit_methods(
    methods: dict[str, object],
    records: dict[str, np.ndarray],
    buckets: np.ndarray,
) -> None:
    for method in methods.values():
        method.fit(
            records["noisy_action"],
            records["quantized_output"],
            records["condition"],
            records["defect"],
            buckets,
        )


def teacher_forced_rows(
    methods: dict[str, object],
    records: dict[str, np.ndarray],
    buckets: np.ndarray,
    denoiser_macs: int,
) -> tuple[list[dict], list[dict]]:
    defect = records["defect"]
    quantized_output = records["quantized_output"]
    full_output = quantized_output + defect
    method_predictions = {"w4_plain": np.zeros_like(defect)}
    for name, method in methods.items():
        method_predictions[name] = method.predict(
            records["noisy_action"],
            quantized_output,
            records["condition"],
            buckets,
        )

    rows = []
    per_bucket = []
    for name, correction in method_predictions.items():
        residual = defect - correction
        action_prediction = quantized_output + correction
        ratios = row_relative_l2(action_prediction, full_output)
        method = methods[name] if name in methods else None
        parameter_count = 0 if method is None else method.parameter_count
        macs = 0 if method is None else method.macs_per_sample
        rows.append(
            {
                "method": name,
                "denoiser_output_relative_l2": relative_l2(
                    action_prediction, full_output
                ),
                "defect_residual_relative_l2": relative_l2(correction, defect),
                "captured_defect_energy": float(
                    1.0
                    - np.linalg.norm(residual) ** 2
                    / (np.linalg.norm(defect) ** 2 + 1e-12)
                ),
                "p95_call_relative_l2": float(np.quantile(ratios, 0.95)),
                "parameter_count": parameter_count,
                "macs_per_sample": macs,
                "mac_fraction_of_denoiser": float(macs / denoiser_macs),
            }
        )
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            per_bucket.append(
                {
                    "method": name,
                    "bucket": int(bucket),
                    "sample_count": int(mask.sum()),
                    "denoiser_output_relative_l2": relative_l2(
                        action_prediction[mask], full_output[mask]
                    ),
                    "defect_residual_relative_l2": relative_l2(
                        correction[mask], defect[mask]
                    ),
                }
            )
    return rows, per_bucket


@torch.inference_mode()
def sample_policy(
    model,
    policy,
    observations: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    device: str,
    predictor=None,
    bucket_count: int = 10,
) -> dict[str, np.ndarray]:
    condition = normalized_condition(policy, observations, device)
    batch_size = len(observations)
    scheduler = copy.deepcopy(policy.noise_scheduler)
    scheduler.set_timesteps(num_inference_steps)
    sampling_step_count = len(scheduler.timesteps)
    generator = torch.Generator(device=device).manual_seed(seed)
    trajectory = torch.randn(
        (batch_size, policy.horizon, policy.action_dim),
        dtype=policy.dtype,
        device=device,
        generator=generator,
    )

    for step_index, timestep in enumerate(scheduler.timesteps):
        model_output = model(trajectory, timestep, condition)
        if predictor is not None:
            current = trajectory.float().cpu().numpy()
            quantized = model_output.float().cpu().numpy()
            cond = condition.reshape(batch_size, -1).float().cpu().numpy()
            bucket = bucket_ids(
                np.full(batch_size, step_index),
                sampling_step_count,
                bucket_count,
            )
            correction = predictor.predict(current, quantized, cond, bucket)
            model_output = model_output + torch.from_numpy(correction).to(
                device=device, dtype=model_output.dtype
            )
        trajectory = scheduler.step(
            model_output,
            timestep,
            trajectory,
            generator=generator,
        ).prev_sample

    normalized_action = trajectory.float()
    action = policy.normalizer["action"].unnormalize(normalized_action)
    start = policy.n_obs_steps - 1
    end = start + policy.n_action_steps
    return {
        "normalized_horizon": normalized_action.cpu().numpy(),
        "action_horizon": action.float().cpu().numpy(),
        "executed_action": action[:, start:end].float().cpu().numpy(),
    }


def rollout_metric_row(name: str, prediction: dict, reference: dict) -> dict:
    pred = prediction["normalized_horizon"]
    ref = reference["normalized_horizon"]
    per_sample = row_relative_l2(pred, ref)
    velocity_pred = np.diff(pred, axis=1)
    velocity_ref = np.diff(ref, axis=1)
    acceleration_pred = np.diff(pred, n=2, axis=1)
    acceleration_ref = np.diff(ref, n=2, axis=1)
    endpoint = np.linalg.norm(pred[:, -1] - ref[:, -1], axis=1)
    executed_mae = np.mean(
        np.abs(prediction["executed_action"] - reference["executed_action"])
    )
    return {
        "method": name,
        "action_relative_l2": relative_l2(pred, ref),
        "p95_sample_relative_l2": float(np.quantile(per_sample, 0.95)),
        "velocity_relative_l2": relative_l2(velocity_pred, velocity_ref),
        "acceleration_relative_l2": relative_l2(
            acceleration_pred, acceleration_ref
        ),
        "mean_endpoint_l2": float(endpoint.mean()),
        "executed_action_mae_raw_units": float(executed_mae),
    }


def estimate_model_macs(cfg) -> dict[str, object]:
    horizon = int(cfg.horizon)
    condition_tokens = 1 + int(cfg.n_obs_steps)
    width = int(cfg.policy.model.n_emb)
    layer_count = int(cfg.policy.model.n_layer)
    feedforward = 4 * width

    input_mac = horizon * int(cfg.action_dim) * width
    condition_embedding_mac = int(cfg.n_obs_steps) * int(cfg.obs_dim) * width
    condition_mlp_mac = 2 * condition_tokens * width * feedforward
    self_attention_mac = 4 * horizon * width * width + 2 * horizon * horizon * width
    cross_attention_mac = (
        2 * horizon * width * width
        + 2 * condition_tokens * width * width
        + 2 * horizon * condition_tokens * width
    )
    decoder_ffn_mac = 2 * horizon * width * feedforward
    head_mac = horizon * width * int(cfg.action_dim)
    total = (
        input_mac
        + condition_embedding_mac
        + condition_mlp_mac
        + layer_count
        * (self_attention_mac + cross_attention_mac + decoder_ffn_mac)
        + head_mac
    )
    selected_ffn = condition_mlp_mac + layer_count * decoder_ffn_mac
    return {
        "estimated_denoiser_macs_per_sample": int(total),
        "estimated_selected_ffn_macs_per_sample": int(selected_ffn),
        "selected_ffn_mac_fraction": float(selected_ffn / total),
        "attention_score_mac_fraction": float(
            layer_count
            * (
                2 * horizon * horizon * width
                + 2 * horizon * condition_tokens * width
            )
            / total
        ),
    }


@torch.inference_mode()
def benchmark_forward_ms(model, policy, observations, device: str) -> float:
    condition = normalized_condition(policy, observations[:1], device)
    sample = torch.randn(
        (1, policy.horizon, policy.action_dim),
        device=device,
        dtype=policy.dtype,
    )
    timestep = torch.tensor(50, device=device)
    for _ in range(20):
        model(sample, timestep, condition)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        model(sample, timestep, condition)
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop) / 100.0)


def classify_gate(
    teacher_rows: list[dict],
    rollout_rows: list[dict],
    denoiser_macs: int,
) -> dict:
    teacher = {row["method"]: row for row in teacher_rows}
    rollout = {row["method"]: row for row in rollout_rows}
    baseline = rollout["w4_plain"]
    deployable = [
        name
        for name in rollout
        if name not in {"full_precision", "w4_plain", "w8_plain", "dense_ridge_ceiling"}
    ]
    best_name = min(deployable, key=lambda name: rollout[name]["action_relative_l2"])
    best = rollout[best_name]
    improvement = 1.0 - best["action_relative_l2"] / max(
        baseline["action_relative_l2"], 1e-12
    )
    guards = {
        "action_relative_improvement_at_least_25pct": bool(improvement >= 0.25),
        "p95_not_worse": bool(
            best["p95_sample_relative_l2"] <= baseline["p95_sample_relative_l2"]
        ),
        "endpoint_not_worse_by_10pct": bool(
            best["mean_endpoint_l2"] <= 1.1 * baseline["mean_endpoint_l2"]
        ),
        "acceleration_not_worse_by_10pct": bool(
            best["acceleration_relative_l2"]
            <= 1.1 * baseline["acceleration_relative_l2"]
        ),
        "teacher_forced_improves": bool(
            teacher[best_name]["denoiser_output_relative_l2"]
            < teacher["w4_plain"]["denoiser_output_relative_l2"]
        ),
        "correction_work_at_most_5pct": bool(
            teacher[best_name]["macs_per_sample"] <= 0.05 * denoiser_macs
        ),
    }
    if all(guards.values()):
        decision = "GO"
    elif improvement > 0:
        decision = "BOUNDARY"
    elif best["action_relative_l2"] >= 1.1 * baseline["action_relative_l2"]:
        decision = "ADVERSE"
    else:
        decision = "NULL"
    return {
        "decision": decision,
        "best_deployable_method": best_name,
        "action_relative_improvement": float(improvement),
        "guards": guards,
    }


def main() -> None:
    args = parse_args()
    args.diffusion_policy_root = args.diffusion_policy_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()

    (
        cfg,
        policy,
        calibration_obs,
        evaluation_obs,
        calibration_indices,
        evaluation_indices,
        train_length,
        validation_length,
    ) = load_policy_and_data(args)

    full_model = copy.deepcopy(policy.model).eval().to(args.device)
    w4_model = copy.deepcopy(policy.model).eval().to(args.device)
    w8_model = copy.deepcopy(policy.model).eval().to(args.device)
    w4_profile = quantize_action_dit_ffn(w4_model, 4)
    w8_profile = quantize_action_dit_ffn(w8_model, 8)
    mac_profile = estimate_model_macs(cfg)
    denoiser_macs = mac_profile["estimated_denoiser_macs_per_sample"]

    calibration = collect_teacher_trajectory(
        full_model,
        w4_model,
        policy,
        calibration_obs,
        args.num_inference_steps,
        args.seed + 1,
        args.device,
    )
    evaluation = collect_teacher_trajectory(
        full_model,
        w4_model,
        policy,
        evaluation_obs,
        args.num_inference_steps,
        args.seed + 2,
        args.device,
    )
    calibration_step_count = observed_step_count(calibration["step_index"])
    evaluation_step_count = observed_step_count(evaluation["step_index"])
    if calibration_step_count != evaluation_step_count:
        raise RuntimeError("calibration and evaluation schedules have different lengths")
    calibration_buckets = bucket_ids(
        calibration["step_index"], calibration_step_count, args.bucket_count
    )
    evaluation_buckets = bucket_ids(
        evaluation["step_index"], evaluation_step_count, args.bucket_count
    )

    methods = make_methods(args.ridge_alpha)
    fit_methods(methods, calibration, calibration_buckets)
    teacher_rows, bucket_rows = teacher_forced_rows(
        methods, evaluation, evaluation_buckets, denoiser_macs
    )
    basis_rows = [
        {
            "rank": row.rank,
            "residual_relative_l2": row.residual_relative_l2,
            "captured_energy": row.captured_energy,
        }
        for row in frozen_basis_projection(
            calibration["defect"],
            calibration_buckets,
            evaluation["defect"],
            evaluation_buckets,
            (1, 2, 4, 8),
        )
    ]

    rollout_seed = args.seed + 3
    full_rollout = sample_policy(
        full_model,
        policy,
        evaluation_obs,
        args.num_inference_steps,
        rollout_seed,
        args.device,
        bucket_count=args.bucket_count,
    )
    rollout_rows = [
        rollout_metric_row("full_precision", full_rollout, full_rollout),
        rollout_metric_row(
            "w4_plain",
            sample_policy(
                w4_model,
                policy,
                evaluation_obs,
                args.num_inference_steps,
                rollout_seed,
                args.device,
                bucket_count=args.bucket_count,
            ),
            full_rollout,
        ),
        rollout_metric_row(
            "w8_plain",
            sample_policy(
                w8_model,
                policy,
                evaluation_obs,
                args.num_inference_steps,
                rollout_seed,
                args.device,
                bucket_count=args.bucket_count,
            ),
            full_rollout,
        ),
    ]
    for name, method in methods.items():
        rollout_rows.append(
            rollout_metric_row(
                name,
                sample_policy(
                    w4_model,
                    policy,
                    evaluation_obs,
                    args.num_inference_steps,
                    rollout_seed,
                    args.device,
                    predictor=method,
                    bucket_count=args.bucket_count,
                ),
                full_rollout,
            )
        )

    gate = classify_gate(teacher_rows, rollout_rows, denoiser_macs)
    selected_fraction = (
        w4_profile["selected_weight_count"] / w4_profile["total_parameter_count"]
    )
    bf16_bytes = 2 * w4_profile["total_parameter_count"]
    selective_w4_bytes = int(
        0.5 * w4_profile["selected_weight_count"]
        + 2
        * (
            w4_profile["total_parameter_count"]
            - w4_profile["selected_weight_count"]
        )
    )
    model_profile = {
        "model_parameter_count": w4_profile["total_parameter_count"],
        "selected_ffn_weight_count": w4_profile["selected_weight_count"],
        "selected_ffn_module_count": w4_profile["selected_module_count"],
        "selected_weight_fraction": selected_fraction,
        "bf16_parameter_bytes": bf16_bytes,
        "selective_w4_parameter_bytes": selective_w4_bytes,
        "parameter_storage_compression": float(bf16_bytes / selective_w4_bytes),
        **mac_profile,
        "ideal_forward_speedup_if_selected_ffn_2x": float(
            1.0
            / (
                1.0
                - mac_profile["selected_ffn_mac_fraction"]
                + mac_profile["selected_ffn_mac_fraction"] / 2.0
            )
        ),
        "ideal_forward_speedup_if_selected_ffn_4x": float(
            1.0
            / (
                1.0
                - mac_profile["selected_ffn_mac_fraction"]
                + mac_profile["selected_ffn_mac_fraction"] / 4.0
            )
        ),
        "measured_fp_forward_ms_batch1_a800": benchmark_forward_ms(
            full_model, policy, evaluation_obs, args.device
        ),
    }

    write_csv(args.output_dir / "teacher_forced_metrics.csv", teacher_rows)
    write_csv(args.output_dir / "per_bucket_metrics.csv", bucket_rows)
    write_csv(args.output_dir / "rollout_metrics.csv", rollout_rows)
    write_csv(args.output_dir / "frozen_basis_transfer.csv", basis_rows)
    with (args.output_dir / "model_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(model_profile, handle, indent=2)
    with (args.output_dir / "split_indices.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_dataset_length": train_length,
                "validation_dataset_length": validation_length,
                "calibration_indices": calibration_indices.tolist(),
                "evaluation_indices": evaluation_indices.tolist(),
            },
            handle,
            indent=2,
        )

    summary = {
        "scope": (
            "Frozen PushT action Diffusion Transformer; fake W4/W8 numerical probe; "
            "no environment-success or integer-kernel speed claim"
        ),
        "checkpoint": str(args.checkpoint),
        "calibration_samples": args.calibration_samples,
        "evaluation_samples": args.evaluation_samples,
        "requested_num_inference_steps": args.num_inference_steps,
        "executed_num_inference_steps": evaluation_step_count,
        "bucket_count": args.bucket_count,
        "gate": gate,
        "teacher_forced_metrics": teacher_rows,
        "rollout_metrics": rollout_rows,
        "frozen_basis_transfer": basis_rows,
        "model_profile": model_profile,
        "elapsed_seconds": float(time.time() - started),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(torch.device(args.device)),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
