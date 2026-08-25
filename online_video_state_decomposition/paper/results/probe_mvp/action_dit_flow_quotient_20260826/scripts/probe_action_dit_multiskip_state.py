from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import probe_action_dit_noise_response_bridge as bridge  # noqa: E402
import probe_action_dit_transport_cache as base  # noqa: E402
from action_dit_transport_cache import (  # noqa: E402
    captured_energy,
    coefficient_r2,
    flatten_feature_groups,
    horizon_shift,
    oracle_gap_recovery,
    overlap_mask,
    relative_l2,
    reuse_with_exact_tail,
    row_relative_l2,
    transfer_basis_coefficients,
)


SKIPS = (1, 2, 4)
METHODS = (
    "raw_reuse",
    "shift_reuse",
    "shift_local_r2",
    "shift_rank8_oracle",
    "state_skip1",
    "state_skip2",
    "state_skip4",
    "state_noise_skip1",
    "state_noise_skip2",
    "state_noise_skip4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-policy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-offset", type=int, default=8)
    parser.add_argument("--calibration-transitions", type=int, default=96)
    parser.add_argument("--evaluation-transitions", type=int, default=48)
    parser.add_argument("--flow-points", type=int, default=10)
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--late-flow-count", type=int, default=3)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


@torch.inference_mode()
def collect_sequences(
    model,
    policy,
    tensors: dict,
    offset: int,
    flow_points: int,
    max_skip: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    previous_condition, current_condition, previous_action, current_action = (
        base.normalized_transition(policy, tensors, device)
    )
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    reference_indices = np.linspace(
        1, len(scheduler.timesteps) - 1, flow_points, dtype=np.int64
    )
    endpoint_indices = reference_indices[reference_indices >= max_skip]
    generator = torch.Generator(device=device).manual_seed(seed)
    previous_noise = torch.randn(
        previous_action.shape,
        dtype=previous_action.dtype,
        device=device,
        generator=generator,
    )
    current_noise = torch.randn(
        current_action.shape,
        dtype=current_action.dtype,
        device=device,
        generator=generator,
    )
    capture = base.FFNCapture(model)
    endpoint_records = {
        "previous_input": [],
        "current_input": [],
        "previous_residual": [],
        "current_residual": [],
        "exact_output": [],
        "current_noisy": [],
    }
    timestep_rows = []
    scheduler_index_rows = []
    for endpoint_index in endpoint_indices:
        sequence = {key: [] for key in endpoint_records}
        timesteps = []
        scheduler_indices = []
        for scheduler_index in range(
            int(endpoint_index) - max_skip, int(endpoint_index) + 1
        ):
            timestep = scheduler.timesteps[scheduler_index]
            timestep_batch = timestep.expand(len(previous_action))
            previous_noisy = scheduler.add_noise(
                previous_action, previous_noise, timestep_batch
            )
            current_noisy = scheduler.add_noise(
                current_action, current_noise, timestep_batch
            )
            _, previous_input, previous_residual = base.captured_forward(
                model, capture, previous_noisy, timestep, previous_condition
            )
            exact_output, current_input, current_residual = base.captured_forward(
                model, capture, current_noisy, timestep, current_condition
            )
            sequence["previous_input"].append(previous_input)
            sequence["current_input"].append(current_input)
            sequence["previous_residual"].append(previous_residual)
            sequence["current_residual"].append(current_residual)
            sequence["exact_output"].append(exact_output)
            sequence["current_noisy"].append(
                current_noisy.float().cpu().numpy()
            )
            timesteps.append(int(timestep.item()))
            scheduler_indices.append(scheduler_index)
        for key, values in sequence.items():
            endpoint_records[key].append(np.stack(values, axis=1))
        timestep_rows.append(timesteps)
        scheduler_index_rows.append(scheduler_indices)
    capture.close()
    output = {
        key: np.stack(values, axis=1)
        for key, values in endpoint_records.items()
    }
    output.update(
        {
            "previous_noise": previous_noise.float().cpu().numpy(),
            "current_noise": current_noise.float().cpu().numpy(),
            "previous_condition": previous_condition.float().cpu().numpy(),
            "current_condition": current_condition.float().cpu().numpy(),
            "timestep": np.asarray(timestep_rows),
            "scheduler_index": np.asarray(scheduler_index_rows),
            "reference_indices": reference_indices,
        }
    )
    return output


def bucket_for_scheduler_index(
    scheduler_index: int,
    reference_indices: np.ndarray,
    bucket_count: int,
) -> int:
    reference_buckets = base.step_buckets(len(reference_indices), bucket_count)
    nearest = int(np.argmin(np.abs(reference_indices - scheduler_index)))
    return int(reference_buckets[nearest])


def cell_innovation(
    records: dict[str, np.ndarray],
    endpoint: int,
    sequence_position: int,
    layer: int,
    offset: int,
) -> np.ndarray:
    current = records["current_residual"][
        :, endpoint, sequence_position, layer
    ]
    previous = records["previous_residual"][
        :, endpoint, sequence_position, layer
    ]
    valid = overlap_mask(current.shape[1], offset)
    return base.zero_tail(current - horizon_shift(previous, offset), valid)


def recurrent_coefficients(
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    endpoint: int,
    layer: int,
    skip: int,
    predictor: str,
    offset: int,
    bucket_count: int,
    timestep_scale: float,
) -> tuple[np.ndarray, object]:
    end_position = records["timestep"].shape[1] - 1
    start_position = end_position - skip
    source_innovation = cell_innovation(
        records, endpoint, start_position, layer, offset
    )
    source_basis = None
    source_coefficients = None
    valid = overlap_mask(source_innovation.shape[1], offset)
    delta_noise = base.zero_tail(
        records["current_noise"]
        - horizon_shift(records["previous_noise"], offset),
        valid,
    )
    for position in range(start_position + 1, end_position + 1):
        scheduler_index = int(records["scheduler_index"][endpoint, position])
        bucket = bucket_for_scheduler_index(
            scheduler_index, records["reference_indices"], bucket_count
        )
        model = models[(layer, bucket)]
        target_basis = model["basis"]
        if source_basis is None:
            state_coefficients = target_basis.coefficients(source_innovation)
        else:
            state_coefficients = transfer_basis_coefficients(
                source_coefficients, source_basis, target_basis
            )
        timesteps = np.full(
            len(state_coefficients),
            records["timestep"][endpoint, position],
        )
        time = bridge.timestep_feature(
            timesteps, timestep_scale, state_coefficients.dtype
        )
        if predictor == "state":
            features = flatten_feature_groups(state_coefficients, time)
        elif predictor == "state_noise":
            features = flatten_feature_groups(
                state_coefficients, delta_noise, time
            )
        else:
            raise ValueError(f"unknown recurrent predictor: {predictor}")
        source_coefficients = model[predictor].predict(features)
        source_basis = target_basis
    return source_coefficients, source_basis


def cell_predictions(
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    endpoint: int,
    layer: int,
    offset: int,
    bucket_count: int,
    timestep_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    end_position = records["timestep"].shape[1] - 1
    scheduler_index = int(records["scheduler_index"][endpoint, end_position])
    bucket = bucket_for_scheduler_index(
        scheduler_index, records["reference_indices"], bucket_count
    )
    model = models[(layer, bucket)]
    basis = model["basis"]
    current = records["current_residual"][:, endpoint, end_position, layer]
    previous = records["previous_residual"][:, endpoint, end_position, layer]
    current_input = records["current_input"][:, endpoint, end_position, layer]
    previous_input = records["previous_input"][:, endpoint, end_position, layer]
    valid = overlap_mask(current.shape[1], offset)
    target_innovation = base.zero_tail(
        current - horizon_shift(previous, offset), valid
    )
    delta_input = base.zero_tail(
        current_input - horizon_shift(previous_input, offset), valid
    )
    shifted = reuse_with_exact_tail(previous, current, offset, aligned=True)
    predictions = {
        "raw_reuse": reuse_with_exact_tail(
            previous, current, offset, aligned=False
        ),
        "shift_reuse": shifted,
        "shift_local_r2": shifted
        + base.zero_tail(model["local"].predict(delta_input), valid),
        "shift_rank8_oracle": shifted
        + base.zero_tail(basis.project(target_innovation), valid),
    }
    coefficient_diagnostics = {}
    recurrent = {}
    for predictor in ("state", "state_noise"):
        for skip in SKIPS:
            coefficients, recurrent_basis = recurrent_coefficients(
                records,
                models,
                endpoint,
                layer,
                skip,
                predictor,
                offset,
                bucket_count,
                timestep_scale,
            )
            if recurrent_basis is not basis:
                coefficients = transfer_basis_coefficients(
                    coefficients, recurrent_basis, basis
                )
            method = f"{predictor}_skip{skip}"
            recurrent[method] = coefficients
            predictions[method] = shifted + base.zero_tail(
                basis.reconstruct(coefficients), valid
            )
            coefficient_diagnostics[f"{method}_coefficient_r2"] = coefficient_r2(
                coefficients, basis.coefficients(target_innovation)
            )
    exact_previous = cell_innovation(
        records, endpoint, end_position - 1, layer, offset
    )
    exact_state = basis.coefficients(exact_previous)
    delta_noise = base.zero_tail(
        records["current_noise"]
        - horizon_shift(records["previous_noise"], offset),
        valid,
    )
    timesteps = np.full(
        len(exact_state), records["timestep"][endpoint, end_position]
    )
    direct_features = flatten_feature_groups(
        exact_state,
        delta_noise,
        bridge.timestep_feature(timesteps, timestep_scale, exact_state.dtype),
    )
    direct = model["state_noise"].predict(direct_features)
    diagnostics = {
        "rank8_energy": captured_energy(
            target_innovation, basis.project(target_innovation)
        ),
        "skip1_teacher_max_abs": float(
            np.max(np.abs(direct - recurrent["state_noise_skip1"]))
        ),
        **coefficient_diagnostics,
    }
    return predictions, diagnostics


def evaluate(
    model,
    policy,
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    offset: int,
    bucket_count: int,
    late_flow_count: int,
    device: str,
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]], float]:
    endpoint_count = records["timestep"].shape[0]
    end_position = records["timestep"].shape[1] - 1
    late_start = endpoint_count - late_flow_count
    timestep_scale = float(np.max(records["timestep"]))
    condition = torch.as_tensor(
        records["current_condition"], dtype=policy.dtype, device=device
    )
    rows = []
    max_teacher_difference = 0.0
    accumulators = {
        region: {
            method: {"numerator": 0.0, "denominator": 0.0, "sample_error": []}
            for method in METHODS
        }
        for region in ("all", "late")
    }
    for endpoint in range(endpoint_count):
        sample = torch.as_tensor(
            records["current_noisy"][:, endpoint, end_position],
            dtype=policy.dtype,
            device=device,
        )
        timestep_value = int(records["timestep"][endpoint, end_position])
        timestep = torch.tensor(timestep_value, device=device)
        exact = records["exact_output"][:, endpoint, end_position]
        for layer in range(records["current_residual"].shape[3]):
            predictions, diagnostics = cell_predictions(
                records,
                models,
                endpoint,
                layer,
                offset,
                bucket_count,
                timestep_scale,
            )
            max_teacher_difference = max(
                max_teacher_difference, diagnostics["skip1_teacher_max_abs"]
            )
            current = records["current_residual"][:, endpoint, end_position, layer]
            for method, replacement in predictions.items():
                output = base.injected_output(
                    model, layer, replacement, sample, timestep, condition
                )
                sample_error = row_relative_l2(output, exact)
                rows.append(
                    {
                        "layer": layer,
                        "endpoint": endpoint,
                        "scheduler_index": int(
                            records["scheduler_index"][endpoint, end_position]
                        ),
                        "timestep": timestep_value,
                        "method": method,
                        "velocity_relative_l2": relative_l2(output, exact),
                        "velocity_mean_relative_l2": float(sample_error.mean()),
                        "velocity_p95_relative_l2": float(
                            np.quantile(sample_error, 0.95)
                        ),
                        "activation_relative_l2": relative_l2(
                            replacement, current
                        ),
                        "rank8_energy": diagnostics["rank8_energy"],
                        "coefficient_r2": diagnostics.get(
                            f"{method}_coefficient_r2", float("nan")
                        ),
                    }
                )
                regions = ["all"]
                if endpoint >= late_start:
                    regions.append("late")
                for region in regions:
                    state = accumulators[region][method]
                    state["numerator"] += float(np.linalg.norm(output - exact) ** 2)
                    state["denominator"] += float(np.linalg.norm(exact) ** 2)
                    state["sample_error"].append(sample_error)
    summaries = {}
    for region, methods in accumulators.items():
        summaries[region] = {}
        for method, state in methods.items():
            sample_error = np.concatenate(state["sample_error"])
            summaries[region][method] = {
                "velocity_relative_l2": float(
                    np.sqrt(state["numerator"] / state["denominator"])
                ),
                "velocity_mean_relative_l2": float(sample_error.mean()),
                "velocity_p95_relative_l2": float(
                    np.quantile(sample_error, 0.95)
                ),
            }
    return rows, summaries, max_teacher_difference


def aggregate_endpoints(rows: list[dict]) -> list[dict]:
    output = []
    for endpoint in sorted({int(row["endpoint"]) for row in rows}):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if int(row["endpoint"]) == endpoint and row["method"] == method
            ]
            output.append(
                {
                    "endpoint": endpoint,
                    "scheduler_index": int(selected[0]["scheduler_index"]),
                    "timestep": int(selected[0]["timestep"]),
                    "method": method,
                    "activation_relative_l2": float(
                        np.mean([row["activation_relative_l2"] for row in selected])
                    ),
                    "velocity_relative_l2": float(
                        np.mean([row["velocity_relative_l2"] for row in selected])
                    ),
                    "velocity_p95_relative_l2": float(
                        np.mean(
                            [row["velocity_p95_relative_l2"] for row in selected]
                        )
                    ),
                    "coefficient_r2": float(
                        np.nanmean([row["coefficient_r2"] for row in selected])
                    )
                    if any(np.isfinite(row["coefficient_r2"]) for row in selected)
                    else float("nan"),
                }
            )
    return output


def decision(
    summary: dict[str, dict[str, dict[str, float]]],
    teacher_difference: float,
) -> dict[str, object]:
    late = summary["late"]
    shifted = late["shift_reuse"]
    local = late["shift_local_r2"]
    oracle = late["shift_rank8_oracle"]
    recoveries = {
        skip: oracle_gap_recovery(
            shifted["velocity_relative_l2"],
            late[f"state_noise_skip{skip}"]["velocity_relative_l2"],
            oracle["velocity_relative_l2"],
        )
        for skip in SKIPS
    }
    one_step_pass = (
        teacher_difference <= 1e-6
        and recoveries[1] >= 0.8
        and late["state_noise_skip1"]["velocity_p95_relative_l2"]
        <= local["velocity_p95_relative_l2"]
    )
    two_step_pass = (
        one_step_pass
        and recoveries[2] >= 0.7
        and late["state_noise_skip2"]["velocity_p95_relative_l2"]
        <= local["velocity_p95_relative_l2"]
    )
    four_step_pass = (
        two_step_pass
        and recoveries[4] >= 0.5
        and late["state_noise_skip4"]["velocity_p95_relative_l2"]
        <= local["velocity_p95_relative_l2"]
        and late["state_noise_skip4"]["velocity_relative_l2"]
        <= 1.5 * late["state_noise_skip1"]["velocity_relative_l2"]
    )
    if four_step_pass:
        gate = "MULTISKIP_4_STABLE"
    elif two_step_pass:
        gate = "MULTISKIP_2_STABLE"
    elif one_step_pass:
        gate = "ONE_STEP_ONLY"
    else:
        gate = "NO_GO"
    return {
        "gate": gate,
        "one_step_pass": bool(one_step_pass),
        "two_step_pass": bool(two_step_pass),
        "four_step_pass": bool(four_step_pass),
        "skip1_teacher_max_abs": float(teacher_difference),
        "state_noise_oracle_gap_recovery": {
            str(skip): float(recoveries[skip]) for skip in SKIPS
        },
    }


def cost_proxy(
    models: dict[tuple[int, int], dict[str, object]],
    horizon: int,
    offset: int,
    width: int,
    feedforward: int,
) -> dict[str, float]:
    model = next(iter(models.values()))
    rank = int(model["basis"].basis.shape[0])
    reusable_tokens = horizon - offset
    basis_macs = reusable_tokens * rank * width
    map_macs = model["state_noise"].macs_per_sample
    replaced_macs = reusable_tokens * feedforward * width
    return {
        "rank": rank,
        "replaced_linear2_macs_per_step": replaced_macs,
        "basis_reconstruction_macs_per_step": basis_macs,
        "state_noise_map_macs_per_step": map_macs,
        "basis_coordinate_macs_at_bucket_change": rank * rank,
        "state_noise_fraction_per_step": float(
            (basis_macs + map_macs + rank * rank) / replaced_macs
        ),
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
        calibration_tensors,
        evaluation_tensors,
        calibration_pair,
        evaluation_pair,
        train_length,
        validation_length,
    ) = base.load_policy_and_data(args)
    if args.control_offset != int(cfg.n_action_steps):
        raise ValueError("multi-skip must use the deployed control offset")
    calibration = base.collect_geometry(
        policy.model,
        policy,
        calibration_tensors,
        args.control_offset,
        args.flow_points,
        args.seed + 1,
        args.device,
        aligned_noise=False,
    )
    models = bridge.fit_models(
        calibration,
        args.control_offset,
        args.bucket_count,
        args.radius,
        args.rank,
        args.ridge_alpha,
    )
    evaluation = collect_sequences(
        policy.model,
        policy,
        evaluation_tensors,
        args.control_offset,
        args.flow_points,
        max(SKIPS),
        args.seed + 2,
        args.device,
    )
    rows, region_summary, teacher_difference = evaluate(
        policy.model,
        policy,
        evaluation,
        models,
        args.control_offset,
        args.bucket_count,
        args.late_flow_count,
        args.device,
    )
    base.write_csv(args.output_dir / "exact_suffix_metrics.csv", rows)
    base.write_csv(
        args.output_dir / "endpoint_summary.csv", aggregate_endpoints(rows)
    )
    with (args.output_dir / "split_indices.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "train_dataset_length": train_length,
                "validation_dataset_length": validation_length,
                "calibration_previous": calibration_pair[0].tolist(),
                "calibration_current": calibration_pair[1].tolist(),
                "evaluation_previous": evaluation_pair[0].tolist(),
                "evaluation_current": evaluation_pair[1].tolist(),
                "endpoint_scheduler_indices": evaluation[
                    "scheduler_index"
                ][:, -1].tolist(),
            },
            handle,
            indent=2,
        )
    summary = {
        "scope": (
            "Frozen PushT teacher-forced denoising-time coefficient recurrence; "
            "exact previous-control cache and exact current latent; no sampler, "
            "environment, repeated-control, quantization, or speed claim"
        ),
        "checkpoint": str(args.checkpoint),
        "control_offset": args.control_offset,
        "horizon": int(cfg.horizon),
        "calibration_transitions": args.calibration_transitions,
        "evaluation_transitions": args.evaluation_transitions,
        "endpoint_count": int(evaluation["timestep"].shape[0]),
        "late_flow_count": args.late_flow_count,
        "skips": list(SKIPS),
        "decision": decision(region_summary, teacher_difference),
        "exact_suffix": region_summary,
        "cost": cost_proxy(
            models,
            int(cfg.horizon),
            args.control_offset,
            int(cfg.policy.model.n_emb),
            4 * int(cfg.policy.model.n_emb),
        ),
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
