from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import probe_action_dit_transport_cache as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from action_dit_transport_cache import (  # noqa: E402
    DepthwiseTemporalRegressor,
    FrozenBasis,
    RidgeMap,
    captured_energy,
    coefficient_r2,
    flatten_feature_groups,
    horizon_shift,
    oracle_gap_recovery,
    overlap_mask,
    relative_l2,
    reuse_with_exact_tail,
    row_relative_l2,
)


METHODS = (
    "raw_reuse",
    "shift_reuse",
    "shift_local_r2",
    "shift_rank8_feature",
    "shift_rank8_noise",
    "shift_rank8_noisy_delta",
    "shift_rank8_state",
    "shift_rank8_state_noise",
    "shift_rank8_state_noisy_condition",
    "shift_rank8_oracle",
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
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--late-flow-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def zero_tail(values: np.ndarray, valid_positions: np.ndarray) -> np.ndarray:
    output = values.copy()
    output[:, ~valid_positions] = 0
    return output


def timestep_feature(values: np.ndarray, scale: float, dtype) -> np.ndarray:
    return (values.reshape(-1, 1) / max(scale, 1.0)).astype(dtype)


def repeated_condition_delta(
    records: dict[str, np.ndarray], timestep_mask: np.ndarray
) -> np.ndarray:
    condition_delta = records["current_condition"] - records["previous_condition"]
    return np.repeat(
        condition_delta[:, None], timestep_mask.sum(), axis=1
    ).reshape((-1,) + condition_delta.shape[1:])


def cell_tensors(
    records: dict[str, np.ndarray],
    layer: int,
    timestep_mask: np.ndarray,
    offset: int,
) -> dict[str, np.ndarray]:
    current = base.bucket_slice(records["current_residual"], timestep_mask, layer)
    previous = base.bucket_slice(records["previous_residual"], timestep_mask, layer)
    current_input = base.bucket_slice(records["current_input"], timestep_mask, layer)
    previous_input = base.bucket_slice(records["previous_input"], timestep_mask, layer)
    current_noisy = base.bucket_slice(records["current_noisy"], timestep_mask, None)
    previous_noisy = base.bucket_slice(records["previous_noisy"], timestep_mask, None)
    current_noise = base.bucket_slice(records["current_noise"], timestep_mask, None)
    previous_noise = base.bucket_slice(records["previous_noise"], timestep_mask, None)
    valid_positions = overlap_mask(current.shape[1], offset)
    shifted_previous = horizon_shift(previous, offset)
    shifted_input = horizon_shift(previous_input, offset)
    innovation = zero_tail(current - shifted_previous, valid_positions)
    delta_input = zero_tail(current_input - shifted_input, valid_positions)
    delta_noise = zero_tail(
        current_noise - horizon_shift(previous_noise, offset), valid_positions
    )
    delta_noisy = zero_tail(
        current_noisy - horizon_shift(previous_noisy, offset), valid_positions
    )
    previous_flow_innovation = zero_tail(
        base.bucket_slice(
            records["current_flow_residual"], timestep_mask, layer
        )
        - horizon_shift(
            base.bucket_slice(
                records["previous_flow_residual"], timestep_mask, layer
            ),
            offset,
        ),
        valid_positions,
    )
    return {
        "current": current,
        "previous": previous,
        "innovation": innovation,
        "delta_input": delta_input,
        "delta_noise": delta_noise,
        "delta_noisy": delta_noisy,
        "current_noisy": current_noisy,
        "condition_delta": repeated_condition_delta(records, timestep_mask),
        "previous_flow_innovation": previous_flow_innovation,
        "valid_positions": valid_positions,
    }


def bridge_features(
    tensors: dict[str, np.ndarray],
    state_coefficients: np.ndarray,
    timesteps: np.ndarray,
    timestep_scale: float,
) -> dict[str, np.ndarray]:
    time = timestep_feature(
        timesteps, timestep_scale, tensors["delta_noise"].dtype
    )
    return {
        "noise": flatten_feature_groups(tensors["delta_noise"], time),
        "noisy_delta": flatten_feature_groups(tensors["delta_noisy"], time),
        "state": flatten_feature_groups(state_coefficients, time),
        "state_noise": flatten_feature_groups(
            state_coefficients, tensors["delta_noise"], time
        ),
        "state_noisy_condition": flatten_feature_groups(
            state_coefficients,
            tensors["delta_noisy"],
            tensors["condition_delta"],
            time,
        ),
    }


def fit_models(
    records: dict[str, np.ndarray],
    offset: int,
    bucket_count: int,
    radius: int,
    rank: int,
    alpha: float,
) -> dict[tuple[int, int], dict[str, object]]:
    buckets = base.step_buckets(len(records["timestep"]), bucket_count)
    timestep_scale = float(np.max(records["timestep"]))
    models = {}
    layer_count = records["current_residual"].shape[2]
    for layer in range(layer_count):
        for bucket in range(bucket_count):
            timestep_mask = buckets == bucket
            tensors = cell_tensors(records, layer, timestep_mask, offset)
            basis = FrozenBasis.fit(tensors["innovation"], rank)
            coefficients = basis.coefficients(tensors["innovation"])
            state_coefficients = basis.coefficients(
                tensors["previous_flow_innovation"]
            )
            timesteps = np.broadcast_to(
                records["timestep"][timestep_mask][None, :],
                (len(records["current_condition"]), timestep_mask.sum()),
            ).reshape(-1)
            features = bridge_features(
                tensors, state_coefficients, timesteps, timestep_scale
            )
            cheap = base.cheap_features(
                tensors["delta_input"],
                tensors["current_noisy"],
                tensors["condition_delta"],
                timesteps,
            )
            models[(layer, bucket)] = {
                "basis": basis,
                "local": DepthwiseTemporalRegressor(
                    radius, False, alpha
                ).fit(
                    tensors["delta_input"],
                    tensors["innovation"],
                    tensors["valid_positions"],
                ),
                "feature": RidgeMap(alpha).fit(cheap, coefficients),
                "noise": RidgeMap(alpha).fit(features["noise"], coefficients),
                "noisy_delta": RidgeMap(alpha).fit(
                    features["noisy_delta"], coefficients
                ),
                "state": RidgeMap(alpha).fit(features["state"], coefficients),
                "state_noise": RidgeMap(alpha).fit(
                    features["state_noise"], coefficients
                ),
                "state_noisy_condition": RidgeMap(alpha).fit(
                    features["state_noisy_condition"], coefficients
                ),
            }
    return models


def cell_predictions(
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    layer: int,
    timestep_index: int,
    bucket: int,
    offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    timestep_mask = np.zeros(len(records["timestep"]), dtype=bool)
    timestep_mask[timestep_index] = True
    tensors = cell_tensors(records, layer, timestep_mask, offset)
    model = models[(layer, bucket)]
    basis = model["basis"]
    target_coefficients = basis.coefficients(tensors["innovation"])
    state_coefficients = basis.coefficients(tensors["previous_flow_innovation"])
    timestep_values = np.full(
        len(tensors["current"]), records["timestep"][timestep_index]
    )
    features = bridge_features(
        tensors,
        state_coefficients,
        timestep_values,
        float(np.max(records["timestep"])),
    )
    cheap = base.cheap_features(
        tensors["delta_input"],
        tensors["current_noisy"],
        tensors["condition_delta"],
        timestep_values,
    )
    coefficients = {
        "feature": model["feature"].predict(cheap),
        "noise": model["noise"].predict(features["noise"]),
        "noisy_delta": model["noisy_delta"].predict(features["noisy_delta"]),
        "state": model["state"].predict(features["state"]),
        "state_noise": model["state_noise"].predict(features["state_noise"]),
        "state_noisy_condition": model["state_noisy_condition"].predict(
            features["state_noisy_condition"]
        ),
    }
    shifted = reuse_with_exact_tail(
        tensors["previous"], tensors["current"], offset, aligned=True
    )
    predictions = {
        "raw_reuse": reuse_with_exact_tail(
            tensors["previous"], tensors["current"], offset, aligned=False
        ),
        "shift_reuse": shifted,
        "shift_local_r2": shifted
        + zero_tail(
            model["local"].predict(tensors["delta_input"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_feature": shifted
        + zero_tail(
            basis.reconstruct(coefficients["feature"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_noise": shifted
        + zero_tail(
            basis.reconstruct(coefficients["noise"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_noisy_delta": shifted
        + zero_tail(
            basis.reconstruct(coefficients["noisy_delta"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_state": shifted
        + zero_tail(
            basis.reconstruct(coefficients["state"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_state_noise": shifted
        + zero_tail(
            basis.reconstruct(coefficients["state_noise"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_state_noisy_condition": shifted
        + zero_tail(
            basis.reconstruct(coefficients["state_noisy_condition"]),
            tensors["valid_positions"],
        ),
        "shift_rank8_oracle": shifted
        + zero_tail(
            basis.project(tensors["innovation"]), tensors["valid_positions"]
        ),
    }
    diagnostics = {
        "rank8_energy": captured_energy(
            tensors["innovation"], basis.project(tensors["innovation"])
        ),
        **{
            f"{name}_coefficient_r2": coefficient_r2(values, target_coefficients)
            for name, values in coefficients.items()
        },
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
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]]]:
    buckets = base.step_buckets(len(records["timestep"]), bucket_count)
    late_start = len(records["timestep"]) - late_flow_count
    condition = torch.as_tensor(
        records["current_condition"], dtype=policy.dtype, device=device
    )
    rows = []
    accumulators = {
        region: {
            method: {
                "numerator": 0.0,
                "denominator": 0.0,
                "sample_error": [],
            }
            for method in METHODS
        }
        for region in ("all", "late")
    }
    for timestep_index, bucket in enumerate(buckets):
        sample = torch.as_tensor(
            records["current_noisy"][:, timestep_index],
            dtype=policy.dtype,
            device=device,
        )
        timestep = torch.tensor(
            int(records["timestep"][timestep_index]), device=device
        )
        exact = records["exact_output"][:, timestep_index]
        for layer in range(records["current_residual"].shape[2]):
            predictions, diagnostics = cell_predictions(
                records, models, layer, timestep_index, int(bucket), offset
            )
            current = records["current_residual"][:, timestep_index, layer]
            for method, replacement in predictions.items():
                output = base.injected_output(
                    model, layer, replacement, sample, timestep, condition
                )
                sample_error = row_relative_l2(output, exact)
                rows.append(
                    {
                        "layer": layer,
                        "flow_point": timestep_index,
                        "timestep": int(records["timestep"][timestep_index]),
                        "method": method,
                        "velocity_relative_l2": relative_l2(output, exact),
                        "velocity_mean_relative_l2": float(sample_error.mean()),
                        "velocity_p95_relative_l2": float(
                            np.quantile(sample_error, 0.95)
                        ),
                        "activation_relative_l2": relative_l2(
                            replacement, current
                        ),
                        **diagnostics,
                    }
                )
                regions = ["all"]
                if timestep_index >= late_start:
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
    return rows, summaries


def aggregate_geometry(rows: list[dict]) -> list[dict]:
    output = []
    for flow_point in sorted({int(row["flow_point"]) for row in rows}):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if int(row["flow_point"]) == flow_point and row["method"] == method
            ]
            output.append(
                {
                    "flow_point": flow_point,
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
                }
            )
    return output


def decision(summary: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    late = summary["late"]
    raw = late["raw_reuse"]
    shifted = late["shift_reuse"]
    local = late["shift_local_r2"]
    noise = late["shift_rank8_noise"]
    state_noise = late["shift_rank8_state_noise"]
    oracle = late["shift_rank8_oracle"]
    noise_recovery = oracle_gap_recovery(
        shifted["velocity_relative_l2"],
        noise["velocity_relative_l2"],
        oracle["velocity_relative_l2"],
    )
    state_recovery = oracle_gap_recovery(
        shifted["velocity_relative_l2"],
        state_noise["velocity_relative_l2"],
        oracle["velocity_relative_l2"],
    )
    late_shift_pass = (
        local["velocity_relative_l2"] <= 0.8 * raw["velocity_relative_l2"]
        and local["velocity_p95_relative_l2"]
        <= raw["velocity_p95_relative_l2"]
    )
    noise_pass = (
        noise_recovery >= 0.5
        and noise["velocity_relative_l2"] <= 0.8 * local["velocity_relative_l2"]
        and noise["velocity_p95_relative_l2"]
        <= local["velocity_p95_relative_l2"]
    )
    state_pass = (
        state_recovery >= 0.8
        and state_noise["velocity_relative_l2"]
        <= 0.8 * local["velocity_relative_l2"]
        and state_noise["velocity_p95_relative_l2"]
        <= local["velocity_p95_relative_l2"]
    )
    if noise_pass:
        gate = "NOISE_BRIDGE_GO"
    elif state_pass:
        gate = "STATE_BRIDGE_BOUNDARY"
    elif late_shift_pass:
        gate = "LATE_SHIFT_BOUNDARY"
    else:
        gate = "NO_GO"
    return {
        "gate": gate,
        "late_shift_pass": bool(late_shift_pass),
        "noise_bridge_pass": bool(noise_pass),
        "state_bridge_pass": bool(state_pass),
        "noise_oracle_gap_recovery": float(noise_recovery),
        "state_noise_oracle_gap_recovery": float(state_recovery),
    }


def model_costs(
    models: dict[tuple[int, int], dict[str, object]],
    horizon: int,
    offset: int,
    width: int,
    feedforward: int,
) -> dict[str, float]:
    model = next(iter(models.values()))
    basis = model["basis"]
    reusable_tokens = horizon - offset
    basis_macs = reusable_tokens * basis.basis.shape[0] * width
    replaced_macs = reusable_tokens * feedforward * width
    costs = {
        name: basis_macs + model[name].macs_per_sample
        for name in (
            "feature",
            "noise",
            "noisy_delta",
            "state",
            "state_noise",
            "state_noisy_condition",
        )
    }
    return {
        "basis_stored_values_per_cell": int(basis.mean.size + basis.basis.size),
        "replaced_linear2_macs_per_sample": int(replaced_macs),
        **{f"{name}_macs_per_sample": int(value) for name, value in costs.items()},
        **{
            f"{name}_correction_fraction": float(value / replaced_macs)
            for name, value in costs.items()
        },
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
        raise ValueError("B0.5 must use the deployed control offset")
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
    evaluation = base.collect_geometry(
        policy.model,
        policy,
        evaluation_tensors,
        args.control_offset,
        args.flow_points,
        args.seed + 2,
        args.device,
        aligned_noise=False,
    )
    models = fit_models(
        calibration,
        args.control_offset,
        args.bucket_count,
        args.radius,
        args.rank,
        args.ridge_alpha,
    )
    rows, region_summary = evaluate(
        policy.model,
        policy,
        evaluation,
        models,
        args.control_offset,
        args.bucket_count,
        args.late_flow_count,
        args.device,
    )
    write_csv(args.output_dir / "exact_suffix_metrics.csv", rows)
    write_csv(args.output_dir / "flow_summary.csv", aggregate_geometry(rows))
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
            },
            handle,
            indent=2,
        )
    summary = {
        "scope": (
            "Frozen PushT independent-noise B0/B0.5 bridge; exact suffix; "
            "no sampler, environment, scheduler, quantization, or speed claim"
        ),
        "checkpoint": str(args.checkpoint),
        "control_offset": args.control_offset,
        "horizon": int(cfg.horizon),
        "calibration_transitions": args.calibration_transitions,
        "evaluation_transitions": args.evaluation_transitions,
        "flow_points": args.flow_points,
        "late_flow_count": args.late_flow_count,
        "rank": args.rank,
        "decision": decision(region_summary),
        "exact_suffix": region_summary,
        "cost": model_costs(
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
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
