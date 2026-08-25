from __future__ import annotations

import argparse
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

from action_dit_transport_cache import (  # noqa: E402
    DepthwiseTemporalRegressor,
    FrozenBasis,
    RidgeMap,
    captured_energy,
    coefficient_r2,
    horizon_shift,
    overlap_mask,
    relative_l2,
    reuse_with_exact_tail,
    row_relative_l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-policy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-offset", type=int, required=True)
    parser.add_argument("--calibration-transitions", type=int, default=96)
    parser.add_argument("--evaluation-transitions", type=int, default=48)
    parser.add_argument("--flow-points", type=int, default=10)
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_subset(values: np.ndarray, count: int) -> np.ndarray:
    if count > len(values):
        raise ValueError(f"requested {count} transitions from {len(values)} candidates")
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[positions]


def transition_pairs(dataset, offset: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = dataset.sampler.indices
    horizon = dataset.horizon
    episode_ends = dataset.replay_buffer.episode_ends[:]
    full_windows = {}
    episode_ids = {}
    for dataset_index, row in enumerate(indices):
        start, end, sample_start, sample_end = [int(value) for value in row]
        if end - start != horizon or sample_start != 0 or sample_end != horizon:
            continue
        full_windows[start] = dataset_index
        episode_ids[start] = int(np.searchsorted(episode_ends, start, side="right"))
    starts = []
    for start in sorted(full_windows):
        current_start = start + offset
        if current_start not in full_windows:
            continue
        if episode_ids[start] != episode_ids[current_start]:
            continue
        starts.append(start)
    selected = deterministic_subset(np.asarray(starts, dtype=np.int64), count)
    previous = np.asarray([full_windows[int(start)] for start in selected], dtype=np.int64)
    current = np.asarray(
        [full_windows[int(start + offset)] for start in selected], dtype=np.int64
    )
    return previous, current


def load_transition_tensors(dataset, previous: np.ndarray, current: np.ndarray):
    previous_items = [dataset[int(index)] for index in previous]
    current_items = [dataset[int(index)] for index in current]
    return {
        "previous_obs": torch.stack([item["obs"] for item in previous_items]),
        "current_obs": torch.stack([item["obs"] for item in current_items]),
        "previous_action": torch.stack([item["action"] for item in previous_items]),
        "current_action": torch.stack([item["action"] for item in current_items]),
    }


def load_policy_and_data(args: argparse.Namespace):
    sys.path.insert(0, str(args.diffusion_policy_root))
    os.chdir(args.diffusion_policy_root)
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    workspace_class = hydra.utils.get_class(payload["cfg"]._target_)
    workspace = workspace_class(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model
    policy.eval().to(args.device)
    dataset = hydra.utils.instantiate(payload["cfg"].task.dataset)
    validation_dataset = dataset.get_validation_dataset()
    calibration_pair = transition_pairs(
        dataset, args.control_offset, args.calibration_transitions
    )
    evaluation_pair = transition_pairs(
        validation_dataset, args.control_offset, args.evaluation_transitions
    )
    calibration = load_transition_tensors(dataset, *calibration_pair)
    evaluation = load_transition_tensors(validation_dataset, *evaluation_pair)
    return (
        payload["cfg"],
        policy,
        calibration,
        evaluation,
        calibration_pair,
        evaluation_pair,
        len(dataset),
        len(validation_dataset),
    )


class FFNCapture:
    def __init__(self, model):
        self.inputs = [None] * len(model.decoder.layers)
        self.outputs = [None] * len(model.decoder.layers)
        self.handles = []
        for layer_index, layer in enumerate(model.decoder.layers):
            self.handles.append(
                layer.linear1.register_forward_pre_hook(self._capture_input(layer_index))
            )
            self.handles.append(
                layer.linear2.register_forward_hook(self._capture_output(layer_index))
            )

    def _capture_input(self, layer_index: int):
        def hook(module, inputs):
            del module
            self.inputs[layer_index] = inputs[0].detach()

        return hook

    def _capture_output(self, layer_index: int):
        def hook(module, inputs, output):
            del module, inputs
            self.outputs[layer_index] = output.detach()

        return hook

    def clear(self) -> None:
        self.inputs = [None] * len(self.inputs)
        self.outputs = [None] * len(self.outputs)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def captured_forward(model, capture: FFNCapture, sample, timestep, condition):
    capture.clear()
    output = model(sample, timestep, condition)
    inputs = torch.stack(capture.inputs, dim=1)
    residuals = torch.stack(capture.outputs, dim=1)
    return (
        output.float().cpu().numpy(),
        inputs.float().cpu().numpy(),
        residuals.float().cpu().numpy(),
    )


def shifted_noise(previous_noise: torch.Tensor, offset: int, generator) -> torch.Tensor:
    output = torch.empty_like(previous_noise)
    output[:, : previous_noise.shape[1] - offset] = previous_noise[:, offset:]
    output[:, previous_noise.shape[1] - offset :] = torch.randn(
        output[:, previous_noise.shape[1] - offset :].shape,
        dtype=output.dtype,
        device=output.device,
        generator=generator,
    )
    return output


def normalized_transition(policy, tensors: dict, device: str):
    previous_obs = policy.normalizer["obs"].normalize(tensors["previous_obs"].to(device))
    current_obs = policy.normalizer["obs"].normalize(tensors["current_obs"].to(device))
    previous_action = policy.normalizer["action"].normalize(
        tensors["previous_action"].to(device)
    )
    current_action = policy.normalizer["action"].normalize(
        tensors["current_action"].to(device)
    )
    return (
        previous_obs[:, : policy.n_obs_steps],
        current_obs[:, : policy.n_obs_steps],
        previous_action,
        current_action,
    )


@torch.inference_mode()
def collect_geometry(
    model,
    policy,
    tensors: dict,
    offset: int,
    flow_points: int,
    seed: int,
    device: str,
    aligned_noise: bool,
) -> dict[str, np.ndarray]:
    previous_condition, current_condition, previous_action, current_action = (
        normalized_transition(policy, tensors, device)
    )
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    if flow_points >= len(scheduler.timesteps):
        raise ValueError("flow_points must be smaller than the inference schedule")
    anchor_indices = np.linspace(
        1, len(scheduler.timesteps) - 1, flow_points, dtype=np.int64
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    previous_noise = torch.randn(
        previous_action.shape,
        dtype=previous_action.dtype,
        device=device,
        generator=generator,
    )
    if aligned_noise:
        current_noise = shifted_noise(previous_noise, offset, generator)
    else:
        current_noise = torch.randn(
            current_action.shape,
            dtype=current_action.dtype,
            device=device,
            generator=generator,
        )

    capture = FFNCapture(model)
    records = {
        "previous_noise": [],
        "current_noise": [],
        "previous_noisy": [],
        "current_noisy": [],
        "exact_output": [],
        "previous_input": [],
        "current_input": [],
        "previous_residual": [],
        "current_residual": [],
        "previous_flow_input": [],
        "current_flow_input": [],
        "previous_flow_residual": [],
        "current_flow_residual": [],
        "timestep": [],
    }
    for anchor_index in anchor_indices:
        timestep = scheduler.timesteps[int(anchor_index)]
        previous_timestep = scheduler.timesteps[int(anchor_index) - 1]
        timestep_batch = timestep.expand(len(previous_action))
        previous_timestep_batch = previous_timestep.expand(len(previous_action))
        previous_noisy = scheduler.add_noise(
            previous_action, previous_noise, timestep_batch
        )
        current_noisy = scheduler.add_noise(current_action, current_noise, timestep_batch)
        previous_flow_noisy = scheduler.add_noise(
            previous_action, previous_noise, previous_timestep_batch
        )
        current_flow_noisy = scheduler.add_noise(
            current_action, current_noise, previous_timestep_batch
        )

        _, previous_input, previous_residual = captured_forward(
            model, capture, previous_noisy, timestep, previous_condition
        )
        exact_output, current_input, current_residual = captured_forward(
            model, capture, current_noisy, timestep, current_condition
        )
        _, previous_flow_input, previous_flow_residual = captured_forward(
            model,
            capture,
            previous_flow_noisy,
            previous_timestep,
            previous_condition,
        )
        _, current_flow_input, current_flow_residual = captured_forward(
            model,
            capture,
            current_flow_noisy,
            previous_timestep,
            current_condition,
        )
        records["previous_noise"].append(previous_noise.float().cpu().numpy())
        records["current_noise"].append(current_noise.float().cpu().numpy())
        records["previous_noisy"].append(previous_noisy.float().cpu().numpy())
        records["current_noisy"].append(current_noisy.float().cpu().numpy())
        records["exact_output"].append(exact_output)
        records["previous_input"].append(previous_input)
        records["current_input"].append(current_input)
        records["previous_residual"].append(previous_residual)
        records["current_residual"].append(current_residual)
        records["previous_flow_input"].append(previous_flow_input)
        records["current_flow_input"].append(current_flow_input)
        records["previous_flow_residual"].append(previous_flow_residual)
        records["current_flow_residual"].append(current_flow_residual)
        records["timestep"].append(int(timestep.item()))
    capture.close()
    output = {
        key: np.stack(value, axis=1) if key != "timestep" else np.asarray(value)
        for key, value in records.items()
    }
    output["previous_condition"] = previous_condition.float().cpu().numpy()
    output["current_condition"] = current_condition.float().cpu().numpy()
    return output


def step_buckets(flow_points: int, bucket_count: int) -> np.ndarray:
    if flow_points % bucket_count != 0:
        raise ValueError("flow_points must be divisible by bucket_count")
    return np.arange(flow_points, dtype=np.int64) * bucket_count // flow_points


def cheap_features(
    delta_input: np.ndarray,
    current_noisy: np.ndarray,
    condition_delta: np.ndarray,
    timestep: np.ndarray,
) -> np.ndarray:
    timestep_feature = timestep.reshape(-1, 1).astype(delta_input.dtype)
    timestep_feature /= max(float(np.max(timestep)), 1.0)
    return np.concatenate(
        [
            delta_input.mean(axis=1),
            delta_input.std(axis=1),
            np.max(np.abs(delta_input), axis=1),
            current_noisy.reshape(len(current_noisy), -1),
            condition_delta.reshape(len(condition_delta), -1),
            timestep_feature,
        ],
        axis=1,
    )


def bucket_slice(values: np.ndarray, timestep_mask: np.ndarray, layer: int | None):
    if layer is None:
        return values[:, timestep_mask].reshape((-1,) + values.shape[2:])
    return values[:, timestep_mask, layer].reshape((-1,) + values.shape[3:])


def zero_tail(values: np.ndarray, valid_positions: np.ndarray) -> np.ndarray:
    output = values.copy()
    output[:, ~valid_positions] = 0
    return output


def fit_cell_models(
    calibration: dict[str, np.ndarray],
    offset: int,
    bucket_count: int,
    radius: int,
    alpha: float,
) -> dict[tuple[int, int], dict[str, object]]:
    buckets = step_buckets(len(calibration["timestep"]), bucket_count)
    valid_positions = overlap_mask(calibration["current_residual"].shape[-2], offset)
    condition_delta_all = (
        calibration["current_condition"] - calibration["previous_condition"]
    )
    models = {}
    layer_count = calibration["current_residual"].shape[2]
    for layer in range(layer_count):
        for bucket in range(bucket_count):
            timestep_mask = buckets == bucket
            current = bucket_slice(calibration["current_residual"], timestep_mask, layer)
            previous = bucket_slice(
                calibration["previous_residual"], timestep_mask, layer
            )
            current_input = bucket_slice(
                calibration["current_input"], timestep_mask, layer
            )
            previous_input = bucket_slice(
                calibration["previous_input"], timestep_mask, layer
            )
            shifted_previous = horizon_shift(previous, offset)
            shifted_input = horizon_shift(previous_input, offset)
            innovation = zero_tail(current - shifted_previous, valid_positions)
            delta_input = zero_tail(current_input - shifted_input, valid_positions)

            basis8 = FrozenBasis.fit(innovation, rank=8)
            coefficients = basis8.coefficients(innovation)
            current_noisy = bucket_slice(
                calibration["current_noisy"], timestep_mask, None
            )
            condition_delta = np.repeat(
                condition_delta_all[:, None], timestep_mask.sum(), axis=1
            ).reshape((-1,) + condition_delta_all.shape[1:])
            timestep_values = np.broadcast_to(
                calibration["timestep"][timestep_mask][None, :],
                (len(condition_delta_all), timestep_mask.sum()),
            ).reshape(-1)
            features = cheap_features(
                delta_input,
                current_noisy,
                condition_delta,
                timestep_values,
            )

            previous_flow_current = bucket_slice(
                calibration["current_flow_residual"], timestep_mask, layer
            )
            previous_flow_previous = bucket_slice(
                calibration["previous_flow_residual"], timestep_mask, layer
            )
            previous_flow_innovation = zero_tail(
                previous_flow_current - horizon_shift(previous_flow_previous, offset),
                valid_positions,
            )
            previous_flow_coefficients = basis8.coefficients(previous_flow_innovation)

            flow_current_input = bucket_slice(
                calibration["current_input"], timestep_mask, layer
            )
            flow_previous_input = bucket_slice(
                calibration["current_flow_input"], timestep_mask, layer
            )
            flow_current_residual = bucket_slice(
                calibration["current_residual"], timestep_mask, layer
            )
            flow_previous_residual = bucket_slice(
                calibration["current_flow_residual"], timestep_mask, layer
            )
            flow_innovation = flow_current_residual - flow_previous_residual
            flow_delta_input = flow_current_input - flow_previous_input

            models[(layer, bucket)] = {
                "toeplitz": DepthwiseTemporalRegressor(radius, False, alpha).fit(
                    delta_input, innovation, valid_positions
                ),
                "circular": DepthwiseTemporalRegressor(radius, True, alpha).fit(
                    delta_input, innovation, valid_positions
                ),
                "basis8": basis8,
                "feature_map": RidgeMap(alpha).fit(features, coefficients),
                "previous_flow_map": RidgeMap(alpha).fit(
                    previous_flow_coefficients, coefficients
                ),
                "flow_toeplitz": DepthwiseTemporalRegressor(
                    radius, False, alpha
                ).fit(
                    flow_delta_input,
                    flow_innovation,
                    np.ones(flow_innovation.shape[1], dtype=bool),
                ),
                "flow_basis8": FrozenBasis.fit(flow_innovation, rank=8),
            }
    return models


def cell_predictions(
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    layer: int,
    timestep_index: int,
    bucket: int,
    offset: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    model = models[(layer, bucket)]
    current = records["current_residual"][:, timestep_index, layer]
    previous = records["previous_residual"][:, timestep_index, layer]
    current_input = records["current_input"][:, timestep_index, layer]
    previous_input = records["previous_input"][:, timestep_index, layer]
    valid_positions = overlap_mask(current.shape[1], offset)
    shifted_previous = horizon_shift(previous, offset)
    shifted_input = horizon_shift(previous_input, offset)
    innovation = zero_tail(current - shifted_previous, valid_positions)
    delta_input = zero_tail(current_input - shifted_input, valid_positions)

    condition_delta = records["current_condition"] - records["previous_condition"]
    timestep_values = np.full(
        len(current), records["timestep"][timestep_index], dtype=np.float32
    )
    features = cheap_features(
        delta_input,
        records["current_noisy"][:, timestep_index],
        condition_delta,
        timestep_values,
    )
    basis8 = model["basis8"]
    target_coefficients = basis8.coefficients(innovation)
    feature_coefficients = model["feature_map"].predict(features)
    previous_flow_innovation = zero_tail(
        records["current_flow_residual"][:, timestep_index, layer]
        - horizon_shift(
            records["previous_flow_residual"][:, timestep_index, layer], offset
        ),
        valid_positions,
    )
    previous_flow_coefficients = basis8.coefficients(previous_flow_innovation)
    memory_coefficients = model["previous_flow_map"].predict(
        previous_flow_coefficients
    )

    flow_previous = records["current_flow_residual"][:, timestep_index, layer]
    flow_delta_input = (
        records["current_input"][:, timestep_index, layer]
        - records["current_flow_input"][:, timestep_index, layer]
    )
    flow_innovation = current - flow_previous
    flow_toeplitz_correction = model["flow_toeplitz"].predict(flow_delta_input)
    flow_basis8 = model["flow_basis8"]

    raw = reuse_with_exact_tail(previous, current, offset, aligned=False)
    shifted = reuse_with_exact_tail(previous, current, offset, aligned=True)
    toeplitz_correction = zero_tail(
        model["toeplitz"].predict(delta_input), valid_positions
    )
    circular_correction = zero_tail(
        model["circular"].predict(delta_input), valid_positions
    )
    predictions = {
        "raw_reuse": raw,
        "shift_reuse": shifted,
        "shift_toeplitz_r2": shifted + toeplitz_correction,
        "shift_circular_r2": shifted + circular_correction,
        "shift_rank8_feature": shifted + basis8.reconstruct(feature_coefficients),
        "shift_rank8_prev_flow": shifted + basis8.reconstruct(memory_coefficients),
        "shift_rank8_oracle": shifted + basis8.project(innovation),
        "flow_reuse": flow_previous,
        "flow_toeplitz_r2": flow_previous + flow_toeplitz_correction,
        "flow_rank8_oracle": flow_previous + flow_basis8.project(flow_innovation),
    }
    diagnostics = {
        "rank8_energy": captured_energy(innovation, basis8.project(innovation)),
        "feature_coefficient_r2": coefficient_r2(
            feature_coefficients, target_coefficients
        ),
        "previous_flow_coefficient_r2": coefficient_r2(
            memory_coefficients, target_coefficients
        ),
        "flow_rank8_energy": captured_energy(
            flow_innovation, flow_basis8.project(flow_innovation)
        ),
    }
    return innovation, predictions, diagnostics


@torch.inference_mode()
def injected_output(model, layer: int, replacement, sample, timestep, condition):
    replacement_tensor = torch.as_tensor(
        replacement, dtype=sample.dtype, device=sample.device
    )

    def replace(module, inputs, output):
        del module, inputs, output
        return replacement_tensor

    handle = model.decoder.layers[layer].linear2.register_forward_hook(replace)
    result = model(sample, timestep, condition)
    handle.remove()
    return result.float().cpu().numpy()


def evaluate_geometry_only(
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    offset: int,
    bucket_count: int,
    noise_mode: str,
) -> list[dict]:
    buckets = step_buckets(len(records["timestep"]), bucket_count)
    rows = []
    for timestep_index, bucket in enumerate(buckets):
        for layer in range(records["current_residual"].shape[2]):
            _, predictions, diagnostics = cell_predictions(
                records, models, layer, timestep_index, int(bucket), offset
            )
            current = records["current_residual"][:, timestep_index, layer]
            for method, prediction in predictions.items():
                sample_error = row_relative_l2(prediction, current)
                rows.append(
                    {
                        "noise_mode": noise_mode,
                        "layer": layer,
                        "flow_point": timestep_index,
                        "timestep": int(records["timestep"][timestep_index]),
                        "method": method,
                        "activation_relative_l2": relative_l2(prediction, current),
                        "activation_mean_relative_l2": float(sample_error.mean()),
                        "activation_p95_relative_l2": float(
                            np.quantile(sample_error, 0.95)
                        ),
                        **diagnostics,
                    }
                )
    return rows


def evaluate_exact_suffix(
    model,
    policy,
    records: dict[str, np.ndarray],
    models: dict[tuple[int, int], dict[str, object]],
    offset: int,
    bucket_count: int,
    device: str,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    buckets = step_buckets(len(records["timestep"]), bucket_count)
    condition = torch.as_tensor(
        records["current_condition"], dtype=policy.dtype, device=device
    )
    rows = []
    aggregate = {}
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
            _, predictions, diagnostics = cell_predictions(
                records, models, layer, timestep_index, int(bucket), offset
            )
            current = records["current_residual"][:, timestep_index, layer]
            for method, replacement in predictions.items():
                output = injected_output(
                    model, layer, replacement, sample, timestep, condition
                )
                sample_error = row_relative_l2(output, exact)
                activation_error = row_relative_l2(replacement, current)
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
                        "activation_relative_l2": relative_l2(replacement, current),
                        **diagnostics,
                    }
                )
                state = aggregate.setdefault(
                    method,
                    {
                        "numerator": 0.0,
                        "denominator": 0.0,
                        "sample_error": [],
                        "activation_error": [],
                    },
                )
                state["numerator"] += float(np.linalg.norm(output - exact) ** 2)
                state["denominator"] += float(np.linalg.norm(exact) ** 2)
                state["sample_error"].append(sample_error)
                state["activation_error"].append(activation_error)
    summary = {}
    for method, state in aggregate.items():
        sample_error = np.concatenate(state["sample_error"])
        activation_error = np.concatenate(state["activation_error"])
        summary[method] = {
            "velocity_relative_l2": float(
                np.sqrt(state["numerator"] / state["denominator"])
            ),
            "velocity_mean_relative_l2": float(sample_error.mean()),
            "velocity_p95_relative_l2": float(np.quantile(sample_error, 0.95)),
            "activation_mean_relative_l2": float(activation_error.mean()),
            "activation_p95_relative_l2": float(
                np.quantile(activation_error, 0.95)
            ),
        }
    return rows, summary


def aggregate_mechanism(
    exact_suffix: dict[str, dict[str, float]],
    geometry_rows: list[dict],
    cfg,
    offset: int,
) -> dict[str, object]:
    raw = exact_suffix["raw_reuse"]
    shifted = exact_suffix["shift_reuse"]
    toeplitz = exact_suffix["shift_toeplitz_r2"]
    circular = exact_suffix["shift_circular_r2"]
    rank8_rows = [row for row in geometry_rows if row["method"] == "shift_reuse"]
    rank8_energy = float(np.mean([row["rank8_energy"] for row in rank8_rows]))
    feature_r2 = float(
        np.mean([row["feature_coefficient_r2"] for row in rank8_rows])
    )
    previous_flow_r2 = float(
        np.mean([row["previous_flow_coefficient_r2"] for row in rank8_rows])
    )
    horizon = int(cfg.horizon)
    condition_tokens = 1 + int(cfg.n_obs_steps)
    width = int(cfg.policy.model.n_emb)
    layers = int(cfg.policy.model.n_layer)
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
    total_mac = (
        input_mac
        + condition_embedding_mac
        + condition_mlp_mac
        + layers * (self_attention_mac + cross_attention_mac + decoder_ffn_mac)
        + head_mac
    )
    overlap_fraction = (horizon - offset) / horizon
    reusable_fraction = layers * decoder_ffn_mac / total_mac * overlap_fraction
    speed_ceiling = 1.0 / (1.0 - reusable_fraction)
    shift_improvement = 1.0 - shifted["velocity_relative_l2"] / raw[
        "velocity_relative_l2"
    ]
    toeplitz_improvement = 1.0 - toeplitz["velocity_relative_l2"] / shifted[
        "velocity_relative_l2"
    ]
    if shift_improvement < 0 or rank8_energy < 0.5:
        gate = "NO_GO"
    elif (
        shift_improvement >= 0.2
        and toeplitz_improvement >= 0.25
        and rank8_energy >= 0.7
        and toeplitz["velocity_p95_relative_l2"]
        <= shifted["velocity_p95_relative_l2"]
        and speed_ceiling >= 1.2
    ):
        gate = "MECHANISM_GO"
    else:
        gate = "BOUNDARY"
    return {
        "gate": gate,
        "shift_risk_improvement_vs_raw": float(shift_improvement),
        "toeplitz_risk_improvement_vs_shift": float(toeplitz_improvement),
        "circular_risk_improvement_vs_shift": float(
            1.0 - circular["velocity_relative_l2"] / shifted["velocity_relative_l2"]
        ),
        "rank8_heldout_energy_mean": rank8_energy,
        "feature_coefficient_r2_mean": feature_r2,
        "previous_flow_coefficient_r2_mean": previous_flow_r2,
        "overlap_fraction": float(overlap_fraction),
        "decoder_ffn_reusable_denoiser_mac_fraction": float(reusable_fraction),
        "control_tick_only_denoiser_speed_ceiling": float(speed_ceiling),
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
    ) = load_policy_and_data(args)
    if args.control_offset != int(cfg.n_action_steps) and args.control_offset != 1:
        raise ValueError("control offset must be deployed n_action_steps or diagnostic 1")

    model = policy.model
    calibration = collect_geometry(
        model,
        policy,
        calibration_tensors,
        args.control_offset,
        args.flow_points,
        args.seed + 1,
        args.device,
        aligned_noise=True,
    )
    evaluation = collect_geometry(
        model,
        policy,
        evaluation_tensors,
        args.control_offset,
        args.flow_points,
        args.seed + 2,
        args.device,
        aligned_noise=True,
    )
    independent_evaluation = collect_geometry(
        model,
        policy,
        evaluation_tensors,
        args.control_offset,
        args.flow_points,
        args.seed + 2,
        args.device,
        aligned_noise=False,
    )
    models = fit_cell_models(
        calibration,
        args.control_offset,
        args.bucket_count,
        args.radius,
        args.ridge_alpha,
    )
    aligned_geometry = evaluate_geometry_only(
        evaluation,
        models,
        args.control_offset,
        args.bucket_count,
        "aligned",
    )
    independent_geometry = evaluate_geometry_only(
        independent_evaluation,
        models,
        args.control_offset,
        args.bucket_count,
        "independent",
    )
    exact_suffix_rows, exact_suffix_summary = evaluate_exact_suffix(
        model,
        policy,
        evaluation,
        models,
        args.control_offset,
        args.bucket_count,
        args.device,
    )
    mechanism = aggregate_mechanism(
        exact_suffix_summary, aligned_geometry, cfg, args.control_offset
    )

    write_csv(args.output_dir / "geometry_metrics.csv", aligned_geometry + independent_geometry)
    write_csv(args.output_dir / "exact_suffix_metrics.csv", exact_suffix_rows)
    with (args.output_dir / "split_indices.json").open("w", encoding="utf-8") as handle:
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
            "Frozen PushT FFN cache-innovation geometry and exact-suffix probe; "
            "teacher-forced aligned noise; no environment, scheduler, or speed claim"
        ),
        "checkpoint": str(args.checkpoint),
        "control_offset": args.control_offset,
        "deployed_control_offset": int(cfg.n_action_steps),
        "horizon": int(cfg.horizon),
        "calibration_transitions": args.calibration_transitions,
        "evaluation_transitions": args.evaluation_transitions,
        "flow_points": args.flow_points,
        "bucket_count": args.bucket_count,
        "mechanism": mechanism,
        "exact_suffix": exact_suffix_summary,
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
