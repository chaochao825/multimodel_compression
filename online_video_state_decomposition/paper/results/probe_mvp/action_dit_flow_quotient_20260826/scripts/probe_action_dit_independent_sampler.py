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
from action_dit_transport_cache import overlap_mask  # noqa: E402


SCHEDULES = ("all_interval5", "late20_interval5")


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
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument("--fixed-subsets", action="store_true")
    return parser.parse_args()


class SelectedStepCapture:
    def __init__(self, model, selected_indices: set[int]):
        self.selected_indices = selected_indices
        self.step_index = -1
        self.inputs = [None] * len(model.decoder.layers)
        self.outputs = [None] * len(model.decoder.layers)
        self.records = {}
        self.handles = []
        for layer_index, layer in enumerate(model.decoder.layers):
            self.handles.append(
                layer.linear1.register_forward_pre_hook(
                    self._capture_input(layer_index)
                )
            )
            self.handles.append(
                layer.linear2.register_forward_hook(
                    self._capture_output(layer_index)
                )
            )

    def _capture_input(self, layer_index: int):
        def hook(module, inputs):
            del module
            if self.step_index in self.selected_indices:
                self.inputs[layer_index] = inputs[0].detach()

        return hook

    def _capture_output(self, layer_index: int):
        def hook(module, inputs, output):
            del module, inputs
            if self.step_index in self.selected_indices:
                self.outputs[layer_index] = output.detach()

        return hook

    def save(self, trajectory: torch.Tensor) -> None:
        if self.step_index not in self.selected_indices:
            return
        self.records[self.step_index] = {
            "trajectory": trajectory.float().cpu().numpy(),
            "input": torch.stack(self.inputs, dim=1).float().cpu().numpy(),
            "residual": torch.stack(self.outputs, dim=1).float().cpu().numpy(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def sample_full(
    policy,
    condition: torch.Tensor,
    initial: torch.Tensor,
    scheduler_seed: int,
    selected_indices: set[int],
    device: str,
) -> tuple[torch.Tensor, dict[int, dict[str, np.ndarray]], np.ndarray]:
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    capture = SelectedStepCapture(policy.model, selected_indices)
    trajectory = initial.clone()
    generator = torch.Generator(device=device).manual_seed(scheduler_seed)
    timesteps = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        capture.step_index = step_index
        model_output = policy.model(trajectory, timestep, condition)
        capture.save(trajectory)
        trajectory = scheduler.step(
            model_output,
            timestep,
            trajectory,
            generator=generator,
            **policy.kwargs,
        ).prev_sample
        timesteps.append(int(timestep.item()))
    capture.close()
    return trajectory, capture.records, np.asarray(timesteps)


def independent_initials(
    shape: tuple[int, ...], dtype, device: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    previous = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    current = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    return previous, current


def torch_horizon_shift(values: torch.Tensor, offset: int) -> torch.Tensor:
    if offset < 0 or offset >= values.shape[-2]:
        raise ValueError("offset must be in [0, horizon)")
    shifted = torch.zeros_like(values)
    shifted[..., : values.shape[-2] - offset, :] = values[..., offset:, :]
    return shifted


def sampler_calibration_records(
    policy,
    tensors: dict,
    flow_points: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    previous_condition, current_condition, previous_action, current_action = (
        base.normalized_transition(policy, tensors, device)
    )
    del previous_action, current_action
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    anchor_indices = np.linspace(
        1, len(scheduler.timesteps) - 1, flow_points, dtype=np.int64
    )
    selected = {int(index) for index in anchor_indices}
    selected.update(int(index) - 1 for index in anchor_indices)
    shape = (len(previous_condition), policy.horizon, policy.action_dim)
    previous_initial, current_initial = independent_initials(
        shape, policy.dtype, device, seed
    )
    _, previous_records, timesteps = sample_full(
        policy,
        previous_condition,
        previous_initial,
        seed + 1,
        selected,
        device,
    )
    _, current_records, _ = sample_full(
        policy,
        current_condition,
        current_initial,
        seed + 2,
        selected,
        device,
    )
    batch = len(previous_condition)
    output = {
        "previous_input": np.stack(
            [previous_records[int(index)]["input"] for index in anchor_indices],
            axis=1,
        ),
        "current_input": np.stack(
            [current_records[int(index)]["input"] for index in anchor_indices],
            axis=1,
        ),
        "previous_residual": np.stack(
            [previous_records[int(index)]["residual"] for index in anchor_indices],
            axis=1,
        ),
        "current_residual": np.stack(
            [current_records[int(index)]["residual"] for index in anchor_indices],
            axis=1,
        ),
        "previous_flow_input": np.stack(
            [previous_records[int(index) - 1]["input"] for index in anchor_indices],
            axis=1,
        ),
        "current_flow_input": np.stack(
            [current_records[int(index) - 1]["input"] for index in anchor_indices],
            axis=1,
        ),
        "previous_flow_residual": np.stack(
            [
                previous_records[int(index) - 1]["residual"]
                for index in anchor_indices
            ],
            axis=1,
        ),
        "current_flow_residual": np.stack(
            [current_records[int(index) - 1]["residual"] for index in anchor_indices],
            axis=1,
        ),
        "previous_noisy": np.stack(
            [previous_records[int(index)]["trajectory"] for index in anchor_indices],
            axis=1,
        ),
        "current_noisy": np.stack(
            [current_records[int(index)]["trajectory"] for index in anchor_indices],
            axis=1,
        ),
        "previous_noise": np.repeat(
            previous_initial.float().cpu().numpy()[:, None], flow_points, axis=1
        ),
        "current_noise": np.repeat(
            current_initial.float().cpu().numpy()[:, None], flow_points, axis=1
        ),
        "timestep": timesteps[anchor_indices],
        "previous_condition": previous_condition.float().cpu().numpy(),
        "current_condition": current_condition.float().cpu().numpy(),
    }
    if output["current_residual"].shape[0] != batch:
        raise ValueError("sampler calibration batch changed during capture")
    return output


def torch_cell(model: dict[str, object], device: str, dtype) -> dict[str, torch.Tensor]:
    basis = model["basis"]
    state_noise = model["state_noise"]
    return {
        "mean": torch.as_tensor(basis.mean, dtype=dtype, device=device),
        "basis": torch.as_tensor(basis.basis, dtype=dtype, device=device),
        "x_mean": torch.as_tensor(
            state_noise.x_mean, dtype=dtype, device=device
        ),
        "x_scale": torch.as_tensor(
            state_noise.x_scale, dtype=dtype, device=device
        ),
        "weight": torch.as_tensor(
            state_noise.weight, dtype=dtype, device=device
        ),
        "y_mean": torch.as_tensor(
            state_noise.y_mean, dtype=dtype, device=device
        ),
    }


class SamplerTQC:
    def __init__(
        self,
        model,
        models: dict[tuple[int, int], dict[str, object]],
        reference_indices: np.ndarray,
        bucket_count: int,
        offset: int,
        delta_noise: torch.Tensor,
        timestep_scale: float,
        device: str,
        dtype,
        active_layers: set[int] | None,
    ):
        self.model = model
        self.reference_indices = reference_indices
        self.bucket_count = bucket_count
        self.offset = offset
        self.delta_noise = delta_noise.reshape(len(delta_noise), -1)
        self.timestep_scale = timestep_scale
        self.cells = {
            key: torch_cell(value, device, dtype) for key, value in models.items()
        }
        self.state = [None] * len(model.decoder.layers)
        self.state_cell = [None] * len(model.decoder.layers)
        self.previous_cache = None
        self.scheduler_index = -1
        self.timestep = 0
        self.approximate = False
        self.approximate_calls = 0
        self.total_calls = 0
        selected_layers = (
            set(range(len(model.decoder.layers)))
            if active_layers is None
            else active_layers
        )
        self.handles = [
            model.decoder.layers[layer_index].linear2.register_forward_hook(
                self._hook(layer_index)
            )
            for layer_index in sorted(selected_layers)
        ]

    def _bucket(self) -> int:
        return base.step_buckets(
            len(self.reference_indices), self.bucket_count
        )[
            int(
                np.argmin(
                    np.abs(self.reference_indices - self.scheduler_index)
                )
            )
        ]

    @staticmethod
    def _transfer(
        coefficients: torch.Tensor,
        source: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        rotation = source["basis"] @ target["basis"].T
        offset = (source["mean"] - target["mean"]) @ target["basis"].T
        return coefficients @ rotation + offset

    def _hook(self, layer_index: int):
        def hook(module, inputs, output):
            del module, inputs
            bucket = int(self._bucket())
            cell = self.cells[(layer_index, bucket)]
            previous = self.previous_cache[:, layer_index]
            overlap = output.shape[1] - self.offset
            shifted = previous[:, self.offset :]
            if not self.approximate:
                innovation = torch.zeros_like(output)
                innovation[:, :overlap] = output[:, :overlap] - shifted
                flat = innovation.reshape(len(output), -1)
                self.state[layer_index] = (flat - cell["mean"]) @ cell[
                    "basis"
                ].T
                self.state_cell[layer_index] = cell
                return None
            coefficients = self.state[layer_index]
            source = self.state_cell[layer_index]
            if source is not cell:
                coefficients = self._transfer(coefficients, source, cell)
            time = torch.full(
                (len(output), 1),
                float(self.timestep) / max(self.timestep_scale, 1.0),
                dtype=output.dtype,
                device=output.device,
            )
            features = torch.cat([coefficients, self.delta_noise, time], dim=1)
            normalized = (features - cell["x_mean"]) / cell["x_scale"]
            coefficients = normalized @ cell["weight"] + cell["y_mean"]
            innovation = (cell["mean"] + coefficients @ cell["basis"]).reshape_as(
                output
            )
            replacement = output.clone()
            replacement[:, :overlap] = shifted + innovation[:, :overlap]
            self.state[layer_index] = coefficients
            self.state_cell[layer_index] = cell
            return replacement

        return hook

    def begin_step(
        self,
        scheduler_index: int,
        timestep: int,
        previous_cache: torch.Tensor,
        approximate: bool,
    ) -> None:
        self.scheduler_index = scheduler_index
        self.timestep = timestep
        self.previous_cache = previous_cache
        self.approximate = approximate
        self.total_calls += 1
        if approximate:
            self.approximate_calls += 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def is_approximate_step(schedule: str, step_index: int, step_count: int) -> bool:
    if schedule == "all_interval5":
        return step_index % 5 != 0
    if schedule == "late20_interval5":
        start = int(np.floor(0.8 * step_count))
        return step_index >= start and (step_index - start) % 5 != 0
    raise ValueError(f"unknown sampler schedule: {schedule}")


@torch.inference_mode()
def sample_tqc(
    policy,
    condition: torch.Tensor,
    initial: torch.Tensor,
    scheduler_seed: int,
    previous_records: dict[int, dict[str, np.ndarray]],
    models: dict[tuple[int, int], dict[str, object]],
    reference_indices: np.ndarray,
    bucket_count: int,
    offset: int,
    previous_initial: torch.Tensor,
    schedule: str,
    timestep_scale: float,
    device: str,
    active_layers: set[int] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    valid = overlap_mask(initial.shape[1], offset)
    delta_noise = initial - torch_horizon_shift(previous_initial, offset)
    delta_noise[:, ~torch.as_tensor(valid, device=device)] = 0
    executor = SamplerTQC(
        policy.model,
        models,
        reference_indices,
        bucket_count,
        offset,
        delta_noise,
        timestep_scale,
        device,
        initial.dtype,
        active_layers,
    )
    trajectory = initial.clone()
    generator = torch.Generator(device=device).manual_seed(scheduler_seed)
    step_count = len(scheduler.timesteps)
    for step_index, timestep in enumerate(scheduler.timesteps):
        previous_cache = torch.as_tensor(
            previous_records[step_index]["residual"],
            dtype=trajectory.dtype,
            device=device,
        )
        executor.begin_step(
            step_index,
            int(timestep.item()),
            previous_cache,
            is_approximate_step(schedule, step_index, step_count),
        )
        model_output = policy.model(trajectory, timestep, condition)
        trajectory = scheduler.step(
            model_output,
            timestep,
            trajectory,
            generator=generator,
            **policy.kwargs,
        ).prev_sample
    diagnostics = {
        "approximate_steps": executor.approximate_calls,
        "total_steps": executor.total_calls,
    }
    executor.close()
    return trajectory, diagnostics


def physical_actions(policy, normalized: torch.Tensor) -> dict[str, np.ndarray]:
    action_pred = policy.normalizer["action"].unnormalize(normalized).float()
    start = policy.n_obs_steps - 1
    end = start + policy.n_action_steps
    executed = action_pred[:, start:end]
    return {
        "horizon": action_pred.cpu().numpy(),
        "executed": executed.cpu().numpy(),
        "first": executed[:, :1].cpu().numpy(),
    }


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    sample_numerator = np.linalg.norm(difference.reshape(len(difference), -1), axis=1)
    sample_denominator = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    sample_relative = sample_numerator / np.maximum(sample_denominator, 1e-12)
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "mean_relative_l2": float(sample_relative.mean()),
        "p95_relative_l2": float(np.quantile(sample_relative, 0.95)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def sampler_decision(metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    passes = {}
    for schedule in SCHEDULES:
        passes[schedule] = bool(
            metrics[schedule]["horizon"]["relative_l2"] <= 0.01
            and metrics[schedule]["horizon"]["p95_relative_l2"] <= 0.02
            and metrics[schedule]["executed"]["relative_l2"] <= 0.01
            and metrics[schedule]["executed"]["p95_relative_l2"] <= 0.02
            and metrics[schedule]["first"]["p95_relative_l2"] <= 0.02
        )
    if passes["all_interval5"]:
        gate = "SAMPLER_ALL_GO"
    elif passes["late20_interval5"]:
        gate = "SAMPLER_LATE_BOUNDARY"
    else:
        gate = "NO_GO"
    return {"gate": gate, "schedule_pass": passes}


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
    if not policy.obs_as_cond:
        raise ValueError("B1a currently requires observation cross-conditioning")
    if args.control_offset != int(cfg.n_action_steps):
        raise ValueError("B1a must use the deployed control offset")
    calibration = sampler_calibration_records(
        policy,
        calibration_tensors,
        args.flow_points,
        args.seed + 10,
        args.device,
    )
    models = bridge.fit_models(
        calibration,
        args.control_offset,
        args.bucket_count,
        args.radius,
        args.rank,
        args.ridge_alpha,
    )
    previous_condition, current_condition, previous_action, current_action = (
        base.normalized_transition(policy, evaluation_tensors, args.device)
    )
    del previous_action, current_action
    shape = (len(previous_condition), policy.horizon, policy.action_dim)
    previous_initial, current_initial = independent_initials(
        shape, policy.dtype, args.device, args.seed + 20
    )
    all_steps = set(range(policy.num_inference_steps))
    _, previous_records, timesteps = sample_full(
        policy,
        previous_condition,
        previous_initial,
        args.seed + 21,
        all_steps,
        args.device,
    )
    full_normalized, _, _ = sample_full(
        policy,
        current_condition,
        current_initial,
        args.seed + 22,
        set(),
        args.device,
    )
    full_actions = physical_actions(policy, full_normalized)
    schedule_metrics = {}
    schedule_diagnostics = {}
    reference_indices = np.linspace(
        1, policy.num_inference_steps - 1, args.flow_points, dtype=np.int64
    )
    for schedule in SCHEDULES:
        candidate_normalized, diagnostics = sample_tqc(
            policy,
            current_condition,
            current_initial,
            args.seed + 22,
            previous_records,
            models,
            reference_indices,
            args.bucket_count,
            args.control_offset,
            previous_initial,
            schedule,
            float(np.max(calibration["timestep"])),
            args.device,
            None,
        )
        candidate_actions = physical_actions(policy, candidate_normalized)
        schedule_metrics[schedule] = {
            region: error_metrics(candidate_actions[region], full_actions[region])
            for region in ("horizon", "executed", "first")
        }
        schedule_diagnostics[schedule] = diagnostics
    layer_sweep = {}
    if args.layer_sweep:
        for layer in range(len(policy.model.decoder.layers)):
            candidate_normalized, diagnostics = sample_tqc(
                policy,
                current_condition,
                current_initial,
                args.seed + 22,
                previous_records,
                models,
                reference_indices,
                args.bucket_count,
                args.control_offset,
                previous_initial,
                "all_interval5",
                float(np.max(calibration["timestep"])),
                args.device,
                {layer},
            )
            candidate_actions = physical_actions(policy, candidate_normalized)
            layer_sweep[str(layer)] = {
                "metrics": {
                    region: error_metrics(
                        candidate_actions[region], full_actions[region]
                    )
                    for region in ("horizon", "executed", "first")
                },
                "schedule": diagnostics,
            }
    fixed_subsets = {}
    if args.fixed_subsets:
        subsets = {
            "layers_1_2": {1, 2},
            "layers_1_7": set(range(1, len(policy.model.decoder.layers))),
        }
        for name, layers in subsets.items():
            candidate_normalized, diagnostics = sample_tqc(
                policy,
                current_condition,
                current_initial,
                args.seed + 22,
                previous_records,
                models,
                reference_indices,
                args.bucket_count,
                args.control_offset,
                previous_initial,
                "all_interval5",
                float(np.max(calibration["timestep"])),
                args.device,
                layers,
            )
            candidate_actions = physical_actions(policy, candidate_normalized)
            fixed_subsets[name] = {
                "layers": sorted(layers),
                "metrics": {
                    region: error_metrics(
                        candidate_actions[region], full_actions[region]
                    )
                    for region in ("horizon", "executed", "first")
                },
                "schedule": diagnostics,
            }
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
            "Frozen PushT independent-noise DDPM sampler with common random "
            "numbers; full versus simulated TQC; no environment or speed claim"
        ),
        "checkpoint": str(args.checkpoint),
        "control_offset": args.control_offset,
        "horizon": int(cfg.horizon),
        "executed_actions": int(cfg.n_action_steps),
        "sampler_steps": int(policy.num_inference_steps),
        "calibration_transitions": args.calibration_transitions,
        "evaluation_transitions": args.evaluation_transitions,
        "rank": args.rank,
        "metrics": schedule_metrics,
        "schedule": schedule_diagnostics,
        "layer_sweep": layer_sweep,
        "fixed_subsets": fixed_subsets,
        "decision": sampler_decision(schedule_metrics),
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
