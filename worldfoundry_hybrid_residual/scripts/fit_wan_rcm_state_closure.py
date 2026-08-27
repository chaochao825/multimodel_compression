#!/usr/bin/env python3
"""Fit calibration-only low-rate closure states for EXP-048."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rcm_state_closure_core import (
    ClosureTrajectory,
    fit_basis,
    fit_stagewise_transitions,
)


MODELS = ("teacher", "rcm")
TRAJECTORIES = ("native4", "rcm4")
BASIS_SCOPES = ("model_specific", "shared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-048" or config["gate_id"] != "G-027":
        raise ValueError("config is not the frozen EXP-048/G-027 configuration")
    return config


def load_payload(path: Path, config: dict[str, object]) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["experiment_id"] != config["experiment_id"]:
        raise ValueError("calibration payload experiment identity mismatch")
    if payload["split"] != "calibration":
        raise ValueError("closure fitting accepts calibration payloads only")
    expected = tuple(config["splits"]["calibration"])
    if tuple(payload["sample_indices"]) != expected:
        raise ValueError("calibration payload does not contain the frozen identities")
    return payload


def trajectory_from_record(record: dict[str, torch.Tensor]) -> ClosureTrajectory:
    trajectory = ClosureTrajectory(
        block_input=record["block_input"],
        residual=record["residual"],
    )
    trajectory.validate()
    return trajectory


def calibration_trajectories(
    payload: dict[str, object], model: str, trajectory: str, block: int
) -> list[ClosureTrajectory]:
    return [
        trajectory_from_record(sample["captures"][model][trajectory][block])
        for sample in payload["samples"]
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    payload = load_payload(args.calibration_payload.resolve(), config)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite closure model: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fitting was requested but CUDA is unavailable")
    ranks = tuple(int(value) for value in config["state"]["ranks"])
    max_rank = max(ranks)
    blocks = tuple(int(value) for value in config["capture"]["blocks"])
    fit_config = config["state"]["fit"]
    bases: dict[str, object] = {"model_specific": {}, "shared": {}}
    diagnostics: dict[str, object] = {"model_specific": {}, "shared": {}}
    seed = int(fit_config["seed"])

    for model_index, model in enumerate(MODELS):
        bases["model_specific"][model] = {}
        diagnostics["model_specific"][model] = {}
        for block in blocks:
            matrices = [
                item.residual
                for trajectory in TRAJECTORIES
                for item in calibration_trajectories(payload, model, trajectory, block)
            ]
            basis, detail = fit_basis(
                matrices,
                rank=max_rank,
                oversampling=int(fit_config["oversampling"]),
                power_iterations=int(fit_config["power_iterations"]),
                seed=seed + 1000 * model_index + block,
                device=device,
            )
            bases["model_specific"][model][block] = basis
            diagnostics["model_specific"][model][block] = detail

    for block in blocks:
        matrices = [
            item.residual
            for model in MODELS
            for trajectory in TRAJECTORIES
            for item in calibration_trajectories(payload, model, trajectory, block)
        ]
        basis, detail = fit_basis(
            matrices,
            rank=max_rank,
            oversampling=int(fit_config["oversampling"]),
            power_iterations=int(fit_config["power_iterations"]),
            seed=seed + 3000 + block,
            device=device,
        )
        bases["shared"][block] = basis
        diagnostics["shared"][block] = detail

    transitions: dict[str, object] = {}
    for scope in BASIS_SCOPES:
        transitions[scope] = {}
        for model in MODELS:
            transitions[scope][model] = {}
            for trajectory in TRAJECTORIES:
                transitions[scope][model][trajectory] = {}
                for block in blocks:
                    basis = (
                        bases["model_specific"][model][block]
                        if scope == "model_specific"
                        else bases["shared"][block]
                    )
                    transitions[scope][model][trajectory][block] = (
                        fit_stagewise_transitions(
                            calibration_trajectories(
                                payload, model, trajectory, block
                            ),
                            basis,
                            ridge=float(fit_config["ridge"]),
                            device=device,
                        )
                    )

    artifact = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "sample_indices": payload["sample_indices"],
        "ranks": ranks,
        "max_rank": max_rank,
        "blocks": blocks,
        "basis_scopes": BASIS_SCOPES,
        "models": MODELS,
        "trajectories": TRAJECTORIES,
        "bases": bases,
        "basis_diagnostics": diagnostics,
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "calibration_samples": list(payload["sample_indices"]),
                "ranks": list(ranks),
                "blocks": list(blocks),
                "basis_diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
