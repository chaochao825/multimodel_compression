#!/usr/bin/env python3
"""Run the EXP-046 target-visible whole-block rank-state capacity probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_SCRIPT_DIR = SCRIPT_DIR.parent.parent / "scripts"
if REPOSITORY_SCRIPT_DIR.is_dir():
    sys.path.insert(0, str(REPOSITORY_SCRIPT_DIR))

from current_input_observability_core import relative_l2_terms, rollout_predict
from generate_wan_h200_v4 import (
    AttentionDispatcher,
    install_grid_compatibility,
    load_backends,
)
from rank_state_capacity_core import (
    estimated_wan_block_macs,
    randomized_rank_state_spectrum,
    state_capacity_rows,
)
from run_wan_current_input_observability import (
    debug_payload,
    parse_indices,
    run_dense_trajectory,
    write_csv,
    write_json,
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


def validate_plan(
    config: dict[str, object], prompts: list[str], sample_indices: tuple[int, ...]
) -> None:
    sample_plan = config["sample_plan"]
    if sample_indices != tuple(sorted(sample_indices)):
        raise ValueError("sample indices must follow the frozen split order")
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(sample_plan):
            raise ValueError("sample index lies outside the frozen plan")
        item = sample_plan[sample_index]
        if item["split"] not in config["allowed_initial_splits"]:
            raise PermissionError(f"split {item['split']} remains locked")
        if int(item["prompt_index"]) >= len(prompts):
            raise ValueError("sample plan references an unavailable prompt")


@torch.inference_mode()
def evaluate_cell(
    trajectory: object,
    *,
    sample_id: str,
    split: str,
    block: int,
    branch: int,
    config: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    ranks = tuple(int(value) for value in config["ranks"])
    svd = config["randomized_svd"]
    max_rank = ranks[-1]
    exact_macs = estimated_wan_block_macs(
        tokens=trajectory.token_count,
        hidden_size=int(config["hidden_size"]),
        ffn_size=int(config["ffn_size"]),
    )
    rows: list[dict[str, object]] = []
    for target_value in config["target_steps"]:
        target_step = int(target_value)
        target_residual = trajectory.residual[target_step]
        target_output = trajectory.block_input[target_step] + target_residual
        residual_target_sq = float(target_residual.double().square().sum())
        output_target_sq = float(target_output.double().square().sum())
        for horizon_value in config["horizons"]:
            horizon = int(horizon_value)
            torch.cuda.synchronize(device)
            base_started = time.perf_counter()
            base = rollout_predict(
                trajectory,
                method="diagonal",
                target_step=target_step,
                horizon=horizon,
                ridge=float(config["ridge"]),
            ).prediction
            torch.cuda.synchronize(device)
            base_ms = 1000 * (time.perf_counter() - base_started)
            defect = target_residual - base
            _, defect_sq, _ = relative_l2_terms(base, target_residual)
            base_renderer_macs = 4 * horizon * trajectory.token_count * trajectory.channel_count
            common = {
                "sample_id": sample_id,
                "split": split,
                "block": block,
                "target_step": target_step,
                "branch": branch,
                "horizon": horizon,
                "target_visible": True,
                "base_renderer": str(config["base_renderer"]),
                "residual_target_sq": residual_target_sq,
                "output_target_sq": output_target_sq,
                "defect_total_energy": defect_sq,
                "tokens": trajectory.token_count,
                "channels": trajectory.channel_count,
                "estimated_exact_block_macs": exact_macs,
                "base_renderer_macs": base_renderer_macs,
                "base_fit_and_rollout_ms": base_ms,
            }
            rows.append(
                {
                    **common,
                    "rank": 0,
                    "error_sq": defect_sq,
                    "residual_relative_l2": (defect_sq / residual_target_sq) ** 0.5,
                    "output_relative_l2": (defect_sq / output_target_sq) ** 0.5,
                    "defect_remaining_energy": 1.0,
                    "state_factor_values": 0,
                    "state_factor_bytes_fp16": 0,
                    "render_macs": 0,
                    "render_to_exact_macs": 0.0,
                    "base_plus_render_macs": base_renderer_macs,
                    "base_plus_render_to_exact_macs": base_renderer_macs / exact_macs,
                    "svd_ms": 0.0,
                }
            )
            cell_seed = (
                int(svd["seed"])
                + 10000 * block
                + 1000 * branch
                + 10 * target_step
                + horizon
            )
            torch.cuda.synchronize(device)
            svd_started = time.perf_counter()
            spectrum = randomized_rank_state_spectrum(
                defect,
                max_rank=max_rank,
                oversample=int(svd["oversample"]),
                power_iterations=int(svd["power_iterations"]),
                seed=cell_seed,
            )
            torch.cuda.synchronize(device)
            svd_ms = 1000 * (time.perf_counter() - svd_started)
            capacity = state_capacity_rows(
                spectrum,
                ranks=ranks,
                residual_target_sq=residual_target_sq,
                output_target_sq=output_target_sq,
                estimated_exact_block_macs=exact_macs,
            )
            for row in capacity:
                render_macs = int(row["render_macs"])
                rows.append(
                    {
                        **common,
                        **row,
                        "state_factor_bytes_fp16": 2 * int(row["state_factor_values"]),
                        "base_plus_render_macs": base_renderer_macs + render_macs,
                        "base_plus_render_to_exact_macs": (
                            base_renderer_macs + render_macs
                        )
                        / exact_macs,
                        "svd_ms": svd_ms,
                    }
                )
            del base, defect, spectrum
    return rows


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
    sample_indices = parse_indices(args.sample_indices)
    validate_plan(config, prompts, sample_indices)
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
    recorder = None
    all_rows: list[dict[str, object]] = []
    sample_summaries: list[dict[str, object]] = []
    load_started = time.perf_counter()
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
        dense_reference = None
        if args.verify_equivalence:
            item = config["sample_plan"][sample_indices[0]]
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
            item = config["sample_plan"][sample_index]
            prompt_index = int(item["prompt_index"])
            seed = int(item["seed"])
            split = str(item["split"])
            sample_id = f"p{prompt_index:02d}_seed{seed}"
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
                equivalence_relative_l2 = relative_l2_terms(latent, dense_reference)[0]
                equivalence_exact = bool(torch.equal(latent, dense_reference))
                if equivalence_relative_l2 > 1e-7:
                    raise RuntimeError(
                        "rank-state capture changed the dense final latent: "
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
                    trajectory = recorder.cell_trajectory(block, branch, device=device)
                    sample_rows.extend(
                        evaluate_cell(
                            trajectory,
                            sample_id=sample_id,
                            split=split,
                            block=block,
                            branch=branch,
                            config=config,
                            device=device,
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
        "warning": "target-visible representation capacity only; no observer, rollout, latency, or speed claim",
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
