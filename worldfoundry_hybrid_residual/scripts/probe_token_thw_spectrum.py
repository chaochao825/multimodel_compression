#!/usr/bin/env python3
"""Audit true spatiotemporal token spectra in captured Wan Q/K/V tensors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("densities must lie in (0, 1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--channel-samples", type=int, default=64)
    parser.add_argument(
        "--densities",
        type=parse_float_list,
        default=parse_float_list("0.015625,0.0625,0.125,0.25"),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_grid(frame_num: int, tokens: int, height: int, width: int) -> tuple[int, int, int]:
    temporal = (frame_num - 1) // 4 + 1
    if temporal * height * width != tokens:
        raise ValueError(
            f"grid mismatch: frame_num={frame_num} gives T={temporal}, "
            f"but {temporal}*{height}*{width}!={tokens}"
        )
    return temporal, height, width


def sampled_channels(tensor: torch.Tensor, count: int, device: torch.device) -> torch.Tensor:
    flat = tensor[0].reshape(tensor.shape[1], -1)
    count = min(count, flat.shape[1])
    indices = torch.linspace(0, flat.shape[1] - 1, count, dtype=torch.float64).round().long()
    return flat.index_select(1, indices).to(device=device, dtype=torch.float32)


def controls(
    grid: torch.Tensor, generator: torch.Generator
) -> list[tuple[str, torch.Tensor]]:
    temporal, height, width, channels = grid.shape
    tokens = temporal * height * width
    flat = grid.reshape(tokens, channels)
    token_order = torch.randperm(tokens, generator=generator, device=grid.device)
    frame_order = torch.randperm(temporal, generator=generator, device=grid.device)
    spatial_order = torch.randperm(height * width, generator=generator, device=grid.device)
    spatial = grid.reshape(temporal, height * width, channels).index_select(1, spatial_order)
    gaussian = torch.randn(
        grid.shape, generator=generator, device=grid.device, dtype=grid.dtype
    )
    gaussian = gaussian * grid.std(unbiased=False).clamp_min(1e-12) + grid.mean()
    return [
        ("original", grid),
        ("token_shuffled", flat.index_select(0, token_order).reshape_as(grid)),
        ("frame_shuffled", grid.index_select(0, frame_order)),
        ("spatial_shuffled", spatial.reshape_as(grid)),
        ("matched_gaussian", gaussian),
    ]


def lowpass_masks(
    shape: tuple[int, int, int], densities: tuple[float, ...], device: torch.device
) -> dict[float, torch.Tensor]:
    axes = [torch.fft.fftfreq(size, device=device).abs() / 0.5 for size in shape]
    radius = (
        axes[0][:, None, None].square()
        + axes[1][None, :, None].square()
        + axes[2][None, None, :].square()
    )
    sorted_radius = radius.flatten().sort().values
    masks: dict[float, torch.Tensor] = {}
    for density in densities:
        requested = max(1, min(radius.numel(), round(radius.numel() * density)))
        threshold = sorted_radius[requested - 1]
        masks[density] = radius <= threshold
    return masks


def neighbor_metrics(grid: torch.Tensor, axis: int) -> tuple[float, float]:
    if grid.shape[axis] <= 1:
        return float("nan"), float("nan")
    left = grid.narrow(axis, 0, grid.shape[axis] - 1).reshape(-1)
    right = grid.narrow(axis, 1, grid.shape[axis] - 1).reshape(-1)
    cosine = torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12)
    centered = grid - grid.mean(dim=(0, 1, 2), keepdim=True)
    left_centered = centered.narrow(axis, 0, centered.shape[axis] - 1)
    right_centered = centered.narrow(axis, 1, centered.shape[axis] - 1)
    variance = centered.square().mean().clamp_min(1e-12)
    normalized_difference = (right_centered - left_centered).square().mean() / (2.0 * variance)
    return float(cosine), float(normalized_difference)


def spectrum_rows(
    *,
    case: str,
    signal: str,
    control: str,
    centering: str,
    grid: torch.Tensor,
    masks: dict[float, torch.Tensor],
) -> list[dict[str, object]]:
    if centering == "centered":
        grid = grid - grid.mean(dim=(0, 1, 2), keepdim=True)
    transformed = torch.fft.fftn(grid, dim=(0, 1, 2), norm="ortho")
    power = transformed.abs().square().sum(dim=-1)
    total = power.sum().clamp_min(1e-30)
    dc_ratio = float(power[0, 0, 0] / total)
    sorted_power = power.flatten().sort(descending=True).values
    cumulative = sorted_power.cumsum(0)
    rows: list[dict[str, object]] = []
    for requested_density, mask in masks.items():
        selected = int(mask.sum().item())
        oracle_count = max(1, min(power.numel(), round(power.numel() * requested_density)))
        retained = float(power[mask].sum() / total)
        oracle = float(cumulative[oracle_count - 1] / total)
        rows.append(
            {
                "case": case,
                "signal": signal,
                "control": control,
                "centering": centering,
                "requested_density": requested_density,
                "actual_lowpass_density": selected / power.numel(),
                "lowpass_retained_energy": retained,
                "lowpass_relative_l2": math.sqrt(max(0.0, 1.0 - retained)),
                "shared_frequency_topk_oracle_energy": oracle,
                "dc_energy_ratio": dc_ratio,
                "tokens": power.numel(),
                "channels_sampled": grid.shape[-1],
            }
        )
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    spectral_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    replay_metadata: list[dict[str, object]] = []
    started = time.time()

    for replay_index, replay_path in enumerate(args.replay):
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        frame_num = int(metadata["frame_num"])
        tokens = int(payload["q"].shape[1])
        shape = infer_grid(frame_num, tokens, args.grid_height, args.grid_width)
        case = f"f{frame_num}_l{metadata.get('layer', 'unknown')}_t{metadata.get('timestep', 'unknown')}"
        replay_metadata.append({"path": str(replay_path), "case": case, **metadata})
        masks = lowpass_masks(shape, args.densities, device)

        for signal in ("q", "k", "v"):
            sampled = sampled_channels(payload[signal], args.channel_samples, device)
            grid = sampled.reshape(*shape, sampled.shape[-1])
            for control_index, (control, controlled) in enumerate(controls(grid, generator)):
                for centering in ("raw", "centered"):
                    spectral_rows.extend(
                        spectrum_rows(
                            case=case,
                            signal=signal,
                            control=control,
                            centering=centering,
                            grid=controlled,
                            masks=masks,
                        )
                    )
                if control in {"original", "token_shuffled", "matched_gaussian"}:
                    axis_names = ("temporal", "height", "width")
                    for axis, axis_name in enumerate(axis_names):
                        cosine, normalized_difference = neighbor_metrics(controlled, axis)
                        geometry_rows.append(
                            {
                                "case": case,
                                "signal": signal,
                                "control": control,
                                "axis": axis_name,
                                "neighbor_cosine": cosine,
                                "normalized_difference_energy": normalized_difference,
                                "axis_length": controlled.shape[axis],
                            }
                        )
                del controlled
            print(f"[thw] case={case} signal={signal}", flush=True)
        del payload

    write_csv(args.output_dir / "token_thw_spectrum.csv", spectral_rows)
    write_csv(args.output_dir / "token_thw_geometry.csv", geometry_rows)
    manifest = {
        "arguments": {
            key: [str(item) for item in value] if key == "replay" else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "replays": replay_metadata,
        "methodology": {
            "channel_sampling": "evenly spaced flattened head channels",
            "frequency_axis": "true Wan latent patch grid T x H x W",
            "lowpass": "shared isotropic radius mask with complete radius ties",
            "controls": "token shuffle, frame shuffle, spatial shuffle, matched Gaussian",
            "centering": "raw and per-channel THW-mean removed",
            "warning": "frequency energy is a representation probe, not an FFT-kernel speedup",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
        "spectral_rows": len(spectral_rows),
        "geometry_rows": len(geometry_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[thw] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
