#!/usr/bin/env python3
"""Test whether THW-lowpass Wan Q/K can preserve attention or route critical keys."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("densities must lie in (0, 1]")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--query-samples", type=int, default=256)
    parser.add_argument(
        "--densities",
        type=parse_float_list,
        default=parse_float_list("0.015625,0.0625,0.125,0.25"),
    )
    parser.add_argument(
        "--router-topk", type=parse_int_list, default=parse_int_list("64,128,256")
    )
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
        raise ValueError("captured token count does not match the requested THW grid")
    return temporal, height, width


def radius_and_thresholds(
    shape: tuple[int, int, int], densities: tuple[float, ...], device: torch.device
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
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
        masks[density] = radius <= sorted_radius[requested - 1]
    return radius, masks


def centered_fft(grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = grid.mean(dim=(0, 1, 2), keepdim=True)
    return torch.fft.fftn(grid - mean, dim=(0, 1, 2), norm="ortho"), mean


def lowpass_reconstruct(
    transformed: torch.Tensor, mean: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    filtered = torch.where(mask[..., None], transformed, torch.zeros((), device=transformed.device))
    return torch.fft.ifftn(filtered, dim=(0, 1, 2), norm="ortho").real + mean


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-30))


def router_metrics(
    exact_attention: torch.Tensor,
    approximate_attention: torch.Tensor,
    topk: int,
) -> tuple[float, float]:
    topk = min(topk, exact_attention.shape[1])
    exact_indices = torch.topk(exact_attention, k=topk, dim=1).indices
    approximate_indices = torch.topk(approximate_attention, k=topk, dim=1).indices
    exact_mask = torch.zeros_like(exact_attention, dtype=torch.bool)
    exact_mask.scatter_(1, exact_indices, True)
    recall = float(exact_mask.gather(1, approximate_indices).float().mean())
    critical_mass = float(exact_attention.gather(1, approximate_indices).sum(dim=1).mean())
    return recall, critical_mass


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    detail_rows: list[dict[str, object]] = []
    router_rows: list[dict[str, object]] = []
    replay_metadata: list[dict[str, object]] = []
    started = time.time()

    for replay_path in args.replay:
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        q_all = payload["q"][0]
        k_all = payload["k"][0]
        v_all = payload["v"][0]
        tokens, heads, dimension = q_all.shape
        frame_num = int(metadata["frame_num"])
        shape = infer_grid(frame_num, tokens, args.grid_height, args.grid_width)
        case = f"f{frame_num}_l{metadata.get('layer', 'unknown')}_t{metadata.get('timestep', 'unknown')}"
        replay_metadata.append({"path": str(replay_path), "case": case, **metadata})
        _, masks = radius_and_thresholds(shape, args.densities, device)
        query_indices = torch.linspace(0, tokens - 1, args.query_samples, dtype=torch.float64).round().long()

        for head in range(heads):
            q = q_all[:, head].to(device=device, dtype=torch.float32)
            k = k_all[:, head].to(device=device, dtype=torch.float32)
            v = v_all[:, head].to(device=device, dtype=torch.float32)
            q_grid = q.reshape(*shape, dimension)
            k_grid = k.reshape(*shape, dimension)
            q_fft, q_mean = centered_fft(q_grid)
            k_fft, k_mean = centered_fft(k_grid)
            q_power = q_fft.abs().square().sum(dim=-1)
            k_power = k_fft.abs().square().sum(dim=-1)
            q_total = q_power.sum().clamp_min(1e-30)
            k_total = k_power.sum().clamp_min(1e-30)
            sampled_q = q.index_select(0, query_indices.to(device))
            scale = float(payload.get("softmax_scale", dimension**-0.5))
            exact_scores = sampled_q @ k.T * scale
            exact_attention = torch.softmax(exact_scores, dim=-1)
            exact_output = exact_attention @ v

            for density, mask in masks.items():
                q_hat = lowpass_reconstruct(q_fft, q_mean, mask).reshape(tokens, dimension)
                k_hat = lowpass_reconstruct(k_fft, k_mean, mask).reshape(tokens, dimension)
                q_retained = float(q_power[mask].sum() / q_total)
                k_retained = float(k_power[mask].sum() / k_total)
                variants = {
                    "q_lowpass": (q_hat.index_select(0, query_indices.to(device)), k),
                    "k_lowpass": (sampled_q, k_hat),
                    "qk_lowpass": (q_hat.index_select(0, query_indices.to(device)), k_hat),
                }
                for variant, (query, key) in variants.items():
                    scores = query @ key.T * scale
                    attention = torch.softmax(scores, dim=-1)
                    output = attention @ v
                    centered_exact_scores = exact_scores - exact_scores.mean(dim=1, keepdim=True)
                    centered_scores = scores - scores.mean(dim=1, keepdim=True)
                    kl = float(
                        (
                            exact_attention
                            * (
                                exact_attention.clamp_min(1e-30).log()
                                - attention.clamp_min(1e-30).log()
                            )
                        )
                        .sum(dim=1)
                        .mean()
                    )
                    detail_rows.append(
                        {
                            "case": case,
                            "head": head,
                            "variant": variant,
                            "requested_density": density,
                            "actual_density": float(mask.float().mean()),
                            "q_retained_energy": q_retained,
                            "k_retained_energy": k_retained,
                            "q_relative_l2": relative_l2(sampled_q, query),
                            "k_relative_l2": relative_l2(k, key),
                            "centered_score_relative_l2": relative_l2(
                                centered_exact_scores, centered_scores
                            ),
                            "attention_relative_fro": relative_l2(exact_attention, attention),
                            "attention_kl_exact_to_approx": kl,
                            "output_relative_l2": relative_l2(exact_output, output),
                            "output_cosine": float(
                                F.cosine_similarity(exact_output.flatten(), output.flatten(), dim=0)
                            ),
                        }
                    )
                    for topk in args.router_topk:
                        recall, mass = router_metrics(exact_attention, attention, topk)
                        router_rows.append(
                            {
                                "case": case,
                                "head": head,
                                "variant": variant,
                                "requested_density": density,
                                "actual_density": float(mask.float().mean()),
                                "router_topk": topk,
                                "exact_topk_recall": recall,
                                "exact_attention_mass_captured": mass,
                            }
                        )
                del q_hat, k_hat
            print(f"[router] case={case} head={head}", flush=True)
            del q, k, v, q_fft, k_fft, exact_scores, exact_attention, exact_output
            torch.cuda.empty_cache()
        del payload

    write_csv(args.output_dir / "spectral_qk_attention.csv", detail_rows)
    write_csv(args.output_dir / "spectral_qk_router.csv", router_rows)
    manifest = {
        "arguments": {
            key: [str(item) for item in value] if key == "replay" else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "replays": replay_metadata,
        "methodology": {
            "capture_scope": "post-RoPE Q/K passed to Wan self-attention; V is unchanged",
            "attention": "dense sampled-query softmax in FP32",
            "router": "top-k selected from approximate attention, scored against exact attention",
            "warning": "FFT reconstruction still computes dense QK and is not a deployable speedup",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
        "detail_rows": len(detail_rows),
        "router_rows": len(router_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[router] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
