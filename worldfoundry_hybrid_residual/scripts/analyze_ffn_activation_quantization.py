#!/usr/bin/env python3
"""Evaluate held-out activation quantization scales from sampled Wan FFN rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


LIMITS = {"fp8_e4m3": 448.0, "int8": 127.0, "int4": 7.0}
FIXED_SCHEMES = (
    "tensor_global",
    "tensor_bucket",
    "tensor_step",
    "tensor_branch_bucket",
    "channel_global",
    "channel_bucket",
)
DYNAMIC_SCHEMES = ("tensor_dynamic", "token_dynamic", "token_group128_dynamic")


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("scale margins must be at least one")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bucket-size", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--scale-margins",
        type=parse_float_list,
        default=parse_float_list("1.0,1.05,1.1,1.25"),
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


def calibration_key(scheme: str, record: dict[str, object], bucket_size: int) -> tuple[object, ...]:
    base = (record["signal"], int(record["block"]))
    bucket = int(record["step"]) // bucket_size
    if scheme in {"tensor_global", "channel_global"}:
        return base
    if scheme in {"tensor_bucket", "channel_bucket"}:
        return base + (bucket,)
    if scheme == "tensor_step":
        return base + (int(record["step"]),)
    if scheme == "tensor_branch_bucket":
        return base + (int(record["branch"]), bucket)
    raise KeyError(scheme)


def build_calibration(
    records: list[dict[str, object]], bucket_size: int
) -> dict[str, dict[tuple[object, ...], torch.Tensor]]:
    calibration: dict[str, dict[tuple[object, ...], torch.Tensor]] = {
        scheme: {} for scheme in FIXED_SCHEMES
    }
    for record in records:
        sample = record["sample"].float()
        tensor_max = sample.abs().amax()
        channel_max = sample.abs().amax(dim=0)
        for scheme in FIXED_SCHEMES:
            key = calibration_key(scheme, record, bucket_size)
            value = channel_max if scheme.startswith("channel") else tensor_max
            previous = calibration[scheme].get(key)
            calibration[scheme][key] = value.clone() if previous is None else torch.maximum(previous, value)
    return calibration


def dynamic_amax(sample: torch.Tensor, scheme: str, group_size: int) -> tuple[torch.Tensor, int, int]:
    if scheme == "tensor_dynamic":
        return sample.abs().amax(), 0, sample.shape[1]
    if scheme == "token_dynamic":
        return sample.abs().amax(dim=1, keepdim=True), 0, sample.shape[1]
    if scheme == "token_group128_dynamic":
        padded_features = math.ceil(sample.shape[1] / group_size) * group_size
        padded = F.pad(sample, (0, padded_features - sample.shape[1]))
        grouped = padded.reshape(sample.shape[0], -1, group_size)
        return grouped.abs().amax(dim=-1, keepdim=True), padded_features, group_size
    raise KeyError(scheme)


def quantize(
    sample: torch.Tensor,
    *,
    dtype: str,
    amax: torch.Tensor,
    padded_features: int = 0,
    group_size: int = 0,
) -> tuple[torch.Tensor, float, int]:
    limit = LIMITS[dtype]
    scale = (amax / limit).clamp_min(torch.finfo(torch.float32).tiny)
    if padded_features:
        padded = F.pad(sample, (0, padded_features - sample.shape[1]))
        grouped = padded.reshape(sample.shape[0], -1, group_size)
        normalized = (grouped / scale).clamp(-limit, limit)
        saturation = float((grouped.abs() > scale * limit).float().mean())
        if dtype == "fp8_e4m3":
            restored = normalized.to(torch.float8_e4m3fn).float() * scale
        else:
            restored = torch.round(normalized) * scale
        return restored.reshape(sample.shape[0], padded_features)[:, : sample.shape[1]], saturation, scale.numel()
    normalized = (sample / scale).clamp(-limit, limit)
    saturation = float((sample.abs() > scale * limit).float().mean())
    if dtype == "fp8_e4m3":
        restored = normalized.to(torch.float8_e4m3fn).float() * scale
    else:
        restored = torch.round(normalized) * scale
    return restored, saturation, scale.numel()


def distribution_row(record: dict[str, object]) -> dict[str, object]:
    sample = record["sample"].float()
    absolute = sample.abs().flatten()
    channel_max = sample.abs().amax(dim=0)
    return {
        "run_id": record["run_id"],
        "step": record["step"],
        "block": record["block"],
        "branch": record["branch"],
        "signal": record["signal"],
        "rows": sample.shape[0],
        "features": sample.shape[1],
        "mean": float(sample.mean()),
        "rms": float(sample.square().mean().sqrt()),
        "abs_mean": float(absolute.mean()),
        "abs_p99": float(torch.quantile(absolute, 0.99)),
        "abs_p999": float(torch.quantile(absolute, 0.999)),
        "abs_max": float(absolute.max()),
        "max_over_rms": float(absolute.max() / sample.square().mean().sqrt().clamp_min(1e-30)),
        "channel_max_p50": float(torch.quantile(channel_max, 0.5)),
        "channel_max_p99": float(torch.quantile(channel_max, 0.99)),
        "channel_max_ratio_p99_p50": float(
            torch.quantile(channel_max, 0.99) / torch.quantile(channel_max, 0.5).clamp_min(1e-30)
        ),
        "near_zero_1e3": float((absolute < 1e-3).float().mean()),
        "exact_zero": float((absolute == 0).float().mean()),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dtype"],
            row["scheme"],
            row["scale_margin"],
            row["signal"],
            row["block"],
        )
        groups[key].append(row)
    summary: list[dict[str, object]] = []
    for (dtype, scheme, scale_margin, signal, block), group in sorted(groups.items(), key=str):
        errors = sorted(float(row["relative_l2"]) for row in group)
        cosines = [float(row["cosine"]) for row in group]
        saturations = [float(row["saturation_ratio"]) for row in group]
        p95_index = min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)
        summary.append(
            {
                "dtype": dtype,
                "scheme": scheme,
                "scale_margin": scale_margin,
                "signal": signal,
                "block": block,
                "records": len(group),
                "mean_relative_l2": statistics.fmean(errors),
                "p95_relative_l2": errors[p95_index],
                "max_relative_l2": max(errors),
                "mean_cosine": statistics.fmean(cosines),
                "mean_saturation_ratio": statistics.fmean(saturations),
                "mean_scale_count": statistics.fmean(float(row["scale_count"]) for row in group),
            }
        )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.bucket_size <= 0 or args.group_size <= 0:
        raise ValueError("bucket and group sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = list(payload["records"])
    run_ids = sorted({str(record["run_id"]) for record in records})
    if len(run_ids) < 2:
        raise ValueError("held-out calibration requires at least two runs")
    calibration_run = run_ids[0]
    calibration_records = [record for record in records if record["run_id"] == calibration_run]
    evaluation_records = [record for record in records if record["run_id"] != calibration_run]
    calibration = build_calibration(calibration_records, args.bucket_size)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    detail_rows: list[dict[str, object]] = []
    distribution_rows = [distribution_row(record) for record in records]
    started = time.time()

    for record_index, record in enumerate(evaluation_records):
        sample = record["sample"].to(device=device, dtype=torch.float32)
        for dtype in LIMITS:
            for scheme in FIXED_SCHEMES:
                key = calibration_key(scheme, record, args.bucket_size)
                calibrated_amax = calibration[scheme][key].to(
                    device=device, dtype=torch.float32
                )
                for margin in args.scale_margins:
                    restored, saturation, scale_count = quantize(
                        sample, dtype=dtype, amax=calibrated_amax * margin
                    )
                    relative_l2 = float(
                        (restored - sample).norm() / sample.norm().clamp_min(1e-30)
                    )
                    cosine = float(
                        F.cosine_similarity(restored.flatten(), sample.flatten(), dim=0)
                    )
                    detail_rows.append(
                        {
                            "run_id": record["run_id"],
                            "step": record["step"],
                            "block": record["block"],
                            "branch": record["branch"],
                            "signal": record["signal"],
                            "dtype": dtype,
                            "scheme": scheme,
                            "scale_margin": margin,
                            "relative_l2": relative_l2,
                            "cosine": cosine,
                            "saturation_ratio": saturation,
                            "scale_count": scale_count,
                        }
                    )
            for scheme in DYNAMIC_SCHEMES:
                amax, padded_features, group_size = dynamic_amax(sample, scheme, args.group_size)
                restored, saturation, scale_count = quantize(
                    sample,
                    dtype=dtype,
                    amax=amax,
                    padded_features=padded_features,
                    group_size=group_size,
                )
                relative_l2 = float((restored - sample).norm() / sample.norm().clamp_min(1e-30))
                cosine = float(F.cosine_similarity(restored.flatten(), sample.flatten(), dim=0))
                detail_rows.append(
                    {
                        "run_id": record["run_id"],
                        "step": record["step"],
                        "block": record["block"],
                        "branch": record["branch"],
                        "signal": record["signal"],
                        "dtype": dtype,
                        "scheme": scheme,
                        "scale_margin": 1.0,
                        "relative_l2": relative_l2,
                        "cosine": cosine,
                        "saturation_ratio": saturation,
                        "scale_count": scale_count,
                    }
                )
        if (record_index + 1) % 80 == 0:
            print(f"[quant] records={record_index + 1}/{len(evaluation_records)}", flush=True)

    summary_rows = summarize(detail_rows)
    write_csv(args.output_dir / "activation_distribution.csv", distribution_rows)
    write_csv(args.output_dir / "activation_quantization_detail.csv", detail_rows)
    write_csv(args.output_dir / "activation_quantization_summary.csv", summary_rows)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "calibration_run": calibration_run,
        "evaluation_runs": sorted(set(run_ids) - {calibration_run}),
        "records_total": len(records),
        "calibration_records": len(calibration_records),
        "evaluation_records": len(evaluation_records),
        "methodology": {
            "split": "first lexicographic run calibrates scales; remaining runs are held out",
            "fixed_scales": list(FIXED_SCHEMES),
            "dynamic_upper_bounds": list(DYNAMIC_SCHEMES),
            "warning": "sampled activation rows; local error is not end-to-end video quality",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[quant] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
