#!/usr/bin/env python3
"""Evaluate mixed-bit FFN weight splits on captured held-out Wan activations."""

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
from safetensors import safe_open


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("integer values must be nonnegative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=parse_int_list, default=parse_int_list("0,12,24,29"))
    parser.add_argument("--ranks", type=parse_int_list, default=parse_int_list("0,8,16,32,64"))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--evaluation-run", default="")
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


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected one safetensors file under {path}")
    return candidates[0]


def orient(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.shape[0] <= weight.shape[1]:
        return weight, False
    return weight.T.contiguous(), True


def groupwise_int4(matrix: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    padded_columns = math.ceil(matrix.shape[1] / group_size) * group_size
    padded = F.pad(matrix, (0, padded_columns - matrix.shape[1]))
    grouped = padded.reshape(padded.shape[0], -1, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 7.0
    restored = torch.round(grouped / scales).clamp(-7, 7) * scales
    return restored.reshape(padded.shape)[:, : matrix.shape[1]], scales.numel()


def tensor_fp8(matrix: torch.Tensor) -> torch.Tensor:
    scale = matrix.abs().amax().clamp_min(1e-12) / 448.0
    return (matrix / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def metrics(
    weight: torch.Tensor, estimate: torch.Tensor, inputs: torch.Tensor
) -> dict[str, float]:
    reference = inputs @ weight.T
    output = inputs @ estimate.T
    return {
        "weight_relative_fro": float((estimate - weight).norm() / weight.norm()),
        "activation_output_relative_l2": float(
            (output - reference).norm() / reference.norm().clamp_min(1e-30)
        ),
        "activation_output_cosine": float(
            F.cosine_similarity(output.flatten(), reference.flatten(), dim=0)
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("group size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint)
    activation_payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = list(activation_payload["records"])
    run_ids = sorted({str(record["run_id"]) for record in records})
    evaluation_run = args.evaluation_run or run_ids[-1]
    if evaluation_run not in run_ids:
        raise ValueError(f"evaluation run {evaluation_run!r} is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    started = time.time()

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for block in args.blocks:
            specifications = (
                (0, "ffn_input"),
                (2, "ffn_hidden_post_gelu"),
            )
            for linear, signal in specifications:
                key = f"blocks.{block}.ffn.{linear}.weight"
                selected = [
                    record["sample"]
                    for record in records
                    if record["run_id"] == evaluation_run
                    and int(record["block"]) == block
                    and record["signal"] == signal
                ]
                if not selected:
                    raise ValueError(f"no activation samples for block={block} signal={signal}")
                inputs = torch.cat(selected, dim=0).to(device=device, dtype=torch.float32)
                weight = handle.get_tensor(key).to(device=device, dtype=torch.float32)
                if inputs.shape[1] != weight.shape[1]:
                    raise ValueError(f"input dimension mismatch for {key}")
                matrix, transposed = orient(weight)
                _, eigenvectors = torch.linalg.eigh(matrix @ matrix.T)
                eigenvectors = eigenvectors.flip(1)

                fp8 = tensor_fp8(weight)
                rows.append(
                    {
                        "evaluation_run": evaluation_run,
                        "weight_key": key,
                        "block": block,
                        "linear": linear,
                        "signal": signal,
                        "method": "fp8_tensor",
                        "rank": 0,
                        "activation_rows": inputs.shape[0],
                        "stored_bits_ratio_vs_bf16": 0.5,
                        **metrics(weight, fp8, inputs),
                    }
                )

                for rank in args.ranks:
                    if rank > matrix.shape[0]:
                        continue
                    if rank == 0:
                        components = [("int4_groupwise", torch.zeros_like(matrix))]
                    else:
                        top = eigenvectors[:, :rank]
                        spectral = (top @ (top.T @ matrix)).to(torch.bfloat16).float()
                        random_basis, _ = torch.linalg.qr(
                            torch.randn(
                                matrix.shape[0],
                                rank,
                                generator=generator,
                                device=device,
                                dtype=matrix.dtype,
                            ),
                            mode="reduced",
                        )
                        random_component = (
                            random_basis @ (random_basis.T @ matrix)
                        ).to(torch.bfloat16).float()
                        components = [
                            ("spectral_fp16_plus_int4", spectral),
                            ("random_fp16_plus_int4", random_component),
                        ]
                    for method, protected in components:
                        quantized_residual, scale_count = groupwise_int4(
                            matrix - protected, args.group_size
                        )
                        estimate_matrix = protected + quantized_residual
                        estimate = estimate_matrix.T.contiguous() if transposed else estimate_matrix
                        lowrank_bits = 16 * rank * (matrix.shape[0] + matrix.shape[1])
                        main_bits = 4 * matrix.numel() + 16 * scale_count
                        rows.append(
                            {
                                "evaluation_run": evaluation_run,
                                "weight_key": key,
                                "block": block,
                                "linear": linear,
                                "signal": signal,
                                "method": method,
                                "rank": rank,
                                "activation_rows": inputs.shape[0],
                                "stored_bits_ratio_vs_bf16": (lowrank_bits + main_bits)
                                / (16 * matrix.numel()),
                                "group_size": args.group_size,
                                "scale_count": scale_count,
                                **metrics(weight, estimate, inputs),
                            }
                        )
                print(f"[activation-split] {key}", flush=True)
                del inputs, weight, matrix, eigenvectors
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "weight_split_activation_metrics.csv", rows)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "checkpoint": str(checkpoint),
        "samples": str(args.samples),
        "evaluation_run": evaluation_run,
        "methodology": {
            "activation": "all sampled FFN rows from held-out run across 20 steps and both CFG branches",
            "mixed_bit": "BF16 rank-r spectral or random left subspace plus row-group INT4 residual",
            "warning": "local linear output error; no fused kernel or diffusion rollout",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
        "rows": len(rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[activation-split] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
