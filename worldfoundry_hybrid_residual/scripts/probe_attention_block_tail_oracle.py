#!/usr/bin/env python3
"""Compare token and GPU-tile sparse critical paths with a low-rank tail."""

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


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty integer list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--query-samples", type=int, default=512)
    parser.add_argument("--token-budgets", type=parse_int_list, default=parse_int_list("64,128,256,512"))
    parser.add_argument("--block-sizes", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--ranks", type=parse_int_list, default=parse_int_list("8,16,32"))
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


def randomized_lowrank(matrix: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=matrix.device).manual_seed(seed)
    oversample = min(matrix.shape[1], rank + 8)
    omega = torch.randn(matrix.shape[1], oversample, device=matrix.device, generator=generator)
    basis, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(2):
        basis, _ = torch.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
    small = basis.T @ matrix
    left, singular, right = torch.linalg.svd(small, full_matrices=False)
    return basis @ left[:, :rank], singular[:rank], right[:rank]


def approximation_metrics(
    attention: torch.Tensor,
    value: torch.Tensor,
    critical: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
) -> dict[str, float]:
    left, singular, right = factors
    left = left[:, :rank]
    singular = singular[:rank]
    right = right[:rank]
    tail_output = (left * singular) @ (right @ value)
    estimate = critical @ value + tail_output
    reference = attention @ value
    lowrank_tail = (left * singular) @ right
    delta_attention = attention - critical - lowrank_tail
    delta_output = estimate - reference
    return {
        "attention_rel_fro": float(delta_attention.norm() / attention.norm()),
        "output_rel_l2": float(delta_output.norm() / reference.norm()),
        "output_cosine": float(F.cosine_similarity(reference.flatten(), estimate.flatten(), dim=0)),
        "critical_mass": float(critical.sum() / attention.sum()),
    }


def token_critical(attention: torch.Tensor, token_budget: int) -> torch.Tensor:
    indices = torch.topk(attention, k=min(token_budget, attention.shape[1]), dim=1).indices
    critical = torch.zeros_like(attention)
    critical.scatter_(1, indices, attention.gather(1, indices))
    return critical


def block_critical(attention: torch.Tensor, token_budget: int, block_size: int) -> tuple[torch.Tensor, int]:
    rows, keys = attention.shape
    blocks = math.ceil(keys / block_size)
    padded_keys = blocks * block_size
    if padded_keys != keys:
        padded = F.pad(attention, (0, padded_keys - keys))
    else:
        padded = attention
    block_mass = padded.view(rows, blocks, block_size).sum(dim=-1)
    selected_blocks = max(1, token_budget // block_size)
    indices = torch.topk(block_mass, k=min(selected_blocks, blocks), dim=1).indices
    mask = torch.zeros_like(block_mass, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    critical = torch.where(
        mask.unsqueeze(-1), padded.view(rows, blocks, block_size), torch.zeros((), device=attention.device)
    ).reshape(rows, padded_keys)[:, :keys]
    return critical, selected_blocks * block_size


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    payload = torch.load(args.qkv_file, map_location="cpu", weights_only=False)
    q = payload["q"].to(device=device, dtype=torch.float32)[0]
    k = payload["k"].to(device=device, dtype=torch.float32)[0]
    v = payload["v"].to(device=device, dtype=torch.float32)[0]
    scale = float(payload.get("softmax_scale", q.shape[-1] ** -0.5))
    query_indices = torch.linspace(0, q.shape[0] - 1, args.query_samples, device=device).round().long()
    rows_out: list[dict[str, object]] = []
    started = time.time()

    for head in range(q.shape[1]):
        scores = q[query_indices, head] @ k[:, head].T * scale
        attention = torch.softmax(scores, dim=-1)
        value = v[:, head]
        for token_budget in args.token_budgets:
            critical_specs = [("token_topk_oracle", token_budget, token_critical(attention, token_budget))]
            for block_size in args.block_sizes:
                critical, actual_tokens = block_critical(attention, token_budget, block_size)
                critical_specs.append((f"block{block_size}_oracle", actual_tokens, critical))
            for method, actual_tokens, critical in critical_specs:
                factors = randomized_lowrank(
                    attention - critical,
                    max(args.ranks),
                    args.seed + head * 101 + token_budget + actual_tokens,
                )
                for rank in args.ranks:
                    metrics = approximation_metrics(
                        attention, value, critical, factors, rank
                    )
                    representation_ratio_full = actual_tokens / attention.shape[1] + 2.0 * rank / attention.shape[1]
                    rows_out.append(
                        {
                            "case": f"f{payload.get('metadata', {}).get('frame_num', 'unknown')}",
                            "head": head,
                            "method": method,
                            "requested_token_budget": token_budget,
                            "actual_critical_tokens": actual_tokens,
                            "rank": rank,
                            "keys": attention.shape[1],
                            "representation_ratio_full_n": representation_ratio_full,
                            **metrics,
                        }
                    )
        print(f"[oracle] head={head}", flush=True)
        write_csv(args.output_dir / "block_tail_oracle.partial.csv", rows_out)

    write_csv(args.output_dir / "block_tail_oracle.csv", rows_out)
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "capture_metadata": payload.get("metadata", {}),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started,
        "scope": "exact-mass representation oracle on sampled queries",
        "warning": "block selection uses dense attention and is not a mask-search or fused-kernel benchmark",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[oracle] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
