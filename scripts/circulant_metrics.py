#!/usr/bin/env python3
"""Compute 2D/3D cyclic-shift attention approximation metrics.

Given an attention matrix A over a grid, the nearest circulant/translation
equivariant matrix under Frobenius norm is obtained by averaging A[i, j] over
equal cyclic offsets j - i. This is the directly testable analogue of BCCB for
2D image tokens and a block-circulant extension for T x H x W video tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_shape(text: str) -> tuple[int, ...]:
    shape = tuple(int(part) for part in text.replace("x", ",").split(",") if part)
    if len(shape) not in (2, 3):
        raise argparse.ArgumentTypeError("shape must be H,W or T,H,W")
    if any(v <= 0 for v in shape):
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, help="npz file containing attention or q/k arrays")
    parser.add_argument("--attention-key", default="attention")
    parser.add_argument("--q-key", default="q")
    parser.add_argument("--k-key", default="k")
    parser.add_argument("--shape", type=parse_shape, required=True, help="H,W or T,H,W")
    parser.add_argument("--softmax", action="store_true", help="apply row softmax to logits")
    parser.add_argument("--head-limit", type=int, default=16)
    parser.add_argument("--synthetic", choices=["perfect", "noisy", "random"], help="self-test input")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


def flatten_heads(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[None, :, :]
    if x.ndim == 3:
        return x
    if x.ndim >= 4:
        return x.reshape((-1,) + x.shape[-2:])
    raise ValueError(f"expected attention with at least 2 dims, got {x.shape}")


def attention_from_qk(q: np.ndarray, k: np.ndarray, apply_softmax: bool) -> np.ndarray:
    qh = q.reshape((-1,) + q.shape[-2:])
    kh = k.reshape((-1,) + k.shape[-2:])
    if qh.shape != kh.shape:
        raise ValueError(f"q and k shape mismatch: {q.shape} vs {k.shape}")
    scale = qh.shape[-1] ** -0.5
    logits = np.matmul(qh, np.swapaxes(kh, -1, -2)) * scale
    return softmax(logits, axis=-1) if apply_softmax else logits


def cyclic_projection(attn: np.ndarray, grid_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.prod(grid_shape))
    if attn.shape != (n, n):
        raise ValueError(f"attention shape {attn.shape} does not match grid size {n}")

    arr = attn.reshape(grid_shape + grid_shape)
    kernel = np.zeros(grid_shape, dtype=np.float64)
    proj = np.zeros_like(arr, dtype=np.float64)
    for offset in np.ndindex(grid_shape):
        shifted = arr
        for axis, delta in enumerate(offset):
            shifted = np.take(shifted, indices=np.arange(grid_shape[axis]) + delta, axis=len(grid_shape) + axis, mode="wrap")
        diag_values = np.diagonal(shifted.reshape(n, n))
        mean_value = float(diag_values.mean())
        kernel[offset] = mean_value
        for axis, delta in enumerate(offset):
            pass
    # Build projection in a second explicit pass for clarity and small probe sizes.
    coords = np.array(list(np.ndindex(grid_shape)), dtype=np.int64)
    offsets = (coords[None, :, :] - coords[:, None, :]) % np.array(grid_shape, dtype=np.int64)
    flat_kernel = kernel.reshape(-1)
    multipliers = np.cumprod((1,) + grid_shape[::-1])[:-1][::-1]
    flat_offsets = (offsets * multipliers).sum(axis=-1)
    proj2d = flat_kernel[flat_offsets]
    return proj2d, kernel


def metrics_for_attention(attn: np.ndarray, grid_shape: tuple[int, ...]) -> dict[str, float]:
    proj, kernel = cyclic_projection(attn.astype(np.float64, copy=False), grid_shape)
    residual = attn - proj
    fro = float(np.linalg.norm(attn))
    err = float(np.linalg.norm(residual) / (fro + 1e-12))
    centered = attn - attn.mean()
    r2 = 1.0 - float(np.sum(residual * residual) / (np.sum(centered * centered) + 1e-12))
    row_entropy = -np.sum(np.clip(attn, 1e-12, None) * np.log(np.clip(attn, 1e-12, None)), axis=-1)
    return {
        "relative_fro_error": err,
        "circulant_r2": r2,
        "kernel_l2": float(np.linalg.norm(kernel)),
        "row_entropy_mean": float(row_entropy.mean()),
        "row_entropy_std": float(row_entropy.std()),
        "diag_mean": float(np.diag(attn).mean()),
    }


def synthetic_attention(kind: str, grid_shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    n = int(np.prod(grid_shape))
    coords = np.array(list(np.ndindex(grid_shape)), dtype=np.int64)
    offsets = (coords[None, :, :] - coords[:, None, :]) % np.array(grid_shape, dtype=np.int64)
    kernel = rng.normal(size=grid_shape)
    multipliers = np.cumprod((1,) + grid_shape[::-1])[:-1][::-1]
    flat_offsets = (offsets * multipliers).sum(axis=-1)
    attn = kernel.reshape(-1)[flat_offsets]
    if kind == "noisy":
        attn = attn + 0.2 * rng.normal(size=(n, n))
    elif kind == "random":
        attn = rng.normal(size=(n, n))
    return softmax(attn, axis=-1)


def load_attention(args: argparse.Namespace) -> np.ndarray:
    if args.synthetic:
        return synthetic_attention(args.synthetic, args.shape, np.random.default_rng(args.seed))[None, :, :]
    if not args.npz:
        raise ValueError("provide --npz or --synthetic")
    data = np.load(args.npz)
    if args.attention_key in data:
        attn = data[args.attention_key]
        return flatten_heads(softmax(attn, axis=-1) if args.softmax else attn)
    if args.q_key in data and args.k_key in data:
        return attention_from_qk(data[args.q_key], data[args.k_key], args.softmax)
    raise KeyError(f"{args.npz} must contain {args.attention_key} or {args.q_key}/{args.k_key}")


def main() -> int:
    args = parse_args()
    heads = load_attention(args)
    rows: list[dict[str, object]] = []
    for idx, head in enumerate(heads[: args.head_limit]):
        row: dict[str, object] = {"head": idx}
        row.update(metrics_for_attention(head, args.shape))
        rows.append(row)
    summary = {
        "shape": args.shape,
        "num_heads_reported": len(rows),
        "mean_relative_fro_error": float(np.mean([r["relative_fro_error"] for r in rows])),
        "mean_circulant_r2": float(np.mean([r["circulant_r2"] for r in rows])),
        "heads": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
