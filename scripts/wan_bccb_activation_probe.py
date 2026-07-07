#!/usr/bin/env python3
"""Direct Wan2.2 self-attention 3D cyclic/BCCB probe.

This script avoids full video generation. It loads one Wan DiT branch, creates a
deterministic latent/context input, runs the transformer blocks sequentially, and
measures whether selected self-attention maps/logits are well approximated by a
3D cyclic relative-offset projection over the latent F x H x W token grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def install_easydict_stub() -> None:
    try:
        import easydict  # noqa: F401
    except Exception:
        module = types.ModuleType("easydict")
        module.EasyDict = EasyDict
        sys.modules["easydict"] = module


def install_wan_package_stub(repo: Path) -> None:
    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [str(repo / "wan")]
    modules_pkg = types.ModuleType("wan.modules")
    modules_pkg.__path__ = [str(repo / "wan" / "modules")]
    sys.modules.setdefault("wan", wan_pkg)
    sys.modules.setdefault("wan.modules", modules_pkg)


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_size(text: str) -> tuple[int, int]:
    if "*" not in text:
        raise argparse.ArgumentTypeError("size must be WIDTH*HEIGHT")
    w, h = text.lower().split("*", 1)
    return int(w), int(h)


def delta_ids_nd(grid: tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(v) for v in grid)
    coords = np.array(list(np.ndindex(shape)), dtype=np.int64)
    shape_np = np.array(shape, dtype=np.int64)
    offsets = (coords[None, :, :] - coords[:, None, :]) % shape_np
    multipliers = np.cumprod((1,) + shape[::-1])[:-1][::-1]
    return (offsets * multipliers).sum(axis=-1).reshape(-1)


def delta_ids_3d(grid: tuple[int, int, int]) -> np.ndarray:
    return delta_ids_nd(grid)


def delta_ids_from_coord_ids(grid: tuple[int, int, int], coord_ids: np.ndarray) -> np.ndarray:
    f, h, w = grid
    ids = np.asarray(coord_ids, dtype=np.int64)
    cf = ids // (h * w)
    ch = (ids // w) % h
    cw = ids % w
    df = (cf[None, :] - cf[:, None]) % f
    dh = (ch[None, :] - ch[:, None]) % h
    dw = (cw[None, :] - cw[:, None]) % w
    return (df * h * w + dh * w + dw).reshape(-1)


def make_delta_variants(grid: tuple[int, int, int], seed: int) -> dict[str, dict[str, Any]]:
    f, h, w = grid
    n = f * h * w
    rng = np.random.default_rng(int(seed))
    random_coord = rng.permutation(n)
    variants = {
        "axis_hfw": {
            "delta_flat": delta_ids_nd((h, f, w)),
            "bins": n,
            "description": "reinterpret flattened tokens as H x F x W instead of F x H x W",
        },
        "axis_fwh": {
            "delta_flat": delta_ids_nd((f, w, h)),
            "bins": n,
            "description": "reinterpret flattened tokens as F x W x H instead of F x H x W",
        },
        "axis_whf": {
            "delta_flat": delta_ids_nd((w, h, f)),
            "bins": n,
            "description": "reinterpret flattened tokens as W x H x F instead of F x H x W",
        },
        "reverse_coord": {
            "delta_flat": delta_ids_from_coord_ids(grid, np.arange(n - 1, -1, -1, dtype=np.int64)),
            "bins": n,
            "description": "assign reversed F x H x W coordinates to token positions",
        },
        "random_coord": {
            "delta_flat": delta_ids_from_coord_ids(grid, random_coord),
            "bins": n,
            "description": "assign deterministic random F x H x W coordinates to token positions",
        },
    }
    return variants


def bccb_metrics(matrix: np.ndarray, delta_flat: np.ndarray, bins: int) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    sums = np.bincount(delta_flat, weights=values, minlength=bins)
    counts = np.bincount(delta_flat, minlength=bins).astype(np.float64)
    kernel = sums / np.maximum(counts, 1.0)
    projected = kernel[delta_flat]
    residual = values - projected
    sse = float(np.dot(residual, residual))
    centered = values - float(values.mean())
    sst = float(np.dot(centered, centered))
    fro = float(np.dot(values, values))
    return {
        "cyclic_r2": 1.0 - sse / sst if sst > 0 else 0.0,
        "relative_fro_error": math.sqrt(sse / fro) if fro > 0 else 0.0,
        "matrix_mean": float(values.mean()),
        "matrix_std": float(values.std()),
        "kernel_std": float(kernel.std()),
    }


class ShardedTensorStore:
    def __init__(self, root: Path):
        self.root = root
        index_path = root / "diffusion_pytorch_model.safetensors.index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = data["weight_map"]

    def state_for_prefix(self, prefix: str) -> dict[str, torch.Tensor]:
        grouped: dict[str, list[str]] = {}
        prefix_dot = prefix + "."
        for key, shard in self.weight_map.items():
            if key.startswith(prefix_dot):
                grouped.setdefault(shard, []).append(key)
        state: dict[str, torch.Tensor] = {}
        for shard, keys in grouped.items():
            with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
                for key in keys:
                    state[key[len(prefix_dot):]] = handle.get_tensor(key)
        return state


def load_module(store: ShardedTensorStore, prefix: str, module: nn.Module, dtype: torch.dtype) -> nn.Module:
    state = store.state_for_prefix(prefix)
    missing, unexpected = module.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{prefix} state mismatch: missing={missing}, unexpected={unexpected}")
    module.to(dtype=dtype)
    return module


def probe_head(
    q: torch.Tensor,
    k: torch.Tensor,
    head: int,
    actual_len: int,
    delta_flat: np.ndarray,
    bins: int,
    delta_variants: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    qh = q[0, :actual_len, head, :].float()
    kh = k[0, :actual_len, head, :].float()
    scale = qh.shape[-1] ** -0.5
    with torch.amp.autocast("cuda", enabled=False):
        logits = torch.matmul(qh, kh.transpose(0, 1)).mul_(scale)
        attn = torch.softmax(logits, dim=-1)
    logits_np = logits.detach().cpu().numpy()
    attn_np = attn.detach().cpu().numpy()
    row_max = logits.amax(dim=-1).detach().float().cpu().numpy()
    row_sum = attn.sum(dim=-1).detach().float().cpu().numpy()
    out = {
        "head": int(head),
        "logits": bccb_metrics(logits_np, delta_flat, bins),
        "attention": bccb_metrics(attn_np, delta_flat, bins),
        "logit_rowmax_mean": float(row_max.mean()),
        "logit_rowmax_std": float(row_max.std()),
        "attention_rowsum_std": float(row_sum.std()),
    }
    if delta_variants:
        out["delta_perturbations"] = {}
        for name, variant in delta_variants.items():
            v_delta = variant["delta_flat"]
            v_bins = int(variant["bins"])
            out["delta_perturbations"][name] = {
                "description": str(variant["description"]),
                "logits": bccb_metrics(logits_np, v_delta, v_bins),
                "attention": bccb_metrics(attn_np, v_delta, v_bins),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--branch", choices=("high_noise", "low_noise"), default="high_noise")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=parse_size, default=parse_size("832*480"))
    parser.add_argument("--frame-num", type=int, default=5)
    parser.add_argument("--layers", default="0,8,20,39")
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--timestep", type=float, default=999.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--perturbation-seed", type=int, default=20260707)
    parser.add_argument("--delta-perturbations", choices=("none", "default"), default="none")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.ckpt = args.ckpt.resolve()
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    install_easydict_stub()
    install_wan_package_stub(args.repo)

    from wan.modules import model as wan_model_module
    from wan.modules.model import WanAttentionBlock, rope_apply, rope_params, sinusoidal_embedding_1d

    layers = parse_int_list(args.layers)
    heads = parse_int_list(args.heads)
    max_layer = max(layers)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subfolder = "high_noise_model" if args.branch == "high_noise" else "low_noise_model"
    branch_root = args.ckpt / subfolder
    cfg = json.loads((branch_root / "config.json").read_text(encoding="utf-8"))
    dim = int(cfg.get("dim", 2048))
    ffn_dim = int(cfg.get("ffn_dim", 8192))
    freq_dim = int(cfg.get("freq_dim", 256))
    text_dim = int(cfg.get("text_dim", 4096))
    text_len = int(cfg.get("text_len", 512))
    in_dim = int(cfg.get("in_dim", 16))
    num_heads = int(cfg.get("num_heads", 16))
    num_layers = int(cfg.get("num_layers", 32))
    patch_size = tuple(cfg.get("patch_size", (1, 2, 2)))
    window_size = tuple(cfg.get("window_size", (-1, -1)))
    qk_norm = bool(cfg.get("qk_norm", True))
    cross_attn_norm = bool(cfg.get("cross_attn_norm", True))
    eps = float(cfg.get("eps", 1e-6))
    store = ShardedTensorStore(branch_root)

    width, height = args.size
    vae_stride = (4, 8, 8)
    latent_shape = (
        in_dim,
        (args.frame_num - 1) // vae_stride[0] + 1,
        height // vae_stride[1],
        width // vae_stride[2],
    )
    patch_grid = (
        latent_shape[1] // patch_size[0],
        latent_shape[2] // patch_size[1],
        latent_shape[3] // patch_size[2],
    )
    seq_len = int(np.prod(patch_grid))
    delta_flat = delta_ids_3d(patch_grid)
    bins = seq_len
    delta_variants = make_delta_variants(patch_grid, args.perturbation_seed) if args.delta_perturbations == "default" else {}
    records: list[dict[str, Any]] = []

    def patched_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        fa_version=None,
        layer_idx=None,
        timestep=None,
        attn_kind=None,
        qk_context=None,
    ):
        del q_lens, window_size, deterministic, dtype, fa_version
        if q_scale is not None:
            q = q * q_scale
        actual_len = int(k_lens[0].detach().cpu().item()) if k_lens is not None else int(k.shape[1])
        if attn_kind == "self" and layer_idx in layers:
            for head in heads:
                if head < q.shape[2]:
                    rec = probe_head(q, k, head, actual_len, delta_flat, bins, delta_variants)
                    rec.update({
                        "branch": args.branch,
                        "layer": int(layer_idx),
                        "timestep": float(args.timestep),
                        "grid": list(patch_grid),
                        "seq_len": int(actual_len),
                        "metric_scope": "3D cyclic projection over RoPE-applied Wan self-attention QK",
                    })
                    records.append(rec)
        query = q.transpose(1, 2).to(v.dtype)
        key = k.transpose(1, 2).to(v.dtype)
        value = v.transpose(1, 2)
        scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=causal,
        )
        return out.transpose(1, 2).contiguous()

    wan_model_module.attention = patched_attention

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        head_dim = dim // num_heads
        freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ], dim=1).to(device)

        patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        patch_embedding = load_module(store, "patch_embedding", patch_embedding, dtype).to(device)
        latent = torch.randn(latent_shape, dtype=dtype, device=device)
        x_conv = patch_embedding(latent.unsqueeze(0))
        patch_embedding.to("cpu")
        grid_sizes = torch.tensor([x_conv.shape[2:]], dtype=torch.long, device=device)
        x = x_conv.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([x.size(1)], dtype=torch.long, device=device)
        if x.size(1) != seq_len:
            raise RuntimeError(f"seq_len mismatch: {x.size(1)} vs {seq_len}")

        time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        time_embedding = load_module(store, "time_embedding", time_embedding, dtype).to(device)
        time_projection = load_module(store, "time_projection", time_projection, dtype).to(device)
        t = torch.tensor([args.timestep], dtype=torch.float32, device=device)
        t_full = t.expand(t.size(0), seq_len)
        e = time_embedding(
            sinusoidal_embedding_1d(freq_dim, t_full.flatten())
            .unflatten(0, (1, seq_len))
            .float()
            .to(device)
        )
        e0 = time_projection(e).unflatten(2, (6, dim)).float()
        time_embedding.to("cpu")
        time_projection.to("cpu")

        text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        text_embedding = load_module(store, "text_embedding", text_embedding, dtype).to(device)
        context_in = torch.randn(text_len, text_dim, dtype=dtype, device=device)
        context = text_embedding(context_in.unsqueeze(0))
        text_embedding.to("cpu")
        context_lens = None

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens,
            timestep=t.detach().clone(),
            quant_context={
                "model_role": args.branch,
                "timestep": float(args.timestep),
                "timestep_idx": 0,
                "cond_type": "synthetic",
                "probe_grid": list(patch_grid),
            },
        )

        if max_layer >= num_layers:
            raise RuntimeError(f"requested layer {max_layer}, but model has {num_layers} layers")

        for layer_idx in range(num_layers):
            if layer_idx > max_layer:
                break
            block = WanAttentionBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
            )
            block = load_module(store, f"blocks.{layer_idx}", block, dtype)
            block.to(device)
            x = block(x, layer_idx=layer_idx, **kwargs)
            block.to("cpu")
            del block
            torch.cuda.empty_cache()

    summary: dict[str, Any] = {
        "repo_name": args.repo.name,
        "ckpt_name": args.ckpt.name,
        "branch": args.branch,
        "size": f"{width}*{height}",
        "frame_num": args.frame_num,
        "latent_shape": list(latent_shape),
        "patch_grid": list(patch_grid),
        "seq_len": seq_len,
        "layers": layers,
        "heads": heads,
        "timestep": args.timestep,
        "seed": args.seed,
        "perturbation_seed": args.perturbation_seed,
        "delta_perturbations": args.delta_perturbations,
        "delta_perturbation_descriptions": {
            name: str(variant["description"]) for name, variant in delta_variants.items()
        },
        "dtype": args.dtype,
        "records": records,
    }
    if records:
        summary["mean_attention_cyclic_r2"] = float(np.mean([r["attention"]["cyclic_r2"] for r in records]))
        summary["mean_attention_relative_fro_error"] = float(
            np.mean([r["attention"]["relative_fro_error"] for r in records])
        )
        summary["mean_logits_cyclic_r2"] = float(np.mean([r["logits"]["cyclic_r2"] for r in records]))
        summary["mean_logits_relative_fro_error"] = float(
            np.mean([r["logits"]["relative_fro_error"] for r in records])
        )
        if delta_variants:
            summary["delta_perturbation_mean_attention_cyclic_r2"] = {
                name: float(np.mean([r["delta_perturbations"][name]["attention"]["cyclic_r2"] for r in records]))
                for name in delta_variants
            }
            summary["delta_perturbation_mean_attention_relative_fro_error"] = {
                name: float(np.mean([r["delta_perturbations"][name]["attention"]["relative_fro_error"] for r in records]))
                for name in delta_variants
            }
            summary["delta_perturbation_mean_logits_cyclic_r2"] = {
                name: float(np.mean([r["delta_perturbations"][name]["logits"]["cyclic_r2"] for r in records]))
                for name in delta_variants
            }
            summary["delta_perturbation_mean_logits_relative_fro_error"] = {
                name: float(np.mean([r["delta_perturbations"][name]["logits"]["relative_fro_error"] for r in records]))
                for name in delta_variants
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
