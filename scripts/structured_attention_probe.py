#!/usr/bin/env python3
"""Structured approximation probes for actual attention matrices.

This is attention-space analysis: it evaluates approximations to A or logits
derived from QK, and for attention matrices also measures the output error of
replacing A in A @ V. It does not fit or replace projection weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_VIT_REPO = "/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628"
DEFAULT_VIT_CKPT = (
    "/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/"
    "aux_value_disc8000_baseline_shift1_d6e192h6_20260629/20260629_123022/final_model.pt"
)
DEFAULT_QWEN_DIR = "/home/wangmeiqi/dqy/Qwen3-VL-30B-A3B-FP8"


def parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while True:
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if chunk_size <= 0:
                break
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stable_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x.float(), dim=dim)


def relative_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    target = target.float()
    approx = approx.float()
    denom = torch.linalg.norm(target)
    if denom.item() == 0:
        return 0.0
    return stable_float(torch.linalg.norm(target - approx) / denom)


def fit_score_from_error(error: float) -> float:
    return 1.0 - error * error


def row_normalize_nonnegative(matrix: torch.Tensor) -> torch.Tensor:
    out = matrix.float().clamp_min(0.0)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def output_relative_error(attn: torch.Tensor, approx: torch.Tensor, value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    base = attn.float() @ value.float()
    repl = row_normalize_nonnegative(approx) @ value.float()
    denom = torch.linalg.norm(base)
    if denom.item() == 0:
        return 0.0
    return stable_float(torch.linalg.norm(base - repl) / denom)


def crop_to_block(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    rows, cols = matrix.shape
    rows2 = (rows // block_size) * block_size
    cols2 = (cols // block_size) * block_size
    if rows2 <= 0 or cols2 <= 0:
        raise ValueError(f"block_size={block_size} too large for {tuple(matrix.shape)}")
    return matrix[:rows2, :cols2].float().contiguous()


def block_circulant_projection(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    m = crop_to_block(matrix, block_size)
    rows, cols = m.shape
    rb = rows // block_size
    cb = cols // block_size
    blocks = m.reshape(rb, block_size, cb, block_size).permute(0, 2, 1, 3).contiguous()
    idx = torch.arange(block_size)
    offsets = (idx[None, :] - idx[:, None]) % block_size
    flat_offsets = offsets.reshape(-1)
    flat_blocks = blocks.reshape(rb, cb, block_size * block_size)
    kernels = []
    for offset in range(block_size):
        kernels.append(flat_blocks[:, :, flat_offsets == offset].mean(dim=-1))
    kernel = torch.stack(kernels, dim=-1)
    approx_blocks = kernel[:, :, flat_offsets].reshape(rb, cb, block_size, block_size)
    return approx_blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()


def grid_cyclic_projection(matrix: torch.Tensor, grid_shape: Tuple[int, ...]) -> torch.Tensor:
    n = int(np.prod(grid_shape))
    if tuple(matrix.shape) != (n, n):
        raise ValueError(f"matrix shape {tuple(matrix.shape)} does not match grid {grid_shape}")
    coords_np = np.array(list(np.ndindex(grid_shape)), dtype=np.int64)
    coords = torch.from_numpy(coords_np)
    shape = torch.tensor(grid_shape, dtype=torch.long)
    offsets = (coords[None, :, :] - coords[:, None, :]) % shape
    multipliers_np = np.cumprod((1,) + tuple(grid_shape[::-1]))[:-1][::-1].copy()
    multipliers = torch.from_numpy(multipliers_np.astype(np.int64))
    offset_ids = (offsets * multipliers).sum(dim=-1).reshape(-1)
    values = matrix.float().reshape(-1)
    bins = n
    sums = torch.zeros(bins, dtype=torch.float32)
    counts = torch.zeros(bins, dtype=torch.float32)
    sums.scatter_add_(0, offset_ids, values)
    counts.scatter_add_(0, offset_ids, torch.ones_like(values))
    kernel = sums / counts.clamp_min(1.0)
    return kernel[offset_ids].reshape(n, n)


def identity_perm(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long)


def bit_reverse_padded_perm(n: int) -> torch.Tensor:
    width = max(1, int(math.ceil(math.log2(max(n, 2)))))

    def rev(x: int) -> int:
        y = 0
        for _ in range(width):
            y = (y << 1) | (x & 1)
            x >>= 1
        return y

    return torch.tensor(sorted(range(n), key=rev), dtype=torch.long)


def stride_perm(n: int, stride: int) -> torch.Tensor:
    if math.gcd(n, stride) != 1:
        return identity_perm(n)
    return torch.tensor([(i * stride) % n for i in range(n)], dtype=torch.long)


def interleave_groups_perm(n: int, groups: int) -> torch.Tensor:
    if groups <= 1 or n % groups != 0:
        return identity_perm(n)
    return torch.arange(n, dtype=torch.long).reshape(groups, n // groups).t().reshape(-1)


def permutation_bank(n: int) -> Dict[str, torch.Tensor]:
    candidates = {
        "identity": identity_perm(n),
        "bit_reverse_padded": bit_reverse_padded_perm(n),
        "stride5": stride_perm(n, 5),
        "stride7": stride_perm(n, 7),
        "interleave_groups4": interleave_groups_perm(n, 4),
        "interleave_groups7": interleave_groups_perm(n, 7),
        "interleave_groups8": interleave_groups_perm(n, 8),
        "interleave_groups14": interleave_groups_perm(n, 14),
        "interleave_groups16": interleave_groups_perm(n, 16),
        "interleave_groups28": interleave_groups_perm(n, 28),
        "interleave_groups32": interleave_groups_perm(n, 32),
    }
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, perm in candidates.items():
        if perm.numel() != n or torch.unique(perm).numel() != n:
            continue
        key = tuple(int(v) for v in perm.tolist())
        if key in seen:
            continue
        seen.add(key)
        out[name] = perm
    return out


def block_diag_mask(rows: int, cols: int, block_size: int) -> torch.Tensor:
    rg = torch.arange(rows, dtype=torch.long) // block_size
    cg = torch.arange(cols, dtype=torch.long) // block_size
    return rg[:, None] == cg[None, :]


def monarch_proxy_mask(rows: int, cols: int, block_size: int, row_perm: torch.Tensor, col_perm: torch.Tensor) -> torch.Tensor:
    base = block_diag_mask(rows, cols, block_size)
    row_rank = torch.empty(rows, dtype=torch.long)
    col_rank = torch.empty(cols, dtype=torch.long)
    row_rank[row_perm[:rows]] = torch.arange(rows, dtype=torch.long)
    col_rank[col_perm[:cols]] = torch.arange(cols, dtype=torch.long)
    permuted = (row_rank[:, None] // block_size) == (col_rank[None, :] // block_size)
    return base | permuted


def result_row(
    base: Dict[str, object],
    method: str,
    matrix: torch.Tensor,
    approx: torch.Tensor,
    params: int,
    block_size: object,
    permutation: str,
    value: Optional[torch.Tensor],
    proxy_definition: Optional[str] = None,
) -> Dict[str, object]:
    cropped = matrix[: approx.shape[0], : approx.shape[1]].float()
    val = value[: approx.shape[1]] if value is not None and value.shape[0] >= approx.shape[1] else None
    err = relative_error(cropped, approx)
    row = {
        **base,
        "method": method,
        "block_size": block_size,
        "permutation": permutation,
        "relative_fro_error": err,
        "fit_score": fit_score_from_error(err),
        "params": int(params),
        "dense_params": int(cropped.numel()),
        "compression_ratio": float(cropped.numel() / max(int(params), 1)),
        "cropped_shape": list(cropped.shape),
        "output_relative_error": output_relative_error(cropped, approx, val) if base.get("matrix_kind") == "attention" else None,
    }
    if proxy_definition is not None:
        row["proxy_definition"] = proxy_definition
    return row


def evaluate_structures(
    matrix: torch.Tensor,
    grid_shape: Tuple[int, ...],
    block_sizes: Sequence[int],
    value: Optional[torch.Tensor],
    base: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return rows
    grid_approx = grid_cyclic_projection(matrix, grid_shape)
    rows.append(result_row(base, "grid_cyclic_bccb", matrix, grid_approx, n, "grid", "identity", value))

    row_perms = list(permutation_bank(n).items())
    col_perms = row_perms
    for block_size in block_sizes:
        if n % block_size != 0:
            continue
        approx = block_circulant_projection(matrix, block_size)
        rows.append(
            result_row(
                base,
                "flat_block_circulant",
                matrix,
                approx,
                (n // block_size) * (n // block_size) * block_size,
                block_size,
                "identity",
                value,
            )
        )
        for row_name, row_perm in row_perms:
            for col_name, col_perm in col_perms:
                if row_name == "identity" and col_name == "identity":
                    continue
                permuted = matrix[row_perm][:, col_perm]
                approx_perm = block_circulant_projection(permuted, block_size)
                rows.append(
                    result_row(
                        base,
                        "permuted_flat_block_circulant",
                        permuted,
                        approx_perm,
                        (n // block_size) * (n // block_size) * block_size,
                        block_size,
                        f"row={row_name};col={col_name}",
                        value[col_perm] if value is not None else None,
                    )
                )

                mask = monarch_proxy_mask(n, n, block_size, row_perm, col_perm)
                proxy = matrix.float() * mask.float()
                rows.append(
                    result_row(
                        base,
                        "monarch_like_mask_proxy",
                        matrix,
                        proxy,
                        int(mask.sum().item()),
                        block_size,
                        f"row={row_name};col={col_name}",
                        value,
                        "attention retained by the union of identity and fixed-permutation block-diagonal masks; not a trained Monarch factorization",
                    )
                )
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["scope"]),
            str(row["matrix_kind"]),
            str(row["method"]),
        )
        groups.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for (family, scope, matrix_kind, method), items in sorted(groups.items()):
        best_by_map: Dict[str, Dict[str, object]] = {}
        for item in items:
            map_id = str(item["map_id"])
            old = best_by_map.get(map_id)
            if old is None or float(item["relative_fro_error"]) < float(old["relative_fro_error"]):
                best_by_map[map_id] = item
        vals = list(best_by_map.values())
        errs = [float(v["relative_fro_error"]) for v in vals]
        fits = [float(v["fit_score"]) for v in vals]
        comps = [float(v["compression_ratio"]) for v in vals]
        out_errs = [float(v["output_relative_error"]) for v in vals if v.get("output_relative_error") is not None]
        summary.append(
            {
                "family": family,
                "scope": scope,
                "matrix_kind": matrix_kind,
                "method": method,
                "maps": len(vals),
                "mean_best_relative_fro_error": float(np.mean(errs)),
                "min_best_relative_fro_error": float(np.min(errs)),
                "max_best_relative_fro_error": float(np.max(errs)),
                "mean_best_fit_score": float(np.mean(fits)),
                "mean_compression_ratio": float(np.mean(comps)),
                "mean_best_output_relative_error": float(np.mean(out_errs)) if out_errs else None,
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def vit_load_images(repo: Path, checkpoint_args: Dict[str, object], samples: int) -> Tuple[torch.Tensor, str]:
    try:
        from torchvision import datasets
        from torchvision.transforms import v2

        dataset = datasets.CIFAR10(
            root=str(repo / "data" / str(checkpoint_args.get("dataset", "cifar-10"))),
            train=False,
            download=False,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        )
        images = torch.stack([dataset[i][0] for i in range(samples)], dim=0)
        return images, "cifar10_test_local"
    except Exception as exc:
        g = torch.Generator().manual_seed(1234)
        return torch.rand(samples, 3, 32, 32, generator=g), f"synthetic_uniform_fallback:{exc.__class__.__name__}:{exc}"


def linear(x: torch.Tensor, state: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.linear(x, state[f"{prefix}.weight"].float(), state.get(f"{prefix}.bias", None).float() if f"{prefix}.bias" in state else None)


def run_vit(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    ckpt_path = Path(args.vit_checkpoint)
    repo = Path(args.vit_repo)
    payload = load_torch(ckpt_path)
    state = payload["model_state"]
    ck_args = dict(payload.get("args", {}))
    layers = parse_ints(args.vit_layers)
    block_sizes = parse_ints(args.vit_block_sizes)
    samples = int(args.vit_samples)
    head_limit = int(args.vit_heads)
    images, input_source = vit_load_images(repo, ck_args, samples)

    patch_w = state["patch_embed.proj.weight"].float()
    patch_b = state["patch_embed.proj.bias"].float()
    patch_size = int(ck_args.get("patch_size", 4))
    embed_dim = int(ck_args.get("embed_dim", patch_w.shape[0]))
    num_heads = int(ck_args.get("num_heads", 6))
    head_dim = embed_dim // num_heads
    depth = int(ck_args.get("depth", max(layers) + 1))

    x = F.conv2d(images.float(), patch_w, patch_b, stride=patch_size)
    x = x.flatten(2).transpose(1, 2)
    cls = state["cls_token"].float().expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1) + state["pos_embed"].float()

    rows: List[Dict[str, object]] = []
    for layer in range(depth):
        prefix = f"blocks.{layer}"
        xn = F.layer_norm(
            x,
            (embed_dim,),
            state[f"{prefix}.norm1.weight"].float(),
            state[f"{prefix}.norm1.bias"].float(),
        )
        q = linear(xn, state, f"{prefix}.attn.q_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        k = linear(xn, state, f"{prefix}.attn.k_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        v = linear(xn, state, f"{prefix}.attn.v_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        full_attn = torch.softmax(logits, dim=-1)

        if layer in layers:
            for sample_idx in range(x.shape[0]):
                for head in range(min(head_limit, num_heads)):
                    patch_logits = logits[sample_idx, head, 1:, 1:].detach().cpu()
                    patch_attn = softmax(patch_logits, dim=-1)
                    value = v[sample_idx, head, 1:, :].detach().cpu()
                    base = {
                        "family": "vit",
                        "scope": "patch_patch_resoftmax_attention_only_rollout" if layer > 0 else "patch_patch_resoftmax_exact_layer0",
                        "sample": int(sample_idx),
                        "layer": int(layer),
                        "head": int(head),
                        "frame": None,
                        "grid_shape": [8, 8],
                        "map_id": f"vit:s{sample_idx}:l{layer}:h{head}",
                        "note": "ViT checkpoint is SCTM; layer0 input is exact, later layers use dense attention-only rollout without logic FFN.",
                    }
                    rows.extend(evaluate_structures(patch_attn, (8, 8), block_sizes, value, {**base, "matrix_kind": "attention"}))
                    rows.extend(evaluate_structures(patch_logits, (8, 8), block_sizes, None, {**base, "matrix_kind": "logits"}))

        dense_out = torch.matmul(full_attn, v).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], embed_dim)
        dense_out = linear(dense_out, state, f"{prefix}.attn.out_proj")
        x = x + dense_out

    metadata = {
        "vit_checkpoint": str(ckpt_path),
        "vit_checkpoint_sha256": sha256_file(ckpt_path),
        "vit_repo": str(repo),
        "checkpoint_args": ck_args,
        "input_source": input_source,
        "samples": samples,
        "head_limit": head_limit,
        "layers": layers,
        "block_sizes": block_sizes,
        "scope_warning": "The selected ViT checkpoint uses SCTM attention. Dense patch-patch maps after layer0 are an attention-only rollout diagnostic, not an exact full-model forward.",
    }
    return rows, metadata


def install_register_fake_shim() -> None:
    if hasattr(torch.library, "register_fake"):
        return

    def register_fake(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    torch.library.register_fake = register_fake  # type: ignore[attr-defined]


def read_video_cv2(path: str, frames: int, size: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = frames
    indices = np.linspace(0, max(total - 1, 0), frames).round().astype(int)
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        out.append(frame)
    cap.release()
    if len(out) < frames:
        raise RuntimeError(f"only decoded {len(out)} frames from {path}")
    return np.stack(out, axis=0)


def video_to_qwen_patches(video: np.ndarray, temporal_patch: int = 2, patch: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    t, h, w, c = video.shape
    assert c == 3
    t = (t // temporal_patch) * temporal_patch
    h = (h // patch) * patch
    w = (w // patch) * patch
    video = video[:t, :h, :w]
    x = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    x = (x - 0.5) / 0.5
    tg, hg, wg = t // temporal_patch, h // patch, w // patch
    patches = x.reshape(tg, temporal_patch, 3, hg, patch, wg, patch)
    patches = patches.permute(0, 3, 5, 2, 1, 4, 6).contiguous()
    return patches.reshape(tg * hg * wg, -1), torch.tensor([[tg, hg, wg]], dtype=torch.long)


def load_qwen_visual_model(model_dir: Path, device: torch.device):
    from safetensors import safe_open

    install_register_fake_shim()
    from transformers import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

    cfg = Qwen3VLMoeConfig.from_pretrained(str(model_dir))
    cfg.vision_config._attn_implementation = "eager"
    model = Qwen3VLMoeVisionModel(cfg.vision_config)
    model.to(dtype=torch.bfloat16)

    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("model.visual."):
            by_shard.setdefault(shard, []).append(key)

    state = {}
    for shard, keys in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as sf:
            for key in keys:
                state[key.removeprefix("model.visual.")] = sf.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def parse_video_specs(specs: Sequence[str]) -> List[Tuple[str, int]]:
    out = []
    for spec in specs:
        if "@" in spec:
            path, size = spec.rsplit("@", 1)
            out.append((path, int(size)))
        else:
            out.append((spec, 128))
    return out


def run_qwen(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    install_register_fake_shim()
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision

    model_dir = Path(args.qwen_model_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, load_info = load_qwen_visual_model(model_dir, device)
    layers = parse_ints(args.qwen_layers)
    block_sizes = parse_ints(args.qwen_block_sizes)
    head_limit = int(args.qwen_heads)
    rows: List[Dict[str, object]] = []
    video_specs = parse_video_specs(args.qwen_video)

    for video_idx, (video_path, size) in enumerate(video_specs):
        video = read_video_cv2(video_path, int(args.qwen_frames), size)
        hidden_states, grid_thw = video_to_qwen_patches(video)
        captures: Dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            module = model.blocks[layer].attn.qkv

            def make_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    captures[layer_idx] = output.detach().float().cpu()

                return hook

            handles.append(module.register_forward_hook(make_hook(layer)))

        with torch.inference_mode():
            _ = model(hidden_states.to(device), grid_thw.to(device))
        for handle in handles:
            handle.remove()

        with torch.inference_mode():
            rotary_pos_emb = model.rot_pos_emb(grid_thw.to(device))
            seq_len = int(grid_thw.prod().item())
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos().cpu(), emb.sin().cpu())

        tg, hg, wg = [int(v) for v in grid_thw[0].tolist()]
        for layer in layers:
            qkv = captures[layer]
            seq_length = qkv.shape[0]
            q, k, v = qkv.reshape(seq_length, 3, model.config.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
            scale = q.shape[-1] ** -0.5
            for frame_idx in range(tg):
                start = frame_idx * hg * wg
                end = start + hg * wg
                qf = q[start:end]
                kf = k[start:end]
                vf = v[start:end]
                logits = torch.einsum("qhd,khd->hqk", qf, kf).float() * scale
                for head in range(min(head_limit, logits.shape[0])):
                    head_logits = logits[head].detach().cpu()
                    head_attn = softmax(head_logits, dim=-1)
                    value = vf[:, head, :].detach().cpu()
                    base = {
                        "family": "qwen3vl_visual",
                        "scope": "per_temporal_slice_spatial_attention",
                        "sample": int(video_idx),
                        "video": str(video_path),
                        "size": int(size),
                        "layer": int(layer),
                        "head": int(head),
                        "frame": int(frame_idx),
                        "grid_shape": [hg, wg],
                        "map_id": f"qwen:v{video_idx}:f{frame_idx}:l{layer}:h{head}",
                        "note": "Qwen3-VL visual attention is per temporal slice; this is 2D spatial attention, not global 3D video attention.",
                    }
                    rows.extend(evaluate_structures(head_attn, (hg, wg), block_sizes, value, {**base, "matrix_kind": "attention"}))
                    rows.extend(evaluate_structures(head_logits, (hg, wg), block_sizes, None, {**base, "matrix_kind": "logits"}))

    metadata = {
        "qwen_model_dir": str(model_dir),
        "qwen_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
        "video_specs": [{"path": p, "size": s, "sha256": sha256_file(Path(p), max_bytes=128 * 1024 * 1024)} for p, s in video_specs],
        "frames": int(args.qwen_frames),
        "layers": layers,
        "head_limit": head_limit,
        "block_sizes": block_sizes,
        "load_info": load_info,
    }
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vit", "qwen"], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--vit-repo", default=DEFAULT_VIT_REPO)
    parser.add_argument("--vit-checkpoint", default=DEFAULT_VIT_CKPT)
    parser.add_argument("--vit-layers", default="0,1,2,5")
    parser.add_argument("--vit-heads", type=int, default=6)
    parser.add_argument("--vit-samples", type=int, default=8)
    parser.add_argument("--vit-block-sizes", default="4,8,16")
    parser.add_argument("--qwen-model-dir", default=DEFAULT_QWEN_DIR)
    parser.add_argument("--qwen-video", action="append", default=[])
    parser.add_argument("--qwen-frames", type=int, default=4)
    parser.add_argument("--qwen-layers", default="0,8,16,26")
    parser.add_argument("--qwen-heads", type=int, default=4)
    parser.add_argument("--qwen-block-sizes", default="4,7,8,14,16,28,32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.mode == "vit":
        rows, input_meta = run_vit(args)
    else:
        if not args.qwen_video:
            raise ValueError("--qwen-video is required for qwen mode; use path@size")
        rows, input_meta = run_qwen(args)

    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "argv": sys.argv,
        "args": vars(args),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_meta": input_meta,
        "method_notes": {
            "grid_cyclic_bccb": "nearest cyclic relative-offset projection over the stated 2D grid; closest to Circulant-Attention's attention-map hypothesis",
            "flat_block_circulant": "nearest per-block circulant projection over flattened token order",
            "permuted_flat_block_circulant": "flat block-circulant projection after fixed row/column permutation cross product; no learned permutation",
            "monarch_like_mask_proxy": "row-renormalized attention retained by two fixed block-diagonal masks for output-error accounting; not a trained Monarch factorization",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, rows)
    print(json.dumps({"rows": len(rows), "summary": len(payload["summary"]), "output_json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
