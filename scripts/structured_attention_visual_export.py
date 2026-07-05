#!/usr/bin/env python3
"""Export representative attention matrices and structured approximations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import hybrid_attention_decomposition as had
import structured_attention_probe as sap


def parse_map_id(map_id: str) -> Dict[str, int | str]:
    if map_id.startswith("vit:"):
        m = re.fullmatch(r"vit:s(\d+):l(\d+):h(\d+)", map_id)
        if not m:
            raise ValueError(f"bad ViT map id: {map_id}")
        return {"family": "vit", "sample": int(m.group(1)), "layer": int(m.group(2)), "head": int(m.group(3))}
    if map_id.startswith("qwen:"):
        m = re.fullmatch(r"qwen:v(\d+):f(\d+):l(\d+):h(\d+)", map_id)
        if not m:
            raise ValueError(f"bad Qwen map id: {map_id}")
        return {
            "family": "qwen",
            "video": int(m.group(1)),
            "frame": int(m.group(2)),
            "layer": int(m.group(3)),
            "head": int(m.group(4)),
        }
    raise ValueError(f"unknown map id: {map_id}")


def best_structures(attn: torch.Tensor, grid_shape: Tuple[int, ...], block_sizes: List[int]) -> Dict[str, Tuple[torch.Tensor, Dict]]:
    n = attn.shape[0]
    out: Dict[str, Tuple[torch.Tensor, Dict]] = {}
    grid = sap.grid_cyclic_projection(attn, grid_shape)
    out["grid_cyclic_bccb"] = (grid, {"method": "grid_cyclic_bccb", "block_size": "grid", "permutation": "identity"})

    row_perms = list(sap.permutation_bank(n).items())
    best_flat = None
    best_perm = None
    best_proxy = None
    for b in block_sizes:
        if n % b != 0:
            continue
        flat = sap.block_circulant_projection(attn, b)
        flat_err = sap.relative_error(attn, flat)
        if best_flat is None or flat_err < best_flat[0]:
            best_flat = (flat_err, flat, {"method": "flat_block_circulant", "block_size": b, "permutation": "identity"})
        for row_name, row_perm in row_perms:
            for col_name, col_perm in row_perms:
                if row_name == "identity" and col_name == "identity":
                    continue
                permuted = attn[row_perm][:, col_perm]
                approx_perm = sap.block_circulant_projection(permuted, b)
                err_perm = sap.relative_error(permuted, approx_perm)
                if best_perm is None or err_perm < best_perm[0]:
                    inv_r = torch.empty_like(row_perm)
                    inv_c = torch.empty_like(col_perm)
                    inv_r[row_perm] = torch.arange(n)
                    inv_c[col_perm] = torch.arange(n)
                    unpermed = approx_perm[inv_r][:, inv_c]
                    best_perm = (
                        err_perm,
                        unpermed,
                        {
                            "method": "permuted_flat_block_circulant",
                            "block_size": b,
                            "permutation": f"row={row_name};col={col_name}",
                        },
                    )
                mask = sap.monarch_proxy_mask(n, n, b, row_perm, col_perm)
                proxy = sap.row_normalize_nonnegative(attn.float() * mask.float())
                err_proxy = sap.relative_error(attn, proxy)
                if best_proxy is None or err_proxy < best_proxy[0]:
                    best_proxy = (
                        err_proxy,
                        proxy,
                        {
                            "method": "monarch_like_mask_proxy",
                            "block_size": b,
                            "permutation": f"row={row_name};col={col_name}",
                        },
                    )
    if best_flat is not None:
        out["flat_block_circulant"] = (best_flat[1], best_flat[2])
    if best_perm is not None:
        out["permuted_flat_block_circulant"] = (best_perm[1], best_perm[2])
    if best_proxy is not None:
        out["monarch_like_mask_proxy"] = (best_proxy[1], best_proxy[2])
    return out


def add_example(
    arrays: Dict[str, np.ndarray],
    items: List[Dict],
    label: str,
    map_id: str,
    attn: torch.Tensor,
    value: torch.Tensor,
    grid_shape: Tuple[int, ...],
    block_sizes: List[int],
    note: str,
) -> None:
    key = f"ex{len(items)}"
    attn = attn.float().cpu()
    value = value.float().cpu()
    arrays[f"{key}_attention"] = attn.numpy()
    structs = best_structures(attn, grid_shape, block_sizes)
    metrics = {}
    for name, (approx, meta) in structs.items():
        approx = approx.float().cpu()
        arrays[f"{key}_{name}"] = approx.numpy()
        arrays[f"{key}_{name}_residual"] = (attn - approx).abs().numpy()
        metrics[name] = {
            **meta,
            "relative_fro_error": sap.relative_error(attn, approx),
            "output_relative_error": sap.output_relative_error(attn, approx, value),
        }
    for config in had.default_configs(attn.shape[0]):
        result = had.decompose(
            attn.numpy(),
            grid_shape,
            int(config["sink_k"]),
            int(config["rank"]),
            int(config["radius"]),
            int(config["sparse_k"]),
        )
        name = str(config["name"])
        approx = torch.from_numpy(result["hybrid"]).float()
        metrics[name] = {
            "method": name,
            "sink_k": int(config["sink_k"]),
            "rank": int(config["rank"]),
            "radius": int(config["radius"]),
            "sparse_k": int(config["sparse_k"]),
            "relative_fro_error": sap.relative_error(attn, approx),
            "output_relative_error": sap.output_relative_error(attn, approx, value),
            "nominal_budget_params": int(result["nominal_budget_params"]),
            "dense_params": int(result["dense_params"]),
            "nominal_budget_ratio": float(result["nominal_budget_ratio"]),
            "global_svd_numeric_rank_after_clipping": int(result["global_svd_numeric_rank_after_clipping"]),
            "sink_mass": float(result["sink_mass"]),
            "local_mass": float(result["local_mass"]),
            "global_svd_mass": float(result["global_svd_mass"]),
            "sparse_mass": float(result["sparse_mass"]),
            "budget_note": str(result["budget_note"]),
            "proxy_definition": (
                "oracle sink/global-SVD + local-cyclic + sparse-routing decomposition; "
                "sinks and sparse routes are selected from observed A; nominal budget is not "
                "a strict compressed representation cost"
            ),
        }
        if name == "hybrid_balanced":
            arrays[f"{key}_{name}"] = np.asarray(result["hybrid"], dtype=np.float32)
            arrays[f"{key}_{name}_residual"] = np.asarray(result["residual"], dtype=np.float32)
            for array_name in ["sink_global_svd", "local_cyclic", "sparse_routing"]:
                arrays[f"{key}_{name}_{array_name}"] = np.asarray(result[array_name], dtype=np.float32)
    entropy = -torch.sum(attn.clamp_min(1e-12) * torch.log(attn.clamp_min(1e-12)), dim=-1)
    items.append(
        {
            "key": key,
            "label": label,
            "map_id": map_id,
            "grid_shape": list(grid_shape),
            "note": note,
            "attention_entropy_mean": float(entropy.mean().item()),
            "attention_entropy_std": float(entropy.std().item()),
            "row_argmax_unique_fraction": float(torch.unique(attn.argmax(dim=-1)).numel() / attn.shape[0]),
            "metrics": metrics,
        }
    )


def export_vit(args: argparse.Namespace, targets: List[Dict], arrays: Dict[str, np.ndarray], items: List[Dict]) -> None:
    if not targets:
        return
    sap.install_register_fake_shim()
    ckpt = Path(args.vit_checkpoint)
    payload = sap.load_torch(ckpt)
    state = payload["model_state"]
    ck_args = dict(payload.get("args", {}))
    max_sample = max(int(t["sample"]) for t in targets)
    max_layer = max(int(t["layer"]) for t in targets)
    images, source = sap.vit_load_images(Path(args.vit_repo), ck_args, max_sample + 1)
    patch_size = int(ck_args.get("patch_size", 4))
    embed_dim = int(ck_args.get("embed_dim", 192))
    num_heads = int(ck_args.get("num_heads", 6))
    head_dim = embed_dim // num_heads
    x = F.conv2d(images.float(), state["patch_embed.proj.weight"].float(), state["patch_embed.proj.bias"].float(), stride=patch_size)
    x = x.flatten(2).transpose(1, 2)
    cls = state["cls_token"].float().expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1) + state["pos_embed"].float()
    by_layer = {}
    for layer in range(max_layer + 1):
        prefix = f"blocks.{layer}"
        xn = F.layer_norm(x, (embed_dim,), state[f"{prefix}.norm1.weight"].float(), state[f"{prefix}.norm1.bias"].float())
        q = sap.linear(xn, state, f"{prefix}.attn.q_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        k = sap.linear(xn, state, f"{prefix}.attn.k_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        v = sap.linear(xn, state, f"{prefix}.attn.v_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        by_layer[layer] = (logits, v)
        dense_out = torch.matmul(torch.softmax(logits, dim=-1), v).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], embed_dim)
        x = x + sap.linear(dense_out, state, f"{prefix}.attn.out_proj")
    for target in targets:
        layer = int(target["layer"])
        sample = int(target["sample"])
        head = int(target["head"])
        logits, v = by_layer[layer]
        patch_logits = logits[sample, head, 1:, 1:]
        attn = sap.softmax(patch_logits, dim=-1)
        value = v[sample, head, 1:, :]
        add_example(
            arrays,
            items,
            f"ViT L{layer} H{head}",
            f"vit:s{sample}:l{layer}:h{head}",
            attn,
            value,
            (8, 8),
            sap.parse_ints(args.vit_block_sizes),
            f"Input source={source}; SCTM checkpoint, layer0 exact, later layers attention-only rollout.",
        )


def export_qwen(args: argparse.Namespace, targets: List[Dict], arrays: Dict[str, np.ndarray], items: List[Dict]) -> None:
    if not targets:
        return
    sap.install_register_fake_shim()
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _load_info = sap.load_qwen_visual_model(Path(args.qwen_model_dir), device)
    specs = sap.parse_video_specs(args.qwen_video)
    by_video: Dict[int, List[Dict]] = {}
    for target in targets:
        by_video.setdefault(int(target["video"]), []).append(target)
    for video_idx, vt in by_video.items():
        video_path, size = specs[video_idx]
        video = sap.read_video_cv2(video_path, int(args.qwen_frames), size)
        hidden_states, grid_thw = sap.video_to_qwen_patches(video)
        layers = sorted({int(t["layer"]) for t in vt})
        captures: Dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            def make_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    captures[layer_idx] = output.detach().float().cpu()
                return hook
            handles.append(model.blocks[layer].attn.qkv.register_forward_hook(make_hook(layer)))
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
        for target in vt:
            layer = int(target["layer"])
            frame = int(target["frame"])
            head = int(target["head"])
            qkv = captures[layer]
            seq_length = qkv.shape[0]
            q, k, v = qkv.reshape(seq_length, 3, model.config.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
            start = frame * hg * wg
            end = start + hg * wg
            logits = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]).float() * (q.shape[-1] ** -0.5)
            attn = sap.softmax(logits[head], dim=-1)
            value = v[start:end, head, :]
            add_example(
                arrays,
                items,
                f"Qwen L{layer} H{head} F{frame}",
                f"qwen:v{video_idx}:f{frame}:l{layer}:h{head}",
                attn,
                value,
                (hg, wg),
                sap.parse_ints(args.qwen_block_sizes),
                f"Video={Path(video_path).name}@{size}; true Qwen3-VL visual forward, per-temporal-slice 2D attention.",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", action="append", default=[])
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--vit-repo", default=sap.DEFAULT_VIT_REPO)
    parser.add_argument("--vit-checkpoint", default=sap.DEFAULT_VIT_CKPT)
    parser.add_argument("--vit-block-sizes", default="4,8,16")
    parser.add_argument("--qwen-model-dir", default=sap.DEFAULT_QWEN_DIR)
    parser.add_argument("--qwen-video", action="append", default=[])
    parser.add_argument("--qwen-frames", type=int, default=4)
    parser.add_argument("--qwen-block-sizes", default="4,7,8,14,16,28,32")
    args = parser.parse_args()

    targets = [parse_map_id(mid) for mid in args.map_id]
    arrays: Dict[str, np.ndarray] = {}
    items: List[Dict] = []
    export_vit(args, [t for t in targets if t["family"] == "vit"], arrays, items)
    export_qwen(args, [t for t in targets if t["family"] == "qwen"], arrays, items)
    out_npz = Path(args.out_npz)
    out_json = Path(args.out_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    out_json.write_text(
        json.dumps({"argv": sys.argv, "items": items, "npz": str(out_npz)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"items": len(items), "out_npz": str(out_npz), "out_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
