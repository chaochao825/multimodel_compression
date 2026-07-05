#!/usr/bin/env python3
"""Export small Qwen3-VL visual attention maps for paper-style figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from circulant_metrics import cyclic_projection, metrics_for_attention, softmax
from introspect_qwen3vl_patched import install_register_fake_shim
from qwen3vl_visual_circulant_probe import (
    load_visual_model,
    read_video_cv2,
    video_to_qwen_patches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--items", required=True, help="comma list of layer:head:frame")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def parse_items(text: str) -> list[tuple[int, int, int]]:
    items = []
    for chunk in text.split(","):
        layer, head, frame = [int(x) for x in chunk.split(":")]
        items.append((layer, head, frame))
    return items


def main() -> int:
    args = parse_args()
    items = parse_items(args.items)
    layers = sorted({layer for layer, _head, _frame in items})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    install_register_fake_shim()
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision

    video = read_video_cv2(args.video, args.frames, args.size)
    hidden_states, grid_thw = video_to_qwen_patches(video)
    model, load_info = load_visual_model(Path(args.model_dir), device)
    model_device = next(model.parameters()).device

    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captures[layer_idx] = output.detach().float().cpu()

            return hook

        handles.append(model.blocks[layer].attn.qkv.register_forward_hook(make_hook(layer)))

    with torch.inference_mode():
        _ = model(hidden_states.to(model_device), grid_thw.to(model_device))

    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        rotary_pos_emb = model.rot_pos_emb(grid_thw.to(model_device))
        seq_len = int(grid_thw.prod().item())
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().cpu(), emb.sin().cpu())

    tg, hg, wg = [int(v) for v in grid_thw[0].tolist()]
    arrays: dict[str, np.ndarray] = {}
    records = []
    for layer, head, frame in items:
        qkv = captures[layer]
        seq_length = qkv.shape[0]
        q, k, _v = qkv.reshape(seq_length, 3, model.config.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        start = frame * hg * wg
        end = start + hg * wg
        logits = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]).numpy() * (q.shape[-1] ** -0.5)
        attn = softmax(logits[head], axis=-1)
        proj, kernel = cyclic_projection(attn, (hg, wg))
        residual = np.abs(attn - proj)
        key = f"layer{layer}_head{head}_frame{frame}"
        arrays[f"{key}_attention"] = attn.astype(np.float32)
        arrays[f"{key}_projection"] = proj.astype(np.float32)
        arrays[f"{key}_residual"] = residual.astype(np.float32)
        arrays[f"{key}_kernel"] = kernel.astype(np.float32)
        metric = metrics_for_attention(attn, (hg, wg))
        records.append({"key": key, "layer": layer, "head": head, "frame": frame, **metric})

    arrays["grid_thw"] = grid_thw.numpy().astype(np.int32)
    np.savez_compressed(args.out_npz, **arrays)
    meta = {
        "model_name": Path(args.model_dir).name,
        "video_name": Path(args.video).name,
        "frames": args.frames,
        "size": args.size,
        "grid_thw": grid_thw.tolist(),
        "items": records,
        "load_info": load_info,
        "metric_scope": "per-temporal-slice 2D attention map and nearest BCCB projection",
    }
    Path(args.out_json).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"exported": len(records), "grid_thw": grid_thw.tolist(), "items": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
