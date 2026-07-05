#!/usr/bin/env python3
"""Run a lightweight Qwen3-VL-MoE visual-attention circulant probe.

The probe loads only the visual tower weights, samples a small real video clip,
captures selected QKV projections, applies the same vision rotary embedding, and
measures per-frame 2D cyclic-shift/BCCB approximation of the attention maps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from circulant_metrics import metrics_for_attention, softmax
from introspect_qwen3vl_patched import install_register_fake_shim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--layers", default="0,8,16,26")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while True:
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if chunk_size <= 0:
                break
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
            if remaining is not None:
                remaining -= len(data)
    return h.hexdigest()


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT_ERROR:{exc.__class__.__name__}:{exc}"
    return str(getattr(module, "__version__", "unknown"))


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


def video_to_qwen_patches(video: np.ndarray, temporal_patch: int = 2, patch: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    # video: T,H,W,C uint8 RGB. Qwen3-VL config uses mean/std=0.5.
    t, h, w, c = video.shape
    assert c == 3
    t = (t // temporal_patch) * temporal_patch
    h = (h // patch) * patch
    w = (w // patch) * patch
    video = video[:t, :h, :w]
    x = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    x = (x - 0.5) / 0.5
    tg, hg, wg = t // temporal_patch, h // patch, w // patch
    # [tg, tp, C, hg, p, wg, p] -> [tg, hg, wg, C, tp, p, p]
    patches = x.reshape(tg, temporal_patch, 3, hg, patch, wg, patch)
    patches = patches.permute(0, 3, 5, 2, 1, 4, 6).contiguous()
    hidden_states = patches.reshape(tg * hg * wg, -1)
    grid_thw = torch.tensor([[tg, hg, wg]], dtype=torch.long)
    return hidden_states, grid_thw


def load_visual_model(model_dir: Path, device: torch.device):
    install_register_fake_shim()
    from transformers import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeVisionModel,
    )

    cfg = Qwen3VLMoeConfig.from_pretrained(str(model_dir))
    cfg.vision_config._attn_implementation = "eager"
    model = Qwen3VLMoeVisionModel(cfg.vision_config)
    model.to(dtype=torch.bfloat16)

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
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


def capture_qkv_metrics(model, hidden_states: torch.Tensor, grid_thw: torch.Tensor, layers: list[int], head_limit: int):
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision

    captures: dict[int, torch.Tensor] = {}
    handles = []
    model_device = next(model.parameters()).device
    for layer in layers:
        module = model.blocks[layer].attn.qkv

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captures[layer_idx] = output.detach().float().cpu()

            return hook

        handles.append(module.register_forward_hook(make_hook(layer)))

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
    rows = []
    for layer in layers:
        qkv = captures[layer]
        seq_length = qkv.shape[0]
        q, k, _v = qkv.reshape(seq_length, 3, model.config.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        d = q.shape[-1]
        scale = d**-0.5
        for frame_idx in range(tg):
            start = frame_idx * hg * wg
            end = start + hg * wg
            qf = q[start:end]
            kf = k[start:end]
            logits = torch.einsum("qhd,khd->hqk", qf, kf).numpy() * scale
            for head_idx in range(min(head_limit, logits.shape[0])):
                attn = softmax(logits[head_idx], axis=-1)
                metric = metrics_for_attention(attn, (hg, wg))
                metric.update({"layer": layer, "frame": frame_idx, "head": head_idx})
                rows.append(metric)
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    by_layer: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_summary = {}
    for layer, vals in sorted(by_layer.items()):
        layer_summary[str(layer)] = {
            "mean_relative_fro_error": float(np.mean([v["relative_fro_error"] for v in vals])),
            "mean_circulant_r2": float(np.mean([v["circulant_r2"] for v in vals])),
            "num_maps": len(vals),
        }
    return {
        "num_maps": len(rows),
        "mean_relative_fro_error": float(np.mean([v["relative_fro_error"] for v in rows])),
        "mean_circulant_r2": float(np.mean([v["circulant_r2"] for v in rows])),
        "layers": layer_summary,
    }


def metadata(args: argparse.Namespace, model_dir: Path, grid_thw: torch.Tensor, load_info: dict[str, object]) -> dict[str, object]:
    video_path = Path(args.video)
    index_path = model_dir / "model.safetensors.index.json"
    script_path = Path(__file__).resolve()
    return {
        "argv": sys.argv,
        "args": vars(args),
        "device_requested": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch": module_version("torch"),
        "transformers": module_version("transformers"),
        "safetensors": module_version("safetensors"),
        "numpy": module_version("numpy"),
        "cv2": module_version("cv2"),
        "script_sha256": sha256_file(script_path),
        "model_index_sha256": sha256_file(index_path) if index_path.exists() else None,
        "video_sha256": sha256_file(video_path) if video_path.exists() else None,
        "video_size_bytes": video_path.stat().st_size if video_path.exists() else None,
        "grid_thw": grid_thw.tolist(),
        "layers": [int(x) for x in args.layers.split(",") if x],
        "head_limit": args.heads,
        "metric_scope": "per-temporal-slice 2D spatial BCCB/cyclic-offset attention",
        "token_order_assumption": (
            "Manual patchification uses temporal-major, raster H/W order and Qwen3-VL "
            "vision forward does not reorder tokens before qkv; this is not an official "
            "processor byte-for-byte validation."
        ),
        "load_info": load_info,
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    layers = [int(x) for x in args.layers.split(",") if x]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    video = read_video_cv2(args.video, args.frames, args.size)
    hidden_states, grid_thw = video_to_qwen_patches(video)
    model, load_info = load_visual_model(model_dir, device)
    rows = capture_qkv_metrics(model, hidden_states, grid_thw, layers, args.heads)
    result = {
        "model_dir": str(model_dir),
        "video": args.video,
        "frames": args.frames,
        "size": args.size,
        "grid_thw": grid_thw.tolist(),
        "load_info": load_info,
        "metadata": metadata(args, model_dir, grid_thw, load_info),
        "summary": summarize(rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
