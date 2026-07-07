#!/usr/bin/env python3
"""Full-sweep attention pattern diagnostics.

This probe reuses the existing ViT/Qwen attention extraction paths, but instead
of testing BCCB/BCM replacements it measures mechanism-oriented quantities:
sink strength, row-argmax collapse, local/cyclic mass, sparse top-k mass,
effective rank, and simple oracle-mask output errors under original and random
value vectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import structured_attention_probe as sap


def stable_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def tensor_gini(values: torch.Tensor) -> float:
    x = values.float().flatten().sort().values
    if x.numel() == 0 or stable_float(x.sum()) == 0.0:
        return 0.0
    idx = torch.arange(1, x.numel() + 1, dtype=torch.float32)
    return stable_float((2.0 * (idx * x).sum() / (x.numel() * x.sum())) - ((x.numel() + 1.0) / x.numel()))


def effective_rank_fraction(matrix: torch.Tensor) -> Tuple[float, float]:
    s = torch.linalg.svdvals(matrix.float())
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    eff = torch.exp(entropy)
    return stable_float(eff), stable_float(eff / matrix.shape[0])


def offset_mask(grid_shape: Tuple[int, int], radius: int, device: torch.device) -> torch.Tensor:
    h, w = grid_shape
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)
    shape = torch.tensor([h, w], device=device)
    diff = (coords[None, :, :] - coords[:, None, :]) % shape
    wrapped = torch.minimum(diff, shape - diff)
    return wrapped.max(dim=-1).values <= radius


def row_normalize_masked(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = attn.float() * mask.float()
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def rel_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    denom = torch.linalg.norm(target.float()).clamp_min(1e-12)
    return stable_float(torch.linalg.norm(target.float() - approx.float()) / denom)


def output_error(attn: torch.Tensor, approx: torch.Tensor, value: torch.Tensor) -> float:
    base = attn.float() @ value.float()
    repl = approx.float() @ value.float()
    denom = torch.linalg.norm(base).clamp_min(1e-12)
    return stable_float(torch.linalg.norm(base - repl) / denom)


def deterministic_randn_like(value: torch.Tensor, key: str) -> torch.Tensor:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(value.shape, generator=gen, dtype=torch.float32)


def topk_row_mask(attn: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, attn.shape[1])
    idx = torch.topk(attn.float(), k=k, dim=-1).indices
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def topk_col_mask(attn: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, attn.shape[1])
    cols = torch.topk(attn.float().sum(dim=0), k=k).indices
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask[:, cols] = True
    return mask, cols


def attention_pattern_metrics(
    attn: torch.Tensor,
    value: torch.Tensor,
    grid_shape: Tuple[int, int],
    map_id: str,
) -> Dict[str, object]:
    attn = attn.detach().float().cpu()
    value = value.detach().float().cpu()
    n = attn.shape[0]
    col_mass = attn.sum(dim=0)
    entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)
    sorted_rows = attn.sort(dim=-1, descending=True).values
    eff_rank, eff_rank_frac = effective_rank_fraction(attn)

    local1_mask = offset_mask(grid_shape, 1, attn.device)
    local2_mask = offset_mask(grid_shape, 2, attn.device)
    sink2_mask, sink2_cols = topk_col_mask(attn, 2)
    sink4_mask, sink4_cols = topk_col_mask(attn, 4)
    row_top4_mask = topk_row_mask(attn, 4)
    local1_approx = row_normalize_masked(attn, local1_mask)
    sink2_approx = row_normalize_masked(attn, sink2_mask)
    top4_approx = row_normalize_masked(attn, row_top4_mask)
    union_mask = local1_mask | sink2_mask | row_top4_mask
    union_approx = row_normalize_masked(attn, union_mask)
    random_value = deterministic_randn_like(value, map_id)

    mass_total = col_mass.sum().clamp_min(1e-12)
    row_argmax_unique_fraction = torch.unique(attn.argmax(dim=-1)).numel() / n
    out: Dict[str, object] = {
        "entropy_mean": stable_float(entropy.mean()),
        "entropy_std": stable_float(entropy.std()),
        "row_argmax_unique_fraction": float(row_argmax_unique_fraction),
        "col_mass_gini": tensor_gini(col_mass),
        "top1_col_mass_fraction": stable_float(col_mass.max() / mass_total),
        "top2_col_mass_fraction": stable_float(torch.topk(col_mass, k=min(2, n)).values.sum() / mass_total),
        "top4_col_mass_fraction": stable_float(torch.topk(col_mass, k=min(4, n)).values.sum() / mass_total),
        "sink2_cols": ",".join(str(int(v)) for v in sink2_cols.tolist()),
        "sink4_cols": ",".join(str(int(v)) for v in sink4_cols.tolist()),
        "local_radius1_mass": stable_float((attn * local1_mask.float()).sum() / mass_total),
        "local_radius2_mass": stable_float((attn * local2_mask.float()).sum() / mass_total),
        "top1_per_row_mass_mean": stable_float(sorted_rows[:, :1].sum(dim=-1).mean()),
        "top2_per_row_mass_mean": stable_float(sorted_rows[:, :2].sum(dim=-1).mean()),
        "top4_per_row_mass_mean": stable_float(sorted_rows[:, :4].sum(dim=-1).mean()),
        "effective_rank": eff_rank,
        "effective_rank_fraction": eff_rank_frac,
        "sink2_matrix_error": rel_error(attn, sink2_approx),
        "local1_matrix_error": rel_error(attn, local1_approx),
        "row_top4_matrix_error": rel_error(attn, top4_approx),
        "union_sink_local_top4_matrix_error": rel_error(attn, union_approx),
        "sink2_output_error": output_error(attn, sink2_approx, value),
        "local1_output_error": output_error(attn, local1_approx, value),
        "row_top4_output_error": output_error(attn, top4_approx, value),
        "union_sink_local_top4_output_error": output_error(attn, union_approx, value),
        "sink2_random_v_output_error": output_error(attn, sink2_approx, random_value),
        "local1_random_v_output_error": output_error(attn, local1_approx, random_value),
        "row_top4_random_v_output_error": output_error(attn, top4_approx, random_value),
        "union_random_v_output_error": output_error(attn, union_approx, random_value),
    }
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_float(rows: Sequence[dict], key: str) -> Optional[float]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def pairwise_jaccard(sets: List[set[int]]) -> Optional[float]:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                vals.append(len(sets[i] & sets[j]) / len(union))
    return float(np.mean(vals)) if vals else None


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object, object, object], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["scope"], row["layer"], row["head"]), []).append(row)
    summary = []
    for (family, scope, layer, head), items in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        sink2_sets = [set(int(v) for v in str(item["sink2_cols"]).split(",") if v != "") for item in items]
        summary.append(
            {
                "family": family,
                "scope": scope,
                "layer": layer,
                "head": head,
                "maps": len(items),
                "mean_top2_col_mass_fraction": mean_float(items, "top2_col_mass_fraction"),
                "mean_row_argmax_unique_fraction": mean_float(items, "row_argmax_unique_fraction"),
                "mean_local_radius1_mass": mean_float(items, "local_radius1_mass"),
                "mean_top4_per_row_mass": mean_float(items, "top4_per_row_mass_mean"),
                "mean_effective_rank_fraction": mean_float(items, "effective_rank_fraction"),
                "mean_union_matrix_error": mean_float(items, "union_sink_local_top4_matrix_error"),
                "mean_union_output_error": mean_float(items, "union_sink_local_top4_output_error"),
                "mean_union_random_v_output_error": mean_float(items, "union_random_v_output_error"),
                "sink2_pairwise_jaccard": pairwise_jaccard(sink2_sets),
            }
        )
    return summary


def run_vit(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], dict]:
    payload = sap.load_torch(Path(args.vit_checkpoint))
    state = payload["model_state"]
    ck_args = dict(payload.get("args", {}))
    layers = parse_ints(args.vit_layers)
    samples = int(args.vit_samples)
    head_limit = int(args.vit_heads)
    images, input_source = sap.vit_load_images(Path(args.vit_repo), ck_args, samples)
    patch_size = int(ck_args.get("patch_size", 4))
    embed_dim = int(ck_args.get("embed_dim", 192))
    num_heads = int(ck_args.get("num_heads", 6))
    head_dim = embed_dim // num_heads
    x = F.conv2d(
        images.float(),
        state["patch_embed.proj.weight"].float(),
        state["patch_embed.proj.bias"].float(),
        stride=patch_size,
    )
    x = x.flatten(2).transpose(1, 2)
    cls = state["cls_token"].float().expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1) + state["pos_embed"].float()
    rows: List[Dict[str, object]] = []
    depth = int(ck_args.get("depth", max(layers) + 1))
    for layer in range(depth):
        prefix = f"blocks.{layer}"
        xn = F.layer_norm(
            x,
            (embed_dim,),
            state[f"{prefix}.norm1.weight"].float(),
            state[f"{prefix}.norm1.bias"].float(),
        )
        q = sap.linear(xn, state, f"{prefix}.attn.q_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        k = sap.linear(xn, state, f"{prefix}.attn.k_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        v = sap.linear(xn, state, f"{prefix}.attn.v_proj").reshape(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        full_attn = torch.softmax(logits, dim=-1)
        if layer in layers:
            for sample_idx in range(x.shape[0]):
                for head in range(min(head_limit, num_heads)):
                    map_id = f"vit:s{sample_idx}:l{layer}:h{head}"
                    patch_attn = torch.softmax(logits[sample_idx, head, 1:, 1:].detach().cpu(), dim=-1)
                    value = v[sample_idx, head, 1:, :].detach().cpu()
                    rows.append(
                        {
                            "family": "vit",
                            "scope": "patch_patch_resoftmax_attention_only_rollout" if layer > 0 else "patch_patch_resoftmax_exact_layer0",
                            "sample": int(sample_idx),
                            "layer": int(layer),
                            "head": int(head),
                            "frame": None,
                            "grid_shape": "8x8",
                            "map_id": map_id,
                            **attention_pattern_metrics(patch_attn, value, (8, 8), map_id),
                        }
                    )
        dense_out = torch.matmul(full_attn, v).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], embed_dim)
        x = x + sap.linear(dense_out, state, f"{prefix}.attn.out_proj")
    meta = {
        "mode": "vit",
        "input_source": input_source,
        "layers": layers,
        "samples": samples,
        "head_limit": head_limit,
        "scope_warning": "Layer0 is exact; later layers are dense attention-only rollout for the SCTM checkpoint.",
    }
    return rows, meta


def run_qwen(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], dict]:
    sap.install_register_fake_shim()
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, load_info = sap.load_qwen_visual_model(Path(args.qwen_model_dir), device)
    layers = parse_ints(args.qwen_layers)
    head_limit = int(args.qwen_heads)
    rows: List[Dict[str, object]] = []
    video_specs = sap.parse_video_specs(args.qwen_video)
    for video_idx, (video_path, size) in enumerate(video_specs):
        video = sap.read_video_cv2(video_path, int(args.qwen_frames), size)
        hidden_states, grid_thw = sap.video_to_qwen_patches(video)
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
        for layer in layers:
            qkv = captures[layer]
            seq_length = qkv.shape[0]
            q, k, v = qkv.reshape(seq_length, 3, model.config.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
            scale = q.shape[-1] ** -0.5
            for frame_idx in range(tg):
                start = frame_idx * hg * wg
                end = start + hg * wg
                logits = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]).float() * scale
                vf = v[start:end]
                for head in range(min(head_limit, logits.shape[0])):
                    map_id = f"qwen:v{video_idx}:f{frame_idx}:l{layer}:h{head}"
                    attn = torch.softmax(logits[head].detach().cpu(), dim=-1)
                    value = vf[:, head, :].detach().cpu()
                    rows.append(
                        {
                            "family": "qwen3vl_visual",
                            "scope": "per_temporal_slice_spatial_attention",
                            "sample": int(video_idx),
                            "video": str(video_path),
                            "size": int(size),
                            "layer": int(layer),
                            "head": int(head),
                            "frame": int(frame_idx),
                            "grid_shape": f"{hg}x{wg}",
                            "map_id": map_id,
                            **attention_pattern_metrics(attn, value, (hg, wg), map_id),
                        }
                    )
    meta = {
        "mode": "qwen",
        "qwen_model_dir": str(args.qwen_model_dir),
        "video_specs": [{"path": p, "size": s} for p, s in video_specs],
        "frames": int(args.qwen_frames),
        "layers": layers,
        "head_limit": head_limit,
        "load_info": load_info,
    }
    return rows, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vit", "qwen"], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--vit-repo", default=sap.DEFAULT_VIT_REPO)
    parser.add_argument("--vit-checkpoint", default=sap.DEFAULT_VIT_CKPT)
    parser.add_argument("--vit-layers", default="0,1,2,5")
    parser.add_argument("--vit-heads", type=int, default=6)
    parser.add_argument("--vit-samples", type=int, default=8)
    parser.add_argument("--qwen-model-dir", default=sap.DEFAULT_QWEN_DIR)
    parser.add_argument("--qwen-layers", default="0,8,16,26")
    parser.add_argument("--qwen-heads", type=int, default=4)
    parser.add_argument("--qwen-frames", type=int, default=4)
    parser.add_argument("--qwen-video", action="append", default=[])
    args = parser.parse_args()

    t0 = time.time()
    if args.mode == "vit":
        rows, input_meta = run_vit(args)
    else:
        if not args.qwen_video:
            raise ValueError("--qwen-video is required for qwen mode; use path@size")
        rows, input_meta = run_qwen(args)
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "created_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input_meta": input_meta,
        "rows": rows,
        "summary": summarize(rows),
        "method_notes": {
            "sink": "top-k key columns by total column attention mass",
            "local": "cyclic wrapped 2D radius mask over the stated spatial grid",
            "row_topk": "oracle per-row top-k retained attention mass",
            "union": "oracle mask union of sink2 columns, radius-1 local mask, and row top-4 entries",
            "random_v": "same attention masks evaluated against deterministic random value vectors to stress-test value-subspace effects",
        },
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, rows)
    print(json.dumps({"rows": len(rows), "summary": len(payload["summary"]), "output_json": str(output_json), "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
