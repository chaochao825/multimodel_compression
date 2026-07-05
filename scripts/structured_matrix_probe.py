#!/usr/bin/env python3
"""Offline structured-matrix probes for ViT and video-model projection weights.

This script measures whether trained attention projection weights are already
close to several lightweight structured families. It is a diagnostic only: it
does not replace model modules, fine-tune adapters, or benchmark kernels.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_VIT_CKPT = (
    "/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/"
    "aux_value_disc8000_baseline_shift1_d6e192h6_20260629/20260629_123022/final_model.pt"
)
DEFAULT_QWEN_DIR = "/home/wangmeiqi/dqy/Qwen3-VL-30B-A3B-FP8"


@dataclass
class MatrixSpec:
    family: str
    model: str
    layer: int
    projection: str
    name: str
    weight: torch.Tensor


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_layers(text: str) -> List[int]:
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def parse_ints(text: str) -> List[int]:
    return parse_layers(text)


def stable_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def matrix_energy(weight: torch.Tensor) -> float:
    return stable_float(torch.sum(weight.float() * weight.float()))


def crop_to_block(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    rows2 = (rows // block_size) * block_size
    cols2 = (cols // block_size) * block_size
    if rows2 <= 0 or cols2 <= 0:
        raise ValueError(f"block_size={block_size} is too large for shape={tuple(weight.shape)}")
    return weight[:rows2, :cols2].float().contiguous()


def circulant_block_projection(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Nearest projection where every b x b block is independently circulant."""
    w = crop_to_block(weight, block_size)
    rows, cols = w.shape
    rb = rows // block_size
    cb = cols // block_size
    blocks = w.reshape(rb, block_size, cb, block_size).permute(0, 2, 1, 3).contiguous()

    idx = torch.arange(block_size, device=w.device)
    offsets = (idx[None, :] - idx[:, None]) % block_size
    flat_offsets = offsets.reshape(-1)
    kernels = []
    flat_blocks = blocks.reshape(rb, cb, block_size * block_size)
    for offset in range(block_size):
        kernels.append(flat_blocks[:, :, flat_offsets == offset].mean(dim=-1))
    kernel = torch.stack(kernels, dim=-1)
    approx_blocks = kernel[:, :, flat_offsets].reshape(rb, cb, block_size, block_size)
    return approx_blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()


def relative_error(target: torch.Tensor, approx: torch.Tensor) -> float:
    target = target.float()
    approx = approx.float()
    denom = torch.linalg.norm(target)
    if denom.item() == 0:
        return 0.0
    return stable_float(torch.linalg.norm(target - approx) / denom)


def fit_score_from_error(error: float) -> float:
    return 1.0 - error * error


def block_circulant_param_count(shape: Tuple[int, int], block_size: int) -> int:
    rows, cols = shape
    rows2 = (rows // block_size) * block_size
    cols2 = (cols // block_size) * block_size
    return (rows2 // block_size) * (cols2 // block_size) * block_size


def is_perm(values: torch.Tensor, n: int) -> bool:
    return values.numel() == n and torch.unique(values).numel() == n


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

    values = sorted(range(n), key=rev)
    return torch.tensor(values, dtype=torch.long)


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
        "interleave_groups16": interleave_groups_perm(n, 16),
        "interleave_groups32": interleave_groups_perm(n, 32),
    }
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, perm in candidates.items():
        if not is_perm(perm, n):
            continue
        key = tuple(int(x) for x in perm[: min(n, 64)].tolist()) + (int(torch.sum(perm).item()),)
        if key in seen:
            continue
        seen.add(key)
        out[name] = perm
    return out


def evaluate_block_circulant(weight: torch.Tensor, block_size: int) -> Dict:
    cropped = crop_to_block(weight, block_size)
    approx = circulant_block_projection(weight, block_size)
    err = relative_error(cropped, approx)
    dense_params = int(cropped.numel())
    params = block_circulant_param_count(tuple(cropped.shape), block_size)
    return {
        "method": "block_circulant",
        "block_size": block_size,
        "permutation": "identity",
        "relative_fro_error": err,
        "fit_score": fit_score_from_error(err),
        "params": params,
        "dense_params": dense_params,
        "compression_ratio": dense_params / max(params, 1),
        "cropped_shape": list(cropped.shape),
    }


def evaluate_permuted_block_circulant(
    weight: torch.Tensor,
    block_size: int,
    perm_name: str,
    row_perm: torch.Tensor,
    col_perm: torch.Tensor,
) -> Dict:
    permuted = weight[row_perm][:, col_perm].contiguous()
    cropped = crop_to_block(permuted, block_size)
    approx = circulant_block_projection(permuted, block_size)
    err = relative_error(cropped, approx)
    dense_params = int(cropped.numel())
    params = block_circulant_param_count(tuple(cropped.shape), block_size)
    return {
        "method": "permuted_block_circulant",
        "block_size": block_size,
        "permutation": perm_name,
        "relative_fro_error": err,
        "fit_score": fit_score_from_error(err),
        "params": params,
        "dense_params": dense_params,
        "compression_ratio": dense_params / max(params, 1),
        "cropped_shape": list(cropped.shape),
    }


def block_diag_mask(rows: int, cols: int, block_size: int) -> torch.Tensor:
    row_group = torch.arange(rows, dtype=torch.long) // block_size
    col_group = torch.arange(cols, dtype=torch.long) // block_size
    return row_group[:, None] == col_group[None, :]


def monarch_proxy_mask(
    rows: int,
    cols: int,
    block_size: int,
    row_perm: torch.Tensor,
    col_perm: torch.Tensor,
) -> torch.Tensor:
    base = block_diag_mask(rows, cols, block_size)
    row_rank = torch.empty(rows, dtype=torch.long)
    col_rank = torch.empty(cols, dtype=torch.long)
    row_rank[row_perm] = torch.arange(rows, dtype=torch.long)
    col_rank[col_perm] = torch.arange(cols, dtype=torch.long)
    permuted = (row_rank[:, None] // block_size) == (col_rank[None, :] // block_size)
    return base | permuted


def evaluate_monarch_proxy(
    weight: torch.Tensor,
    block_size: int,
    perm_name: str,
    row_perm: torch.Tensor,
    col_perm: torch.Tensor,
) -> Dict:
    cropped = crop_to_block(weight, block_size)
    rows, cols = cropped.shape
    mask = monarch_proxy_mask(rows, cols, block_size, row_perm[:rows], col_perm[:cols])
    total = matrix_energy(cropped)
    kept = stable_float(torch.sum(cropped[mask] * cropped[mask])) if total > 0 else 0.0
    frac = kept / total if total > 0 else 1.0
    err = math.sqrt(max(0.0, 1.0 - frac))
    params = int(mask.sum().item())
    dense_params = int(cropped.numel())
    return {
        "method": "monarch_like_proxy",
        "block_size": block_size,
        "permutation": perm_name,
        "relative_fro_error": err,
        "fit_score": frac,
        "params": params,
        "dense_params": dense_params,
        "compression_ratio": dense_params / max(params, 1),
        "cropped_shape": list(cropped.shape),
        "proxy_definition": "energy retained by union of identity and fixed-permutation block-diagonal layouts; not a trained Monarch factorization",
    }


def load_vit_matrices(checkpoint: Path, layers: Sequence[int]) -> Tuple[List[MatrixSpec], Dict]:
    payload = load_torch(checkpoint)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    metadata = {}
    if isinstance(payload, dict) and isinstance(payload.get("args"), dict):
        metadata = dict(payload["args"])
    elif isinstance(payload, dict) and payload.get("args") is not None:
        args_obj = payload["args"]
        metadata = vars(args_obj) if hasattr(args_obj, "__dict__") else {"args_repr": repr(args_obj)}
    matrices: List[MatrixSpec] = []
    for layer in layers:
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            key = f"blocks.{layer}.attn.{proj}.weight"
            if key not in state:
                continue
            matrices.append(
                MatrixSpec(
                    family="vit",
                    model=str(checkpoint),
                    layer=layer,
                    projection=proj,
                    name=key,
                    weight=state[key].detach().float().cpu(),
                )
            )
    return matrices, metadata


def load_safetensor(model_dir: Path, key: str) -> torch.Tensor:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    filename = index["weight_map"][key]
    shard_path = model_dir / filename
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).detach().float().cpu()


def load_qwen_matrices(model_dir: Path, layers: Sequence[int]) -> Tuple[List[MatrixSpec], Dict]:
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]
    metadata = {
        "model_dir": str(model_dir),
        "index_sha256": sha256_file(index_path),
        "num_weight_keys": len(weight_map),
    }
    matrices: List[MatrixSpec] = []
    for layer in layers:
        qkv_key = f"model.visual.blocks.{layer}.attn.qkv.weight"
        if qkv_key in weight_map:
            qkv = load_safetensor(model_dir, qkv_key)
            if qkv.shape[0] % 3 == 0:
                chunk = qkv.shape[0] // 3
                for idx, proj in enumerate(["q_proj", "k_proj", "v_proj"]):
                    matrices.append(
                        MatrixSpec(
                            family="qwen3vl_visual",
                            model=str(model_dir),
                            layer=layer,
                            projection=proj,
                            name=f"{qkv_key}:{proj}",
                            weight=qkv[idx * chunk : (idx + 1) * chunk].contiguous(),
                        )
                    )
        out_key = f"model.visual.blocks.{layer}.attn.proj.weight"
        if out_key in weight_map:
            matrices.append(
                MatrixSpec(
                    family="qwen3vl_visual",
                    model=str(model_dir),
                    layer=layer,
                    projection="out_proj",
                    name=out_key,
                    weight=load_safetensor(model_dir, out_key),
                )
            )
    return matrices, metadata


def evaluate_matrix(spec: MatrixSpec, block_sizes: Sequence[int]) -> List[Dict]:
    rows, cols = spec.weight.shape
    row_perms = permutation_bank(rows)
    col_perms = permutation_bank(cols)
    row_perm_items = list(row_perms.items())
    col_perm_items = list(col_perms.items())
    rows_out: List[Dict] = []
    base = {
        "family": spec.family,
        "model": spec.model,
        "layer": spec.layer,
        "projection": spec.projection,
        "name": spec.name,
        "shape": list(spec.weight.shape),
        "weight_energy": matrix_energy(spec.weight),
    }
    for block_size in block_sizes:
        if rows < block_size or cols < block_size:
            continue
        if rows % block_size != 0 or cols % block_size != 0:
            continue
        bcm = evaluate_block_circulant(spec.weight, block_size)
        rows_out.append({**base, **bcm})

        for row_name, row_perm in row_perm_items:
            for col_name, col_perm in col_perm_items:
                if row_name == "identity" and col_name == "identity":
                    continue
                perm_name = f"row={row_name};col={col_name}"
                permed = evaluate_permuted_block_circulant(
                    spec.weight,
                    block_size,
                    perm_name,
                    row_perm,
                    col_perm,
                )
                rows_out.append({**base, **permed})

        for row_name, row_perm in row_perm_items:
            for col_name, col_perm in col_perm_items:
                if row_name == "identity" and col_name == "identity":
                    continue
                perm_name = f"row={row_name};col={col_name}"
                proxy = evaluate_monarch_proxy(
                    spec.weight,
                    block_size,
                    perm_name,
                    row_perm,
                    col_perm,
                )
                rows_out.append({**base, **proxy})
    return rows_out


def summarize(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str, int], List[Dict]] = {}
    for row in rows:
        key = (row["family"], row["method"], int(row["block_size"]))
        groups.setdefault(key, []).append(row)
    summary: List[Dict] = []
    for (family, method, block_size), items in sorted(groups.items()):
        best_by_matrix: Dict[str, Dict] = {}
        for item in items:
            key = item["name"]
            old = best_by_matrix.get(key)
            if old is None or item["relative_fro_error"] < old["relative_fro_error"]:
                best_by_matrix[key] = item
        vals = list(best_by_matrix.values())
        errs = [float(v["relative_fro_error"]) for v in vals]
        fits = [float(v["fit_score"]) for v in vals]
        compressions = [float(v["compression_ratio"]) for v in vals]
        summary.append(
            {
                "family": family,
                "method": method,
                "block_size": block_size,
                "matrices": len(vals),
                "mean_best_relative_fro_error": sum(errs) / len(errs),
                "min_best_relative_fro_error": min(errs),
                "max_best_relative_fro_error": max(errs),
                "mean_best_fit_score": sum(fits) / len(fits),
                "mean_compression_ratio": sum(compressions) / len(compressions),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "vit", "qwen"], default="all")
    parser.add_argument("--vit-checkpoint", default=DEFAULT_VIT_CKPT)
    parser.add_argument("--qwen-model-dir", default=DEFAULT_QWEN_DIR)
    parser.add_argument("--vit-layers", default="0,1,2,5")
    parser.add_argument("--qwen-layers", default="0,8,16,26")
    parser.add_argument("--vit-block-sizes", default="8,16,32")
    parser.add_argument("--qwen-block-sizes", default="16,32,64")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    started = time.time()
    all_rows: List[Dict] = []
    inputs: Dict[str, Dict] = {}

    if args.mode in ("all", "vit"):
        vit_path = Path(args.vit_checkpoint)
        matrices, metadata = load_vit_matrices(vit_path, parse_layers(args.vit_layers))
        inputs["vit"] = {
            "path": str(vit_path),
            "sha256": sha256_file(vit_path),
            "checkpoint_metadata": metadata,
            "num_matrices": len(matrices),
        }
        for spec in matrices:
            all_rows.extend(evaluate_matrix(spec, parse_ints(args.vit_block_sizes)))

    if args.mode in ("all", "qwen"):
        model_dir = Path(args.qwen_model_dir)
        matrices, metadata = load_qwen_matrices(model_dir, parse_layers(args.qwen_layers))
        inputs["qwen"] = {**metadata, "num_matrices": len(matrices)}
        for spec in matrices:
            all_rows.extend(evaluate_matrix(spec, parse_ints(args.qwen_block_sizes)))

    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "script_sha256": sha256_file(Path(__file__)),
        "argv": sys.argv,
        "args": vars(args),
        "mode": args.mode,
        "method_notes": {
            "block_circulant": "nearest Frobenius projection where each b x b block is independently circulant, matching the BlockCirculantLinear parameterization used on 210",
            "permuted_block_circulant": "same projection after the cross product of fixed deterministic row and column permutations; no learned permutation",
            "monarch_like_proxy": "energy retained by the union of two block-diagonal layouts, identity plus a fixed row/column permutation pair; this is a proxy score, not a trained Monarch product factorization",
        },
        "inputs": inputs,
        "rows": all_rows,
        "summary": summarize(all_rows),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_csv(output_csv, all_rows)
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
