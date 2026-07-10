#!/usr/bin/env python3
"""Task-level causal intervention probe for the actual ViT/SCTM route path.

This script loads the saved ViT-LGN/SCTM checkpoint from the original 210
workspace, evaluates it on CIFAR, and monkeypatches the real
AuxAccumCLSPatchTokenMixer forward path to remove selected CLS-to-patch routes.

It is intentionally not a dense-attention proxy: all measurements go through the
checkpoint's SCTM token mixer, logic FFN, norm, residual, and classifier head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_VIT_REPO = "/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628"
DEFAULT_CHECKPOINT = (
    "/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/"
    "runs/aux_value_disc8000_baseline_shift1_d6e192h6_20260629/"
    "20260629_123022/final_model.pt"
)


FALLBACK_ARGS: Dict[str, Any] = {
    "dataset": "cifar-10",
    "seed": 0,
    "data_encoding": "real-input",
    "augment": False,
    "batch_size": 128,
    "num_workers": 0,
    "valid_set_size": 0.1,
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
    "drop_path_rate": 0.0,
    "ffn_type": "logic",
    "logic_ffn_layers": 2,
    "logic_mlp_ratio": 1.0,
    "num_forest_layers": 2,
    "logic_connections": "random",
    "logic_n_thresholds": 7,
    "logic_use_thermometer": True,
    "logic_encoding_temperature": 10.0,
    "logic_act_fn": "SIN01",
    "logic_weight_init": "ri",
    "logic_weight_init_sigma": 0.5,
    "logic_shift_init": False,
    "logic_shift_init_type": "ri",
    "logic_shift_init_shift": 1.2,
    "logic_shift_init_direction": "0101",
    "logic_resconnection_init": False,
    "logic_res_connect_fraction": 0.0,
    "grad_factor": 1.0,
    "logic_connectivity": "fixed",
    "learnable_conn_k": 64,
    "learnable_conn_use_skip_bias": True,
    "logic_conn_lr_multiplier": 0.2,
    "head_type": "linear",
    "signed_head_topk": 32,
    "signed_head_use_topk_mask": False,
    "post_mask_head_calibrate_iters": 0,
    "post_mask_head_calibrate_lr": 0.0,
    "post_mask_head_calibrate_target": "student",
    "teacher_iters": 1,
    "student_iters": 0,
    "learning_rate": 1e-3,
    "student_lr": 3e-4,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "temp_start": 1.0,
    "temp_end": 0.3,
    "temp_warmup_ratio": 0.02,
    "temp_cooldown_ratio": 0.05,
    "distill_alpha": 1.0,
    "distill_tau": 2.0,
    "student_scope": "logic_decode_head",
    "student_hard_thermometer": False,
    "eval_split": "test",
    "eval_max_batches": 20,
    "out_dir": "runs/goal6plus_vit",
    "mixer": "sctm_aux_accum",
    "sctm_topk": 8,
    "sctm_weight_bits": 2,
    "sctm_patch_path": "local3x3",
    "sctm_score_mode": "continuous",
    "sat_state_bits": 4,
    "sat_delta_bits": 4,
    "sat_state_clip": 4.0,
    "sat_delta_clip": 2.0,
    "sat_residual_shift": 1,
    "sat_leak": "none",
    "sat_leak_shift": 2,
    "sat_use_out_proj": True,
    "aux_accum_bits": 4,
    "aux_accum_clip": 2.0,
    "aux_delta_clip": 1.0,
    "aux_scale_shift": 3,
    "aux_accum_init": "cls",
}


VARIANT_LABELS = {
    "baseline": "baseline",
    "drop_top1": "drop top-1 route",
    "drop_top2": "drop top-2 routes",
    "drop_tail1": "drop weakest selected route",
    "drop_random1": "drop random selected route",
    "zero_all": "zero all selected routes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-repo", default=DEFAULT_VIT_REPO)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--random-repeats", type=int, default=8)
    parser.add_argument(
        "--variants",
        default="baseline,drop_top1,drop_top2,drop_tail1,drop_random1,zero_all",
        help="Comma-separated intervention list.",
    )
    parser.add_argument("--out-json", default="remote_logs/vit_sctm_route_causal_20260708.json")
    parser.add_argument("--out-csv", default="remote_logs/vit_sctm_route_causal_20260708.csv")
    return parser.parse_args()


def load_checkpoint_args(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    cfg = dict(FALLBACK_ARGS)
    cfg.update(payload.get("args", {}))
    return cfg


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(x: torch.Tensor) -> str:
    array = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def namespace_from_checkpoint(cfg: Dict[str, Any], batch_size: int, num_workers: int, split: str) -> SimpleNamespace:
    merged = dict(FALLBACK_ARGS)
    merged.update(cfg)
    merged["batch_size"] = int(batch_size)
    merged["num_workers"] = int(num_workers)
    merged["eval_split"] = split
    return SimpleNamespace(**merged)


def import_vit_modules(vit_repo: str):
    repo = str(Path(vit_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import sctm_token_mixer_experiment as sctm  # type: ignore

    return sctm


def apply_weight_intervention(weights: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "baseline":
        return weights
    if variant == "zero_all":
        return torch.zeros_like(weights)

    edited = weights.clone()
    topk = edited.shape[-1]
    if variant == "drop_top1":
        edited[:, :, 0] = 0
    elif variant == "drop_top2":
        edited[:, :, : min(2, topk)] = 0
    elif variant == "drop_tail1":
        edited[:, :, -1] = 0
    elif variant == "drop_random1":
        choices = torch.randint(0, topk, (edited.shape[0], edited.shape[1], 1), device=edited.device)
        edited.scatter_(dim=2, index=choices, value=0.0)
    else:
        raise ValueError("unknown intervention variant: {}".format(variant))

    denom = edited.sum(dim=-1, keepdim=True)
    return torch.where(denom > 1e-12, edited / denom.clamp_min(1e-12), edited)


def init_route_record(num_heads: int, num_patches: int) -> Dict[str, Any]:
    return {
        "num_heads": int(num_heads),
        "num_patches": int(num_patches),
        "counts": torch.zeros(num_heads, num_patches, dtype=torch.long),
        "route_total": 0,
        "weight_entropy_sum": 0.0,
        "weight_entropy_count": 0,
        "top1_weight_sum": 0.0,
        "score_gap_sum": 0.0,
        "batch_calls": 0,
    }


def record_routes(store: Dict[int, Dict[str, Any]], module: Any, indices: torch.Tensor, weights: torch.Tensor, selected_scores: torch.Tensor) -> None:
    block_id = int(getattr(module, "block_id", len(store)))
    if block_id not in store:
        store[block_id] = init_route_record(int(module.num_heads), int(module.num_patches))
    rec = store[block_id]
    idx_cpu = indices.detach().cpu()
    for head in range(int(module.num_heads)):
        flat = idx_cpu[:, head, :].reshape(-1)
        rec["counts"][head].add_(torch.bincount(flat, minlength=int(module.num_patches)))
    rec["route_total"] += int(indices.numel())
    entropy = -(weights.detach() * weights.detach().clamp_min(1e-12).log()).sum(dim=-1)
    rec["weight_entropy_sum"] += float(entropy.sum().item())
    rec["weight_entropy_count"] += int(entropy.numel())
    rec["top1_weight_sum"] += float(weights.detach()[:, :, 0].sum().item())
    if selected_scores.shape[-1] > 1:
        gap = selected_scores.detach()[:, :, 0] - selected_scores.detach()[:, :, -1]
        rec["score_gap_sum"] += float(gap.sum().item())
    rec["batch_calls"] += 1


def summarize_route_records(store: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block_id in sorted(store):
        rec = store[block_id]
        counts = rec["counts"].float()
        total_per_head = counts.sum(dim=1).clamp_min(1.0)
        probs = counts / total_per_head.unsqueeze(1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        entropy_norm = entropy / math.log(max(int(rec["num_patches"]), 2))
        top_mass, top_idx = probs.max(dim=1)
        topk = min(3, int(rec["num_patches"]))
        top_vals, top_indices = probs.topk(topk, dim=1)
        denom = max(int(rec["weight_entropy_count"]), 1)
        row = {
            "block": int(block_id),
            "num_heads": int(rec["num_heads"]),
            "num_patches": int(rec["num_patches"]),
            "route_total": int(rec["route_total"]),
            "mean_route_entropy_norm": float(entropy_norm.mean().item()),
            "mean_top_patch_mass": float(top_mass.mean().item()),
            "max_top_patch_mass": float(top_mass.max().item()),
            "mean_selected_weight_entropy": float(rec["weight_entropy_sum"] / denom),
            "mean_top1_selected_weight": float(rec["top1_weight_sum"] / denom),
            "mean_score_gap_top1_tail": float(rec["score_gap_sum"] / denom),
            "heads": [],
        }
        for head in range(int(rec["num_heads"])):
            row["heads"].append(
                {
                    "head": int(head),
                    "top_patch": int(top_idx[head].item()),
                    "top_patch_mass": float(top_mass[head].item()),
                    "route_entropy_norm": float(entropy_norm[head].item()),
                    "top3_patches": [int(v) for v in top_indices[head].tolist()],
                    "top3_masses": [float(v) for v in top_vals[head].tolist()],
                }
            )
        rows.append(row)
    return rows


def make_intervened_forward(variant: str, route_store: Optional[Dict[int, Dict[str, Any]]]):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_mixer:
            return torch.zeros_like(x)
        batch, tokens, channels = x.shape
        if tokens != self.num_patches + 1 or channels != self.embed_dim:
            raise ValueError("SCTM expected {}, got {}".format((self.num_patches + 1, self.embed_dim), (tokens, channels)))
        cls = x[:, 0]
        patches = x[:, 1:]
        scores = self.route_scores(cls, patches)
        indices, selected_scores = self._topk(scores)
        weights = self._aux_selected_weights(selected_scores)
        if route_store is not None:
            record_routes(route_store, self, indices, weights, selected_scores)
        weights = apply_weight_intervention(weights, variant)

        values = self.v_proj(patches).reshape(batch, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        values, q_int, scale = self._quantize_values(values)
        selected_values = self._gather_values(values, indices)
        if self.bitplane_aggregation:
            if q_int is None or scale is None:
                raise RuntimeError("bitplane aggregation requires quantized value integers")
            selected_q_int = self._gather_values(q_int, indices)
            mixed_heads = self._bitplane_aggregate(selected_values, selected_q_int, scale, weights)
        else:
            mixed_heads = (weights.unsqueeze(-1) * selected_values).sum(dim=2)
        mixed = mixed_heads.reshape(batch, channels)
        baseline_cls_delta = self.out_proj(mixed)
        aux_delta, q_acc, shifted, prev_acc = self._aux_accumulate(cls, baseline_cls_delta)
        cls_delta = baseline_cls_delta + aux_delta
        patch_delta = self._patch_delta(patches)
        self._record_diagnostics(indices.detach(), weights.detach(), patches.detach(), patch_delta.detach(), cls_delta.detach())
        self._record_aux_diagnostics(baseline_cls_delta.detach(), aux_delta.detach(), q_acc.detach(), shifted.detach(), prev_acc.detach())
        out = x.new_zeros(batch, tokens, channels)
        out[:, 0] = cls_delta
        out[:, 1:] = patch_delta
        return out

    return forward


def install_intervention(modules: Iterable[Any], variant: str, route_store: Optional[Dict[int, Dict[str, Any]]]):
    handles = []
    for module in modules:
        original = module.forward
        module.forward = MethodType(make_intervened_forward(variant, route_store), module)
        handles.append((module, original))
    return handles


def restore_intervention(handles: Iterable[Tuple[Any, Any]]) -> None:
    for module, original in handles:
        module.forward = original


def build_loader(sctm: Any, args: SimpleNamespace, split: str):
    train_args = sctm.make_train_args(args)
    _train_loader, valid_loader, test_loader, _train_eval_loader, _num_batches, transform = sctm.load_dataset(train_args)
    if split == "valid":
        if valid_loader is None:
            raise RuntimeError("valid split requested but no validation loader is available")
        return valid_loader, transform
    return test_loader, transform


def evaluate_variant(
    sctm: Any,
    model: Any,
    modules: List[Any],
    loader: Any,
    transform: Any,
    variant: str,
    max_batches: int,
    seed: int,
    baseline_refs: Optional[List[Dict[str, Any]]] = None,
    collect_routes: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    route_store: Optional[Dict[int, Dict[str, Any]]] = {} if collect_routes else None
    handles = install_intervention(modules, variant, route_store)
    refs: List[Dict[str, Any]] = []
    try:
        sctm.prepare_forward(model, "relaxed_eval")
        model.eval()
        total = 0
        correct = 0
        ce_sum = 0.0
        logit_l2_sum = 0.0
        true_logit_drop_sum = 0.0
        baseline_pred_prob_drop_sum = 0.0
        pred_flip = 0
        ref_batches = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                images = sctm.preprocess_batch(images, transform)
                targets = targets.to(sctm.DEVICE, non_blocking=True)
                logits = model(images)
                loss = F.cross_entropy(logits, targets, reduction="sum")
                ce_sum += float(loss.item())
                pred = logits.argmax(dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())

                if baseline_refs is None:
                    probs = F.softmax(logits, dim=1)
                    refs.append(
                        {
                            "image_sha256": tensor_sha256(images),
                            "targets": targets.detach().cpu(),
                            "logits": logits.detach().cpu(),
                            "pred": pred.detach().cpu(),
                            "pred_prob": probs.gather(1, pred[:, None]).squeeze(1).detach().cpu(),
                            "true_logit": logits.gather(1, targets[:, None]).squeeze(1).detach().cpu(),
                        }
                    )
                else:
                    ref = baseline_refs[batch_idx]
                    image_sha = tensor_sha256(images)
                    if image_sha != ref["image_sha256"]:
                        raise RuntimeError("loader image mismatch at batch {}".format(batch_idx))
                    ref_targets = ref["targets"].to(targets.device)
                    if not torch.equal(ref_targets, targets):
                        raise RuntimeError("loader order mismatch at batch {}".format(batch_idx))
                    base_logits = ref["logits"].to(logits.device)
                    base_pred = ref["pred"].to(logits.device)
                    base_pred_prob = ref["pred_prob"].to(logits.device)
                    base_true_logit = ref["true_logit"].to(logits.device)
                    probs = F.softmax(logits, dim=1)
                    logit_l2_sum += float((logits - base_logits).norm(dim=1).sum().item())
                    true_now = logits.gather(1, targets[:, None]).squeeze(1)
                    true_logit_drop_sum += float((base_true_logit - true_now).sum().item())
                    pred_prob_now = probs.gather(1, base_pred[:, None]).squeeze(1)
                    baseline_pred_prob_drop_sum += float((base_pred_prob - pred_prob_now).sum().item())
                    pred_flip += int((pred != base_pred).sum().item())
                    ref_batches += 1

        denom = max(total, 1)
        row: Dict[str, Any] = {
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "samples": int(total),
            "loss": float(ce_sum / denom),
            "acc": float(correct / denom),
        }
        if baseline_refs is not None:
            row.update(
                {
                    "mean_logit_l2_delta": float(logit_l2_sum / denom),
                    "mean_true_logit_drop": float(true_logit_drop_sum / denom),
                    "mean_baseline_pred_prob_drop": float(baseline_pred_prob_drop_sum / denom),
                    "pred_flip_rate": float(pred_flip / denom),
                    "matched_ref_batches": int(ref_batches),
                }
            )
        route_rows = summarize_route_records(route_store or {}) if collect_routes else []
        return row, refs, route_rows
    finally:
        restore_intervention(handles)


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, float(math.sqrt(var))


def aggregate_repeat_rows(variant: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty repeat rows")
    first = rows[0]
    out: Dict[str, Any] = {
        "variant": variant,
        "label": first.get("label", VARIANT_LABELS.get(variant, variant)),
        "samples": int(first["samples"]),
        "random_repeats": int(len(rows)),
        "matched_ref_batches": int(first.get("matched_ref_batches", 0)),
    }
    for key in [
        "loss",
        "acc",
        "mean_logit_l2_delta",
        "mean_true_logit_drop",
        "mean_baseline_pred_prob_drop",
        "pred_flip_rate",
    ]:
        mean, std = mean_std([float(row[key]) for row in rows])
        out[key] = mean
        out[f"{key}_std"] = std
    return out


def aggregate_hash(items: List[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


def add_delta_columns(rows: List[Dict[str, Any]]) -> None:
    base = next(row for row in rows if row["variant"] == "baseline")
    for row in rows:
        row["loss_delta_vs_baseline"] = float(row["loss"] - base["loss"])
        row["acc_delta_vs_baseline"] = float(row["acc"] - base["acc"])
        if "loss_std" in row:
            row["loss_delta_std_vs_baseline"] = float(row["loss_std"])
        if "acc_std" in row:
            row["acc_delta_std_vs_baseline"] = float(row["acc_std"])
        if row["variant"] == "baseline":
            row["mean_logit_l2_delta"] = 0.0
            row["mean_true_logit_drop"] = 0.0
            row["mean_baseline_pred_prob_drop"] = 0.0
            row["pred_flip_rate"] = 0.0
            row["matched_ref_batches"] = 0


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "variant",
        "label",
        "samples",
        "loss",
        "acc",
        "loss_delta_vs_baseline",
        "acc_delta_vs_baseline",
        "mean_logit_l2_delta",
        "mean_true_logit_drop",
        "mean_baseline_pred_prob_drop",
        "pred_flip_rate",
        "matched_ref_batches",
        "random_repeats",
        "loss_std",
        "acc_std",
        "loss_delta_std_vs_baseline",
        "acc_delta_std_vs_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def summarize(rows: List[Dict[str, Any]], route_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = next(row for row in rows if row["variant"] == "baseline")
    non_base = [row for row in rows if row["variant"] != "baseline"]
    strongest = max(non_base, key=lambda row: float(row["loss_delta_vs_baseline"])) if non_base else base
    top1 = next((row for row in rows if row["variant"] == "drop_top1"), None)
    random1 = next((row for row in rows if row["variant"] == "drop_random1"), None)
    route_mass = [float(row["mean_top_patch_mass"]) for row in route_stats]
    route_entropy = [float(row["mean_route_entropy_norm"]) for row in route_stats]
    return {
        "baseline_loss": float(base["loss"]),
        "baseline_acc": float(base["acc"]),
        "strongest_loss_variant": strongest["variant"],
        "strongest_loss_delta": float(strongest["loss_delta_vs_baseline"]),
        "drop_top1_loss_delta": float(top1["loss_delta_vs_baseline"]) if top1 is not None else None,
        "drop_random1_loss_delta": float(random1["loss_delta_vs_baseline"]) if random1 is not None else None,
        "drop_random1_loss_delta_std": float(random1.get("loss_delta_std_vs_baseline", 0.0)) if random1 is not None else None,
        "drop_random1_repeats": int(random1.get("random_repeats", 1)) if random1 is not None else 0,
        "drop_top1_minus_random1_loss_delta": (
            float(top1["loss_delta_vs_baseline"] - random1["loss_delta_vs_baseline"])
            if top1 is not None and random1 is not None
            else None
        ),
        "mean_block_top_patch_mass": float(sum(route_mass) / max(len(route_mass), 1)),
        "mean_block_route_entropy_norm": float(sum(route_entropy) / max(len(route_entropy), 1)),
    }


def main() -> int:
    args = parse_args()
    start = time.time()
    sctm = import_vit_modules(args.vit_repo)
    torch.manual_seed(args.seed)

    checkpoint_args = load_checkpoint_args(args.checkpoint)
    run_args = namespace_from_checkpoint(checkpoint_args, args.batch_size, args.num_workers, args.split)
    loader, transform = build_loader(sctm, run_args, args.split)
    model = sctm.load_checkpoint_model(args.checkpoint, SimpleNamespace(**checkpoint_args))
    model.to(sctm.DEVICE)
    modules = list(sctm.aux_accum_modules(model))
    if not modules:
        raise RuntimeError("checkpoint model has no AuxAccumCLSPatchTokenMixer modules")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if "baseline" not in variants:
        variants.insert(0, "baseline")
    elif variants[0] != "baseline":
        variants = ["baseline"] + [variant for variant in variants if variant != "baseline"]

    rows: List[Dict[str, Any]] = []
    baseline_refs: Optional[List[Dict[str, Any]]] = None
    route_stats: List[Dict[str, Any]] = []
    repeat_rows: Dict[str, List[Dict[str, Any]]] = {}
    variant_seeds: Dict[str, Any] = {}
    for idx, variant in enumerate(variants):
        base_seed = int(args.seed + idx * 1009)
        if variant == "drop_random1" and int(args.random_repeats) > 1:
            current_repeat_rows = []
            current_seeds = []
            for repeat in range(int(args.random_repeats)):
                seed = int(base_seed + repeat * 9176)
                current_seeds.append(seed)
                row, _refs, _route_rows = evaluate_variant(
                    sctm=sctm,
                    model=model,
                    modules=modules,
                    loader=loader,
                    transform=transform,
                    variant=variant,
                    max_batches=args.max_batches,
                    seed=seed,
                    baseline_refs=baseline_refs,
                    collect_routes=False,
                )
                row["repeat"] = int(repeat)
                row["seed"] = seed
                current_repeat_rows.append(row)
            rows.append(aggregate_repeat_rows(variant, current_repeat_rows))
            repeat_rows[variant] = current_repeat_rows
            variant_seeds[variant] = current_seeds
            continue

        variant_seeds[variant] = base_seed
        row, refs, route_rows = evaluate_variant(
            sctm=sctm,
            model=model,
            modules=modules,
            loader=loader,
            transform=transform,
            variant=variant,
            max_batches=args.max_batches,
            seed=base_seed,
            baseline_refs=baseline_refs,
            collect_routes=(variant == "baseline"),
        )
        row["seed"] = base_seed
        rows.append(row)
        if variant == "baseline":
            baseline_refs = refs
            route_stats = route_rows

    add_delta_columns(rows)
    base_for_repeats = next(row for row in rows if row["variant"] == "baseline")
    for current_repeat_rows in repeat_rows.values():
        for row in current_repeat_rows:
            row["loss_delta_vs_baseline"] = float(row["loss"] - base_for_repeats["loss"])
            row["acc_delta_vs_baseline"] = float(row["acc"] - base_for_repeats["acc"])

    source_hashes = {}
    for rel in ["sctm_token_mixer_experiment.py", "goal6plus_vit_experiment.py", "logic_vit_tiny.py"]:
        source_path = Path(args.vit_repo) / rel
        if source_path.exists():
            source_hashes[rel] = file_sha256(source_path)
    baseline_batch_image_sha256 = [str(ref["image_sha256"]) for ref in baseline_refs or []]
    payload = {
        "created_unix": time.time(),
        "elapsed_sec": time.time() - start,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(sctm.DEVICE),
        "seed": int(args.seed),
        "variant_seeds": variant_seeds,
        "variants": variants,
        "random_repeats": int(args.random_repeats),
        "probe_script_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "vit_repo_source_sha256": source_hashes,
        "baseline_batch_image_sha256": baseline_batch_image_sha256,
        "baseline_batch_image_sha256_digest": aggregate_hash(baseline_batch_image_sha256),
        "vit_repo": str(Path(args.vit_repo).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "max_batches": int(args.max_batches),
        "batch_size": int(args.batch_size),
        "checkpoint_args_subset": {
            key: checkpoint_args.get(key)
            for key in [
                "dataset",
                "img_size",
                "patch_size",
                "embed_dim",
                "depth",
                "num_heads",
                "mixer",
                "sctm_topk",
                "sctm_patch_path",
                "sctm_score_mode",
                "aux_accum_bits",
                "sat_residual_shift",
                "head_type",
            ]
        },
        "rows": rows,
        "repeat_rows": repeat_rows,
        "route_stats": route_stats,
        "summary": summarize(rows, route_stats),
        "method_note": (
            "Task-level probe on the actual saved SCTM forward path. "
            "Route interventions edit the selected-route weights after top-k and before V aggregation; "
            "drop_* variants renormalize remaining selected weights, while zero_all zeros all selected-route weights."
        ),
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_csv, rows)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print("wrote {} and {}".format(out_json, out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
