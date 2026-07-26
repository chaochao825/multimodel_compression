#!/usr/bin/env python3
"""Probe a content-derived block-moment marginal attention path.

The deployable candidate never freezes an output correction basis.  It keeps
exact attention for router-selected 64-key blocks and approximates every
omitted block with moments computed from the current K/V tensors.  Centroid
and diagonal-Gaussian moments are evaluated against an oracle-mass router to
separate routing error from marginal approximation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch

from probe_dynamic_sparse_lowrank_oracle import aligned_query_tile_starts
from summarize_attention_head_stability import classify_head


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return values


def parse_strings(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated string list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--head-stats-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_ints, default=(0, 14, 29))
    parser.add_argument("--steps", type=parse_ints, default=(0, 9, 19))
    parser.add_argument("--branches", type=parse_strings, default=("cond", "uncond"))
    parser.add_argument("--query-tile-size", type=int, default=64)
    parser.add_argument("--query-tiles", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tail-group-sizes", type=parse_ints, default=(64, 32, 16))
    parser.add_argument("--densities", type=parse_floats, default=(0.0, 0.03125, 0.0625, 0.125))
    parser.add_argument("--methods", type=parse_strings, default=("centroid", "diag_gaussian"))
    parser.add_argument("--routers", type=parse_strings, default=("moment", "oracle_mass"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-target", type=float, default=0.01)
    parser.add_argument("--worst-head-target", type=float, default=0.02)
    parser.add_argument("--max-work-ratio", type=float, default=0.50)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resolve_index_path(index_path: Path, raw_path: str, label: str | None = None) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    candidates = [index_path.parent / path.name]
    if label:
        candidates.insert(0, index_path.parent / label / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def read_capture_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    index_path = args.capture_index.resolve()
    with index_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        layer = int(row["layer"])
        step = int(row["sampling_step"])
        branch = row["branch"]
        if layer not in args.layers or step not in args.steps or branch not in args.branches:
            continue
        rows.append(
            {
                **row,
                "prompt_index": int(row["prompt_index"]),
                "seed": int(row["seed"]),
                "sampling_step": step,
                "layer": layer,
                "path": resolve_index_path(index_path, row["path"], row["sample_id"]),
            }
        )
    if not rows:
        raise ValueError("capture selection is empty")
    return rows


def read_head_roles(index_path: Path | None) -> dict[tuple[str, int, str, int, int], str]:
    if index_path is None:
        return {}
    index_path = index_path.resolve()
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    roles: dict[tuple[str, int, str, int, int], str] = {}
    for row in rows:
        head_path = resolve_index_path(index_path, row["head_csv"], row["label"])
        with head_path.open(newline="", encoding="utf-8") as handle:
            heads = list(csv.DictReader(handle))
        for head in heads:
            features = {
                "actual_normalized_entropy_mean": float(
                    head["actual_normalized_entropy_mean"]
                ),
                "geometry_mass_mean": float(head["geometry_mass_mean"]),
            }
            roles[
                (
                    row["sample_id"],
                    int(row["sampling_step"]),
                    row["branch"],
                    int(row["layer"]),
                    int(head["head"]),
                )
            ] = classify_head(features)
    return roles


def split_name(sample_id: str) -> str:
    if sample_id.startswith("s00_"):
        return "calibration"
    if sample_id.startswith("s01_"):
        return "validation"
    return "test"


def padded_groups(
    keys: torch.Tensor, values: torch.Tensor, group_size: int
) -> dict[str, torch.Tensor]:
    if group_size <= 0:
        raise ValueError("group size must be positive")
    tokens, dimension = keys.shape
    groups = math.ceil(tokens / group_size)
    padded_tokens = groups * group_size
    padding = padded_tokens - tokens
    if padding:
        keys = torch.cat([keys, keys.new_zeros((padding, dimension))])
        values = torch.cat([values, values.new_zeros((padding, dimension))])
    valid = torch.arange(padded_tokens, device=keys.device) < tokens
    valid = valid.reshape(groups, group_size)
    counts = valid.sum(dim=1).to(keys.dtype)
    mask = valid.unsqueeze(-1).to(keys.dtype)
    grouped_keys = keys.reshape(groups, group_size, dimension)
    grouped_values = values.reshape(groups, group_size, dimension)
    key_mean = (grouped_keys * mask).sum(dim=1) / counts[:, None]
    value_mean = (grouped_values * mask).sum(dim=1) / counts[:, None]
    centered_keys = (grouped_keys - key_mean[:, None]) * mask
    centered_values = (grouped_values - value_mean[:, None]) * mask
    key_variance = centered_keys.square().sum(dim=1) / counts[:, None]
    diagonal_cross_covariance = (
        centered_keys * centered_values
    ).sum(dim=1) / counts[:, None]
    return {
        "counts": counts,
        "key_mean": key_mean,
        "value_mean": value_mean,
        "key_variance": key_variance,
        "diagonal_cross_covariance": diagonal_cross_covariance,
    }


def moment_logits(
    queries: torch.Tensor,
    moments: dict[str, torch.Tensor],
    scale: float,
    method: str,
) -> torch.Tensor:
    logits = queries @ moments["key_mean"].T * scale
    logits = logits + moments["counts"].log()[None]
    if method == "diag_gaussian":
        logits = logits + 0.5 * (
            queries.square() @ moments["key_variance"].T
        ) * (scale * scale)
    elif method != "centroid":
        raise ValueError(f"unsupported moment method: {method}")
    return logits


def choose_blocks(
    router_logits: torch.Tensor, density: float
) -> torch.Tensor:
    blocks = router_logits.shape[1]
    selected_count = max(0, min(blocks, int(round(blocks * density))))
    selected = torch.zeros(blocks, dtype=torch.bool, device=router_logits.device)
    if selected_count:
        indices = router_logits.mean(dim=0).topk(selected_count).indices
        selected[indices] = True
    return selected


def combine_exact_and_moments(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int,
    tail_moments: dict[str, torch.Tensor],
    tail_group_size: int,
    scale: float,
    method: str,
) -> tuple[torch.Tensor, float, int, int]:
    tokens = keys.shape[0]
    key_block_ids = torch.arange(tokens, device=keys.device) // block_size
    selected_keys = selected_blocks.index_select(0, key_block_ids)
    exact_keys = keys[selected_keys]
    exact_values = values[selected_keys]
    exact_logits = queries @ exact_keys.T * scale if exact_keys.numel() else None

    groups_per_block = block_size // tail_group_size
    tail_parent_blocks = (
        torch.arange(tail_moments["counts"].shape[0], device=keys.device)
        // groups_per_block
    )
    tail_active = ~selected_blocks.index_select(0, tail_parent_blocks)
    tail_logits = moment_logits(queries, tail_moments, scale, method)
    tail_logits = tail_logits.masked_fill(~tail_active[None], float("-inf"))

    maxima = tail_logits.max(dim=1).values
    if exact_logits is not None:
        maxima = torch.maximum(maxima, exact_logits.max(dim=1).values)
    tail_weights = torch.exp(tail_logits - maxima[:, None])
    tail_weights = torch.where(tail_active[None], tail_weights, torch.zeros_like(tail_weights))
    numerator = tail_weights @ tail_moments["value_mean"]
    if method == "diag_gaussian":
        numerator = numerator + (
            tail_weights @ tail_moments["diagonal_cross_covariance"]
        ) * (queries * scale)
    denominator = tail_weights.sum(dim=1)
    if exact_logits is not None:
        exact_weights = torch.exp(exact_logits - maxima[:, None])
        numerator = numerator + exact_weights @ exact_values
        denominator = denominator + exact_weights.sum(dim=1)
    output = numerator / denominator.clamp_min(1e-30)[:, None]

    selected_key_count = int(selected_keys.sum())
    active_tail_groups = int(tail_active.sum())
    selected_mass_proxy = selected_key_count / max(tokens, 1)
    return output, selected_mass_proxy, selected_key_count, active_tail_groups


def oracle_block_logits(
    probabilities: torch.Tensor, tokens: int, block_size: int
) -> torch.Tensor:
    blocks = math.ceil(tokens / block_size)
    padding = blocks * block_size - tokens
    if padding:
        probabilities = torch.cat(
            [probabilities, probabilities.new_zeros((probabilities.shape[0], padding))],
            dim=1,
        )
    return probabilities.reshape(probabilities.shape[0], blocks, block_size).sum(dim=2).log()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@torch.inference_mode()
def process_capture(
    row: dict[str, object],
    args: argparse.Namespace,
    roles: dict[tuple[str, int, str, int, int], str],
    device: torch.device,
) -> list[dict[str, object]]:
    payload = torch.load(row["path"], map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all = payload["q"][0]
    k_all = payload["k"][0]
    v_all = payload["v"][0]
    tokens, heads, dimension = q_all.shape
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    starts = aligned_query_tile_starts(tokens, args.query_tile_size, args.query_tiles)
    output: list[dict[str, object]] = []
    for head in range(heads):
        keys = k_all[:, head].to(device=device, dtype=torch.float32)
        values = v_all[:, head].to(device=device, dtype=torch.float32)
        router_moments = padded_groups(keys, values, args.block_size)
        tail_by_size = {
            size: padded_groups(keys, values, size) for size in args.tail_group_sizes
        }
        accumulators: dict[tuple[str, str, float, int], dict[str, float]] = defaultdict(
            lambda: {
                "residual_sq": 0.0,
                "reference_sq": 0.0,
                "attention_mass": 0.0,
                "router_mass_proxy": 0.0,
                "tiles": 0.0,
                "selected_keys": 0.0,
                "tail_landmarks": 0.0,
            }
        )
        for start in starts:
            queries = q_all[start : start + args.query_tile_size, head].to(
                device=device, dtype=torch.float32
            )
            scores = queries @ keys.T * scale
            probabilities = torch.softmax(scores, dim=1)
            reference = probabilities @ values
            reference_sq = float(reference.square().sum())
            oracle_logits = oracle_block_logits(probabilities, tokens, args.block_size)
            for method in args.methods:
                router_logits = moment_logits(queries, router_moments, scale, method)
                for router in args.routers:
                    active_router = router_logits if router == "moment" else oracle_logits
                    for density in args.densities:
                        selected = choose_blocks(active_router, density)
                        router_probabilities = torch.softmax(active_router, dim=1)
                        router_mass_proxy = float(
                            router_probabilities[:, selected].sum(dim=1).mean()
                        )
                        selected_token_mask = selected.index_select(
                            0, torch.arange(tokens, device=device) // args.block_size
                        )
                        selected_mass = float(probabilities[:, selected_token_mask].sum(dim=1).mean())
                        for tail_size, tail_moments in tail_by_size.items():
                            approximation, _, selected_keys, tail_groups = combine_exact_and_moments(
                                queries,
                                keys,
                                values,
                                selected,
                                args.block_size,
                                tail_moments,
                                tail_size,
                                scale,
                                method,
                            )
                            residual_sq = float((reference - approximation).square().sum())
                            record = accumulators[(method, router, density, tail_size)]
                            record["residual_sq"] += residual_sq
                            record["reference_sq"] += reference_sq
                            record["attention_mass"] += selected_mass
                            record["router_mass_proxy"] += router_mass_proxy
                            record["tiles"] += 1
                            record["selected_keys"] += selected_keys
                            record["tail_landmarks"] += tail_groups
        role = roles.get(
            (
                str(row["sample_id"]),
                int(row["sampling_step"]),
                str(row["branch"]),
                int(row["layer"]),
                head,
            ),
            "unknown",
        )
        for (method, router, density, tail_size), record in accumulators.items():
            tiles = record["tiles"]
            selected_keys = record["selected_keys"] / tiles
            tail_landmarks = record["tail_landmarks"] / tiles
            router_landmarks = math.ceil(tokens / args.block_size)
            work_ratio = (selected_keys + tail_landmarks + router_landmarks) / tokens
            output.append(
                {
                    "split": split_name(str(row["sample_id"])),
                    "sample_id": row["sample_id"],
                    "prompt_index": row["prompt_index"],
                    "seed": row["seed"],
                    "sampling_step": row["sampling_step"],
                    "branch": row["branch"],
                    "layer": row["layer"],
                    "head": head,
                    "head_role": role,
                    "method": method,
                    "router": router,
                    "density": density,
                    "tail_group_size": tail_size,
                    "query_tiles": len(starts),
                    "query_tile_size": args.query_tile_size,
                    "selected_attention_mass": record["attention_mass"] / tiles,
                    "router_selected_mass_proxy": record["router_mass_proxy"] / tiles,
                    "selected_keys": selected_keys,
                    "tail_landmarks": tail_landmarks,
                    "router_landmarks": router_landmarks,
                    "attention_work_ratio": work_ratio,
                    "flop_proxy_speedup": 1.0 / work_ratio,
                    "residual_sq": record["residual_sq"],
                    "reference_sq": record["reference_sq"],
                    "output_relative_l2": math.sqrt(
                        record["residual_sq"] / max(record["reference_sq"], 1e-30)
                    ),
                }
            )
        del keys, values, router_moments, tail_by_size
    print(
        f"[block-moment] sample={row['sample_id']} step={row['sampling_step']} "
        f"branch={row['branch']} layer={row['layer']} heads={heads}",
        flush=True,
    )
    return output


def aggregate_rows(
    rows: list[dict[str, object]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        errors = [float(row["output_relative_l2"]) for row in group]
        output.append(
            {
                **dict(zip(group_fields, key)),
                "records": len(group),
                "aggregate_output_relative_l2": math.sqrt(
                    residual_sq / max(reference_sq, 1e-30)
                ),
                "record_error_p95": quantile(errors, 0.95),
                "record_error_max": max(errors),
                "selected_attention_mass_mean": sum(
                    float(row["selected_attention_mass"]) for row in group
                )
                / len(group),
                "router_selected_mass_proxy_mean": sum(
                    float(row["router_selected_mass_proxy"]) for row in group
                )
                / len(group),
                "attention_work_ratio_mean": sum(
                    float(row["attention_work_ratio"]) for row in group
                )
                / len(group),
                "flop_proxy_speedup": 1.0
                / (
                    sum(float(row["attention_work_ratio"]) for row in group)
                    / len(group)
                ),
            }
        )
    return output


def make_decision(
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    test_summary = [
        row
        for row in summary
        if row["split"] == "test"
        and row["router"] == "moment"
        and float(row["attention_work_ratio_mean"]) <= args.max_work_ratio
    ]
    if not test_summary:
        return {
            "verdict": "NO_GO_NO_DEPLOYABLE_CONFIGURATION",
            "best_deployable_configuration": None,
            "best_by_head_role": {},
            "gates": {
                "aggregate_output_relative_l2": args.aggregate_target,
                "every_record_output_relative_l2": args.worst_head_target,
                "attention_work_ratio": args.max_work_ratio,
                "real_h200_local_speedup_continue": 1.5,
                "real_h200_local_speedup_target": 2.0,
            },
            "warning": "no moment-router configuration satisfied the arithmetic work gate",
        }
    best = min(
        test_summary,
        key=lambda row: (
            float(row["aggregate_output_relative_l2"]),
            float(row["record_error_max"]),
        ),
    )
    role_summary = aggregate_rows(
        [row for row in rows if row["split"] == "test"],
        (
            "split",
            "head_role",
            "method",
            "router",
            "density",
            "tail_group_size",
        ),
    )
    role_best = {}
    for role in ("localized", "transitional", "diffuse"):
        candidates = [
            row
            for row in role_summary
            if row["head_role"] == role
            and row["router"] == "moment"
            and float(row["attention_work_ratio_mean"]) <= args.max_work_ratio
        ]
        if candidates:
            role_best[role] = min(
                candidates,
                key=lambda row: (
                    float(row["aggregate_output_relative_l2"]),
                    float(row["record_error_max"]),
                ),
            )
    strict_go = (
        float(best["aggregate_output_relative_l2"]) <= args.aggregate_target
        and float(best["record_error_max"]) <= args.worst_head_target
    )
    return {
        "verdict": (
            "GO_H200_KERNEL_PILOT_NUMERICAL_ONLY"
            if strict_go
            else "NO_GO_CURRENT_BLOCK_MOMENTS"
        ),
        "best_deployable_configuration": best,
        "best_by_head_role": role_best,
        "gates": {
            "aggregate_output_relative_l2": args.aggregate_target,
            "every_record_output_relative_l2": args.worst_head_target,
            "attention_work_ratio": args.max_work_ratio,
            "real_h200_local_speedup_continue": 1.5,
            "real_h200_local_speedup_target": 2.0,
        },
        "warning": (
            "a numerical GO only permits a fused-kernel pilot; the arithmetic work ratio "
            "excludes K/V moment reduction, routing, gather, softmax, and launch overhead, "
            "so it is not an H200 speedup claim"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.query_tile_size <= 0 or args.query_tiles <= 0:
        raise ValueError("block and query tile sizes must be positive")
    if any(args.block_size % size for size in args.tail_group_sizes):
        raise ValueError("every tail group size must divide block size")
    if any(not 0.0 <= density <= 1.0 for density in args.densities):
        raise ValueError("densities must lie in [0, 1]")
    if not set(args.methods) <= {"centroid", "diag_gaussian"}:
        raise ValueError("unsupported moment method")
    if not set(args.routers) <= {"moment", "oracle_mass"}:
        raise ValueError("unsupported router")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_capture_rows(args)
    roles = read_head_roles(args.head_stats_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started = time.time()
    detail = []
    for row in rows:
        detail.extend(process_capture(row, args, roles, device))
    summary = aggregate_rows(
        detail,
        ("split", "method", "router", "density", "tail_group_size"),
    )
    cells = aggregate_rows(
        detail,
        (
            "split",
            "sampling_step",
            "branch",
            "layer",
            "method",
            "router",
            "density",
            "tail_group_size",
        ),
    )
    roles_summary = aggregate_rows(
        detail,
        (
            "split",
            "head_role",
            "method",
            "router",
            "density",
            "tail_group_size",
        ),
    )
    decision = make_decision(detail, summary, args)
    write_csv(args.output_dir / "block_moment_marginal_heads.csv", detail)
    write_csv(args.output_dir / "block_moment_marginal_summary.csv", summary)
    write_csv(args.output_dir / "block_moment_marginal_cells.csv", cells)
    write_csv(args.output_dir / "block_moment_marginal_roles.csv", roles_summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "scope": "F81 content-derived exact-critical plus block-moment marginal probe",
        "capture_index": str(args.capture_index.resolve()),
        "head_stats_index": str(args.head_stats_index.resolve())
        if args.head_stats_index
        else None,
        "captures": len(rows),
        "arguments": {
            key: list(value) if isinstance(value, tuple) else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "methodology": (
            "moment router uses current pooled K; selected blocks use exact scores; "
            "unselected blocks use current K/V centroid or diagonal-Gaussian moments"
        ),
        "oracle_boundary": (
            "oracle_mass router is diagnostic only; moment router does not read dense attention"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[block-moment] verdict={decision['verdict']}", flush=True)
    print(f"[block-moment] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
