#!/usr/bin/env python3
"""Probe deployable THW geometry masks and an oracle activation-defect tail."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    spatial_radius_tiles: int
    window_temporal_radius: int
    full_temporal_tube: bool
    global_anchor_tiles: int


DEFAULT_SPECS = (
    GeometrySpec("s3", 1, 0, False, 0),
    GeometrySpec("s5", 2, 0, False, 0),
    GeometrySpec("s3_tfull", 1, 0, True, 0),
    GeometrySpec("s5_tfull", 2, 0, True, 0),
    GeometrySpec("s3_temporal_pm2", 1, 2, False, 0),
    GeometrySpec("s3_tfull_anchor12", 1, 0, True, 12),
)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return values


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument(
        "--replay-dir",
        type=Path,
        action="append",
        default=[],
        help="Recursively add *_self.pt replay files from this directory.",
    )
    parser.add_argument(
        "--replay-index",
        type=Path,
        action="append",
        default=[],
        help="CSV capture index whose path column explicitly selects replay files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--tile-height", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=8)
    parser.add_argument("--query-samples", type=int, default=256)
    parser.add_argument(
        "--tail-ranks", type=parse_int_list, default=parse_int_list("0,4,8,16")
    )
    parser.add_argument(
        "--error-targets", type=parse_float_list, default=parse_float_list("0.02,0.05")
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def replay_paths_from_index(path: Path) -> list[Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "path" not in rows[0]:
        raise ValueError(f"replay index has no path rows: {path}")
    replay_paths: list[Path] = []
    for row in rows:
        replay_path = Path(row["path"])
        if not replay_path.is_absolute():
            replay_path = path.parent / replay_path
        replay_paths.append(replay_path.resolve())
    return replay_paths


def infer_grid(frame_num: int, tokens: int, height: int, width: int) -> tuple[int, int, int]:
    temporal = (frame_num - 1) // 4 + 1
    if temporal * height * width != tokens:
        raise ValueError(
            f"token count {tokens} does not match inferred grid "
            f"{temporal}x{height}x{width}"
        )
    return temporal, height, width


def grid_from_metadata(
    metadata: dict[str, object],
    tokens: int,
    fallback_height: int,
    fallback_width: int,
) -> tuple[int, int, int]:
    raw_grid = metadata.get("grid_size")
    if raw_grid is None:
        return infer_grid(
            int(metadata["frame_num"]), tokens, fallback_height, fallback_width
        )
    if not isinstance(raw_grid, (list, tuple)) or len(raw_grid) != 3:
        raise ValueError(f"invalid replay grid_size: {raw_grid!r}")
    shape = tuple(int(value) for value in raw_grid)
    if any(value <= 0 for value in shape) or math.prod(shape) != tokens:
        raise ValueError(
            f"replay grid_size {shape} does not match token count {tokens}"
        )
    return shape


def token_coordinates(
    indices: torch.Tensor, shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, height, width = shape
    spatial = height * width
    temporal = torch.div(indices, spatial, rounding_mode="floor")
    remainder = torch.remainder(indices, spatial)
    row = torch.div(remainder, width, rounding_mode="floor")
    column = torch.remainder(remainder, width)
    return temporal, row, column


def geometry_mask(
    query_indices: torch.Tensor,
    shape: tuple[int, int, int],
    spec: GeometrySpec,
    tile_height: int = 8,
    tile_width: int = 8,
    anchor_phase: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    tokens = shape[0] * shape[1] * shape[2]
    keys = torch.arange(tokens, device=query_indices.device)
    query_t, query_h, query_w = token_coordinates(query_indices, shape)
    key_t, key_h, key_w = token_coordinates(keys, shape)
    tile_rows = (shape[1] + tile_height - 1) // tile_height
    tile_columns = (shape[2] + tile_width - 1) // tile_width
    tile_count = shape[0] * tile_rows * tile_columns
    query_tile_h = torch.div(query_h, tile_height, rounding_mode="floor")
    query_tile_w = torch.div(query_w, tile_width, rounding_mode="floor")
    key_tile_h = torch.div(key_h, tile_height, rounding_mode="floor")
    key_tile_w = torch.div(key_w, tile_width, rounding_mode="floor")
    key_tile_ids = (key_t * tile_rows + key_tile_h) * tile_columns + key_tile_w

    tile_ids = torch.arange(tile_count, device=query_indices.device)
    tile_t = torch.div(tile_ids, tile_rows * tile_columns, rounding_mode="floor")
    tile_remainder = torch.remainder(tile_ids, tile_rows * tile_columns)
    tile_h = torch.div(tile_remainder, tile_columns, rounding_mode="floor")
    tile_w = torch.remainder(tile_remainder, tile_columns)
    anchors = torch.zeros(tile_count, dtype=torch.bool, device=query_indices.device)
    if spec.global_anchor_tiles:
        anchor_count = min(spec.global_anchor_tiles, tile_count)
        anchor_ids = torch.div(
            torch.arange(anchor_count, device=query_indices.device) * tile_count,
            anchor_count,
            rounding_mode="floor",
        )
        anchor_ids = torch.remainder(anchor_ids + anchor_phase, tile_count)
        anchors[anchor_ids] = True

    def select_tiles(
        source_t: torch.Tensor,
        source_h: torch.Tensor,
        source_w: torch.Tensor,
    ) -> torch.Tensor:
        dt = (source_t[:, None] - tile_t[None, :]).abs()
        dh = (source_h[:, None] - tile_h[None, :]).abs()
        dw = (source_w[:, None] - tile_w[None, :]).abs()
        window = (
            (dt <= spec.window_temporal_radius)
            & (dh <= spec.spatial_radius_tiles)
            & (dw <= spec.spatial_radius_tiles)
        )
        tube = (dh == 0) & (dw == 0) if spec.full_temporal_tube else torch.zeros_like(window)
        return window | tube | anchors[None, :]

    tile_mask = select_tiles(query_t, query_tile_h, query_tile_w)
    mask = tile_mask.gather(1, key_tile_ids.expand(query_indices.numel(), -1))
    mask.scatter_(1, query_indices[:, None], True)
    all_tile_mask = select_tiles(tile_t, tile_h, tile_w)
    valid_tokens_per_tile = torch.bincount(key_tile_ids, minlength=tile_count)
    logical_operations = (
        all_tile_mask
        * valid_tokens_per_tile[:, None]
        * valid_tokens_per_tile[None, :]
    ).sum()
    tile_size = tile_height * tile_width
    selected_blocks = all_tile_mask.sum()
    logical_density = float(logical_operations / (tokens * tokens))
    padded_block_density = float(selected_blocks / (tile_count * tile_count))
    execution_density = float(selected_blocks * (tile_size * tile_size) / (tokens * tokens))
    selected_tiles = all_tile_mask.sum(dim=1)
    return mask, {
        "tile_height": tile_height,
        "tile_width": tile_width,
        "tile_rows": tile_rows,
        "tile_columns": tile_columns,
        "tile_count": tile_count,
        "padded_tokens": tile_count * tile_height * tile_width,
        "token_padding_overhead": tile_count * tile_height * tile_width / tokens - 1.0,
        "selected_tiles_mean": float(selected_tiles.float().mean()),
        "selected_tiles_min": int(selected_tiles.min()),
        "selected_tiles_max": int(selected_tiles.max()),
        "padded_block_density": padded_block_density,
        "sampled_query_logical_density": float(mask.float().mean()),
        "logical_density": logical_density,
        "execution_density": execution_density,
    }


def stratified_query_indices(
    shape: tuple[int, int, int],
    samples: int,
    tile_height: int,
    tile_width: int,
    device: torch.device,
) -> torch.Tensor:
    tile_rows = (shape[1] + tile_height - 1) // tile_height
    tile_columns = (shape[2] + tile_width - 1) // tile_width
    tile_count = shape[0] * tile_rows * tile_columns
    tile_ids = (
        torch.linspace(0, tile_count - 1, samples, dtype=torch.float64, device=device)
        .round()
        .long()
    )
    temporal = torch.div(tile_ids, tile_rows * tile_columns, rounding_mode="floor")
    remainder = torch.remainder(tile_ids, tile_rows * tile_columns)
    tile_row = torch.div(remainder, tile_columns, rounding_mode="floor")
    tile_column = torch.remainder(remainder, tile_columns)
    offsets = torch.tensor(
        [[0, 0], [0, tile_width - 1], [tile_height - 1, 0],
         [tile_height - 1, tile_width - 1], [tile_height // 2, tile_width // 2]],
        device=device,
    )
    selected_offsets = offsets[torch.arange(samples, device=device) % offsets.shape[0]]
    row = (tile_row * tile_height + selected_offsets[:, 0]).clamp_max(shape[1] - 1)
    column = (tile_column * tile_width + selected_offsets[:, 1]).clamp_max(shape[2] - 1)
    return temporal * (shape[1] * shape[2]) + row * shape[2] + column


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-30))


def lowrank_defect_corrections(
    defect: torch.Tensor, ranks: tuple[int, ...]
) -> dict[int, tuple[torch.Tensor, float]]:
    results = {rank: (torch.zeros_like(defect), 0.0) for rank in ranks if rank <= 0}
    positive = sorted({rank for rank in ranks if rank > 0})
    if not positive:
        return results
    u, singular, vh = torch.linalg.svd(defect, full_matrices=False)
    energy = singular.square()
    total_energy = energy.sum().clamp_min(1e-30)
    for rank in positive:
        used = min(rank, singular.numel())
        reconstruction = (u[:, :used] * singular[:used]) @ vh[:used]
        results[rank] = (
            reconstruction,
            float(energy[:used].sum() / total_energy),
        )
    return results


def fallback_curve(
    reference: torch.Tensor,
    estimates: torch.Tensor,
    sparse_density: float,
) -> list[dict[str, float | int]]:
    defect = estimates - reference
    head_energy = defect.square().sum(dim=(0, 2))
    order = torch.argsort(head_energy, descending=True)
    rows: list[dict[str, float | int]] = []
    for fallback_heads in range(reference.shape[1] + 1):
        remaining = defect.clone()
        if fallback_heads:
            remaining[:, order[:fallback_heads]] = 0
        estimate = reference + remaining
        rows.append(
            {
                "dense_fallback_heads": fallback_heads,
                "dense_fallback_fraction": fallback_heads / reference.shape[1],
                "effective_attention_density": (
                    fallback_heads
                    + (reference.shape[1] - fallback_heads) * sparse_density
                )
                / reference.shape[1],
                "output_relative_l2": relative_l2(reference, estimate),
                "output_cosine": float(
                    F.cosine_similarity(reference.flatten(), estimate.flatten(), dim=0)
                ),
            }
        )
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    replay_paths = list(args.replay)
    for replay_dir in args.replay_dir:
        replay_paths.extend(sorted(replay_dir.rglob("*_self.pt")))
    for replay_index in args.replay_index:
        replay_paths.extend(replay_paths_from_index(replay_index.resolve()))
    replay_paths = list(dict.fromkeys(path.resolve() for path in replay_paths))
    if not replay_paths:
        raise ValueError(
            "provide at least one --replay, --replay-dir, or --replay-index"
        )
    missing = [path for path in replay_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing replay files: {missing[:3]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    head_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    mask_catalog_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    replay_metadata: list[dict[str, object]] = []
    started = time.time()

    for replay_path in replay_paths:
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        q_all = payload["q"][0]
        k_all = payload["k"][0]
        v_all = payload["v"][0]
        tokens, heads, dimension = q_all.shape
        frame_num = int(metadata["frame_num"])
        shape = grid_from_metadata(
            metadata, tokens, args.grid_height, args.grid_width
        )
        base_case = (
            f"f{frame_num}_l{metadata.get('layer', 'unknown')}_"
            f"t{metadata.get('timestep', 'unknown')}_{metadata.get('branch', 'unknown')}"
        )
        sample_id = str(metadata.get("sample_id", ""))
        case = f"{sample_id}_{base_case}" if sample_id else base_case
        capture_fields = {
            "sample_id": sample_id,
            "prompt_index": metadata.get("prompt_index", ""),
            "seed": metadata.get("seed", ""),
            "layer": metadata.get("layer", ""),
            "sampling_step": metadata.get("sampling_step", ""),
            "timestep": metadata.get("timestep", ""),
            "branch": metadata.get("branch", ""),
        }
        replay_metadata.append({"path": str(replay_path), "case": case, **metadata})
        query_indices = stratified_query_indices(
            shape,
            args.query_samples,
            args.tile_height,
            args.tile_width,
            device,
        )
        masks = {
            spec.name: geometry_mask(
                query_indices,
                shape,
                spec,
                args.tile_height,
                args.tile_width,
                anchor_phase=int(metadata.get("layer", 0)),
            )
            for spec in DEFAULT_SPECS
        }

        states: dict[str, dict[str, object]] = {}
        for spec in DEFAULT_SPECS:
            mask, mask_info = masks[spec.name]
            mask_catalog_rows.append(
                {"case": case, **capture_fields, "mask": spec.name, **asdict(spec), **mask_info}
            )
            states[spec.name] = {
                "spec": spec,
                "mask": mask,
                "mask_info": mask_info,
                "references": [],
                "estimates_by_rank": {rank: [] for rank in args.tail_ranks},
            }

        query_index_values = query_indices.cpu().tolist()
        scale = float(payload.get("softmax_scale", dimension**-0.5))
        for head in range(heads):
            q = q_all[:, head].to(device=device, dtype=torch.float32)
            k = k_all[:, head].to(device=device, dtype=torch.float32)
            v = v_all[:, head].to(device=device, dtype=torch.float32)
            sampled_q = q.index_select(0, query_indices)
            scores = sampled_q @ k.T * scale
            exact_attention = torch.softmax(scores, dim=-1)
            exact_output = exact_attention @ v
            exact_lse = torch.logsumexp(scores, dim=1)
            score_max = scores.max(dim=1, keepdim=True).values
            stable_weights = torch.exp(scores - score_max)
            exact_numerator = stable_weights @ v

            for spec in DEFAULT_SPECS:
                state = states[spec.name]
                mask = state["mask"]
                mask_info = state["mask_info"]
                assert isinstance(mask, torch.Tensor) and isinstance(mask_info, dict)
                logical_density = float(mask_info["logical_density"])
                execution_density = float(mask_info["execution_density"])
                sparse_scores = scores.masked_fill(~mask, float("-inf"))
                sparse_attention = torch.softmax(sparse_scores, dim=-1)
                sparse_output = sparse_attention @ v
                defect = exact_output - sparse_output
                references = state["references"]
                assert isinstance(references, list)
                references.append(exact_output)
                per_query_mass = (exact_attention * mask).sum(dim=1)
                sparse_lse = torch.logsumexp(sparse_scores, dim=1)
                query_error = (sparse_output - exact_output).norm(dim=1) / exact_output.norm(
                    dim=1
                ).clamp_min(1e-30)
                query_cosine = F.cosine_similarity(exact_output, sparse_output, dim=1)
                sparse_numerator = (stable_weights * mask) @ v
                numerator_error = (sparse_numerator - exact_numerator).norm(dim=1) / exact_numerator.norm(
                    dim=1
                ).clamp_min(1e-30)
                query_metrics = torch.stack(
                    [
                        per_query_mass,
                        query_error,
                        query_cosine,
                        (sparse_lse - exact_lse).abs(),
                        numerator_error,
                    ],
                    dim=1,
                ).cpu().tolist()
                for query_index, values in zip(query_index_values, query_metrics):
                    query_rows.append(
                        {
                            "case": case,
                            **capture_fields,
                            "head": head,
                            "mask": spec.name,
                            "query_index": query_index,
                            "logical_density": logical_density,
                            "execution_density": execution_density,
                            "exact_attention_mass": values[0],
                            "output_relative_l2": values[1],
                            "output_cosine": values[2],
                            "absolute_lse_error": values[3],
                            "numerator_relative_l2": values[4],
                        }
                    )

                corrections = lowrank_defect_corrections(defect, args.tail_ranks)
                estimates_by_rank = state["estimates_by_rank"]
                assert isinstance(estimates_by_rank, dict)
                for rank in args.tail_ranks:
                    correction, explained = corrections[rank]
                    estimate = sparse_output + correction
                    estimates_by_rank[rank].append(estimate)
                    tail_rows.append(
                        {
                            "case": case,
                            **capture_fields,
                            "head": head,
                            "mask": spec.name,
                            "tail_rank": rank,
                            "tokens": tokens,
                            "query_samples": query_indices.numel(),
                            "oracle_only": rank > 0,
                            "logical_density": logical_density,
                            "execution_density": execution_density,
                            "keys_mean": float(mask.sum(dim=1).float().mean()),
                            "keys_min": int(mask.sum(dim=1).min()),
                            "keys_max": int(mask.sum(dim=1).max()),
                            "exact_attention_mass": float(per_query_mass.mean()),
                            "tail_defect_energy_explained": explained,
                            "output_relative_l2": relative_l2(exact_output, estimate),
                            "output_cosine": float(
                                F.cosine_similarity(
                                    exact_output.flatten(), estimate.flatten(), dim=0
                                )
                            ),
                        }
                    )
                p95_error = float(torch.quantile(query_error, 0.95))
                p99_error = float(torch.quantile(query_error, 0.99))
                p05_cosine = float(torch.quantile(query_cosine, 0.05))
                p95_lse = float(torch.quantile((sparse_lse - exact_lse).abs(), 0.95))
                head_rows.append(
                    {
                        "case": case,
                        **capture_fields,
                        "head": head,
                        "mask": spec.name,
                        "logical_density": logical_density,
                        "execution_density": execution_density,
                        "exact_attention_mass_mean": float(per_query_mass.mean()),
                        "output_relative_l2": relative_l2(exact_output, sparse_output),
                        "defect_squared_norm": float(defect.square().sum()),
                        "reference_squared_norm": float(exact_output.square().sum()),
                        "query_output_nrmse_p95": p95_error,
                        "query_output_nrmse_p99": p99_error,
                        "query_cosine_p05": p05_cosine,
                        "absolute_lse_error_p95": p95_lse,
                        "numerator_relative_l2_p95": float(
                            torch.quantile(numerator_error, 0.95)
                        ),
                        "static_sparse_head_gate": (
                            p95_error <= 0.03
                            and p99_error <= 0.075
                            and p05_cosine >= 0.995
                            and p95_lse <= 0.10
                        ),
                    }
                )
                del sparse_scores, sparse_attention, sparse_output, defect, corrections

            del q, k, v, sampled_q, scores, exact_attention, stable_weights
            del exact_output, exact_lse, exact_numerator

        for spec in DEFAULT_SPECS:
            state = states[spec.name]
            mask_info = state["mask_info"]
            references = state["references"]
            estimates_by_rank = state["estimates_by_rank"]
            assert isinstance(mask_info, dict)
            assert isinstance(references, list) and isinstance(estimates_by_rank, dict)
            logical_density = float(mask_info["logical_density"])
            execution_density = float(mask_info["execution_density"])
            reference = torch.stack(references, dim=1)
            for rank, head_estimates in estimates_by_rank.items():
                estimates = torch.stack(head_estimates, dim=1)
                curve = fallback_curve(reference, estimates, execution_density)
                for row in curve:
                    curve_rows.append(
                        {
                            "case": case,
                            **capture_fields,
                            "mask": spec.name,
                            "tail_rank": rank,
                            "fallback_selection": "oracle_current_replay_defect_energy",
                            "logical_sparse_density": logical_density,
                            "execution_sparse_density": execution_density,
                            **row,
                        }
                    )
                for target in args.error_targets:
                    feasible = [row for row in curve if float(row["output_relative_l2"]) <= target]
                    selected = feasible[0] if feasible else curve[-1]
                    gate_rows.append(
                        {
                            "case": case,
                            **capture_fields,
                            "mask": spec.name,
                            "tail_rank": rank,
                            "error_target": target,
                            "target_reached": bool(feasible),
                            **selected,
                        }
                    )
            print(
                f"[geometry] case={case} mask={spec.name} "
                f"logical={logical_density:.5f} execution={execution_density:.5f}",
                flush=True,
            )
            del reference
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del payload, q_all, k_all, v_all, masks, states

    write_csv(args.output_dir / "geometry_mask_catalog.csv", mask_catalog_rows)
    write_csv(args.output_dir / "geometry_attention_queries.csv", query_rows)
    write_csv(args.output_dir / "geometry_attention_heads.csv", head_rows)
    write_csv(args.output_dir / "geometry_attention_oracle_tail.csv", tail_rows)
    write_csv(args.output_dir / "geometry_attention_fallback_curve.csv", curve_rows)
    write_csv(args.output_dir / "geometry_attention_gates.csv", gate_rows)
    manifest = {
        "arguments": {
            key: [str(item) for item in value]
            if key in {"replay", "replay_dir"}
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "geometry_specs": [asdict(spec) for spec in DEFAULT_SPECS],
        "replays": replay_metadata,
        "resolved_replay_paths": [str(path) for path in replay_paths],
        "methodology": {
            "mask": (
                "content-independent 1x8x8 THW tile union of a spatial/temporal window, "
                "an optional all-frame same-space tube, and optional fixed global tiles"
            ),
            "sparse_attention": "exact sampled-query softmax restricted and renormalized on the geometry mask",
            "tail": (
                "best rank-r SVD of the sampled pre-output-projection activation defect; "
                "its coefficients are oracle-only and are not a deployable speedup"
            ),
            "fallback": "highest defect-energy heads are restored to dense attention",
            "grid_provenance": (
                "T is derived from frame_num and Wan temporal stride 4; H/W are explicit CLI "
                "values validated against the 480x832, VAE-stride-8, patch-size-2 capture"
            ),
            "warning": (
                "quality metrics are local pre-o-projection replay probes, not end-to-end video evidence; "
                "actual acceleration requires a fused block-sparse kernel"
            ),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "elapsed_seconds": time.time() - started,
        "head_rows": len(head_rows),
        "query_rows": len(query_rows),
        "tail_rows": len(tail_rows),
        "curve_rows": len(curve_rows),
        "gate_rows": len(gate_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[geometry] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
