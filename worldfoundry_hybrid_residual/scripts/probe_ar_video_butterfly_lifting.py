"""Evaluate causal Butterfly-lifting K/V memory on LongLive captures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from ar_video_butterfly_lifting_core import (
    build_lifting_tree,
    canonicalize_rope_keys,
    estimate_storage,
    iter_merge_nodes,
    middle_frame_indices,
    reconstruct_lifting_tree,
    restore_rope_keys,
    select_detail_tiles,
)
from ar_video_residual_memory_core import adaptive_rank_projection, dense_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--max-captures", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_key(metadata: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(metadata["prompt_id"]),
        int(metadata["layer"]),
        int(metadata["current_start_frame"]),
        int(metadata["denoising_call_index"]),
    )


def required_capture_keys(protocol: dict[str, Any]) -> set[tuple[str, int, int, int]]:
    scope = protocol["scope"]
    return {
        (
            str(prompt_id),
            int(cell["layer"]),
            int(cell["current_start_frame"]),
            int(cell["denoising_call_index"]),
        )
        for prompt_id in scope["prompt_ids"]
        for cell in scope["cells"]
    }


def _error_components(
    target: torch.Tensor, estimate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.shape != estimate.shape or target.ndim != 3:
        raise ValueError("target/estimate must match [queries, heads, dim]")
    delta = target.float() - estimate.float()
    numerator_sq = delta.square().sum(dim=(0, 2))
    denominator_sq = target.float().square().sum(dim=(0, 2)).clamp_min(1e-24)
    return (numerator_sq / denominator_sq).sqrt(), numerator_sq, denominator_sq


def _grid(metadata: dict[str, Any]) -> tuple[int, int]:
    grids = metadata.get("grid_sizes")
    if not isinstance(grids, list) or len(grids) != 1 or len(grids[0]) != 3:
        raise ValueError("capture requires one [frames, height, width] grid")
    return int(grids[0][1]), int(grids[0][2])


def _validate_capture(
    payload: dict[str, Any], source_protocol: dict[str, Any], protocol: dict[str, Any]
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("capture metadata is missing")
    if metadata.get("protocol_id") != source_protocol["protocol_id"]:
        raise ValueError("capture source protocol mismatch")
    for name in ("query", "key", "value", "dense_output", "rope_freqs"):
        if name not in payload or not isinstance(payload[name], torch.Tensor):
            raise ValueError(f"capture tensor is missing: {name}")
    height, width = _grid(metadata)
    scope = protocol["scope"]
    if (height, width) != (int(scope["height"]), int(scope["width"])):
        raise ValueError("capture grid differs from the frozen protocol")
    frames = int(metadata["key_frames"])
    spatial = height * width
    if payload["key"].shape[0] != frames * spatial:
        raise ValueError("captured key count does not match frame/grid metadata")
    if payload["key"].shape != payload["value"].shape:
        raise ValueError("this probe requires matching K/V shapes")


def _variant_groups(
    protocol: dict[str, Any],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    names: set[str] = set()
    for variant in protocol["methods"]["variants"]:
        name = str(variant["name"])
        if name in names:
            raise ValueError(f"duplicate variant name: {name}")
        names.add(name)
        domain = str(variant["key_domain"])
        scope = str(variant["shift_scope"])
        window_schedule = str(variant.get("window_schedule", "global"))
        if domain not in {"post_rope", "canonical_pre_rope"}:
            raise ValueError(f"unsupported key domain: {domain}")
        if scope not in {"identity", "shared", "per_head", "window_shared"}:
            raise ValueError(f"unsupported shift scope: {scope}")
        if scope == "window_shared" and window_schedule not in {"fixed", "staggered"}:
            raise ValueError(f"unsupported window schedule: {window_schedule}")
        if scope != "window_shared" and window_schedule != "global":
            raise ValueError("non-window shifts must use the global schedule")
        fraction = float(variant["detail_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("detail fraction must lie in [0, 1]")
        groups[(domain, scope, window_schedule)].append(variant)
    return dict(groups)


def _shifts_json(root) -> str:
    payload = []
    for node in iter_merge_nodes(root):
        assert node.shifts is not None
        payload.append(
            {
                "frames": list(node.frame_indices),
                "shifts": node.shifts.detach().cpu().tolist(),
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def evaluate_capture(
    path: Path,
    payload: dict[str, Any],
    protocol: dict[str, Any],
    source_protocol: dict[str, Any],
    device: torch.device,
    query_chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_capture(payload, source_protocol, protocol)
    metadata = payload["metadata"]
    query = payload["query"].to(device)
    key_flat = payload["key"].to(device)
    value_flat = payload["value"].to(device)
    target = payload["dense_output"].to(device)
    rope_freqs = payload["rope_freqs"].to(device)
    height, width = _grid(metadata)
    frames = int(metadata["key_frames"])
    spatial = height * width
    heads, key_dim = key_flat.shape[1:]
    value_dim = value_flat.shape[-1]
    key = key_flat.reshape(frames, spatial, heads, key_dim)
    value = value_flat.reshape(frames, spatial, heads, value_dim)

    dense_math = dense_attention(query, key_flat, value_flat, query_chunk_size)
    parity, _, _ = _error_components(target, dense_math)
    scope = protocol["scope"]
    exact_indices, middle_indices = middle_frame_indices(
        frames,
        int(scope["exact_sink_frames"]),
        int(scope["exact_recent_frames"]),
    )
    if not middle_indices:
        raise ValueError("frozen protocol leaves no frames for lifting")

    transform = protocol["transform"]
    candidates = [tuple(int(value) for value in pair) for pair in transform["shift_candidates"]]
    key_weight = float(transform["key_weight"])
    value_weight = float(transform["value_weight"])
    tile_size = int(scope["detail_tile_size"])
    rank = int(protocol["methods"]["adaptive_residual_rank"])

    canonical_key: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    variant_debug: list[dict[str, Any]] = []
    window_shape = tuple(int(value) for value in transform.get("window_shape", [8, 8]))
    schedules = transform.get(
        "window_offset_schedules",
        {"fixed": [[0, 0]], "staggered": [[0, 0], [4, 4], [0, 4]]},
    )
    for (key_domain, shift_scope, window_schedule), variants in _variant_groups(protocol).items():
        if key_domain == "canonical_pre_rope":
            if canonical_key is None:
                canonical_key = canonicalize_rope_keys(
                    key.float(),
                    metadata["key_frame_ids"],
                    height,
                    width,
                    rope_freqs,
                )
            transform_key = canonical_key
        else:
            transform_key = key.float()
        window_offsets = (
            [tuple(int(value) for value in pair) for pair in schedules[window_schedule]]
            if shift_scope == "window_shared"
            else [(0, 0)]
        )
        middle_tensor = torch.tensor(middle_indices, dtype=torch.long, device=device)
        tree = build_lifting_tree(
            transform_key.index_select(0, middle_tensor),
            value.float().index_select(0, middle_tensor),
            middle_indices,
            candidates,
            height,
            width,
            shift_scope,
            key_weight,
            value_weight,
            window_shape if shift_scope == "window_shared" else None,
            window_offsets,
        )
        shift_json = _shifts_json(tree)
        for variant in variants:
            fraction = float(variant["detail_fraction"])
            selection = select_detail_tiles(
                tree, tile_size, fraction, key_weight, value_weight
            )
            reconstructed_key_map, reconstructed_value_map = reconstruct_lifting_tree(
                tree, selection, height, width
            )
            reconstructed_domain_key = transform_key.clone()
            reconstructed_value = value.float().clone()
            for frame in middle_indices:
                reconstructed_domain_key[frame] = reconstructed_key_map[frame]
                reconstructed_value[frame] = reconstructed_value_map[frame]
            if key_domain == "canonical_pre_rope":
                reconstructed_key = restore_rope_keys(
                    reconstructed_domain_key,
                    metadata["key_frame_ids"],
                    height,
                    width,
                    rope_freqs,
                    key.dtype,
                )
                # Exact branches should remain bit-identical to the captured cache.
                exact_tensor = torch.tensor(exact_indices, dtype=torch.long, device=device)
                reconstructed_key.index_copy_(0, exact_tensor, key.index_select(0, exact_tensor))
            else:
                reconstructed_key = reconstructed_domain_key.to(key.dtype)
            exact_tensor = torch.tensor(exact_indices, dtype=torch.long, device=device)
            reconstructed_value.index_copy_(0, exact_tensor, value.float().index_select(0, exact_tensor))
            estimate = dense_attention(
                query,
                reconstructed_key.reshape_as(key_flat),
                reconstructed_value.to(value.dtype).reshape_as(value_flat),
                query_chunk_size,
            )
            defect = target - estimate
            adaptive = estimate + adaptive_rank_projection(defect, rank)
            storage = estimate_storage(
                frames=frames,
                exact_frames=len(exact_indices),
                spatial=spatial,
                heads=heads,
                key_dim=key_dim,
                value_dim=value_dim,
                root=tree,
                selection=selection,
            )
            variant_debug.append(
                {
                    "name": variant["name"],
                    "retained_blocks": selection.retained_blocks,
                    "candidate_blocks": selection.candidate_blocks,
                    "retained_tokens": selection.retained_tokens,
                    "compression_ratio": storage.compression_ratio,
                    "shifts": json.loads(shift_json),
                }
            )
            for correction, output, correction_rank in (
                ("none", estimate, 0),
                (f"adaptive_rank{rank}", adaptive, rank),
            ):
                error, numerator_sq, denominator_sq = _error_components(target, output)
                for local_head, head_index in enumerate(metadata["head_indices"]):
                    records.append(
                        {
                            "protocol_id": protocol["protocol_id"],
                            "source_protocol_id": source_protocol["protocol_id"],
                            "capture_path": str(path),
                            "prompt_id": metadata["prompt_id"],
                            "split": metadata["prompt_split"],
                            "seed": int(metadata["seed"]),
                            "layer": int(metadata["layer"]),
                            "current_start_frame": int(metadata["current_start_frame"]),
                            "denoising_call_index": int(metadata["denoising_call_index"]),
                            "head_index": int(head_index),
                            "method": variant["name"],
                            "key_domain": key_domain,
                            "shift_scope": shift_scope,
                            "window_schedule": window_schedule,
                            "detail_fraction": fraction,
                            "correction": correction,
                            "rank": correction_rank,
                            "relative_av_l2": float(error[local_head].item()),
                            "numerator_sq": float(numerator_sq[local_head].item()),
                            "denominator_sq": float(denominator_sq[local_head].item()),
                            "dense_reference_parity": float(parity[local_head].item()),
                            "cache_compression_ratio": storage.compression_ratio,
                            "dense_cache_bytes": storage.dense_bytes,
                            "compressed_cache_bytes": storage.compressed_bytes,
                            "exact_bytes": storage.exact_bytes,
                            "coarse_bytes": storage.coarse_bytes,
                            "detail_bytes": storage.detail_bytes,
                            "metadata_bytes": storage.metadata_bytes,
                            "candidate_detail_blocks": selection.candidate_blocks,
                            "retained_detail_blocks": selection.retained_blocks,
                            "retained_detail_tokens": selection.retained_tokens,
                            "shift_payload": shift_json,
                            "oracle_access": (
                                "same_record_dense_AV_defect"
                                if correction_rank
                                else "none_runtime_KV_only"
                            ),
                        }
                    )
    return records, {
        "capture": capture_key(metadata),
        "worst_dense_reference_parity": float(parity.max().item()),
        "variants": variant_debug,
    }


def _scopes(records: list[dict[str, Any]]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    for split in ("calibration", "validation", "test"):
        selected = [record for record in records if record["split"] == split]
        if selected:
            yield split, selected
    held_out = [record for record in records if record["split"] != "calibration"]
    if held_out:
        yield "held_out", held_out
    yield "all", records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope_name, scoped in _scopes(records):
        groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in scoped:
            groups[(record["method"], record["correction"])].append(record)
        for (method, correction), group in sorted(groups.items()):
            rows.append(
                {
                    "scope": scope_name,
                    "method": method,
                    "correction": correction,
                    "records": len(group),
                    "captures": len({record["capture_path"] for record in group}),
                    "aggregate_relative_av_l2": math.sqrt(
                        sum(float(record["numerator_sq"]) for record in group)
                        / max(sum(float(record["denominator_sq"]) for record in group), 1e-24)
                    ),
                    "mean_head_relative_av_l2": sum(
                        float(record["relative_av_l2"]) for record in group
                    )
                    / len(group),
                    "worst_head_relative_av_l2": max(
                        float(record["relative_av_l2"]) for record in group
                    ),
                    "minimum_cache_compression_ratio": min(
                        float(record["cache_compression_ratio"]) for record in group
                    ),
                    "maximum_dense_reference_parity": max(
                        float(record["dense_reference_parity"]) for record in group
                    ),
                }
            )
    return rows


def _find_summary(
    summary: list[dict[str, Any]], scope: str, method: str, correction: str
) -> dict[str, Any]:
    selected = [
        row
        for row in summary
        if row["scope"] == scope
        and row["method"] == method
        and row["correction"] == correction
    ]
    if len(selected) != 1:
        raise ValueError(f"summary row is not unique: {scope}/{method}/{correction}")
    return selected[0]


def decide(
    protocol: dict[str, Any], summary: list[dict[str, Any]], complete: bool
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    primary = str(protocol["methods"]["primary"]["name"])
    baseline = str(protocol["methods"]["baseline"]["name"])
    rank = int(protocol["methods"]["adaptive_residual_rank"])
    correction = f"adaptive_rank{rank}"
    decision_scope = "held_out" if complete else "all"
    primary_direct = _find_summary(summary, decision_scope, primary, "none")
    primary_adaptive = _find_summary(summary, decision_scope, primary, correction)
    baseline_adaptive = _find_summary(summary, decision_scope, baseline, correction)
    relative_improvement = 1.0 - float(
        primary_adaptive["aggregate_relative_av_l2"]
    ) / max(float(baseline_adaptive["aggregate_relative_av_l2"]), 1e-24)
    parity_pass = complete and float(primary_direct["maximum_dense_reference_parity"]) <= float(
        evaluation["dense_reference_parity_gate"]
    )
    compression_pass = float(primary_direct["minimum_cache_compression_ratio"]) >= float(
        evaluation["minimum_cache_compression"]
    )
    direct_pass = float(primary_direct["aggregate_relative_av_l2"]) <= float(
        evaluation["direct_aggregate_gate"]
    ) and float(primary_direct["worst_head_relative_av_l2"]) <= float(
        evaluation["direct_worst_head_gate"]
    )
    adaptive_pass = float(primary_adaptive["aggregate_relative_av_l2"]) <= float(
        evaluation["adaptive_aggregate_gate"]
    ) and float(primary_adaptive["worst_head_relative_av_l2"]) <= float(
        evaluation["adaptive_worst_head_gate"]
    )
    shift_pass = relative_improvement >= float(
        evaluation["minimum_shift_relative_improvement"]
    )
    if not complete:
        classification, action = "invalid", "complete_registered_capture_scope"
    elif not parity_pass:
        classification, action = "invalid", "repair_capture_or_dense_parity_once"
    elif adaptive_pass and compression_pass and direct_pass and shift_pass:
        classification, action = "pass", "register_new_captures_before_lazy_decode"
    elif adaptive_pass and compression_pass:
        classification, action = "boundary", "open_one_content_residual_gate"
    else:
        classification, action = "null", "stop_lifting_predictor_kernel_and_rollout"
    return {
        "protocol_id": protocol["protocol_id"],
        "classification": classification,
        "action": action,
        "complete": complete,
        "decision_scope": decision_scope,
        "primary": primary_direct,
        "primary_adaptive": primary_adaptive,
        "same_budget_identity_adaptive": baseline_adaptive,
        "relative_shift_improvement_over_identity": relative_improvement,
        "guards": {
            "parity_pass": parity_pass,
            "compression_pass": compression_pass,
            "direct_quality_pass": direct_pass,
            "adaptive_capacity_pass": adaptive_pass,
            "circulant_mechanism_pass": shift_pass,
        },
        "supports": [
            "A support statement is added only if the corresponding guard passes."
        ],
        "does_not_support": [
            "No attention, H200 kernel, rollout, or end-to-end speedup claim.",
            "No generalization beyond the registered LongLive cells and development-exposed captures.",
        ],
        "unknown": [
            "Query-guided lazy detail decoding and token-count reduction were not evaluated.",
            "Long-horizon rollout quality was not evaluated.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.query_chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    if args.max_captures is not None and args.max_captures <= 0:
        raise ValueError("max captures must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    if protocol["source_protocol_id"] != source_protocol["protocol_id"]:
        raise ValueError("candidate/source protocol IDs disagree")
    required = required_capture_keys(protocol)
    selected: dict[tuple[str, int, int, int], Path] = {}
    for path in sorted(args.capture_dir.glob("**/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            key = capture_key(metadata)
            if key in required:
                if key in selected:
                    raise ValueError(f"duplicate required capture: {key}")
                selected[key] = path
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"missing {len(missing)} required captures; first={missing[0]}")
    selected_items = sorted(selected.items())
    if args.max_captures is not None:
        if not args.allow_partial:
            raise ValueError("max captures requires --allow-partial")
        selected_items = selected_items[: args.max_captures]
    complete = len(selected_items) == len(required)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    capture_hashes: list[dict[str, Any]] = []
    for index, (identity, path) in enumerate(selected_items, start=1):
        print(f"[evaluate {index}/{len(selected_items)}] {identity}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        capture_records, capture_debug = evaluate_capture(
            path,
            payload,
            protocol,
            source_protocol,
            device,
            args.query_chunk_size,
        )
        records.extend(capture_records)
        debug.append(capture_debug)
        capture_hashes.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    summary = summarize(records)
    decision = decide(protocol, summary, complete)
    write_csv(args.output_dir / "metrics.csv", records)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "debug.json").write_text(
        json.dumps(debug, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "source_protocol_path": str(args.source_protocol),
        "source_protocol_sha256": sha256_file(args.source_protocol),
        "capture_dir": str(args.capture_dir),
        "required_capture_count": len(required),
        "evaluated_capture_count": len(selected_items),
        "complete": complete,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "capture_files": capture_hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "SUCCESS.json").write_text(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "classification": decision["classification"],
                "artifacts": [
                    "metrics.csv",
                    "summary.csv",
                    "decision.json",
                    "debug.json",
                    "manifest.json",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
