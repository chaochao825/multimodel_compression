"""Dense-defect singleton oracle for Butterfly-lifting detail blocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from ar_video_butterfly_lifting_core import (
    build_lifting_tree,
    canonicalize_rope_keys,
    detail_selection_from_indices,
    enumerate_detail_blocks,
    estimate_storage,
    middle_frame_indices,
    reconstruct_lifting_tree,
    restore_rope_keys,
    select_detail_tiles,
)
from ar_video_residual_memory_core import adaptive_rank_projection, dense_attention
from probe_ar_video_butterfly_lifting import (
    _error_components,
    _grid,
    _validate_capture,
    capture_key,
    required_capture_keys,
    sha256_file,
    summarize,
    write_csv,
)


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


def _selection_indices(root, tile_size: int, selection) -> list[int]:
    indices = []
    for index, (node, start, end) in enumerate(enumerate_detail_blocks(root, tile_size)):
        if bool(selection.masks[id(node)][start:end].all()):
            indices.append(index)
    return indices


def _replay(
    query: torch.Tensor,
    original_key: torch.Tensor,
    value: torch.Tensor,
    canonical_key: torch.Tensor,
    tree,
    selection,
    middle_indices: tuple[int, ...],
    exact_indices: tuple[int, ...],
    frame_ids: list[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
    query_chunk_size: int,
) -> torch.Tensor:
    key_maps, value_maps = reconstruct_lifting_tree(tree, selection, height, width)
    reconstructed_key = canonical_key.clone()
    reconstructed_value = value.float().clone()
    for frame in middle_indices:
        reconstructed_key[frame] = key_maps[frame]
        reconstructed_value[frame] = value_maps[frame]
    post_key = restore_rope_keys(
        reconstructed_key,
        frame_ids,
        height,
        width,
        rope_freqs,
        original_key.dtype,
    )
    exact = torch.tensor(exact_indices, dtype=torch.long, device=query.device)
    post_key.index_copy_(0, exact, original_key.index_select(0, exact))
    reconstructed_value.index_copy_(0, exact, value.float().index_select(0, exact))
    return dense_attention(
        query,
        post_key.reshape(-1, post_key.shape[2], post_key.shape[3]),
        reconstructed_value.to(value.dtype).reshape(
            -1, reconstructed_value.shape[2], reconstructed_value.shape[3]
        ),
        query_chunk_size,
    )


def _adaptive_sse(target: torch.Tensor, estimate: torch.Tensor, rank: int) -> float:
    corrected = estimate + adaptive_rank_projection(target - estimate, rank)
    _, numerator_sq, _ = _error_components(target, corrected)
    return float(numerator_sq.sum().item())


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
    heads, dim = key_flat.shape[1:]
    key = key_flat.reshape(frames, spatial, heads, dim)
    value = value_flat.reshape(frames, spatial, heads, dim)
    dense_math = dense_attention(query, key_flat, value_flat, query_chunk_size)
    parity, _, _ = _error_components(target, dense_math)
    scope = protocol["scope"]
    exact_indices, middle_indices = middle_frame_indices(
        frames,
        int(scope["exact_sink_frames"]),
        int(scope["exact_recent_frames"]),
    )
    canonical_key = canonicalize_rope_keys(
        key.float(), metadata["key_frame_ids"], height, width, rope_freqs
    )
    transform = protocol["transform"]
    candidates = [tuple(int(value) for value in pair) for pair in transform["shift_candidates"]]
    middle = torch.tensor(middle_indices, dtype=torch.long, device=device)
    tree = build_lifting_tree(
        canonical_key.index_select(0, middle),
        value.float().index_select(0, middle),
        middle_indices,
        candidates,
        height,
        width,
        str(transform["shift_scope"]),
        float(transform["key_weight"]),
        float(transform["value_weight"]),
    )
    tile_size = int(scope["detail_tile_size"])
    fraction = float(scope["detail_fraction"])
    blocks = enumerate_detail_blocks(tree, tile_size)
    budget = int(math.ceil(len(blocks) * fraction))
    rank = int(protocol["methods"]["adaptive_residual_rank"])
    energy = select_detail_tiles(
        tree,
        tile_size,
        fraction,
        float(transform["key_weight"]),
        float(transform["value_weight"]),
    )
    energy_indices = _selection_indices(tree, tile_size, energy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    oracle_start = time.perf_counter()
    singleton_scores = []
    for index in range(len(blocks)):
        selection = detail_selection_from_indices(tree, tile_size, [index])
        estimate = _replay(
            query,
            key,
            value,
            canonical_key,
            tree,
            selection,
            middle_indices,
            exact_indices,
            metadata["key_frame_ids"],
            height,
            width,
            rope_freqs,
            query_chunk_size,
        )
        singleton_scores.append(_adaptive_sse(target, estimate, rank))
        if (index + 1) % 25 == 0:
            print(f"  singleton {index + 1}/{len(blocks)}", flush=True)
    oracle_indices = sorted(
        range(len(blocks)), key=lambda index: singleton_scores[index]
    )[:budget]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    oracle_seconds = time.perf_counter() - oracle_start

    records: list[dict[str, Any]] = []
    selector_payload = {}
    for selector, indices in (
        ("kv_detail_energy", energy_indices),
        ("adaptive_tail_singleton_oracle", oracle_indices),
    ):
        selection = detail_selection_from_indices(tree, tile_size, indices)
        estimate = _replay(
            query,
            key,
            value,
            canonical_key,
            tree,
            selection,
            middle_indices,
            exact_indices,
            metadata["key_frame_ids"],
            height,
            width,
            rope_freqs,
            query_chunk_size,
        )
        adaptive = estimate + adaptive_rank_projection(target - estimate, rank)
        storage = estimate_storage(
            frames,
            len(exact_indices),
            spatial,
            heads,
            dim,
            dim,
            tree,
            selection,
        )
        selector_payload[selector] = indices
        for correction, output, correction_rank in (
            ("none", estimate, 0),
            (f"adaptive_rank{rank}", adaptive, rank),
        ):
            error, numerator_sq, denominator_sq = _error_components(target, output)
            for local_head, head_index in enumerate(metadata["head_indices"]):
                records.append(
                    {
                        "protocol_id": protocol["protocol_id"],
                        "prompt_id": metadata["prompt_id"],
                        "split": metadata["prompt_split"],
                        "seed": int(metadata["seed"]),
                        "layer": int(metadata["layer"]),
                        "current_start_frame": int(metadata["current_start_frame"]),
                        "denoising_call_index": int(metadata["denoising_call_index"]),
                        "head_index": int(head_index),
                        "capture_path": str(path),
                        "method": selector,
                        "correction": correction,
                        "rank": correction_rank,
                        "relative_av_l2": float(error[local_head].item()),
                        "numerator_sq": float(numerator_sq[local_head].item()),
                        "denominator_sq": float(denominator_sq[local_head].item()),
                        "dense_reference_parity": float(parity[local_head].item()),
                        "cache_compression_ratio": storage.compression_ratio,
                        "dense_cache_bytes": storage.dense_bytes,
                        "compressed_cache_bytes": storage.compressed_bytes,
                        "candidate_detail_blocks": len(blocks),
                        "retained_detail_blocks": len(indices),
                        "selector_seconds_per_capture": (
                            oracle_seconds if selector.endswith("oracle") else 0.0
                        ),
                        "selected_indices": json.dumps(indices),
                        "oracle_access": (
                            "same_record_dense_AV_defect"
                            if selector.endswith("oracle") or correction_rank
                            else "none_runtime_KV_only"
                        ),
                    }
                )
    overlap = len(set(energy_indices) & set(oracle_indices)) / max(
        len(set(energy_indices) | set(oracle_indices)), 1
    )
    return records, {
        "capture": capture_key(metadata),
        "candidate_blocks": len(blocks),
        "budget": budget,
        "energy_indices": energy_indices,
        "oracle_indices": oracle_indices,
        "energy_oracle_jaccard": overlap,
        "oracle_seconds": oracle_seconds,
        "worst_dense_reference_parity": float(parity.max().item()),
    }


def _find(summary, scope: str, method: str, correction: str):
    rows = [
        row
        for row in summary
        if row["scope"] == scope
        and row["method"] == method
        and row["correction"] == correction
    ]
    if len(rows) != 1:
        raise ValueError(f"summary row is not unique: {scope}/{method}/{correction}")
    return rows[0]


def decide(protocol, summary, complete: bool):
    decision_scope = "held_out" if complete else "all"
    rank = int(protocol["methods"]["adaptive_residual_rank"])
    correction = f"adaptive_rank{rank}"
    primary = _find(summary, decision_scope, protocol["methods"]["primary"], correction)
    baseline = _find(summary, decision_scope, protocol["methods"]["baseline"], correction)
    evaluation = protocol["evaluation"]
    improvement = 1.0 - float(primary["aggregate_relative_av_l2"]) / max(
        float(baseline["aggregate_relative_av_l2"]), 1e-24
    )
    parity_pass = complete and float(primary["maximum_dense_reference_parity"]) <= float(
        evaluation["dense_reference_parity_gate"]
    )
    quality_pass = float(primary["aggregate_relative_av_l2"]) <= float(
        evaluation["adaptive_aggregate_gate"]
    ) and float(primary["worst_head_relative_av_l2"]) <= float(
        evaluation["adaptive_worst_head_gate"]
    )
    compression_pass = float(primary["minimum_cache_compression_ratio"]) >= float(
        evaluation["minimum_cache_compression"]
    )
    mechanism_pass = improvement >= float(
        evaluation["minimum_relative_improvement_over_energy"]
    )
    if not complete:
        classification, action = "invalid", "complete_registered_capture_scope"
    elif not parity_pass:
        classification, action = "invalid", "repair_dense_reference_parity_once"
    elif quality_pass and compression_pass and mechanism_pass:
        classification, action = "pass", "open_runtime_proxy_gate_on_new_captures"
    elif quality_pass and compression_pass:
        classification, action = "boundary", "do_not_train_selector_without_mechanism_gain"
    else:
        classification, action = "null", "stop_butterfly_lifting_and_detail_routing"
    return {
        "protocol_id": protocol["protocol_id"],
        "classification": classification,
        "action": action,
        "complete": complete,
        "decision_scope": decision_scope,
        "primary": primary,
        "baseline": baseline,
        "relative_improvement_over_energy": improvement,
        "guards": {
            "parity_pass": parity_pass,
            "quality_pass": quality_pass,
            "compression_pass": compression_pass,
            "mechanism_pass": mechanism_pass,
        },
        "does_not_support": [
            "The oracle is not a deployable selector.",
            "No attention speedup or rollout-quality claim is permitted."
        ],
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    if protocol["source_protocol_id"] != source_protocol["protocol_id"]:
        raise ValueError("candidate/source protocol mismatch")
    required = required_capture_keys(protocol)
    selected = {}
    for path in sorted(args.capture_dir.glob("**/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and capture_key(metadata) in required:
            key = capture_key(metadata)
            if key in selected:
                raise ValueError(f"duplicate required capture: {key}")
            selected[key] = path
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"missing {len(missing)} captures; first={missing[0]}")
    items = sorted(selected.items())
    if args.max_captures is not None:
        if not args.allow_partial:
            raise ValueError("max captures requires --allow-partial")
        items = items[: args.max_captures]
    complete = len(items) == len(required)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records = []
    debug = []
    hashes = []
    for index, (identity, path) in enumerate(items, start=1):
        print(f"[evaluate {index}/{len(items)}] {identity}", flush=True)
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
        hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
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
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": sha256_file(args.protocol),
                "source_protocol_sha256": sha256_file(args.source_protocol),
                "capture_dir": str(args.capture_dir),
                "required_capture_count": len(required),
                "evaluated_capture_count": len(items),
                "complete": complete,
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "capture_files": hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "SUCCESS.json").write_text(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "classification": decision["classification"],
                "artifacts": ["metrics.csv", "summary.csv", "decision.json", "debug.json", "manifest.json"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
