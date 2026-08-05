"""Forward-marginal detail oracle for causal Butterfly-lifting memory."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import torch

from ar_video_butterfly_lifting_core import (
    LiftingNode,
    build_lifting_tree,
    canonicalize_rope_keys,
    detail_selection_from_indices,
    enumerate_detail_blocks,
    estimate_storage,
    middle_frame_indices,
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
from probe_ar_video_lifting_detail_oracle import _replay, _selection_indices


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


def _adaptive_output(
    target: torch.Tensor, estimate: torch.Tensor, rank: int
) -> tuple[torch.Tensor, float]:
    corrected = estimate + adaptive_rank_projection(target - estimate, rank)
    _, numerator_sq, _ = _error_components(target, corrected)
    return corrected, float(numerator_sq.sum().item())


def _roundtrip_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(dtype).float()


def roundtrip_lifting_payload(
    node: LiftingNode, dtype: torch.dtype, *, is_root: bool = True
) -> LiftingNode:
    """Clone only stored lifting payload after a dtype round trip."""

    left = (
        roundtrip_lifting_payload(node.left, dtype, is_root=False)
        if node.left is not None
        else None
    )
    right = (
        roundtrip_lifting_payload(node.right, dtype, is_root=False)
        if node.right is not None
        else None
    )
    return replace(
        node,
        coarse_key=(
            _roundtrip_tensor(node.coarse_key, dtype)
            if is_root
            else node.coarse_key
        ),
        coarse_value=(
            _roundtrip_tensor(node.coarse_value, dtype)
            if is_root
            else node.coarse_value
        ),
        left=left,
        right=right,
        detail_key=(
            _roundtrip_tensor(node.detail_key, dtype)
            if node.detail_key is not None
            else None
        ),
        detail_value=(
            _roundtrip_tensor(node.detail_value, dtype)
            if node.detail_value is not None
            else None
        ),
    )


def _evaluate_indices(
    indices: list[int],
    tree: LiftingNode,
    tile_size: int,
    query: torch.Tensor,
    original_key: torch.Tensor,
    value: torch.Tensor,
    canonical_key: torch.Tensor,
    middle_indices: tuple[int, ...],
    exact_indices: tuple[int, ...],
    frame_ids: list[int],
    height: int,
    width: int,
    rope_freqs: torch.Tensor,
    query_chunk_size: int,
    target: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, float]:
    selection = detail_selection_from_indices(tree, tile_size, indices)
    estimate = _replay(
        query,
        original_key,
        value,
        canonical_key,
        tree,
        selection,
        middle_indices,
        exact_indices,
        frame_ids,
        height,
        width,
        rope_freqs,
        query_chunk_size,
    )
    _, sse = _adaptive_output(target, estimate, rank)
    return estimate, sse


def forward_marginal_search(
    *,
    budget: int,
    candidate_count: int,
    evaluate,
    denominator_sq: float,
) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    """Select each block by its marginal error with the current joint support."""

    if budget <= 0 or budget > candidate_count:
        raise ValueError("invalid forward-marginal budget")
    selected: list[int] = []
    singleton_scores: list[float] = []
    trajectory: list[dict[str, Any]] = []
    remaining = set(range(candidate_count))
    for step in range(budget):
        best_index = -1
        best_sse = float("inf")
        step_scores: list[tuple[int, float]] = []
        for candidate in sorted(remaining):
            score = float(evaluate(selected + [candidate]))
            step_scores.append((candidate, score))
            if score < best_sse:
                best_index = candidate
                best_sse = score
        if step == 0:
            singleton_scores = [0.0] * candidate_count
            for candidate, score in step_scores:
                singleton_scores[candidate] = score
        selected.append(best_index)
        remaining.remove(best_index)
        relative_error = (best_sse / max(denominator_sq, 1e-24)) ** 0.5
        trajectory.append(
            {
                "step": step + 1,
                "selected_index": best_index,
                "selected_indices": list(selected),
                "adaptive_sse": best_sse,
                "aggregate_relative_av_l2": relative_error,
                "evaluated_candidates": len(step_scores),
            }
        )
        print(
            f"  marginal {step + 1}/{budget}: block={best_index} "
            f"aggregate={100.0 * relative_error:.4f}%",
            flush=True,
        )
    return selected, singleton_scores, trajectory


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
    canonical_key = canonicalize_rope_keys(
        key.float(), metadata["key_frame_ids"], height, width, rope_freqs
    )
    transform = protocol["transform"]
    candidates = [
        tuple(int(value) for value in pair)
        for pair in transform["shift_candidates"]
    ]
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
    blocks = enumerate_detail_blocks(tree, tile_size)
    budget = int(scope["retained_detail_blocks"])
    rank = int(protocol["methods"]["adaptive_residual_rank"])
    energy = select_detail_tiles(
        tree,
        tile_size,
        budget / len(blocks),
        float(transform["key_weight"]),
        float(transform["value_weight"]),
    )
    if energy.retained_blocks != budget:
        raise ValueError("energy selector did not match the frozen block budget")
    energy_indices = _selection_indices(tree, tile_size, energy)

    _, _, denominator_sq = _error_components(target, torch.zeros_like(target))
    total_denominator_sq = float(denominator_sq.sum().item())

    def score(indices: list[int]) -> float:
        _, sse = _evaluate_indices(
            indices,
            tree,
            tile_size,
            query,
            key,
            value,
            canonical_key,
            middle_indices,
            exact_indices,
            metadata["key_frame_ids"],
            height,
            width,
            rope_freqs,
            query_chunk_size,
            target,
            rank,
        )
        return sse

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    search_start = time.perf_counter()
    marginal_indices, singleton_scores, trajectory = forward_marginal_search(
        budget=budget,
        candidate_count=len(blocks),
        evaluate=score,
        denominator_sq=total_denominator_sq,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    search_seconds = time.perf_counter() - search_start
    singleton_indices = sorted(
        range(len(blocks)), key=lambda index: singleton_scores[index]
    )[:budget]

    methods = (
        ("kv_detail_energy", energy_indices),
        ("adaptive_tail_singleton_selector", singleton_indices),
        ("adaptive_tail_forward_marginal_greedy", marginal_indices),
    )
    records: list[dict[str, Any]] = []
    for method, indices in methods:
        selection = detail_selection_from_indices(tree, tile_size, indices)
        storage = estimate_storage(
            frames,
            len(exact_indices),
            spatial,
            heads,
            key_dim,
            value_dim,
            tree,
            selection,
            element_bytes=2,
            padded_detail_tile_size=(
                tile_size if bool(scope["pad_partial_detail_tiles"]) else None
            ),
        )
        for coefficient_dtype, replay_tree in (
            ("fp32_coeff", tree),
            ("bf16_coeff", roundtrip_lifting_payload(tree, torch.bfloat16)),
        ):
            replay_selection = detail_selection_from_indices(
                replay_tree, tile_size, indices
            )
            estimate = _replay(
                query,
                key,
                value,
                canonical_key,
                replay_tree,
                replay_selection,
                middle_indices,
                exact_indices,
                metadata["key_frame_ids"],
                height,
                width,
                rope_freqs,
                query_chunk_size,
            )
            adaptive, _ = _adaptive_output(target, estimate, rank)
            for correction, output, correction_rank in (
                (f"none_{coefficient_dtype}", estimate, 0),
                (f"adaptive_rank{rank}_{coefficient_dtype}", adaptive, rank),
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
                            "method": method,
                            "correction": correction,
                            "rank": correction_rank,
                            "relative_av_l2": float(error[local_head].item()),
                            "numerator_sq": float(numerator_sq[local_head].item()),
                            "denominator_sq": float(denominator_sq[local_head].item()),
                            "dense_reference_parity": float(parity[local_head].item()),
                            "cache_compression_ratio": storage.compression_ratio,
                            "logical_bf16_cache_compression_ratio": storage.compression_ratio,
                            "dense_cache_bytes": storage.dense_bytes,
                            "compressed_cache_bytes": storage.compressed_bytes,
                            "candidate_detail_blocks": len(blocks),
                            "retained_detail_blocks": len(indices),
                            "retained_detail_tokens": selection.retained_tokens,
                            "padded_detail_tokens": len(indices) * tile_size,
                            "search_seconds_per_capture": (
                                search_seconds
                                if method == "adaptive_tail_forward_marginal_greedy"
                                else 0.0
                            ),
                            "selected_indices": json.dumps(indices),
                            "oracle_access": "same_record_dense_AV_defect",
                        }
                    )

    def jaccard(left: list[int], right: list[int]) -> float:
        return len(set(left) & set(right)) / max(len(set(left) | set(right)), 1)

    return records, {
        "capture": capture_key(metadata),
        "candidate_blocks": len(blocks),
        "budget": budget,
        "energy_indices": energy_indices,
        "singleton_indices": singleton_indices,
        "forward_marginal_indices": marginal_indices,
        "energy_singleton_jaccard": jaccard(energy_indices, singleton_indices),
        "singleton_marginal_jaccard": jaccard(singleton_indices, marginal_indices),
        "energy_marginal_jaccard": jaccard(energy_indices, marginal_indices),
        "forward_marginal_trajectory": trajectory,
        "search_seconds": search_seconds,
        "worst_dense_reference_parity": float(parity.max().item()),
    }


def _find(
    summary: list[dict[str, Any]], scope: str, method: str, correction: str
) -> dict[str, Any]:
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


def decide(
    protocol: dict[str, Any], summary: list[dict[str, Any]], complete: bool
) -> dict[str, Any]:
    decision_scope = "held_out" if complete else "all"
    rank = int(protocol["methods"]["adaptive_residual_rank"])
    correction = f"adaptive_rank{rank}_bf16_coeff"
    primary = _find(
        summary, decision_scope, protocol["methods"]["primary"], correction
    )
    baseline = _find(
        summary, decision_scope, protocol["methods"]["baseline"], correction
    )
    energy = _find(
        summary, decision_scope, protocol["methods"]["energy_control"], correction
    )
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
        evaluation["minimum_logical_bf16_cache_compression"]
    )
    mechanism_pass = improvement >= float(
        evaluation["minimum_relative_improvement_over_singleton"]
    )
    if not complete:
        classification, action = "invalid", "complete_registered_capture_scope"
    elif not parity_pass:
        classification, action = "invalid", "repair_dense_reference_parity_once"
    elif quality_pass and compression_pass and mechanism_pass:
        classification, action = "pass", "register_new_captures_before_router_training"
    elif quality_pass and compression_pass:
        classification, action = "boundary", "do_not_train_without_joint_search_gain"
    else:
        classification, action = "null", "stop_router_and_kernel_for_current_transform"
    return {
        "protocol_id": protocol["protocol_id"],
        "classification": classification,
        "action": action,
        "complete": complete,
        "decision_scope": decision_scope,
        "primary": primary,
        "singleton_baseline": baseline,
        "energy_control": energy,
        "relative_improvement_over_singleton": improvement,
        "guards": {
            "parity_pass": parity_pass,
            "quality_pass": quality_pass,
            "compression_pass": compression_pass,
            "mechanism_pass": mechanism_pass,
        },
        "does_not_support": [
            "Forward greedy is not a globally optimal combinatorial oracle.",
            "The selector and adaptive correction read same-record dense AV.",
            "No deployability, attention speedup, rollout quality, or prompt generalization claim is permitted.",
        ],
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    if protocol["source_protocol_id"] != source_protocol["protocol_id"]:
        raise ValueError("candidate/source protocol mismatch")
    required = required_capture_keys(protocol)
    selected: dict[tuple[str, int, int, int], Path] = {}
    for path in sorted(args.capture_dir.glob("**/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and capture_key(metadata) in required:
            identity = capture_key(metadata)
            if identity in selected:
                raise ValueError(f"duplicate required capture: {identity}")
            selected[identity] = path
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
    records: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
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
        hashes.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
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
