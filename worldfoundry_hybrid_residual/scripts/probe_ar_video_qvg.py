"""Evaluate official QuantVideoGen representations on captured LongLive K/V."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch

from ar_video_qvg_core import (
    compression_ratio,
    gather_frames_for_qvg,
    logical_rtn_bytes,
    quantized_frame_indices,
    relative_l2_by_head as kv_relative_l2_by_head,
    scatter_qvg_frames,
    tensor_tree_nbytes,
    transform_key_rope,
)
from ar_video_residual_memory_core import dense_attention, relative_l2_by_head
from probe_ar_video_residual_memory import (
    capture_identity,
    expected_identities,
    sha256_file,
    validate_manifests,
    validate_metadata,
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    bits: int
    stages: int
    block_size: int
    exact_policy: str
    primary: bool


@dataclass(frozen=True)
class QVGFunctions:
    compress_kv_cache: Any
    get_quantize_fn: Any
    QuantizeConfig: Any
    triton_prq_dequantize_tensor: Any
    blockwise_intx_quantize_triton: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--qvg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--max-captures", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qvg_functions(root: Path, expected_commit: str) -> QVGFunctions:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise ValueError(
            f"QuantVideoGen commit mismatch: {actual_commit} != {expected_commit}"
        )
    sys.path.insert(0, str(root))
    from quant_videogen.compress import compress_kv_cache, get_quantize_fn
    from quant_videogen.functions import triton_prq_dequantize_tensor
    from quant_videogen.sim.quant.lowbit_quantize import (
        blockwise_intx_quantize_triton,
    )
    from quant_videogen.sim.quant.quantize_config import QuantizeConfig

    return QVGFunctions(
        compress_kv_cache=compress_kv_cache,
        get_quantize_fn=get_quantize_fn,
        QuantizeConfig=QuantizeConfig,
        triton_prq_dequantize_tensor=triton_prq_dequantize_tensor,
        blockwise_intx_quantize_triton=blockwise_intx_quantize_triton,
    )


def method_specs(protocol: dict[str, Any], selected: Iterable[str]) -> list[MethodSpec]:
    requested = set(selected)
    specs = [
        MethodSpec(
            name=str(item["name"]),
            kind=str(item["kind"]),
            bits=int(item["bits"]),
            stages=int(item["stages"]),
            block_size=int(item["block_size"]),
            exact_policy=str(item["exact_policy"]),
            primary=bool(item.get("primary", False)),
        )
        for item in protocol["methods"]
    ]
    known = {item.name for item in specs}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown requested methods: {sorted(unknown)}")
    return [item for item in specs if not requested or item.name in requested]


def stable_record_seed(base_seed: int, metadata: dict[str, Any], method: str) -> int:
    identity = "|".join(map(str, capture_identity(metadata))) + "|" + method
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**31))


def squared_error_by_head(
    reference: torch.Tensor, estimate: torch.Tensor, head_axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate shapes differ")
    reduce_axes = tuple(axis for axis in range(reference.ndim) if axis != head_axis)
    numerator = (reference.float() - estimate.float()).square().sum(dim=reduce_axes)
    denominator = reference.float().square().sum(dim=reduce_axes).clamp_min(1e-24)
    return numerator, denominator


def quantize_reconstruct(
    canonical_key: torch.Tensor,
    value: torch.Tensor,
    spec: MethodSpec,
    protocol: dict[str, Any],
    qvg: QVGFunctions,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    capture = protocol["capture"]
    frames = quantized_frame_indices(
        canonical_key.shape[0],
        spec.exact_policy,
        int(capture["exact_sink_frames"]),
        int(capture["exact_recent_frames"]),
    )
    key_input = gather_frames_for_qvg(canonical_key, frames)
    value_input = gather_frames_for_qvg(value, frames)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if spec.kind == "naive_block_quantization":
        key_reconstructed = qvg.blockwise_intx_quantize_triton(
            key_input, num_bits=spec.bits, block_size=spec.block_size
        )
        value_reconstructed = qvg.blockwise_intx_quantize_triton(
            value_input, num_bits=spec.bits, block_size=spec.block_size
        )
        packed_bytes = logical_rtn_bytes(key_input, spec.bits, spec.block_size)
        packed_bytes += logical_rtn_bytes(value_input, spec.bits, spec.block_size)
    elif spec.kind == "semantic_residual_quantization":
        quant_type = f"triton-nstages-kmeans-int{spec.bits}"
        quant_config = qvg.QuantizeConfig(
            quant_type=quant_type,
            cache_num_k_centroids=int(protocol["quantization"]["centroids_per_head"]),
            cache_num_v_centroids=int(protocol["quantization"]["centroids_per_head"]),
            kmeans_max_iters=int(protocol["quantization"]["kmeans_iterations"]),
            quant_block_size=spec.block_size,
            num_prq_stages=spec.stages,
        )
        quantize_fn = qvg.get_quantize_fn(quant_type, quant_config)
        key_state, value_state = qvg.compress_kv_cache(
            key_input, value_input, quant_type, quant_config, quantize_fn
        )
        key_reconstructed = qvg.triton_prq_dequantize_tensor(
            key_state, spec.block_size, spec.bits, output_dtype=canonical_key.dtype
        )
        value_reconstructed = qvg.triton_prq_dequantize_tensor(
            value_state, spec.block_size, spec.bits, output_dtype=value.dtype
        )
        packed_bytes = tensor_tree_nbytes(key_state) + tensor_tree_nbytes(value_state)
    else:
        raise ValueError(f"unsupported method kind: {spec.kind}")

    canonical_reconstructed = scatter_qvg_frames(
        canonical_key, key_reconstructed, frames
    )
    value_reconstructed_full = scatter_qvg_frames(value, value_reconstructed, frames)
    exact_frames = canonical_key.shape[0] - len(frames)
    if exact_frames:
        bytes_per_frame = (
            canonical_key[0].numel() * canonical_key.element_size()
            + value[0].numel() * value.element_size()
        )
        packed_bytes += int(exact_frames * bytes_per_frame)
    return canonical_reconstructed, value_reconstructed_full, packed_bytes, len(frames)


def evaluate_method(
    payload: dict[str, Any],
    spec: MethodSpec,
    protocol: dict[str, Any],
    qvg: QVGFunctions,
    device: torch.device,
    query_chunk_size: int,
    parity_by_head: torch.Tensor,
) -> list[dict[str, Any]]:
    metadata = payload["metadata"]
    query = payload["query"].to(device)
    target = payload["dense_output"].to(device)
    key_flat = payload["key"].to(device)
    value_flat = payload["value"].to(device)
    capture = protocol["capture"]
    key = key_flat.reshape(
        int(capture["key_frames"]),
        int(capture["height"]) * int(capture["width"]),
        key_flat.shape[1],
        key_flat.shape[2],
    )
    value = value_flat.reshape_as(key)
    rope_freqs = payload["rope_freqs"].to(device)
    canonical_key = transform_key_rope(
        key,
        metadata["key_frame_ids"],
        int(capture["height"]),
        int(capture["width"]),
        rope_freqs,
        inverse=True,
    )
    seed = stable_record_seed(
        int(protocol["quantization"]["deterministic_seed"]), metadata, spec.name
    )
    canonical_reconstructed, value_reconstructed, packed_bytes, quantized_frames = (
        quantize_reconstruct(canonical_key, value, spec, protocol, qvg, seed)
    )
    key_reconstructed = transform_key_rope(
        canonical_reconstructed,
        metadata["key_frame_ids"],
        int(capture["height"]),
        int(capture["width"]),
        rope_freqs,
        inverse=False,
    )
    estimate = dense_attention(
        query,
        key_reconstructed.reshape_as(key_flat),
        value_reconstructed.reshape_as(value_flat),
        query_chunk_size,
    )
    av_error = relative_l2_by_head(target, estimate)
    key_error = kv_relative_l2_by_head(key, key_reconstructed)
    value_error = kv_relative_l2_by_head(value, value_reconstructed)
    av_num, av_den = squared_error_by_head(target, estimate, 1)
    key_num, key_den = squared_error_by_head(key, key_reconstructed, 2)
    value_num, value_den = squared_error_by_head(value, value_reconstructed, 2)
    ratio = compression_ratio([key, value], packed_bytes)
    rows = []
    for local_head, head_index in enumerate(metadata["head_indices"]):
        rows.append(
            {
                "protocol_id": protocol["protocol_id"],
                "prompt_id": metadata["prompt_id"],
                "split": metadata["prompt_split"],
                "seed": int(metadata["seed"]),
                "layer": int(metadata["layer"]),
                "current_start_frame": int(metadata["current_start_frame"]),
                "denoising_call_index": int(metadata["denoising_call_index"]),
                "denoising_timestep": int(metadata["denoising_timestep"]),
                "head_index": int(head_index),
                "method": spec.name,
                "kind": spec.kind,
                "bits": spec.bits,
                "stages": spec.stages,
                "block_size": spec.block_size,
                "exact_policy": spec.exact_policy,
                "quantized_frames": quantized_frames,
                "record_seed": seed,
                "relative_av_l2": float(av_error[local_head]),
                "av_numerator_sq": float(av_num[local_head]),
                "av_denominator_sq": float(av_den[local_head]),
                "relative_k_l2": float(key_error[local_head]),
                "k_numerator_sq": float(key_num[local_head]),
                "k_denominator_sq": float(key_den[local_head]),
                "relative_v_l2": float(value_error[local_head]),
                "v_numerator_sq": float(value_num[local_head]),
                "v_denominator_sq": float(value_den[local_head]),
                "dense_reference_parity": float(parity_by_head[local_head]),
                "packed_kv_bytes": packed_bytes,
                "bf16_kv_bytes": int(
                    key.numel() * key.element_size()
                    + value.numel() * value.element_size()
                ),
                "cache_compression_ratio": ratio,
                "measured_attention_speedup": "",
            }
        )
    return rows


def scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all":
        return rows
    if scope == "held_out":
        return [row for row in rows if row["split"] in {"validation", "test"}]
    return [row for row in rows if row["split"] == scope]


def aggregate_error(rows: list[dict[str, Any]], prefix: str) -> float:
    numerator = sum(float(row[f"{prefix}_numerator_sq"]) for row in rows)
    denominator = sum(float(row[f"{prefix}_denominator_sq"]) for row in rows)
    return math.sqrt(numerator / denominator)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    methods = sorted({str(row["method"]) for row in rows})
    for scope in ("calibration", "validation", "test", "held_out", "all"):
        scoped = scope_rows(rows, scope)
        for method in methods:
            group = [row for row in scoped if row["method"] == method]
            if not group:
                continue
            av_values = [float(row["relative_av_l2"]) for row in group]
            unique_records = {
                (
                    row["prompt_id"],
                    row["layer"],
                    row["current_start_frame"],
                    row["denoising_call_index"],
                ): row
                for row in group
            }
            compression = [
                float(row["cache_compression_ratio"])
                for row in unique_records.values()
            ]
            output.append(
                {
                    "scope": scope,
                    "method": method,
                    "aggregate_relative_av_l2": aggregate_error(group, "av"),
                    "mean_head_relative_av_l2": sum(av_values) / len(av_values),
                    "p95_head_relative_av_l2": percentile(av_values, 0.95),
                    "worst_head_relative_av_l2": max(av_values),
                    "aggregate_relative_k_l2": aggregate_error(group, "k"),
                    "aggregate_relative_v_l2": aggregate_error(group, "v"),
                    "minimum_cache_compression_ratio": min(compression),
                    "mean_cache_compression_ratio": sum(compression) / len(compression),
                    "capture_records": len(unique_records),
                    "head_records": len(group),
                    "measured_attention_speedup": None,
                }
            )
    return output


def decide_gate(
    protocol: dict[str, Any],
    summaries: list[dict[str, Any]],
    capture_validation: dict[str, Any],
) -> dict[str, Any]:
    primary = next(item for item in protocol["methods"] if item.get("primary"))
    match = [
        item
        for item in summaries
        if item["scope"] == "held_out" and item["method"] == primary["name"]
    ]
    decision: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "primary_method": primary["name"],
        "primary_held_out": match[0] if match else None,
        "measured_attention_speedup_claim_allowed": False,
    }
    evaluation = protocol["evaluation"]
    parity_pass = (
        capture_validation["aggregate_dense_reference_parity"]
        <= float(evaluation["dense_reference_parity_gate"])
        and capture_validation["worst_dense_reference_parity"]
        <= float(evaluation["dense_reference_parity_gate"])
    )
    decision["capture_parity_pass"] = parity_pass
    if not capture_validation["complete"] or not parity_pass or not match:
        decision["classification"] = "invalid_or_incomplete"
        decision["action"] = "repair_capture_or_complete_primary_evaluation"
        return decision
    result = match[0]
    quality_pass = (
        result["aggregate_relative_av_l2"] <= float(evaluation["aggregate_av_gate"])
        and result["worst_head_relative_av_l2"]
        <= float(evaluation["worst_head_av_gate"])
    )
    compression_pass = result["minimum_cache_compression_ratio"] >= float(
        evaluation["minimum_primary_cache_compression"]
    )
    decision["quality_pass"] = quality_pass
    decision["compression_pass"] = compression_pass
    if not quality_pass:
        decision["classification"] = "null"
        decision["action"] = "stop_query_selective_residual_decoding_for_qvg_representation"
    elif not compression_pass:
        decision["classification"] = "boundary"
        decision["action"] = "audit_metadata_overhead_and_revise_memory_representation"
    else:
        decision["classification"] = "cache_representation_pass"
        decision["action"] = "open_separate_query_adaptive_progressive_decode_gate"
    return decision


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no metrics were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.query_chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    if args.max_captures is not None and args.max_captures <= 0:
        raise ValueError("max captures must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        args.output_dir / "metrics.csv",
        args.output_dir / "summary.json",
        args.output_dir / "gate_decision.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing result artifacts: "
            + ", ".join(map(str, existing))
        )

    protocol = load_json(args.protocol)
    source_protocol = load_json(args.source_protocol)
    qvg = load_qvg_functions(
        args.qvg_root, str(protocol["reference"]["repository_commit"])
    )
    manifests, paths = validate_manifests(
        args.capture_dir, source_protocol, sha256_file(args.source_protocol)
    )
    expected = expected_identities(source_protocol)
    if len(expected) != int(protocol["capture"]["expected_records"]):
        raise ValueError("QVG and source protocols disagree on capture count")
    if args.max_captures is not None:
        paths = paths[: args.max_captures]
    specs = method_specs(protocol, args.method)
    if not specs:
        raise ValueError("no methods selected")
    if args.method and not args.allow_partial:
        raise ValueError("method subsets require --allow-partial")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested but CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)

    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int, int], Path] = {}
    parity_values: list[float] = []
    for index, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_metadata(payload["metadata"], payload, source_protocol)
        identity = capture_identity(payload["metadata"])
        if identity in seen:
            raise ValueError(f"duplicate capture identity in {seen[identity]} and {path}")
        if identity not in expected:
            raise ValueError(f"capture identity outside source protocol: {identity}")
        seen[identity] = path
        print(f"[evaluate {index}/{len(paths)}] {identity}", flush=True)
        query = payload["query"].to(device)
        key = payload["key"].to(device)
        value = payload["value"].to(device)
        target = payload["dense_output"].to(device)
        dense_math = dense_attention(query, key, value, args.query_chunk_size)
        parity = relative_l2_by_head(target, dense_math)
        parity_values.extend(float(item) for item in parity)
        del query, key, value, target, dense_math
        for spec in specs:
            print(f"  [method] {spec.name}", flush=True)
            rows.extend(
                evaluate_method(
                    payload,
                    spec,
                    protocol,
                    qvg,
                    device,
                    args.query_chunk_size,
                    parity,
                )
            )
        del payload

    missing = sorted(expected - set(seen))
    complete = not missing and len(specs) == len(protocol["methods"])
    if not complete and not args.allow_partial:
        raise ValueError(
            f"incomplete protocol: missing captures={len(missing)}, methods={len(specs)}"
        )
    summaries = summarize(rows)
    capture_validation = {
        "expected_capture_count": len(expected),
        "found_capture_count": len(seen),
        "complete": complete,
        "missing_identities": missing,
        "manifest_count": len(manifests),
        "aggregate_dense_reference_parity": math.sqrt(
            sum(value * value for value in parity_values) / len(parity_values)
        ),
        "worst_dense_reference_parity": max(parity_values),
    }
    gate = decide_gate(protocol, summaries, capture_validation)
    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "qvg_repository_commit": protocol["reference"]["repository_commit"],
                "capture_validation": capture_validation,
                "manifests": manifests,
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
