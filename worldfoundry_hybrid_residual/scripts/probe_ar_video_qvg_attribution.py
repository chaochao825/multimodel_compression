"""Calibration-only K/V and temporal error attribution for the LongLive QVG probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
from typing import Any

import torch

from ar_video_qvg_core import (
    gather_frames_for_qvg,
    scatter_qvg_frames,
    tensor_tree_nbytes,
    transform_key_rope,
)
from ar_video_residual_memory_core import dense_attention, relative_l2_by_head
from probe_ar_video_qvg import load_json, load_qvg_functions, stable_record_seed
from probe_ar_video_residual_memory import (
    capture_identity,
    sha256_file,
    validate_manifests,
    validate_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qvg-protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--qvg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    return parser.parse_args()


def centroid_reconstruction(state: dict[str, Any]) -> torch.Tensor:
    """Reconstruct the semantic centroid contribution without the low-bit residual."""

    centroids_list = state["centroids_list"]
    cluster_ids_list = state["cluster_ids_list"]
    if not centroids_list or len(centroids_list) != len(cluster_ids_list):
        raise ValueError("invalid progressive centroid state")
    batch, heads, sequence = cluster_ids_list[0].shape
    dim = centroids_list[0].shape[-1]
    output = torch.zeros(
        batch,
        heads,
        sequence,
        dim,
        dtype=centroids_list[0].dtype,
        device=centroids_list[0].device,
    )
    for centroids, cluster_ids in zip(centroids_list, cluster_ids_list):
        index = cluster_ids.long().unsqueeze(-1).expand(-1, -1, -1, dim)
        output.add_(torch.gather(centroids, dim=2, index=index))
    return output


def top_level_bytes(state: dict[str, Any]) -> dict[str, int]:
    return {name: tensor_tree_nbytes(value) for name, value in state.items()}


def squared_error_by_head(
    reference: torch.Tensor, estimate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = (reference.float() - estimate.float()).square().sum(dim=(0, 2))
    denominator = reference.float().square().sum(dim=(0, 2)).clamp_min(1e-24)
    return numerator, denominator


def frame_statistics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_frames: int,
    query_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head attention mass and a squared-value leverage proxy by frame."""

    key_tokens, heads, dim = key.shape
    if key_tokens % key_frames:
        raise ValueError("key sequence is not frame aligned")
    spatial = key_tokens // key_frames
    mass = torch.zeros(heads, key_frames, dtype=torch.float64, device=query.device)
    leverage = torch.zeros_like(mass)
    value_energy = value.float().square().sum(-1).transpose(0, 1)
    for start in range(0, query.shape[0], query_chunk_size):
        q_chunk = query[start : start + query_chunk_size].float()
        logits = torch.einsum("qhd,khd->hqk", q_chunk, key.float()) / math.sqrt(dim)
        probabilities = torch.softmax(logits, dim=-1)
        shape = (heads, q_chunk.shape[0], key_frames, spatial)
        mass += probabilities.reshape(shape).sum(dim=(1, 3)).double()
        weighted = probabilities.square() * value_energy[:, None, :]
        leverage += weighted.reshape(shape).sum(dim=(1, 3)).double()
    mass /= query.shape[0]
    leverage /= leverage.sum(dim=1, keepdim=True).clamp_min(1e-24)
    return mass.cpu(), leverage.cpu()


def select_paths(
    paths: list[Path], source_protocol: dict[str, Any], selection: dict[str, Any]
) -> list[tuple[Path, dict[str, Any]]]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        validate_metadata(metadata, payload, source_protocol)
        matches = (
            metadata["prompt_id"] == selection["prompt_id"]
            and int(metadata["layer"]) in selection["layers"]
            and int(metadata["current_start_frame"])
            == int(selection["current_start_frame"])
            and int(metadata["denoising_call_index"])
            == int(selection["denoising_call_index"])
        )
        if matches:
            if metadata["prompt_split"] != "calibration":
                raise ValueError("attribution selection escaped the calibration split")
            selected.append((path, metadata))
    selected.sort(key=lambda item: int(item[1]["layer"]))
    if len(selected) != int(selection["expected_records"]):
        raise ValueError(
            f"expected {selection['expected_records']} records, found {len(selected)}"
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.query_chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        args.output_dir / "variant_metrics.csv",
        args.output_dir / "frame_statistics.csv",
        args.output_dir / "summary.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")

    protocol = load_json(args.protocol)
    qvg_protocol = load_json(args.qvg_protocol)
    source_protocol = load_json(args.source_protocol)
    expected_qvg_path = str(protocol["source_qvg_protocol"])
    if args.qvg_protocol.as_posix() != expected_qvg_path:
        raise ValueError("attribution protocol references a different QVG protocol path")
    qvg = load_qvg_functions(
        args.qvg_root, str(qvg_protocol["reference"]["repository_commit"])
    )
    manifests, paths = validate_manifests(
        args.capture_dir, source_protocol, sha256_file(args.source_protocol)
    )
    selected = select_paths(paths, source_protocol, protocol["selection"])

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)

    method = protocol["method"]
    quant_type = f"triton-nstages-kmeans-int{int(method['bits'])}"
    config = qvg.QuantizeConfig(
        quant_type=quant_type,
        cache_num_k_centroids=int(method["centroids_per_head"]),
        cache_num_v_centroids=int(method["centroids_per_head"]),
        kmeans_max_iters=int(method["kmeans_iterations"]),
        quant_block_size=int(method["block_size"]),
        num_prq_stages=int(method["stages"]),
        asymmetric=bool(method["asymmetric"]),
    )
    quantize_fn = qvg.get_quantize_fn(quant_type, config)
    variants = list(protocol["variants"])
    variant_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for ordinal, (path, expected_metadata) in enumerate(selected, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if capture_identity(metadata) != capture_identity(expected_metadata):
            raise ValueError("capture changed between validation and evaluation")
        print(f"[attribution {ordinal}/{len(selected)}] {capture_identity(metadata)}")
        query = payload["query"].to(device)
        key_flat = payload["key"].to(device)
        value_flat = payload["value"].to(device)
        target = payload["dense_output"].to(device)
        capture = qvg_protocol["capture"]
        key = key_flat.reshape(
            int(capture["key_frames"]),
            int(capture["height"]) * int(capture["width"]),
            key_flat.shape[1],
            key_flat.shape[2],
        )
        value = value_flat.reshape_as(key)
        canonical_key = transform_key_rope(
            key,
            metadata["key_frame_ids"],
            int(capture["height"]),
            int(capture["width"]),
            payload["rope_freqs"].to(device),
            inverse=True,
        )
        frames = tuple(range(int(capture["key_frames"])))
        key_input = gather_frames_for_qvg(canonical_key, frames)
        value_input = gather_frames_for_qvg(value, frames)
        seed = stable_record_seed(
            int(qvg_protocol["quantization"]["deterministic_seed"]),
            metadata,
            str(method["name"]),
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        key_state, value_state = qvg.compress_kv_cache(
            key_input, value_input, quant_type, config, quantize_fn
        )
        key_full_qvg = qvg.triton_prq_dequantize_tensor(
            key_state,
            int(method["block_size"]),
            int(method["bits"]),
            output_dtype=canonical_key.dtype,
        )
        value_full_qvg = qvg.triton_prq_dequantize_tensor(
            value_state,
            int(method["block_size"]),
            int(method["bits"]),
            output_dtype=value.dtype,
        )
        key_centroid_qvg = centroid_reconstruction(key_state)
        value_centroid_qvg = centroid_reconstruction(value_state)

        canonical_variants = {
            "qvg_k_only": (key_full_qvg, value_input),
            "qvg_v_only": (key_input, value_full_qvg),
            "qvg_both": (key_full_qvg, value_full_qvg),
            "centroid_k_only": (key_centroid_qvg, value_input),
            "centroid_v_only": (key_input, value_centroid_qvg),
            "centroid_both": (key_centroid_qvg, value_centroid_qvg),
        }
        dense_math = dense_attention(
            query, key_flat, value_flat, args.query_chunk_size
        )
        parity = relative_l2_by_head(target, dense_math)
        for name in variants:
            key_qvg, value_qvg = canonical_variants[name]
            canonical_full = scatter_qvg_frames(canonical_key, key_qvg, frames)
            value_full = scatter_qvg_frames(value, value_qvg, frames)
            key_full = transform_key_rope(
                canonical_full,
                metadata["key_frame_ids"],
                int(capture["height"]),
                int(capture["width"]),
                payload["rope_freqs"].to(device),
                inverse=False,
            )
            estimate = dense_attention(
                query,
                key_full.reshape_as(key_flat),
                value_full.reshape_as(value_flat),
                args.query_chunk_size,
            )
            error = relative_l2_by_head(dense_math, estimate)
            numerator, denominator = squared_error_by_head(dense_math, estimate)
            for local_head, head_index in enumerate(metadata["head_indices"]):
                variant_rows.append(
                    {
                        "prompt_id": metadata["prompt_id"],
                        "layer": int(metadata["layer"]),
                        "head_index": int(head_index),
                        "variant": name,
                        "relative_av_l2": float(error[local_head]),
                        "av_numerator_sq": float(numerator[local_head]),
                        "av_denominator_sq": float(denominator[local_head]),
                        "dense_capture_parity": float(parity[local_head]),
                    }
                )

        mass, leverage = frame_statistics(
            query,
            key_flat,
            value_flat,
            int(capture["key_frames"]),
            args.query_chunk_size,
        )
        for local_head, head_index in enumerate(metadata["head_indices"]):
            for frame_offset, absolute_frame in enumerate(metadata["key_frame_ids"]):
                frame_rows.append(
                    {
                        "prompt_id": metadata["prompt_id"],
                        "layer": int(metadata["layer"]),
                        "head_index": int(head_index),
                        "frame_offset": frame_offset,
                        "absolute_frame": int(absolute_frame),
                        "attention_mass": float(mass[local_head, frame_offset]),
                        "value_leverage_fraction": float(
                            leverage[local_head, frame_offset]
                        ),
                    }
                )
        records.append(
            {
                "identity": list(capture_identity(metadata)),
                "record_seed": seed,
                "dense_capture_parity_aggregate": float(
                    parity.square().mean().sqrt()
                ),
                "key_state_bytes": top_level_bytes(key_state),
                "value_state_bytes": top_level_bytes(value_state),
                "bf16_kv_bytes": int(
                    key_flat.numel() * key_flat.element_size()
                    + value_flat.numel() * value_flat.element_size()
                ),
                "packed_kv_bytes": tensor_tree_nbytes(key_state)
                + tensor_tree_nbytes(value_state),
            }
        )

    summaries = []
    for name in variants:
        rows = [row for row in variant_rows if row["variant"] == name]
        numerator = sum(float(row["av_numerator_sq"]) for row in rows)
        denominator = sum(float(row["av_denominator_sq"]) for row in rows)
        summaries.append(
            {
                "variant": name,
                "aggregate_relative_av_l2": math.sqrt(numerator / denominator),
                "worst_head_relative_av_l2": max(
                    float(row["relative_av_l2"]) for row in rows
                ),
                "head_records": len(rows),
            }
        )

    for path, rows in [
        (args.output_dir / "variant_metrics.csv", variant_rows),
        (args.output_dir / "frame_statistics.csv", frame_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "qvg_protocol_sha256": sha256_file(args.qvg_protocol),
        "source_protocol_sha256": sha256_file(args.source_protocol),
        "qvg_repository_commit": qvg_protocol["reference"]["repository_commit"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "manifests": manifests,
        "records": records,
        "variant_summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
