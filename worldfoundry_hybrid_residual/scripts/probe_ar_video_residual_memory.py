"""Evaluate residual-memory approximations on captured LongLive attention.

The evaluator keeps three questions separate:

1. Does the train-free temporal summary plus event rule work without a
   correction?
2. Is the remaining output defect low rank for each record (adaptive oracle)?
3. Does an output-channel basis fitted on calibration prompts transfer to
   validation/test records (frozen-basis oracle coefficients)?

The last two diagnostics use the dense defect to fit coefficients and are not
deployable methods. Validation/test defects never select a support rule, rank,
basis, threshold, or primary candidate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from ar_video_residual_memory_core import (
    adaptive_rank_projection,
    arithmetic_reduction,
    attention_from_representatives,
    build_representatives,
    dense_attention,
    fit_output_basis,
    make_recency_plan,
    phase_align_keys_for_temporal_summaries,
    project_onto_output_basis,
    select_residual_event_tiles,
)


@dataclass(frozen=True)
class Variant:
    method: str
    summary_key_mode: str
    summary_groups: int
    event_fraction: float


@dataclass
class CandidateResult:
    metadata: dict[str, Any]
    variant: Variant
    target: torch.Tensor
    estimate: torch.Tensor
    defect: torch.Tensor
    dense_tokens: int
    representative_tokens: int
    exact_tokens: int
    summary_tokens: int
    parity_by_head: torch.Tensor

    @property
    def cell_key(self) -> tuple[int, int, int, str]:
        return (
            int(self.metadata["layer"]),
            int(self.metadata["current_start_frame"]),
            int(self.metadata["denoising_call_index"]),
            self.variant.method,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a strict subset of preregistered captures for smoke tests.",
    )
    return parser.parse_args()


def event_label(value: float) -> str:
    return format(value, ".6g").replace(".", "p")


def make_variants(protocol: dict[str, Any]) -> list[Variant]:
    methods = protocol["methods"]
    variants = [Variant("sink_recent_drop_middle", "post_rope", 0, 0.0)]
    for mode in methods["summary_key_modes"]:
        prefix = "postrope" if mode == "post_rope" else "phasealigned"
        if mode not in {"post_rope", "phase_aligned"}:
            raise ValueError(f"unsupported summary key mode {mode}")
        for groups in methods["summary_group_counts"]:
            for fraction in methods["event_tile_fractions"]:
                variants.append(
                    Variant(
                        method=(
                            f"{prefix}_recency_g{int(groups)}"
                            f"_event_{event_label(float(fraction))}"
                        ),
                        summary_key_mode=mode,
                        summary_groups=int(groups),
                        event_fraction=float(fraction),
                    )
                )
    return variants


def primary_method(protocol: dict[str, Any]) -> str:
    primary = protocol["methods"]["primary_candidate"]
    prefix = (
        "phasealigned"
        if primary["summary_key_mode"] == "phase_aligned"
        else "postrope"
    )
    return (
        f"{prefix}_recency_g{int(primary['summary_group_count'])}"
        f"_event_{event_label(float(primary['event_tile_fraction']))}"
    )


def expected_identities(protocol: dict[str, Any]) -> set[tuple[str, int, int, int]]:
    capture = protocol["capture"]
    return {
        (prompt["id"], int(layer), int(start), int(call))
        for prompt in capture["prompts"]
        for layer in capture["layer_indices"]
        for start in capture["current_start_frames"]
        for call in capture["denoising_call_indices"]
    }


def capture_identity(metadata: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(metadata["prompt_id"]),
        int(metadata["layer"]),
        int(metadata["current_start_frame"]),
        int(metadata["denoising_call_index"]),
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifests(
    capture_dir: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(capture_dir.rglob("capture_manifest.json"))
    if not paths:
        raise ValueError(f"no capture manifests found under {capture_dir}")
    manifests = []
    artifact_paths: list[Path] = []
    common_signature: tuple[str, str, str, str] | None = None
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError(f"protocol mismatch in {path}")
        if manifest.get("source_commit") != protocol["model"]["code_commit"]:
            raise ValueError(f"source commit mismatch in {path}")
        expected_hashes = {
            "protocol_sha256": protocol_sha256,
            "generator_sha256": protocol["model"]["generator_sha256"],
            "lora_sha256": protocol["model"]["lora_sha256"],
        }
        for name, expected in expected_hashes.items():
            if manifest.get(name) != expected:
                raise ValueError(f"{name} mismatch in {path}")
        signature = (
            str(manifest.get("protocol_sha256")),
            str(manifest.get("runtime_config_sha256")),
            str(manifest.get("generator_sha256")),
            str(manifest.get("lora_sha256")),
        )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError(f"runtime or checkpoint signature mismatch in {path}")

        names = manifest.get("captures")
        if not isinstance(names, list) or not names:
            raise ValueError(f"capture membership is missing in {path}")
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate capture membership in {path}")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
                raise ValueError(f"unsafe capture member {name!r} in {path}")
            artifact = path.parent / name
            if not artifact.is_file():
                raise ValueError(f"manifest member does not exist: {artifact}")
            artifact_paths.append(artifact)
        manifest["manifest_path"] = str(path)
        manifests.append(manifest)
    resolved = [item.resolve() for item in artifact_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("a capture artifact is claimed by multiple manifests")
    return manifests, sorted(artifact_paths, key=str)


def validate_metadata(
    metadata: dict[str, Any],
    payload: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    capture = protocol["capture"]
    prompts = {item["id"]: item for item in capture["prompts"]}
    prompt_id = metadata.get("prompt_id")
    if prompt_id not in prompts:
        raise ValueError(f"unexpected prompt id {prompt_id}")
    prompt = prompts[prompt_id]
    required_equal = {
        "protocol_id": protocol["protocol_id"],
        "prompt_split": prompt["split"],
        "prompt_text": prompt["text"],
        "seed": int(prompt["seed"]),
        "frame_seq_len": int(capture["frame_seq_len"]),
        "head_indices": list(capture["heads"]),
    }
    for key, expected in required_equal.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"capture metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    if int(metadata["layer"]) not in capture["layer_indices"]:
        raise ValueError("unexpected layer")
    if int(metadata["current_start_frame"]) not in capture["current_start_frames"]:
        raise ValueError("unexpected current start frame")
    if int(metadata["denoising_call_index"]) not in capture["denoising_call_indices"]:
        raise ValueError("unexpected denoising call index")
    if metadata.get("denoising_timestep") is None:
        raise ValueError("warped denoising timestep is missing")

    query = payload["query"]
    key = payload["key"]
    value = payload["value"]
    dense_output = payload["dense_output"]
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("captured Q/K/V must have shape [tokens, heads, dim]")
    if query.shape != dense_output.shape:
        raise ValueError("captured query and dense output shapes differ")
    if key.shape != value.shape:
        raise ValueError("captured key and value shapes differ")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("captured Q/K/V head dimensions differ")
    if int(metadata["query_tokens_saved"]) != query.shape[0]:
        raise ValueError("saved query token count mismatch")
    if int(metadata["key_tokens"]) != key.shape[0]:
        raise ValueError("saved key token count mismatch")
    if key.shape[0] % int(capture["frame_seq_len"]) != 0:
        raise ValueError("captured key tokens are not frame aligned")
    if int(metadata["key_frames"]) != key.shape[0] // int(capture["frame_seq_len"]):
        raise ValueError("captured key frame count mismatch")
    if int(metadata["key_frames"]) != int(capture["local_attention_frames"]):
        raise ValueError("capture does not expose the preregistered local window")
    key_frame_ids = [int(item) for item in metadata.get("key_frame_ids", [])]
    query_frame_ids = [int(item) for item in metadata.get("query_frame_ids", [])]
    if len(key_frame_ids) != int(metadata["key_frames"]):
        raise ValueError("absolute key frame IDs are missing or inconsistent")
    if len(set(key_frame_ids)) != len(key_frame_ids):
        raise ValueError("absolute key frame IDs contain duplicates")
    if key_frame_ids[: int(capture["sink_frames"])] != list(
        range(int(capture["sink_frames"]))
    ):
        raise ValueError("captured key frame IDs do not begin with the frozen sink")

    query_frames, remainder = divmod(
        int(metadata["query_tokens_full"]), int(capture["frame_seq_len"])
    )
    if remainder:
        raise ValueError("full query count is not frame aligned")
    expected_saved = (
        query_frames
        * int(capture["query_tiles_per_record"])
        * int(capture["query_tile_size"])
    )
    if query.shape[0] != expected_saved:
        raise ValueError("captured query tile count mismatch")
    expected_query_frames = list(
        range(
            int(metadata["current_start_frame"]),
            int(metadata["current_start_frame"]) + query_frames,
        )
    )
    if query_frame_ids != expected_query_frames:
        raise ValueError("absolute query frame IDs are inconsistent")
    rope_freqs = payload.get("rope_freqs")
    if not isinstance(rope_freqs, torch.Tensor) or not rope_freqs.is_complex():
        raise ValueError("complex RoPE frequencies are required for phase-aware baselines")
    if not bool(torch.isfinite(rope_freqs).all()):
        raise ValueError("non-finite values in captured RoPE frequencies")
    for name in ("query", "key", "value", "dense_output"):
        if not bool(torch.isfinite(payload[name].float()).all()):
            raise ValueError(f"non-finite values in captured {name}")


def error_components(
    target: torch.Tensor,
    estimate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = target.float() - estimate.float()
    numerator_sq = difference.square().sum(dim=(0, 2))
    denominator_sq = target.float().square().sum(dim=(0, 2)).clamp_min(1e-24)
    return numerator_sq.sqrt() / denominator_sq.sqrt(), numerator_sq, denominator_sq


def append_metric_rows(
    rows: list[dict[str, Any]],
    result: CandidateResult,
    estimate: torch.Tensor,
    correction: str,
    rank: int,
    coefficient_scope: str,
) -> None:
    error, numerator_sq, denominator_sq = error_components(result.target, estimate)
    metadata = result.metadata
    for local_head, head_index in enumerate(metadata["head_indices"]):
        rows.append(
            {
                "protocol_id": metadata["protocol_id"],
                "prompt_id": metadata["prompt_id"],
                "split": metadata["prompt_split"],
                "seed": int(metadata["seed"]),
                "layer": int(metadata["layer"]),
                "current_start_frame": int(metadata["current_start_frame"]),
                "denoising_call_index": int(metadata["denoising_call_index"]),
                "denoising_timestep": int(metadata["denoising_timestep"]),
                "head_index": int(head_index),
                "method": result.variant.method,
                "summary_key_mode": result.variant.summary_key_mode,
                "summary_groups": result.variant.summary_groups,
                "event_fraction": result.variant.event_fraction,
                "correction": correction,
                "rank": rank,
                "coefficient_scope": coefficient_scope,
                "deployable": correction == "none",
                "relative_av_l2": float(error[local_head].item()),
                "numerator_sq": float(numerator_sq[local_head].item()),
                "denominator_sq": float(denominator_sq[local_head].item()),
                "dense_reference_parity": float(result.parity_by_head[local_head].item()),
                "dense_key_tokens": result.dense_tokens,
                "representative_tokens": result.representative_tokens,
                "exact_tokens": result.exact_tokens,
                "summary_tokens": result.summary_tokens,
                "arithmetic_reduction": arithmetic_reduction(
                    result.dense_tokens, result.representative_tokens
                ),
            }
        )


def evaluate_capture(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    variants: Iterable[Variant],
    device: torch.device,
    query_chunk_size: int,
    rows: list[dict[str, Any]],
) -> list[CandidateResult]:
    metadata = payload["metadata"]
    query = payload["query"].to(device)
    key_flat = payload["key"].to(device)
    value_flat = payload["value"].to(device)
    target = payload["dense_output"].to(device)

    dense_math = dense_attention(query, key_flat, value_flat, query_chunk_size)
    parity_by_head, _, _ = error_components(target, dense_math)

    capture = protocol["capture"]
    methods = protocol["methods"]
    frame_seq_len = int(capture["frame_seq_len"])
    key_frames = key_flat.shape[0] // frame_seq_len
    key = key_flat.reshape(key_frames, frame_seq_len, key_flat.shape[1], key_flat.shape[2])
    value = value_flat.reshape(
        key_frames, frame_seq_len, value_flat.shape[1], value_flat.shape[2]
    )

    results: list[CandidateResult] = []
    phase_aligned_cache: dict[int, torch.Tensor] = {}
    ranks = [int(rank) for rank in methods["low_rank_residual_ranks"] if int(rank) > 0]
    for variant in variants:
        plan = make_recency_plan(
            num_frames=key_frames,
            sink_frames=int(methods["exact_sink_frames"]),
            recent_frames=int(methods["exact_recent_frames"]),
            max_summary_groups=variant.summary_groups,
        )
        summary_key = key
        if variant.summary_key_mode == "phase_aligned":
            if variant.summary_groups not in phase_aligned_cache:
                grid = metadata["grid_sizes"]
                if len(grid) != 1 or len(grid[0]) != 3:
                    raise ValueError("capture requires one [frames, height, width] grid")
                phase_aligned_cache[variant.summary_groups] = (
                    phase_align_keys_for_temporal_summaries(
                        key=key,
                        plan=plan,
                        absolute_frame_ids=metadata["key_frame_ids"],
                        height=int(grid[0][1]),
                        width=int(grid[0][2]),
                        rope_freqs=payload["rope_freqs"].to(device),
                    )
                )
            summary_key = phase_aligned_cache[variant.summary_groups]
        event_mask = None
        if variant.event_fraction > 0:
            event_mask = select_residual_event_tiles(
                key=summary_key,
                value=value,
                plan=plan,
                tile_size=int(methods["event_tile_size"]),
                tile_fraction=variant.event_fraction,
                query=query,
            )
        representatives = build_representatives(
            key, value, plan, event_mask, summary_key=summary_key
        )
        estimate = attention_from_representatives(
            query, representatives, query_chunk_size=query_chunk_size
        )
        defect = target - estimate
        result = CandidateResult(
            metadata=metadata,
            variant=variant,
            target=target.detach().cpu(),
            estimate=estimate.detach().cpu(),
            defect=defect.detach().cpu(),
            dense_tokens=int(key_flat.shape[0]),
            representative_tokens=representatives.token_count,
            exact_tokens=representatives.exact_token_count,
            summary_tokens=representatives.summary_token_count,
            parity_by_head=parity_by_head.detach().cpu(),
        )
        results.append(result)
        append_metric_rows(
            rows, result, result.estimate, "none", 0, "train_free_rule"
        )
        for rank in ranks:
            projection = adaptive_rank_projection(defect, rank)
            append_metric_rows(
                rows,
                result,
                (estimate + projection).detach().cpu(),
                "adaptive_rank_oracle",
                rank,
                "same_record_dense_defect",
            )
    return results


def add_frozen_basis_metrics(
    candidates: list[CandidateResult],
    protocol: dict[str, Any],
    device: torch.device,
    rows: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    by_cell: defaultdict[tuple[int, int, int, str], list[CandidateResult]] = defaultdict(list)
    for result in candidates:
        by_cell[result.cell_key].append(result)

    ranks = [
        int(rank)
        for rank in protocol["methods"]["low_rank_residual_ranks"]
        if int(rank) > 0
    ]
    basis_artifact: dict[str, torch.Tensor] = {}
    for cell_key, cell_results in sorted(by_cell.items()):
        calibration = [
            item for item in cell_results if item.metadata["prompt_split"] == "calibration"
        ]
        held_out = [
            item for item in cell_results if item.metadata["prompt_split"] != "calibration"
        ]
        if not calibration or not held_out:
            continue
        for rank in ranks:
            basis = fit_output_basis(
                [item.defect.to(device) for item in calibration], rank
            )
            artifact_key = (
                f"l{cell_key[0]:02d}_f{cell_key[1]:03d}_c{cell_key[2]:02d}"
                f"__{cell_key[3]}__r{rank}"
            )
            basis_artifact[artifact_key] = basis.detach().cpu().float()
            for item in cell_results:
                projection = project_onto_output_basis(item.defect.to(device), basis)
                corrected = item.estimate.to(device) + projection
                coefficient_scope = (
                    "calibration_in_sample_dense_defect"
                    if item.metadata["prompt_split"] == "calibration"
                    else "held_out_dense_defect_with_calibration_basis"
                )
                append_metric_rows(
                    rows,
                    item,
                    corrected.detach().cpu(),
                    "frozen_calibration_basis_oracle_coefficients",
                    rank,
                    coefficient_scope,
                )
    return basis_artifact


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: dict[str, set[str]] = {
        "calibration": {"calibration"},
        "validation": {"validation"},
        "test": {"test"},
        "held_out": {"validation", "test"},
        "all": {"calibration", "validation", "test"},
    }
    summaries: list[dict[str, Any]] = []
    for scope, splits in scopes.items():
        selected = [row for row in rows if row["split"] in splits]
        grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(row["method"], row["correction"], int(row["rank"]))].append(row)
        for (method, correction, rank), group in sorted(grouped.items()):
            numerator_sq = sum(float(row["numerator_sq"]) for row in group)
            denominator_sq = sum(float(row["denominator_sq"]) for row in group)
            errors = [float(row["relative_av_l2"]) for row in group]
            reductions = [float(row["arithmetic_reduction"]) for row in group]
            summaries.append(
                {
                    "scope": scope,
                    "method": method,
                    "correction": correction,
                    "rank": rank,
                    "aggregate_relative_av_l2": math.sqrt(numerator_sq / denominator_sq),
                    "mean_head_relative_av_l2": sum(errors) / len(errors),
                    "p95_head_relative_av_l2": percentile(errors, 0.95),
                    "worst_head_relative_av_l2": max(errors),
                    "minimum_arithmetic_reduction": min(reductions),
                    "mean_arithmetic_reduction": sum(reductions) / len(reductions),
                    "head_records": len(group),
                }
            )
    return summaries


def find_summary(
    summaries: list[dict[str, Any]],
    scope: str,
    method: str,
    correction: str,
    rank: int,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in summaries
        if item["scope"] == scope
        and item["method"] == method
        and item["correction"] == correction
        and int(item["rank"]) == rank
    ]
    if len(matches) > 1:
        raise AssertionError("summary key is not unique")
    return matches[0] if matches else None


def decide_gate(
    protocol: dict[str, Any],
    summaries: list[dict[str, Any]],
    capture_validation: dict[str, Any],
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    primary = primary_method(protocol)
    rank = int(protocol["methods"]["primary_candidate"]["low_rank_residual_rank"])
    adaptive = find_summary(
        summaries, "held_out", primary, "adaptive_rank_oracle", rank
    )
    frozen = find_summary(
        summaries,
        "held_out",
        primary,
        "frozen_calibration_basis_oracle_coefficients",
        rank,
    )
    decision: dict[str, Any] = {
        "primary_method": primary,
        "primary_rank": rank,
        "adaptive_oracle": adaptive,
        "frozen_basis_transfer_oracle_coefficients": frozen,
        "speedup_claim_allowed": False,
        "note": (
            "Adaptive and frozen-basis coefficients use dense defects. Passing this gate "
            "would justify predictor work, not a deployment or speedup claim."
        ),
    }
    if not capture_validation["complete"]:
        decision["classification"] = "incomplete"
        decision["action"] = "finish_preregistered_capture"
        return decision
    if capture_validation["worst_dense_reference_parity"] > float(
        evaluation["dense_reference_parity_gate"]
    ):
        decision["classification"] = "invalid"
        decision["action"] = "repair_dense_reference_before_interpretation"
        return decision
    if adaptive is None or frozen is None:
        decision["classification"] = "invalid"
        decision["action"] = "repair_missing_calibration_or_transfer_metrics"
        return decision

    adaptive_quality = (
        adaptive["aggregate_relative_av_l2"] <= float(evaluation["aggregate_oracle_gate"])
        and adaptive["worst_head_relative_av_l2"] <= float(evaluation["worst_oracle_gate"])
    )
    arithmetic = adaptive["minimum_arithmetic_reduction"] >= float(
        evaluation["minimum_primary_arithmetic_reduction"]
    )
    frozen_quality = (
        frozen["aggregate_relative_av_l2"] <= float(evaluation["aggregate_transfer_gate"])
        and frozen["worst_head_relative_av_l2"] <= float(evaluation["worst_transfer_gate"])
    )
    decision["adaptive_quality_pass"] = adaptive_quality
    decision["arithmetic_screen_pass"] = arithmetic
    decision["frozen_basis_transfer_pass"] = frozen_quality
    if not adaptive_quality or not arithmetic:
        decision["classification"] = "null"
        decision["action"] = "stop_predictor_and_kernel_work_for_primary_candidate"
    elif not frozen_quality:
        decision["classification"] = "boundary"
        decision["action"] = "narrow_or_reject_static_low_rank_basis"
    else:
        decision["classification"] = "representation_pass"
        decision["action"] = "open_bounded_content_conditioned_predictor_gate"
    return decision


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no metric rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.query_chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        args.output_dir / "metrics.csv",
        args.output_dir / "summary.json",
        args.output_dir / "gate_decision.json",
        args.output_dir / "calibration_output_bases.pt",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing result artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifests, paths = validate_manifests(
        args.capture_dir, protocol, sha256_file(args.protocol)
    )

    expected = expected_identities(protocol)
    seen: dict[tuple[str, int, int, int], Path] = {}
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "metadata" not in payload:
            continue
        validate_metadata(payload["metadata"], payload, protocol)
        identity = capture_identity(payload["metadata"])
        if identity in seen:
            raise ValueError(f"duplicate capture identity in {seen[identity]} and {path}")
        if identity not in expected:
            raise ValueError(f"capture identity is outside the frozen protocol: {identity}")
        seen[identity] = path
        payloads.append(payload)

    missing = sorted(expected - set(seen))
    if missing and not args.allow_partial:
        raise ValueError(f"missing {len(missing)} preregistered captures; first={missing[0]}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested but CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)
    variants = make_variants(protocol)
    rows: list[dict[str, Any]] = []
    candidates: list[CandidateResult] = []
    for index, payload in enumerate(payloads, start=1):
        identity = capture_identity(payload["metadata"])
        print(f"[evaluate {index}/{len(payloads)}] {identity}")
        candidates.extend(
            evaluate_capture(
                payload,
                protocol,
                variants,
                device,
                args.query_chunk_size,
                rows,
            )
        )

    bases = add_frozen_basis_metrics(candidates, protocol, device, rows)
    summaries = summarize_rows(rows)
    parity_values = [
        float(value)
        for result in candidates[::len(variants)]
        for value in result.parity_by_head.tolist()
    ]
    capture_validation = {
        "expected_capture_count": len(expected),
        "found_capture_count": len(seen),
        "complete": not missing,
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
    torch.save(
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "fit_split": "calibration",
            "bases": bases,
        },
        args.output_dir / "calibration_output_bases.pt",
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
