#!/usr/bin/env python3
"""Probe position-bucketed defect basis banks across independent Wan seeds.

The sparse mask is frozen from the calibration replay so this experiment
isolates low-rank-basis transfer.  A bank with K entries fits one rank-r output
basis per contiguous query-tile group on calibration data and applies the same
position bucket to a different-seed replay with oracle coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from probe_dynamic_sparse_lowrank_oracle import (
    aligned_query_tile_starts,
    block_output_contributions,
    mask_jaccard,
    outputs_from_selected_blocks,
    residual_after_basis,
    resolve_capture,
    right_singular_basis,
    selection_orders,
    subspace_overlap,
    write_csv,
)


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--calibration-sample-id", required=True)
    parser.add_argument("--test-sample-id", required=True)
    parser.add_argument("--branch", default="cond")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sampling-step", type=int, default=0)
    parser.add_argument("--heads", type=parse_ints, default=())
    parser.add_argument("--method", choices=("mass_topk", "contribution_norm"), default="mass_topk")
    parser.add_argument("--density", type=float, default=0.125)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--bank-counts", type=parse_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument("--query-tile-size", type=int, default=64)
    parser.add_argument("--key-block-size", type=int, default=64)
    parser.add_argument("--query-tiles", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--error-target", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tile_group(tile_index: int, tile_count: int, bank_count: int) -> int:
    if bank_count <= 0 or bank_count > tile_count:
        raise ValueError("bank count must be in [1, query_tiles]")
    return min(bank_count - 1, tile_index * bank_count // tile_count)


@torch.inference_mode()
def capture_residuals(
    path: Path,
    args: argparse.Namespace,
    frozen_masks: dict[int, torch.Tensor] | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all, k_all, v_all = payload["q"][0], payload["k"][0], payload["v"][0]
    tokens, total_heads, dimension = q_all.shape
    heads = args.heads or tuple(range(total_heads))
    starts = aligned_query_tile_starts(tokens, args.query_tile_size, args.query_tiles)
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    device = torch.device(args.device)
    records = {}
    for head in heads:
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        residuals = []
        references = []
        dynamic_masks = []
        applied_masks = []
        masses = []
        for tile_index, start in enumerate(starts):
            scores = q[start : start + args.query_tile_size] @ k.T * scale
            attention = torch.softmax(scores, dim=1)
            reference = attention @ v
            contributions, block_mass, _ = block_output_contributions(
                attention, v, args.key_block_size
            )
            blocks = contributions.shape[0]
            budget = max(1, round(args.density * blocks))
            order = selection_orders(
                contributions, block_mass, (args.method,), budget
            )[args.method]
            dynamic = torch.zeros(blocks, dtype=torch.bool, device=device)
            dynamic[order[:budget]] = True
            applied = (
                dynamic
                if frozen_masks is None
                else frozen_masks[head][tile_index].to(device)
            )
            selected = applied.nonzero(as_tuple=False).flatten()
            _, estimate, mass = outputs_from_selected_blocks(
                contributions, block_mass, selected
            )
            residuals.append((reference - estimate).cpu())
            references.append(reference.cpu())
            dynamic_masks.append(dynamic.cpu())
            applied_masks.append(applied.cpu())
            masses.append(float(mass.mean()))
        records[head] = {
            "residuals": tuple(residuals),
            "references": tuple(references),
            "dynamic_masks": torch.stack(dynamic_masks),
            "applied_masks": torch.stack(applied_masks),
            "attention_mass": sum(masses) / len(masses),
            "head_dim": dimension,
        }
        print(
            f"[basis-bank] sample={metadata.get('sample_id')} head={head} tiles={len(starts)}",
            flush=True,
        )
        del q, k, v
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del payload, q_all, k_all, v_all
    return {"metadata": metadata, "records": records, "query_starts": starts}


def frozen_masks(capture: dict[str, object]) -> dict[int, torch.Tensor]:
    return {
        head: record["dynamic_masks"]
        for head, record in capture["records"].items()
    }


def evaluate_bank(
    calibration_tiles: tuple[torch.Tensor, ...],
    test_tiles: tuple[torch.Tensor, ...],
    test_references: tuple[torch.Tensor, ...],
    bank_count: int,
    rank: int,
) -> dict[str, object]:
    tile_count = len(calibration_tiles)
    bases = []
    test_oracle_bases = []
    group_overlaps = []
    group_energies = []
    final_tiles: list[torch.Tensor] = [torch.empty(0)] * tile_count
    for group in range(bank_count):
        indices = [
            index
            for index in range(tile_count)
            if tile_group(index, tile_count, bank_count) == group
        ]
        calibration_defect = torch.cat([calibration_tiles[index] for index in indices])
        test_defect = torch.cat([test_tiles[index] for index in indices])
        basis, _ = right_singular_basis(calibration_defect, rank)
        test_basis, _ = right_singular_basis(test_defect, rank)
        group_overlaps.append(subspace_overlap(basis, test_basis))
        projected = residual_after_basis(test_defect, basis)
        group_energies.append(
            1.0 - float(projected.square().sum() / test_defect.square().sum().clamp_min(1e-30))
        )
        offset = 0
        for index in indices:
            rows = test_tiles[index].shape[0]
            final_tiles[index] = projected[offset : offset + rows]
            offset += rows
        bases.append(basis)
        test_oracle_bases.append(test_basis)
    reference_sq = sum(float(reference.square().sum()) for reference in test_references)
    final_sq = sum(float(residual.square().sum()) for residual in final_tiles)
    tile_errors = [
        math.sqrt(
            float(residual.square().sum())
            / max(float(reference.square().sum()), 1e-30)
        )
        for residual, reference in zip(final_tiles, test_references)
    ]
    return {
        "residual_sq": final_sq,
        "reference_sq": reference_sq,
        "relative_l2": math.sqrt(final_sq / max(reference_sq, 1e-30)),
        "tile_error_mean": sum(tile_errors) / len(tile_errors),
        "tile_error_max": max(tile_errors),
        "basis_overlap_mean": sum(group_overlaps) / len(group_overlaps),
        "captured_energy_mean": sum(group_energies) / len(group_energies),
    }


def aggregate(rows: list[dict[str, object]], target: float) -> list[dict[str, object]]:
    summary = []
    for bank_count in sorted({int(row["bank_count"]) for row in rows}):
        group = [row for row in rows if int(row["bank_count"]) == bank_count]
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        residual_sq = sum(float(row["residual_sq"]) for row in group)
        head_errors = [float(row["relative_l2"]) for row in group]
        aggregate_error = math.sqrt(residual_sq / max(reference_sq, 1e-30))
        summary.append(
            {
                "bank_count": bank_count,
                "heads": len(group),
                "aggregate_relative_l2": aggregate_error,
                "head_error_median": statistics.median(head_errors),
                "head_error_max": max(head_errors),
                "worst_head": group[max(range(len(group)), key=lambda i: head_errors[i])]["head"],
                "tile_error_max": max(float(row["tile_error_max"]) for row in group),
                "mask_jaccard_mean": sum(float(row["mask_jaccard"]) for row in group) / len(group),
                "basis_overlap_mean": sum(float(row["basis_overlap_mean"]) for row in group) / len(group),
                "captured_energy_mean": sum(float(row["captured_energy_mean"]) for row in group) / len(group),
                "basis_parameters": sum(int(row["basis_parameters"]) for row in group),
                "basis_fp16_mib": sum(int(row["basis_parameters"]) for row in group) * 2 / 2**20,
                "aggregate_go": aggregate_error <= target,
                "all_heads_go": max(head_errors) <= target,
            }
        )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if any(count > args.query_tiles for count in args.bank_counts):
        raise ValueError("bank counts cannot exceed query tiles")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    calibration_path = resolve_capture(
        args.capture_index, args.calibration_sample_id, args.branch, args.layer, args.sampling_step
    )
    test_path = resolve_capture(
        args.capture_index, args.test_sample_id, args.branch, args.layer, args.sampling_step
    )
    started = time.time()
    calibration = capture_residuals(calibration_path, args)
    test = capture_residuals(test_path, args, frozen_masks(calibration))
    rows = []
    for head, calibration_record in calibration["records"].items():
        test_record = test["records"][head]
        for bank_count in args.bank_counts:
            metrics = evaluate_bank(
                calibration_record["residuals"],
                test_record["residuals"],
                test_record["references"],
                bank_count,
                args.rank,
            )
            rows.append(
                {
                    "head": head,
                    "bank_count": bank_count,
                    "rank": args.rank,
                    "density": args.density,
                    "method": args.method,
                    "mask_jaccard": mask_jaccard(
                        calibration_record["dynamic_masks"],
                        test_record["dynamic_masks"],
                    ),
                    "attention_mass": test_record["attention_mass"],
                    "basis_parameters": bank_count * int(test_record["head_dim"]) * args.rank,
                    **metrics,
                }
            )
    summary = aggregate(rows, args.error_target)
    feasible = [row for row in summary if row["aggregate_go"] and row["all_heads_go"]]
    decision = {
        "verdict": "GO_POSITION_BUCKETED_BANK" if feasible else "NO_GO_POSITION_ONLY_BANK",
        "selected": feasible[0] if feasible else min(summary, key=lambda row: float(row["head_error_max"])),
        "error_target": args.error_target,
        "interpretation": "oracle coefficients with a frozen position bucket; coefficient prediction and H200 timing remain unverified",
    }
    write_csv(args.output_dir / "conditional_basis_bank_heads.csv", rows)
    write_csv(args.output_dir / "conditional_basis_bank_summary.csv", summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "calibration_path": str(calibration_path),
        "test_path": str(test_path),
        "query_starts": calibration["query_starts"],
        "methodology": "frozen calibration sparse masks and position-bucketed calibration defect bases, test oracle coefficients",
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[basis-bank] verdict={decision['verdict']}", flush=True)
    print(f"[basis-bank] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
