#!/usr/bin/env python3
"""Alternating sparse routing that minimizes the low-rank-unrepairable defect.

For a fixed selected-key density, the probe alternates between (1) fitting the
best output-channel rank-r defect basis and (2) rerouting key blocks after
projecting outputs onto the orthogonal complement of that basis.  This tests
whether a sparse router designed jointly with its low-rank tail improves over
routers that minimize critical-only error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from probe_dynamic_sparse_lowrank_oracle import (
    aligned_query_tile_starts,
    block_output_contributions,
    mask_jaccard,
    outputs_from_selected_blocks,
    renorm_output_greedy_order,
    resolve_capture,
    residual_after_basis,
    right_singular_basis,
    selection_orders,
    subspace_overlap,
    write_csv,
)


@dataclass
class TileData:
    contributions: torch.Tensor
    block_mass: torch.Tensor
    reference: torch.Tensor


@dataclass
class RoutedResult:
    selections: tuple[torch.Tensor, ...]
    defect: torch.Tensor
    basis: torch.Tensor
    final_residual: torch.Tensor
    initial_method: str
    initial_relative_l2: float
    final_relative_l2: float
    accepted_iterations: int
    tail_energy: float


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


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
    parser.add_argument("--densities", type=parse_floats, default=(0.0625, 0.125))
    parser.add_argument("--ranks", type=parse_ints, default=(8, 16))
    parser.add_argument("--query-tile-size", type=int, default=64)
    parser.add_argument("--key-block-size", type=int, default=64)
    parser.add_argument("--query-tiles", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--error-target", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_l2(residual: torch.Tensor, reference_sq: float) -> float:
    return math.sqrt(float(residual.square().sum()) / max(reference_sq, 1e-30))


def residual_for_selections(
    tiles: tuple[TileData, ...], selections: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    residuals = []
    for tile, selected in zip(tiles, selections):
        _, estimate, _ = outputs_from_selected_blocks(
            tile.contributions, tile.block_mass, selected
        )
        residuals.append(tile.reference - estimate)
    return torch.cat(residuals, dim=0)


def evaluate_selections(
    tiles: tuple[TileData, ...],
    selections: tuple[torch.Tensor, ...],
    rank: int,
    reference_sq: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    defect = residual_for_selections(tiles, selections)
    basis, energy = right_singular_basis(defect, rank)
    final = residual_after_basis(defect, basis)
    return defect, basis, final, relative_l2(final, reference_sq), energy


def tail_aware_route(
    tiles: tuple[TileData, ...],
    budget: int,
    rank: int,
    iterations: int,
) -> RoutedResult:
    reference_sq = sum(float(tile.reference.square().sum()) for tile in tiles)
    initial_methods = ("mass_topk", "contribution_norm", "renorm_output_greedy")
    initial_orders = [
        selection_orders(
            tile.contributions, tile.block_mass, initial_methods, budget
        )
        for tile in tiles
    ]
    candidates: list[tuple[float, str, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    for method in initial_methods:
        selections = tuple(orders[method][:budget] for orders in initial_orders)
        defect, basis, final, error, energy = evaluate_selections(
            tiles, selections, rank, reference_sq
        )
        candidates.append((error, method, selections, defect, basis, final, energy))
    error, method, selections, defect, basis, final, energy = min(
        candidates, key=lambda item: item[0]
    )
    initial_error = error
    accepted = 0

    for _ in range(iterations):
        projected_tiles = []
        for tile in tiles:
            projected = tile.contributions - (
                (tile.contributions @ basis) @ basis.T
            )
            projected_tiles.append(projected)
        proposed = tuple(
            renorm_output_greedy_order(projected, tile.block_mass, budget)
            for projected, tile in zip(projected_tiles, tiles)
        )
        proposed_defect, proposed_basis, proposed_final, proposed_error, proposed_energy = (
            evaluate_selections(tiles, proposed, rank, reference_sq)
        )
        if proposed_error >= error - 1e-10:
            break
        selections = proposed
        defect = proposed_defect
        basis = proposed_basis
        final = proposed_final
        error = proposed_error
        energy = proposed_energy
        accepted += 1

    return RoutedResult(
        selections=selections,
        defect=defect,
        basis=basis,
        final_residual=final,
        initial_method=method,
        initial_relative_l2=initial_error,
        final_relative_l2=error,
        accepted_iterations=accepted,
        tail_energy=energy,
    )


def evaluate_with_basis(
    tiles: tuple[TileData, ...],
    selections: tuple[torch.Tensor, ...],
    basis: torch.Tensor,
    reference_sq: float,
) -> tuple[torch.Tensor, float]:
    defect = residual_for_selections(tiles, selections)
    final = residual_after_basis(defect, basis)
    return defect, relative_l2(final, reference_sq)


@torch.inference_mode()
def capture_tiles(
    path: Path,
    args: argparse.Namespace,
    callback,
) -> dict[int, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all, k_all, v_all = payload["q"][0], payload["k"][0], payload["v"][0]
    tokens, total_heads, dimension = q_all.shape
    heads = args.heads or tuple(range(total_heads))
    starts = aligned_query_tile_starts(tokens, args.query_tile_size, args.query_tiles)
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    device = torch.device(args.device)
    outputs = {}
    for head in heads:
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        tiles = []
        for start in starts:
            scores = q[start : start + args.query_tile_size] @ k.T * scale
            attention = torch.softmax(scores, dim=1)
            reference = attention @ v
            contributions, block_mass, _ = block_output_contributions(
                attention, v, args.key_block_size
            )
            tiles.append(TileData(contributions, block_mass, reference))
        outputs[head] = callback(head, tuple(tiles), metadata)
        print(
            f"[tail-aware] sample={metadata.get('sample_id')} head={head} tiles={len(tiles)}",
            flush=True,
        )
        del q, k, v, tiles
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del payload, q_all, k_all, v_all
    return outputs


def mask_tensor(result: RoutedResult, blocks: int) -> torch.Tensor:
    masks = torch.zeros((len(result.selections), blocks), dtype=torch.bool)
    for tile, selected in enumerate(result.selections):
        masks[tile, selected.cpu()] = True
    return masks


def aggregate(rows: list[dict[str, object]], target: float) -> list[dict[str, object]]:
    summary = []
    keys = sorted({(float(row["density"]), int(row["rank"])) for row in rows})
    error_fields = (
        "test_dynamic_initial_residual_sq",
        "test_dynamic_final_residual_sq",
        "test_dynamic_calibration_basis_residual_sq",
        "test_frozen_self_residual_sq",
        "test_frozen_calibration_basis_residual_sq",
    )
    for density, rank in keys:
        group = [row for row in rows if float(row["density"]) == density and int(row["rank"]) == rank]
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        metrics = {
            field.replace("_residual_sq", "_relative_l2"): math.sqrt(
                sum(float(row[field]) for row in group) / max(reference_sq, 1e-30)
            )
            for field in error_fields
        }
        summary.append(
            {
                "density": density,
                "rank": rank,
                "heads": len(group),
                "accepted_iterations_mean": sum(float(row["test_accepted_iterations"]) for row in group) / len(group),
                "mask_jaccard_mean": sum(float(row["mask_jaccard"]) for row in group) / len(group),
                "basis_overlap_mean": sum(float(row["basis_overlap"]) for row in group) / len(group),
                "test_tail_energy_mean": sum(float(row["test_tail_energy"]) for row in group) / len(group),
                **metrics,
                "dynamic_go": metrics["test_dynamic_final_relative_l2"] <= target,
                "frozen_structure_go": metrics["test_frozen_calibration_basis_relative_l2"] <= target,
            }
        )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.calibration_sample_id == args.test_sample_id:
        raise ValueError("calibration and test samples must differ")
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

    def calibrate(head: int, tiles: tuple[TileData, ...], metadata: dict[str, object]):
        blocks = tiles[0].contributions.shape[0]
        return {
            (density, rank): tail_aware_route(
                tiles, max(1, round(density * blocks)), rank, args.iterations
            )
            for density in args.densities
            for rank in args.ranks
        }

    calibration = capture_tiles(calibration_path, args, calibrate)
    rows: list[dict[str, object]] = []

    def test_callback(head: int, tiles: tuple[TileData, ...], metadata: dict[str, object]):
        blocks = tiles[0].contributions.shape[0]
        reference_sq = sum(float(tile.reference.square().sum()) for tile in tiles)
        head_results = {}
        for density in args.densities:
            for rank in args.ranks:
                calibration_result = calibration[head][(density, rank)]
                test_result = tail_aware_route(
                    tiles, max(1, round(density * blocks)), rank, args.iterations
                )
                calibration_basis = calibration_result.basis.to(device)
                dynamic_defect, _ = evaluate_with_basis(
                    tiles, test_result.selections, calibration_basis, reference_sq
                )
                frozen_selections = tuple(
                    selected.to(device) for selected in calibration_result.selections
                )
                frozen_defect = residual_for_selections(tiles, frozen_selections)
                frozen_basis, _ = right_singular_basis(frozen_defect, rank)
                frozen_self = residual_after_basis(frozen_defect, frozen_basis)
                frozen_calibration = residual_after_basis(frozen_defect, calibration_basis)
                initial_residual_sq = (
                    test_result.initial_relative_l2**2 * reference_sq
                )
                rows.append(
                    {
                        "head": head,
                        "density": density,
                        "rank": rank,
                        "calibration_initial_method": calibration_result.initial_method,
                        "test_initial_method": test_result.initial_method,
                        "calibration_accepted_iterations": calibration_result.accepted_iterations,
                        "test_accepted_iterations": test_result.accepted_iterations,
                        "mask_jaccard": mask_jaccard(
                            mask_tensor(calibration_result, blocks),
                            mask_tensor(test_result, blocks),
                        ),
                        "basis_overlap": subspace_overlap(
                            calibration_result.basis.cpu(), test_result.basis.cpu()
                        ),
                        "test_tail_energy": test_result.tail_energy,
                        "test_dynamic_initial_residual_sq": initial_residual_sq,
                        "test_dynamic_final_residual_sq": float(test_result.final_residual.square().sum()),
                        "test_dynamic_calibration_basis_residual_sq": float(
                            residual_after_basis(dynamic_defect, calibration_basis).square().sum()
                        ),
                        "test_frozen_self_residual_sq": float(frozen_self.square().sum()),
                        "test_frozen_calibration_basis_residual_sq": float(
                            frozen_calibration.square().sum()
                        ),
                        "reference_sq": reference_sq,
                    }
                )
                head_results[(density, rank)] = test_result
        return head_results

    capture_tiles(test_path, args, test_callback)
    summary = aggregate(rows, args.error_target)
    best_dynamic = min(summary, key=lambda row: float(row["test_dynamic_final_relative_l2"]))
    best_static = min(
        summary, key=lambda row: float(row["test_frozen_calibration_basis_relative_l2"])
    )
    decision = {
        "best_dynamic": best_dynamic,
        "best_frozen_structure": best_static,
        "error_target": args.error_target,
        "dynamic_go": bool(best_dynamic["dynamic_go"]),
        "frozen_structure_go": bool(best_static["frozen_structure_go"]),
        "warning": "alternating greedy routing is a heuristic witness, not a globally optimal mask certificate",
    }
    write_csv(args.output_dir / "tail_aware_sparse_router_heads.csv", rows)
    write_csv(args.output_dir / "tail_aware_sparse_router_summary.csv", summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "calibration_path": str(calibration_path),
        "test_path": str(test_path),
        "method": "alternating renormalized sparse routing on the rank-r basis orthogonal complement",
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[tail-aware] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
