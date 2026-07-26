#!/usr/bin/env python3
"""Output-aware oracle for dynamic block-sparse attention plus a low-rank tail.

This probe intentionally separates representation and transfer questions.  A
per-sample oracle chooses hardware-aligned key blocks for each query tile, then
fits the best output-channel low-rank correction.  A held-out replay is also
evaluated with the calibration mask and/or calibration output basis frozen.

The ``dense_probability`` path is a generous representation upper bound: it
uses probabilities normalized over all keys.  The ``renormalized`` path only
renormalizes selected keys and is the relevant numerical target for a sparse
attention implementation.  Neither path by itself establishes acceleration;
that requires a fused kernel benchmark.
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
import torch.nn.functional as F


DEFAULT_METHODS = "mass_topk,contribution_norm,dense_output_greedy,renorm_output_greedy"
DEFAULT_DENSITIES = "0.03125,0.0625,0.125,0.25"
DEFAULT_RANKS = "0,8,16"


def parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in parse_csv_strings(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def parse_floats(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in parse_csv_strings(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--calibration-sample-id", required=True)
    parser.add_argument("--test-sample-id", required=True)
    parser.add_argument("--branch", default="cond")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sampling-step", type=int, default=0)
    parser.add_argument("--heads", type=parse_ints, default=())
    parser.add_argument("--methods", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_METHODS))
    parser.add_argument("--densities", type=parse_floats, default=parse_floats(DEFAULT_DENSITIES))
    parser.add_argument("--ranks", type=parse_ints, default=parse_ints(DEFAULT_RANKS))
    parser.add_argument("--query-tile-size", type=int, default=64)
    parser.add_argument("--key-block-size", type=int, default=64)
    parser.add_argument("--query-tiles", type=int, default=4)
    parser.add_argument("--error-target", type=float, default=0.02)
    parser.add_argument("--max-deployable-density", type=float, default=0.125)
    parser.add_argument("--max-deployable-rank", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aligned_query_tile_starts(tokens: int, tile_size: int, tile_count: int) -> tuple[int, ...]:
    if tile_size <= 0 or tile_count <= 0:
        raise ValueError("tile size and count must be positive")
    full_tiles = tokens // tile_size
    if tile_count > full_tiles:
        raise ValueError(f"requested {tile_count} query tiles but only {full_tiles} fit")
    if tile_count == 1:
        return (0,)
    tile_ids = torch.linspace(0, full_tiles - 1, tile_count, dtype=torch.float64).round().long()
    if tile_ids.unique().numel() != tile_count:
        raise RuntimeError("stratified query tile selection produced duplicate tiles")
    return tuple(int(index) * tile_size for index in tile_ids.tolist())


def padded_attention_and_values(
    attention: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if attention.ndim != 2 or value.ndim != 2:
        raise ValueError("attention and value must be rank-2")
    if attention.shape[1] != value.shape[0]:
        raise ValueError("attention key dimension and value tokens must match")
    tokens = value.shape[0]
    blocks = math.ceil(tokens / block_size)
    padded_tokens = blocks * block_size
    padding = padded_tokens - tokens
    attention_padded = F.pad(attention, (0, padding))
    value_padded = F.pad(value, (0, 0, 0, padding))
    return (
        attention_padded.reshape(attention.shape[0], blocks, block_size),
        value_padded.reshape(blocks, block_size, value.shape[1]),
        padding,
    )


def block_output_contributions(
    attention: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    attention_blocks, value_blocks, padding = padded_attention_and_values(
        attention, value, block_size
    )
    contributions = torch.einsum("qbk,bkd->bqd", attention_blocks, value_blocks)
    block_mass = attention_blocks.sum(dim=2).T.contiguous()
    return contributions, block_mass, padding


def dense_output_greedy_order(contributions: torch.Tensor, budget: int) -> torch.Tensor:
    """Greedily minimize ||sum(all blocks) - sum(selected blocks)||_F."""
    budget = min(budget, contributions.shape[0])
    target = contributions.sum(dim=0)
    residual = target.clone()
    energy = contributions.square().sum(dim=(1, 2))
    available = torch.ones(contributions.shape[0], dtype=torch.bool, device=contributions.device)
    selected: list[torch.Tensor] = []
    for _ in range(budget):
        gain = 2.0 * torch.einsum("bqd,qd->b", contributions, residual) - energy
        gain.masked_fill_(~available, float("-inf"))
        index = gain.argmax()
        selected.append(index)
        available[index] = False
        residual = residual - contributions[index]
    return torch.stack(selected).long()


def renorm_output_greedy_order(
    contributions: torch.Tensor,
    block_mass: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Greedily minimize output error after selected-key renormalization."""
    budget = min(budget, contributions.shape[0])
    target = contributions.sum(dim=0)
    numerator = torch.zeros_like(target)
    mass = torch.zeros(target.shape[0], dtype=target.dtype, device=target.device)
    available = torch.ones(contributions.shape[0], dtype=torch.bool, device=target.device)
    selected: list[torch.Tensor] = []
    for _ in range(budget):
        candidate_numerator = numerator.unsqueeze(0) + contributions
        candidate_mass = mass.unsqueeze(0) + block_mass
        estimates = candidate_numerator / candidate_mass.clamp_min(1e-30).unsqueeze(2)
        error = (estimates - target.unsqueeze(0)).square().sum(dim=(1, 2))
        error.masked_fill_(~available, float("inf"))
        index = error.argmin()
        selected.append(index)
        available[index] = False
        numerator = numerator + contributions[index]
        mass = mass + block_mass[index]
    return torch.stack(selected).long()


def selection_orders(
    contributions: torch.Tensor,
    block_mass: torch.Tensor,
    methods: tuple[str, ...],
    max_budget: int,
) -> dict[str, torch.Tensor]:
    supported = {
        "mass_topk",
        "contribution_norm",
        "dense_output_greedy",
        "renorm_output_greedy",
    }
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(f"unsupported selection methods: {sorted(unknown)}")
    orders: dict[str, torch.Tensor] = {}
    if "mass_topk" in methods:
        orders["mass_topk"] = block_mass.sum(dim=1).argsort(descending=True)[:max_budget]
    if "contribution_norm" in methods:
        orders["contribution_norm"] = (
            contributions.square().sum(dim=(1, 2)).argsort(descending=True)[:max_budget]
        )
    if "dense_output_greedy" in methods:
        orders["dense_output_greedy"] = dense_output_greedy_order(contributions, max_budget)
    if "renorm_output_greedy" in methods:
        orders["renorm_output_greedy"] = renorm_output_greedy_order(
            contributions, block_mass, max_budget
        )
    return orders


def outputs_from_selected_blocks(
    contributions: torch.Tensor,
    block_mass: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = contributions.index_select(0, selected).sum(dim=0)
    mass = block_mass.index_select(0, selected).sum(dim=0)
    renormalized = numerator / mass.clamp_min(1e-30).unsqueeze(1)
    return numerator, renormalized, mass


def right_singular_basis(defect: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    return right_singular_bases(defect, (rank,))[rank]


def right_singular_bases(
    defect: torch.Tensor, ranks: tuple[int, ...]
) -> dict[int, tuple[torch.Tensor, float]]:
    results = {
        rank: (defect.new_zeros((defect.shape[1], 0)), 0.0)
        for rank in ranks
        if rank <= 0
    }
    positive_ranks = sorted({rank for rank in ranks if rank > 0})
    if not positive_ranks:
        return results
    covariance = defect.T @ defect
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    positive = eigenvalues.clamp_min(0)
    total = positive.sum().clamp_min(1e-30)
    for rank in positive_ranks:
        used = min(rank, defect.shape[1])
        basis = eigenvectors[:, order[:used]].contiguous()
        explained = float(positive[order[:used]].sum() / total)
        results[rank] = basis, explained
    return results


def residual_after_basis(defect: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[1] == 0:
        return defect
    return defect - (defect @ basis) @ basis.T


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    used = min(left.shape[1], right.shape[1])
    if used == 0:
        return 1.0
    return float((left.T @ right).square().sum() / used)


def mask_jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = torch.logical_and(left, right).sum(dim=1).float()
    union = torch.logical_or(left, right).sum(dim=1).float().clamp_min(1)
    return float((intersection / union).mean())


def resolve_capture(
    index_path: Path,
    sample_id: str,
    branch: str,
    layer: int,
    sampling_step: int,
) -> Path:
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row["sample_id"] == sample_id
        and row["branch"] == branch
        and int(row["layer"]) == layer
        and int(row["sampling_step"]) == sampling_step
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one capture for {sample_id}, found {len(matches)}")
    path = Path(matches[0]["path"])
    if not path.is_absolute():
        path = index_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


@torch.inference_mode()
def process_capture(
    path: Path,
    args: argparse.Namespace,
    frozen_masks: dict[tuple[int, str, float], torch.Tensor] | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    q_all, k_all, v_all = payload["q"][0], payload["k"][0], payload["v"][0]
    tokens, total_heads, dimension = q_all.shape
    heads = args.heads or tuple(range(total_heads))
    if any(head < 0 or head >= total_heads for head in heads):
        raise ValueError(f"heads must be in [0, {total_heads})")
    query_starts = aligned_query_tile_starts(tokens, args.query_tile_size, args.query_tiles)
    key_blocks = math.ceil(tokens / args.key_block_size)
    budgets = {
        density: max(1, min(key_blocks, int(round(density * key_blocks))))
        for density in args.densities
    }
    max_budget = max(budgets.values())
    scale = float(payload.get("softmax_scale", dimension**-0.5))
    device = torch.device(args.device)
    result_heads: dict[int, dict[str, object]] = {}

    for head in heads:
        q = q_all[:, head].to(device=device, dtype=torch.float32)
        k = k_all[:, head].to(device=device, dtype=torch.float32)
        v = v_all[:, head].to(device=device, dtype=torch.float32)
        references: list[torch.Tensor] = []
        residuals: dict[tuple[str, float, str, str], list[torch.Tensor]] = defaultdict(list)
        masses: dict[tuple[str, float, str], list[float]] = defaultdict(list)
        masks: dict[tuple[str, float], list[torch.Tensor]] = defaultdict(list)

        for tile_index, start in enumerate(query_starts):
            sampled_q = q[start : start + args.query_tile_size]
            scores = sampled_q @ k.T * scale
            attention = torch.softmax(scores, dim=1)
            reference = attention @ v
            contributions, block_mass, _ = block_output_contributions(
                attention, v, args.key_block_size
            )
            orders = selection_orders(contributions, block_mass, args.methods, max_budget)
            references.append(reference.cpu())

            for method, order in orders.items():
                for density, budget in budgets.items():
                    selected = order[:budget]
                    selected_mask = torch.zeros(key_blocks, dtype=torch.bool, device=device)
                    selected_mask[selected] = True
                    masks[(method, density)].append(selected_mask.cpu())
                    dense_probability, renormalized, mass = outputs_from_selected_blocks(
                        contributions, block_mass, selected
                    )
                    for normalization, estimate in (
                        ("dense_probability", dense_probability),
                        ("renormalized", renormalized),
                    ):
                        residuals[(method, density, normalization, "oracle_route")].append(
                            (reference - estimate).cpu()
                        )
                        masses[(method, density, "oracle_route")].append(float(mass.mean()))

                    if frozen_masks is None:
                        continue
                    frozen = frozen_masks[(head, method, density)][tile_index].to(device)
                    frozen_selected = frozen.nonzero(as_tuple=False).flatten()
                    frozen_dense, frozen_renorm, frozen_mass = outputs_from_selected_blocks(
                        contributions, block_mass, frozen_selected
                    )
                    for normalization, estimate in (
                        ("dense_probability", frozen_dense),
                        ("renormalized", frozen_renorm),
                    ):
                        residuals[(method, density, normalization, "frozen_route")].append(
                            (reference - estimate).cpu()
                        )
                        masses[(method, density, "frozen_route")].append(
                            float(frozen_mass.mean())
                        )
            del scores, attention, reference, contributions, block_mass, orders

        reference_matrix = torch.cat(references, dim=0)
        result_heads[head] = {
            "reference": reference_matrix,
            "reference_sq": float(reference_matrix.square().sum()),
            "residuals": {key: torch.cat(value, dim=0) for key, value in residuals.items()},
            "masses": {key: sum(value) / len(value) for key, value in masses.items()},
            "masks": {key: torch.stack(value) for key, value in masks.items()},
        }
        print(
            f"[dynamic-oracle] sample={metadata.get('sample_id')} head={head} "
            f"tiles={len(query_starts)} blocks={key_blocks}",
            flush=True,
        )
        del q, k, v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del payload, q_all, k_all, v_all
    return {
        "metadata": metadata,
        "path": str(path),
        "tokens": tokens,
        "heads": result_heads,
        "query_starts": query_starts,
        "key_blocks": key_blocks,
        "budgets": budgets,
    }


def calibration_masks(result: dict[str, object]) -> dict[tuple[int, str, float], torch.Tensor]:
    masks: dict[tuple[int, str, float], torch.Tensor] = {}
    for head, record in result["heads"].items():
        for (method, density), value in record["masks"].items():
            masks[(head, method, density)] = value
    return masks


def relative_error_sq(residual: torch.Tensor, reference_sq: float) -> float:
    return float(residual.square().sum() / max(reference_sq, 1e-30))


def build_detail_rows(
    sample: dict[str, object],
    split: str,
    ranks: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata = sample["metadata"]
    for head, record in sample["heads"].items():
        reference_sq = float(record["reference_sq"])
        for key, defect in record["residuals"].items():
            method, density, normalization, route = key
            bases = right_singular_bases(defect, ranks)
            for rank in ranks:
                basis, energy = bases[rank]
                final = residual_after_basis(defect, basis)
                rows.append(
                    {
                        "split": split,
                        "sample_id": metadata.get("sample_id", ""),
                        "seed": metadata.get("seed", ""),
                        "head": head,
                        "method": method,
                        "density": density,
                        "normalization": normalization,
                        "route": route,
                        "rank": rank,
                        "attention_mass": record["masses"][(method, density, route)],
                        "tail_energy_explained": energy,
                        "critical_output_relative_l2": math.sqrt(relative_error_sq(defect, reference_sq)),
                        "final_output_relative_l2": math.sqrt(relative_error_sq(final, reference_sq)),
                        "final_residual_sq": float(final.square().sum()),
                        "reference_sq": reference_sq,
                    }
                )
    return rows


def build_transfer_rows(
    calibration: dict[str, object],
    test: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, calibration_record in calibration["heads"].items():
        test_record = test["heads"][head]
        reference_sq = float(test_record["reference_sq"])
        for method in args.methods:
            for density in args.densities:
                jaccard = mask_jaccard(
                    calibration_record["masks"][(method, density)],
                    test_record["masks"][(method, density)],
                )
                for normalization in ("dense_probability", "renormalized"):
                    calibration_defect = calibration_record["residuals"][
                        (method, density, normalization, "oracle_route")
                    ]
                    test_oracle = test_record["residuals"][
                        (method, density, normalization, "oracle_route")
                    ]
                    test_frozen = test_record["residuals"][
                        (method, density, normalization, "frozen_route")
                    ]
                    calibration_bases = right_singular_bases(calibration_defect, args.ranks)
                    test_bases = right_singular_bases(test_oracle, args.ranks)
                    frozen_route_bases = right_singular_bases(test_frozen, args.ranks)
                    for rank in args.ranks:
                        calibration_basis, calibration_energy = calibration_bases[rank]
                        test_basis, test_energy = test_bases[rank]
                        frozen_route_basis, frozen_route_energy = frozen_route_bases[rank]
                        oracle_self = residual_after_basis(test_oracle, test_basis)
                        oracle_calibration = residual_after_basis(
                            test_oracle, calibration_basis
                        )
                        frozen_self = residual_after_basis(test_frozen, frozen_route_basis)
                        frozen_calibration = residual_after_basis(
                            test_frozen, calibration_basis
                        )
                        rows.append(
                            {
                                "head": head,
                                "method": method,
                                "density": density,
                                "normalization": normalization,
                                "rank": rank,
                                "mask_jaccard": jaccard,
                                "calibration_tail_energy": calibration_energy,
                                "test_oracle_tail_energy": test_energy,
                                "test_frozen_route_tail_energy": frozen_route_energy,
                                "basis_overlap": subspace_overlap(
                                    calibration_basis, test_basis
                                ),
                                "test_oracle_attention_mass": test_record["masses"][
                                    (method, density, "oracle_route")
                                ],
                                "test_frozen_attention_mass": test_record["masses"][
                                    (method, density, "frozen_route")
                                ],
                                "test_oracle_critical_residual_sq": float(test_oracle.square().sum()),
                                "test_frozen_critical_residual_sq": float(test_frozen.square().sum()),
                                "test_oracle_self_residual_sq": float(oracle_self.square().sum()),
                                "test_oracle_calibration_basis_residual_sq": float(
                                    oracle_calibration.square().sum()
                                ),
                                "test_frozen_self_residual_sq": float(frozen_self.square().sum()),
                                "test_frozen_calibration_basis_residual_sq": float(
                                    frozen_calibration.square().sum()
                                ),
                                "reference_sq": reference_sq,
                            }
                        )
    return rows


def aggregate_transfer(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["density"]), str(row["normalization"]), int(row["rank"]))].append(row)
    summary: list[dict[str, object]] = []
    residual_fields = (
        "test_oracle_critical_residual_sq",
        "test_frozen_critical_residual_sq",
        "test_oracle_self_residual_sq",
        "test_oracle_calibration_basis_residual_sq",
        "test_frozen_self_residual_sq",
        "test_frozen_calibration_basis_residual_sq",
    )
    for (method, density, normalization, rank), group in sorted(grouped.items()):
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        aggregate = {
            field.replace("_residual_sq", "_relative_l2"): math.sqrt(
                sum(float(row[field]) for row in group) / max(reference_sq, 1e-30)
            )
            for field in residual_fields
        }
        per_sample_error = aggregate["test_oracle_self_relative_l2"]
        static_error = aggregate["test_frozen_calibration_basis_relative_l2"]
        summary.append(
            {
                "method": method,
                "density": density,
                "normalization": normalization,
                "rank": rank,
                "heads": len(group),
                "mask_jaccard_mean": sum(float(row["mask_jaccard"]) for row in group) / len(group),
                "basis_overlap_mean": sum(float(row["basis_overlap"]) for row in group) / len(group),
                "test_oracle_tail_energy_mean": sum(float(row["test_oracle_tail_energy"]) for row in group) / len(group),
                "test_oracle_attention_mass_mean": sum(float(row["test_oracle_attention_mass"]) for row in group) / len(group),
                "test_frozen_attention_mass_mean": sum(float(row["test_frozen_attention_mass"]) for row in group) / len(group),
                **aggregate,
                "per_sample_oracle_go": per_sample_error <= args.error_target,
                "frozen_static_upper_bound_go": static_error <= args.error_target,
            }
        )
    return summary


def make_decision(summary: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    eligible = [
        row
        for row in summary
        if row["normalization"] == "renormalized"
        and float(row["density"]) <= args.max_deployable_density
        and int(row["rank"]) <= args.max_deployable_rank
    ]
    best_per_sample = min(
        eligible, key=lambda row: float(row["test_oracle_self_relative_l2"])
    )
    best_static = min(
        eligible,
        key=lambda row: float(row["test_frozen_calibration_basis_relative_l2"]),
    )
    best_frozen_route = min(
        eligible, key=lambda row: float(row["test_frozen_self_relative_l2"])
    )
    best_frozen_basis = min(
        eligible,
        key=lambda row: float(row["test_oracle_calibration_basis_relative_l2"]),
    )
    per_sample_error = float(best_per_sample["test_oracle_self_relative_l2"])
    static_error = float(best_static["test_frozen_calibration_basis_relative_l2"])
    if per_sample_error > args.error_target:
        verdict = "HEURISTIC_NO_GO_REQUIRES_STRONGER_SEARCH"
    elif static_error > args.error_target:
        verdict = "GO_CONDITIONAL_MASK_OR_BASIS_REQUIRED"
    else:
        verdict = "GO_FROZEN_STRUCTURE_COEFFICIENT_MODEL_REQUIRED"
    return {
        "verdict": verdict,
        "error_target": args.error_target,
        "budget": {
            "max_density": args.max_deployable_density,
            "max_rank": args.max_deployable_rank,
        },
        "best_per_sample_configuration": best_per_sample,
        "best_frozen_structure_configuration": best_static,
        "best_frozen_route_configuration": best_frozen_route,
        "best_frozen_basis_configuration": best_frozen_basis,
        "interpretation": {
            "per_sample_oracle": "dynamic mask plus best sample-specific output basis",
            "search_warning": "block subset selection is greedy rather than a certified global optimum; failure is not a proof of impossibility",
            "static_upper_bound": "calibration mask plus calibration basis, with oracle basis coefficients",
            "dense_probability_warning": "uses the all-key softmax denominator and is representation-only",
            "renormalized_warning": "numerically sparse-compatible but mask routing and tail coefficients remain oracle",
            "latency_warning": "density is not measured H200 latency; no speedup is claimed without a fused kernel",
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.calibration_sample_id == args.test_sample_id:
        raise ValueError("calibration and test sample IDs must differ")
    if any(density <= 0 or density > 1 for density in args.densities):
        raise ValueError("densities must be in (0, 1]")
    if any(rank < 0 for rank in args.ranks):
        raise ValueError("ranks must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started = time.time()
    calibration_path = resolve_capture(
        args.capture_index, args.calibration_sample_id, args.branch, args.layer, args.sampling_step
    )
    test_path = resolve_capture(
        args.capture_index, args.test_sample_id, args.branch, args.layer, args.sampling_step
    )
    calibration = process_capture(calibration_path, args)
    test = process_capture(test_path, args, calibration_masks(calibration))
    detail_rows = build_detail_rows(calibration, "calibration", args.ranks)
    detail_rows.extend(build_detail_rows(test, "test", args.ranks))
    transfer_rows = build_transfer_rows(calibration, test, args)
    summary_rows = aggregate_transfer(transfer_rows, args)
    decision = make_decision(summary_rows, args)
    write_csv(args.output_dir / "dynamic_sparse_lowrank_detail.csv", detail_rows)
    write_csv(args.output_dir / "dynamic_sparse_lowrank_transfer.csv", transfer_rows)
    write_csv(args.output_dir / "dynamic_sparse_lowrank_summary.csv", summary_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "calibration_path": str(calibration_path),
        "test_path": str(test_path),
        "query_starts": calibration["query_starts"],
        "key_blocks": calibration["key_blocks"],
        "budgets": {str(key): value for key, value in calibration["budgets"].items()},
        "methodology": {
            "critical_granularity": "one shared contiguous key-block mask per contiguous query tile and head",
            "objective": "pre-output-projection AV relative L2, not attention-probability Frobenius error",
            "tail": "best output-channel rank-r projection; coefficients are oracle-only",
            "transfer": "calibration mask and/or right singular basis frozen on a different-seed replay",
            "limitations": [
                "only captured layer/step/CFG cells are represented",
                "only two bit-distinct step-0 contents currently exist",
                "no quantization is included in this first sparse-plus-tail oracle",
                "no end-to-end trajectory or fused-kernel timing is inferred",
            ],
        },
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[dynamic-oracle] verdict={decision['verdict']}", flush=True)
    print(f"[dynamic-oracle] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
