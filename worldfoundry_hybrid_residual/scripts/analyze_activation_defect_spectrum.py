#!/usr/bin/env python3
"""Analyze token-sampled Q/cache activation-defect covariance spectra."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def operator_name(record: dict[str, object]) -> str:
    kind = str(record["kind"])
    if kind != "C_FORECAST":
        return kind
    return f"C_FORECAST_{float(record['forecast_scale']):g}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def covariance_spectrum(
    records: list[dict[str, object]], device: torch.device
) -> tuple[torch.Tensor, float, int]:
    features = int(records[0]["features"])
    covariance = torch.zeros((features, features), device=device, dtype=torch.float32)
    rows = 0
    energy = 0.0
    for record in records:
        sample = record["sample"]
        if not isinstance(sample, torch.Tensor):
            raise TypeError("defect sample must be a tensor")
        matrix = sample.to(device=device, dtype=torch.float32)
        covariance.addmm_(matrix.T, matrix)
        rows += matrix.shape[0]
        energy += float(matrix.square().sum().item())
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min_(0).flip(0)
    return eigenvalues.cpu(), energy, rows


def effective_rank(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    return float(math.exp(float(-np.sum(values * np.log(values)))))


def rank_for_ratio(cumulative: np.ndarray, target: float) -> int:
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = payload["records"]
    if not records:
        raise ValueError("sample file contains no defect records")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        operator = operator_name(record)
        grouped[(operator, f"block_{int(record['block']):02d}")].append(record)
        grouped[(operator, "all_blocks")].append(record)

    spectrum_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (operator, group), values in sorted(grouped.items()):
        eigenvalues, energy, sampled_rows = covariance_spectrum(values, device)
        values_np = eigenvalues.numpy().astype(np.float64)
        total = float(values_np.sum())
        if total <= 0:
            probabilities = np.zeros_like(values_np)
            cumulative = np.zeros_like(values_np)
        else:
            probabilities = values_np / total
            cumulative = np.cumsum(probabilities)
        curves[(operator, group)] = (np.arange(1, len(cumulative) + 1), cumulative)
        summary: dict[str, object] = {
            "operator": operator,
            "group": group,
            "records": len(values),
            "sampled_rows": sampled_rows,
            "features": len(values_np),
            "sample_energy_accumulated": energy,
            "covariance_trace": total,
            "effective_rank_entropy": effective_rank(probabilities),
            "rank_at_90pct": rank_for_ratio(cumulative, 0.90),
            "rank_at_95pct": rank_for_ratio(cumulative, 0.95),
            "rank_at_99pct": rank_for_ratio(cumulative, 0.99),
        }
        for rank in RANKS:
            index = min(rank, len(cumulative)) - 1
            ratio = float(cumulative[index]) if cumulative.size else 0.0
            summary[f"energy_rank_{rank}"] = ratio
            summary[f"relative_fro_after_rank_{rank}"] = math.sqrt(max(0.0, 1.0 - ratio))
            spectrum_rows.append(
                {
                    "operator": operator,
                    "group": group,
                    "rank": rank,
                    "energy_ratio": ratio,
                    "relative_fro_residual": math.sqrt(max(0.0, 1.0 - ratio)),
                }
            )
        summary_rows.append(summary)
        print(
            f"SPECTRUM operator={operator:16s} group={group:10s} "
            f"r8={summary['energy_rank_8']:.4f} "
            f"r16={summary['energy_rank_16']:.4f} "
            f"r90={summary['rank_at_90pct']}",
            flush=True,
        )

    write_csv(out_dir / "defect_spectrum.csv", spectrum_rows)
    write_csv(out_dir / "defect_spectrum_summary.csv", summary_rows)

    import matplotlib.pyplot as plt

    all_operators = sorted({operator for operator, group in curves if group == "all_blocks"})
    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(all_operators))))
    for color, operator in zip(palette, all_operators, strict=True):
        ranks, cumulative = curves[(operator, "all_blocks")]
        limit = min(128, len(ranks))
        axis.plot(ranks[:limit], cumulative[:limit], label=operator, color=color, linewidth=2)
    for rank in (8, 16, 32, 64):
        axis.axvline(rank, color="#aaaaaa", linewidth=0.7, alpha=0.35)
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xscale("log", base=2)
    axis.set_xlim(1, 128)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("defect basis rank")
    axis.set_ylabel("explained sampled defect energy")
    axis.set_title("Wan activation-defect subspace spectrum (all probed blocks)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "defect_spectrum_all_blocks.png", dpi=190)
    plt.close(figure)

    blocks = sorted({group for _, group in curves if group.startswith("block_")})
    matrix = np.full((len(all_operators), len(blocks)), np.nan)
    lookup = {(row["operator"], row["group"]): row for row in summary_rows}
    for operator_index, operator in enumerate(all_operators):
        for block_index, block in enumerate(blocks):
            row = lookup.get((operator, block))
            if row is not None:
                matrix[operator_index, block_index] = float(row["energy_rank_16"])
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(range(len(blocks)), blocks)
    axis.set_yticks(range(len(all_operators)), all_operators)
    axis.set_title("Rank-16 explained defect energy")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(out_dir / "defect_rank16_heatmap.png", dpi=190)
    plt.close(figure)

    decision = {
        "scope": "token-row sampled activation defects on an unchanged dense trajectory",
        "records": len(records),
        "operators": all_operators,
        "blocks": blocks,
        "ranks": list(RANKS),
        "interpretation_gate": {
            "retain_low_rank_correction": "rank-8/16 should explain a large majority of weighted defect energy and coefficients must be cheaply predictable",
            "reject_low_rank_correction": "flat spectrum or correction compute approaching dense block cost",
        },
    }
    (out_dir / "defect_spectrum_manifest.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
