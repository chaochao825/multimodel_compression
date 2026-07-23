#!/usr/bin/env python3
"""Create a compact evidence dashboard for the tri-mode Oracle probes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def evidence_path(root: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"none of the evidence paths exist under {root}: {relative_paths}"
    )


def best(rows: list[dict[str, str]], action: str | None = None) -> float:
    selected = rows if action is None else [row for row in rows if row["action"] == action]
    return max(float(row["ssim_min"]) for row in selected)


def main() -> None:
    args = parse_args()
    root = args.results_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    coarse = read_csv(
        evidence_path(
            root,
            "tri_mode_oracle_f17_single_cells_combined_v1/tri_mode_cell_metrics.csv",
            "coarse_cells/tri_mode_cell_metrics.csv",
        )
    )
    fine = read_csv(
        evidence_path(
            root,
            "tri_mode_oracle_f17_cache_refine_combined_v1/tri_mode_cell_metrics.csv",
            "cache_refine/tri_mode_cell_metrics.csv",
        )
    )
    branch = read_csv(
        evidence_path(
            root,
            "tri_mode_oracle_f17_cache_branch_probe_combined_v1/tri_mode_cell_metrics.csv",
            "branch_probe/tri_mode_cell_metrics.csv",
        )
    )
    forecast = read_csv(
        evidence_path(
            root,
            "tri_mode_oracle_f17_forecast_probe_combined_v1/tri_mode_cell_metrics.csv",
            "forecast_probe/tri_mode_cell_metrics.csv",
        )
    )
    anchor = read_csv(
        evidence_path(
            root,
            "tri_mode_global_anchor_serial_v1/analysis/method_summary.csv",
            "global_anchor/method_summary.csv",
        )
    )
    defect = read_csv(
        evidence_path(
            root,
            "activation_defect_subspace_f17_v1/analysis/defect_spectrum_summary.csv",
            "activation_defect/defect_spectrum_summary.csv",
        )
    )
    multisample = read_csv(
        evidence_path(
            root,
            "tri_mode_oracle_f17_multisample_gate_v1/analysis/method_summary.csv",
            "multisample_gate/method_summary.csv",
        )
    )

    granularity = [
        {"probe": "Q 4x6 both", "best_ssim": best(coarse, "Q")},
        {"probe": "C 4x6 both", "best_ssim": best(coarse, "C")},
        {"probe": "C 1x3 both", "best_ssim": best(fine)},
        {"probe": "C 1x1 one CFG", "best_ssim": best(branch)},
        {"probe": "forecast 1x1 one CFG", "best_ssim": best(forecast)},
        {
            "probe": "forecast 1x1, 4-sample worst",
            "best_ssim": max(
                float(row["ssim_min"])
                for row in multisample
                if row["method"] != "dense"
            ),
        },
    ]
    write_csv(out / "granularity_best_ssim.csv", granularity)

    temporal_rows: list[dict[str, object]] = []
    for label, rows, action in (("Q 4x6", coarse, "Q"), ("C 4x6", coarse, "C"), ("C 1x3", fine, "C")):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row["action"] == action:
                grouped[int(row["step_start"])].append(float(row["ssim_min"]))
        for step, values in sorted(grouped.items()):
            temporal_rows.append(
                {
                    "probe": label,
                    "step": step,
                    "max_ssim": max(values),
                    "mean_ssim": float(np.mean(values)),
                }
            )
    write_csv(out / "temporal_sensitivity.csv", temporal_rows)

    anchor_rows = [
        {
            "method": row["method"],
            "speedup": float(row["speedup_geomean"]),
            "ssim": float(row["ssim_min"]),
            "samples": int(row["samples"]),
        }
        for row in anchor
        if row["method"] in {"dense", "all_q", "all_c"}
    ]
    write_csv(out / "global_anchor_tradeoff.csv", anchor_rows)

    defect_rows = [
        {
            "operator": row["operator"],
            "rank16_energy": float(row["energy_rank_16"]),
            "rank_at_90pct": int(row["rank_at_90pct"]),
        }
        for row in defect
        if row["group"] == "all_blocks"
    ]
    write_csv(out / "defect_rank16.csv", defect_rows)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = [
        "#466c8a",
        "#c44e38",
        "#d18343",
        "#698f3f",
        "#16817a",
        "#2a9d8f",
    ]

    axis = axes[0, 0]
    labels = [row["probe"] for row in granularity]
    values = [float(row["best_ssim"]) for row in granularity]
    bars = axis.barh(labels, values, color=colors)
    axis.axvline(0.98, color="#202020", linestyle="--", linewidth=1.4, label="SSIM 0.98 gate")
    axis.set_xlim(min(values) - 0.01, 1.001)
    axis.set_xlabel("best measured paired SSIM")
    axis.set_title("A. No tested action granularity clears 0.98")
    for bar, value in zip(bars, values, strict=True):
        axis.text(value + 0.0004, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8)
    axis.legend(fontsize=8, loc="lower right")

    axis = axes[0, 1]
    for label in ("Q 4x6", "C 4x6", "C 1x3"):
        rows = [row for row in temporal_rows if row["probe"] == label]
        axis.plot(
            [int(row["step"]) for row in rows],
            [float(row["max_ssim"]) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.axhline(0.98, color="#202020", linestyle="--", linewidth=1.2)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel("best SSIM across block groups")
    axis.set_title("B. Strong non-stationarity, but no safe late window")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for color, row in zip(("#202020", "#466c8a", "#c44e38"), anchor_rows, strict=True):
        axis.scatter(row["speedup"], row["ssim"], s=95, color=color)
        axis.annotate(row["method"], (row["speedup"], row["ssim"]), xytext=(5, 5), textcoords="offset points")
    axis.axhline(0.98, color="#202020", linestyle="--", linewidth=1.2)
    axis.axvline(1.2, color="#6b705c", linestyle=":", linewidth=1.2)
    axis.set_xlabel("serial H200 end-to-end speedup")
    axis.set_ylabel("paired SSIM")
    axis.set_title("C. Available speed leverage is quality-infeasible")
    axis.grid(alpha=0.2)

    axis = axes[1, 1]
    labels = [row["operator"] for row in defect_rows]
    values = [float(row["rank16_energy"]) for row in defect_rows]
    bars = axis.barh(labels, values, color="#16817a")
    axis.axvline(0.9, color="#202020", linestyle="--", linewidth=1.2, label="90% energy")
    axis.set_xlim(0, 1)
    axis.set_xlabel("rank-16 explained defect energy")
    axis.set_title("D. Uniform rank-16 correction is not supported")
    for bar, value in zip(bars, values, strict=True):
        axis.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    axis.legend(fontsize=8, loc="lower right")

    figure.suptitle("Trajectory-budgeted tri-mode runtime: measured feasibility audit", fontsize=16, y=0.995)
    figure.tight_layout()
    figure.savefig(out / "tri_mode_evidence_dashboard.png", dpi=200, bbox_inches="tight")
    figure.savefig(out / "tri_mode_evidence_dashboard.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
