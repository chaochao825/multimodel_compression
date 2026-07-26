#!/usr/bin/env python3
"""Create a compact decision dashboard for the FFN and F81 attention probes."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "navy": "#18354C",
    "orange": "#D2693C",
    "teal": "#25806D",
    "gold": "#C89A2B",
    "red": "#B4443C",
    "gray": "#77828C",
    "cream": "#F4F0E6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-csv", type=Path, required=True)
    parser.add_argument("--structure-csv", type=Path, required=True)
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--h200-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, str]], keys: tuple[str, ...], value: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row[value]))
    output = []
    for group, values in sorted(groups.items()):
        output.append(
            {
                **dict(zip(keys, group)),
                "metric": value,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        )
    return output


def select_mean(
    rows: list[dict[str, str]], value: str, **filters: str | int | float
) -> tuple[float, float, int]:
    selected = []
    for row in rows:
        match = True
        for key, expected in filters.items():
            actual = row[key]
            if isinstance(expected, float):
                match &= math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
            else:
                match &= actual == str(expected)
        if match:
            selected.append(float(row[value]))
    return statistics.mean(selected), statistics.pstdev(selected), len(selected)


def h200_value(rows: list[dict[str, str]], row_count: int, operation: str, width: int | None = None) -> float:
    selected = [
        row
        for row in rows
        if row["rows"] == str(row_count)
        and row["operation"] == operation
        and (width is None or row["width"] == str(width))
        and row["status"] == "ok"
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one H200 row for {row_count=} {operation=} {width=}")
    return float(selected[0]["latency_ms_median"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectral = read_csv(args.spectral_csv)
    structure = read_csv(args.structure_csv)
    oracle = read_csv(args.oracle_csv)
    h200 = read_csv(args.h200_csv)

    spectral_summary = summarize(
        spectral,
        ("model", "role", "control", "method", "target_scalar_density"),
        "retained_energy",
    )
    structure_rows = []
    for metric in (
        "row_neighbor_cosine",
        "column_neighbor_cosine",
        "circulant_retained_energy",
    ):
        structure_rows.extend(summarize(structure, ("model", "role", "control"), metric))
    oracle_summary = summarize(
        oracle,
        ("method", "requested_token_budget", "actual_critical_tokens", "rank"),
        "output_rel_l2",
    )
    oracle_mass_summary = summarize(
        oracle,
        ("method", "requested_token_budget", "actual_critical_tokens", "rank"),
        "critical_mass",
    )
    write_csv(args.output_dir / "ffn_spectral_summary.csv", spectral_summary)
    write_csv(args.output_dir / "ffn_structure_summary.csv", structure_rows)
    write_csv(args.output_dir / "attention_block_tail_summary.csv", oracle_summary + oracle_mass_summary)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#263746",
            "axes.linewidth": 0.8,
            "xtick.color": "#263746",
            "ytick.color": "#263746",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), constrained_layout=True)
    figure.patch.set_facecolor(COLORS["cream"])
    for axis in axes.flat:
        axis.set_facecolor("#FCFBF7")
        axis.grid(True, color="#D9D6CC", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)

    densities = [0.015625, 0.0625, 0.125, 0.25]
    axis = axes[0, 0]
    for model, label, color, marker in (
        ("wan2.1-1.3b", "Wan FFN", COLORS["orange"], "o"),
        ("llama2-7b", "Llama FFN", COLORS["navy"], "s"),
    ):
        means, stds = [], []
        for density in densities:
            mean, std, _ = select_mean(
                spectral,
                "retained_energy",
                model=model,
                control="original",
                method="fft2_lowfreq_static",
                target_scalar_density=density,
            )
            means.append(mean)
            stds.append(std)
        axis.errorbar(
            np.asarray(densities) * 100,
            np.asarray(means) * 100,
            yerr=np.asarray(stds) * 100,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=2,
        )
    axis.plot(np.asarray(densities) * 100, np.asarray(densities) * 100, "--", color=COLORS["gray"], label="white-noise expectation")
    axis.set_title("A. Static 2D low frequencies carry no excess energy")
    axis.set_xlabel("Real-scalar budget (%)")
    axis.set_ylabel("Retained weight energy (%)")
    axis.legend(frameon=False, fontsize=8.5)

    axis = axes[0, 1]
    for method, label, color, marker in (
        ("identity_topk_oracle", "Weight top-k oracle", COLORS["orange"], "o"),
        ("fft2_topk_oracle", "FFT top-k oracle", COLORS["teal"], "s"),
        ("fft2_lowfreq_static", "FFT static low-pass", COLORS["navy"], "^"),
    ):
        means = [
            select_mean(
                spectral,
                "retained_energy",
                model="wan2.1-1.3b",
                control="original",
                method=method,
                target_scalar_density=density,
            )[0]
            for density in densities
        ]
        axis.plot(np.asarray(densities) * 100, np.asarray(means) * 100, marker=marker, color=color, linewidth=2, label=label)
    axis.set_title("B. FFT oracle gain is irregular and not kernel-ready")
    axis.set_xlabel("Real-scalar budget (%)")
    axis.set_ylabel("Wan retained energy (%)")
    axis.legend(frameon=False, fontsize=8.3)

    axis = axes[0, 2]
    controls = ["original", "rowcol_shuffled", "matched_gaussian"]
    control_labels = ["Original", "Channel-shuffled", "Gaussian"]
    x = np.arange(len(controls))
    width = 0.34
    for offset, model, label, color in (
        (-width / 2, "wan2.1-1.3b", "Wan", COLORS["orange"]),
        (width / 2, "llama2-7b", "Llama", COLORS["navy"]),
    ):
        values = [
            select_mean(
                spectral,
                "retained_energy",
                model=model,
                control=control,
                method="fft2_topk_oracle",
                target_scalar_density=0.125,
            )[0]
            for control in controls
        ]
        axis.bar(x + offset, np.asarray(values) * 100, width, label=label, color=color)
    axis.set_xticks(x, control_labels, rotation=12)
    axis.set_ylim(35, 41)
    axis.set_ylabel("FFT top-k retained energy (%)")
    axis.set_title("C. 12.5% FFT top-k survives shuffling because it is generic")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    method_specs = (
        ("token_topk_oracle", "Token oracle", COLORS["orange"], "o"),
        ("block16_oracle", "Block 16", COLORS["teal"], "s"),
        ("block32_oracle", "Block 32", COLORS["gold"], "^"),
        ("block64_oracle", "Block 64", COLORS["red"], "D"),
        ("block128_oracle", "Block 128", COLORS["navy"], "v"),
    )
    token_budgets = [64, 128, 256, 512, 1024]
    for method, label, color, marker in method_specs:
        means, maxima = [], []
        for token_budget in token_budgets:
            values = [
                float(row["output_rel_l2"])
                for row in oracle
                if row["method"] == method
                and row["requested_token_budget"] == str(token_budget)
                and row["rank"] == "16"
            ]
            means.append(statistics.mean(values))
            maxima.append(max(values))
        axis.plot(token_budgets, np.asarray(means) * 100, marker=marker, color=color, linewidth=1.9, label=label)
        axis.fill_between(token_budgets, np.asarray(means) * 100, np.asarray(maxima) * 100, color=color, alpha=0.08)
    axis.axhline(10, color=COLORS["gray"], linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(token_budgets, [str(item) for item in token_budgets])
    axis.set_xlabel("Critical tokens per query")
    axis.set_ylabel("F81 output relative L2 (%)")
    axis.set_title("D. GPU tiles need far more budget than token top-k")
    axis.legend(frameon=False, fontsize=7.7, ncol=2)

    axis = axes[1, 1]
    labels, full_ffn, fft_floor = [], [], []
    for row_count, label in ((7800, "F17"), (32760, "F81")):
        labels.append(label)
        full_ffn.append(h200_value(h200, row_count, "ffn_full_eager"))
        fft_floor.append(
            h200_value(h200, row_count, "rfft_fp32_roundtrip", 1536)
            + h200_value(h200, row_count, "rfft_fp32_roundtrip", 8960)
        )
    x = np.arange(2)
    axis.bar(x - 0.18, full_ffn, 0.36, color=COLORS["navy"], label="Complete BF16 FFN")
    axis.bar(x + 0.18, fft_floor, 0.36, color=COLORS["red"], label="Input + output FFT only")
    axis.set_xticks(x, labels)
    axis.set_ylabel("H200 latency (ms)")
    axis.set_title("E. Online FFT loses before spectral multiplication")
    axis.legend(frameon=False, fontsize=8.3)
    for index in range(2):
        axis.text(index + 0.18, fft_floor[index] + 0.08, f"{fft_floor[index] / full_ffn[index]:.2f}x FFN", ha="center", fontsize=8, color=COLORS["red"])

    axis = axes[1, 2]
    row_controls = ["original", "rowcol_shuffled", "matched_gaussian"]
    x = np.arange(len(row_controls))
    width = 0.34
    for offset, model, label, color in (
        (-width / 2, "wan2.1-1.3b", "Wan", COLORS["orange"]),
        (width / 2, "llama2-7b", "Llama", COLORS["navy"]),
    ):
        values = [
            select_mean(
                structure,
                "circulant_retained_energy",
                model=model,
                control=control,
            )[0]
            for control in row_controls
        ]
        axis.bar(x + offset, np.asarray(values) * 100, width, color=color, label=label)
    axis.axhline(100 / 64, color=COLORS["gray"], linestyle="--", label="random 1/64")
    axis.set_xticks(x, control_labels, rotation=12)
    axis.set_ylabel("Nearest-circulant energy (%)")
    axis.set_title("F. CM/BCM capture equals random subspace dimension")
    axis.legend(frameon=False, fontsize=8.3)

    figure.suptitle(
        "Wan FFN frequency audit and F81 sparse-tail deployability gate",
        fontsize=15,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.savefig(args.output_dir / "ffn_attention_decision_dashboard.png", dpi=210)
    figure.savefig(args.output_dir / "ffn_attention_decision_dashboard.pdf")
    plt.close(figure)

    decision_rows = [
        {
            "finding": "wan_fft2_lowfreq_12p5",
            "value": select_mean(
                spectral,
                "retained_energy",
                model="wan2.1-1.3b",
                control="original",
                method="fft2_lowfreq_static",
                target_scalar_density=0.125,
            )[0],
            "criterion": "must materially exceed scalar density and shuffled/Gaussian controls",
            "decision": "fail",
        },
        {
            "finding": "wan_circulant_capture",
            "value": select_mean(
                structure,
                "circulant_retained_energy",
                model="wan2.1-1.3b",
                control="original",
            )[0],
            "criterion": "must materially exceed 1/block_size random expectation",
            "decision": "fail",
        },
        {
            "finding": "f17_fft_only_over_full_ffn",
            "value": fft_floor[0] / full_ffn[0],
            "criterion": "must be below one before adding spectral contraction",
            "decision": "fail",
        },
        {
            "finding": "f81_fft_only_over_full_ffn",
            "value": fft_floor[1] / full_ffn[1],
            "criterion": "must be below one before adding spectral contraction",
            "decision": "fail",
        },
    ]
    write_csv(args.output_dir / "decision_gates.csv", decision_rows)
    print(f"[plot] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
