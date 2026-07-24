#!/usr/bin/env python3
"""Plot the Wan attention bottleneck and low-rank/sparse oracle audit."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    SCRIPT_ROOT
    if (SCRIPT_ROOT / "raw").exists()
    else SCRIPT_ROOT / "results" / "attention_lowrank_audit_v1"
)
RAW = ROOT / "raw"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "data"

COLORS = {
    "self_attention_core": "#C14924",
    "cross_attention_core": "#E08E45",
    "linear_gemm": "#287271",
    "elementwise_memory": "#6C91BF",
    "normalization": "#9B7EBD",
    "other": "#9B9B93",
    "lowrank": "#A44A3F",
    "topk_sparse": "#3B7A57",
    "topk_plus_lowrank": "#245A8D",
}


def classify_kernel(name: str) -> str:
    token = name.lower()
    if "flash::compute_attn_ws" in token:
        return "self_attention_core"
    if "pytorch_flash::flash_fwd" in token:
        return "cross_attention_core"
    if "nvjet" in token or "cublas" in token or "cutlass_3x" in token:
        return "linear_gemm"
    if "layer_norm" in token or "rms" in token:
        return "normalization"
    if "cudnn" in token or "fmha_cutlass" in token or "nhwc" in token:
        return "vae_decode"
    if any(
        marker in token
        for marker in ("elementwise", "copy", "catarray", "_cat_pad")
    ):
        return "elementwise_memory"
    return "other"


def read_nsys_categories(path: Path) -> dict[str, int]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("Generating ", "Processing ["))
    ]
    totals: dict[str, int] = defaultdict(int)
    for row in csv.DictReader(lines):
        try:
            total_ns = int(row["Total Time (ns)"])
        except (KeyError, TypeError, ValueError):
            continue
        totals[classify_kernel(row["Name"])] += total_ns
    return dict(totals)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def profile_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in ("f17", "f81"):
        one = read_nsys_categories(RAW / f"{case}_1step_cuda_gpu_kern_sum.csv")
        twenty = read_nsys_categories(RAW / f"{case}_20step_cuda_gpu_kern_sum.csv")
        delta = {key: twenty.get(key, 0) - one.get(key, 0) for key in set(one) | set(twenty)}
        # Both traces contain one warmup generation and one VAE decode per timed
        # generation. Their difference therefore isolates 19 denoising steps.
        delta["vae_decode"] = 0
        kept = {
            key: max(value, 0)
            for key, value in delta.items()
            if key != "vae_decode"
        }
        total = sum(kept.values())
        for component, value in sorted(kept.items()):
            rows.append(
                {
                    "case": case.upper(),
                    "component": component,
                    "delta_time_ms": value / 1e6,
                    "per_denoise_step_ms": value / 19e6,
                    "share_percent": 100.0 * value / total,
                }
            )
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_latency(path: Path, method: str) -> float:
    row = next(row for row in read_rows(path) if row["method"] == method)
    return float(row["milliseconds"])


def speedup_rows() -> list[dict[str, object]]:
    f17 = RAW / "attention_f17.csv"
    f81 = RAW / "attention_f81.csv"
    f17_base = method_latency(f17, "fa3_bf16")
    f81_base = method_latency(f81, "fa3_bf16")
    rows = [
        {
            "label": "F17 FA3 FP8 attention",
            "family": "attention",
            "speedup": f17_base / method_latency(f17, "fa3_fp8"),
        },
        {
            "label": "F17 SageAttention",
            "family": "attention",
            "speedup": f17_base / method_latency(f17, "sage_sm90_no_smooth"),
        },
        {
            "label": "F81 FA3 FP8 attention",
            "family": "attention",
            "speedup": f81_base / method_latency(f81, "fa3_fp8"),
        },
    ]

    wf_rows = read_rows(RAW / "worldfoundry_linear_benchmark.csv")
    for shape, label in (
        ("f17_qkv", "F17 WorldFoundry dynamic FP8 QKV"),
        ("f17_ffn_down", "F17 WorldFoundry dynamic FP8 FFN-down"),
    ):
        row = next(
            row
            for row in wf_rows
            if row["shape"] == shape and row["method"] == "worldfoundry_fp8"
        )
        rows.append(
            {"label": label, "family": "linear", "speedup": float(row["speedup_vs_bf16"])}
        )

    h200_rows = read_rows(RAW / "h200_bcm_cached.csv")
    static = next(
        row
        for row in h200_rows
        if row["rows"] == "7800" and row["method"] == "fp8_static_input_main"
    )
    residual = next(
        row
        for row in h200_rows
        if row["rows"] == "7800" and row["method"] == "fp8_static_input_svd_r16"
    )
    rows.extend(
        [
            {
                "label": "F17 static-scale FP8 Q lower bound",
                "family": "linear",
                "speedup": float(static["speedup_vs_bf16"]),
            },
            {
                "label": "F17 static FP8 + rank-16 Q",
                "family": "linear",
                "speedup": float(residual["speedup_vs_bf16"]),
            },
        ]
    )
    return rows


def pareto_front(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: float(row["representation_ratio_mean"]))
    frontier: list[dict[str, str]] = []
    best_error = float("inf")
    for row in ordered:
        error = float(row["output_rel_l2_mean"])
        if error < best_error:
            frontier.append(row)
            best_error = error
    return frontier


def plot_oracle(ax: plt.Axes, case: str, summary: list[dict[str, str]]) -> None:
    for method, marker, label in (
        ("lowrank", "o", "Pure low rank"),
        ("topk_sparse", "s", "Oracle top-k sparse"),
        ("topk_plus_lowrank", "D", "Top-k + low-rank residual"),
    ):
        selected = [
            row for row in summary if row["case"] == case and row["method"] == method
        ]
        selected = pareto_front(selected)
        x = [100.0 * float(row["representation_ratio_mean"]) for row in selected]
        y = [100.0 * float(row["output_rel_l2_mean"]) for row in selected]
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            color=COLORS[method],
            label=label,
        )
    ax.set_xlabel("Oracle representation budget (%)")
    ax.set_ylabel("Attention output relative L2 (%)")
    ax.grid(alpha=0.22, linewidth=0.7)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    profiles = profile_rows()
    speedups = speedup_rows()
    write_csv(RESULTS / "profile_component_shares.csv", profiles)
    write_csv(RESULTS / "operator_speedups.csv", speedups)

    summary = read_rows(RAW / "attention_lowrank_sparse_oracle_v1" / "oracle_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    order = [
        "self_attention_core",
        "cross_attention_core",
        "linear_gemm",
        "elementwise_memory",
        "normalization",
        "other",
    ]
    x = np.arange(2)
    bottom = np.zeros(2)
    for component in order:
        values = [
            next(
                (float(row["share_percent"]) for row in profiles if row["case"] == case and row["component"] == component),
                0.0,
            )
            for case in ("F17", "F81")
        ]
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.66,
            color=COLORS[component],
            label=component.replace("_", " "),
        )
        bottom += np.array(values)
    ax.set_xticks(x, ["F17 (7,800 tokens)", "F81 (32,760 tokens)"])
    ax.set_ylabel("Incremental denoiser GPU-kernel share (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    ax = axes[0, 1]
    labels = [row["label"] for row in speedups]
    values = [float(row["speedup"]) for row in speedups]
    colors = ["#287271" if row["family"] == "attention" else "#C14924" for row in speedups]
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Measured speedup over the matching BF16 operator")
    for pos, value in zip(positions, values):
        ax.text(value + 0.025, pos, f"{value:.2f}x", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)

    plot_oracle(axes[1, 0], "f17", summary)
    plot_oracle(axes[1, 1], "f81", summary)
    axes[1, 0].set_title("F17: 7,800-token real Wan attention", loc="left", fontsize=10)
    axes[1, 1].set_title("F81: 32,760-token real Wan attention", loc="left", fontsize=10)
    axes[1, 0].legend(frameon=False, fontsize=8)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold")

    png = FIGURES / "attention_lowrank_audit.png"
    pdf = FIGURES / "attention_lowrank_audit.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
