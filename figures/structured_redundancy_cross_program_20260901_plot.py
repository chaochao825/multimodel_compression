from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "worldfoundry_hybrid_residual"
    / "results"
    / "wan_rcm_baseline_exp047_20260901"
)
FIGURE_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quality_rows() -> list[dict]:
    payload = load_json(
        RESULT_ROOT / "quality" / "evaluation_v1" / "vbench_summary.json"
    )
    rows = []
    for dimension, values in payload["dimensions"].items():
        for method, ratio in values["teacher_normalized"].items():
            rows.append(
                {
                    "dimension": dimension,
                    "method": method,
                    "teacher_normalized": ratio,
                }
            )
    return rows


def timing_rows() -> list[dict]:
    rows = []
    for method in ("teacher20", "native4", "rcm4"):
        payload = load_json(
            RESULT_ROOT
            / "timing"
            / "outputs"
            / "timing_f81"
            / method
            / "generation_manifest.json"
        )
        summary = payload["summary"]
        known = {
            "text": summary["median_text_seconds"],
            "denoiser": summary["median_denoiser_seconds"],
            "vae": summary["median_vae_seconds"],
            "serialization": summary["median_serialization_seconds"],
        }
        known["other"] = summary["median_warm_e2e_seconds"] - sum(known.values())
        for component, seconds in known.items():
            rows.append(
                {"method": method, "component": component, "seconds": seconds}
            )
    return rows


def understanding_rows() -> list[dict]:
    # Source: RESULT-EXP-050 and RESULT-EXP-051 under the frozen 0.5/1/2% gates.
    raw = {
        "EXP-050 fixed-state residual support": (6.759, 12.566, 19.383),
        "EXP-051 best independent": (4.019, 7.651, 10.536),
        "EXP-051 joint residual": (4.281, 13.642, 21.254),
    }
    thresholds = {"mean": 0.5, "p95": 1.0, "worst": 2.0}
    rows = []
    for method, values in raw.items():
        for metric, value in zip(("mean", "p95", "worst"), values, strict=True):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "error_percent": value,
                    "gate_percent": thresholds[metric],
                    "error_to_gate": value / thresholds[metric],
                }
            )
    return rows


def main() -> None:
    quality = quality_rows()
    timing = timing_rows()
    understanding = understanding_rows()
    write_csv(
        FIGURE_ROOT / "structured_redundancy_wan_quality_20260901.csv",
        ["dimension", "method", "teacher_normalized"],
        quality,
    )
    write_csv(
        FIGURE_ROOT / "structured_redundancy_wan_timing_20260901.csv",
        ["method", "component", "seconds"],
        timing,
    )
    write_csv(
        FIGURE_ROOT / "structured_redundancy_understanding_capacity_20260901.csv",
        ["method", "metric", "error_percent", "gate_percent", "error_to_gate"],
        understanding,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)

    dimensions = list(dict.fromkeys(row["dimension"] for row in quality))
    labels = [
        "subject",
        "background",
        "flicker",
        "motion",
        "dynamic",
        "aesthetic",
        "imaging",
        "text-video",
    ]
    x = np.arange(len(dimensions))
    width = 0.36
    method_style = {
        "native4": ("Native 4-step", "#D55E00"),
        "rcm4": ("rCM 4-step", "#0072B2"),
    }
    for offset, method in zip((-0.5, 0.5), method_style, strict=True):
        values = [
            next(
                row["teacher_normalized"]
                for row in quality
                if row["dimension"] == dimension and row["method"] == method
            )
            for dimension in dimensions
        ]
        label, color = method_style[method]
        axes[0].bar(x + offset * width, values, width, label=label, color=color)
    axes[0].axhline(1.0, color="#333333", linewidth=1.0, label="Teacher")
    axes[0].axhline(0.8, color="#009E73", linewidth=1.0, linestyle="--", label="Dim. floor")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylim(0.45, 1.10)
    axes[0].set_ylabel("Teacher-normalized score")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=12)

    methods = ("teacher20", "native4", "rcm4")
    method_labels = ("Teacher 20", "Native 4", "rCM 4")
    components = ("text", "denoiser", "vae", "serialization", "other")
    component_colors = {
        "text": "#56B4E9",
        "denoiser": "#0072B2",
        "vae": "#E69F00",
        "serialization": "#CC79A7",
        "other": "#999999",
    }
    bottom = np.zeros(len(methods))
    totals = np.zeros(len(methods))
    for component in components:
        values = np.array(
            [
                next(
                    row["seconds"]
                    for row in timing
                    if row["method"] == method and row["component"] == component
                )
                for method in methods
            ]
        )
        axes[1].bar(
            method_labels,
            values,
            bottom=bottom,
            color=component_colors[component],
            label=component,
        )
        bottom += values
        totals += values
    for index, total in enumerate(totals):
        axes[1].text(index, total + 1.1, f"{total:.1f}s", ha="center", fontsize=8)
    axes[1].set_ylim(0, max(totals) * 1.13)
    axes[1].set_ylabel("Warm end-to-end time (s)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].text(-0.12, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=12)

    understanding_methods = list(
        dict.fromkeys(row["method"] for row in understanding)
    )
    understanding_labels = ("Fixed-state\nresidual support", "Best\nindependent", "Joint\nresidual")
    ux = np.arange(len(understanding_methods))
    metric_style = {
        "mean": ("Mean", "#0072B2"),
        "p95": ("P95", "#E69F00"),
        "worst": ("Worst", "#CC79A7"),
    }
    uwidth = 0.24
    for offset, metric in zip((-1, 0, 1), metric_style, strict=True):
        values = [
            next(
                row["error_to_gate"]
                for row in understanding
                if row["method"] == method and row["metric"] == metric
            )
            for method in understanding_methods
        ]
        label, color = metric_style[metric]
        axes[2].bar(ux + offset * uwidth, values, uwidth, color=color, label=label)
    axes[2].axhline(1.0, color="#009E73", linewidth=1.0, linestyle="--", label="Pass")
    axes[2].set_xticks(ux, understanding_labels)
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.8, 30)
    axes[2].set_ylabel("Visual error / registered gate")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].text(-0.12, 1.04, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=12)

    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_ROOT / f"structured_redundancy_cross_program_20260901.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
