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
RUNTIME_ROOT = (
    ROOT
    / "worldfoundry_hybrid_residual"
    / "results"
    / "wan_rcm_exact_runtime_exp052_20260901"
)
ATTENTION_ROOT = (
    ROOT
    / "worldfoundry_hybrid_residual"
    / "results"
    / "wan_rcm_onpolicy_attention_exp054_20260901"
    / "analysis_v1"
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
    payload = load_json(
        RUNTIME_ROOT / "outputs_exp052" / "evaluation_v1" / "gate_summary.json"
    )
    method_rows = {row["method"]: row for row in payload["method_rows"]}
    rows = []
    for method in ("teacher20", "native4", "rcm4"):
        summary = method_rows[method]
        known = {
            "text": summary["median_text_seconds"],
            "denoiser": summary["median_denoiser_seconds"],
            "vae": summary["median_vae_seconds"],
            "transfer": summary["median_cpu_transfer_seconds"],
            "serialization": summary["median_serialization_seconds"],
        }
        known["other"] = summary["median_request_seconds"] - sum(known.values())
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


def attention_rows() -> list[dict]:
    rows = []
    with (ATTENTION_ROOT / "cell_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            for split in ("calibration", "evaluation"):
                rows.append(
                    {
                        "step": int(raw["step"]),
                        "layer": int(raw["layer"]),
                        "split": split,
                        "threshold_ratio": float(raw[f"{split}_threshold_ratio"]),
                        "passes": raw[f"{split}_passes"] == "True",
                    }
                )
    return rows


def main() -> None:
    quality = quality_rows()
    timing = timing_rows()
    understanding = understanding_rows()
    attention = attention_rows()
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
    write_csv(
        FIGURE_ROOT / "structured_redundancy_wan_attention_exp054_20260901.csv",
        ["step", "layer", "split", "threshold_ratio", "passes"],
        attention,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes_grid = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    axes = axes_grid.ravel()

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
    components = ("text", "denoiser", "vae", "transfer", "serialization", "other")
    component_colors = {
        "text": "#56B4E9",
        "denoiser": "#0072B2",
        "vae": "#E69F00",
        "transfer": "#009E73",
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

    layer_ids = np.arange(30)
    attention_style = {
        "calibration": ("Calibration", "#0072B2"),
        "evaluation": ("Evaluation", "#D55E00"),
    }
    for split, (label, color) in attention_style.items():
        layer_values = [
            [
                row["threshold_ratio"]
                for row in attention
                if row["split"] == split and row["layer"] == layer
            ]
            for layer in layer_ids
        ]
        minimum = np.asarray([min(values) for values in layer_values])
        maximum = np.asarray([max(values) for values in layer_values])
        axes[3].plot(layer_ids, minimum, color=color, label=f"{label} best step")
        axes[3].fill_between(layer_ids, minimum, maximum, color=color, alpha=0.12)
    axes[3].axhline(1.0, color="#009E73", linewidth=1.0, linestyle="--", label="Pass boundary")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Wan layer")
    axes[3].set_ylabel("Cell error / registered threshold")
    axes[3].legend(frameon=False, fontsize=8)
    axes[3].text(-0.12, 1.04, "d", transform=axes[3].transAxes, fontweight="bold", fontsize=12)

    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_ROOT / f"structured_redundancy_cross_program_20260901.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
