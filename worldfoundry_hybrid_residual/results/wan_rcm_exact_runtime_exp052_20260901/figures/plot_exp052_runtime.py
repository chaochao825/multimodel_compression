from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
EXP047_ROOT = REPO_ROOT / "worldfoundry_hybrid_residual/results/wan_rcm_baseline_exp047_20260901"
EXP052_EVAL = RESULT_ROOT / "local_evaluation_v1"

METHODS = ["teacher20", "native4", "rcm4"]
LABELS = ["Teacher-20", "Native-4", "rCM-4"]
COLORS = {
    "text": "#56B4E9",
    "denoiser": "#0072B2",
    "vae": "#E69F00",
    "transfer": "#009E73",
    "serialization": "#D55E00",
    "other": "#999999",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_resident_rows() -> dict[str, dict[str, float]]:
    with (EXP052_EVAL / "method_summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def load_legacy_e2e() -> dict[str, float]:
    values = {}
    for method in METHODS:
        path = EXP047_ROOT / f"timing/outputs/timing_f81/{method}/generation_manifest.json"
        values[method] = load_json(path)["summary"]["median_warm_e2e_seconds"]
    return values


def write_plot_data(
    resident: dict[str, dict[str, float]],
    legacy: dict[str, float],
    quality: dict[str, float],
) -> None:
    with (HERE / "runtime_pareto_data.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "legacy_request_seconds",
            "resident_request_seconds",
            "resident_text_seconds",
            "resident_denoiser_seconds",
            "resident_vae_seconds",
            "resident_cpu_transfer_seconds",
            "resident_serialization_seconds",
            "mean_teacher_normalized_quality",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            row = resident[method]
            writer.writerow(
                {
                    "method": method,
                    "legacy_request_seconds": legacy[method],
                    "resident_request_seconds": row["median_request_seconds"],
                    "resident_text_seconds": row["median_text_seconds"],
                    "resident_denoiser_seconds": row["median_denoiser_seconds"],
                    "resident_vae_seconds": row["median_vae_seconds"],
                    "resident_cpu_transfer_seconds": row["median_cpu_transfer_seconds"],
                    "resident_serialization_seconds": row["median_serialization_seconds"],
                    "mean_teacher_normalized_quality": quality[method],
                }
            )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    resident = load_resident_rows()
    legacy = load_legacy_e2e()
    quality_json = load_json(EXP047_ROOT / "quality/evaluation_v1/quality_gate_summary.json")
    quality = {
        "teacher20": 1.0,
        "native4": quality_json["methods"]["native4"]["quality"]["mean_teacher_normalized"],
        "rcm4": quality_json["methods"]["rcm4"]["quality"]["mean_teacher_normalized"],
    }
    write_plot_data(resident, legacy, quality)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=True)
    x = np.arange(len(METHODS))

    old = np.array([legacy[method] for method in METHODS])
    new = np.array([float(resident[method]["median_request_seconds"]) for method in METHODS])
    width = 0.34
    axes[0].bar(x - width / 2, old, width, color="#BBBBBB", label="EXP-047 clear policy")
    axes[0].bar(x + width / 2, new, width, color="#0072B2", label="EXP-052 resident")
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Warm request latency (s)")
    axes[0].legend(frameon=False, fontsize=8)
    for index, value in enumerate(new):
        axes[0].text(index + width / 2, value + 1.1, f"{value:.1f}", ha="center", fontsize=8)
    axes[0].text(0.02, 0.96, "a", transform=axes[0].transAxes, va="top", fontweight="bold")

    component_keys = ["text", "denoiser", "vae", "transfer", "serialization"]
    component_fields = {
        "text": "median_text_seconds",
        "denoiser": "median_denoiser_seconds",
        "vae": "median_vae_seconds",
        "transfer": "median_cpu_transfer_seconds",
        "serialization": "median_serialization_seconds",
    }
    bottom = np.zeros(len(METHODS))
    for key in component_keys:
        values = np.array([float(resident[method][component_fields[key]]) for method in METHODS])
        axes[1].bar(x, values, bottom=bottom, color=COLORS[key], label=key.capitalize())
        bottom += values
    other = new - bottom
    axes[1].bar(x, other, bottom=bottom, color=COLORS["other"], label="Other")
    axes[1].set_xticks(x, LABELS)
    axes[1].set_ylabel("Resident latency breakdown (s)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].text(0.02, 0.96, "b", transform=axes[1].transAxes, va="top", fontweight="bold")

    teacher = float(resident["teacher20"]["median_request_seconds"])
    speedups = np.array([teacher / float(resident[method]["median_request_seconds"]) for method in METHODS])
    qualities = np.array([quality[method] for method in METHODS])
    axes[2].scatter(speedups, qualities, s=[55, 65, 75], c=["#666666", "#E69F00", "#009E73"])
    for index, label in enumerate(LABELS):
        axes[2].annotate(label, (speedups[index], qualities[index]), xytext=(5, 5), textcoords="offset points")
    axes[2].axhline(0.90, color="#888888", linestyle="--", linewidth=1, label="Quality gate")
    axes[2].axvline(2.5, color="#888888", linestyle=":", linewidth=1, label="Speed gate")
    axes[2].set_xlabel("Warm end-to-end speedup vs Teacher-20")
    axes[2].set_ylabel("Mean teacher-normalized VBench")
    axes[2].set_xlim(0.75, 4.35)
    axes[2].set_ylim(0.82, 1.025)
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    axes[2].text(0.02, 0.96, "c", transform=axes[2].transAxes, va="top", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.set_axisbelow(True)

    fig.savefig(HERE / "wan_rcm_exact_runtime_pareto.png", bbox_inches="tight")
    fig.savefig(HERE / "wan_rcm_exact_runtime_pareto.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
