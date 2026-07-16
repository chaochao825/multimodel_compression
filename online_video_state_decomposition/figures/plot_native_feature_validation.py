from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "p50": "#0072B2",
    "p95": "#E69F00",
    "p99": "#D55E00",
    "agreement": "#009E73",
    "cache": "#56B4E9",
    "selector": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name",
        default="native_feature_memory_validation",
    )
    return parser.parse_args()


def write_source_csv(
    path: Path,
    *,
    latency: dict[str, dict[str, float]],
    consistency: dict[str, float],
    cache_bytes: int,
    selector_bytes: int,
    total_bytes: int,
) -> None:
    rows: list[dict[str, object]] = []
    for stage in ("decode", "preprocess", "vision_encode", "read"):
        for statistic in ("mean", "p50", "p95", "p99"):
            rows.append(
                {
                    "panel": "latency",
                    "metric": stage,
                    "statistic": statistic,
                    "value": latency[stage][statistic],
                    "unit": "seconds",
                }
            )
    for metric, value in consistency.items():
        rows.append(
            {
                "panel": "consistency",
                "metric": metric,
                "statistic": "agreement",
                "value": value,
                "unit": "percent",
            }
        )
    for metric, value in (
        ("native_feature_cache", cache_bytes),
        ("provisioned_selector", selector_bytes),
        ("total_persistent_state", total_bytes),
    ):
        rows.append(
            {
                "panel": "state",
                "metric": metric,
                "statistic": "bytes",
                "value": value,
                "unit": "bytes",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    payload = json.loads(
        args.validation_json.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency = payload["latency_seconds"]
    policy_rows = int(payload["policy_rows"])
    consistency = {
        "Selected frames": 100.0
        * (
            policy_rows
            - int(payload["selected_frame_mismatches_vs_raw"])
        )
        / policy_rows,
        "Predictions": 100.0
        * (
            policy_rows
            - int(payload["prediction_mismatches_vs_raw"])
        )
        / policy_rows,
        "Correctness": 100.0
        * (
            policy_rows
            - int(payload["correctness_mismatches_vs_raw"])
        )
        / policy_rows,
    }
    total_bytes = int(payload["state_bytes_values"][0])
    cache_bytes = int(payload["feature_cache_bytes_values"][0])
    selector_bytes = total_bytes - cache_bytes
    csv_path = args.output_dir / f"{args.name}.csv"
    write_source_csv(
        csv_path,
        latency=latency,
        consistency=consistency,
        cache_bytes=cache_bytes,
        selector_bytes=selector_bytes,
        total_bytes=total_bytes,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.6),
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]},
    )

    stages = ("decode", "preprocess", "vision_encode", "read")
    stage_labels = ("Decode", "Preprocess", "Vision encode", "Cached read")
    positions = np.arange(len(stages))
    width = 0.23
    for offset, statistic in enumerate(("p50", "p95", "p99")):
        values = [float(latency[stage][statistic]) for stage in stages]
        axes[0].bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=statistic.upper(),
            color=COLORS[statistic],
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Latency (seconds, log scale)")
    axes[0].set_xticks(positions, stage_labels, rotation=22, ha="right")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25, which="both")
    axes[0].text(
        -0.12,
        1.03,
        "a",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    labels = list(consistency)
    values = [consistency[label] for label in labels]
    agreement_positions = np.arange(len(labels))
    axes[1].barh(
        agreement_positions,
        values,
        color=COLORS["agreement"],
        height=0.58,
    )
    axes[1].set_yticks(agreement_positions, labels)
    axes[1].set_xlim(0.0, 100.0)
    axes[1].set_xlabel("Agreement with raw path (%)")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)
    for position, value in zip(agreement_positions, values):
        axes[1].text(
            min(value + 1.0, 97.5),
            position,
            f"{value:.3f}%",
            va="center",
            ha="right" if value > 96.0 else "left",
        )
    axes[1].text(
        -0.18,
        1.03,
        "b",
        transform=axes[1].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    cache_mib = cache_bytes / (1024**2)
    selector_mib = selector_bytes / (1024**2)
    axes[2].barh(
        [0],
        [cache_mib],
        color=COLORS["cache"],
        label="Visual feature cache",
        height=0.5,
    )
    axes[2].barh(
        [0],
        [selector_mib],
        left=[cache_mib],
        color=COLORS["selector"],
        label="Provisioned selector",
        height=0.5,
    )
    axes[2].set_yticks([])
    axes[2].set_xlabel("Persistent state (MiB)")
    axes[2].set_xlim(0.0, max(8.5, total_bytes / (1024**2) * 1.08))
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend(frameon=False, loc="upper center")
    axes[2].text(
        cache_mib / 2,
        0,
        f"{cache_mib:.2f} MiB\nfeature cache",
        ha="center",
        va="center",
    )
    axes[2].text(
        0.5,
        -0.48,
        f"Total: {total_bytes / 1024:.2f} KiB; "
        f"selector: {selector_bytes / 1024:.2f} KiB",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
    )
    axes[2].text(
        -0.16,
        1.03,
        "c",
        transform=axes[2].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{args.name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
