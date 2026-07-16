from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="instant_frames|directory containing memory_summary.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=64)
    return parser.parse_args()


def parse_input(value: str) -> tuple[int, Path]:
    fields = value.split("|", 1)
    if len(fields) != 2:
        raise ValueError("input must be instant_frames|directory")
    return int(fields[0]), Path(fields[1]).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(
    inputs: list[tuple[int, Path]],
    *,
    capacity: int,
) -> list[dict[str, object]]:
    output = []
    for instant_frames, directory in sorted(inputs):
        rows = read_csv(directory / "memory_summary.csv")
        selected = [
            row
            for row in rows
            if row["category"] == "__all__"
            and row["mode"] == "frame_centered_unit"
            and int(row["capacity"]) == capacity
            and row["method"] in ("instant_adaptive", "instant_oja")
        ]
        if not selected:
            raise ValueError(
                f"no capacity={capacity} hybrid rows in {directory}"
            )
        for row in selected:
            output.append(
                {
                    "instant_frames": instant_frames,
                    "capacity": capacity,
                    "long_term_capacity": max(
                        0,
                        capacity - 16 * instant_frames,
                    ),
                    "layer": int(row["layer"]),
                    "delay": int(row["delay"]),
                    "method": row["method"],
                    "mean_cosine": float(row["mean_cosine_mean"]),
                    "cosine_std": float(row["mean_cosine_std"]),
                    "coverage": float(row["coverage_mean"]),
                    "cosine_gain_vs_recent": float(
                        row["cosine_gain_vs_recent"]
                    ),
                    "run_count": int(row["run_count"]),
                }
            )
    return output


def plot(rows: list[dict[str, object]], path: Path) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    instant_values = sorted({int(row["instant_frames"]) for row in rows})
    colors = {
        1: "#0072B2",
        2: "#E69F00",
        3: "#009E73",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    axes_flat = axes.reshape(-1)
    for axis, layer in zip(axes_flat[:5], layers, strict=True):
        for instant_frames in instant_values:
            selected = [
                row
                for row in rows
                if int(row["layer"]) == layer
                and int(row["instant_frames"]) == instant_frames
                and row["method"] == "instant_oja"
            ]
            selected.sort(key=lambda row: int(row["delay"]))
            axis.plot(
                [int(row["delay"]) for row in selected],
                [float(row["cosine_gain_vs_recent"]) for row in selected],
                marker="o",
                linewidth=1.8,
                color=colors[instant_frames],
                label=(
                    f"{instant_frames} instant frame"
                    f"{'s' if instant_frames > 1 else ''}"
                ),
            )
        axis.axhline(0.0, color="#495057", linewidth=0.8)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Query delay (hidden frames)")
        axis.set_ylabel("Cosine gain vs recent")
        axis.grid(axis="y", alpha=0.20)

    summary_axis = axes_flat[5]
    method_styles = {
        "instant_adaptive": ("#D55E00", "o", "adaptive slots"),
        "instant_oja": ("#6A4C93", "s", "Oja subspace"),
    }
    for method, (color, marker, label) in method_styles.items():
        means = []
        for instant_frames in instant_values:
            samples = [
                float(row["cosine_gain_vs_recent"])
                for row in rows
                if int(row["instant_frames"]) == instant_frames
                and int(row["delay"]) == 8
                and row["method"] == method
            ]
            means.append(float(np.mean(samples)))
        summary_axis.plot(
            instant_values,
            means,
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    summary_axis.axhline(0.0, color="#495057", linewidth=0.8)
    summary_axis.set_xticks(instant_values)
    summary_axis.set_xlabel("Instant-cache frames")
    summary_axis.set_ylabel("Mean delay-8 gain vs recent")
    summary_axis.set_title("Average across layers")
    summary_axis.grid(axis="y", alpha=0.20)
    summary_axis.legend(frameon=False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=len(instant_values),
        bbox_to_anchor=(0.5, -0.01),
    )
    for axis in axes_flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    inputs = [parse_input(value) for value in args.input]
    rows = collect(inputs, capacity=args.capacity)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "memory_instant_window_ablation.csv"
    plot_path = args.out_dir / "memory_instant_window_ablation.png"
    write_csv(csv_path, rows)
    plot(rows, plot_path)
    print(
        {
            "inputs": len(inputs),
            "rows": len(rows),
            "csv": str(csv_path.resolve()),
            "png": str(plot_path.resolve()),
            "pdf": str(plot_path.with_suffix(".pdf").resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
