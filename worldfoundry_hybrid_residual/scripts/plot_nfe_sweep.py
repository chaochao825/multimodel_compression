#!/usr/bin/env python3
"""Summarize and visualize the training-free Wan NFE sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--probe-results-root", required=True)
    parser.add_argument("--quality-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    baseline = Path(args.baseline_dir)
    probe = Path(args.probe_results_root)
    quality = Path(args.quality_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline_run = next(
        row for row in read_csv(baseline / "generation_runs.csv") if row["method"] == "fa3_bf16"
    )
    baseline_seconds = float(baseline_run["seconds_including_text_and_vae"])
    metric_rows = {
        int(row["video"].removeprefix("step").removesuffix(".mp4")): row
        for row in read_csv(quality / "paired_video_metrics.csv")
        if row["video"].startswith("step")
    }
    summary = []
    videos = {20: baseline / baseline_run["video_file"]}
    for steps in (20, 12, 8, 6, 4):
        if steps == 20:
            run = baseline_run
        else:
            run_dir = probe / f"nfe_sweep_f17_step{steps}_v1"
            run = read_csv(run_dir / "generation_runs.csv")[0]
            videos[steps] = run_dir / run["video_file"]
        metric = metric_rows[steps]
        summary.append(
            {
                "sampling_steps": steps,
                "seconds_including_text_and_vae": float(run["seconds_including_text_and_vae"]),
                "speedup_vs_20step": baseline_seconds / float(run["seconds_including_text_and_vae"]),
                "self_attention_calls": int(run["self_attention_calls"]),
                "pixel_psnr_db": metric["pixel_psnr_db"],
                "frame_ssim_mean": float(metric["frame_ssim_mean"]),
                "temporal_delta_mae": float(metric["temporal_delta_mae"]),
                "candidate_frame_delta_mae": float(metric["candidate_frame_delta_mae"]),
                "video_file": str(videos[steps]),
            }
        )
    write_csv(output / "nfe_sweep_summary.csv", summary)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )
    figure, left = plt.subplots(figsize=(9.6, 5.8))
    candidates = [row for row in summary if row["sampling_steps"] != 20]
    speedups = [row["speedup_vs_20step"] for row in candidates]
    ssim = [row["frame_ssim_mean"] for row in candidates]
    psnr = [float(row["pixel_psnr_db"]) for row in candidates]
    left.plot(speedups, ssim, marker="o", linewidth=2.2, color="#D1495B", label="frame SSIM")
    left.set_xlabel("End-to-end speedup vs 20-step")
    left.set_ylabel("Frame SSIM vs 20-step", color="#D1495B")
    left.tick_params(axis="y", labelcolor="#D1495B")
    right = left.twinx()
    right.spines.top.set_visible(False)
    right.plot(speedups, psnr, marker="s", linewidth=2.0, color="#30638E", label="pixel PSNR")
    right.set_ylabel("Pixel PSNR (dB) vs 20-step", color="#30638E")
    right.tick_params(axis="y", labelcolor="#30638E")
    for row in candidates:
        left.annotate(
            f"{row['sampling_steps']} steps",
            (row["speedup_vs_20step"], row["frame_ssim_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    left.set_title("Training-free NFE reduction: speed rises before paired-video fidelity recovers")
    figure.tight_layout()
    figure.savefig(output / "nfe_speed_quality.png")
    plt.close(figure)

    selected_frames = (0, 8, 16)
    ordered_steps = (20, 12, 8, 6, 4)
    figure, axes = plt.subplots(len(ordered_steps), len(selected_frames), figsize=(13.5, 11.5))
    for row_index, steps in enumerate(ordered_steps):
        frames = imageio.mimread(videos[steps])
        for col_index, frame_index in enumerate(selected_frames):
            axis = axes[row_index, col_index]
            axis.imshow(frames[min(frame_index, len(frames) - 1)])
            axis.axis("off")
            if row_index == 0:
                axis.set_title(f"frame {frame_index}")
            if col_index == 0:
                metric = metric_rows[steps]
                label = f"{steps} steps"
                if steps != 20:
                    label += f"\nSSIM {float(metric['frame_ssim_mean']):.3f}"
                axis.text(
                    -0.03,
                    0.5,
                    label,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    figure.suptitle("Wan F17 fixed-seed NFE sweep: first, middle, and last decoded frames", y=0.995)
    figure.tight_layout()
    figure.savefig(output / "nfe_frame_contact_sheet.png", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote NFE figures to {output}")


if __name__ == "__main__":
    main()
