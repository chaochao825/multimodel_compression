#!/usr/bin/env python3
"""Render representative frames and dense-relative errors for the worst pair."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "results"
    / "h200_live"
    / "worldfoundry_ffn_hybrid_f17_multiseed_v2"
)
VIDEO_DIR = RUN / "sample_videos" / "worst_p01_seed20260723"
OUT = ROOT / "results" / "h200_live" / "figures"

VIDEOS = (
    ("Dense", "p01_dense_seed20260723.mp4"),
    ("FP8 middle-1", "p01_fp8_middle1_seed20260723.mp4"),
    ("Hybrid middle-1", "p01_hybrid_middle1_seed20260723.mp4"),
    ("Hybrid + cache .08", "p01_hybrid_middle1_cache008_seed20260723.mp4"),
)
FRAME_INDICES = (0, 8, 16)


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"could not decode frames from {path}")
    return np.stack(frames)


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    decoded = {label: read_video(VIDEO_DIR / filename) for label, filename in VIDEOS}
    shapes = {frames.shape for frames in decoded.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"video shapes do not match: {sorted(shapes)}")
    frame_count = next(iter(shapes))[0]
    if max(FRAME_INDICES) >= frame_count:
        raise RuntimeError(f"requested frame exceeds decoded length {frame_count}")

    figure, axes = plt.subplots(len(VIDEOS), len(FRAME_INDICES), figsize=(12.0, 8.0))
    for row, (label, _) in enumerate(VIDEOS):
        for column, frame_index in enumerate(FRAME_INDICES):
            axis = axes[row, column]
            axis.imshow(decoded[label][frame_index])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"Frame {frame_index + 1}")
            if column == 0:
                axis.set_ylabel(label, rotation=90, labelpad=10)
    figure.subplots_adjust(wspace=0.025, hspace=0.08)
    save_figure(figure, "worst_case_contact_sheet")

    dense = decoded["Dense"].astype(np.float32)
    candidates = VIDEOS[1:]
    all_errors = [
        np.abs(decoded[label].astype(np.float32) - dense).mean(axis=-1)
        for label, _ in candidates
    ]
    vmax = float(np.quantile(np.concatenate([error.ravel() for error in all_errors]), 0.99))
    figure, axes = plt.subplots(len(candidates), len(FRAME_INDICES), figsize=(12.0, 6.0))
    rows: list[dict[str, object]] = []
    image = None
    for row, ((label, _), error) in enumerate(zip(candidates, all_errors)):
        for column, frame_index in enumerate(FRAME_INDICES):
            axis = axes[row, column]
            image = axis.imshow(error[frame_index], cmap="inferno", vmin=0.0, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"Frame {frame_index + 1}")
            if column == 0:
                axis.set_ylabel(label, rotation=90, labelpad=10)
            pixel_error = np.abs(
                decoded[label][frame_index].astype(np.float32)
                - dense[frame_index]
            )
            rows.append(
                {
                    "method": label,
                    "frame_index": frame_index,
                    "channel_mae": float(pixel_error.mean()),
                    "channel_rmse": float(np.sqrt(np.square(pixel_error).mean())),
                    "channel_abs_p95": float(np.quantile(pixel_error, 0.95)),
                    "channel_abs_max": float(pixel_error.max()),
                }
            )
    if image is None:
        raise RuntimeError("no difference image was rendered")
    figure.subplots_adjust(wspace=0.025, hspace=0.08, right=0.90)
    colorbar_axis = figure.add_axes((0.915, 0.15, 0.015, 0.70))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean absolute RGB difference")
    save_figure(figure, "worst_case_difference_heatmap")

    with (OUT / "worst_case_frame_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CONTACT_SHEET_OK frames={frame_count} vmax={vmax:.4f}")


if __name__ == "__main__":
    main()
