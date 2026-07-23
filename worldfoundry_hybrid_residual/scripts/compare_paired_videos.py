#!/usr/bin/env python3
"""Compare same-prompt/same-seed decoded videos against a dense reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--reference", default="0000_sdpa_seed20260723.mp4")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.mp4")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> tuple[np.ndarray, float]:
    if cv2 is None:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        try:
            metadata = reader.get_meta_data()
            frames = [frame for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        return np.stack(frames), float(metadata["fps"])

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames), fps


def fallback_structural_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Match scikit-image's default 7x7, sample-covariance SSIM for RGB uint8."""

    from scipy.ndimage import uniform_filter

    channel_scores: list[float] = []
    window_size = 7
    covariance_normalization = (window_size**2) / (window_size**2 - 1)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    border = (window_size - 1) // 2
    for channel in range(reference.shape[2]):
        ref = reference[..., channel].astype(np.float64)
        cand = candidate[..., channel].astype(np.float64)
        ref_mean = uniform_filter(ref, size=window_size)
        candidate_mean = uniform_filter(cand, size=window_size)
        ref_variance = covariance_normalization * (
            uniform_filter(ref * ref, size=window_size) - ref_mean * ref_mean
        )
        candidate_variance = covariance_normalization * (
            uniform_filter(cand * cand, size=window_size)
            - candidate_mean * candidate_mean
        )
        covariance = covariance_normalization * (
            uniform_filter(ref * cand, size=window_size)
            - ref_mean * candidate_mean
        )
        numerator = (
            (2.0 * ref_mean * candidate_mean + c1)
            * (2.0 * covariance + c2)
        )
        denominator = (
            (ref_mean * ref_mean + candidate_mean * candidate_mean + c1)
            * (ref_variance + candidate_variance + c2)
        )
        similarity = numerator / denominator
        channel_scores.append(
            float(np.mean(similarity[border:-border, border:-border]))
        )
    return float(np.mean(channel_scores))


def frame_structural_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    if structural_similarity is not None:
        return float(
            structural_similarity(
                reference, candidate, channel_axis=2, data_range=255
            )
        )
    return fallback_structural_similarity(reference, candidate)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"video shape mismatch: {reference.shape} versus {candidate.shape}")
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    delta = cand - ref
    abs_delta = np.abs(delta)
    mse = float(np.mean(np.square(delta), dtype=np.float64))
    frame_mse = np.mean(np.square(delta), axis=(1, 2, 3), dtype=np.float64)
    frame_psnr = [
        float("inf") if value == 0 else 10.0 * math.log10((255.0**2) / float(value))
        for value in frame_mse
    ]
    frame_ssim = [
        frame_structural_similarity(r, c)
        for r, c in zip(reference, candidate)
    ]
    ref_motion = np.diff(ref, axis=0)
    cand_motion = np.diff(cand, axis=0)
    temporal_delta = cand_motion - ref_motion
    return {
        "frames": int(reference.shape[0]),
        "height": int(reference.shape[1]),
        "width": int(reference.shape[2]),
        "pixel_mae": float(np.mean(abs_delta, dtype=np.float64)),
        "pixel_rmse": math.sqrt(mse),
        "pixel_psnr_db": float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse),
        "pixel_max_abs": float(np.max(abs_delta)),
        "exact_pixel_fraction": float(np.mean(delta == 0)),
        "frame_psnr_mean_db": float(np.mean(frame_psnr)),
        "frame_psnr_p05_db": percentile(frame_psnr, 0.05),
        "frame_ssim_mean": float(np.mean(frame_ssim)),
        "frame_ssim_p05": percentile(frame_ssim, 0.05),
        "frame_ssim_min": float(np.min(frame_ssim)),
        "temporal_delta_mae": float(np.mean(np.abs(temporal_delta), dtype=np.float64)),
        "reference_frame_delta_mae": float(np.mean(np.abs(ref_motion), dtype=np.float64)),
        "candidate_frame_delta_mae": float(np.mean(np.abs(cand_motion), dtype=np.float64)),
    }


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    reference_path = video_dir / args.reference
    reference, reference_fps = read_video(reference_path)
    rows: list[dict[str, object]] = []
    for path in sorted(video_dir.glob(args.glob)):
        candidate, fps = read_video(path)
        row: dict[str, object] = {
            "video": path.name,
            "reference": reference_path.name,
            "status": "ok",
            "fps": fps,
            "video_sha256": sha256(path),
        }
        try:
            row.update(compare(reference, candidate))
            if fps != reference_fps:
                row["status"] = "fps_mismatch"
        except Exception as error:
            row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        rows.append(row)
        print(
            f"{path.name:45s} {row['status']:12s} "
            f"PSNR={row.get('pixel_psnr_db', float('nan')):.4f} "
            f"SSIM={row.get('frame_ssim_mean', float('nan')):.6f}",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (args.out_dir / "paired_video_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "scope": "decoded-frame paired comparison; same prompt, seed, sampler, and encoding settings",
        "warning": "PSNR/SSIM measure deviation from dense SDPA, not standalone semantic or perceptual quality",
        "video_dir": str(video_dir),
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "reference_fps": reference_fps,
        "decoder": "opencv" if cv2 is not None else "imageio-ffmpeg",
        "opencv": None if cv2 is None else cv2.__version__,
        "ssim_backend": (
            "scikit-image"
            if structural_similarity is not None
            else "scipy-uniform-filter-compatible"
        ),
        "numpy": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (args.out_dir / "paired_video_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
