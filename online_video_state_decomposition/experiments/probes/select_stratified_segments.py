from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = (
    "scene_cut",
    "static",
    "camera_motion",
    "object_motion",
    "high_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-videos", type=int, default=120)
    parser.add_argument("--segments-per-video", type=int, default=3)
    parser.add_argument("--screen-frames", type=int, default=8)
    parser.add_argument("--screen-stride", type=int, default=8)
    parser.add_argument("--analysis-size", type=int, default=96)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--formal-frames", type=int, default=16)
    parser.add_argument("--formal-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_videos(root: Path) -> list[Path]:
    suffixes = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    videos = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not videos:
        raise FileNotFoundError(f"no videos found directly under {root}")
    return videos


def sample_evenly(
    videos: list[Path],
    *,
    count: int,
    seed: int,
) -> list[Path]:
    if count >= len(videos):
        return videos
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(videos), size=count, replace=False))
    return [videos[int(index)] for index in indices]


def segment_fractions(count: int) -> list[float]:
    if count <= 0:
        raise ValueError("segments_per_video must be positive")
    return np.linspace(0.15, 0.85, count).tolist()


def decode_frames(
    path: Path,
    *,
    frame_indices: list[int],
    size: int,
) -> list[np.ndarray]:
    if not frame_indices:
        return []
    ordered_indices = sorted({int(index) for index in frame_indices})
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    capture.set(cv2.CAP_PROP_POS_FRAMES, ordered_indices[0])
    decoded: dict[int, np.ndarray] = {}
    target_indices = set(ordered_indices)
    for index in range(ordered_indices[0], ordered_indices[-1] + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index in target_indices:
            decoded[index] = cv2.resize(
                frame,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )
    capture.release()
    return [decoded[index] for index in frame_indices if index in decoded]


def histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_hist = cv2.calcHist([left], [0], None, [32], [0.0, 1.0])
    right_hist = cv2.calcHist([right], [0], None, [32], [0.0, 1.0])
    cv2.normalize(left_hist, left_hist)
    cv2.normalize(right_hist, right_hist)
    return float(
        cv2.compareHist(
            left_hist,
            right_hist,
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def aligned_error(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> float:
    height, width = previous.shape
    candidates = []
    for sign in (1.0, -1.0):
        transform = np.asarray(
            [[1.0, 0.0, sign * dx], [0.0, 1.0, sign * dy]],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            previous,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        candidates.append(float(np.mean(np.abs(warped - current))))
    return min(candidates)


def segment_metrics(frames_bgr: list[np.ndarray]) -> dict[str, float]:
    grayscale = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for frame in frames_bgr
    ]
    raw_errors = []
    aligned_errors = []
    shift_magnitudes = []
    phase_responses = []
    cut_scores = []
    diagonal = float(np.hypot(grayscale[0].shape[0], grayscale[0].shape[1]))
    for previous, current in zip(grayscale[:-1], grayscale[1:], strict=True):
        raw_errors.append(float(np.mean(np.abs(previous - current))))
        (dx, dy), response = cv2.phaseCorrelate(previous, current)
        aligned_errors.append(
            aligned_error(previous, current, dx=dx, dy=dy)
        )
        shift_magnitudes.append(float(np.hypot(dx, dy) / diagonal))
        phase_responses.append(float(response))
        cut_scores.append(histogram_distance(previous, current))
    raw_change = float(np.mean(raw_errors))
    aligned_change = float(np.mean(aligned_errors))
    return {
        "raw_change": raw_change,
        "max_raw_change": float(np.max(raw_errors)),
        "aligned_change": aligned_change,
        "alignment_gain": float(
            np.clip(
                (raw_change - aligned_change) / max(raw_change, 1e-12),
                -1.0,
                1.0,
            )
        ),
        "mean_shift_normalized": float(np.mean(shift_magnitudes)),
        "max_shift_normalized": float(np.max(shift_magnitudes)),
        "mean_phase_response": float(np.mean(phase_responses)),
        "max_cut_score": float(np.max(cut_scores)),
        "mean_cut_score": float(np.mean(cut_scores)),
    }


def scan_video(
    path: Path,
    *,
    fractions: list[float],
    frames: int,
    stride: int,
    size: int,
    formal_frames: int,
    formal_stride: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    screen_span = (frames - 1) * stride
    formal_span = (formal_frames - 1) * formal_stride
    required_span = max(screen_span, formal_span)
    max_start = total_frames - required_span - 1
    if max_start <= 0:
        return []
    output = []
    for fraction in fractions:
        start_frame = int(round(max_start * fraction))
        indices = [start_frame + index * stride for index in range(frames)]
        decoded = decode_frames(path, frame_indices=indices, size=size)
        if len(decoded) != frames:
            continue
        metrics = segment_metrics(decoded)
        output.append(
            {
                "video": str(path),
                "video_id": path.stem,
                "start_fraction": float(fraction),
                "start_frame": start_frame,
                "total_frames": total_frames,
                "fps": fps,
                "screen_frames": frames,
                "screen_stride": stride,
                "formal_frames": formal_frames,
                "formal_stride": formal_stride,
                **metrics,
            }
        )
    return output


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def add_selection_scores(rows: list[dict[str, object]]) -> None:
    metric_names = (
        "raw_change",
        "max_raw_change",
        "aligned_change",
        "alignment_gain",
        "mean_shift_normalized",
        "mean_phase_response",
        "max_cut_score",
    )
    percentiles = {
        name: percentile_ranks(
            np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        )
        for name in metric_names
    }
    for index, row in enumerate(rows):
        raw = percentiles["raw_change"][index]
        max_raw = percentiles["max_raw_change"][index]
        aligned = percentiles["aligned_change"][index]
        gain = percentiles["alignment_gain"][index]
        shift = percentiles["mean_shift_normalized"][index]
        response = percentiles["mean_phase_response"][index]
        cut = percentiles["max_cut_score"][index]
        row.update(
            {
                "score_static": float(
                    0.65 * (1.0 - raw)
                    + 0.20 * (1.0 - max_raw)
                    + 0.15 * (1.0 - cut)
                ),
                "score_camera_motion": float(
                    0.40 * shift
                    + 0.30 * gain
                    + 0.20 * response
                    + 0.10 * raw
                ),
                "score_object_motion": float(
                    0.45 * aligned
                    + 0.20 * raw
                    + 0.15 * (1.0 - gain)
                    + 0.20 * (1.0 - cut)
                ),
                "score_scene_cut": float(
                    0.60 * cut + 0.30 * max_raw + 0.10 * raw
                ),
                "score_high_change": float(
                    0.45 * raw + 0.40 * aligned + 0.15 * max_raw
                ),
            }
        )


def select_balanced(
    rows: list[dict[str, object]],
    *,
    per_category: int,
) -> list[dict[str, object]]:
    selected = []
    used_videos: set[str] = set()
    for category in CATEGORIES:
        score_name = f"score_{category}"
        ordered = sorted(
            rows,
            key=lambda row: float(row[score_name]),
            reverse=True,
        )
        category_rows = []
        for row in ordered:
            video_id = str(row["video_id"])
            if video_id in used_videos:
                continue
            chosen = dict(row)
            chosen["category"] = category
            chosen["category_score"] = float(row[score_name])
            chosen["run_name"] = (
                f"{category}_{video_id}_f{int(row['start_frame']):07d}"
            )
            category_rows.append(chosen)
            used_videos.add(video_id)
            if len(category_rows) == per_category:
                break
        selected.extend(category_rows)
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(
    rows: list[dict[str, object]],
    *,
    path: Path,
    size: int = 112,
    columns: int = 3,
) -> None:
    label_height = 46
    frames_per_tile = 3
    tile_width = size * frames_per_tile
    tile_height = size + label_height
    grid_rows = int(np.ceil(len(rows) / columns))
    sheet = np.full(
        (grid_rows * tile_height, columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, row in enumerate(rows):
        formal_frames = int(row["formal_frames"])
        formal_stride = int(row["formal_stride"])
        start_frame = int(row["start_frame"])
        offsets = (0, formal_frames // 2, formal_frames - 1)
        frame_indices = [
            start_frame + offset * formal_stride for offset in offsets
        ]
        frames = decode_frames(
            Path(str(row["video"])),
            frame_indices=frame_indices,
            size=size,
        )
        if len(frames) != frames_per_tile:
            continue
        row_index = index // columns
        column_index = index % columns
        y0 = row_index * tile_height
        x0 = column_index * tile_width
        strip = np.concatenate(frames, axis=1)
        sheet[y0 : y0 + size, x0 : x0 + tile_width] = strip
        label = f"{row['category']} | {row['video_id'][:15]}"
        metrics = (
            f"raw {float(row['raw_change']):.3f} "
            f"align {float(row['aligned_change']):.3f} "
            f"shift {float(row['mean_shift_normalized']):.3f} "
            f"cut {float(row['max_cut_score']):.3f}"
        )
        cv2.putText(
            sheet,
            label,
            (x0 + 4, y0 + size + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            metrics,
            (x0 + 4, y0 + size + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write contact sheet: {path}")


def main() -> int:
    args = parse_args()
    videos = sample_evenly(
        discover_videos(args.video_root),
        count=args.sample_videos,
        seed=args.seed,
    )
    fractions = segment_fractions(args.segments_per_video)
    candidates: list[dict[str, object]] = []
    failures = []
    for index, video in enumerate(videos, start=1):
        scanned = scan_video(
            video,
            fractions=fractions,
            frames=args.screen_frames,
            stride=args.screen_stride,
            size=args.analysis_size,
            formal_frames=args.formal_frames,
            formal_stride=args.formal_stride,
        )
        if not scanned:
            failures.append(str(video))
        candidates.extend(scanned)
        if index % 20 == 0:
            print(
                json.dumps(
                    {
                        "progress_videos": index,
                        "candidate_segments": len(candidates),
                    }
                ),
                flush=True,
            )
    if not candidates:
        raise RuntimeError("no candidate segments were decoded")
    add_selection_scores(candidates)
    selected = select_balanced(
        candidates,
        per_category=args.per_category,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "all_candidate_segments.csv", candidates)
    write_csv(args.out_dir / "selected_segments.csv", selected)
    (args.out_dir / "selected_segments.json").write_text(
        json.dumps(
            {
                "config": vars(args)
                | {
                    "video_root": str(args.video_root),
                    "out_dir": str(args.out_dir),
                },
                "sampled_videos": len(videos),
                "candidate_segments": len(candidates),
                "decode_failures": failures,
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    contact_sheet(
        selected,
        path=args.out_dir / "selected_segments_contact_sheet.jpg",
    )
    print(
        json.dumps(
            {
                "sampled_videos": len(videos),
                "candidate_segments": len(candidates),
                "decode_failures": len(failures),
                "selected_segments": len(selected),
                "category_counts": {
                    category: sum(
                        row["category"] == category for row in selected
                    )
                    for category in CATEGORIES
                },
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
