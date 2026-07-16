from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.metrics import (  # noqa: E402
    relative_fro_error,
    residual_concentration,
    singular_value_metrics,
)
from probes.transport import (  # noqa: E402
    apply_global_bccb,
    apply_shift_basis,
    best_integer_shift,
    fit_global_bccb,
    fit_shift_basis,
    local_offsets,
    shift_grid,
    warp_grid_bilinear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64")
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--top-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--history-window", type=int, default=2)
    parser.add_argument("--use-optical-flow", action="store_true")
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool_frames_to_grid(
    frames_rgb: np.ndarray,
    *,
    time_grid: int,
    height_grid: int,
    width_grid: int,
    temporal_patch: int,
) -> np.ndarray:
    frames = np.asarray(frames_rgb, dtype=np.float64)
    required_frames = time_grid * temporal_patch
    if frames.shape[0] < required_frames:
        raise ValueError(
            f"need {required_frames} frames for grid but only have {frames.shape[0]}"
        )
    height = frames.shape[1]
    width = frames.shape[2]
    if height % height_grid or width % width_grid:
        raise ValueError(
            f"frame size {(height, width)} is not divisible by grid {(height_grid, width_grid)}"
        )
    patch_height = height // height_grid
    patch_width = width // width_grid
    pooled = frames[:required_frames].reshape(
        time_grid,
        temporal_patch,
        height_grid,
        patch_height,
        width_grid,
        patch_width,
        3,
    )
    return pooled.mean(axis=(1, 3, 5)) / 255.0


def motion_steps(
    sequence: np.ndarray,
    *,
    max_shift: int,
    cyclic: bool,
) -> tuple[list[tuple[int, int]], list[float]]:
    steps: list[tuple[int, int]] = [(0, 0)]
    errors: list[float] = [0.0]
    for frame in range(1, sequence.shape[0]):
        dy, dx, error = best_integer_shift(
            sequence[frame - 1],
            sequence[frame],
            max_shift=max_shift,
            cyclic=cyclic,
        )
        steps.append((dy, dx))
        errors.append(error)
    return steps, errors


def align_by_steps(
    sequence: np.ndarray,
    steps: list[tuple[int, int]],
    *,
    cyclic: bool,
) -> np.ndarray:
    if len(steps) != sequence.shape[0]:
        raise ValueError("step count must equal sequence length")
    cumulative_y = 0
    cumulative_x = 0
    aligned = []
    for frame, (dy, dx) in enumerate(steps):
        if frame:
            cumulative_y += int(dy)
            cumulative_x += int(dx)
        aligned.append(
            shift_grid(
                sequence[frame],
                -cumulative_y,
                -cumulative_x,
                cyclic=cyclic,
            )
        )
    return np.stack(aligned)


def causal_subspace_rows(
    sequence: np.ndarray,
    *,
    layer: int,
    ranks: list[int],
    history_window: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(1, sequence.shape[0]):
        start = max(0, frame - history_window)
        history = sequence[start:frame].reshape(-1, sequence.shape[-1]).astype(
            np.float64
        )
        current = sequence[frame].reshape(-1, sequence.shape[-1]).astype(
            np.float64
        )
        for mode in ("raw", "history_centered"):
            center = (
                np.zeros((1, history.shape[1]), dtype=np.float64)
                if mode == "raw"
                else history.mean(axis=0, keepdims=True)
            )
            history_work = history - center
            current_work = current - center
            _u, _s, vh = np.linalg.svd(history_work, full_matrices=False)
            for rank in ranks:
                kept = min(rank, vh.shape[0])
                basis = vh[:kept]
                prediction_work = (current_work @ basis.T) @ basis
                rows.append(
                    {
                        "layer": layer,
                        "frame": frame,
                        "history_start": start,
                        "history_frames": frame - start,
                        "rank": rank,
                        "mode": mode,
                        "relative_projection_error": relative_fro_error(
                            current_work,
                            prediction_work,
                        ),
                        "raw_reconstruction_error": relative_fro_error(
                            current,
                            prediction_work + center,
                        ),
                    }
                )
    return rows


def pixel_change_proxy(
    pixel_grid: np.ndarray,
    steps: list[tuple[int, int]],
    *,
    fraction: float = 0.10,
) -> tuple[list[np.ndarray], list[float]]:
    masks: list[np.ndarray] = [
        np.zeros(pixel_grid.shape[1:3], dtype=bool)
    ]
    scores: list[float] = [0.0]
    for frame in range(1, pixel_grid.shape[0]):
        dy, dx = steps[frame]
        prediction = shift_grid(
            pixel_grid[frame - 1],
            dy,
            dx,
            cyclic=False,
        )
        residual = pixel_grid[frame] - prediction
        energy = np.sum(residual * residual, axis=-1)
        count = max(1, int(math.ceil(energy.size * fraction)))
        selected = np.argpartition(energy.reshape(-1), -count)[-count:]
        mask = np.zeros(energy.size, dtype=bool)
        mask[selected] = True
        masks.append(mask.reshape(energy.shape))
        scores.append(float(np.mean(energy)))
    return masks, scores


def optical_flow_fields(
    frames_rgb: np.ndarray,
    *,
    time_grid: int,
    height_grid: int,
    width_grid: int,
    temporal_patch: int,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    import cv2

    frames = np.asarray(frames_rgb, dtype=np.float32)
    required_frames = time_grid * temporal_patch
    if frames.shape[0] < required_frames:
        raise ValueError(
            f"need {required_frames} frames for optical flow, "
            f"got {frames.shape[0]}"
        )
    temporal_frames = frames[:required_frames].reshape(
        time_grid,
        temporal_patch,
        frames.shape[1],
        frames.shape[2],
        3,
    ).mean(axis=1)
    grayscale = [
        cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for frame in temporal_frames
    ]
    fields = [
        np.zeros((height_grid, width_grid, 2), dtype=np.float64)
    ]
    metrics = [
        {
            "mean_magnitude": 0.0,
            "max_magnitude": 0.0,
        }
    ]
    source_height, source_width = grayscale[0].shape
    for frame in range(1, time_grid):
        # current -> previous gives the backward field required for sampling.
        backward = cv2.calcOpticalFlowFarneback(
            grayscale[frame],
            grayscale[frame - 1],
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        resized = cv2.resize(
            backward,
            (width_grid, height_grid),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
        resized[..., 0] *= width_grid / source_width
        resized[..., 1] *= height_grid / source_height
        magnitude = np.linalg.norm(resized, axis=-1)
        fields.append(resized)
        metrics.append(
            {
                "mean_magnitude": float(np.mean(magnitude)),
                "max_magnitude": float(np.max(magnitude)),
            }
        )
    return fields, metrics


def mean_frame_spectrum(
    sequence: np.ndarray,
    ranks: list[int],
) -> dict[str, object]:
    metrics = [
        singular_value_metrics(
            frame.reshape(-1, frame.shape[-1]),
            ranks,
        )
        for frame in sequence
    ]
    return {
        "shape": [
            int(sequence.shape[1] * sequence.shape[2]),
            int(sequence.shape[3]),
        ],
        "effective_rank": float(
            np.mean([row["effective_rank"] for row in metrics])
        ),
        "stable_rank": float(
            np.mean([row["stable_rank"] for row in metrics])
        ),
        "energy_at_rank": {
            str(rank): float(
                np.mean(
                    [
                        row["energy_at_rank"][str(rank)]
                        for row in metrics
                    ]
                )
            )
            for rank in ranks
        },
        "singular_values": [],
    }


def analyze_layer(
    hidden: np.ndarray,
    pixel_grid: np.ndarray,
    pixel_steps: list[tuple[int, int]],
    pixel_event_masks: list[np.ndarray],
    pixel_change_scores: list[float],
    optical_flows: list[np.ndarray] | None,
    optical_flow_metrics: list[dict[str, float]] | None,
    *,
    layer: int,
    ranks: list[int],
    max_shift: int,
    local_radius: int,
    block_size: int,
    top_fractions: list[float],
    ridge: float,
    history_window: int,
) -> dict[str, object]:
    hidden64 = np.asarray(hidden, dtype=np.float64)
    hidden_steps, hidden_shift_errors = motion_steps(
        hidden64,
        max_shift=max_shift,
        cyclic=True,
    )
    pixel_aligned = align_by_steps(hidden64, pixel_steps, cyclic=True)
    hidden_aligned = align_by_steps(hidden64, hidden_steps, cyclic=True)
    frame_centered = hidden64 - hidden64.mean(
        axis=(1, 2),
        keepdims=True,
    )
    frame_centered_norm = np.linalg.norm(
        frame_centered,
        axis=-1,
        keepdims=True,
    )
    token_normalized = frame_centered / (frame_centered_norm + 1e-12)

    def temporal_change(sequence: np.ndarray) -> np.ndarray:
        matrix = sequence.reshape(sequence.shape[0], -1)
        return matrix - matrix.mean(axis=0, keepdims=True)

    history_raw = hidden_aligned.reshape(-1, hidden_aligned.shape[-1])
    history_centered = history_raw - history_raw.mean(axis=0, keepdims=True)
    history_normalized = token_normalized.reshape(
        -1,
        token_normalized.shape[-1],
    )

    rank_summary = {
        "unaligned_temporal": singular_value_metrics(
            hidden64.reshape(hidden64.shape[0], -1),
            ranks,
        ),
        "pixel_aligned_temporal": singular_value_metrics(
            pixel_aligned.reshape(pixel_aligned.shape[0], -1),
            ranks,
        ),
        "hidden_aligned_temporal": singular_value_metrics(
            hidden_aligned.reshape(hidden_aligned.shape[0], -1),
            ranks,
        ),
        "unaligned_temporal_change": singular_value_metrics(
            temporal_change(hidden64),
            ranks,
        ),
        "pixel_aligned_temporal_change": singular_value_metrics(
            temporal_change(pixel_aligned),
            ranks,
        ),
        "hidden_aligned_temporal_change": singular_value_metrics(
            temporal_change(hidden_aligned),
            ranks,
        ),
        "history_feature_subspace": singular_value_metrics(
            history_raw,
            ranks,
        ),
        "history_feature_subspace_centered": singular_value_metrics(
            history_centered,
            ranks,
        ),
        "history_feature_subspace_token_normalized": singular_value_metrics(
            history_normalized,
            ranks,
        ),
        "frame_spatial_raw": mean_frame_spectrum(
            hidden_aligned,
            ranks,
        ),
        "frame_spatial_centered": mean_frame_spectrum(
            frame_centered,
            ranks,
        ),
        "frame_spatial_token_normalized": mean_frame_spectrum(
            token_normalized,
            ranks,
        ),
        "pixel_steps": [[int(dy), int(dx)] for dy, dx in pixel_steps],
        "hidden_steps": [[int(dy), int(dx)] for dy, dx in hidden_steps],
        "hidden_shift_errors": hidden_shift_errors,
    }
    subspace_rows = causal_subspace_rows(
        hidden_aligned,
        layer=layer,
        ranks=ranks,
        history_window=history_window,
    )

    offsets = local_offsets(local_radius)
    transport_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    previous_local_bccb_weights: np.ndarray | None = None
    previous_local_bttb_weights: np.ndarray | None = None
    previous_global_kernel: np.ndarray | None = None

    for frame in range(1, hidden64.shape[0]):
        previous = hidden64[frame - 1]
        current = hidden64[frame]
        pixel_dy, pixel_dx = pixel_steps[frame]
        hidden_dy, hidden_dx = hidden_steps[frame]
        pair_bccb, bccb_weights = fit_shift_basis(
            previous,
            current,
            offsets,
            cyclic=True,
            ridge=ridge,
        )
        pair_bttb, bttb_weights = fit_shift_basis(
            previous,
            current,
            offsets,
            cyclic=False,
            ridge=ridge,
        )
        pair_global, global_kernel = fit_global_bccb(
            previous,
            current,
            ridge=ridge,
        )
        predictions: dict[str, tuple[np.ndarray, str]] = {
            "identity": (previous, "fixed"),
            "pixel_shift_zero": (
                shift_grid(
                    previous,
                    pixel_dy,
                    pixel_dx,
                    cyclic=False,
                ),
                "pixel_pair",
            ),
            "pixel_shift_cyclic": (
                shift_grid(
                    previous,
                    pixel_dy,
                    pixel_dx,
                    cyclic=True,
                ),
                "pixel_pair",
            ),
            "hidden_shift_cyclic": (
                shift_grid(
                    previous,
                    hidden_dy,
                    hidden_dx,
                    cyclic=True,
                ),
                "hidden_pair",
            ),
            "local_bccb_pair": (pair_bccb, "pair_oracle"),
            "local_bttb_pair": (pair_bttb, "pair_oracle"),
            "global_bccb_pair": (pair_global, "pair_oracle"),
        }
        if optical_flows is not None:
            predictions["optical_flow_warp"] = (
                warp_grid_bilinear(previous, optical_flows[frame]),
                "pixel_pair_flow",
            )
        if previous_local_bccb_weights is not None:
            predictions["local_bccb_causal"] = (
                apply_shift_basis(
                    previous,
                    offsets,
                    previous_local_bccb_weights,
                    cyclic=True,
                ),
                "previous_transition",
            )
        if previous_local_bttb_weights is not None:
            predictions["local_bttb_causal"] = (
                apply_shift_basis(
                    previous,
                    offsets,
                    previous_local_bttb_weights,
                    cyclic=False,
                ),
                "previous_transition",
            )
        if previous_global_kernel is not None:
            predictions["global_bccb_causal"] = (
                apply_global_bccb(previous, previous_global_kernel),
                "previous_transition",
            )

        for method, (prediction, scope) in predictions.items():
            error = relative_fro_error(current, prediction)
            current_centered = current - current.mean(
                axis=(0, 1),
                keepdims=True,
            )
            prediction_centered = prediction - prediction.mean(
                axis=(0, 1),
                keepdims=True,
            )
            centered_error = relative_fro_error(
                current_centered,
                prediction_centered,
            )
            transport_rows.append(
                {
                    "layer": layer,
                    "frame": frame,
                    "method": method,
                    "scope": scope,
                    "relative_error": error,
                    "centered_relative_error": centered_error,
                    "pixel_change_score": pixel_change_scores[frame],
                    "pixel_dy": pixel_dy,
                    "pixel_dx": pixel_dx,
                    "hidden_dy": hidden_dy,
                    "hidden_dx": hidden_dx,
                    "optical_flow_mean_magnitude": (
                        optical_flow_metrics[frame]["mean_magnitude"]
                        if optical_flow_metrics is not None
                        else None
                    ),
                }
            )
            if method in {
                "identity",
                "pixel_shift_zero",
                "optical_flow_warp",
                "local_bccb_causal",
                "local_bttb_causal",
            }:
                concentration = residual_concentration(
                    current - prediction,
                    block_size,
                    top_fractions,
                    event_mask=pixel_event_masks[frame],
                )
                for fraction, values in concentration["top"].items():
                    residual_rows.append(
                        {
                            "layer": layer,
                            "frame": frame,
                            "method": method,
                            "fraction": float(fraction),
                            "energy_ratio": float(values["energy_ratio"]),
                            "pixel_change_recall": float(
                                values["event_recall"]
                            ),
                            "gini": float(concentration["gini"]),
                            "pixel_change_score": pixel_change_scores[frame],
                        }
                    )

        previous_local_bccb_weights = bccb_weights
        previous_local_bttb_weights = bttb_weights
        previous_global_kernel = global_kernel

    change_values = np.asarray(pixel_change_scores[1:], dtype=np.float64)
    stable_threshold = float(np.median(change_values))
    event_threshold = float(np.quantile(change_values, 0.75))
    method_summary: dict[str, dict[str, float | None]] = {}
    for method in sorted({str(row["method"]) for row in transport_rows}):
        selected = [row for row in transport_rows if row["method"] == method]
        stable = [
            row
            for row in selected
            if float(row["pixel_change_score"]) <= stable_threshold
        ]
        event_like = [
            row
            for row in selected
            if float(row["pixel_change_score"]) >= event_threshold
        ]
        method_summary[method] = {
            "mean_error": mean_or_none(
                [float(row["relative_error"]) for row in selected]
            ),
            "stable_error": mean_or_none(
                [float(row["relative_error"]) for row in stable]
            ),
            "event_like_error": mean_or_none(
                [float(row["relative_error"]) for row in event_like]
            ),
            "mean_centered_error": mean_or_none(
                [float(row["centered_relative_error"]) for row in selected]
            ),
            "stable_centered_error": mean_or_none(
                [float(row["centered_relative_error"]) for row in stable]
            ),
            "event_like_centered_error": mean_or_none(
                [float(row["centered_relative_error"]) for row in event_like]
            ),
        }

    return {
        "layer": layer,
        "rank_summary": rank_summary,
        "method_summary": method_summary,
        "stable_pixel_change_threshold": stable_threshold,
        "event_pixel_change_threshold": event_threshold,
        "transport_rows": transport_rows,
        "residual_rows": residual_rows,
        "subspace_rows": subspace_rows,
    }


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    data = np.load(args.npz)
    available_layers = sorted(
        int(key.removeprefix("hidden_layer_"))
        for key in data.files
        if key.startswith("hidden_layer_")
    )
    requested_layers = (
        [int(value) for value in args.layers.split(",") if value]
        if args.layers
        else available_layers
    )
    missing = sorted(set(requested_layers) - set(available_layers))
    if missing:
        raise ValueError(f"requested layers are missing from export: {missing}")
    ranks = [int(value) for value in args.ranks.split(",") if value]
    top_fractions = [
        float(value) for value in args.top_fractions.split(",") if value
    ]
    time_grid, height_grid, width_grid = [
        int(value) for value in np.asarray(data["grid_thw"]).reshape(-1)
    ]
    pixel_grid = pool_frames_to_grid(
        data["frames_rgb"],
        time_grid=time_grid,
        height_grid=height_grid,
        width_grid=width_grid,
        temporal_patch=int(metadata.get("temporal_patch", 2)),
    )
    pixel_steps, pixel_shift_errors = motion_steps(
        pixel_grid,
        max_shift=args.max_shift,
        cyclic=False,
    )
    pixel_event_masks, pixel_change_scores = pixel_change_proxy(
        pixel_grid,
        pixel_steps,
    )
    optical_flows = None
    optical_flow_metrics = None
    if args.use_optical_flow:
        optical_flows, optical_flow_metrics = optical_flow_fields(
            data["frames_rgb"],
            time_grid=time_grid,
            height_grid=height_grid,
            width_grid=width_grid,
            temporal_patch=int(metadata.get("temporal_patch", 2)),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer_summaries = {}
    all_transport_rows: list[dict[str, object]] = []
    all_residual_rows: list[dict[str, object]] = []
    all_subspace_rows: list[dict[str, object]] = []
    for layer in requested_layers:
        analysis = analyze_layer(
            data[f"hidden_layer_{layer}"],
            pixel_grid,
            pixel_steps,
            pixel_event_masks,
            pixel_change_scores,
            optical_flows,
            optical_flow_metrics,
            layer=layer,
            ranks=ranks,
            max_shift=args.max_shift,
            local_radius=args.local_radius,
            block_size=args.block_size,
            top_fractions=top_fractions,
            ridge=args.ridge,
            history_window=args.history_window,
        )
        layer_summaries[str(layer)] = {
            key: value
            for key, value in analysis.items()
            if key not in {"transport_rows", "residual_rows", "subspace_rows"}
        }
        all_transport_rows.extend(analysis["transport_rows"])
        all_residual_rows.extend(analysis["residual_rows"])
        all_subspace_rows.extend(analysis["subspace_rows"])

    result = {
        "source_npz": str(args.npz.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "model_dir": metadata.get("model_dir"),
        "video": metadata.get("video"),
        "frame_indices": metadata.get("frame_indices"),
        "grid_thw": [time_grid, height_grid, width_grid],
        "ranks": ranks,
        "pixel_steps": [[int(dy), int(dx)] for dy, dx in pixel_steps],
        "pixel_shift_errors": pixel_shift_errors,
        "pixel_change_scores": pixel_change_scores,
        "optical_flow_enabled": args.use_optical_flow,
        "optical_flow_metrics": optical_flow_metrics,
        "layers": layer_summaries,
        "scope": (
            "Activation diagnostics. Pair-fitted operators are oracle upper bounds; "
            "causal operators reuse only the previous transition. Pixel-change recall "
            "is a motion/change proxy and not semantic event recall."
        ),
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "transport_metrics.csv", all_transport_rows)
    write_csv(args.out_dir / "residual_metrics.csv", all_residual_rows)
    write_csv(args.out_dir / "causal_subspace_metrics.csv", all_subspace_rows)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "layers": requested_layers,
                "grid_thw": result["grid_thw"],
                "transport_rows": len(all_transport_rows),
                "residual_rows": len(all_residual_rows),
                "subspace_rows": len(all_subspace_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
