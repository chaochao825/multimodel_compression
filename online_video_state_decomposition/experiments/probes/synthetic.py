from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transport import shift_grid


@dataclass(frozen=True)
class SyntheticSequence:
    observed: np.ndarray
    clean: np.ndarray
    event_mask: np.ndarray
    shifts: np.ndarray
    event_frames: tuple[int, ...]
    scene_cut_frame: int | None
    metadata: dict[str, object]


def _smooth_spatial_basis(
    rng: np.random.Generator,
    height: int,
    width: int,
    rank: int,
) -> np.ndarray:
    basis = rng.normal(size=(height, width, rank))
    for _ in range(5):
        basis = (
            4.0 * basis
            + np.roll(basis, 1, axis=0)
            + np.roll(basis, -1, axis=0)
            + np.roll(basis, 1, axis=1)
            + np.roll(basis, -1, axis=1)
        ) / 8.0
    flat = basis.reshape(height * width, rank)
    q, _ = np.linalg.qr(flat)
    return q[:, :rank].reshape(height, width, rank)


def _temporal_coefficients(
    frame: int,
    total_frames: int,
    modes: int,
) -> np.ndarray:
    phase = 2.0 * np.pi * frame / max(total_frames - 1, 1)
    values = [1.0]
    for mode in range(1, modes + 1):
        values.append(np.sin(mode * phase))
        values.append(np.cos(mode * phase))
    return np.asarray(values, dtype=np.float64)


def generate_synthetic_sequence(
    *,
    seed: int,
    frames: int,
    height: int,
    width: int,
    hidden_dim: int,
    spatial_rank: int,
    temporal_modes: int,
    event_frames: tuple[int, ...],
    scene_cut_frame: int | None,
    event_block_size: int,
    event_amplitude: float,
    max_step_shift: int,
    constant_step: tuple[int, int] | None = None,
) -> SyntheticSequence:
    rng = np.random.default_rng(seed)
    basis = _smooth_spatial_basis(rng, height, width, spatial_rank)
    mode_count = 1 + 2 * temporal_modes
    feature_modes = rng.normal(size=(mode_count, spatial_rank, hidden_dim))
    feature_modes /= np.linalg.norm(feature_modes, axis=2, keepdims=True) + 1e-12

    clean = np.zeros((frames, height, width, hidden_dim), dtype=np.float64)
    observed = np.zeros_like(clean)
    event_mask = np.zeros((frames, height, width), dtype=bool)
    shifts = np.zeros((frames, 2), dtype=np.int64)

    event_locations: dict[int, tuple[int, int]] = {}
    for event_frame in event_frames:
        max_y = max(height - event_block_size, 0)
        max_x = max(width - event_block_size, 0)
        event_locations[event_frame] = (
            int(rng.integers(0, max_y + 1)),
            int(rng.integers(0, max_x + 1)),
        )

    for frame in range(frames):
        if frame > 0:
            step = (
                np.asarray(constant_step, dtype=np.int64)
                if constant_step is not None
                else rng.integers(
                    -max_step_shift,
                    max_step_shift + 1,
                    size=2,
                )
            )
            shifts[frame] = shifts[frame - 1] + step

        if scene_cut_frame is not None and frame == scene_cut_frame:
            basis = _smooth_spatial_basis(rng, height, width, spatial_rank)
            feature_modes = rng.normal(
                size=(mode_count, spatial_rank, hidden_dim)
            )
            feature_modes /= (
                np.linalg.norm(feature_modes, axis=2, keepdims=True) + 1e-12
            )

        coefficients = _temporal_coefficients(frame, frames, temporal_modes)
        latent_features = np.tensordot(
            coefficients,
            feature_modes,
            axes=(0, 0),
        )
        canonical = np.einsum("hwr,rd->hwd", basis, latent_features)
        clean[frame] = shift_grid(
            canonical,
            int(shifts[frame, 0]),
            int(shifts[frame, 1]),
            cyclic=True,
        )
        observed[frame] = clean[frame]

        if frame in event_locations:
            y, x = event_locations[frame]
            event_feature = rng.normal(size=(hidden_dim,))
            event_feature /= np.linalg.norm(event_feature) + 1e-12
            observed[
                frame,
                y : y + event_block_size,
                x : x + event_block_size,
            ] += event_amplitude * event_feature
            event_mask[
                frame,
                y : y + event_block_size,
                x : x + event_block_size,
            ] = True

        if scene_cut_frame is not None and frame == scene_cut_frame:
            event_mask[frame] = True

    return SyntheticSequence(
        observed=observed,
        clean=clean,
        event_mask=event_mask,
        shifts=shifts,
        event_frames=event_frames,
        scene_cut_frame=scene_cut_frame,
        metadata={
            "seed": seed,
            "frames": frames,
            "height": height,
            "width": width,
            "hidden_dim": hidden_dim,
            "spatial_rank": spatial_rank,
            "temporal_modes": temporal_modes,
            "event_frames": list(event_frames),
            "scene_cut_frame": scene_cut_frame,
            "event_block_size": event_block_size,
            "event_amplitude": event_amplitude,
            "max_step_shift": max_step_shift,
            "constant_step": list(constant_step) if constant_step is not None else None,
        },
    )
