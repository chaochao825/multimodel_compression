from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np


SCENARIOS = (
    "static",
    "camera_slow",
    "camera_fast",
    "lighting_drift",
    "periodic_motion",
    "object_enter",
    "object_disappear",
    "scene_cut",
    "one_frame_ocr",
    "brief_action",
)

NEGATIVE_SCENARIOS = frozenset(
    {
        "static",
        "camera_slow",
        "camera_fast",
        "lighting_drift",
        "periodic_motion",
    }
)
CAMERA_SCENARIOS = frozenset({"camera_slow", "camera_fast"})
RARE_EVENT_SCENARIOS = frozenset({"one_frame_ocr", "brief_action"})
OBJECT_EVENT_SCENARIOS = frozenset({"object_enter", "object_disappear"})


@dataclass(frozen=True)
class ControlledSequence:
    scenario: str
    values: np.ndarray
    event_onsets: tuple[int, ...]
    metadata: dict[str, object]


def _smooth_coefficients(
    rng: np.random.Generator,
    height: int,
    width: int,
    rank: int,
) -> np.ndarray:
    values = rng.normal(size=(height, width, rank))
    for _ in range(4):
        values = (
            4.0 * values
            + np.roll(values, 1, axis=0)
            + np.roll(values, -1, axis=0)
            + np.roll(values, 1, axis=1)
            + np.roll(values, -1, axis=1)
        ) / 8.0
    flat = values.reshape(height * width, rank)
    flat /= np.maximum(np.linalg.norm(flat, axis=0, keepdims=True), 1e-12)
    return flat


def _feature_basis(
    rng: np.random.Generator,
    hidden_dim: int,
    rank: int,
) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(hidden_dim, rank)), mode="reduced")[0]


def _event_direction(
    rng: np.random.Generator,
    basis: np.ndarray,
) -> np.ndarray:
    value = rng.normal(size=(basis.shape[0],))
    value -= basis @ (basis.T @ value)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise RuntimeError("failed to create an event direction")
    return value / norm


def _add_block(
    frame: np.ndarray,
    direction: np.ndarray,
    *,
    y: int,
    x: int,
    block_size: int,
    amplitude: float,
) -> None:
    height, width = frame.shape[:2]
    y = min(max(y, 0), max(0, height - block_size))
    x = min(max(x, 0), max(0, width - block_size))
    frame[y : y + block_size, x : x + block_size] += amplitude * direction


def _camera_shift(scenario: str, frame: int) -> tuple[int, int]:
    if scenario == "camera_slow":
        return frame // 4, -(frame // 5)
    if scenario == "camera_fast":
        return frame, -frame
    if scenario not in NEGATIVE_SCENARIOS:
        return frame // 8, -(frame // 10)
    return 0, 0


def generate_controlled_sequence(
    *,
    scenario: str,
    seed: int,
    frames: int,
    height: int,
    width: int,
    hidden_dim: int,
    base_rank: int,
    event_frame: int,
    event_block_size: int,
    event_amplitude: float,
    noise_std: float,
) -> ControlledSequence:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown controlled scenario: {scenario}")
    if not 0 < base_rank < hidden_dim:
        raise ValueError("base_rank must be in (0, hidden_dim)")
    if not 0 <= event_frame < frames:
        raise ValueError("event_frame must lie inside the sequence")

    rng = np.random.default_rng(seed)
    coefficients = _smooth_coefficients(rng, height, width, base_rank)
    basis = _feature_basis(rng, hidden_dim, base_rank)
    base = (coefficients @ basis.T).reshape(height, width, hidden_dim)
    base /= np.sqrt(np.mean(base**2) + 1e-12)

    cut_coefficients = _smooth_coefficients(rng, height, width, base_rank)
    cut_basis = _feature_basis(rng, hidden_dim, base_rank)
    cut = (cut_coefficients @ cut_basis.T).reshape(height, width, hidden_dim)
    cut /= np.sqrt(np.mean(cut**2) + 1e-12)
    event_direction = _event_direction(rng, basis)
    output = np.empty((frames, height * width, hidden_dim), dtype=np.float64)

    event_onsets: tuple[int, ...]
    if scenario in NEGATIVE_SCENARIOS:
        event_onsets = ()
    else:
        event_onsets = (event_frame,)

    fixed_y = max(0, height // 3 - event_block_size // 2)
    fixed_x = max(0, width // 2 - event_block_size // 2)
    for frame_index in range(frames):
        current = (
            cut.copy()
            if scenario == "scene_cut" and frame_index >= event_frame
            else base.copy()
        )
        if scenario == "lighting_drift":
            scale = 0.8 + 0.4 * frame_index / max(frames - 1, 1)
            current *= scale
        if scenario == "periodic_motion":
            y = (frame_index // 2) % max(1, height - event_block_size + 1)
            x = (2 * frame_index) % max(1, width - event_block_size + 1)
            _add_block(
                current,
                event_direction,
                y=y,
                x=x,
                block_size=event_block_size,
                amplitude=0.6 * event_amplitude,
            )
        if scenario == "object_enter" and frame_index >= event_frame:
            _add_block(
                current,
                event_direction,
                y=fixed_y,
                x=fixed_x,
                block_size=event_block_size,
                amplitude=event_amplitude,
            )
        if scenario == "object_disappear" and frame_index < event_frame:
            _add_block(
                current,
                event_direction,
                y=fixed_y,
                x=fixed_x,
                block_size=event_block_size,
                amplitude=event_amplitude,
            )
        if scenario == "one_frame_ocr" and frame_index == event_frame:
            _add_block(
                current,
                event_direction,
                y=fixed_y,
                x=fixed_x,
                block_size=event_block_size,
                amplitude=1.25 * event_amplitude,
            )
        if scenario == "brief_action" and event_frame <= frame_index < event_frame + 3:
            direction = event_direction * (1.0 if frame_index % 2 == 0 else -1.0)
            _add_block(
                current,
                direction,
                y=fixed_y,
                x=fixed_x,
                block_size=event_block_size,
                amplitude=event_amplitude,
            )

        if noise_std > 0:
            current += rng.normal(scale=noise_std, size=current.shape)
        dy, dx = _camera_shift(scenario, frame_index)
        current = np.roll(current, shift=(dy, dx), axis=(0, 1))
        output[frame_index] = current.reshape(height * width, hidden_dim)

    return ControlledSequence(
        scenario=scenario,
        values=output,
        event_onsets=event_onsets,
        metadata={
            "seed": seed,
            "frames": frames,
            "height": height,
            "width": width,
            "hidden_dim": hidden_dim,
            "base_rank": base_rank,
            "event_frame": event_frame,
            "event_block_size": event_block_size,
            "event_amplitude": event_amplitude,
            "noise_std": noise_std,
        },
    )


def residual_ratio(values: np.ndarray, basis: np.ndarray | None) -> float:
    if basis is None:
        return 0.0
    projection = values @ basis
    residual = values - projection @ basis.T
    return float(
        np.sum(residual**2) / max(float(np.sum(values**2)), 1e-12)
    )


class OnlineOjaSubspace:
    def __init__(
        self,
        *,
        rank: int,
        hidden_dim: int,
        beta: float,
        learning_rate: float,
        seed: int,
    ) -> None:
        if not 0 < rank <= hidden_dim:
            raise ValueError("rank must be in [1, hidden_dim]")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0,1)")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self.basis: np.ndarray | None = None
        self.spectrum: np.ndarray | None = None

    def _initialize(self, values: np.ndarray) -> None:
        _, _, right = np.linalg.svd(values, full_matrices=False)
        retained = min(self.rank, right.shape[0])
        basis = right[:retained].T
        while basis.shape[1] < self.rank:
            candidate = self._rng.normal(
                size=(self.hidden_dim, self.rank - basis.shape[1] + 2)
            )
            if basis.size:
                candidate -= basis @ (basis.T @ candidate)
            extra = np.linalg.qr(candidate, mode="reduced")[0]
            basis = np.concatenate(
                (basis, extra[:, : self.rank - basis.shape[1]]), axis=1
            )
        self.basis = basis[:, : self.rank]
        self.spectrum = np.mean((values @ self.basis) ** 2, axis=0)

    def update(self, values: np.ndarray) -> None:
        if self.basis is None:
            self._initialize(values)
            return
        projected = values @ self.basis
        covariance_action = values.T @ projected / values.shape[0]
        in_span = self.basis @ (projected.T @ projected / values.shape[0])
        candidate = self.basis + self.learning_rate * (
            covariance_action - in_span
        )
        self.basis = np.linalg.qr(candidate, mode="reduced")[0][
            :, : self.rank
        ]
        instantaneous = np.mean((values @ self.basis) ** 2, axis=0)
        assert self.spectrum is not None
        self.spectrum = (
            self.beta * self.spectrum + (1.0 - self.beta) * instantaneous
        )


class CausalActivityBasis:
    """Independent CausalMem-style activity-basis trigger proxy."""

    def __init__(
        self,
        *,
        rank: int,
        hidden_dim: int,
        activity_beta: float,
        max_new_directions: int,
    ) -> None:
        self.rank = min(rank, hidden_dim)
        self.hidden_dim = hidden_dim
        self.activity_beta = activity_beta
        self.max_new_directions = max_new_directions
        self.basis: np.ndarray | None = None
        self.activity: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        if self.basis is None:
            _, _, right = np.linalg.svd(values, full_matrices=False)
            retained = min(self.rank, right.shape[0])
            self.basis = right[:retained].T
            self.activity = np.mean(np.abs(values @ self.basis), axis=0)
            return

        assert self.activity is not None
        projection = values @ self.basis
        current_activity = np.mean(np.abs(projection), axis=0)
        self.activity = (
            self.activity_beta * self.activity
            + (1.0 - self.activity_beta) * current_activity
        )
        residual = values - projection @ self.basis.T
        residual_energy = np.sum(residual**2, axis=1)
        candidates = np.argsort(residual_energy)[::-1][
            : self.max_new_directions
        ]
        for candidate_index in candidates:
            direction = residual[candidate_index].copy()
            replace = int(np.argmin(self.activity))
            keep = np.arange(self.basis.shape[1]) != replace
            other = self.basis[:, keep]
            if other.size:
                direction -= other @ (other.T @ direction)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                continue
            self.basis[:, replace] = direction / norm
            self.activity[replace] = float(np.sqrt(residual_energy[candidate_index]))


def dual_spectral_components(
    values: np.ndarray,
    fast: OnlineOjaSubspace,
    slow: OnlineOjaSubspace,
) -> dict[str, float]:
    residual = residual_ratio(values, slow.basis)
    if fast.basis is None or slow.basis is None:
        return {
            "residual_component": residual,
            "angle_component": 0.0,
            "spectrum_component": 0.0,
        }
    overlap = fast.basis.T @ slow.basis
    angle = max(
        0.0,
        1.0 - float(np.sum(overlap**2)) / max(fast.rank, 1),
    )
    assert fast.spectrum is not None and slow.spectrum is not None
    spectrum = float(
        np.mean(
            np.abs(
                np.log(fast.spectrum + 1e-8)
                - np.log(slow.spectrum + 1e-8)
            )
        )
    )
    return {
        "residual_component": residual,
        "angle_component": angle,
        "spectrum_component": spectrum,
    }


def _basis_state_bytes(
    rank: int,
    hidden_dim: int,
    storage_bits: int,
    scalar_count: int,
) -> int:
    return rank * hidden_dim * storage_bits // 8 + 4 * scalar_count


def trace_controlled_sequence(
    sequence: ControlledSequence,
    *,
    total_rank_budget: int,
    storage_bits: int,
    fast_beta: float,
    slow_beta: float,
    fast_learning_rate: float,
    slow_learning_rate: float,
    causal_activity_beta: float,
    causal_max_new_directions: int,
) -> list[dict[str, object]]:
    values = sequence.values
    hidden_dim = values.shape[2]
    token_count = values.shape[1]
    dual_rank = max(1, total_rank_budget // 2)
    causal = CausalActivityBasis(
        rank=total_rank_budget,
        hidden_dim=hidden_dim,
        activity_beta=causal_activity_beta,
        max_new_directions=causal_max_new_directions,
    )
    single = OnlineOjaSubspace(
        rank=total_rank_budget,
        hidden_dim=hidden_dim,
        beta=slow_beta,
        learning_rate=slow_learning_rate,
        seed=17,
    )
    fast = OnlineOjaSubspace(
        rank=dual_rank,
        hidden_dim=hidden_dim,
        beta=fast_beta,
        learning_rate=fast_learning_rate,
        seed=23,
    )
    slow = OnlineOjaSubspace(
        rank=dual_rank,
        hidden_dim=hidden_dim,
        beta=slow_beta,
        learning_rate=slow_learning_rate,
        seed=23,
    )
    rows: list[dict[str, object]] = []
    previous: np.ndarray | None = None

    frame_delta_bytes = token_count * hidden_dim * storage_bits // 8
    causal_bytes = _basis_state_bytes(
        total_rank_budget,
        hidden_dim,
        storage_bits,
        total_rank_budget,
    )
    single_bytes = _basis_state_bytes(
        total_rank_budget,
        hidden_dim,
        storage_bits,
        total_rank_budget,
    )
    dual_bytes = _basis_state_bytes(
        2 * dual_rank,
        hidden_dim,
        storage_bits,
        2 * dual_rank,
    )

    for frame_index, current in enumerate(values):
        start = perf_counter_ns()
        frame_delta = (
            0.0
            if previous is None
            else float(
                np.sum((current - previous) ** 2)
                / max(float(np.sum(current**2)), 1e-12)
            )
        )
        previous = current.copy()
        frame_delta_us = (perf_counter_ns() - start) / 1000.0

        start = perf_counter_ns()
        causal_score = residual_ratio(current, causal.basis)
        causal.update(current)
        causal_us = (perf_counter_ns() - start) / 1000.0

        start = perf_counter_ns()
        single_score = residual_ratio(current, single.basis)
        single.update(current)
        single_us = (perf_counter_ns() - start) / 1000.0

        start = perf_counter_ns()
        pre_update_residual = residual_ratio(current, slow.basis)
        fast.update(current)
        slow.update(current)
        components = dual_spectral_components(current, fast, slow)
        components["residual_component"] = pre_update_residual
        dual_us = (perf_counter_ns() - start) / 1000.0

        common = {
            "scenario": sequence.scenario,
            "seed": int(sequence.metadata["seed"]),
            "frame": frame_index,
            "total_rank_budget": total_rank_budget,
            "dual_rank_each": dual_rank,
            "is_event_onset": int(frame_index in sequence.event_onsets),
        }
        rows.extend(
            [
                {
                    **common,
                    "method": "frame_delta",
                    "raw_score": frame_delta,
                    "residual_component": frame_delta,
                    "angle_component": 0.0,
                    "spectrum_component": 0.0,
                    "update_us": frame_delta_us,
                    "state_bytes": frame_delta_bytes,
                    "estimated_update_flops": token_count * hidden_dim,
                },
                {
                    **common,
                    "method": "causalmem_residual_proxy",
                    "raw_score": causal_score,
                    "residual_component": causal_score,
                    "angle_component": 0.0,
                    "spectrum_component": 0.0,
                    "update_us": causal_us,
                    "state_bytes": causal_bytes,
                    "estimated_update_flops": (
                        4 * token_count * hidden_dim * total_rank_budget
                    ),
                },
                {
                    **common,
                    "method": "single_oja_residual",
                    "raw_score": single_score,
                    "residual_component": single_score,
                    "angle_component": 0.0,
                    "spectrum_component": 0.0,
                    "update_us": single_us,
                    "state_bytes": single_bytes,
                    "estimated_update_flops": (
                        6 * token_count * hidden_dim * total_rank_budget
                    ),
                },
                {
                    **common,
                    "method": "dual_slow_residual",
                    "raw_score": components["residual_component"],
                    **components,
                    "update_us": dual_us,
                    "state_bytes": dual_bytes,
                    "estimated_update_flops": (
                        6 * token_count * hidden_dim * 2 * dual_rank
                    ),
                },
                {
                    **common,
                    "method": "dual_spectral",
                    "raw_score": 0.0,
                    **components,
                    "update_us": dual_us,
                    "state_bytes": dual_bytes,
                    "estimated_update_flops": (
                        6 * token_count * hidden_dim * 2 * dual_rank
                        + hidden_dim * dual_rank * dual_rank
                    ),
                },
            ]
        )
    return rows
