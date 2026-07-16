from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHODS = (
    "recent_window",
    "uniform_reservoir",
    "adaptive_slots",
    "oja_subspace",
    "instant_oja",
)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(values),
    )


def softmax_pool_scores(
    text_vectors: np.ndarray,
    prototypes: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    text_vectors = normalize_rows(text_vectors)
    prototypes = normalize_rows(prototypes)
    if prototypes.shape[0] == 0:
        return np.zeros(text_vectors.shape[0], dtype=np.float64)
    logits = temperature * (text_vectors @ prototypes.T)
    maxima = np.max(logits, axis=1, keepdims=True)
    pooled = maxima[:, 0] + np.log(
        np.mean(np.exp(logits - maxima), axis=1)
    )
    return pooled / temperature


class VectorMemory:
    def update(self, values: np.ndarray) -> None:
        raise NotImplementedError

    def prototypes(self) -> np.ndarray:
        raise NotImplementedError

    def score(
        self,
        text_vectors: np.ndarray,
        *,
        temperature: float,
    ) -> np.ndarray:
        return softmax_pool_scores(
            text_vectors,
            self.prototypes(),
            temperature=temperature,
        )


class RecentWindowMemory(VectorMemory):
    def __init__(self, capacity: int, hidden_dim: int) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.values = np.empty((0, hidden_dim), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = normalize_rows(values)
        self.values = np.concatenate((self.values, values), axis=0)[
            -self.capacity :
        ]

    def prototypes(self) -> np.ndarray:
        return self.values


class ReservoirMemory(VectorMemory):
    def __init__(self, capacity: int, hidden_dim: int, *, seed: int) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.values = np.empty((0, hidden_dim), dtype=np.float64)
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        for value in normalize_rows(values):
            self.seen += 1
            if self.values.shape[0] < self.capacity:
                self.values = np.concatenate(
                    (self.values, value[None]),
                    axis=0,
                )
                continue
            replacement = int(self.rng.integers(0, self.seen))
            if replacement < self.capacity:
                self.values[replacement] = value

    def prototypes(self) -> np.ndarray:
        return self.values


class AdaptiveSlotMemory(VectorMemory):
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        min_lr: float,
        replace_similarity: float,
    ) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.min_lr = min_lr
        self.replace_similarity = replace_similarity
        self.values = np.empty((0, hidden_dim), dtype=np.float64)
        self.counts = np.empty((0,), dtype=np.int64)

    def _redundant_slot(self) -> int:
        similarity = self.values @ self.values.T
        np.fill_diagonal(similarity, -np.inf)
        left, right = np.unravel_index(
            int(np.argmax(similarity)),
            similarity.shape,
        )
        return (
            int(left)
            if self.counts[left] <= self.counts[right]
            else int(right)
        )

    def update(self, values: np.ndarray) -> None:
        for value in normalize_rows(values):
            if self.values.shape[0] < self.capacity:
                self.values = np.concatenate(
                    (self.values, value[None]),
                    axis=0,
                )
                self.counts = np.concatenate(
                    (self.counts, np.ones(1, dtype=np.int64))
                )
                continue
            similarity = self.values @ value
            index = int(np.argmax(similarity))
            if float(similarity[index]) < self.replace_similarity:
                index = self._redundant_slot()
                self.values[index] = value
                self.counts[index] = 1
                continue
            self.counts[index] += 1
            rate = max(
                self.min_lr,
                1.0 / np.sqrt(float(self.counts[index])),
            )
            self.values[index] = normalize_rows(
                (1.0 - rate) * self.values[index] + rate * value
            )[0]

    def prototypes(self) -> np.ndarray:
        return self.values


class OjaEllipsoidMemory(VectorMemory):
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        learning_rate: float,
        prototype_scale: float,
    ) -> None:
        self.capacity = capacity
        self.rank = max(0, capacity - 1)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.prototype_scale = prototype_scale
        self.count = 0
        self.mean = np.zeros(hidden_dim, dtype=np.float64)
        self.basis = np.empty((hidden_dim, 0), dtype=np.float64)
        self.variance = np.empty((0,), dtype=np.float64)

    def _append_direction(self, vector: np.ndarray) -> bool:
        residual = vector.copy()
        if self.basis.shape[1]:
            residual -= self.basis @ (self.basis.T @ residual)
        norm = float(np.linalg.norm(residual))
        if norm <= 1e-8:
            return False
        direction = residual / norm
        self.basis = np.concatenate(
            (self.basis, direction[:, None]),
            axis=1,
        )
        self.variance = np.concatenate(
            (self.variance, np.asarray([norm * norm])),
        )
        return True

    def update(self, values: np.ndarray) -> None:
        for value in normalize_rows(values):
            self.count += 1
            previous_mean = self.mean.copy()
            self.mean += (value - self.mean) / float(self.count)
            centered = value - previous_mean
            if self.rank == 0:
                continue
            if self.basis.shape[1] < self.rank:
                if self._append_direction(centered):
                    continue
            if self.basis.shape[1] == 0:
                continue
            projected = self.basis.T @ (value - self.mean)
            rate = self.learning_rate / np.sqrt(float(self.count))
            covariance_action = np.outer(value - self.mean, projected)
            in_span = self.basis @ np.outer(projected, projected)
            candidate = self.basis + rate * (
                covariance_action - in_span
            )
            self.basis = np.linalg.qr(candidate, mode="reduced")[0][
                :, : self.rank
            ]
            projected = self.basis.T @ (value - self.mean)
            if self.variance.shape[0] != self.basis.shape[1]:
                self.variance = np.ones(
                    self.basis.shape[1],
                    dtype=np.float64,
                )
            self.variance += (
                projected * projected - self.variance
            ) / float(self.count)

    def prototypes(self) -> np.ndarray:
        if self.count == 0:
            return np.empty((0, self.hidden_dim), dtype=np.float64)
        output = [self.mean]
        if self.basis.shape[1]:
            scales = np.sqrt(np.maximum(self.variance, 1e-12))
            scales /= max(float(np.max(scales)), 1e-12)
            for index in range(self.basis.shape[1]):
                offset = (
                    self.prototype_scale
                    * scales[index]
                    * self.basis[:, index]
                )
                output.extend((self.mean + offset, self.mean - offset))
        return normalize_rows(np.stack(output))


class InstantOjaMemory(VectorMemory):
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        instant_capacity: int,
        learning_rate: float,
        prototype_scale: float,
    ) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.instant_capacity = min(instant_capacity, capacity)
        self.instant = np.empty((0, hidden_dim), dtype=np.float64)
        long_capacity = capacity - self.instant_capacity
        self.long_term = (
            OjaEllipsoidMemory(
                long_capacity,
                hidden_dim,
                learning_rate=learning_rate,
                prototype_scale=prototype_scale,
            )
            if long_capacity > 0
            else None
        )

    def update(self, values: np.ndarray) -> None:
        combined = np.concatenate(
            (self.instant, normalize_rows(values)),
            axis=0,
        )
        overflow = max(0, combined.shape[0] - self.instant_capacity)
        if self.long_term is not None and overflow:
            self.long_term.update(combined[:overflow])
        self.instant = combined[-self.instant_capacity :].copy()

    def prototypes(self) -> np.ndarray:
        if self.long_term is None:
            return self.instant
        return np.concatenate(
            (self.instant, self.long_term.prototypes()),
            axis=0,
        )


@dataclass(frozen=True)
class StateAccounting:
    payload_bytes: int
    metadata_bytes: int
    total_state_bytes: int
    read_flops_per_option: int
    update_flops_per_frame: int


def state_accounting(
    method: str,
    *,
    capacity: int,
    hidden_dim: int,
    storage_bits: int,
    instant_capacity: int,
) -> StateAccounting:
    payload_bytes = capacity * hidden_dim * storage_bits // 8
    if method in {"recent_window", "uniform_reservoir"}:
        metadata_bytes = 16
        read_flops = 2 * capacity * hidden_dim
        update_flops = 0
    elif method == "adaptive_slots":
        metadata_bytes = 8 * capacity + 8
        read_flops = 2 * capacity * hidden_dim
        update_flops = 2 * capacity * hidden_dim + 4 * hidden_dim
    elif method == "oja_subspace":
        rank = max(0, capacity - 1)
        metadata_bytes = 8 + 4 * rank
        read_flops = 4 * capacity * hidden_dim
        update_flops = 6 * rank * hidden_dim
    elif method == "instant_oja":
        long_capacity = capacity - min(instant_capacity, capacity)
        rank = max(0, long_capacity - 1)
        metadata_bytes = 24 + 4 * rank
        read_flops = 2 * instant_capacity * hidden_dim
        read_flops += 4 * long_capacity * hidden_dim
        update_flops = 6 * rank * hidden_dim
    else:
        raise ValueError(f"unknown task memory method: {method}")
    return StateAccounting(
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_state_bytes=payload_bytes + metadata_bytes,
        read_flops_per_option=read_flops,
        update_flops_per_frame=update_flops,
    )


def make_memory(
    method: str,
    *,
    capacity: int,
    hidden_dim: int,
    seed: int,
    instant_capacity: int,
    oja_lr: float,
    prototype_scale: float,
    slot_min_lr: float,
    slot_replace_similarity: float,
) -> VectorMemory:
    if method == "recent_window":
        return RecentWindowMemory(capacity, hidden_dim)
    if method == "uniform_reservoir":
        return ReservoirMemory(capacity, hidden_dim, seed=seed)
    if method == "adaptive_slots":
        return AdaptiveSlotMemory(
            capacity,
            hidden_dim,
            min_lr=slot_min_lr,
            replace_similarity=slot_replace_similarity,
        )
    if method == "oja_subspace":
        return OjaEllipsoidMemory(
            capacity,
            hidden_dim,
            learning_rate=oja_lr,
            prototype_scale=prototype_scale,
        )
    if method == "instant_oja":
        return InstantOjaMemory(
            capacity,
            hidden_dim,
            instant_capacity=instant_capacity,
            learning_rate=oja_lr,
            prototype_scale=prototype_scale,
        )
    raise ValueError(f"unknown task memory method: {method}")
