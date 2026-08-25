from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha
    return np.linalg.solve(gram, x.T @ y)


class RidgeRegressor:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        self.y_mean = y.mean(axis=0)
        xs = (x - self.x_mean) / self.x_scale
        self.weight = _ridge_solve(xs, y - self.y_mean, self.alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_scale @ self.weight + self.y_mean

    @property
    def parameter_count(self) -> int:
        return int(self.weight.size + self.y_mean.size)


class ReducedRankRegressor:
    def __init__(self, rank: int, alpha: float):
        self.rank = rank
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ReducedRankRegressor":
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        self.y_mean = y.mean(axis=0)
        xs = (x - self.x_mean) / self.x_scale
        yc = y - self.y_mean
        full_weight = _ridge_solve(xs, yc, self.alpha)
        fitted = xs @ full_weight
        _, _, vt = np.linalg.svd(fitted, full_matrices=False)
        self.output_basis = vt[: self.rank]
        coefficients = yc @ self.output_basis.T
        self.coefficient_weight = _ridge_solve(xs, coefficients, self.alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.x_mean) / self.x_scale
        return xs @ self.coefficient_weight @ self.output_basis + self.y_mean

    @property
    def parameter_count(self) -> int:
        return int(
            self.coefficient_weight.size
            + self.output_basis.size
            + self.y_mean.size
        )


def transition_view(states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    n, state_count, width = states.shape
    transition_count = state_count - 1
    action_width = actions.shape[-1]
    current = states[:, :-1].reshape(n * transition_count, width)
    previous_1 = np.zeros_like(current)
    previous_2 = np.zeros_like(current)
    positions = np.zeros((n * transition_count, transition_count), dtype=states.dtype)
    action_context = np.zeros(
        (n * transition_count, transition_count * action_width), dtype=actions.dtype
    )

    for sample in range(n):
        for step in range(transition_count):
            row = sample * transition_count + step
            positions[row, step] = 1.0
            if step >= 1:
                previous_1[row] = states[sample, step - 1]
            if step >= 2:
                previous_2[row] = states[sample, step - 2]
            used_actions = actions[sample, : step + 1].reshape(-1)
            action_context[row, : used_actions.size] = used_actions

    return {
        "current": current,
        "previous_1": previous_1,
        "previous_2": previous_2,
        "position": positions,
        "action_context": action_context,
    }


def causal_context(sequence: np.ndarray) -> np.ndarray:
    n, step_count, width = sequence.shape
    context = np.zeros((n * step_count, step_count * width), dtype=sequence.dtype)
    for sample in range(n):
        for step in range(step_count):
            row = sample * step_count + step
            values = sequence[sample, : step + 1].reshape(-1)
            context[row, : values.size] = values
    return context


def history_features(view: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            view["current"],
            view["current"] - view["previous_1"],
            view["current"] - view["previous_2"],
        ],
        axis=1,
    )


class IdentityRegressor:
    def fit(self, view: dict[str, np.ndarray], y: np.ndarray) -> "IdentityRegressor":
        self.width = y.shape[1]
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return view["current"]

    @property
    def parameter_count(self) -> int:
        return 0


class VelocityRegressor:
    def fit(self, view: dict[str, np.ndarray], y: np.ndarray) -> "VelocityRegressor":
        self.width = y.shape[1]
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        has_history = (view["position"][:, 0] == 0.0)[:, None]
        velocity = view["current"] - view["previous_1"]
        return view["current"] + has_history * velocity

    @property
    def parameter_count(self) -> int:
        return 0


class DiagonalHistoryRegressor:
    def __init__(self, alpha: float, use_action: bool, interaction_rank: int):
        self.alpha = alpha
        self.use_action = use_action
        self.interaction_rank = interaction_rank

    def _fit_action_basis(self, action: np.ndarray) -> None:
        self.action_mean = action.mean(axis=0)
        centered = action - self.action_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.action_basis = vt[: self.interaction_rank]

    def _features(self, view: dict[str, np.ndarray], channel: int) -> np.ndarray:
        columns = [
            view["current"][:, channel : channel + 1],
            view["previous_1"][:, channel : channel + 1],
            view["previous_2"][:, channel : channel + 1],
            view["position"],
        ]
        if self.use_action:
            action = view["action_context"]
            columns.append(action)
            if self.interaction_rank:
                gate = (action - self.action_mean) @ self.action_basis.T
                columns.append(gate * view["current"][:, channel : channel + 1])
        return np.concatenate(columns, axis=1)

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "DiagonalHistoryRegressor":
        if self.use_action:
            self._fit_action_basis(view["action_context"])
        self.models = []
        for channel in range(y.shape[1]):
            model = RidgeRegressor(self.alpha)
            model.fit(self._features(view, channel), y[:, channel : channel + 1])
            self.models.append(model)
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                model.predict(self._features(view, channel))
                for channel, model in enumerate(self.models)
            ],
            axis=1,
        )

    @property
    def parameter_count(self) -> int:
        return sum(model.parameter_count for model in self.models)


class FrozenConditionDiagonalRegressor:
    def __init__(self, alpha: float, condition_rank: int):
        self.alpha = alpha
        self.condition_rank = condition_rank

    def _fit_basis(self, condition: np.ndarray) -> None:
        self.condition_mean = condition.mean(axis=0)
        centered = condition - self.condition_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.condition_basis = vt[: self.condition_rank]

    def _gates(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return (
            view["condition_context"] - self.condition_mean
        ) @ self.condition_basis.T

    def _features(self, view: dict[str, np.ndarray], channel: int) -> np.ndarray:
        gates = self._gates(view)
        return np.concatenate(
            [
                view["current"][:, channel : channel + 1],
                view["previous_1"][:, channel : channel + 1],
                view["previous_2"][:, channel : channel + 1],
                view["position"],
                gates,
                gates * view["current"][:, channel : channel + 1],
            ],
            axis=1,
        )

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "FrozenConditionDiagonalRegressor":
        self._fit_basis(view["condition_context"])
        self.models = []
        for channel in range(y.shape[1]):
            self.models.append(
                RidgeRegressor(self.alpha).fit(
                    self._features(view, channel), y[:, channel : channel + 1]
                )
            )
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                model.predict(self._features(view, channel))
                for channel, model in enumerate(self.models)
            ],
            axis=1,
        )

    @property
    def parameter_count(self) -> int:
        return sum(model.parameter_count for model in self.models)


class FrozenConditionLowRankRegressor:
    def __init__(self, condition_rank: int, alpha: float):
        self.condition_rank = condition_rank
        self.alpha = alpha

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "FrozenConditionLowRankRegressor":
        self.base = DiagonalHistoryRegressor(self.alpha, False, 0).fit(view, y)
        residual = y - self.base.predict(view)
        self.condition = ReducedRankRegressor(
            self.condition_rank, self.alpha
        ).fit(view["condition_context"], residual)
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return self.base.predict(view) + self.condition.predict(
            view["condition_context"]
        )

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count + self.condition.parameter_count


def _bcm_design(current: np.ndarray, block_size: int) -> np.ndarray:
    n, width = current.shape
    block_count = width // block_size
    blocked = current.reshape(n, block_count, block_size)
    design = np.empty((n, block_size, width), dtype=current.dtype)
    for output_row in range(block_size):
        design[:, output_row] = np.concatenate(
            [
                np.roll(blocked[:, input_block], -output_row, axis=1)
                for input_block in range(block_count)
            ],
            axis=1,
        )
    return design


class BCMRegressor:
    def __init__(
        self,
        block_size: int,
        alpha: float,
        gate_rank: int = 0,
        include_base: bool = True,
        gate_key: str = "action_context",
    ):
        self.block_size = block_size
        self.alpha = alpha
        self.gate_rank = gate_rank
        self.include_base = include_base
        self.gate_key = gate_key

    def _fit_gates(self, action: np.ndarray) -> None:
        self.action_mean = action.mean(axis=0)
        centered = action - self.action_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.action_basis = vt[: self.gate_rank]

    def _gates(self, view: dict[str, np.ndarray]) -> np.ndarray:
        if self.gate_rank == 0:
            return np.empty((view["current"].shape[0], 0))
        return (view[self.gate_key] - self.action_mean) @ self.action_basis.T

    def _features(
        self,
        base_design: np.ndarray,
        gates: np.ndarray,
    ) -> np.ndarray:
        n = base_design.shape[0]
        parts = []
        if self.include_base:
            parts.append(base_design)
        for gate in range(gates.shape[1]):
            parts.append(base_design * gates[:, None, gate : gate + 1])
        row_identity = np.broadcast_to(
            np.eye(self.block_size, dtype=base_design.dtype)[None],
            (n, self.block_size, self.block_size),
        )
        parts.append(row_identity)
        return np.concatenate(parts, axis=2).reshape(n * self.block_size, -1)

    def fit(self, view: dict[str, np.ndarray], y: np.ndarray) -> "BCMRegressor":
        width = y.shape[1]
        if width % self.block_size:
            raise ValueError("state width must be divisible by BCM block size")
        if self.gate_rank:
            self._fit_gates(view[self.gate_key])
        base_design = _bcm_design(view["current"], self.block_size)
        gates = self._gates(view)
        features = self._features(base_design, gates)
        self.models = []
        for output_block in range(width // self.block_size):
            target = y[
                :,
                output_block * self.block_size : (output_block + 1) * self.block_size,
            ].reshape(-1, 1)
            self.models.append(RidgeRegressor(self.alpha).fit(features, target))
        self.width = width
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        base_design = _bcm_design(view["current"], self.block_size)
        features = self._features(base_design, self._gates(view))
        n = view["current"].shape[0]
        blocks = [model.predict(features).reshape(n, self.block_size) for model in self.models]
        return np.concatenate(blocks, axis=1)

    @property
    def parameter_count(self) -> int:
        return sum(model.parameter_count for model in self.models)


class ActionBCMMixtureRegressor:
    def __init__(self, block_size: int, gate_rank: int, alpha: float):
        self.block_size = block_size
        self.gate_rank = gate_rank
        self.alpha = alpha

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "ActionBCMMixtureRegressor":
        self.base = BCMRegressor(self.block_size, self.alpha).fit(view, y)
        residual = y - self.base.predict(view)
        self.action = RidgeRegressor(self.alpha).fit(
            view["action_context"], residual
        )
        residual = residual - self.action.predict(view["action_context"])
        self.gated = BCMRegressor(
            self.block_size,
            self.alpha,
            gate_rank=self.gate_rank,
            include_base=False,
        ).fit(view, residual)
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return (
            self.base.predict(view)
            + self.action.predict(view["action_context"])
            + self.gated.predict(view)
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.base.parameter_count
            + self.action.parameter_count
            + self.gated.parameter_count
        )


class FrozenConditionBCMMixtureRegressor:
    def __init__(
        self,
        block_size: int,
        condition_rank: int,
        gate_rank: int,
        alpha: float,
    ):
        self.block_size = block_size
        self.condition_rank = condition_rank
        self.gate_rank = gate_rank
        self.alpha = alpha

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "FrozenConditionBCMMixtureRegressor":
        self.base = BCMRegressor(self.block_size, self.alpha).fit(view, y)
        residual = y - self.base.predict(view)
        self.condition = ReducedRankRegressor(
            self.condition_rank, self.alpha
        ).fit(view["condition_context"], residual)
        residual = residual - self.condition.predict(view["condition_context"])
        self.gated = BCMRegressor(
            self.block_size,
            self.alpha,
            gate_rank=self.gate_rank,
            include_base=False,
            gate_key="condition_context",
        ).fit(view, residual)
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return (
            self.base.predict(view)
            + self.condition.predict(view["condition_context"])
            + self.gated.predict(view)
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.base.parameter_count
            + self.condition.parameter_count
            + self.gated.parameter_count
        )


class ResidualWrapper:
    def __init__(
        self,
        base,
        rank: int,
        alpha: float,
        use_action_features: bool,
    ):
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.use_action_features = use_action_features

    def _features(self, view: dict[str, np.ndarray]) -> np.ndarray:
        features = history_features(view)
        if self.use_action_features:
            features = np.concatenate([features, view["action_context"]], axis=1)
        return features

    def fit(self, view: dict[str, np.ndarray], y: np.ndarray) -> "ResidualWrapper":
        self.base.fit(view, y)
        residual = y - self.base.predict(view)
        self.residual = ReducedRankRegressor(self.rank, self.alpha).fit(
            self._features(view), residual
        )
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return self.base.predict(view) + self.residual.predict(self._features(view))

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count + self.residual.parameter_count


class DenseCapacityRegressor:
    def __init__(self, alpha: float, use_action: bool):
        self.alpha = alpha
        self.use_action = use_action

    def _features(self, view: dict[str, np.ndarray]) -> np.ndarray:
        parts = [history_features(view), view["position"]]
        if self.use_action:
            parts.append(view["action_context"])
        return np.concatenate(parts, axis=1)

    def fit(
        self, view: dict[str, np.ndarray], y: np.ndarray
    ) -> "DenseCapacityRegressor":
        self.model = RidgeRegressor(self.alpha).fit(self._features(view), y)
        return self

    def predict(self, view: dict[str, np.ndarray]) -> np.ndarray:
        return self.model.predict(self._features(view))

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    uses_action: bool
    constructor: object


def make_method_specs(config: dict) -> list[MethodSpec]:
    alpha = float(config["ridge_alpha"])
    gate_rank = int(config["action_gate_rank"])
    specs = [
        MethodSpec("identity", "fixed", False, lambda: IdentityRegressor()),
        MethodSpec("velocity", "fixed", False, lambda: VelocityRegressor()),
        MethodSpec(
            "diagonal_history",
            "diagonal",
            False,
            lambda: DiagonalHistoryRegressor(alpha, False, 0),
        ),
        MethodSpec(
            "action_diagonal",
            "action_diagonal",
            True,
            lambda: DiagonalHistoryRegressor(alpha, True, gate_rank),
        ),
    ]
    for block_size in config["bcm_block_sizes"]:
        specs.append(
            MethodSpec(
                f"bcm_b{block_size}",
                "bcm",
                False,
                lambda block_size=block_size: BCMRegressor(block_size, alpha),
            )
        )
    for rank in config["low_ranks"]:
        specs.extend(
            [
                MethodSpec(
                    f"diagonal_lr{rank}",
                    "diagonal_low_rank",
                    False,
                    lambda rank=rank: ResidualWrapper(
                        DiagonalHistoryRegressor(alpha, False, 0),
                        rank,
                        alpha,
                        False,
                    ),
                ),
                MethodSpec(
                    f"bcm_b32_lr{rank}",
                    "bcm_low_rank",
                    False,
                    lambda rank=rank: ResidualWrapper(
                        BCMRegressor(32, alpha), rank, alpha, False
                    ),
                ),
            ]
        )
    specs.extend(
        [
            MethodSpec(
                "action_bcm_mixture",
                "action_bcm",
                True,
                lambda: ActionBCMMixtureRegressor(32, gate_rank, alpha),
            ),
            MethodSpec(
                "action_diagonal_lr8",
                "action_diagonal_low_rank",
                True,
                lambda: ResidualWrapper(
                    DiagonalHistoryRegressor(alpha, True, gate_rank),
                    8,
                    alpha,
                    True,
                ),
            ),
            MethodSpec(
                "action_bcm_mixture_lr8",
                "action_bcm_low_rank",
                True,
                lambda: ResidualWrapper(
                    ActionBCMMixtureRegressor(32, gate_rank, alpha),
                    8,
                    alpha,
                    True,
                ),
            ),
            MethodSpec(
                "dense_history",
                "capacity_control",
                False,
                lambda: DenseCapacityRegressor(alpha, False),
            ),
            MethodSpec(
                "dense_action",
                "capacity_control",
                True,
                lambda: DenseCapacityRegressor(alpha, True),
            ),
        ]
    )
    for condition_rank in config["condition_ranks"]:
        specs.extend(
            [
                MethodSpec(
                    f"frozen_condition_diagonal_r{condition_rank}",
                    "frozen_condition_diagonal",
                    True,
                    lambda condition_rank=condition_rank: FrozenConditionDiagonalRegressor(
                        alpha, condition_rank
                    ),
                ),
                MethodSpec(
                    f"frozen_condition_lr{condition_rank}",
                    "frozen_condition_low_rank",
                    True,
                    lambda condition_rank=condition_rank: FrozenConditionLowRankRegressor(
                        condition_rank, alpha
                    ),
                ),
            ]
        )
    specs.append(
        MethodSpec(
            "frozen_condition_bcm_lr8_gate2",
            "frozen_condition_bcm",
            True,
            lambda: FrozenConditionBCMMixtureRegressor(
                32,
                8,
                int(config["condition_bcm_gate_rank"]),
                alpha,
            ),
        )
    )
    return specs


def residual_spectrum(residual: np.ndarray, rank: int = 16) -> tuple[float, int]:
    singular = np.linalg.svd(residual, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total < EPS:
        return 1.0, 0
    rank_energy = float(energy[:rank].sum() / total)
    cumulative = np.cumsum(energy) / total
    effective_rank = int(np.searchsorted(cumulative, 0.99) + 1)
    return rank_energy, effective_rank


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + EPS))


def innovation_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(prediction - target)
        / (np.linalg.norm(target - current) + EPS)
    )


def cosine_summary(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, EPS)))
