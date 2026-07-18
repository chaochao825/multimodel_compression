"""Deployable block features and routers for TileLogic-RVQ.

The MLP is a calibration-only teacher/upper bound.  The logic router consists
of calibrated scalar comparators followed by bounded-depth binary regression
trees.  Neither router receives gradients, labels, answers, or model outputs
at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .core import block_query_relevance, enumerate_blocks


BLOCK_FEATURE_NAMES = (
    "log_energy",
    "query_relevance",
    "log_variance",
    "log_rms",
    "tile_0",
    "tile_1",
    "tile_2",
    "tile_3",
    "normalized_row",
    "normalized_col",
    "spatial_radius",
    "thumbnail_agreement",
    "curvature_prior",
)


def _safe_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_norm = lhs.float() / lhs.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rhs_norm = rhs.float() / rhs.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (lhs_norm * rhs_norm).sum(dim=-1)


def block_router_features(
    crop_tiles: torch.Tensor,
    residual: torch.Tensor,
    query_embedding: torch.Tensor,
    thumbnail: torch.Tensor,
    *,
    curvature_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[tuple[int, int, int], ...]]:
    """Build low-cost inference features for all 2x2 residual blocks."""

    if crop_tiles.shape != residual.shape or crop_tiles.ndim != 4:
        raise ValueError("crop_tiles and residual must share [tiles,H,W,C] shape")
    tiles, height, width, channels = crop_tiles.shape
    if thumbnail.shape == (height * width, channels):
        thumbnail = thumbnail.reshape(height, width, channels)
    if thumbnail.shape != (height, width, channels):
        raise ValueError(
            "thumbnail must match one crop tile after the visual merger"
        )
    blocks, locations, energy = enumerate_blocks(residual)
    crop_blocks, crop_locations, _ = enumerate_blocks(crop_tiles)
    if locations != crop_locations:
        raise AssertionError("crop and residual layouts differ")
    relevance = block_query_relevance(blocks, query_embedding)
    flattened = blocks.float().reshape(blocks.shape[0], -1)
    variance = flattened.var(dim=1, unbiased=False)
    rms = flattened.square().mean(dim=1).sqrt()

    tile_indices = torch.tensor(
        [item[0] for item in locations], device=residual.device, dtype=torch.long
    )
    rows = torch.tensor(
        [item[1] for item in locations], device=residual.device, dtype=torch.float32
    )
    cols = torch.tensor(
        [item[2] for item in locations], device=residual.device, dtype=torch.float32
    )
    normalized_row = rows / max(1, height - 2)
    normalized_col = cols / max(1, width - 2)
    radius = (normalized_row.square() + normalized_col.square()).sqrt() / math.sqrt(2)
    tile_one_hot = torch.nn.functional.one_hot(tile_indices, num_classes=tiles).float()

    crop_means = crop_blocks.float().mean(dim=(1, 2))
    thumbnail_vectors = []
    for tile, row, col in locations:
        global_row = row + (height if tile >= 2 else 0)
        global_col = col + (width if tile % 2 else 0)
        thumbnail_row = min(height - 1, global_row // 2)
        thumbnail_col = min(width - 1, global_col // 2)
        thumbnail_vectors.append(thumbnail[thumbnail_row, thumbnail_col])
    thumbnail_tensor = torch.stack(thumbnail_vectors)
    thumbnail_agreement = ((_safe_cosine(crop_means, thumbnail_tensor) + 1) * 0.5).clamp(0, 1)

    if curvature_prior is None:
        curvature = torch.zeros(len(locations), device=residual.device)
    else:
        if curvature_prior.shape != (len(locations),):
            raise ValueError(
                f"curvature_prior must have shape [{len(locations)}]"
            )
        curvature = curvature_prior.to(residual.device, torch.float32)

    features = torch.cat(
        (
            energy.float().clamp_min(1e-12).log().unsqueeze(1),
            relevance.float().unsqueeze(1),
            variance.clamp_min(1e-12).log().unsqueeze(1),
            rms.clamp_min(1e-12).log().unsqueeze(1),
            tile_one_hot,
            normalized_row.unsqueeze(1),
            normalized_col.unsqueeze(1),
            radius.unsqueeze(1),
            thumbnail_agreement.unsqueeze(1),
            curvature.unsqueeze(1),
        ),
        dim=1,
    )
    if features.shape[1] != len(BLOCK_FEATURE_NAMES):
        raise AssertionError("router feature schema drift")
    if not torch.isfinite(features).all():
        raise ValueError("router features contain non-finite values")
    return features, locations


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @staticmethod
    def fit(features: torch.Tensor) -> "FeatureNormalizer":
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("features must have shape [N,F] with N >= 2")
        mean = features.float().mean(dim=0)
        scale = features.float().std(dim=0, unbiased=False).clamp_min(1e-6)
        return FeatureNormalizer(mean, scale)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("feature dimension does not match normalizer")
        return (features.float() - self.mean.to(features.device)) / self.scale.to(
            features.device
        )

    def state_dict(self) -> dict[str, torch.Tensor | str]:
        return {
            "format": "tilespec_feature_normalizer_v1",
            "mean": self.mean.detach().cpu(),
            "scale": self.scale.detach().cpu(),
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "FeatureNormalizer":
        if state.get("format") != "tilespec_feature_normalizer_v1":
            raise ValueError("unsupported feature-normalizer format")
        return FeatureNormalizer(state["mean"].float(), state["scale"].float())


class RouterMLP(nn.Module):
    """Small continuous teacher predicting marginal coding benefits."""

    def __init__(self, input_dim: int, output_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) <= 0:
            raise ValueError("router dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def export_state(self) -> dict[str, Any]:
        return {
            "format": "tilespec_router_mlp_v1",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @staticmethod
    def from_export(state: dict[str, Any]) -> "RouterMLP":
        if state.get("format") != "tilespec_router_mlp_v1":
            raise ValueError("unsupported router-MLP format")
        model = RouterMLP(
            int(state["input_dim"]),
            int(state["output_dim"]),
            int(state["hidden_dim"]),
        )
        model.load_state_dict(state["state_dict"])
        return model


@dataclass(frozen=True)
class RouterTrainingSummary:
    epochs: int
    best_validation_loss: float
    final_training_loss: float


def fit_router_mlp(
    model: RouterMLP,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    epochs: int = 300,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    seed: int = 20260718,
) -> RouterTrainingSummary:
    if train_features.ndim != 2 or train_targets.ndim != 2:
        raise ValueError("router train tensors must be matrices")
    if validation_features.ndim != 2 or validation_targets.ndim != 2:
        raise ValueError("router validation tensors must be matrices")
    if train_features.shape[0] != train_targets.shape[0]:
        raise ValueError("training feature/target counts differ")
    if validation_features.shape[0] != validation_targets.shape[0]:
        raise ValueError("validation feature/target counts differ")
    if train_features.shape[1] != model.input_dim:
        raise ValueError("training feature dimension differs from router")
    if train_targets.shape[1] != model.output_dim:
        raise ValueError("training target dimension differs from router")
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    stale = 0
    final_training_loss = math.inf
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_features)
        scale = train_targets.detach().mean(dim=0).clamp_min(1e-6)
        training_loss = ((prediction - train_targets) / scale).square().mean()
        training_loss.backward()
        optimizer.step()
        final_training_loss = float(training_loss.detach().item())

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_features)
            validation_scale = validation_targets.mean(dim=0).clamp_min(1e-6)
            validation_loss = (
                ((validation_prediction - validation_targets) / validation_scale)
                .square()
                .mean()
            )
        value = float(validation_loss.item())
        if value < best_loss - 1e-8:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("router training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return RouterTrainingSummary(
        epochs=epochs - max(0, stale),
        best_validation_loss=best_loss,
        final_training_loss=final_training_loss,
    )


@dataclass(frozen=True)
class FeatureBinarizer:
    thresholds: torch.Tensor
    threshold_storage_bits: int = 16

    def __post_init__(self) -> None:
        if self.threshold_storage_bits not in {16, 32}:
            raise ValueError("threshold storage supports exactly 16 or 32 bits")

    @staticmethod
    def fit(
        features: torch.Tensor,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    ) -> "FeatureBinarizer":
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("features must have shape [N,F] with N >= 2")
        if not quantiles or any(not 0 < value < 1 for value in quantiles):
            raise ValueError("quantiles must be non-empty and within (0,1)")
        q = torch.tensor(tuple(quantiles), device=features.device)
        thresholds = torch.quantile(features.float(), q, dim=0).t().contiguous()
        return FeatureBinarizer(thresholds)

    @property
    def num_features(self) -> int:
        return int(self.thresholds.shape[0])

    @property
    def thresholds_per_feature(self) -> int:
        return int(self.thresholds.shape[1])

    @property
    def num_bits(self) -> int:
        return int(self.thresholds.numel())

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError("feature dimension does not match binarizer")
        bits = features.float().unsqueeze(2) > self.thresholds.to(features.device)
        return bits.reshape(features.shape[0], -1)

    def state_dict(self) -> dict[str, Any]:
        storage_dtype = (
            torch.float16 if self.threshold_storage_bits == 16 else torch.float32
        )
        return {
            "format": "tilespec_feature_binarizer_v1",
            "thresholds": self.thresholds.detach().cpu().to(storage_dtype),
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "FeatureBinarizer":
        if state.get("format") != "tilespec_feature_binarizer_v1":
            raise ValueError("unsupported feature-binarizer format")
        thresholds = state["thresholds"]
        return FeatureBinarizer(
            thresholds.float(), thresholds.element_size() * 8
        )


@dataclass(frozen=True)
class LogicTreeNode:
    bit_index: int
    left: int
    right: int
    value: float

    @property
    def is_leaf(self) -> bool:
        return self.bit_index < 0


@dataclass(frozen=True)
class LogicRegressionTree:
    nodes: tuple[LogicTreeNode, ...]
    num_input_bits: int
    leaf_storage_bits: int = 32

    def __post_init__(self) -> None:
        if self.leaf_storage_bits != 32:
            raise ValueError("logic leaves execute as float32 logical values")
        for node in self.nodes:
            encoded = torch.tensor(node.value, dtype=torch.float32)
            if not bool(torch.isfinite(encoded)) or float(encoded.item()) != node.value:
                raise ValueError("logic-tree values must round-trip exactly through float32")

    def predict(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.num_input_bits:
            raise ValueError("logic input shape differs from tree")
        device = bits.device
        bit_index = torch.tensor(
            [node.bit_index for node in self.nodes], device=device, dtype=torch.long
        )
        left = torch.tensor(
            [node.left for node in self.nodes], device=device, dtype=torch.long
        )
        right = torch.tensor(
            [node.right for node in self.nodes], device=device, dtype=torch.long
        )
        values = torch.tensor(
            [node.value for node in self.nodes], device=device, dtype=torch.float32
        )
        current = torch.zeros(bits.shape[0], device=device, dtype=torch.long)
        for _ in range(len(self.nodes)):
            current_bits = bit_index[current]
            active = current_bits >= 0
            if not bool(active.any()):
                break
            rows = torch.nonzero(active, as_tuple=False).flatten()
            choices = bits[rows, current_bits[rows]]
            current[rows] = torch.where(
                choices, right[current[rows]], left[current[rows]]
            )
        if bool((bit_index[current] >= 0).any()):
            raise RuntimeError("logic tree traversal did not reach every leaf")
        return values[current]

    @property
    def internal_nodes(self) -> int:
        return sum(not node.is_leaf for node in self.nodes)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "tilespec_logic_regression_tree_v2",
            "num_input_bits": self.num_input_bits,
            "leaf_storage_bits": self.leaf_storage_bits,
            "nodes": [
                {
                    "bit_index": node.bit_index,
                    "left": node.left,
                    "right": node.right,
                }
                for node in self.nodes
            ],
            "values": torch.tensor(
                [node.value for node in self.nodes], dtype=torch.float32
            ),
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "LogicRegressionTree":
        state_format = state.get("format")
        if state_format not in {
            "tilespec_logic_regression_tree_v1",
            "tilespec_logic_regression_tree_v2",
        }:
            raise ValueError("unsupported logic-tree format")
        raw_nodes = state["nodes"]
        if state_format == "tilespec_logic_regression_tree_v1":
            nodes = tuple(LogicTreeNode(**item) for item in raw_nodes)
        else:
            values = state.get("values")
            if (
                not isinstance(values, torch.Tensor)
                or values.dtype != torch.float32
                or values.shape != (len(raw_nodes),)
            ):
                raise ValueError("logic-tree v2 values must be a float32 tensor")
            nodes = tuple(
                LogicTreeNode(
                    bit_index=int(item["bit_index"]),
                    left=int(item["left"]),
                    right=int(item["right"]),
                    value=float(values[index].item()),
                )
                for index, item in enumerate(raw_nodes)
            )
        return LogicRegressionTree(
            nodes,
            int(state["num_input_bits"]),
            int(state.get("leaf_storage_bits", 32)),
        )


def fit_logic_regression_tree(
    bits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_depth: int = 6,
    min_leaf: int = 32,
) -> LogicRegressionTree:
    """Fit a deterministic binary regression tree by variance reduction."""

    if bits.ndim != 2 or bits.dtype != torch.bool:
        raise ValueError("bits must be a boolean [N,B] tensor")
    if targets.ndim != 1 or targets.shape[0] != bits.shape[0]:
        raise ValueError("targets must have shape [N]")
    if max_depth <= 0 or min_leaf <= 0:
        raise ValueError("max_depth and min_leaf must be positive")
    bits_cpu = bits.cpu()
    targets_cpu = targets.float().cpu()
    nodes: list[LogicTreeNode | None] = []

    def build(rows: torch.Tensor, depth: int, available: tuple[int, ...]) -> int:
        node_index = len(nodes)
        nodes.append(None)
        mean = float(targets_cpu[rows].mean().item())
        if depth >= max_depth or rows.numel() < 2 * min_leaf or not available:
            nodes[node_index] = LogicTreeNode(-1, -1, -1, mean)
            return node_index
        parent_error = float(
            (targets_cpu[rows] - targets_cpu[rows].mean()).square().sum().item()
        )
        best: tuple[float, int, torch.Tensor, torch.Tensor] | None = None
        for bit_index in available:
            values = bits_cpu[rows, bit_index]
            left_rows = rows[~values]
            right_rows = rows[values]
            if left_rows.numel() < min_leaf or right_rows.numel() < min_leaf:
                continue
            error = float(
                (targets_cpu[left_rows] - targets_cpu[left_rows].mean())
                .square()
                .sum()
                .item()
                + (targets_cpu[right_rows] - targets_cpu[right_rows].mean())
                .square()
                .sum()
                .item()
            )
            gain = parent_error - error
            if best is None or gain > best[0] + 1e-12 or (
                abs(gain - best[0]) <= 1e-12 and bit_index < best[1]
            ):
                best = (gain, bit_index, left_rows, right_rows)
        if best is None or best[0] <= 1e-12:
            nodes[node_index] = LogicTreeNode(-1, -1, -1, mean)
            return node_index
        _, bit_index, left_rows, right_rows = best
        remaining = tuple(index for index in available if index != bit_index)
        left = build(left_rows, depth + 1, remaining)
        right = build(right_rows, depth + 1, remaining)
        nodes[node_index] = LogicTreeNode(bit_index, left, right, mean)
        return node_index

    build(
        torch.arange(bits.shape[0], dtype=torch.long),
        0,
        tuple(range(bits.shape[1])),
    )
    if any(node is None for node in nodes):
        raise AssertionError("logic tree contains unfinished nodes")
    return LogicRegressionTree(tuple(node for node in nodes if node is not None), bits.shape[1])


@dataclass(frozen=True)
class LogicRouter:
    binarizer: FeatureBinarizer
    trees: tuple[LogicRegressionTree, ...]

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        bits = self.binarizer.transform(features)
        return torch.stack([tree.predict(bits) for tree in self.trees], dim=1).clamp_min(0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "tilespec_logic_router_v1",
            "binarizer": self.binarizer.state_dict(),
            "trees": [tree.state_dict() for tree in self.trees],
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "LogicRouter":
        if state.get("format") != "tilespec_logic_router_v1":
            raise ValueError("unsupported logic-router format")
        return LogicRouter(
            FeatureBinarizer.from_state_dict(state["binarizer"]),
            tuple(LogicRegressionTree.from_state_dict(item) for item in state["trees"]),
        )


def fit_logic_router(
    features: torch.Tensor,
    teacher_marginal_benefits: torch.Tensor,
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    max_depth: int = 6,
    min_leaf: int = 32,
) -> LogicRouter:
    if teacher_marginal_benefits.ndim != 2:
        raise ValueError("teacher benefits must have shape [N,A]")
    if teacher_marginal_benefits.shape[0] != features.shape[0]:
        raise ValueError("feature and teacher counts differ")
    binarizer = FeatureBinarizer.fit(features, quantiles)
    bits = binarizer.transform(features)
    trees = tuple(
        fit_logic_regression_tree(
            bits,
            teacher_marginal_benefits[:, action],
            max_depth=max_depth,
            min_leaf=min_leaf,
        )
        for action in range(teacher_marginal_benefits.shape[1])
    )
    return LogicRouter(binarizer, trees)


def allocate_variable_depth(
    marginal_benefits: torch.Tensor,
    incremental_cost_bits: torch.Tensor,
    budget_bits: int,
    *,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Greedily allocate precedence-constrained coding upgrades.

    Action ``a`` for a block is available only after actions ``0..a-1`` have
    been selected.  This models drop -> RVQ1 -> RVQ2 -> exact.
    """

    if marginal_benefits.ndim != 2 or incremental_cost_bits.shape != marginal_benefits.shape:
        raise ValueError("benefits and costs must share [blocks,actions] shape")
    if budget_bits < 0:
        raise ValueError("budget_bits must be non-negative")
    if bool((incremental_cost_bits <= 0).any()):
        raise ValueError("incremental costs must be strictly positive")
    blocks, actions = marginal_benefits.shape
    if candidate_mask is None:
        candidate_mask = torch.ones(blocks, dtype=torch.bool, device=marginal_benefits.device)
    if candidate_mask.shape != (blocks,):
        raise ValueError("candidate_mask must have shape [blocks]")
    depths = torch.zeros(blocks, dtype=torch.long, device=marginal_benefits.device)
    spent = 0
    while True:
        best_block = -1
        best_ratio = -math.inf
        best_benefit = -math.inf
        for block in range(blocks):
            if not bool(candidate_mask[block]):
                continue
            action = int(depths[block].item())
            if action >= actions:
                continue
            cost = int(incremental_cost_bits[block, action].item())
            if spent + cost > budget_bits:
                continue
            benefit = float(marginal_benefits[block, action].item())
            ratio = benefit / cost
            if ratio > best_ratio + 1e-15 or (
                abs(ratio - best_ratio) <= 1e-15
                and (benefit > best_benefit + 1e-15 or block < best_block)
            ):
                best_block = block
                best_ratio = ratio
                best_benefit = benefit
        if best_block < 0 or best_benefit <= 0:
            break
        action = int(depths[best_block].item())
        spent += int(incremental_cost_bits[best_block, action].item())
        depths[best_block] += 1
    return depths, spent


def fixed_slot_mask(
    calibration_importance: torch.Tensor,
    locations: Iterable[tuple[int, int, int]],
    *,
    slots_per_tile: int,
) -> torch.Tensor:
    """Choose immutable positions independently within each tile."""

    locations = tuple(locations)
    if calibration_importance.shape != (len(locations),):
        raise ValueError("calibration importance and locations differ")
    if slots_per_tile <= 0:
        raise ValueError("slots_per_tile must be positive")
    mask = torch.zeros(len(locations), dtype=torch.bool, device=calibration_importance.device)
    for tile in range(4):
        indices = torch.tensor(
            [index for index, location in enumerate(locations) if location[0] == tile],
            device=calibration_importance.device,
            dtype=torch.long,
        )
        if slots_per_tile > indices.numel():
            raise ValueError("slots_per_tile exceeds available positions")
        selected = indices[
            torch.topk(calibration_importance[indices], k=slots_per_tile).indices
        ]
        mask[selected] = True
    return mask


def logic_router_storage_bits(
    router: LogicRouter,
    *,
    threshold_bits: int | None = None,
    leaf_value_bits: int | None = None,
) -> int:
    threshold_bits = (
        router.binarizer.threshold_storage_bits
        if threshold_bits is None
        else threshold_bits
    )
    if threshold_bits <= 0 or (leaf_value_bits is not None and leaf_value_bits <= 0):
        raise ValueError("storage bit widths must be positive")
    bits = router.binarizer.thresholds.numel() * threshold_bits
    for tree in router.trees:
        tree_leaf_bits = tree.leaf_storage_bits if leaf_value_bits is None else leaf_value_bits
        index_bits = max(1, math.ceil(math.log2(tree.num_input_bits)))
        node_pointer_bits = max(1, math.ceil(math.log2(len(tree.nodes))))
        for node in tree.nodes:
            if node.is_leaf:
                bits += 1 + tree_leaf_bits
            else:
                bits += 1 + index_bits + 2 * node_pointer_bits
    return int(bits)
