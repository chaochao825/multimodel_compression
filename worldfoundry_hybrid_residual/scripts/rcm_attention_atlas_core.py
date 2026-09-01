#!/usr/bin/env python3
"""Pure helpers for the EXP-054 rCM attention atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


CELL_STEPS = 4
CELL_LAYERS = 30
CELL_COUNT = CELL_STEPS * CELL_LAYERS


@dataclass(frozen=True)
class ErrorThresholds:
    aggregate: float
    worst_head: float
    worst_query_tile: float

    def validate(self) -> None:
        if min(self.aggregate, self.worst_head, self.worst_query_tile) <= 0:
            raise ValueError("all error thresholds must be positive")


@dataclass(frozen=True)
class CellMetric:
    identity: str
    split: str
    step: int
    layer: int
    aggregate: float
    worst_head: float
    worst_query_tile: float

    @property
    def cell(self) -> tuple[int, int]:
        return self.step, self.layer

    def passes(self, thresholds: ErrorThresholds) -> bool:
        return (
            self.aggregate <= thresholds.aggregate
            and self.worst_head <= thresholds.worst_head
            and self.worst_query_tile <= thresholds.worst_query_tile
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "split": self.split,
            "step": self.step,
            "layer": self.layer,
            "aggregate": self.aggregate,
            "worst_head": self.worst_head,
            "worst_query_tile": self.worst_query_tile,
        }


def _relative_l2(numerator: Any, denominator: Any) -> Any:
    import torch

    zeros = torch.zeros_like(numerator)
    infinities = torch.full_like(numerator, float("inf"))
    ratio = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.where(numerator == 0, zeros, infinities),
    )
    return torch.sqrt(ratio)


def output_error_metrics(
    reference: Any,
    candidate: Any,
    query_tile_size: int = 64,
) -> tuple[Any, Any, Any]:
    """Return aggregate, worst-head, and worst-query-tile relative L2."""

    import torch

    if reference.shape != candidate.shape:
        raise ValueError(
            f"attention output shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.ndim != 4:
        raise ValueError(f"expected [B,S,H,D], got {tuple(reference.shape)}")
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")

    reference_f = reference.float()
    difference = candidate.float() - reference_f
    squared_reference = reference_f.square()
    squared_difference = difference.square()

    aggregate = _relative_l2(
        squared_difference.sum(), squared_reference.sum()
    )
    head = _relative_l2(
        squared_difference.sum(dim=(0, 1, 3)),
        squared_reference.sum(dim=(0, 1, 3)),
    ).max()

    batch, sequence, heads, width = reference.shape
    tile_count = (sequence + query_tile_size - 1) // query_tile_size
    padded_sequence = tile_count * query_tile_size
    if padded_sequence != sequence:
        pad_shape = (batch, padded_sequence - sequence, heads, width)
        squared_difference = torch.cat(
            [squared_difference, squared_difference.new_zeros(pad_shape)], dim=1
        )
        squared_reference = torch.cat(
            [squared_reference, squared_reference.new_zeros(pad_shape)], dim=1
        )
    squared_difference = squared_difference.reshape(
        batch, tile_count, query_tile_size, heads, width
    )
    squared_reference = squared_reference.reshape(
        batch, tile_count, query_tile_size, heads, width
    )
    query_tile = _relative_l2(
        squared_difference.sum(dim=(2, 3, 4)),
        squared_reference.sum(dim=(2, 3, 4)),
    ).max()
    return aggregate, head, query_tile


def validate_record_grid(
    records: Sequence[CellMetric],
    identities: Sequence[str],
    split: str,
    steps: int = CELL_STEPS,
    layers: int = CELL_LAYERS,
) -> None:
    expected = {
        (identity, step, layer)
        for identity in identities
        for step in range(steps)
        for layer in range(layers)
    }
    observed: set[tuple[str, int, int]] = set()
    duplicates: list[tuple[str, int, int]] = []
    for record in records:
        if record.split != split:
            raise ValueError(
                f"record {record.identity}/{record.cell} has split {record.split}, "
                f"expected {split}"
            )
        key = (record.identity, record.step, record.layer)
        if key in observed:
            duplicates.append(key)
        observed.add(key)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "attention atlas grid mismatch: "
            f"duplicates={duplicates[:5]}, missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )


def freeze_and_evaluate_atlas(
    calibration_records: Sequence[CellMetric],
    evaluation_records: Sequence[CellMetric],
    calibration_identities: Sequence[str],
    evaluation_identities: Sequence[str],
    calibration_thresholds: ErrorThresholds,
    evaluation_thresholds: ErrorThresholds,
    minimum_selected_cells: int,
) -> dict[str, object]:
    calibration_thresholds.validate()
    evaluation_thresholds.validate()
    if set(calibration_identities) & set(evaluation_identities):
        raise ValueError("calibration and evaluation identities overlap")
    if not 0 < minimum_selected_cells <= CELL_COUNT:
        raise ValueError("minimum_selected_cells lies outside the cell grid")
    validate_record_grid(calibration_records, calibration_identities, "calibration")
    validate_record_grid(evaluation_records, evaluation_identities, "evaluation")

    grouped_calibration: dict[tuple[int, int], list[CellMetric]] = {}
    for record in calibration_records:
        grouped_calibration.setdefault(record.cell, []).append(record)
    selected = tuple(
        sorted(
            cell
            for cell, rows in grouped_calibration.items()
            if all(row.passes(calibration_thresholds) for row in rows)
        )
    )
    selected_set = set(selected)
    false_safe = tuple(
        record
        for record in evaluation_records
        if record.cell in selected_set and not record.passes(evaluation_thresholds)
    )
    return {
        "selected_cells": [list(cell) for cell in selected],
        "selected_cell_count": len(selected),
        "coverage": len(selected) / CELL_COUNT,
        "false_safe_count": len(false_safe),
        "false_safe": [record.as_dict() for record in false_safe],
        "passes_transfer_and_count": (
            len(false_safe) == 0 and len(selected) >= minimum_selected_cells
        ),
    }


def projected_request(
    baseline_request_seconds: float,
    baseline_denoiser_seconds: float,
    attention_share: float,
    selected_coverage: float,
    attention_speedup: float,
) -> tuple[float, float]:
    positive_values = (
        baseline_request_seconds,
        baseline_denoiser_seconds,
        attention_share,
        attention_speedup,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("time, attention share, and speedup must be positive")
    if selected_coverage < 0:
        raise ValueError("selected coverage must be nonnegative")
    if attention_share > 1 or selected_coverage > 1:
        raise ValueError("attention share and selected coverage must not exceed one")
    saved = (
        baseline_denoiser_seconds
        * attention_share
        * selected_coverage
        * (1.0 - 1.0 / attention_speedup)
    )
    projected_seconds = baseline_request_seconds - saved
    return projected_seconds, baseline_request_seconds / projected_seconds


class WanSelfAttentionPatch:
    """Patch only instantiated Wan self-attention callables and restore them."""

    def __init__(self, network: object, dispatcher: Callable[..., Any]) -> None:
        self.network = network
        self.dispatcher = dispatcher
        self.originals: list[tuple[object, Callable[..., Any]]] = []

    def install(self) -> None:
        if self.originals:
            raise RuntimeError("self-attention patch is already installed")
        blocks = self.network.blocks
        if len(blocks) != CELL_LAYERS:
            raise ValueError(f"expected {CELL_LAYERS} Wan blocks, got {len(blocks)}")
        for layer, block in enumerate(blocks):
            local_attention = block.self_attn.attn_op.local_attn
            original = local_attention.attn
            if not callable(original):
                raise TypeError(f"layer {layer} self-attention is not callable")

            def wrapped(
                q: Any,
                k: Any,
                v: Any,
                *,
                _layer: int = layer,
                _original: Callable[..., Any] = original,
            ) -> Any:
                return self.dispatcher(_layer, _original, q, k, v)

            self.originals.append((local_attention, original))
            local_attention.attn = wrapped

    def restore(self) -> None:
        for local_attention, original in self.originals:
            local_attention.attn = original
        self.originals.clear()

    def __enter__(self) -> "WanSelfAttentionPatch":
        self.install()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.restore()
