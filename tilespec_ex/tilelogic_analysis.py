"""Pure analysis helpers for the TileLogic-RVQ experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import mean
from typing import Any


RATES = (0.125, 0.25)
DATASETS = ("gqa", "textvqa", "chartqa")


def relative_increase(candidate: float, baseline: float) -> float:
    """Return the signed relative increase, handling an exact-zero baseline."""

    if not math.isfinite(candidate) or not math.isfinite(baseline):
        raise ValueError("relative-increase inputs must be finite")
    if baseline < 0:
        raise ValueError("relative-increase baseline must be nonnegative")
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    return candidate / baseline - 1.0


def topk_recall(scores: Sequence[float], target: Sequence[float], k: int) -> float:
    """Compute deterministic top-k set recall with index-stable tie breaking."""

    if len(scores) != len(target) or not 0 < k <= len(scores):
        raise ValueError("top-k inputs must have equal nonzero length and valid k")
    if not all(math.isfinite(float(value)) for value in (*scores, *target)):
        raise ValueError("top-k inputs must be finite")
    predicted = set(
        sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k]
    )
    expected = set(
        sorted(range(len(target)), key=lambda index: (-float(target[index]), index))[:k]
    )
    return len(predicted & expected) / k


def strict_frontier_extension(
    candidate: Mapping[str, float],
    baselines: Sequence[Mapping[str, float]],
    *,
    rate_key: str,
    metric_key: str,
) -> bool:
    """Return whether a point strictly extends a lower-is-better baseline frontier."""

    rate = float(candidate[rate_key])
    metric = float(candidate[metric_key])
    if not math.isfinite(rate) or not math.isfinite(metric):
        raise ValueError("candidate frontier values must be finite")
    if not baselines:
        raise ValueError("frontier comparison needs at least one baseline")
    comparable = []
    for point in baselines:
        point_rate = float(point[rate_key])
        point_metric = float(point[metric_key])
        if not math.isfinite(point_rate) or not math.isfinite(point_metric):
            raise ValueError("baseline frontier values must be finite")
        if point_rate <= rate:
            comparable.append(point_metric)
    if not comparable:
        return rate < min(float(point[rate_key]) for point in baselines)
    return metric < min(comparable)


def quality_guardrail(
    candidate_scores: Mapping[str, float],
    baseline_scores: Mapping[str, float],
    *,
    candidate_nll: float,
    baseline_nll: float,
    candidate_nmse: float,
    baseline_nmse: float,
    score_loss_limit: float = 0.005,
    nll_increase_limit: float = 0.01,
    nmse_increase_limit: float = 0.02,
) -> dict[str, Any]:
    """Apply the predeclared score, NLL, and feature-NMSE guardrails."""

    missing = [
        dataset
        for dataset in DATASETS
        if dataset not in candidate_scores or dataset not in baseline_scores
    ]
    if missing:
        raise ValueError(f"guardrail dataset scores are missing: {missing}")
    score_deltas = {
        dataset: float(candidate_scores[dataset]) - float(baseline_scores[dataset])
        for dataset in DATASETS
    }
    nll_increase = relative_increase(float(candidate_nll), float(baseline_nll))
    nmse_increase = relative_increase(float(candidate_nmse), float(baseline_nmse))
    score_pass = all(delta >= -score_loss_limit for delta in score_deltas.values())
    nll_pass = nll_increase <= nll_increase_limit
    nmse_pass = nmse_increase <= nmse_increase_limit
    return {
        "score_deltas": score_deltas,
        "max_dataset_score_loss": max(0.0, -min(score_deltas.values())),
        "teacher_nll_relative_increase": nll_increase,
        "feature_nmse_relative_increase": nmse_increase,
        "score_pass": score_pass,
        "teacher_nll_pass": nll_pass,
        "feature_nmse_pass": nmse_pass,
        "pass": bool(score_pass and nll_pass and nmse_pass),
    }


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted mean needs equal nonempty arrays")
    denominator = sum(float(weight) for weight in weights)
    if denominator <= 0:
        raise ValueError("weighted mean needs positive total weight")
    return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / denominator


def aggregate_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError(f"no values for aggregate key: {key}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite values for aggregate key: {key}")
    return mean(values)


def status_from_evidence(*, available: bool, passed: bool) -> str:
    if not available:
        return "INCONCLUSIVE"
    return "PASS" if passed else "FAIL"
