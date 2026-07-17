from __future__ import annotations

import pytest

from scripts.rescore_tilespec_ex_quality import rescore_record
from tilespec_ex.metrics import (
    dataset_score,
    normalize_answer,
    paired_bootstrap_interval,
    relaxed_chart_match,
    spearman_correlation,
    textvqa_score,
)


def test_normalize_answer_removes_articles_and_punctuation() -> None:
    assert normalize_answer("The, BLUE car!") == "blue car"
    assert normalize_answer("Two") == "2"
    assert normalize_answer("1,234.50") == "1234.50"
    assert normalize_answer("-12.5") == "-12.5"


def test_dataset_metrics() -> None:
    assert relaxed_chart_match("105", "100") == 1.0
    assert relaxed_chart_match("106", "100") == 0.0
    assert relaxed_chart_match("0.58", "0.57") == 1.0
    assert relaxed_chart_match("-10.4", "-10") == 1.0
    assert textvqa_score("Exit", ["exit", "Exit", "EXIT", "door"]) == 1.0
    assert dataset_score("gqa", "A parrot.", ["parrot"]) == 1.0


def test_spearman_handles_ties_and_order() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_paired_bootstrap_detects_clear_improvement() -> None:
    mean, lower, upper = paired_bootstrap_interval(
        [2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0], samples=2_000
    )
    assert mean > 0
    assert lower > 0
    assert upper >= lower


def test_rescore_updates_all_derived_answer_fields() -> None:
    record = {
        "dataset": "chartqa",
        "answers": ["-10"],
        "variants": [
            {
                "method": "none",
                "prediction": "-10.4",
                "score": 0.0,
                "normalized_prediction": "10.4",
                "agrees_with_full": 0.0,
            },
            {
                "method": "tile_lowpass",
                "prediction": "-10.4",
                "score": 0.0,
                "normalized_prediction": "10.4",
                "agrees_with_full": 0.0,
            },
        ],
    }
    changed = rescore_record(record)
    assert changed == {"scores": 2, "normalized_predictions": 2, "agreements": 2}
    assert all(item["score"] == 1.0 for item in record["variants"])
    assert all(item["normalized_prediction"] == "-10.4" for item in record["variants"])
    assert all(item["agrees_with_full"] == 1.0 for item in record["variants"])
