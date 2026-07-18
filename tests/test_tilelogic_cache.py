from __future__ import annotations

import pytest

from scripts.cache_tilelogic_features import assign_oracle_subsets, assign_splits
from scripts.train_tilelogic_rvq import _exception_budget
from tilespec_ex.cache import CACHE_FORMAT, CacheEntry


def _records() -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "dataset_index": index,
            "sample_id": f"{dataset}-{index}",
            "image_sha256": f"{index:064x}",
        }
        for dataset in ("gqa", "textvqa", "chartqa")
        for index in range(20)
    ]


def test_stable_split_is_exact_and_order_independent() -> None:
    forward = assign_splits(_records(), calibration_per_dataset=8)
    reverse = assign_splits(list(reversed(_records())), calibration_per_dataset=8)
    forward_map = {
        (item["dataset"], item["dataset_index"]): item["split"] for item in forward
    }
    reverse_map = {
        (item["dataset"], item["dataset_index"]): item["split"] for item in reverse
    }
    assert forward_map == reverse_map
    for dataset in ("gqa", "textvqa", "chartqa"):
        calibration = [
            item
            for item in forward
            if item["dataset"] == dataset and item["split"] == "calibration"
        ]
        evaluation = [
            item
            for item in forward
            if item["dataset"] == dataset and item["split"] == "evaluation"
        ]
        assert len(calibration) == 8
        assert len(evaluation) == 12
        assert {item["sample_id"] for item in calibration}.isdisjoint(
            {item["sample_id"] for item in evaluation}
        )


def test_oracle_subsets_are_disjoint_and_exact() -> None:
    records = assign_splits(_records(), calibration_per_dataset=8)
    assign_oracle_subsets(records, oracle_per_dataset_split=3)
    for dataset in ("gqa", "textvqa", "chartqa"):
        calibration = {
            item["sample_id"]
            for item in records
            if item["dataset"] == dataset
            and item["split"] == "calibration"
            and item["oracle"]
        }
        evaluation = {
            item["sample_id"]
            for item in records
            if item["dataset"] == dataset
            and item["split"] == "evaluation"
            and item["oracle"]
        }
        assert len(calibration) == 3
        assert len(evaluation) == 3
        assert calibration.isdisjoint(evaluation)


def test_training_exception_budgets_match_original_contract() -> None:
    assert _exception_budget(0.125) == (128, 96, 32)
    assert _exception_budget(0.25) == (256, 192, 64)


def test_cache_entry_requires_per_record_source_model_and_dtype_provenance() -> None:
    record = {
        "format": CACHE_FORMAT,
        "dataset": "gqa",
        "dataset_index": 1,
        "sample_id": "gqa-1",
        "image_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "model_revision": "revision-1",
        "split": "evaluation",
        "split_rank": 1,
        "oracle": False,
        "cache_file": "entries/gqa-1.pt",
        "bytes": 10,
        "sha256": "3" * 64,
        "thumbnail_shape": [256, 8],
        "crop_shape": [4, 16, 16, 8],
        "query_shape": [8],
        "gradient_shape": None,
        "tensor_dtypes": {
            "thumbnail": "float16",
            "crops": "float16",
            "query": "float16",
        },
    }
    entry = CacheEntry.from_record(record)
    assert entry.source_manifest_sha256 == "2" * 64
    assert entry.tensor_dtypes == record["tensor_dtypes"]
    del record["model_revision"]
    with pytest.raises(KeyError):
        CacheEntry.from_record(record)
