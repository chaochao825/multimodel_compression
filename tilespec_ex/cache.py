"""Validated reader for TileLogic feature caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


CACHE_FORMAT = "tilelogic_feature_cache_v1"
DATASETS = ("gqa", "textvqa", "chartqa")
SPLITS = ("calibration", "evaluation")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    dataset: str
    dataset_index: int
    sample_id: str
    image_sha256: str
    source_manifest_sha256: str
    model_revision: str
    split: str
    split_rank: int
    oracle: bool
    cache_file: str
    bytes: int
    sha256: str
    thumbnail_shape: tuple[int, ...]
    crop_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...] | None
    tensor_dtypes: dict[str, str]

    @property
    def key(self) -> tuple[str, int]:
        return self.dataset, self.dataset_index

    @staticmethod
    def from_record(record: dict[str, Any]) -> "CacheEntry":
        if record.get("format") != CACHE_FORMAT:
            raise ValueError("unsupported cache-manifest format")
        gradient = record.get("gradient_shape")
        entry = CacheEntry(
            dataset=str(record["dataset"]),
            dataset_index=int(record["dataset_index"]),
            sample_id=str(record["sample_id"]),
            image_sha256=str(record["image_sha256"]),
            source_manifest_sha256=str(record["source_manifest_sha256"]),
            model_revision=str(record["model_revision"]),
            split=str(record["split"]),
            split_rank=int(record["split_rank"]),
            oracle=bool(record["oracle"]),
            cache_file=str(record["cache_file"]),
            bytes=int(record["bytes"]),
            sha256=str(record["sha256"]),
            thumbnail_shape=tuple(int(item) for item in record["thumbnail_shape"]),
            crop_shape=tuple(int(item) for item in record["crop_shape"]),
            query_shape=tuple(int(item) for item in record["query_shape"]),
            gradient_shape=(
                tuple(int(item) for item in gradient) if gradient is not None else None
            ),
            tensor_dtypes={
                str(name): str(dtype)
                for name, dtype in record["tensor_dtypes"].items()
            },
        )
        if entry.dataset not in DATASETS or entry.split not in SPLITS:
            raise ValueError("cache entry has unsupported dataset or split")
        if entry.bytes <= 0 or len(entry.sha256) != 64:
            raise ValueError("cache entry has invalid size or SHA-256")
        if len(entry.source_manifest_sha256) != 64 or not entry.model_revision:
            raise ValueError("cache entry has invalid source/model provenance")
        if entry.oracle != (entry.gradient_shape is not None):
            raise ValueError("oracle flag and gradient shape disagree")
        expected_dtype_fields = {"thumbnail", "crops", "query"}
        if entry.oracle:
            expected_dtype_fields.add("crop_gradient")
        if set(entry.tensor_dtypes) != expected_dtype_fields:
            raise ValueError("cache entry tensor dtype fields are incomplete")
        return entry


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def load_cache_manifest(
    cache_dir: Path,
    *,
    verify_files: bool = False,
) -> list[CacheEntry]:
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "cache_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    entries = [CacheEntry.from_record(item) for item in _jsonl(manifest_path)]
    keys: set[tuple[str, int]] = set()
    for entry in entries:
        if entry.key in keys:
            raise RuntimeError(f"duplicate cache key: {entry.key}")
        keys.add(entry.key)
        path = cache_dir / entry.cache_file
        if not path.is_file() or path.stat().st_size != entry.bytes:
            raise RuntimeError(f"missing or truncated cache file: {path}")
        if verify_files and file_sha256(path) != entry.sha256:
            raise RuntimeError(f"cache SHA-256 mismatch: {path}")
    return sorted(entries, key=lambda item: (item.dataset, item.dataset_index))


def validate_split_contract(
    entries: Iterable[CacheEntry],
    *,
    calibration_per_dataset: int,
    evaluation_per_dataset: int,
    oracle_per_dataset_split: int,
) -> dict[str, dict[str, int]]:
    entries = tuple(entries)
    counts: dict[str, dict[str, int]] = {}
    keys = {entry.key for entry in entries}
    if len(keys) != len(entries):
        raise RuntimeError("cache entries contain duplicate keys")
    for dataset in DATASETS:
        counts[dataset] = {}
        calibration_keys = {
            entry.key
            for entry in entries
            if entry.dataset == dataset and entry.split == "calibration"
        }
        evaluation_keys = {
            entry.key
            for entry in entries
            if entry.dataset == dataset and entry.split == "evaluation"
        }
        if calibration_keys & evaluation_keys:
            raise RuntimeError(f"split overlap for {dataset}")
        for split, expected in (
            ("calibration", calibration_per_dataset),
            ("evaluation", evaluation_per_dataset),
        ):
            selected = [
                entry
                for entry in entries
                if entry.dataset == dataset and entry.split == split
            ]
            oracle = sum(entry.oracle for entry in selected)
            if len(selected) != expected or oracle != oracle_per_dataset_split:
                raise RuntimeError(
                    f"cache split contract mismatch for {dataset}/{split}: "
                    f"records={len(selected)}, oracle={oracle}"
                )
            counts[dataset][split] = len(selected)
            counts[dataset][f"{split}_oracle"] = oracle
    return counts


def load_cache_payload(
    cache_dir: Path,
    entry: CacheEntry,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    path = cache_dir.resolve() / entry.cache_file
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"unsupported feature payload: {path}")
    expected = {
        "dataset": entry.dataset,
        "dataset_index": entry.dataset_index,
        "sample_id": entry.sample_id,
        "image_sha256": entry.image_sha256,
        "split": entry.split,
        "split_rank": entry.split_rank,
        "oracle": entry.oracle,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"cache payload mismatch for {entry.key}: {name}")
    optional_provenance = {
        "source_manifest_sha256": entry.source_manifest_sha256,
        "model_revision": entry.model_revision,
        "tensor_dtypes": entry.tensor_dtypes,
    }
    for name, value in optional_provenance.items():
        if name in payload and payload[name] != value:
            raise RuntimeError(
                f"cache payload provenance mismatch for {entry.key}: {name}"
            )
    shape_contract = {
        "thumbnail": entry.thumbnail_shape,
        "crops": entry.crop_shape,
        "query": entry.query_shape,
    }
    for name, shape in shape_contract.items():
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise RuntimeError(f"cache tensor shape mismatch for {entry.key}: {name}")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise RuntimeError(f"invalid cache tensor for {entry.key}: {name}")
        if str(tensor.dtype).removeprefix("torch.") != entry.tensor_dtypes[name]:
            raise RuntimeError(f"cache tensor dtype mismatch for {entry.key}: {name}")
    gradient = payload.get("crop_gradient")
    if entry.oracle:
        if not isinstance(gradient, torch.Tensor) or tuple(gradient.shape) != entry.gradient_shape:
            raise RuntimeError(f"oracle gradient mismatch for {entry.key}")
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite oracle gradient for {entry.key}")
        if (
            str(gradient.dtype).removeprefix("torch.")
            != entry.tensor_dtypes["crop_gradient"]
        ):
            raise RuntimeError(f"oracle gradient dtype mismatch for {entry.key}")
    elif gradient is not None:
        raise RuntimeError(f"non-oracle cache contains gradient: {entry.key}")
    return payload


def manifest_sha256(cache_dir: Path) -> str:
    return hashlib.sha256(
        (cache_dir.resolve() / "cache_manifest.jsonl").read_bytes()
    ).hexdigest()
