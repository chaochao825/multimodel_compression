#!/usr/bin/env python3
"""Small, dependency-free helpers for traceable experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write beside the target and atomically publish with os.replace()."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def require_fresh_output_dir(path: Path) -> None:
    """Reject reuse so stale SUCCESS markers cannot overlap a new run."""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; stage it in trash before rerun: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


class JsonlEventLog:
    """Append-only progress log; final artifacts remain guarded by SUCCESS.json."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: object) -> None:
        record = {
            "event": event,
            "run_id": self.run_id,
            "time_unix": time.time(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()


@dataclass(frozen=True)
class SplitProtocol:
    name: str
    calibration: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    claim_boundary: str

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset((*self.calibration, *self.validation, *self.test))

    def validate(self) -> None:
        groups = {
            "calibration": self.calibration,
            "validation": self.validation,
            "test": self.test,
        }
        if not self.name:
            raise ValueError("split protocol name must not be empty")
        if any(not values for values in groups.values()):
            raise ValueError(f"protocol {self.name} has an empty split")
        flattened = [sample for values in groups.values() for sample in values]
        duplicates = sorted(
            sample for sample in set(flattened) if flattened.count(sample) > 1
        )
        if duplicates:
            raise ValueError(
                f"protocol {self.name} assigns samples more than once: {duplicates}"
            )

    def assert_exact_coverage(self, observed: Iterable[str]) -> None:
        observed_ids = frozenset(observed)
        missing = sorted(self.sample_ids - observed_ids)
        unexpected = sorted(observed_ids - self.sample_ids)
        if missing or unexpected:
            raise ValueError(
                f"protocol {self.name} coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def split_for(self, sample_id: str) -> str:
        if sample_id in self.calibration:
            return "calibration"
        if sample_id in self.validation:
            return "validation"
        if sample_id in self.test:
            return "test"
        raise KeyError(f"sample {sample_id!r} is not registered in {self.name}")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "calibration": list(self.calibration),
            "validation": list(self.validation),
            "test": list(self.test),
            "claim_boundary": self.claim_boundary,
        }


def load_split_protocols(path: Path) -> tuple[SplitProtocol, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported split schema in {path}")
    protocols = tuple(
        SplitProtocol(
            name=str(raw["name"]),
            calibration=tuple(map(str, raw["calibration"])),
            validation=tuple(map(str, raw["validation"])),
            test=tuple(map(str, raw["test"])),
            claim_boundary=str(raw["claim_boundary"]),
        )
        for raw in payload["protocols"]
    )
    if not protocols:
        raise ValueError("split config must contain at least one protocol")
    names = [protocol.name for protocol in protocols]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate split protocol names: {names}")
    for protocol in protocols:
        protocol.validate()
    return protocols
