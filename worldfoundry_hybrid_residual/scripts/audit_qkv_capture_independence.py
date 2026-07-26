#!/usr/bin/env python3
"""Audit whether nominal QKV replay samples contain independent tensor content."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


Cell = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-sample-id", action="append", default=[])
    parser.add_argument("--validation-sample-id", action="append", default=[])
    parser.add_argument("--test-sample-id", action="append", default=[])
    parser.add_argument("--fail-on-cross-split-duplicate", action="store_true")
    return parser.parse_args()


def flatten(values: list[str]) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    )


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "layer", "sampling_step", "branch", "path"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"capture index is empty or missing columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_replay(index_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = index_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class DisjointSet:
    def __init__(self, labels: list[str]) -> None:
        self.parent = {label: label for label in labels}

    def find(self, label: str) -> str:
        parent = self.parent[label]
        if parent != label:
            self.parent[label] = self.find(parent)
        return self.parent[label]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for label in self.parent:
            grouped[self.find(label)].append(label)
        return sorted((sorted(values) for values in grouped.values()), key=lambda item: item[0])


def split_map(args: argparse.Namespace) -> dict[str, str]:
    groups = {
        "calibration": flatten(args.calibration_sample_id),
        "validation": flatten(args.validation_sample_id),
        "test": flatten(args.test_sample_id),
    }
    mapping: dict[str, str] = {}
    for split, samples in groups.items():
        for sample in samples:
            if sample in mapping:
                raise ValueError(f"sample {sample} appears in multiple nominal splits")
            mapping[sample] = split
    return mapping


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.capture_index = args.capture_index.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(args.capture_index)
    splits = split_map(args)
    by_cell: dict[Cell, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cell[(int(row["layer"]), int(row["sampling_step"]))].append(row)

    pair_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    cell_summaries: list[dict[str, object]] = []
    cross_split_duplicate_pairs = 0

    for cell, selected in sorted(by_cell.items()):
        payloads: dict[str, dict[str, torch.Tensor]] = {}
        sample_for_label: dict[str, str] = {}
        branch_for_label: dict[str, str] = {}
        for row in selected:
            label = f"{row['sample_id']}:{row['branch']}"
            if label in payloads:
                raise RuntimeError(f"duplicate replay label for cell={cell}: {label}")
            payload = torch.load(
                resolve_replay(args.capture_index, row["path"]),
                map_location="cpu",
                weights_only=False,
            )
            payloads[label] = {key: payload[key] for key in ("q", "k", "v")}
            sample_for_label[label] = row["sample_id"]
            branch_for_label[label] = row["branch"]

        labels = sorted(payloads)
        disjoint = DisjointSet(labels)
        for left, right in itertools.combinations(labels, 2):
            left_payload = payloads[left]
            right_payload = payloads[right]
            equality = {
                key: torch.equal(left_payload[key], right_payload[key])
                for key in ("q", "k", "v")
            }
            exact_qkv = all(equality.values())
            if exact_qkv:
                disjoint.union(left, right)
            left_split = splits.get(sample_for_label[left], "unspecified")
            right_split = splits.get(sample_for_label[right], "unspecified")
            cross_split = left_split != right_split
            if exact_qkv and cross_split:
                cross_split_duplicate_pairs += 1
            pair_rows.append(
                {
                    "layer": cell[0],
                    "sampling_step": cell[1],
                    "left": left,
                    "right": right,
                    "left_split": left_split,
                    "right_split": right_split,
                    "cross_split": cross_split,
                    "q_exact": equality["q"],
                    "k_exact": equality["k"],
                    "v_exact": equality["v"],
                    "qkv_exact": exact_qkv,
                    "q_max_abs": float(
                        (left_payload["q"].float() - right_payload["q"].float()).abs().max()
                    ),
                    "k_max_abs": float(
                        (left_payload["k"].float() - right_payload["k"].float()).abs().max()
                    ),
                    "v_max_abs": float(
                        (left_payload["v"].float() - right_payload["v"].float()).abs().max()
                    ),
                }
            )

        groups = disjoint.groups()
        for group_index, group in enumerate(groups):
            samples = sorted({sample_for_label[label] for label in group})
            branches = sorted({branch_for_label[label] for label in group})
            nominal_splits = sorted({splits.get(sample, "unspecified") for sample in samples})
            group_rows.append(
                {
                    "layer": cell[0],
                    "sampling_step": cell[1],
                    "content_group": group_index,
                    "replay_count": len(group),
                    "sample_count": len(samples),
                    "labels": "|".join(group),
                    "samples": "|".join(samples),
                    "branches": "|".join(branches),
                    "nominal_splits": "|".join(nominal_splits),
                    "cross_split_duplicate": len(nominal_splits) > 1,
                }
            )
        test_groups = {
            disjoint.find(label)
            for label in labels
            if splits.get(sample_for_label[label]) == "test"
        }
        cell_summaries.append(
            {
                "layer": cell[0],
                "sampling_step": cell[1],
                "nominal_replays": len(labels),
                "unique_qkv_content_groups": len(groups),
                "nominal_test_samples": len(
                    {sample_for_label[label] for label in labels if splits.get(sample_for_label[label]) == "test"}
                ),
                "unique_test_qkv_content_groups": len(test_groups),
                "content_independence_ratio": len(groups) / len(labels),
            }
        )

    write_csv(args.output_dir / "qkv_capture_independence_pairs.csv", pair_rows)
    write_csv(args.output_dir / "qkv_capture_content_groups.csv", group_rows)
    summary = {
        "capture_index": str(args.capture_index),
        "nominal_splits": splits,
        "cells": cell_summaries,
        "cross_split_duplicate_pairs": cross_split_duplicate_pairs,
        "cross_split_content_independent": cross_split_duplicate_pairs == 0,
        "interpretation": (
            "Sample IDs are not independent evidence when their captured Q/K/V tensors "
            "are bit-exact. At layer 0 before cross-attention, prompt and CFG branch may "
            "be structurally unobservable."
        ),
    }
    (args.output_dir / "qkv_capture_independence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.fail_on_cross_split_duplicate and cross_split_duplicate_pairs:
        sys.exit(2)


if __name__ == "__main__":
    main()
