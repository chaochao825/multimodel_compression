#!/usr/bin/env python3
"""Re-evaluate saved content-tail checkpoints with both tile and shared bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from content_generated_tail_core import PositiveLinearTail
from probe_content_generated_tail import (
    aggregate,
    atomic_csv,
    atomic_json,
    build_decision,
    build_instances,
    capture_paths,
    load_captures,
    load_protocol,
    sha256_file,
    split_map,
    evaluate_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> PositiveLinearTail:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    projection = state["q_projection"]
    model = PositiveLinearTail(
        projection.shape[0],
        projection.shape[1],
        projection.shape[2],
        state["q_rms"],
        state["k_rms"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = load_protocol(args.protocol_config, args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    paths = capture_paths(args.capture_index, protocol)
    captures = load_captures(paths, device)
    instances = build_instances(captures, protocol)
    models = {
        "calibration_frozen": load_model(
            args.checkpoint_dir / "calibration_frozen.pt", device
        ),
        "transductive_capacity": load_model(
            args.checkpoint_dir / "transductive_capacity.pt", device
        ),
    }
    mapping = split_map(protocol)
    rows = []
    for sample_id in map(str, protocol["scope"]["sample_ids"]):
        rows.extend(
            evaluate_sample(
                models["calibration_frozen"],
                "calibration_frozen",
                sample_id,
                instances,
                protocol,
                mapping[sample_id],
            )
        )
        rows.extend(
            evaluate_sample(
                models["transductive_capacity"],
                "transductive_capacity",
                sample_id,
                instances,
                protocol,
                "transductive_fit",
            )
        )
    summary = aggregate(rows, protocol)
    atomic_csv(args.output_dir / "content_tail_records.csv", rows)
    atomic_csv(args.output_dir / "content_tail_summary.csv", summary)
    atomic_json(args.output_dir / "decision.json", build_decision(summary, protocol))
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "rank": args.rank,
            "protocol_sha256": sha256_file(args.protocol_config),
            "source_checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "source_checkpoint_sha256": {
                name: sha256_file(args.checkpoint_dir / f"{name}.pt")
                for name in models
            },
            "metrics": {
                "per_tile": "one adaptive rank-16 basis independently fit to each 64-query tile",
                "shared": "one adaptive rank-16 basis shared by all three sampled tiles per sample/head",
            },
            "warning": "This is deterministic checkpoint re-evaluation, not retraining.",
        },
    )
    atomic_json(
        args.output_dir / "SUCCESS.json",
        {
            "artifact_status": "SUCCESS",
            "decision_sha256": sha256_file(args.output_dir / "decision.json"),
            "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
