from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvbench_llava_anchor import write_json_atomic
from onevision_reader_quotient_stage_a import descending_eigenspace, tail_energy_fraction
from vsi_onevision_protocol import PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.input_dir) < 2:
        raise ValueError("at least two reader-risk artifacts are required")
    device = torch.device(args.device)
    summaries = []
    artifacts = []
    sample_ids = []
    for input_dir in args.input_dir:
        summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
        if summary["protocol_id"] != PROTOCOL_ID:
            raise ValueError("reader-risk protocol identity mismatch")
        overlap = set(sample_ids) & set(summary["sample_ids"])
        if overlap:
            raise ValueError(f"reader-risk inputs overlap: {sorted(overlap)}")
        summaries.append(summary)
        sample_ids.extend(summary["sample_ids"])
        artifacts.append(
            torch.load(
                input_dir / "reader_risk_artifact.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    ranks = {int(summary["rank"]) for summary in summaries}
    margin_floors = {float(summary["margin_floor"]) for summary in summaries}
    if len(ranks) != 1 or len(margin_floors) != 1:
        raise ValueError("reader-risk inputs use different rank or margin floor")
    sample_counts = [int(summary["sample_count"]) for summary in summaries]
    total_samples = sum(sample_counts)
    risk = torch.zeros_like(artifacts[0]["risk_matrix"], device=device)
    for artifact, sample_count in zip(artifacts, sample_counts, strict=True):
        risk.add_(
            artifact["risk_matrix"].to(device=device, dtype=torch.float32),
            alpha=sample_count / total_samples,
        )
    rank = ranks.pop()
    eigenvalues, basis = descending_eigenspace(risk, rank=rank)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "risk_matrix": risk.cpu(),
            "risk_eigenvalues": eigenvalues.cpu(),
            "risk_basis": basis.cpu(),
        },
        args.out_dir / "reader_risk_artifact.pt",
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_only_merged_reader_risk",
        "sample_count": total_samples,
        "sample_ids": sample_ids,
        "margin_floor": margin_floors.pop(),
        "rank": rank,
        "risk_tail_energy_fraction": tail_energy_fraction(
            eigenvalues.cpu(), rank=rank
        ),
        "input_dirs": [str(path) for path in args.input_dir],
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
