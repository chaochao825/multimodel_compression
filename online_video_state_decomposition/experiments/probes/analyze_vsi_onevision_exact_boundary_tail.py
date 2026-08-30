from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from train_vsi_onevision_additive_nz_feature_state import (
    DEVELOPMENT_POSITIONS,
    capture_paths,
)
from vsi_onevision_protocol import PROTOCOL_ID


ROLE = "development_exact_boundary_tail_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--learned-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exact-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def support_geometry(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    exact_visual_output: torch.Tensor,
    *,
    exact_fraction: float,
) -> dict[str, torch.Tensor]:
    scores = torch.einsum("hd,hnd->hn", query, key).double()
    weights = torch.softmax(scores, dim=1)
    effect = weights * torch.linalg.vector_norm(
        value.double() - exact_visual_output.double().unsqueeze(1), dim=-1
    )
    token_count = key.shape[1]
    selected_count = int(round(token_count * exact_fraction))
    mass_indices = torch.topk(weights, k=selected_count, dim=1).indices
    effect_indices = torch.topk(effect, k=selected_count, dim=1).indices
    mass_mask = torch.zeros_like(weights, dtype=torch.bool)
    effect_mask = torch.zeros_like(weights, dtype=torch.bool)
    mass_mask.scatter_(1, mass_indices, True)
    effect_mask.scatter_(1, effect_indices, True)

    intersection = (mass_mask & effect_mask).sum(dim=1)
    union = (mass_mask | effect_mask).sum(dim=1)
    mass_retained = (weights * mass_mask).sum(dim=1)
    effect_retained_mass = (weights * effect_mask).sum(dim=1)
    tail_scores = scores.masked_fill(mass_mask, -torch.inf)
    tail_probability = torch.softmax(tail_scores, dim=1)
    tail_count = token_count - selected_count
    tail_ess = 1.0 / tail_probability.square().sum(dim=1).clamp_min(1e-12)
    tail_entropy = -(
        tail_probability * torch.log(tail_probability.clamp_min(1e-12))
    ).sum(dim=1) / np.log(tail_count)
    return {
        "mass_retained": mass_retained,
        "effect_retained_mass": effect_retained_mass,
        "support_jaccard": intersection / union,
        "tail_ess": tail_ess,
        "tail_ess_fraction": tail_ess / tail_count,
        "tail_normalized_entropy": tail_entropy,
        "tail_max_probability": tail_probability.max(dim=1).values,
    }


def read_learned_errors(path: Path) -> dict[tuple[str, int, int], float]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["selector"], int(row["position"]), int(row["layer_index"]))
            values[key] = float(row["visual_relative_l2"])
    return values


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    if args.exact_fraction != 0.25:
        raise ValueError("registered exact-boundary tail geometry changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("tail-geometry output must be empty")
    paths = capture_paths(args.capture_dir)
    learned_errors = read_learned_errors(args.learned_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []

    for position in DEVELOPMENT_POSITIONS:
        payload = torch.load(paths[position], map_location="cpu", weights_only=False)
        if payload["protocol_id"] != PROTOCOL_ID:
            raise ValueError("capture protocol identity mismatch")
        for layer_index in LAYERS:
            layer = payload["layers"][layer_index]
            geometry = support_geometry(
                layer["query_scaled"].to(device=device, dtype=torch.float32),
                layer["visual_key"].to(device=device, dtype=torch.float32),
                layer["visual_value"].to(device=device, dtype=torch.float32),
                layer["exact_visual_output"].to(device=device, dtype=torch.float32),
                exact_fraction=args.exact_fraction,
            )
            for head_index in range(layer["query_scaled"].shape[0]):
                row = {
                    "position": position,
                    "layer_index": layer_index,
                    "head_index": head_index,
                }
                row.update(
                    {
                        key: float(value[head_index].item())
                        for key, value in geometry.items()
                    }
                )
                row["mass_topk_visual_relative_l2"] = learned_errors[
                    ("mass_topk", position, layer_index)
                ]
                row["effect_topk_visual_relative_l2"] = learned_errors[
                    ("effect_topk", position, layer_index)
                ]
                rows.append(row)

    output_path = args.out_dir / "tail_geometry_head_rows.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer_index"])].append(row)
    metric_names = (
        "mass_retained",
        "effect_retained_mass",
        "support_jaccard",
        "tail_ess",
        "tail_ess_fraction",
        "tail_normalized_entropy",
        "tail_max_probability",
    )
    layer_summaries = {
        str(layer): {
            metric: summarize([float(row[metric]) for row in layer_rows])
            for metric in metric_names
        }
        for layer, layer_rows in sorted(by_layer.items())
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "exact_fraction": args.exact_fraction,
        "positions": [73, 96],
        "layers": list(LAYERS),
        "head_row_count": len(rows),
        "layer_summaries": layer_summaries,
        "claim_boundary": (
            "Post-result geometry diagnostic on the already exposed development "
            "partition. It cannot change the frozen Gate decision and reads no "
            "confirmation, selection, or formal example."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
