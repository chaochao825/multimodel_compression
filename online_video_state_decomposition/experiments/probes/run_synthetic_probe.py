from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.metrics import (  # noqa: E402
    relative_fro_error,
    residual_concentration,
    singular_value_metrics,
)
from probes.synthetic import SyntheticSequence, generate_synthetic_sequence  # noqa: E402
from probes.transport import (  # noqa: E402
    apply_global_bccb,
    apply_shift_basis,
    best_integer_shift,
    fit_global_bccb,
    fit_shift_basis,
    local_offsets,
    shift_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "probe_mvp.yaml",
    )
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def rank_metrics(sequence: SyntheticSequence, ranks: list[int]) -> dict[str, object]:
    observed = sequence.observed
    aligned = np.stack(
        [
            shift_grid(
                observed[frame],
                -int(sequence.shifts[frame, 0]),
                -int(sequence.shifts[frame, 1]),
                cyclic=True,
            )
            for frame in range(observed.shape[0])
        ],
        axis=0,
    )
    unaligned_temporal = observed.reshape(observed.shape[0], -1)
    aligned_temporal = aligned.reshape(aligned.shape[0], -1)
    history_tokens = aligned.reshape(-1, aligned.shape[-1])
    frame_spatial = [
        singular_value_metrics(
            aligned[frame].reshape(-1, aligned.shape[-1]),
            ranks,
        )
        for frame in range(aligned.shape[0])
    ]
    return {
        "suite": "state",
        "seed": int(sequence.metadata["seed"]),
        "unaligned_temporal": singular_value_metrics(unaligned_temporal, ranks),
        "aligned_temporal": singular_value_metrics(aligned_temporal, ranks),
        "aligned_history_tokens": singular_value_metrics(history_tokens, ranks),
        "mean_frame_spatial_effective_rank": mean(
            [float(row["effective_rank"]) for row in frame_spatial]
        ),
        "mean_frame_spatial_stable_rank": mean(
            [float(row["stable_rank"]) for row in frame_spatial]
        ),
    }


def transition_metrics(
    sequence: SyntheticSequence,
    cfg: dict[str, object],
    *,
    suite: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    probe_cfg = cfg["probe"]
    radius = int(probe_cfg["local_radius"])
    ridge = float(probe_cfg["ridge"])
    block_size = int(probe_cfg["block_size"])
    fractions = [float(value) for value in probe_cfg["top_fractions"]]
    offsets = local_offsets(radius)
    rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    previous_local_bccb_weights: np.ndarray | None = None
    previous_local_bttb_weights: np.ndarray | None = None
    previous_global_bccb_kernel: np.ndarray | None = None

    for frame in range(1, sequence.observed.shape[0]):
        previous = sequence.observed[frame - 1]
        current = sequence.observed[frame]
        is_event = bool(sequence.event_mask[frame].any())
        previous_is_event = bool(sequence.event_mask[frame - 1].any())
        event_transition = is_event or previous_is_event
        is_scene_cut = frame == sequence.scene_cut_frame
        delta = sequence.shifts[frame] - sequence.shifts[frame - 1]
        est_dy, est_dx, _est_error = best_integer_shift(
            previous,
            current,
            max_shift=int(sequence.metadata["max_step_shift"]),
            cyclic=True,
        )

        pair_local_bccb, local_bccb_weights = fit_shift_basis(
            previous,
            current,
            offsets,
            cyclic=True,
            ridge=ridge,
        )
        pair_local_bttb, local_bttb_weights = fit_shift_basis(
            previous,
            current,
            offsets,
            cyclic=False,
            ridge=ridge,
        )
        pair_global_bccb, global_bccb_kernel = fit_global_bccb(
            previous,
            current,
            ridge=ridge,
        )
        predictions: dict[str, tuple[np.ndarray, str]] = {
            "identity": (previous, "fixed"),
            "oracle_shift": (
                shift_grid(
                    previous,
                    int(delta[0]),
                    int(delta[1]),
                    cyclic=True,
                ),
                "ground_truth",
            ),
            "estimated_shift": (
                shift_grid(
                    previous,
                    est_dy,
                    est_dx,
                    cyclic=True,
                ),
                "pair_estimated",
            ),
            "local_bccb_pair": (pair_local_bccb, "pair_oracle"),
            "local_bttb_pair": (pair_local_bttb, "pair_oracle"),
            "global_bccb_pair": (pair_global_bccb, "pair_oracle"),
        }
        if previous_local_bccb_weights is not None:
            predictions["local_bccb_causal"] = (
                apply_shift_basis(
                    previous,
                    offsets,
                    previous_local_bccb_weights,
                    cyclic=True,
                ),
                "previous_transition",
            )
        if previous_local_bttb_weights is not None:
            predictions["local_bttb_causal"] = (
                apply_shift_basis(
                    previous,
                    offsets,
                    previous_local_bttb_weights,
                    cyclic=False,
                ),
                "previous_transition",
            )
        if previous_global_bccb_kernel is not None:
            predictions["global_bccb_causal"] = (
                apply_global_bccb(previous, previous_global_bccb_kernel),
                "previous_transition",
            )

        for method, (prediction, scope) in predictions.items():
            residual = current - prediction
            rows.append(
                {
                    "suite": suite,
                    "seed": int(sequence.metadata["seed"]),
                    "frame": frame,
                    "method": method,
                    "scope": scope,
                    "relative_error": relative_fro_error(current, prediction),
                    "is_event": int(is_event),
                    "previous_is_event": int(previous_is_event),
                    "event_transition": int(event_transition),
                    "is_scene_cut": int(is_scene_cut),
                    "true_dy": int(delta[0]),
                    "true_dx": int(delta[1]),
                    "estimated_dy": int(est_dy),
                    "estimated_dx": int(est_dx),
                }
            )
            if method == "oracle_shift":
                concentration = residual_concentration(
                    residual,
                    block_size,
                    fractions,
                    event_mask=sequence.event_mask[frame],
                )
                for fraction, values in concentration["top"].items():
                    concentration_rows.append(
                        {
                            "suite": suite,
                            "seed": int(sequence.metadata["seed"]),
                            "frame": frame,
                            "fraction": float(fraction),
                            "energy_ratio": float(values["energy_ratio"]),
                            "event_recall": float(values["event_recall"]),
                            "event_blocks": int(values["event_blocks"]),
                            "gini": float(concentration["gini"]),
                            "is_event": int(is_event),
                            "previous_is_event": int(previous_is_event),
                            "event_transition": int(event_transition),
                            "is_scene_cut": int(is_scene_cut),
                        }
                    )

        previous_local_bccb_weights = local_bccb_weights
        previous_local_bttb_weights = local_bttb_weights
        previous_global_bccb_kernel = global_bccb_kernel
    return rows, concentration_rows


def build_suite(
    cfg: dict[str, object],
    *,
    seed: int,
    suite: str,
) -> SyntheticSequence:
    synthetic_cfg = cfg["synthetic"]
    common = {
        "seed": seed,
        "frames": int(synthetic_cfg["frames"]),
        "height": int(synthetic_cfg["height"]),
        "width": int(synthetic_cfg["width"]),
        "hidden_dim": int(synthetic_cfg["hidden_dim"]),
        "spatial_rank": int(synthetic_cfg["spatial_rank"]),
        "event_block_size": int(synthetic_cfg["event_block_size"]),
        "event_amplitude": float(synthetic_cfg["event_amplitude"]),
        "max_step_shift": int(synthetic_cfg["max_step_shift"]),
    }
    constant_step = tuple(int(v) for v in synthetic_cfg["transport_step"])
    if suite == "transport":
        return generate_synthetic_sequence(
            **common,
            temporal_modes=0,
            event_frames=(),
            scene_cut_frame=None,
            constant_step=constant_step,
        )
    if suite == "state":
        return generate_synthetic_sequence(
            **common,
            temporal_modes=int(synthetic_cfg["temporal_modes"]),
            event_frames=(),
            scene_cut_frame=None,
        )
    if suite == "event":
        return generate_synthetic_sequence(
            **common,
            temporal_modes=0,
            event_frames=tuple(
                int(value) for value in synthetic_cfg["event_frames"]
            ),
            scene_cut_frame=None,
            constant_step=constant_step,
        )
    if suite == "scene_cut":
        return generate_synthetic_sequence(
            **common,
            temporal_modes=0,
            event_frames=(),
            scene_cut_frame=int(synthetic_cfg["scene_cut_frame"]),
            constant_step=constant_step,
        )
    raise ValueError(f"unknown suite: {suite}")


def summarize(
    transition_rows: list[dict[str, object]],
    concentration_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    gates: dict[str, float],
) -> dict[str, object]:
    method_summary: dict[str, dict[str, dict[str, float]]] = {}
    for suite in sorted({str(row["suite"]) for row in transition_rows}):
        method_summary[suite] = {}
        suite_rows = [row for row in transition_rows if row["suite"] == suite]
        for method in sorted({str(row["method"]) for row in suite_rows}):
            selected = [row for row in suite_rows if row["method"] == method]
            stable = [
                row
                for row in selected
                if not bool(row["event_transition"])
                and not bool(row["is_scene_cut"])
            ]
            events = [row for row in selected if bool(row["is_event"])]
            method_summary[suite][method] = {
                "mean_error": mean(
                    [float(row["relative_error"]) for row in selected]
                ),
                "stable_error": mean(
                    [float(row["relative_error"]) for row in stable]
                ),
                "event_error": mean_or_none(
                    [float(row["relative_error"]) for row in events]
                ),
            }

    top10_events = [
        row
        for row in concentration_rows
        if row["suite"] == "event"
        and abs(float(row["fraction"]) - 0.1) < 1e-9
        and bool(row["is_event"])
    ]
    aligned_rank4 = mean(
        [
            float(row["aligned_temporal"]["energy_at_rank"]["4"])
            for row in rank_rows
        ]
    )
    event_top10_energy = mean(
        [float(row["energy_ratio"]) for row in top10_events]
    )
    event_top10_recall = mean(
        [float(row["event_recall"]) for row in top10_events]
    )
    identity_stable = float(
        method_summary["transport"]["identity"]["stable_error"]
    )
    structured_stable = min(
        float(
            method_summary["transport"]["local_bccb_causal"]["stable_error"]
        ),
        float(
            method_summary["transport"]["local_bttb_causal"]["stable_error"]
        ),
    )
    structured_improvement = (
        (identity_stable - structured_stable) / (identity_stable + 1e-12)
    )
    oracle_transport_error = float(
        method_summary["transport"]["oracle_shift"]["stable_error"]
    )

    gate_results = {
        "aligned_temporal_energy_rank4": {
            "value": aligned_rank4,
            "threshold": float(gates["aligned_temporal_energy_rank4"]),
            "passed": aligned_rank4
            >= float(gates["aligned_temporal_energy_rank4"]),
        },
        "event_top10_energy": {
            "value": event_top10_energy,
            "threshold": float(gates["event_top10_energy"]),
            "passed": event_top10_energy >= float(gates["event_top10_energy"]),
        },
        "event_top10_recall": {
            "value": event_top10_recall,
            "threshold": float(gates["event_top10_recall"]),
            "passed": event_top10_recall >= float(gates["event_top10_recall"]),
        },
        "structured_relative_improvement": {
            "value": structured_improvement,
            "threshold": float(gates["structured_relative_improvement"]),
            "passed": structured_improvement
            >= float(gates["structured_relative_improvement"]),
        },
        "oracle_transport_sanity": {
            "value": oracle_transport_error,
            "threshold": 1e-8,
            "passed": oracle_transport_error <= 1e-8,
        },
    }
    return {
        "method_summary": method_summary,
        "rank_summary": {
            "aligned_temporal_energy_rank4": aligned_rank4,
            "mean_aligned_temporal_effective_rank": mean(
                [
                    float(row["aligned_temporal"]["effective_rank"])
                    for row in rank_rows
                ]
            ),
            "mean_unaligned_temporal_effective_rank": mean(
                [
                    float(row["unaligned_temporal"]["effective_rank"])
                    for row in rank_rows
                ]
            ),
        },
        "event_summary": {
            "top10_energy": event_top10_energy,
            "top10_event_recall": event_top10_recall,
        },
        "gate_results": gate_results,
        "all_gates_passed": all(
            bool(row["passed"]) for row in gate_results.values()
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.out_dir
    if output_root is None:
        output_root = (
            args.config.parent / str(cfg["outputs"]["root"])
        ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ranks = [int(value) for value in cfg["probe"]["ranks"]]
    all_transition_rows: list[dict[str, object]] = []
    all_concentration_rows: list[dict[str, object]] = []
    all_rank_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed_value in cfg["synthetic"]["seeds"]:
        seed = int(seed_value)
        for suite in ("transport", "state", "event", "scene_cut"):
            sequence = build_suite(cfg, seed=seed, suite=suite)
            transition_rows, concentration_rows = transition_metrics(
                sequence,
                cfg,
                suite=suite,
            )
            all_transition_rows.extend(transition_rows)
            all_concentration_rows.extend(concentration_rows)
            metadata_rows.append({"suite": suite, **sequence.metadata})
            if suite == "state":
                all_rank_rows.append(rank_metrics(sequence, ranks))

    summary = summarize(
        all_transition_rows,
        all_concentration_rows,
        all_rank_rows,
        cfg["gates"],
    )
    result = {
        "config": str(args.config.resolve()),
        "metadata": metadata_rows,
        "summary": summary,
        "rank_metrics": all_rank_rows,
    }
    (output_root / "synthetic_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(output_root / "synthetic_transport.csv", all_transition_rows)
    write_csv(
        output_root / "synthetic_residual_concentration.csv",
        all_concentration_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
