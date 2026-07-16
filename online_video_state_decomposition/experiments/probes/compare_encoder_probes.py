from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        action="append",
        required=True,
        help="label|aggregate_directory|hidden_diagnostics_json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_encoder(value: str) -> tuple[str, Path, Path]:
    fields = value.split("|", 2)
    if len(fields) != 3:
        raise ValueError(
            "encoder must be label|aggregate_directory|diagnostics_json"
        )
    return fields[0], Path(fields[1]).resolve(), Path(fields[2]).resolve()


def mean_diagnostics(path: Path) -> dict[int, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, list[float]]] = {}
    metrics = (
        "centered_top1_energy_fraction_mean",
        "centered_effective_rank_mean",
    )
    for row in data["summary"]:
        layer = int(row["layer"])
        layer_values = grouped.setdefault(
            layer,
            {metric: [] for metric in metrics},
        )
        for metric in metrics:
            layer_values[metric].append(float(row[metric]))
    return {
        layer: {
            metric: float(np.mean(values[metric]))
            for metric in metrics
        }
        for layer, values in grouped.items()
    }


def build_rows(
    label: str,
    aggregate_dir: Path,
    diagnostics_path: Path,
) -> list[dict[str, object]]:
    rank_rows = read_csv(aggregate_dir / "rank_summary.csv")
    subspace_rows = read_csv(aggregate_dir / "causal_subspace_summary.csv")
    transport_rows = read_csv(aggregate_dir / "transport_summary.csv")
    residual_rows = read_csv(aggregate_dir / "residual_summary.csv")
    diagnostics = mean_diagnostics(diagnostics_path)
    layers = sorted({int(row["layer"]) for row in rank_rows})
    output = []
    for layer in layers:
        def rank_value(alignment: str) -> float:
            row = next(
                row
                for row in rank_rows
                if int(row["layer"]) == layer
                and row["alignment"] == alignment
                and int(row["rank"]) == 32
            )
            return float(row["energy_ratio_mean"])

        causal_row = next(
            row
            for row in subspace_rows
            if int(row["layer"]) == layer
            and row["mode"] == "history_centered"
            and int(row["rank"]) == 32
        )
        transport = {
            row["method"]: row
            for row in transport_rows
            if int(row["layer"]) == layer
        }
        residual = next(
            row
            for row in residual_rows
            if int(row["layer"]) == layer
            and row["method"] == "identity"
            and abs(float(row["fraction"]) - 0.10) < 1e-9
        )
        identity_error = float(
            transport["identity"]["stable_centered_error_mean"]
        )
        bttb_error = float(
            transport["local_bttb_causal"]["stable_centered_error_mean"]
        )
        bccb_error = float(
            transport["local_bccb_causal"]["stable_centered_error_mean"]
        )
        layer_diagnostics = diagnostics[layer]
        output.append(
            {
                "encoder": label,
                "layer": layer,
                "history_centered_rank32_energy": rank_value(
                    "history_feature_subspace_centered"
                ),
                "history_token_normalized_rank32_energy": rank_value(
                    "history_feature_subspace_token_normalized"
                ),
                "causal_rank32_projection_error": float(
                    causal_row["relative_projection_error_mean"]
                ),
                "identity_stable_centered_error": identity_error,
                "local_bttb_stable_centered_error": bttb_error,
                "local_bccb_stable_centered_error": bccb_error,
                "local_bttb_relative_improvement": (
                    identity_error - bttb_error
                )
                / identity_error,
                "local_bccb_relative_improvement": (
                    identity_error - bccb_error
                )
                / identity_error,
                "residual_top10_energy": float(
                    residual["energy_ratio_mean"]
                ),
                "pixel_change_proxy_recall": float(
                    residual["pixel_change_recall_mean"]
                ),
                "centered_top1_energy": layer_diagnostics[
                    "centered_top1_energy_fraction_mean"
                ],
                "centered_effective_rank": layer_diagnostics[
                    "centered_effective_rank_mean"
                ],
                "collapsed_representation": (
                    layer_diagnostics[
                        "centered_top1_energy_fraction_mean"
                    ]
                    >= 0.95
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [
        f"{row['encoder']}\nL{row['layer']}"
        + ("*" if row["collapsed_representation"] else "")
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.4))

    axes[0, 0].bar(
        x - width / 2,
        [row["history_centered_rank32_energy"] for row in rows],
        width,
        color="#264653",
        label="centered history",
    )
    axes[0, 0].bar(
        x + width / 2,
        [row["history_token_normalized_rank32_energy"] for row in rows],
        width,
        color="#2a9d8f",
        label="token-normalized history",
    )
    axes[0, 0].axhline(
        0.70,
        color="#d1495b",
        linestyle="--",
        linewidth=1.2,
        label="70% spectral gate",
    )
    axes[0, 0].set_ylabel("Rank-32 explained energy")
    axes[0, 0].set_ylim(0, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=8)

    encoder_colors = {
        label: color
        for label, color in zip(
            sorted({str(row["encoder"]) for row in rows}),
            ["#e76f51", "#457b9d", "#8d5a97"],
            strict=False,
        )
    }
    bars = axes[0, 1].bar(
        x,
        [row["causal_rank32_projection_error"] for row in rows],
        color=[encoder_colors[str(row["encoder"])] for row in rows],
    )
    axes[0, 1].set_ylabel("Causal rank-32 projection error")
    axes[0, 1].set_ylim(0, 1.03)

    axes[1, 0].bar(
        x - width / 2,
        [row["local_bttb_relative_improvement"] for row in rows],
        width,
        color="#e9c46a",
        label="local BTTB",
    )
    axes[1, 0].bar(
        x + width / 2,
        [row["local_bccb_relative_improvement"] for row in rows],
        width,
        color="#e76f51",
        label="local BCCB",
    )
    axes[1, 0].axhline(
        0.10,
        color="#d1495b",
        linestyle="--",
        linewidth=1.2,
        label="10% transport gate",
    )
    axes[1, 0].axhline(0.0, color="#495057", linewidth=0.8)
    axes[1, 0].set_ylabel("Stable-error improvement vs identity")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(
        x - width / 2,
        [row["residual_top10_energy"] for row in rows],
        width,
        color="#8d5a97",
        label="top-10% residual energy",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["pixel_change_proxy_recall"] for row in rows],
        width,
        color="#2a9d8f",
        label="pixel-change proxy recall",
    )
    axes[1, 1].axhline(
        0.70,
        color="#d1495b",
        linestyle="--",
        linewidth=1.2,
        label="70% energy gate",
    )
    axes[1, 1].axhline(
        0.80,
        color="#f4a261",
        linestyle=":",
        linewidth=1.2,
        label="80% event-recall gate",
    )
    axes[1, 1].set_ylabel("Residual concentration / proxy recall")
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=42, ha="right", fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for bar, row in zip(bars, rows, strict=True):
        if row["collapsed_representation"]:
            bar.set_hatch("///")
            bar.set_edgecolor("#212529")
    fig.text(
        0.01,
        0.01,
        "* centered top-1 energy >=95%; treat as representation collapse, "
        "not compressor evidence.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    inputs = []
    for value in args.encoder:
        label, aggregate_dir, diagnostics_path = parse_encoder(value)
        rows.extend(build_rows(label, aggregate_dir, diagnostics_path))
        inputs.append(
            {
                "label": label,
                "aggregate_dir": str(aggregate_dir),
                "diagnostics_json": str(diagnostics_path),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "cross_encoder_probe_summary.csv", rows)
    plot_rows(args.out_dir / "cross_encoder_probe_summary.png", rows)
    (args.out_dir / "cross_encoder_probe_summary.json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "rows": rows,
                "thresholds": {
                    "spectral_energy": 0.70,
                    "transport_improvement": 0.10,
                    "residual_top10_energy": 0.70,
                    "event_recall": 0.80,
                    "collapse_top1_energy": 0.95,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "encoders": len(args.encoder),
                "rows": len(rows),
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
