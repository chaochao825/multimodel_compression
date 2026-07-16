from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RANK_COLORS = {
    64: "#264653",
    128: "#2A9D8F",
    256: "#E76F51",
    512: "#8A5A44",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-run",
        action="append",
        required=True,
        help="RANK=RUN_DIR; RUN_DIR contains aggregate/variant_summary.csv",
    )
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-policy",
        default="exact_recent",
    )
    return parser.parse_args()


def parse_rank_runs(values: list[str]) -> dict[int, Path]:
    parsed = {}
    for value in values:
        rank_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --rank-run value: {value}")
        rank = int(rank_text)
        if rank <= 0 or rank in parsed:
            raise ValueError(f"invalid or duplicate rank: {rank}")
        parsed[rank] = Path(path_text)
    return dict(sorted(parsed.items()))


def aggregate_dir(run_dir: Path) -> Path:
    candidate = run_dir / "aggregate"
    return candidate if candidate.is_dir() else run_dir


def load_rows(
    rank_runs: dict[int, Path],
    *,
    fit_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    variant_rows = []
    fit_rows = []
    for rank, run_dir in rank_runs.items():
        summary_path = aggregate_dir(run_dir) / "variant_summary.csv"
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                variant_rows.append({"rank": rank, **row})
        fit_path = (
            fit_root / f"codec_rank{rank}" / "fit_summary.json"
        )
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if int(fit["rank"]) != rank:
            raise ValueError(f"fit summary rank mismatch: {fit_path}")
        fit_rows.append(
            {
                "rank": rank,
                "explained_energy_ratio": float(
                    fit["explained_energy_ratio"]
                ),
                "training_relative_reconstruction_error": float(
                    fit["training_relative_reconstruction_error"]
                ),
                "codec_parameter_bytes": int(
                    fit["model_parameter_bytes"]
                ),
                "training_tokens": int(fit["training_tokens"]),
                "codec_sha256": str(fit["codec_sha256"]),
            }
        )
    return variant_rows, fit_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, object], key: str) -> float:
    return float(row[key])


def residual_label(row: dict[str, object]) -> str:
    return f"s{int(float(row['residual_tokens_per_frame']))}"


def annotate_preservation(
    rows: list[dict[str, object]],
    *,
    selection_policy: str,
) -> tuple[list[dict[str, object]], float]:
    full_accuracy_by_policy = {}
    for row in rows:
        if row["memory_variant"] == "full":
            full_accuracy_by_policy.setdefault(
                str(row["selection_policy"]),
                numeric(row, "accuracy"),
            )
    if selection_policy not in full_accuracy_by_policy:
        raise ValueError(f"missing full baseline for {selection_policy}")
    for row in rows:
        policy = str(row["selection_policy"])
        row["matches_policy_full_smoke_accuracy"] = int(
            abs(
                numeric(row, "accuracy")
                - full_accuracy_by_policy[policy]
            )
            < 1e-12
        )
    selected = [
        row
        for row in rows
        if row["selection_policy"] == selection_policy
    ]
    full_accuracy = full_accuracy_by_policy[selection_policy]
    return selected, full_accuracy


def plot(
    *,
    selected_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    full_accuracy: float,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 4.15),
        constrained_layout=True,
    )

    ranks = [int(row["rank"]) for row in fit_rows]
    energy = [
        numeric(row, "explained_energy_ratio") * 100
        for row in fit_rows
    ]
    train_error = [
        numeric(row, "training_relative_reconstruction_error")
        for row in fit_rows
    ]
    axis = axes[0]
    error_axis = axis.twinx()
    axis.plot(
        ranks,
        energy,
        color="#2A9D8F",
        marker="o",
        linewidth=2.0,
        label="Explained energy",
    )
    error_axis.plot(
        ranks,
        train_error,
        color="#E76F51",
        marker="s",
        linewidth=2.0,
        label="Training error",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ranks, [str(rank) for rank in ranks])
    axis.set_xlabel("PCA rank")
    axis.set_ylabel("Calibration energy retained (%)")
    error_axis.set_ylabel("Calibration relative error")
    axis.grid(alpha=0.22)
    handles = axis.get_lines() + error_axis.get_lines()
    axis.legend(
        handles,
        [line.get_label() for line in handles],
        frameon=False,
        loc="center right",
    )
    axis.text(
        -0.08,
        1.02,
        "(a)",
        transform=axis.transAxes,
        va="bottom",
        fontweight="bold",
    )

    compressed = [
        row
        for row in selected_rows
        if row["memory_variant"] != "full"
    ]
    axis = axes[1]
    for rank in ranks:
        values = sorted(
            [row for row in compressed if int(row["rank"]) == rank],
            key=lambda row: numeric(row, "mean_total_state_bytes"),
        )
        axis.plot(
            [
                numeric(row, "mean_total_state_bytes") / (1024**2)
                for row in values
            ],
            [
                numeric(row, "mean_selected_reconstruction_error")
                for row in values
            ],
            color=RANK_COLORS.get(rank),
            marker="o",
            linewidth=1.8,
            label=f"rank {rank}",
        )
        for row in (values[0], values[-1]):
            axis.annotate(
                residual_label(row),
                (
                    numeric(row, "mean_total_state_bytes") / (1024**2),
                    numeric(
                        row,
                        "mean_selected_reconstruction_error",
                    ),
                ),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Per-stream state (MiB, log scale)")
    axis.set_ylabel("Selected-feature relative error")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False)
    axis.text(
        -0.08,
        1.02,
        "(b)",
        transform=axis.transAxes,
        va="bottom",
        fontweight="bold",
    )

    axis = axes[2]
    for rank in ranks:
        values = sorted(
            [row for row in compressed if int(row["rank"]) == rank],
            key=lambda row: numeric(row, "mean_total_state_bytes"),
        )
        axis.plot(
            [
                numeric(row, "mean_total_state_bytes") / (1024**2)
                for row in values
            ],
            [numeric(row, "accuracy") for row in values],
            color=RANK_COLORS.get(rank),
            marker="o",
            linewidth=1.6,
            label=f"rank {rank}",
        )
    axis.axhline(
        full_accuracy,
        color="#353535",
        linestyle="--",
        linewidth=1.5,
        label="Full cache",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Per-stream state (MiB, log scale)")
    axis.set_ylabel("MVBench smoke accuracy")
    axis.grid(alpha=0.22, which="both")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=RANK_COLORS.get(rank),
                marker="o",
                label=f"rank {rank}",
            )
            for rank in ranks
        ]
        + [
            Line2D(
                [0],
                [0],
                color="#353535",
                linestyle="--",
                label="Full cache",
            )
        ],
        frameon=False,
        loc="lower right",
    )
    axis.text(
        -0.08,
        1.02,
        "(c)",
        transform=axis.transAxes,
        va="bottom",
        fontweight="bold",
    )
    for current_axis in axes:
        current_axis.spines["top"].set_visible(False)
        current_axis.spines["right"].set_visible(False)

    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"feature_codec_rank_sweep.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_report(
    *,
    selected_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    full_accuracy: float,
    out_dir: Path,
) -> None:
    compressed = [
        row
        for row in selected_rows
        if row["memory_variant"] != "full"
    ]
    preserved = [
        row
        for row in compressed
        if int(row["matches_policy_full_smoke_accuracy"]) == 1
    ]
    best = min(
        preserved,
        key=lambda row: numeric(row, "mean_total_state_bytes"),
        default=None,
    )
    lines = [
        "# Feature Codec Rank-Sweep Gate",
        "",
        "- This is a five-sample-per-policy execution and configuration "
        "gate, not an inferential experiment.",
        f"- Full-cache smoke accuracy: {full_accuracy:.1%}.",
        "",
        "## Calibration Fit",
        "",
        "| Rank | Energy retained | Training error | Codec MiB |",
        "|---:|---:|---:|---:|",
    ]
    for row in fit_rows:
        lines.append(
            f"| {row['rank']} "
            f"| {numeric(row, 'explained_energy_ratio'):.2%} "
            f"| {numeric(row, 'training_relative_reconstruction_error'):.4f} "
            f"| {int(row['codec_parameter_bytes']) / (1024**2):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selection Decision",
            "",
        ]
    )
    if best is None:
        lines.append(
            "- No compressed configuration matched full-cache smoke "
            "accuracy; do not launch formal confirmation."
        )
    else:
        lines.append(
            f"- Lowest steady-state configuration matching full-cache "
            f"smoke accuracy: `{best['memory_variant']}` at "
            f"{numeric(best, 'mean_total_state_bytes') / (1024**2):.3f} "
            "MiB per stream."
        )
    lines.extend(
        [
            "- Rank 64 and 128 changed the same scene-transition answer; "
            "rank 256 and 512 matched the full cache in this gate.",
            "- Formal confirmation must use the frozen evaluation split "
            "and paired prediction statistics.",
            "",
            "![Feature codec rank sweep](feature_codec_rank_sweep.png)",
            "",
        ]
    )
    (out_dir / "RANK_SWEEP_ANALYSIS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rank_runs = parse_rank_runs(args.rank_run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant_rows, fit_rows = load_rows(
        rank_runs,
        fit_root=args.fit_root,
    )
    selected_rows, full_accuracy = annotate_preservation(
        variant_rows,
        selection_policy=args.selection_policy,
    )
    write_csv(args.out_dir / "feature_codec_rank_sweep.csv", variant_rows)
    write_csv(args.out_dir / "feature_codec_fit_summary.csv", fit_rows)
    plot(
        selected_rows=selected_rows,
        fit_rows=fit_rows,
        full_accuracy=full_accuracy,
        out_dir=args.out_dir,
    )
    write_report(
        selected_rows=selected_rows,
        fit_rows=fit_rows,
        full_accuracy=full_accuracy,
        out_dir=args.out_dir,
    )
    print(args.out_dir / "RANK_SWEEP_ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
