from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COLORS = {
    "control": "#7A7A7A",
    "noise": "#D55E00",
    "state": "#0072B2",
    "state_noise": "#009E73",
    "oracle": "#CC79A7",
}


def summaries(directory: str) -> list[tuple[str, dict]]:
    root = RESULTS / directory
    return [
        (path.parent.name, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("train*/summary.json"))
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def save_figure(figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            FIGURES / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    bridge_runs = summaries("action_dit_noise_response_bridge_20260826")
    multiskip_runs = summaries("action_dit_multiskip_state_20260826")
    sampler_runs = summaries("action_dit_independent_sampler_20260826")
    bound_rows = []

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    axis = axes[0, 0]
    bridge_methods = [
        ("Local", "shift_local_r2", "control"),
        ("Noise", "shift_rank8_noise", "noise"),
        ("State", "shift_rank8_state", "state"),
        ("State+noise", "shift_rank8_state_noise", "state_noise"),
        ("Oracle", "shift_rank8_oracle", "oracle"),
    ]
    for position, (label, method, color) in enumerate(bridge_methods):
        values = [
            run["exact_suffix"]["late"][method]["velocity_relative_l2"] * 100
            for _, run in bridge_runs
        ]
        axis.bar(position, np.mean(values), color=COLORS[color], width=0.68)
        axis.scatter(
            np.full(len(values), position),
            values,
            facecolors="white",
            edgecolors="#222222",
            zorder=3,
            s=30,
        )
        for (checkpoint, _), value in zip(bridge_runs, values):
            bound_rows.append(
                {
                    "stage": "B0.5",
                    "checkpoint": checkpoint,
                    "method": method,
                    "metric": "late_velocity_relative_l2_percent",
                    "value": value,
                }
            )
    axis.set_xticks(range(len(bridge_methods)), [item[0] for item in bridge_methods])
    axis.tick_params(axis="x", rotation=18)
    axis.set_ylabel("Late exact-suffix rel-L2 (%)")
    axis.text(-0.13, 1.04, "a", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)

    axis = axes[0, 1]
    skips = np.asarray([1, 2, 4])
    for label, method, color in (
        ("State", "state", "state"),
        ("State+noise", "state_noise", "state_noise"),
    ):
        matrix = np.asarray(
            [
                [
                    run["exact_suffix"]["late"][f"{method}_skip{skip}"][
                        "velocity_relative_l2"
                    ]
                    * 100
                    for skip in skips
                ]
                for _, run in multiskip_runs
            ]
        )
        axis.plot(
            skips,
            matrix.mean(axis=0),
            marker="o",
            linewidth=2,
            color=COLORS[color],
            label=label,
        )
        axis.fill_between(
            skips,
            matrix.min(axis=0),
            matrix.max(axis=0),
            color=COLORS[color],
            alpha=0.15,
        )
        for run_index, (checkpoint, _) in enumerate(multiskip_runs):
            for skip_index, skip in enumerate(skips):
                bound_rows.append(
                    {
                        "stage": "multi-skip",
                        "checkpoint": checkpoint,
                        "method": method,
                        "metric": f"late_skip{skip}_velocity_relative_l2_percent",
                        "value": matrix[run_index, skip_index],
                    }
                )
    oracle = np.mean(
        [
            run["exact_suffix"]["late"]["shift_rank8_oracle"][
                "velocity_relative_l2"
            ]
            * 100
            for _, run in multiskip_runs
        ]
    )
    axis.axhline(
        oracle,
        color=COLORS["oracle"],
        linestyle="--",
        linewidth=1.5,
        label="Rank-8 oracle",
    )
    axis.set_xticks(skips)
    axis.set_xlabel("Consecutive recurrent denoising steps")
    axis.set_ylabel("Late exact-suffix rel-L2 (%)")
    axis.legend(frameon=False)
    axis.text(-0.13, 1.04, "b", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)

    axis = axes[1, 0]
    for label, method, color in (
        ("State", "state", "state"),
        ("State+noise", "state_noise", "state_noise"),
    ):
        matrix = []
        for checkpoint, _ in multiskip_runs:
            rows = list(
                csv.DictReader(
                    (
                        RESULTS
                        / "action_dit_multiskip_state_20260826"
                        / checkpoint
                        / "endpoint_summary.csv"
                    ).open(encoding="utf-8")
                )
            )
            matrix.append(
                [
                    np.mean(
                        [
                            float(row["coefficient_r2"])
                            for row in rows
                            if row["method"] == f"{method}_skip{skip}"
                            and int(row["endpoint"]) >= 6
                        ]
                    )
                    for skip in skips
                ]
            )
        matrix = np.asarray(matrix)
        axis.plot(
            skips,
            matrix.mean(axis=0),
            marker="o",
            linewidth=2,
            color=COLORS[color],
            label=label,
        )
        axis.fill_between(
            skips,
            matrix.min(axis=0),
            matrix.max(axis=0),
            color=COLORS[color],
            alpha=0.15,
        )
        for run_index, (checkpoint, _) in enumerate(multiskip_runs):
            for skip_index, skip in enumerate(skips):
                bound_rows.append(
                    {
                        "stage": "multi-skip",
                        "checkpoint": checkpoint,
                        "method": method,
                        "metric": f"late_skip{skip}_coefficient_r2",
                        "value": matrix[run_index, skip_index],
                    }
                )
    axis.set_xticks(skips)
    axis.set_ylim(0.88, 1.005)
    axis.set_xlabel("Consecutive recurrent denoising steps")
    axis.set_ylabel("Late coefficient R-squared")
    axis.legend(frameon=False)
    axis.text(-0.13, 1.04, "c", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)

    axis = axes[1, 1]
    positions = np.arange(len(sampler_runs))
    width = 0.19
    entries = [
        ("All: aggregate", "all_interval5", "relative_l2", "state"),
        ("All: P95", "all_interval5", "p95_relative_l2", "noise"),
        ("Late20: aggregate", "late20_interval5", "relative_l2", "state_noise"),
        ("Late20: P95", "late20_interval5", "p95_relative_l2", "oracle"),
    ]
    for entry_index, (label, schedule, metric, color) in enumerate(entries):
        values = np.asarray(
            [run["metrics"][schedule]["executed"][metric] * 100 for _, run in sampler_runs]
        )
        x = positions + (entry_index - 1.5) * width
        axis.bar(x, values, width=width, color=COLORS[color], label=label)
        for (checkpoint, _), value in zip(sampler_runs, values):
            bound_rows.append(
                {
                    "stage": "B1a sampler",
                    "checkpoint": checkpoint,
                    "method": schedule,
                    "metric": f"executed_{metric}_percent",
                    "value": value,
                }
            )
    axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1, label="Aggregate gate")
    axis.axhline(2.0, color="#666666", linestyle=":", linewidth=1, label="P95 gate")
    axis.set_xticks(positions, [name.replace("_m8_v1", "") for name, _ in sampler_runs])
    axis.set_ylabel("Executed-action error (%)")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(-0.13, 1.04, "d", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)

    figure.tight_layout()
    save_figure(figure, "action_dit_tqc_gates")
    plt.close(figure)
    write_csv(FIGURES / "action_dit_tqc_gate_summary.csv", bound_rows)

    layer_path = (
        RESULTS
        / "action_dit_independent_sampler_layer_sweep_20260826"
        / "train0_m8_v1"
        / "summary.json"
    )
    layer_summary = json.loads(layer_path.read_text(encoding="utf-8"))
    subset_root = RESULTS / "action_dit_independent_sampler_subset_transfer_20260826"
    subset_runs = [
        (path.parent.name, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(subset_root.glob("train*/summary.json"))
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    axis = axes[0]
    layers = np.arange(len(layer_summary["layer_sweep"]))
    aggregate = np.asarray(
        [
            layer_summary["layer_sweep"][str(layer)]["metrics"]["executed"][
                "relative_l2"
            ]
            * 100
            for layer in layers
        ]
    )
    p95 = np.asarray(
        [
            layer_summary["layer_sweep"][str(layer)]["metrics"]["executed"][
                "p95_relative_l2"
            ]
            * 100
            for layer in layers
        ]
    )
    axis.bar(layers - 0.18, aggregate, width=0.36, color=COLORS["state"], label="Aggregate")
    axis.bar(layers + 0.18, p95, width=0.36, color=COLORS["noise"], label="P95")
    axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    axis.axhline(2.0, color="#666666", linestyle=":", linewidth=1)
    axis.set_xlabel("Approximated FFN layer (train-0 discovery)")
    axis.set_ylabel("Executed-action error (%)")
    axis.legend(frameon=False)
    axis.text(-0.12, 1.04, "a", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)
    layer_rows = []
    for layer, aggregate_value, p95_value in zip(layers, aggregate, p95):
        layer_rows.extend(
            [
                {"checkpoint": "train0_m8_v1", "candidate": f"layer_{layer}", "metric": "executed_relative_l2_percent", "value": aggregate_value},
                {"checkpoint": "train0_m8_v1", "candidate": f"layer_{layer}", "metric": "executed_p95_percent", "value": p95_value},
            ]
        )

    axis = axes[1]
    candidates = ["layers_1_2", "layers_1_7"]
    x = np.arange(len(subset_runs))
    for candidate_index, candidate in enumerate(candidates):
        values = np.asarray(
            [
                run["fixed_subsets"][candidate]["metrics"]["executed"][
                    "relative_l2"
                ]
                * 100
                for _, run in subset_runs
            ]
        )
        axis.bar(
            x + (candidate_index - 0.5) * 0.34,
            values,
            width=0.34,
            color=(COLORS["state_noise"], COLORS["control"])[candidate_index],
            label=candidate.replace("layers_", "layers ").replace("_", "-"),
        )
        for (checkpoint, _), value in zip(subset_runs, values):
            layer_rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": candidate,
                    "metric": "executed_relative_l2_percent",
                    "value": value,
                }
            )
    axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    axis.set_xticks(x, [name.replace("_m8_v1", "") for name, _ in subset_runs])
    axis.set_ylabel("Executed-action aggregate rel-L2 (%)")
    axis.legend(frameon=False)
    axis.text(-0.12, 1.04, "b", transform=axis.transAxes, fontweight="bold")
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, "action_dit_tqc_layer_attribution")
    plt.close(figure)
    write_csv(FIGURES / "action_dit_tqc_layer_attribution.csv", layer_rows)


if __name__ == "__main__":
    main()
