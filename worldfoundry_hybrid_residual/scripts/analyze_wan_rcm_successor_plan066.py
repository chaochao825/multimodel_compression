#!/usr/bin/env python3
"""Build the PLAN-066 exact-system successor decision surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


REQUEST_TARGETS = (1.05, 1.10, 1.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "worldfoundry_hybrid_residual/results/wan_rcm_successor_plan066_20260901"
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def required_local_speedup(
    total: float, component: float, request_speedup: float
) -> float:
    target_component = total / request_speedup - (total - component)
    if target_component <= 0:
        return math.inf
    return component / target_component


def projected_request_speedup(
    total: float, component: float, local_speedup: float
) -> float:
    return total / (total - component + component / local_speedup)


def load_inputs(repo_root: Path) -> tuple[dict[str, float], float, float]:
    runtime_path = (
        repo_root
        / "worldfoundry_hybrid_residual/results/wan_rcm_exact_runtime_exp052_20260901"
        / "local_evaluation_v1/method_summary.csv"
    )
    runtime_rows = read_csv_rows(runtime_path)
    rcm_row = next(row for row in runtime_rows if row["method"] == "rcm4")
    runtime = {
        "request": float(rcm_row["median_request_seconds"]),
        "text": float(rcm_row["median_text_seconds"]),
        "denoiser": float(rcm_row["median_denoiser_seconds"]),
        "vae": float(rcm_row["median_vae_seconds"]),
        "transfer": float(rcm_row["median_cpu_transfer_seconds"]),
        "serialization": float(rcm_row["median_serialization_seconds"]),
    }

    attention_manifest_path = (
        repo_root
        / "worldfoundry_hybrid_residual/results/wan_rcm_onpolicy_attention_exp054_20260901"
        / "s0-smoke/manifest.json"
    )
    with attention_manifest_path.open(encoding="utf-8") as handle:
        attention_manifest = json.load(handle)
    attention_speedup = float(
        attention_manifest["result"]["benchmark"]["attention_speedup"]
    )

    attention_config_path = (
        repo_root
        / "worldfoundry_hybrid_residual/configs/wan_rcm_onpolicy_attention_exp054_v1.json"
    )
    with attention_config_path.open(encoding="utf-8") as handle:
        attention_config = json.load(handle)
    self_attention_share = float(
        attention_config["materiality"]["historical_self_attention_share"]
    )
    return runtime, attention_speedup, self_attention_share


def build_component_rows(
    runtime: dict[str, float], attention_speedup: float, self_attention_share: float
) -> list[dict[str, object]]:
    total = runtime["request"]
    components = [
        ("Exact VAE", runtime["vae"], math.nan, "same-graph CUDA replay; unmeasured"),
        (
            "Transfer + serialization",
            runtime["transfer"] + runtime["serialization"],
            math.nan,
            "same request is sequential; codec exactness constrains changes",
        ),
        (
            "Serialization",
            runtime["serialization"],
            math.nan,
            "CPU libx264 path; bitstream contract matters",
        ),
        (
            "Trainable low-precision attention",
            runtime["denoiser"] * self_attention_share,
            attention_speedup,
            "measured Sage local speed; train-free safe coverage was 0/120",
        ),
        (
            "CPU transfer",
            runtime["transfer"],
            math.nan,
            "too small as a standalone gate",
        ),
    ]
    rows: list[dict[str, object]] = []
    for priority, (name, seconds, measured_speedup, note) in enumerate(
        components, start=1
    ):
        measured_request_speedup = (
            projected_request_speedup(total, seconds, measured_speedup)
            if math.isfinite(measured_speedup)
            else math.nan
        )
        rows.append(
            {
                "priority_order": priority,
                "candidate": name,
                "eligible_seconds": seconds,
                "request_share": seconds / total,
                "zero_cost_request_ceiling": total / (total - seconds),
                "measured_related_local_speedup": measured_speedup,
                "measured_related_request_speedup": measured_request_speedup,
                "evidence_boundary": note,
            }
        )
    return rows


def build_required_rows(
    runtime: dict[str, float], self_attention_share: float
) -> list[dict[str, object]]:
    total = runtime["request"]
    components = {
        "Exact VAE": runtime["vae"],
        "Transfer + serialization": runtime["transfer"] + runtime["serialization"],
        "Serialization": runtime["serialization"],
        "Self-attention": runtime["denoiser"] * self_attention_share,
    }
    rows: list[dict[str, object]] = []
    for name, seconds in components.items():
        for request_target in REQUEST_TARGETS:
            rows.append(
                {
                    "candidate": name,
                    "eligible_seconds": seconds,
                    "request_speedup_target": request_target,
                    "required_local_speedup": required_local_speedup(
                        total, seconds, request_target
                    ),
                    "feasible_if_component_zero": total / (total - seconds)
                    >= request_target,
                }
            )
    return rows


def plot_decision_surface(
    component_rows: list[dict[str, object]],
    required_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    names = [str(row["candidate"]) for row in component_rows]
    shares = [100.0 * float(row["request_share"]) for row in component_rows]
    colors = ["#0072B2", "#009E73", "#56B4E9", "#D55E00", "#999999"]
    y_positions = list(range(len(names)))
    axes[0].barh(y_positions, shares, color=colors)
    axes[0].set_yticks(y_positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Eligible share of resident rCM request (%)")
    axes[0].set_xlim(0, 50)
    for y, share in zip(y_positions, shares, strict=True):
        axes[0].text(share + 0.8, y, f"{share:.1f}%", va="center")

    line_colors = {
        "Exact VAE": "#0072B2",
        "Transfer + serialization": "#009E73",
        "Serialization": "#56B4E9",
        "Self-attention": "#D55E00",
    }
    for name, color in line_colors.items():
        selected = [row for row in required_rows if row["candidate"] == name]
        x_values = [float(row["request_speedup_target"]) for row in selected]
        y_values = [
            float(row["required_local_speedup"])
            if bool(row["feasible_if_component_zero"])
            else math.nan
            for row in selected
        ]
        axes[1].plot(
            x_values, y_values, marker="o", linewidth=2, color=color, label=name
        )
    axes[1].axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Target resident-request speedup")
    axes[1].set_ylabel("Required local component speedup")
    axes[1].set_xticks(REQUEST_TARGETS)
    axes[1].set_ylim(0.9, 15.5)
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(
            axis="x" if axis is axes[0] else "y",
            color="#dddddd",
            linewidth=0.7,
            alpha=0.7,
        )

    fig.tight_layout()
    fig.savefig(
        output_dir / "wan_rcm_successor_decision_surface.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "wan_rcm_successor_decision_surface.pdf", bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime, attention_speedup, self_attention_share = load_inputs(repo_root)
    component_rows = build_component_rows(
        runtime, attention_speedup, self_attention_share
    )
    required_rows = build_required_rows(runtime, self_attention_share)
    write_csv(output_dir / "candidate_frontier.csv", component_rows)
    write_csv(output_dir / "required_component_speedup.csv", required_rows)
    plot_decision_surface(component_rows, required_rows, output_dir)

    attention_seconds = runtime["denoiser"] * self_attention_share
    summary = {
        "baseline_request_seconds": runtime["request"],
        "baseline_components_seconds": runtime,
        "historical_self_attention_share_of_denoiser": self_attention_share,
        "eligible_self_attention_seconds": attention_seconds,
        "measured_sage_local_speedup": attention_speedup,
        "measured_sage_full_coverage_request_speedup_ceiling": projected_request_speedup(
            runtime["request"], attention_seconds, attention_speedup
        ),
        "selected_successor": "exact_full_f81_vae_cuda_graph_replay",
        "selection_reason": (
            "VAE is the largest remaining exact component; CUDA Graph replay preserves the registered temporal "
            "schedule and cache contents while testing launch-bound redundancy."
        ),
        "decision_boundary": (
            "This is an Amdahl and source-readiness selection, not a measured VAE speed or exactness result."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
