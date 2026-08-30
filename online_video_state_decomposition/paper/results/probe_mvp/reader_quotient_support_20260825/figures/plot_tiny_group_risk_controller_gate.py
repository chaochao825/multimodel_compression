from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "tiny_group_risk_controller_prospective_v1"
)
OUT = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(RUN / "training_history.csv")

    selector_rows = []
    for role in ("validation", "prospective"):
        diagnostics = summary[f"{role}_selector_diagnostics"]
        for method, values in diagnostics.items():
            selector_rows.append(
                {
                    "role": role,
                    "method": method,
                    "topk_recall": float(values["mean_topk_recall"]),
                    "risk_mass_capture": float(values["mean_risk_mass_capture"]),
                }
            )
    selectors = pd.DataFrame(selector_rows)

    reader_rows = []
    for method, values in summary["prospective_reader"].items():
        reader_rows.append(
            {
                "method": method,
                "agreement": float(values["agreement"]),
                "candidate_kl_mean": float(values["candidate_kl_mean"]),
                "harmful_count": int(values["harmful_count"]),
            }
        )
    readers = pd.DataFrame(reader_rows)

    progressive = summary["prospective_progressive"]
    gate_rows = pd.DataFrame(
        [
            {
                "metric": "Delivered agreement",
                "actual": float(progressive["delivered_agreement"]),
                "gate": 0.98,
                "direction": "minimum",
            },
            {
                "metric": "Fallback rate",
                "actual": float(progressive["fallback_rate"]),
                "gate": 0.15,
                "direction": "maximum",
            },
            {
                "metric": "Token retention",
                "actual": float(progressive["effective_token_retention"]),
                "gate": 0.53,
                "direction": "maximum",
            },
        ]
    )

    selectors.to_csv(OUT / "tiny_group_risk_controller_selector_metrics.csv", index=False)
    readers.to_csv(OUT / "tiny_group_risk_controller_reader_metrics.csv", index=False)
    gate_rows.to_csv(OUT / "tiny_group_risk_controller_gate_metrics.csv", index=False)

    labels = {
        "query_cosine": "Question cosine",
        "residual_energy": "Residual energy",
        "tiny_controller": "Tiny controller",
    }
    colors = {
        "query_cosine": "#F2A541",
        "residual_energy": "#4C78A8",
        "tiny_controller": "#2A9D8F",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        history["epoch"],
        100 * history["validation_topk_recall"],
        color="#2A9D8F",
        linewidth=1.8,
        label="Validation top-k recall",
    )
    ax.plot(
        history["epoch"],
        100 * history["validation_risk_mass_capture"],
        color="#7A5195",
        linewidth=1.5,
        label="Validation risk mass",
    )
    ax.axhline(25, color="#777777", linestyle="--", linewidth=1, label="Random recall")
    ax.axvline(
        int(summary["best_epoch"]),
        color="#333333",
        linestyle=":",
        linewidth=1,
        label=f"Selected epoch {summary['best_epoch']}",
    )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(20, 38)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    methods = ["query_cosine", "residual_energy", "tiny_controller"]
    x = np.arange(len(methods))
    width = 0.34
    for offset, role, color in (
        (-width / 2, "validation", "#9ECAE1"),
        (width / 2, "prospective", "#2A9D8F"),
    ):
        values = [
            100
            * float(
                selectors[
                    (selectors["role"] == role) & (selectors["method"] == method)
                ]["topk_recall"].iloc[0]
            )
            for method in methods
        ]
        ax.bar(x + offset, values, width, color=color, label=role.capitalize())
    ax.axhline(25, color="#777777", linestyle="--", linewidth=1, label="Random")
    ax.set_xticks(x, [labels[method] for method in methods], rotation=10)
    ax.set_ylabel("Top-98 risk-group recall (%)")
    ax.set_ylim(20, 35)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    annotation_offsets = {
        "query_cosine": (-4, -14),
        "residual_energy": (5, 5),
        "tiny_controller": (5, 5),
    }
    for _, row in readers.iterrows():
        method = str(row["method"])
        ax.scatter(
            row["candidate_kl_mean"],
            100 * row["agreement"],
            s=76,
            color=colors[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            labels[method],
            (row["candidate_kl_mean"], 100 * row["agreement"]),
            xytext=annotation_offsets[method],
            textcoords="offset points",
            fontsize=8,
            ha="right" if method == "query_cosine" else "left",
        )
    ax.axhline(98, color="#333333", linestyle="--", linewidth=1, label="98% gate")
    ax.set_xlabel("Prospective mean candidate KL")
    ax.set_ylabel("Raw reader agreement (%)")
    ax.set_xlim(0.035, 0.072)
    ax.set_ylim(78, 100)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    x = np.arange(len(gate_rows))
    ax.bar(
        x - 0.18,
        100 * gate_rows["actual"],
        0.36,
        color="#C44E52",
        label="Prospective actual",
    )
    ax.bar(
        x + 0.18,
        100 * gate_rows["gate"],
        0.36,
        color="#BFC5CA",
        label="Registered gate",
    )
    ax.set_xticks(x, ["Agreement\n(min)", "Fallback\n(max)", "Retention\n(max)"])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontweight="bold")

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"tiny_group_risk_controller_gate.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    svg_path = OUT / "tiny_group_risk_controller_gate.svg"
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
