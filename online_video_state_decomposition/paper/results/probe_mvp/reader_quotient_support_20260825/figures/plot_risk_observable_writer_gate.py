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
    / "risk_observable_writer_prospective_v2"
)
OUT = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(RUN / "training_history.csv")
    selector_rows = []
    for role in ("validation", "prospective"):
        for method, values in summary[f"{role}_selector_diagnostics"].items():
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
    selectors.to_csv(OUT / "risk_observable_writer_selector_metrics.csv", index=False)
    readers.to_csv(OUT / "risk_observable_writer_reader_metrics.csv", index=False)

    labels = {
        "residual_energy": "Residual energy",
        "query_cosine": "Question cosine",
        "fixed_controller": "Fixed writer + controller",
        "writer_dot": "Learned writer + dot",
        "joint_writer_controller": "Joint writer-controller",
        "target_gradient_risk": "Gradient-risk oracle",
    }
    colors = {
        "residual_energy": "#4C78A8",
        "query_cosine": "#F2A541",
        "fixed_controller": "#7A5195",
        "writer_dot": "#59A14F",
        "joint_writer_controller": "#2A9D8F",
        "target_gradient_risk": "#C44E52",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)

    ax = axes[0, 0]
    for method in ("fixed_controller", "writer_dot", "joint_writer_controller"):
        frame = history[history["method"] == method]
        ax.plot(
            frame["epoch"],
            100 * frame["validation_topk_recall"],
            color=colors[method],
            linewidth=1.6,
            label=labels[method],
        )
    ax.axhline(25, color="#777777", linestyle="--", linewidth=1, label="Random")
    ax.axhline(45, color="#333333", linestyle=":", linewidth=1, label="45% gate")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation top-98 recall (%)")
    ax.set_ylim(20, 47)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    methods = [
        "query_cosine",
        "residual_energy",
        "fixed_controller",
        "writer_dot",
        "joint_writer_controller",
    ]
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
    ax.axhline(45, color="#333333", linestyle=":", linewidth=1, label="45% gate")
    ax.set_xticks(x, [labels[method] for method in methods], rotation=22, ha="right")
    ax.set_ylabel("Top-98 risk-group recall (%)")
    ax.set_ylim(20, 48)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    prospective = selectors[selectors["role"] == "prospective"].set_index("method")
    methods_with_oracle = methods + ["target_gradient_risk"]
    x = np.arange(len(methods_with_oracle))
    ax.bar(
        x,
        [100 * prospective.loc[method, "risk_mass_capture"] for method in methods_with_oracle],
        color=[colors[method] for method in methods_with_oracle],
    )
    ax.axhline(50, color="#333333", linestyle=":", linewidth=1, label="50% gate")
    ax.set_xticks(
        x,
        [labels[method] for method in methods_with_oracle],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel("Prospective teacher-risk mass (%)")
    ax.set_ylim(0, 88)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    offsets = {
        "residual_energy": (5, -14),
        "query_cosine": (5, 7),
        "fixed_controller": (-5, 7),
        "writer_dot": (-5, -14),
        "joint_writer_controller": (5, -14),
        "target_gradient_risk": (5, 5),
    }
    right_aligned = {"fixed_controller", "writer_dot"}
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
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if method in right_aligned else "left",
        )
    ax.axhline(100 * 22 / 24, color="#333333", linestyle=":", linewidth=1)
    ax.set_xlabel("Prospective mean candidate KL")
    ax.set_ylabel("Full-reader agreement (%)")
    ax.set_xlim(0.025, 0.073)
    ax.set_ylim(66, 95)
    ax.grid(alpha=0.2)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontweight="bold")

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"risk_observable_writer_gate.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    svg_path = OUT / "risk_observable_writer_gate.svg"
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
