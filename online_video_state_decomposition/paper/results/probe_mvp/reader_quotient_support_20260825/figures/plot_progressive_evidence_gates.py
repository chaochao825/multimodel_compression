from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "onevision_reader_quotient_stage_a_20260830"
OUT = Path(__file__).resolve().parent


def load_summary(name: str) -> dict[str, object]:
    return json.loads((ANALYSIS / name / "summary.json").read_text(encoding="utf-8"))


def method_row(
    label: str,
    family: str,
    values: dict[str, float | int],
    *,
    fallback_rate: float = 0.0,
) -> dict[str, object]:
    return {
        "label": label,
        "family": family,
        "token_retention": float(values["token_retention"]),
        "candidate_kl_mean": float(values["candidate_kl_mean"]),
        "agreement": float(values["agreement"]),
        "harmful_count": int(values["harmful_count"]),
        "fallback_rate": fallback_rate,
    }


def main() -> None:
    one = load_summary("progressive_evidence_capacity_v1")
    pair = load_summary("progressive_evidence_pair_capacity_v1")
    risk = load_summary("risk_guided_exact_groups_capacity_v1")
    transfer = load_summary("query_group_margin_fallback_transfer_v1")

    rows = [
        method_row(
            "Quotient pool",
            "coarse",
            one["method_summaries"]["quotient_pool49"],
        ),
        method_row(
            "Oracle frame-1",
            "frame oracle",
            one["method_summaries"]["oracle_frame1"],
        ),
        method_row(
            "Oracle frame-2",
            "frame oracle",
            pair["method_summaries"]["oracle_frame2"],
        ),
        method_row(
            "Residual groups",
            "group proxy",
            risk["method_summaries"]["residual_energy_groups"],
        ),
        method_row(
            "Query groups",
            "group proxy",
            risk["method_summaries"]["query_score_groups"],
        ),
        method_row(
            "Gradient-risk groups",
            "risk oracle",
            risk["method_summaries"]["target_gradient_risk_groups"],
        ),
        {
            "label": "Query+fallback eval",
            "family": "transfer",
            "token_retention": float(
                transfer["progressive_evaluation"]["effective_token_retention"]
            ),
            "candidate_kl_mean": float(
                transfer["progressive_evaluation"]["delivered_candidate_kl_mean"]
            ),
            "agreement": float(
                transfer["progressive_evaluation"]["delivered_agreement"]
            ),
            "harmful_count": int(
                transfer["progressive_evaluation"]["remaining_harmful_count"]
            ),
            "fallback_rate": float(
                transfer["progressive_evaluation"]["fallback_rate"]
            ),
        },
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "progressive_evidence_gate_metrics.csv", index=False)

    groups = pd.read_csv(ANALYSIS / "risk_guided_exact_groups_capacity_v1" / "group_metrics.csv")
    residual_overlap = []
    query_overlap = []
    for _, frame in groups.groupby("sample_id", sort=False):
        risk_count = frame["selected_by_risk"].sum()
        residual_overlap.append(
            (frame["selected_by_residual"] * frame["selected_by_risk"]).sum()
            / risk_count
        )
        query_overlap.append(
            (frame["selected_by_query"] * frame["selected_by_risk"]).sum()
            / risk_count
        )
    correlations = pd.DataFrame(
        {
            "proxy": ["Residual energy", "Question cosine"],
            "spearman_to_target_risk": [
                groups["residual_energy"].rank().corr(
                    groups["target_gradient_risk"].rank()
                ),
                groups["query_score"].rank().corr(
                    groups["target_gradient_risk"].rank()
                ),
            ],
            "mean_topk_overlap": [np.mean(residual_overlap), np.mean(query_overlap)],
        }
    )
    correlations.to_csv(OUT / "group_proxy_correlations.csv", index=False)

    colors = {
        "coarse": "#7F8C8D",
        "frame oracle": "#4C78A8",
        "group proxy": "#F2A541",
        "risk oracle": "#2A9D8F",
        "transfer": "#C44E52",
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
    for _, row in metrics.iterrows():
        ax.scatter(
            100 * row["token_retention"],
            row["candidate_kl_mean"],
            s=62,
            color=colors[row["family"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (100 * row["token_retention"], row["candidate_kl_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Visual-token retention (%)")
    ax.set_ylabel("Mean candidate KL")
    ax.grid(alpha=0.2)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    for _, row in metrics.iterrows():
        ax.scatter(
            100 * row["token_retention"],
            100 * row["agreement"],
            s=62,
            color=colors[row["family"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (100 * row["token_retention"], 100 * row["agreement"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(98, color="#333333", linestyle="--", linewidth=1, label="98% gate")
    ax.set_xlabel("Visual-token retention (%)")
    ax.set_ylabel("Full-reader agreement (%)")
    ax.set_ylim(50, 102)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    x = np.arange(len(correlations))
    width = 0.36
    ax.bar(
        x - width / 2,
        correlations["spearman_to_target_risk"],
        width,
        color="#4C78A8",
        label="Spearman to target risk",
    )
    ax.bar(
        x + width / 2,
        correlations["mean_topk_overlap"],
        width,
        color="#F2A541",
        label="Top-k overlap",
    )
    ax.set_xticks(x, correlations["proxy"])
    ax.set_ylabel("Association")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylim(-0.05, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    labels = ["Fit", "Fresh eval"]
    raw = [
        transfer["raw_fit"]["agreement"],
        transfer["raw_evaluation"]["agreement"],
    ]
    delivered = [
        transfer["progressive_fit"]["delivered_agreement"],
        transfer["progressive_evaluation"]["delivered_agreement"],
    ]
    fallback = [
        transfer["progressive_fit"]["fallback_rate"],
        transfer["progressive_evaluation"]["fallback_rate"],
    ]
    x = np.arange(2)
    ax.bar(x - 0.25, np.asarray(raw) * 100, 0.25, color="#F2A541", label="Raw agreement")
    ax.bar(x, np.asarray(delivered) * 100, 0.25, color="#2A9D8F", label="Delivered agreement")
    ax.bar(x + 0.25, np.asarray(fallback) * 100, 0.25, color="#C44E52", label="Fallback")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontweight="bold")

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"progressive_evidence_gates.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
