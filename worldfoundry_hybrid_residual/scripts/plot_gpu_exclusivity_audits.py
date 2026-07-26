#!/usr/bin/env python3
"""Plot foreign-process overlap from one or more GPU exclusivity audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, object]] = []
    for path in args.audit:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": payload["stage"],
                "timing_valid": bool(payload["timing_valid"]),
                "snapshots_total": int(payload["snapshots_total"]),
                "snapshots_with_foreign_process": int(
                    payload["snapshots_with_foreign_process"]
                ),
                "foreign_snapshot_fraction": float(
                    payload["foreign_snapshot_fraction"]
                ),
                "estimated_foreign_overlap_seconds": payload.get(
                    "estimated_foreign_overlap_seconds"
                ),
                "foreign_pid_count": len(payload.get("foreign_processes", [])),
                "source": str(path.resolve()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gpu_exclusivity_audit_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["stage"]).replace("_retro", "") for row in rows]
    fractions = [100.0 * float(row["foreign_snapshot_fraction"]) for row in rows]
    colors = ["#b3261e" if not row["timing_valid"] else "#1b7f5c" for row in rows]
    figure, axis = plt.subplots(figsize=(9.2, max(3.2, 1.15 * len(rows) + 1.5)))
    bars = axis.barh(labels, fractions, color=colors, height=0.58)
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Telemetry snapshots with a foreign process (%)")
    axis.set_title("H200 timing admissibility audit")
    axis.grid(axis="x", alpha=0.22)
    axis.invert_yaxis()
    for bar, row, fraction in zip(bars, rows, fractions):
        overlap = row["estimated_foreign_overlap_seconds"]
        seconds = "unknown" if overlap is None else f"~{float(overlap):.0f}s"
        axis.text(
            min(fraction + 1.5, 96.0),
            bar.get_y() + bar.get_height() / 2,
            f"{fraction:.1f}% ({seconds}); timing INVALID",
            va="center",
            ha="left" if fraction < 78.0 else "right",
            color="#202124",
            fontsize=9,
        )
    figure.text(
        0.01,
        0.01,
        "Correctness artifacts remain usable; latency and speedup are excluded unless timing_valid=true.",
        fontsize=8.5,
        color="#4a4a4a",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"gpu_exclusivity_audit.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps({"rows": rows, "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
