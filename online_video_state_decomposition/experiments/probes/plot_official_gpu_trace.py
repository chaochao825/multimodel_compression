from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_FIELDS = (
    "monotonic_seconds",
    "gpu_total_memory_mib",
    "gpu_utilization_percent",
    "process_memory_mib",
)
NORMALIZED_FIELDS = (
    "elapsed_seconds",
    "gpu_memory_used_mib",
    "process_memory_mib",
    "gpu_utilization_percent",
)
COLORS = ("#0072B2", "#D55E00", "#009E73")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and plot an audited official-run GPU trace"
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="gpu_trace")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric: {value!r}") from error
    if not math.isfinite(observed) or observed < minimum:
        raise ValueError(f"invalid {label}: {value!r}")
    return observed


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_samples(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(
                f"unexpected GPU trace columns: {reader.fieldnames}; "
                f"expected {EXPECTED_FIELDS}"
            )
        raw_rows = list(reader)
    if len(raw_rows) < 2:
        raise ValueError("GPU trace must contain at least two samples")

    rows: list[dict[str, float]] = []
    previous_timestamp: float | None = None
    first_timestamp: float | None = None
    for index, raw in enumerate(raw_rows):
        timestamp = _number(raw["monotonic_seconds"], label=f"row {index} timestamp")
        total_used = _number(
            raw["gpu_total_memory_mib"], label=f"row {index} total GPU memory"
        )
        process_used = _number(
            raw["process_memory_mib"], label=f"row {index} process memory"
        )
        utilization = _number(
            raw["gpu_utilization_percent"], label=f"row {index} utilization"
        )
        if utilization > 100.0:
            raise ValueError(f"row {index} utilization exceeds 100%")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("GPU trace timestamps must be strictly increasing")
        if first_timestamp is None:
            first_timestamp = timestamp
        rows.append(
            {
                "elapsed_seconds": timestamp - first_timestamp,
                "gpu_memory_used_mib": total_used,
                "process_memory_mib": process_used,
                "gpu_utilization_percent": utilization,
            }
        )
        previous_timestamp = timestamp
    return rows


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("result JSON must describe a complete run")
    record = payload.get("run_record")
    if not isinstance(record, dict):
        raise ValueError("result JSON lacks an audited run record")
    if record.get("run_fingerprint") != payload.get("run_fingerprint"):
        raise ValueError("result and run-record fingerprints differ")
    monitor = record.get("gpu_monitor")
    if not isinstance(monitor, dict):
        raise ValueError("result JSON lacks GPU monitor metrics")
    return payload


def _validate_result(rows: list[dict[str, float]], payload: dict[str, Any]) -> None:
    monitor = payload["run_record"]["gpu_monitor"]
    expected = {
        "sample_count": len(rows),
        "gpu_peak_process_mib_sampled": int(
            max(row["process_memory_mib"] for row in rows)
        ),
        "gpu_peak_total_mib_sampled": int(
            max(row["gpu_memory_used_mib"] for row in rows)
        ),
        "gpu_peak_utilization_percent_sampled": int(
            max(row["gpu_utilization_percent"] for row in rows)
        ),
    }
    for field, observed in expected.items():
        if monitor.get(field) != observed:
            raise ValueError(
                f"GPU trace/result mismatch for {field}: "
                f"trace={observed}, result={monitor.get(field)!r}"
            )


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float]], stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed = [row["elapsed_seconds"] for row in rows]
    total_gib = [row["gpu_memory_used_mib"] / 1024.0 for row in rows]
    process_gib = [row["process_memory_mib"] / 1024.0 for row in rows]
    utilization = [row["gpu_utilization_percent"] for row in rows]

    fig, (memory_axis, util_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.5),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (1.45, 1.0), "hspace": 0.08},
    )
    memory_axis.plot(
        elapsed,
        total_gib,
        color=COLORS[0],
        linewidth=1.8,
        label="Total GPU memory used",
    )
    memory_axis.plot(
        elapsed,
        process_gib,
        color=COLORS[1],
        linewidth=1.5,
        linestyle="--",
        label="Evaluator process",
    )
    memory_axis.set_ylabel("GPU memory used (GiB)")
    memory_axis.legend(frameon=False, loc="upper left", ncol=2)

    util_axis.plot(elapsed, utilization, color=COLORS[2], linewidth=1.2)
    util_axis.fill_between(elapsed, utilization, color=COLORS[2], alpha=0.16)
    util_axis.set_ylim(0.0, 105.0)
    util_axis.set_ylabel("GPU utilization (%)")
    util_axis.set_xlabel("Elapsed wall time (s)")

    for axis in (memory_axis, util_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def analyze_trace(
    *, samples_path: Path, result_path: Path | None, out_dir: Path, stem: str
) -> dict[str, Any]:
    rows = load_samples(samples_path.resolve())
    result = _load_result(result_path.resolve()) if result_path is not None else None
    if result is not None:
        _validate_result(rows, result)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = out_dir / f"{stem}.csv"
    _write_csv(normalized_csv, rows)
    plots = _plot(rows, out_dir / stem)

    cadence = [
        rows[index]["elapsed_seconds"] - rows[index - 1]["elapsed_seconds"]
        for index in range(1, len(rows))
    ]
    process_values = [row["process_memory_mib"] for row in rows]
    total_values = [row["gpu_memory_used_mib"] for row in rows]
    utilization = [row["gpu_utilization_percent"] for row in rows]
    summary = {
        "format_version": 1,
        "source_samples": str(samples_path),
        "source_samples_sha256": _sha256(samples_path),
        "source_result": str(result_path) if result_path is not None else None,
        "source_result_sha256": _sha256(result_path) if result_path is not None else None,
        "run_fingerprint": result.get("run_fingerprint") if result is not None else None,
        "sample_count": len(rows),
        "sampled_duration_seconds": rows[-1]["elapsed_seconds"],
        "median_sample_interval_seconds": statistics.median(cadence),
        "peak_gpu_memory_used_mib": max(total_values),
        "peak_process_memory_mib": max(process_values),
        "p95_process_memory_mib": _percentile(process_values, 0.95),
        "peak_gpu_utilization_percent": max(utilization),
        "median_gpu_utilization_percent": statistics.median(utilization),
        "active_sample_fraction_ge_20_percent": sum(value >= 20.0 for value in utilization)
        / len(utilization),
        "normalized_csv": str(normalized_csv),
        "plots": plots,
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = analyze_trace(
        samples_path=args.samples,
        result_path=args.result,
        out_dir=args.out_dir,
        stem=args.stem,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
