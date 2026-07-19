from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_FIELDS = (
    "method",
    "variant",
    "role",
    "benchmark",
    "scope",
    "status",
    "evidence_tier",
    "run_fingerprint",
    "source_path",
    "source_sha256",
    "quality_accuracy",
    "quality_correct",
    "quality_scored",
    "quality_expected",
    "quality_coverage",
    "whole_run_seconds",
    "whole_run_semantics",
    "tail_latency_available",
    "stage_latency_available",
    "peak_memory_value",
    "peak_memory_unit",
    "peak_memory_semantics",
    "notes",
)

QUALITY_FIELDS = (
    "method",
    "variant",
    "scope",
    "accuracy",
    "correct",
    "scored",
    "expected",
    "coverage",
    "evidence_tier",
    "source_path",
)

LATENCY_FIELDS = (
    "method",
    "mode",
    "stage",
    "count",
    "min_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    "max_ms",
    "source_path",
)

COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9")
METHOD_STYLES = {
    "CausalMem": (COLORS[0], "//"),
    "OASIS": (COLORS[1], "xx"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate audited model-level streaming baseline results"
    )
    parser.add_argument("--causalmem-metrics", type=Path, action="append", default=[])
    parser.add_argument("--stc-result", type=Path, action="append", default=[])
    parser.add_argument("--oasis-result", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or (minimum is not None and observed < minimum):
        raise ValueError(f"invalid {label}: {value}")
    return observed


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {label}: {value}")
    return value


def _probability(value: Any, *, label: str) -> float:
    observed = _finite_number(value, label=label)
    if not 0.0 <= observed <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]: {value}")
    return observed


def _scope(expected_questions: int) -> str:
    if expected_questions == 250:
        return "formal_50x5"
    return f"smoke_{expected_questions}q"


def _gpu_peak(monitor: Any) -> int | None:
    if not isinstance(monitor, dict):
        return None
    value = monitor.get("gpu_peak_process_mib_sampled")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _quality_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": run["method"],
        "variant": run["variant"],
        "scope": run["scope"],
        "accuracy": run["quality_accuracy"],
        "correct": run["quality_correct"],
        "scored": run["quality_scored"],
        "expected": run["quality_expected"],
        "coverage": run["quality_coverage"],
        "evidence_tier": run["evidence_tier"],
        "source_path": run["source_path"],
    }


def parse_causalmem(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("format_version") != 2 or payload.get("method") != "causal_mem":
        raise ValueError(f"not an audited CausalMem metrics file: {path}")
    if payload.get("success") is not True or payload.get("returncode") != 0:
        raise ValueError(f"CausalMem run is not successful: {path}")
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        raise ValueError(f"CausalMem quality object is missing: {path}")
    integrity_fields = (
        "parse_errors",
        "invalid_records",
        "duplicate_ids",
        "unexpected_ids",
        "missing_question_ids",
    )
    for field in integrity_fields:
        value = quality.get(field)
        if value not in (0, []):
            raise ValueError(f"CausalMem quality integrity failure in {field}: {value}")
    scored = _integer(
        quality.get("completed_questions"), label="CausalMem completed_questions", minimum=1
    )
    expected = _integer(
        quality.get("expected_questions"), label="CausalMem expected_questions", minimum=1
    )
    correct = _integer(quality.get("correct"), label="CausalMem correct")
    if scored != expected or correct > scored:
        raise ValueError("CausalMem result must be complete and internally consistent")
    accuracy = _probability(quality.get("accuracy"), label="CausalMem accuracy")
    if not math.isclose(accuracy, correct / scored, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("CausalMem accuracy is inconsistent")
    latency_scope = payload.get("latency_scope")
    if not isinstance(latency_scope, dict) or latency_scope.get(
        "per_sample_p50_p95_p99_available"
    ) is not False:
        raise ValueError("CausalMem latency scope must explicitly reject tail latency")
    monitor = payload.get("gpu_monitor")
    peak_memory = _gpu_peak(monitor)
    wall_seconds = _finite_number(
        payload.get("wall_seconds"), label="CausalMem wall_seconds", minimum=0.0
    )
    source = str(path.resolve())
    run = {
        "method": "CausalMem",
        "variant": "causal_mem",
        "role": "quality",
        "benchmark": "StreamingBench RTU",
        "scope": _scope(expected),
        "status": "complete",
        "evidence_tier": str(payload.get("evidence_tier", "")),
        "run_fingerprint": str(payload.get("run_fingerprint", "")),
        "source_path": source,
        "source_sha256": _sha256(path),
        "quality_accuracy": accuracy,
        "quality_correct": correct,
        "quality_scored": scored,
        "quality_expected": expected,
        "quality_coverage": scored / expected,
        "whole_run_seconds": wall_seconds,
        "whole_run_semantics": "evaluator process wall time; no per-sample tails",
        "tail_latency_available": False,
        "stage_latency_available": False,
        "peak_memory_value": peak_memory,
        "peak_memory_unit": "MiB" if peak_memory is not None else None,
        "peak_memory_semantics": (
            "sampled evaluator-process GPU memory" if peak_memory is not None else None
        ),
        "notes": "Not comparable to STC stage latency or OASIS request latency.",
    }
    return run, _quality_row(run)


def _validate_stage(stage: Any, *, label: str, source_path: str, mode: str) -> dict[str, Any]:
    if not isinstance(stage, dict):
        raise ValueError(f"missing STC stage summary: {label}")
    count = _integer(stage.get("count"), label=f"{label}.count", minimum=1)
    values = {
        key: _finite_number(stage.get(key), label=f"{label}.{key}", minimum=0.0)
        for key in ("min", "p50", "p95", "p99", "mean", "std", "max")
    }
    if not (
        values["min"]
        <= values["p50"]
        <= values["p95"]
        <= values["p99"]
        <= values["max"]
    ):
        raise ValueError(f"STC quantiles are not ordered for {label}")
    return {
        "method": "STC ReKV",
        "mode": mode,
        "stage": label,
        "count": count,
        "min_ms": values["min"],
        "p50_ms": values["p50"],
        "p95_ms": values["p95"],
        "p99_ms": values["p99"],
        "mean_ms": values["mean"],
        "std_ms": values["std"],
        "max_ms": values["max"],
        "source_path": source_path,
    }


def parse_stc(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    mode = payload.get("mode")
    if payload.get("format_version") != 1 or payload.get("status") not in {
        "complete",
        "recovered_from_valid_official_raw",
    }:
        raise ValueError(f"STC result is not complete: {path}")
    if mode not in {"rekv", "stc"}:
        raise ValueError(f"unexpected STC mode: {mode}")
    derived = payload.get("derived")
    if not isinstance(derived, dict):
        raise ValueError(f"STC derived metrics are missing: {path}")
    source = str(path.resolve())
    stage_rows = [
        _validate_stage(derived.get(stage), label=stage, source_path=source, mode=mode)
        for stage in (
            "vit_encode_ms",
            "llm_prefill_ms",
            "instrumented_stage_sum_ms",
        )
    ]
    peak_memory = _finite_number(
        derived.get("peak_mem_gb_official"),
        label="STC peak_mem_gb_official",
        minimum=0.0,
    )
    run_record_path = path.parent / "run_record.json"
    wall_seconds = None
    gpu_peak = None
    if run_record_path.is_file():
        record = _load_json(run_record_path)
        if record.get("run_fingerprint") != payload.get("run_fingerprint"):
            raise ValueError("STC run record fingerprint mismatch")
        wall_seconds = _finite_number(
            record.get("elapsed_wall_seconds"),
            label="STC elapsed_wall_seconds",
            minimum=0.0,
        )
        gpu_peak = _gpu_peak(record.get("gpu_monitor"))
    run = {
        "method": "STC ReKV",
        "variant": str(mode),
        "role": "stage_latency",
        "benchmark": "STC official LLaVA-OneVision ReKV benchmark",
        "scope": "official_model_stage",
        "status": "complete",
        "evidence_tier": "official_model_stage_latency",
        "run_fingerprint": str(payload.get("run_fingerprint", "")),
        "source_path": source,
        "source_sha256": _sha256(path),
        "quality_accuracy": None,
        "quality_correct": None,
        "quality_scored": None,
        "quality_expected": None,
        "quality_coverage": None,
        "whole_run_seconds": wall_seconds,
        "whole_run_semantics": (
            "benchmark process wall time; not TTFT or request latency"
            if wall_seconds is not None
            else None
        ),
        "tail_latency_available": False,
        "stage_latency_available": True,
        "peak_memory_value": peak_memory,
        "peak_memory_unit": "GB (official field)",
        "peak_memory_semantics": (
            "official benchmark peak_mem_gb; sampled process peak is "
            f"{gpu_peak} MiB" if gpu_peak is not None else "official benchmark peak_mem_gb"
        ),
        "notes": (
            "Stage P50/P95/P99 cover ViT encode plus visual-token prefill only; "
            "not request-tail latency, TTFT, decode, or quality."
        ),
    }
    return run, stage_rows


def parse_oasis(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("format_version") != 1 or payload.get("status") not in {
        "complete",
        "recovered_from_valid_official_output",
    }:
        raise ValueError(f"OASIS result is not complete: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("complete") is not True:
        raise ValueError(f"OASIS metrics are incomplete: {path}")
    if metrics.get("errors") not in (None, []):
        raise ValueError(f"OASIS result contains failed questions: {metrics.get('errors')}")
    scored = _integer(metrics.get("scored_questions"), label="OASIS scored_questions", minimum=1)
    expected = _integer(
        metrics.get("expected_questions"), label="OASIS expected_questions", minimum=1
    )
    correct = _integer(metrics.get("correct"), label="OASIS correct")
    if scored != expected or correct > scored:
        raise ValueError("OASIS result must be complete and fully scored")
    accuracy = _probability(metrics.get("accuracy"), label="OASIS accuracy")
    coverage = _probability(metrics.get("scored_coverage"), label="OASIS coverage")
    if not math.isclose(accuracy, correct / scored, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("OASIS accuracy is inconsistent")
    if not math.isclose(coverage, scored / expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("OASIS coverage is inconsistent")
    run_record = payload.get("run_record")
    if not isinstance(run_record, dict):
        raise ValueError("OASIS result lacks its audited run record")
    if run_record.get("run_fingerprint") != payload.get("run_fingerprint"):
        raise ValueError("OASIS run record fingerprint mismatch")
    wall_seconds = _finite_number(
        run_record.get("elapsed_wall_seconds"),
        label="OASIS elapsed_wall_seconds",
        minimum=0.0,
    )
    peak_memory = _gpu_peak(run_record.get("gpu_monitor"))
    source = str(path.resolve())
    run = {
        "method": "OASIS",
        "variant": "event_archive",
        "role": "quality",
        "benchmark": "StreamingBench RTU",
        "scope": _scope(expected),
        "status": "complete",
        "evidence_tier": (
            "official_model_level_rt_1_50"
            if expected == 250
            else "official_model_level_smoke"
        ),
        "run_fingerprint": str(payload.get("run_fingerprint", "")),
        "source_path": source,
        "source_sha256": _sha256(path),
        "quality_accuracy": accuracy,
        "quality_correct": correct,
        "quality_scored": scored,
        "quality_expected": expected,
        "quality_coverage": coverage,
        "whole_run_seconds": wall_seconds,
        "whole_run_semantics": "whole official pace=0 evaluation; not request latency",
        "tail_latency_available": False,
        "stage_latency_available": False,
        "peak_memory_value": peak_memory,
        "peak_memory_unit": "MiB" if peak_memory is not None else None,
        "peak_memory_semantics": (
            "sampled evaluator-process GPU memory" if peak_memory is not None else None
        ),
        "notes": "Slow event-archive quality baseline; wall time is not TTFT or SLO latency.",
    }
    return run, _quality_row(run)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: Any, stem: Path) -> list[str]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [str(png), str(pdf)]


def _plot_quality(
    rows: list[dict[str, Any]], stem: Path, *, scope_label: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: (str(row["method"]), str(row["variant"])))
    labels = [str(row["method"]) for row in ordered]
    accuracy = [100.0 * float(row["accuracy"]) for row in ordered]
    coverage = [100.0 * float(row["coverage"]) for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(max(7.2, 1.3 * len(rows) + 5.0), 4.2))
    for axis, values, ylabel in zip(
        axes,
        (accuracy, coverage),
        ("Multiple-choice accuracy (%)", "Scored coverage (%)"),
        strict=True,
    ):
        bars = axis.bar(
            labels,
            values,
            width=0.56,
            color=[METHOD_STYLES.get(label, (COLORS[2], ".."))[0] for label in labels],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, label in zip(bars, labels, strict=True):
            bar.set_hatch(METHOD_STYLES.get(label, (COLORS[2], ".."))[1])
        axis.set_xlim(-0.65, max(0.65, len(values) - 0.35))
        axis.set_ylim(0.0, 109.0)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(scope_label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=0 if len(values) <= 2 else 18)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(106.0, value + 1.8),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def _plot_stc_latency(rows: list[dict[str, Any]], stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    totals = [row for row in rows if row["stage"] == "instrumented_stage_sum_ms"]
    totals.sort(key=lambda row: str(row["mode"]))
    labels = [str(row["mode"]).upper() for row in totals]
    positions = np.arange(len(totals), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(max(6.3, 1.45 * len(totals) + 3.8), 4.4))
    for offset, field, label, color, hatch in zip(
        (-width, 0.0, width),
        ("p50_ms", "p95_ms", "p99_ms"),
        ("P50", "P95", "P99"),
        COLORS[:3],
        ("//", "xx", ".."),
        strict=True,
    ):
        axis.bar(
            positions + offset,
            [float(row[field]) for row in totals],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
        )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Official mode")
    axis.set_ylabel("ViT encode + visual-token prefill (ms)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    maximum = max(float(row["p99_ms"]) for row in totals)
    axis.set_ylim(0.0, 1.12 * maximum)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def aggregate_results(
    *,
    causalmem_metrics: list[Path],
    stc_results: list[Path],
    oasis_results: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    if not (causalmem_metrics or stc_results or oasis_results):
        raise ValueError("at least one official result path is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    latency: list[dict[str, Any]] = []
    for path in causalmem_metrics:
        run, quality_row = parse_causalmem(path.resolve())
        runs.append(run)
        quality.append(quality_row)
    for path in stc_results:
        run, stage_rows = parse_stc(path.resolve())
        runs.append(run)
        latency.extend(stage_rows)
    for path in oasis_results:
        run, quality_row = parse_oasis(path.resolve())
        runs.append(run)
        quality.append(quality_row)
    identities = [(row["method"], row["variant"], row["scope"]) for row in runs]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate official run identities: {identities}")

    runs.sort(key=lambda row: (str(row["role"]), str(row["method"]), str(row["variant"])))
    quality.sort(key=lambda row: (str(row["scope"]), str(row["method"])))
    latency.sort(key=lambda row: (str(row["mode"]), str(row["stage"])))
    formal_quality = [row for row in quality if row["scope"] == "formal_50x5"]
    smoke_quality = [row for row in quality if row["scope"] != "formal_50x5"]

    _write_csv(out_dir / "official_runs.csv", runs, RUN_FIELDS)
    _write_csv(out_dir / "official_quality_formal.csv", formal_quality, QUALITY_FIELDS)
    _write_csv(out_dir / "official_quality_smoke.csv", smoke_quality, QUALITY_FIELDS)
    _write_csv(out_dir / "official_stc_stage_latency.csv", latency, LATENCY_FIELDS)

    plots: dict[str, list[str]] = {}
    if formal_quality:
        plots["quality_formal"] = _plot_quality(
            formal_quality,
            out_dir / "official_quality_formal",
            scope_label="Formal 50-video/250-question runs",
        )
    if smoke_quality:
        plots["quality_smoke"] = _plot_quality(
            smoke_quality,
            out_dir / "official_quality_smoke",
            scope_label="Smoke scope (not a formal comparison)",
        )
    if latency:
        plots["stc_stage_latency"] = _plot_stc_latency(
            latency, out_dir / "official_stc_stage_latency"
        )

    summary = {
        "format_version": 1,
        "run_count": len(runs),
        "formal_quality_run_count": len(formal_quality),
        "smoke_quality_run_count": len(smoke_quality),
        "stc_stage_row_count": len(latency),
        "runs": runs,
        "quality_formal": formal_quality,
        "quality_smoke": smoke_quality,
        "stc_stage_latency": latency,
        "plots": plots,
        "cautions": [
            "CausalMem and OASIS quality are compared only within the formal 50x5 scope.",
            "Smoke quality is emitted separately and is not a formal method comparison.",
            "STC values cover ViT encode and visual-token prefill stages only.",
            "Whole-run wall time is not mixed with stage P50/P95/P99 or request latency.",
            "Peak-memory fields have method-specific semantics and are not plotted together.",
            "Proxy results are intentionally excluded from this official-result aggregate.",
        ],
    }
    (out_dir / "aggregation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = aggregate_results(
        causalmem_metrics=args.causalmem_metrics,
        stc_results=args.stc_result,
        oasis_results=args.oasis_result,
        out_dir=args.out_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
