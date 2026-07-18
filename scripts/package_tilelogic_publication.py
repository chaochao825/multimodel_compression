#!/usr/bin/env python3
"""Build a sanitized, bounded public result bundle for TileLogic-RVQ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable
from urllib.parse import unquote


PUBLICATION_FORMAT = "tilelogic_rvq_publication_v1"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".log"}
BINARY_PRIVATE_MARKERS = (
    b"/home/",
    b"/root/",
    b"/mnt/",
    b"/data6/",
    b"wangmeiqi",
    b"spco",
    b"172.25.",
    b"C:\\Users\\",
)
REQUIRED_FILES = (
    "cache_provenance_backfill.json",
    "rate_precision_correction_validation.json",
    "cache/cache_summary.json",
    "training/training_summary.json",
    "feature_eval/feature_eval_summary.json",
    "quality/quality_environment.json",
    "quality/quality_summary.json",
    "latency/latency_environment.json",
    "latency/latency_samples.csv",
    "latency/gpu_co_residency_during_run.log",
    "analysis/feature_metrics.csv",
    "analysis/quality_metrics.csv",
    "analysis/rate_components.csv",
    "analysis/router_sample_metrics.csv",
    "analysis/router_metrics.csv",
    "analysis/latency_metrics.csv",
    "analysis/latency_paired_dynamic_fixed.csv",
    "analysis/method_points.csv",
    "analysis/decision_summary.json",
    "analysis/decision_summary.csv",
    "analysis/analysis_environment.json",
    "analysis/TILELOGIC_RVQ_FINAL_REPORT.md",
    "analysis/result_audit_findings.json",
    "analysis/result_audit_report.md",
    "analysis/independent_review_report.md",
)
PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|root|mnt|tmp|data[0-9]*)/"
    r"[A-Za-z0-9_.+/@=-]+(?:/[A-Za-z0-9_.+/@=-]+)*"
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SOURCE_REVIEW_PASS_MARKER = "Source review verdict: **PASS**"
FINAL_RELEASE_PASS_MARKER = "Final release verdict: **PASS**"
CORRECTED_RATE_SENTINELS = {
    ("base_scalar_quant", "0.125"): 0.501953125,
    ("base_scalar_quant", "0.25"): 1.00390625,
    ("base_vq", "0.125"): 0.011930677625868056,
    ("base_vq", "0.25"): 0.012663099500868056,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_name(path_text: str, category: str = "private") -> str:
    name = Path(path_text.rstrip("/")).name or "root"
    return f"external://{category}/{name}"


def sanitize_string(value: str, *, key: str | None = None) -> str:
    """Remove machine-specific absolute paths while retaining artifact identity."""

    if value.startswith("/"):
        category = "models" if key == "model_dir" else "private"
        return _external_name(value, category)
    return PRIVATE_PATH.sub(lambda match: _external_name(match.group(0)), value)


def sanitize_json(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: sanitize_json(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json(item, key=key) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, key=key)
    return value


def sanitize_text(value: str) -> str:
    sanitized = sanitize_string(value)
    trailing_newline = "\n" if sanitized.endswith(("\n", "\r")) else ""
    return "\n".join(line.rstrip() for line in sanitized.splitlines()) + trailing_newline


def _copy_sanitized(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        payload = sanitize_json(json.loads(source.read_text(encoding="utf-8")))
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if source.suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {source}")
            rows = [
                {
                    key: sanitize_string(value, key=key)
                    for key, value in row.items()
                }
                for row in reader
            ]
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    if source.suffix in {".md", ".txt", ".log"}:
        destination.write_text(
            sanitize_text(source.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        return
    shutil.copy2(source, destination)


def _iter_public_files(output_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "PUBLICATION_MANIFEST.json"
    )


def _validate_publication_source(run_dir: Path, *, final_release: bool = False) -> None:
    correction = json.loads(
        (run_dir / "rate_precision_correction_validation.json").read_text(
            encoding="utf-8"
        )
    )
    if not (
        correction.get("records") == 360
        and correction.get("compared_variants") == 360 * 23
        and correction.get("non_rate_semantics_identical") is True
        and not correction.get("errors")
    ):
        raise RuntimeError("rate-precision correction evidence is incomplete")
    provenance = json.loads(
        (run_dir / "cache_provenance_backfill.json").read_text(encoding="utf-8")
    )
    if not (
        provenance.get("records") == 600
        and provenance.get("payload_tensors_unchanged") is True
    ):
        raise RuntimeError("cache-provenance evidence is incomplete")
    review = (run_dir / "analysis/independent_review_report.md").read_text(
        encoding="utf-8"
    )
    if SOURCE_REVIEW_PASS_MARKER not in review:
        raise RuntimeError("sole Review Agent has not approved the source candidate")
    if final_release and FINAL_RELEASE_PASS_MARKER not in review:
        raise RuntimeError("sole Review Agent has not approved the generated bundle")
    audit = json.loads(
        (run_dir / "analysis/result_audit_findings.json").read_text(encoding="utf-8")
    )
    audit_checks = audit.get("checks", [])
    if not (
        audit.get("overall") == "PASS"
        and audit.get("major_failures") == 0
        and len(audit_checks) == 22
        and all(check.get("status") == "PASS" for check in audit_checks)
    ):
        raise RuntimeError("machine audit is not a complete 22-check PASS")

    with (run_dir / "analysis/method_points.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    points = {
        (str(row["method"]), str(row["retention_rate"])): float(
            row["effective_bits_per_original_value"]
        )
        for row in rows
    }
    stale = {
        key: points.get(key)
        for key, expected in CORRECTED_RATE_SENTINELS.items()
        if key not in points
        or not math.isclose(points[key], expected, rel_tol=0.0, abs_tol=1e-12)
    }
    if stale:
        raise RuntimeError(f"corrected rate sentinels are missing or stale: {stale}")


def _validate_markdown_links(output_dir: Path) -> None:
    broken: list[str] = []
    for path in sorted(output_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            target_path = (path.parent / unquote(target)).resolve()
            try:
                target_path.relative_to(output_dir.resolve())
            except ValueError:
                broken.append(f"{path.relative_to(output_dir)} -> {target}")
                continue
            if not target_path.exists():
                broken.append(f"{path.relative_to(output_dir)} -> {target}")
    if broken:
        raise RuntimeError(f"broken or escaping Markdown links: {broken}")


def _read_public_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_public_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _public_record(output_dir: Path, relative: str) -> dict[str, Any]:
    path = output_dir / relative
    return {
        "file": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _move_source_field(payload: dict[str, Any], field: str) -> None:
    if field in payload:
        payload[f"source_{field}"] = payload.pop(field)


def _set_public_reference(
    payload: dict[str, Any], name: str, record: dict[str, Any]
) -> None:
    payload[f"public_{name}_file"] = record["file"]
    payload[f"public_{name}_bytes"] = record["bytes"]
    payload[f"public_{name}_sha256"] = record["sha256"]


def _rewrite_public_provenance(output_dir: Path) -> None:
    training_path = output_dir / "training/training_summary.json"
    training = _read_public_json(training_path)
    _move_source_field(training, "cache_manifest_sha256")
    training["artifact_hash_scope"] = "private_source_run_excluded_artifacts"
    _write_public_json(training_path, training)
    training_record = _public_record(output_dir, "training/training_summary.json")

    cache_provenance_path = output_dir / "cache_provenance_backfill.json"
    cache_provenance = _read_public_json(cache_provenance_path)
    _move_source_field(cache_provenance, "training_summary_sha256")
    cache_provenance["cache_manifest_hash_scope"] = (
        "private_source_run_excluded_cache_manifests"
    )
    _set_public_reference(cache_provenance, "training_summary", training_record)
    _write_public_json(cache_provenance_path, cache_provenance)

    cache_summary_path = output_dir / "cache/cache_summary.json"
    cache_summary = _read_public_json(cache_summary_path)
    _move_source_field(cache_summary, "cache_manifest_sha256")
    _move_source_field(cache_summary, "split_manifest_sha256")
    cache_summary["manifest_hash_scope"] = "private_source_run_excluded_manifests"
    _write_public_json(cache_summary_path, cache_summary)

    feature_path = output_dir / "feature_eval/feature_eval_summary.json"
    feature = _read_public_json(feature_path)
    _move_source_field(feature, "cache_manifest_sha256")
    _move_source_field(feature, "training_summary_sha256")
    _set_public_reference(feature, "training_summary", training_record)
    _write_public_json(feature_path, feature)
    feature_record = _public_record(output_dir, "feature_eval/feature_eval_summary.json")

    quality_path = output_dir / "quality/quality_environment.json"
    quality = _read_public_json(quality_path)
    _move_source_field(quality, "manifest_sha256")
    _move_source_field(quality, "cache_manifest_sha256")
    _move_source_field(quality, "feature_eval_summary_sha256")
    _set_public_reference(quality, "feature_eval_summary", feature_record)
    _write_public_json(quality_path, quality)

    latency_path = output_dir / "latency/latency_environment.json"
    latency = _read_public_json(latency_path)
    _move_source_field(latency, "manifest_sha256")
    source_log = dict(latency["gpu_co_residency_log"])
    public_log = _public_record(
        output_dir, "latency/gpu_co_residency_during_run.log"
    )
    latency["gpu_co_residency_log"] = {
        "source_file": source_log["file"],
        "source_bytes": source_log["bytes"],
        "source_sha256": source_log["sha256"],
        "public_file": public_log["file"],
        "public_bytes": public_log["bytes"],
        "public_sha256": public_log["sha256"],
    }
    _write_public_json(latency_path, latency)

    correction_path = output_dir / "rate_precision_correction_validation.json"
    correction = _read_public_json(correction_path)
    correction["feature_sample_hash_scope"] = (
        "private_source_run_excluded_feature_jsonl"
    )
    _write_public_json(correction_path, correction)

    public_provenance = {
        "training_summary": training_record,
        "feature_eval_summary": feature_record,
        "gpu_co_residency_log": public_log,
    }
    audit_path = output_dir / "analysis/result_audit_findings.json"
    audit = _read_public_json(audit_path)
    audit["evidence_hash_scope"] = "private_source_run"
    audit["public_bundle_provenance"] = public_provenance
    _write_public_json(audit_path, audit)

    audit_report_path = output_dir / "analysis/result_audit_report.md"
    audit_report = audit_report_path.read_text(encoding="utf-8")
    audit_note = (
        "> Public provenance note: hashes in the machine-audit findings describe "
        "the private source run. Hashes for sanitized bundled files are recorded "
        "in `PUBLICATION_MANIFEST.json` and `public_bundle_provenance`.\n\n"
    )
    audit_report_path.write_text(audit_note + audit_report, encoding="utf-8")

    report_path = output_dir / "analysis/TILELOGIC_RVQ_FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    latency_note = (
        "Public-provenance note: the GPU co-residency log has private source "
        f"SHA256 `{source_log['sha256']}` and sanitized public SHA256 "
        f"`{public_log['sha256']}`. The public copy is {public_log['bytes']} bytes."
    )
    report = report.replace(
        "\n## 8. Evidence Boundaries", f"\n{latency_note}\n\n## 8. Evidence Boundaries"
    )
    report_path.write_text(report, encoding="utf-8")

    environment_path = output_dir / "analysis/analysis_environment.json"
    environment = _read_public_json(environment_path)
    environment.update(
        {
            "training": training,
            "feature": feature,
            "quality": quality,
            "latency": latency,
            "cache_provenance_backfill": cache_provenance,
            "rate_precision_correction": correction,
            "public_bundle_provenance": public_provenance,
            "evidence_hash_scope": (
                "source_* fields identify private inputs; public_* fields identify "
                "sanitized bundled bytes"
            ),
        }
    )
    _write_public_json(environment_path, environment)


def _validate_public_provenance(output_dir: Path) -> None:
    training = _read_public_json(output_dir / "training/training_summary.json")
    training_record = _public_record(output_dir, "training/training_summary.json")
    feature = _read_public_json(output_dir / "feature_eval/feature_eval_summary.json")
    feature_record = _public_record(output_dir, "feature_eval/feature_eval_summary.json")
    quality = _read_public_json(output_dir / "quality/quality_environment.json")
    latency = _read_public_json(output_dir / "latency/latency_environment.json")
    cache_provenance = _read_public_json(output_dir / "cache_provenance_backfill.json")
    correction = _read_public_json(
        output_dir / "rate_precision_correction_validation.json"
    )
    audit = _read_public_json(output_dir / "analysis/result_audit_findings.json")
    environment = _read_public_json(output_dir / "analysis/analysis_environment.json")
    public_log = _public_record(
        output_dir, "latency/gpu_co_residency_during_run.log"
    )

    errors: list[str] = []

    def check_reference(
        payload: dict[str, Any], name: str, expected: dict[str, Any], label: str
    ) -> None:
        actual = {
            "file": payload.get(f"public_{name}_file"),
            "bytes": payload.get(f"public_{name}_bytes"),
            "sha256": payload.get(f"public_{name}_sha256"),
        }
        if actual != expected:
            errors.append(f"{label}: {actual} != {expected}")

    check_reference(
        cache_provenance,
        "training_summary",
        training_record,
        "cache provenance -> training summary",
    )
    check_reference(
        feature,
        "training_summary",
        training_record,
        "feature summary -> training summary",
    )
    check_reference(
        quality,
        "feature_eval_summary",
        feature_record,
        "quality environment -> feature summary",
    )
    latency_public = {
        "file": latency["gpu_co_residency_log"].get("public_file"),
        "bytes": latency["gpu_co_residency_log"].get("public_bytes"),
        "sha256": latency["gpu_co_residency_log"].get("public_sha256"),
    }
    if latency_public != public_log:
        errors.append(f"latency environment -> public log: {latency_public} != {public_log}")
    for field in ("source_file", "source_bytes", "source_sha256"):
        if not latency["gpu_co_residency_log"].get(field):
            errors.append(f"latency environment missing {field}")
    if environment.get("training") != training:
        errors.append("analysis environment training copy differs")
    if environment.get("feature") != feature:
        errors.append("analysis environment feature copy differs")
    if environment.get("quality") != quality:
        errors.append("analysis environment quality copy differs")
    if environment.get("latency") != latency:
        errors.append("analysis environment latency copy differs")
    if environment.get("cache_provenance_backfill") != cache_provenance:
        errors.append("analysis environment cache provenance copy differs")
    if environment.get("rate_precision_correction") != correction:
        errors.append("analysis environment rate correction copy differs")
    expected_public = {
        "training_summary": training_record,
        "feature_eval_summary": feature_record,
        "gpu_co_residency_log": public_log,
    }
    if audit.get("public_bundle_provenance") != expected_public:
        errors.append("machine audit public provenance differs")
    if environment.get("public_bundle_provenance") != expected_public:
        errors.append("analysis environment public provenance differs")
    report = (output_dir / "analysis/TILELOGIC_RVQ_FINAL_REPORT.md").read_text(
        encoding="utf-8"
    )
    log_state = latency["gpu_co_residency_log"]
    if (
        log_state["source_sha256"] not in report
        or log_state["public_sha256"] not in report
    ):
        errors.append("formal report omits source/public latency hashes")
    ambiguous_fields = {
        "cache_provenance.training_summary_sha256": cache_provenance.get(
            "training_summary_sha256"
        ),
        "feature.training_summary_sha256": feature.get("training_summary_sha256"),
        "quality.feature_eval_summary_sha256": quality.get(
            "feature_eval_summary_sha256"
        ),
    }
    if any(value is not None for value in ambiguous_fields.values()):
        errors.append(f"ambiguous cross-file hash fields remain: {ambiguous_fields}")
    if errors:
        raise RuntimeError(f"public provenance validation failed: {errors}")


def _write_readme(output_dir: Path, decisions: dict[str, Any]) -> None:
    statuses = "\n".join(
        f"- Q{item['id']}: **{item['status']}** - {item['question']}"
        for item in decisions["questions"]
    )
    text = f"""# TileLogic-RVQ Formal Result Bundle

This directory is the sanitized public evidence bundle for the formal
TileLogic-RVQ experiment. The method combines tile-local DCT, scaled base VQ,
sequential residual VQ, calibration-only MLP and discrete-logic routing,
fixed slots, and a fully charged sparse FP16 fallback.

## Frozen Decisions

{statuses}

Aggregate positive claim allowed: **{'YES' if decisions['aggregate_positive_claim_allowed'] else 'NO'}**.

## Included

- Aggregate feature, rate, Fisher, quality, router, latency, TTFT, and memory tables.
- Six frozen-rule decisions, the independent machine audit, and the sole
  Review Agent's initial and final verdicts.
- The cache-provenance migration record and artifact-precision audit.
- PDF/PNG figures, environment summaries, per-sample latency rows, and the
  during-run GPU co-residency snapshot.
- `PUBLICATION_MANIFEST.json` with SHA256 and byte count for every public file.

## Deliberately Excluded

- Model checkpoints, images, cached visual tensors, and private dataset paths.
- Trained `.pt` codebooks/router payloads and build caches.
- Per-example questions, answers, predictions, and 20 MiB feature JSONL.
- Credentials, private server paths, and unrelated experiment logs.

The private run was audited against those excluded records before this bundle
was produced. The public bundle supports result and claim inspection, but does
not by itself reproduce cache extraction or model inference.

The rate ledger charges FP32 for scalar scales, VQ metric weights,
MLP/normalizer state, logic leaves, and curvature priors; FP16 is charged only
for codewords, scale tables, logic thresholds, and exact residual values that
are explicitly serialized or exactly round-tripped at that precision.
Per-entry cache provenance was
backfilled without changing any cached tensor payload and is hash-linked in the
included migration record.

Hashes named `source_*` describe the private pre-sanitization evidence used by
the machine audit. Hashes named `public_*` are recomputed from the actual files
in this bundle after path and whitespace sanitization. Both scopes are retained
so the public provenance chain does not reinterpret private source hashes.

The quality path reconstructs all 1,280 visual tokens. Latency is a PyTorch
implementation diagnostic, not native compact-prefill evidence, isolated-GPU
signoff, kernel-fusion evidence, PPA, or post-layout hardware evidence.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--final-release",
        action="store_true",
        help="require the sole Review Agent's bundle-level PASS marker",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {output_dir}")
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"required publication inputs are missing: {missing}")
    _validate_publication_source(run_dir, final_release=args.final_release)

    for relative in REQUIRED_FILES:
        _copy_sanitized(run_dir / relative, output_dir / relative)
    figure_sources = sorted((run_dir / "analysis/figures").glob("*"))
    if not figure_sources:
        raise SystemExit("analysis figures are missing")
    for source in figure_sources:
        if source.is_file() and source.suffix.lower() in {".png", ".pdf"}:
            _copy_sanitized(source, output_dir / "analysis/figures" / source.name)
    _rewrite_public_provenance(output_dir)

    decisions = json.loads(
        (output_dir / "analysis/decision_summary.json").read_text(encoding="utf-8")
    )
    _write_readme(output_dir, decisions)
    _validate_public_provenance(output_dir)
    _validate_markdown_links(output_dir)
    private_hits = []
    for path in _iter_public_files(output_dir):
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if PRIVATE_PATH.search(text):
                private_hits.append(str(path.relative_to(output_dir)))
    if private_hits:
        raise RuntimeError(f"private absolute paths remain: {private_hits}")
    binary_private_hits = []
    for path in _iter_public_files(output_dir):
        payload = path.read_bytes()
        if any(marker in payload for marker in BINARY_PRIVATE_MARKERS):
            binary_private_hits.append(str(path.relative_to(output_dir)))
    if binary_private_hits:
        raise RuntimeError(
            f"private markers remain in binary scan: {binary_private_hits}"
        )

    files = [
        {
            "path": str(path.relative_to(output_dir)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _iter_public_files(output_dir)
    ]
    manifest = {
        "format": PUBLICATION_FORMAT,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "private_path_scan_pass": True,
        "binary_private_marker_scan_pass": True,
        "markdown_link_scan_pass": True,
        "source_public_provenance_scan_pass": True,
        "release_ready": args.final_release,
        "review_stage": "final" if args.final_release else "source-reviewed-candidate",
        "excluded": [
            "cache payload tensors",
            "training .pt artifacts",
            "feature_samples.jsonl",
            "quality_samples.jsonl",
            "model checkpoints and datasets",
        ],
    }
    (output_dir / "PUBLICATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
