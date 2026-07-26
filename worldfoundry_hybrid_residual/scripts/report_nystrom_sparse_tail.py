#!/usr/bin/env python3
"""Generate a claim-bounded Markdown report from frozen-selection artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment_artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    require_fresh_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "pilot", "full"), required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def pct(value: str) -> str:
    return f"{100 * float(value):.3f}%"


def selected_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Protocol | Split | Head scope | Frozen method | m | Density | Aggregate | Worst record | Arithmetic upper bound |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["split"] not in {"validation", "test"}:
            continue
        row_head_scope = row.get("head_scope", "all")
        lines.append(
            "| {protocol} | {split} | {head_scope} | `{method}` | {landmarks} | {density:.1%} | "
            "{aggregate} | {worst} | {speed:.2f}x |".format(
                protocol=row["protocol"],
                split=row["split"],
                head_scope=(
                    f"{row.get('calibration_role', 'unknown')} "
                    f"({row.get('frozen_head_keys', 'unknown')})"
                    if row_head_scope == "calibration_role"
                    else "all"
                ),
                method=row["method"],
                landmarks=row["landmarks"],
                density=float(row["density"]),
                aggregate=pct(row["aggregate_output_relative_l2"]),
                worst=pct(row["record_error_max"]),
                speed=float(row["arithmetic_speedup_upper_bound"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    success_path = selection_dir / "SUCCESS.json"
    manifest_path = selection_dir / "selection_manifest.json"
    selected_path = selection_dir / "selected_protocol_metrics.csv"
    for path in (success_path, manifest_path, selected_path):
        if not path.is_file():
            raise ValueError(f"incomplete selector artifact: missing {path}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(selected_path)
    if success.get("status") != "SUCCESS":
        raise ValueError(f"selector artifact is not complete: {success}")

    numerical_gate = str(success["numerical_gate"])
    deployment_gate = str(success["deployment_gate"])
    scientific_gate = str(success["scientific_gate"])
    if args.run_kind == "smoke":
        next_action = (
            "NO_SCIENTIFIC_DECISION: this run verifies integration and artifacts only."
        )
        bounded_conclusion = (
            "Smoke metrics cannot accept or reject the method because the registered "
            "pilot capacity and coverage were not evaluated."
        )
        displayed_scientific_gate = "NOT_EVALUATED"
    elif numerical_gate == "PASS":
        next_action = (
            "PROCEED_CONDITIONALLY: expand the frozen protocol to all planned cells, "
            "then require a fused H200 kernel benchmark before any acceleration claim."
        )
        bounded_conclusion = (
            "The pilot supports further evaluation of this train-free family under the "
            "registered numerical thresholds; it does not establish H200 acceleration."
        )
        displayed_scientific_gate = scientific_gate
    else:
        next_action = (
            "STOP_THIS_TRAIN_FREE_FAMILY: do not expand this Nystrom/landmark sweep. "
            "A learned low-cost tail remains a separate, untested hypothesis."
        )
        bounded_conclusion = (
            "The validation-frozen train-free candidates failed the registered pilot "
            "gate. This does not falsify content-conditioned learned tails in general."
        )
        displayed_scientific_gate = scientific_gate

    protocol_lines = []
    for protocol in manifest["protocols"]:
        protocol_lines.append(
            f"- `{protocol['name']}`: {protocol['claim_boundary']}"
        )
    head_scope = rows[0].get("head_scope", "all")
    if head_scope == "calibration_role":
        role_statement = (
            "Dense roles were used only on calibration to freeze a static head map; "
            "validation/test roles did not alter routing or selection."
        )
    else:
        role_statement = "Dense-derived head-role labels were ignored."
    markdown = f"""# {args.experiment_label}

## Gate Status

| Gate | Status |
|---|---|
| Artifact completeness | `{success['status']}` |
| Numerical quality + arithmetic upper bound | `{numerical_gate}` |
| Measured H200 deployment | `{deployment_gate}` |
| Scientific claim | `{displayed_scientific_gate}` |

**Decision:** {next_action}

{bounded_conclusion}

## Frozen Results

Configurations were selected on validation only. Test metrics were evaluated after
the configuration tuple was frozen. {role_statement}

{selected_table(rows)}

## Claim Boundaries

{chr(10).join(protocol_lines)}

- The dataset contains four captures and is pilot evidence, not a population estimate.
- These captures were reused from exploratory work; this is not an untouched external test.
- A confirmatory claim requires new prompts and seeds registered before inspection.
- Run kind is `{args.run_kind}`; smoke runs never trigger a scientific stop/go decision.
- Reported speed is an arithmetic upper bound, not wall-clock H200 speed.
- `dense_mass_*_diagnostic` isolates selected-mass error but is not a quality oracle.
- Full-matrix nonnegative clamping is diagnostic-only because it cannot preserve the low-rank association.
- Passing this report cannot produce a final acceleration claim while deployment is `UNMEASURED`.

## Reproducibility

- Selection run: `{manifest['run_id']}`
- Probe run: `{manifest['probe_run_id']}`
- Probe config SHA256: `{manifest['probe_config_sha256']}`
- Probe resource mode: `{manifest['probe_execution_resource_note']}`
- Probe detail SHA256: `{manifest['probe_detail_sha256']}`
- Split config SHA256: `{manifest['split_config_sha256']}`
"""
    require_fresh_output_dir(output_dir)
    report_path = output_dir / "report.md"
    atomic_write_text(report_path, markdown)
    decision = {
        "schema_version": 1,
        "artifact_status": success["status"],
        "run_kind": args.run_kind,
        "numerical_gate": numerical_gate,
        "deployment_gate": deployment_gate,
        "scientific_gate": displayed_scientific_gate,
        "next_action": next_action,
        "selection_run_id": manifest["run_id"],
        "report_sha256": file_sha256(report_path),
    }
    atomic_write_json(output_dir / "decision.json", decision)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "scientific_gate": displayed_scientific_gate,
            "report": report_path.name,
        },
    )
    print(
        f"[report] wrote {report_path} numerical_gate={numerical_gate} "
        f"scientific_gate={displayed_scientific_gate}",
        flush=True,
    )


if __name__ == "__main__":
    main()
