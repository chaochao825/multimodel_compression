#!/usr/bin/env python3
"""Freeze Nystrom/sparse-tail configurations on validation, then score test.

The selector deliberately ignores dense-derived head-role labels.  Test rows
are unavailable to configuration selection and are read only after the
validation choice has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiment_artifacts import (
    JsonlEventLog,
    SplitProtocol,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_split_protocols,
    object_sha256,
    require_fresh_output_dir,
)


CONFIG_FIELDS = ("method", "landmark_mode", "landmarks", "pinv_rtol", "density")
DEPLOYABLE_METHODS = frozenset(
    {
        "nystrom_signed",
        "landmark_linear",
        "proxy_mass_nystrom_mixture",
        "proxy_mass_landmark_partition",
    }
)
DIAGNOSTIC_METHODS = frozenset(
    {
        "nystrom_nonnegative_clamped",
        "dense_mass_nystrom_mixture_diagnostic",
        "dense_mass_landmark_partition_diagnostic",
    }
)


@dataclass(frozen=True, order=True)
class CandidateConfig:
    method: str
    landmark_mode: str
    landmarks: int
    pinv_rtol: float
    density: float

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "CandidateConfig":
        return cls(
            method=str(row["method"]),
            landmark_mode=str(row["landmark_mode"]),
            landmarks=int(row["landmarks"]),
            pinv_rtol=float(row["pinv_rtol"]),
            density=float(row["density"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "landmark_mode": self.landmark_mode,
            "landmarks": self.landmarks,
            "pinv_rtol": self.pinv_rtol,
            "density": self.density,
        }


@dataclass(frozen=True)
class SelectionThresholds:
    aggregate_error: float
    record_error: float
    speedup: float
    max_work_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "aggregate_error": self.aggregate_error,
            "record_error": self.record_error,
            "speedup": self.speedup,
            "max_work_ratio": self.max_work_ratio,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-target", type=float, default=0.01)
    parser.add_argument("--record-target", type=float, default=0.02)
    parser.add_argument("--speed-target", type=float, default=1.5)
    parser.add_argument("--max-work-ratio", type=float, default=0.50)
    parser.add_argument(
        "--head-scope",
        choices=("all", "calibration_role"),
        default="all",
    )
    parser.add_argument(
        "--calibration-role",
        choices=("localized", "transitional", "diffuse"),
        default="transitional",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def read_probe_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"probe CSV is empty: {path}")

    rows: list[dict[str, object]] = []
    numeric_float_fields = (
        "pinv_rtol",
        "density",
        "residual_sq",
        "reference_sq",
        "output_relative_l2",
        "projected_attention_work_ratio",
        "arithmetic_speedup_upper_bound",
    )
    numeric_int_fields = ("landmarks", "sampling_step", "layer", "head")
    for index, raw in enumerate(raw_rows, start=2):
        row: dict[str, object] = dict(raw)
        try:
            for field in numeric_float_fields:
                row[field] = float(raw[field])
            for field in numeric_int_fields:
                row[field] = int(raw[field])
            row["deployable_candidate"] = parse_bool(raw["deployable_candidate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid probe row {path}:{index}: {error}") from error
        finite_fields = (*numeric_float_fields,)
        if any(not math.isfinite(float(row[field])) for field in finite_fields):
            raise ValueError(f"non-finite metric in {path}:{index}")
        if float(row["reference_sq"]) <= 0 or float(row["residual_sq"]) < 0:
            raise ValueError(f"invalid squared norm in {path}:{index}")
        if float(row["projected_attention_work_ratio"]) <= 0:
            raise ValueError(f"invalid work ratio in {path}:{index}")
        method = str(row["method"])
        expected_deployable = method in DEPLOYABLE_METHODS
        if bool(row["deployable_candidate"]) != expected_deployable:
            raise ValueError(
                f"deployability flag disagrees with method in {path}:{index}: "
                f"method={method!r}"
            )
        rows.append(row)
    return rows


def evaluation_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["sample_id"],
        row["sampling_step"],
        row["branch"],
        row["layer"],
        row["head"],
    )


def head_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["sampling_step"],
        row["branch"],
        row["layer"],
        row["head"],
    )


def unique_role_observations(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[object, ...], str]:
    observations: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for row in rows:
        observations[(row["sample_id"], *head_key(row))].add(
            str(row["head_role_diagnostic_only"])
        )
    inconsistent = {
        key: sorted(roles) for key, roles in observations.items() if len(roles) != 1
    }
    if inconsistent:
        raise ValueError(f"head role changes across configurations: {inconsistent}")
    return {key: next(iter(roles)) for key, roles in observations.items()}


def frozen_calibration_role_heads(
    rows: Sequence[Mapping[str, object]],
    protocol: SplitProtocol,
    target_role: str,
) -> tuple[frozenset[tuple[object, ...]], list[dict[str, object]]]:
    """Freeze a static head map using calibration rows only."""

    observations = unique_role_observations(rows)
    all_head_keys = sorted({head_key(row) for row in rows}, key=lambda key: tuple(map(str, key)))
    selected: set[tuple[object, ...]] = set()
    routing_rows: list[dict[str, object]] = []
    for key in all_head_keys:
        calibration_roles = []
        for sample_id in protocol.calibration:
            observation_key = (sample_id, *key)
            if observation_key not in observations:
                raise ValueError(
                    f"missing calibration role for protocol={protocol.name}, "
                    f"sample={sample_id}, head_key={key}"
                )
            calibration_roles.append(observations[observation_key])
        selected_by_calibration = all(
            role == target_role for role in calibration_roles
        )
        if selected_by_calibration:
            selected.add(key)
        for sample_id in sorted(protocol.sample_ids):
            observed_role = observations[(sample_id, *key)]
            routing_rows.append(
                {
                    "protocol": protocol.name,
                    "target_role": target_role,
                    "sample_id": sample_id,
                    "split": protocol.split_for(sample_id),
                    "sampling_step": key[0],
                    "branch": key[1],
                    "layer": key[2],
                    "head": key[3],
                    "calibration_roles": ";".join(calibration_roles),
                    "selected_by_calibration": selected_by_calibration,
                    "observed_dense_role_diagnostic_only": observed_role,
                    "target_role_match_diagnostic_only": observed_role == target_role,
                }
            )
    if not selected:
        raise ValueError(
            f"protocol {protocol.name} calibration selected no {target_role} heads"
        )
    return frozenset(selected), routing_rows


def validate_rectangular_sweep(rows: Sequence[Mapping[str, object]]) -> None:
    by_evaluation: dict[tuple[object, ...], set[CandidateConfig]] = defaultdict(set)
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        config = CandidateConfig.from_row(row)
        unique = (*evaluation_key(row), config)
        if unique in seen:
            raise ValueError(f"duplicate evaluation/config row: {unique}")
        seen.add(unique)
        by_evaluation[evaluation_key(row)].add(config)
    expected = next(iter(by_evaluation.values()))
    for key, configs in by_evaluation.items():
        if configs != expected:
            missing = sorted(expected - configs)
            extra = sorted(configs - expected)
            raise ValueError(
                f"non-rectangular sweep at {key}: missing={missing}, extra={extra}"
            )


def metric_for_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate an empty row set")
    residual_sq = sum(float(row["residual_sq"]) for row in rows)
    reference_sq = sum(float(row["reference_sq"]) for row in rows)
    errors = [float(row["output_relative_l2"]) for row in rows]
    work = sum(
        float(row["projected_attention_work_ratio"]) for row in rows
    ) / len(rows)
    return {
        "records": len(rows),
        "aggregate_output_relative_l2": math.sqrt(
            residual_sq / max(reference_sq, 1e-30)
        ),
        "record_error_max": max(errors),
        "projected_attention_work_ratio_mean": work,
        "arithmetic_speedup_upper_bound": 1.0 / work,
    }


def metrics_by_config(
    rows: Sequence[Mapping[str, object]],
    sample_ids: Iterable[str],
    allowed_methods: frozenset[str],
    allowed_head_keys: frozenset[tuple[object, ...]] | None = None,
) -> dict[CandidateConfig, dict[str, float | int]]:
    selected_samples = frozenset(sample_ids)
    groups: dict[CandidateConfig, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        config = CandidateConfig.from_row(row)
        if (
            str(row["sample_id"]) in selected_samples
            and config.method in allowed_methods
            and (allowed_head_keys is None or head_key(row) in allowed_head_keys)
        ):
            groups[config].append(row)
    if not groups:
        raise ValueError(
            f"no rows for samples={sorted(selected_samples)}, methods={sorted(allowed_methods)}"
        )
    return {config: metric_for_rows(group) for config, group in groups.items()}


def metric_passes(
    metric: Mapping[str, float | int], thresholds: SelectionThresholds
) -> bool:
    return (
        float(metric["aggregate_output_relative_l2"])
        <= thresholds.aggregate_error
        and float(metric["record_error_max"]) <= thresholds.record_error
        and float(metric["arithmetic_speedup_upper_bound"]) >= thresholds.speedup
        and float(metric["projected_attention_work_ratio_mean"])
        <= thresholds.max_work_ratio
    )


def choose_validation_config(
    validation_metrics: Mapping[CandidateConfig, Mapping[str, float | int]],
    thresholds: SelectionThresholds,
) -> tuple[CandidateConfig, str]:
    feasible = [
        (config, metric)
        for config, metric in validation_metrics.items()
        if metric_passes(metric, thresholds)
    ]
    if feasible:
        feasible.sort(
            key=lambda item: (
                float(item[1]["projected_attention_work_ratio_mean"]),
                float(item[1]["aggregate_output_relative_l2"]),
                float(item[1]["record_error_max"]),
                item[0],
            )
        )
        return feasible[0][0], "PASS"

    within_cost = [
        (config, metric)
        for config, metric in validation_metrics.items()
        if float(metric["arithmetic_speedup_upper_bound"]) >= thresholds.speedup
        and float(metric["projected_attention_work_ratio_mean"])
        <= thresholds.max_work_ratio
    ]
    pool = within_cost or list(validation_metrics.items())
    pool.sort(
        key=lambda item: (
            float(item[1]["aggregate_output_relative_l2"]),
            float(item[1]["record_error_max"]),
            float(item[1]["projected_attention_work_ratio_mean"]),
            item[0],
        )
    )
    status = "FAIL_QUALITY" if within_cost else "FAIL_COST_AND_QUALITY"
    return pool[0][0], status


def select_for_protocol(
    rows: Sequence[Mapping[str, object]],
    protocol: SplitProtocol,
    allowed_methods: frozenset[str],
    thresholds: SelectionThresholds,
    allowed_head_keys: frozenset[tuple[object, ...]] | None = None,
) -> tuple[CandidateConfig, str, dict[str, dict[str, float | int]]]:
    """Select using validation only; test rows cannot influence the choice."""

    validation_metrics = metrics_by_config(
        rows, protocol.validation, allowed_methods, allowed_head_keys
    )
    chosen, validation_status = choose_validation_config(
        validation_metrics, thresholds
    )
    split_metrics: dict[str, dict[str, float | int]] = {}
    for split_name, sample_ids in (
        ("calibration", protocol.calibration),
        ("validation", protocol.validation),
        ("test", protocol.test),
    ):
        candidates = metrics_by_config(
            rows, sample_ids, allowed_methods, allowed_head_keys
        )
        if chosen not in candidates:
            raise ValueError(f"selected config missing from {split_name}: {chosen}")
        split_metrics[split_name] = candidates[chosen]
    return chosen, validation_status, split_metrics


def all_config_metrics(
    rows: Sequence[Mapping[str, object]],
    protocol: SplitProtocol,
    allowed_head_keys: frozenset[tuple[object, ...]] | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for split_name, sample_ids in (
        ("calibration", protocol.calibration),
        ("validation", protocol.validation),
        ("test", protocol.test),
    ):
        for config, metric in metrics_by_config(
            rows,
            sample_ids,
            DEPLOYABLE_METHODS | DIAGNOSTIC_METHODS,
            allowed_head_keys,
        ).items():
            output.append(
                {
                    "protocol": protocol.name,
                    "split": split_name,
                    "candidate_class": (
                        "deployable"
                        if config.method in DEPLOYABLE_METHODS
                        else "diagnostic_only"
                    ),
                    **config.as_dict(),
                    **metric,
                }
            )
    return output


def validate_probe_artifacts(probe_dir: Path, split_config: Path) -> dict[str, object]:
    success_path = probe_dir / "SUCCESS.json"
    manifest_path = probe_dir / "manifest.json"
    input_manifest_path = probe_dir / "input_manifest.json"
    if not (
        success_path.is_file()
        and manifest_path.is_file()
        and input_manifest_path.is_file()
    ):
        raise ValueError(
            f"probe is incomplete; SUCCESS/manifest/input manifest missing in {probe_dir}"
        )
    success = json.loads(success_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if success.get("status") != "SUCCESS":
        raise ValueError(f"probe did not finish successfully: {success}")
    if success.get("config_sha256") != manifest.get("config_sha256"):
        raise ValueError("probe SUCCESS and manifest config hashes disagree")
    if manifest.get("split_config_sha256") != file_sha256(split_config):
        raise ValueError("selector split config differs from the probe split config")
    if manifest.get("input_manifest_sha256") != file_sha256(input_manifest_path):
        raise ValueError("probe input manifest hash disagrees with probe manifest")
    return manifest


def main() -> None:
    args = parse_args()
    thresholds = SelectionThresholds(
        aggregate_error=args.aggregate_target,
        record_error=args.record_target,
        speedup=args.speed_target,
        max_work_ratio=args.max_work_ratio,
    )
    if min(thresholds.as_dict().values()) <= 0:
        raise ValueError("all selection thresholds must be positive")

    probe_dir = args.probe_dir.resolve()
    split_config = args.split_config.resolve()
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    manifest = validate_probe_artifacts(probe_dir, split_config)
    detail_path = probe_dir / "nystrom_sparse_tail_heads.csv"
    rows = read_probe_rows(detail_path)
    validate_rectangular_sweep(rows)
    observed_samples = {str(row["sample_id"]) for row in rows}
    protocols = load_split_protocols(split_config)
    for protocol in protocols:
        protocol.assert_exact_coverage(observed_samples)

    selector_config = {
        "probe_dir": str(probe_dir),
        "split_config": str(split_config),
        "thresholds": thresholds.as_dict(),
        "deployable_methods": sorted(DEPLOYABLE_METHODS),
        "diagnostic_methods": sorted(DIAGNOSTIC_METHODS),
        "head_scope": args.head_scope,
        "calibration_role": args.calibration_role,
    }
    selector_hash = object_sha256(selector_config)
    run_id = f"nystrom-select-{selector_hash[:12]}-{int(time.time())}"
    event_log = JsonlEventLog(output_dir / "progress.jsonl", run_id)
    atomic_write_json(
        output_dir / "run_state.json",
        {"status": "RUNNING", "run_id": run_id, "config_sha256": selector_hash},
    )

    selected_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    numerical_pass = True
    for protocol in protocols:
        if args.head_scope == "calibration_role":
            allowed_head_keys, protocol_routing_rows = frozen_calibration_role_heads(
                rows, protocol, args.calibration_role
            )
            routing_rows.extend(protocol_routing_rows)
        else:
            allowed_head_keys = None
        chosen, validation_status, split_metrics = select_for_protocol(
            rows,
            protocol,
            DEPLOYABLE_METHODS,
            thresholds,
            allowed_head_keys,
        )
        test_pass = metric_passes(split_metrics["test"], thresholds)
        numerical_pass = numerical_pass and validation_status == "PASS" and test_pass
        for split_name, metric in split_metrics.items():
            selected_rows.append(
                {
                    "protocol": protocol.name,
                    "claim_boundary": protocol.claim_boundary,
                    "selection_source": "validation_only",
                    "head_scope": args.head_scope,
                    "calibration_role": (
                        args.calibration_role
                        if args.head_scope == "calibration_role"
                        else "not_applicable"
                    ),
                    "frozen_head_keys": (
                        len(allowed_head_keys)
                        if allowed_head_keys is not None
                        else "all"
                    ),
                    "validation_status": validation_status,
                    "test_status": "PASS" if test_pass else "FAIL",
                    "split": split_name,
                    **chosen.as_dict(),
                    **metric,
                }
            )
        protocol_config_rows = all_config_metrics(
            rows, protocol, allowed_head_keys
        )
        for config_row in protocol_config_rows:
            config_row["head_scope"] = args.head_scope
            config_row["calibration_role"] = (
                args.calibration_role
                if args.head_scope == "calibration_role"
                else "not_applicable"
            )
            config_row["frozen_head_keys"] = (
                len(allowed_head_keys)
                if allowed_head_keys is not None
                else "all"
            )
        config_rows.extend(protocol_config_rows)
        event_log.emit(
            "protocol_selected",
            protocol=protocol.name,
            selected_config=chosen.as_dict(),
            validation_status=validation_status,
            test_status="PASS" if test_pass else "FAIL",
            frozen_head_keys=(
                len(allowed_head_keys)
                if allowed_head_keys is not None
                else "all"
            ),
        )

    atomic_write_csv(output_dir / "selected_protocol_metrics.csv", selected_rows)
    atomic_write_csv(output_dir / "all_config_split_metrics.csv", config_rows)
    if routing_rows:
        atomic_write_csv(output_dir / "frozen_head_routing.csv", routing_rows)
    selection_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "artifact_status": "SUCCESS",
        "numerical_gate": "PASS" if numerical_pass else "FAIL",
        "deployment_gate": "UNMEASURED",
        "scientific_gate": "INCOMPLETE" if numerical_pass else "FAIL",
        "selector_config": selector_config,
        "config_sha256": selector_hash,
        "probe_run_id": manifest["run_id"],
        "probe_config_sha256": manifest["config_sha256"],
        "probe_execution_resource_note": manifest.get(
            "execution_resource_note", "unspecified"
        ),
        "probe_detail_sha256": file_sha256(detail_path),
        "split_config_sha256": file_sha256(split_config),
        "protocols": [protocol.as_dict() for protocol in protocols],
        "selection_rule": (
            "validation only: fastest candidate satisfying all thresholds; "
            "otherwise lowest validation aggregate error within the cost budget"
        ),
        "speed_claim_boundary": (
            "speed thresholds use arithmetic upper bounds; measured H200 latency "
            "is required before the scientific gate can pass"
        ),
        "test_isolation": (
            "test metrics are computed only after CandidateConfig is frozen"
        ),
        "role_isolation": (
            "all-head mode ignores dense roles; calibration-role mode freezes a "
            "static head map from calibration before validation/test aggregation"
        ),
        "claim_warning": (
            "Four captures support pilot evidence only, not a population-level claim"
        ),
        "development_reuse_warning": (
            "These captures were used in prior exploratory probes; test isolation is "
            "within-run only and an untouched confirmatory set is still required"
        ),
        "python": platform.python_version(),
    }
    atomic_write_json(output_dir / "selection_manifest.json", selection_manifest)
    success = {
        "status": "SUCCESS",
        "numerical_gate": selection_manifest["numerical_gate"],
        "deployment_gate": selection_manifest["deployment_gate"],
        "scientific_gate": selection_manifest["scientific_gate"],
        "run_id": run_id,
        "config_sha256": selector_hash,
        "protocols": len(protocols),
    }
    atomic_write_json(output_dir / "run_state.json", success)
    atomic_write_json(output_dir / "SUCCESS.json", success)
    event_log.emit("selection_completed", **success)
    print(
        f"[selector] artifact=SUCCESS numerical_gate={selection_manifest['numerical_gate']} "
        f"scientific_gate={selection_manifest['scientific_gate']} "
        f"protocols={len(protocols)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
