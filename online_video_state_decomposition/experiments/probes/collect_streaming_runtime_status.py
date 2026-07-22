from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aggregate_official_streaming_results import (
    STREAMINGTOM_COMMIT,
    STREAMINGTOM_SPECS,
    parse_causalmem,
    parse_oasis,
    parse_stc,
    parse_streamingtom,
)
from run_oasis_streamingbench import summarize_official_output


WAITING_STATUSES = {
    "waiting_for_dependency",
    "waiting_for_flash_attn",
    "waiting_for_idle",
}
RUNNING_STATUSES = {"launching", "running"}
COMPLETED_STATUSES = {"complete", "completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a read-only runtime snapshot for the streaming evidence matrix"
        )
    )
    parser.add_argument("--oasis-run", type=Path)
    parser.add_argument("--oasis-metadata", type=Path)
    parser.add_argument("--stc-run", type=Path)
    parser.add_argument("--causalmem-run", type=Path)
    parser.add_argument("--streamingtom-run", type=Path)
    parser.add_argument("--streamingtom-preflight-dir", type=Path)
    parser.add_argument("--observed-at")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _observed_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid --observed-at timestamp: {value}") from error
    if observed.tzinfo is None:
        raise ValueError("--observed-at must include a UTC offset")
    return observed.astimezone(timezone.utc).isoformat()


def _read_status(run_dir: Path) -> tuple[str | None, Path]:
    path = run_dir / "queue_status"
    if not path.is_file():
        return None, path
    status = path.read_text(encoding="utf-8").strip()
    return status or None, path


def _read_pid(
    run_dir: Path, candidates: tuple[str, ...]
) -> tuple[int | None, Path | None]:
    for name in candidates:
        path = run_dir / name
        if not path.is_file():
            continue
        value = path.read_text(encoding="utf-8").strip()
        try:
            pid = int(value)
        except ValueError as error:
            raise ValueError(f"invalid PID in {path}: {value!r}") from error
        if pid <= 0:
            raise ValueError(f"PID must be positive in {path}: {pid}")
        return pid, path
    return None, None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _last_nonempty_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), None)


def _record(
    *,
    method_id: str,
    stage: str,
    status: str,
    detail: str,
    source_path: Path,
) -> dict[str, str]:
    return {
        "method_id": method_id,
        "stage": stage,
        "status": status,
        "detail": detail,
        "source_path": str(source_path.resolve()),
    }


def _queue_snapshot(
    run_dir: Path,
    *,
    pid_candidates: tuple[str, ...],
    complete_validator: Callable[[], str],
) -> tuple[str, str, Path, dict[str, Any], str | None]:
    status_value, status_path = _read_status(run_dir)
    pid, pid_path = _read_pid(run_dir, pid_candidates)
    alive = None
    diagnostic = {
        "run_dir": str(run_dir.resolve()),
        "queue_status": status_value,
        "queue_status_path": str(status_path.resolve()),
        "pid": pid,
        "pid_path": str(pid_path.resolve()) if pid_path is not None else None,
        "pid_alive": alive,
        "last_idle_sample": _last_nonempty_line(run_dir / "idle_samples.log"),
    }
    if status_value is None:
        return "OPEN", "queue status is missing", status_path, diagnostic, None
    if status_value in COMPLETED_STATUSES:
        try:
            validation_detail = complete_validator()
        except (OSError, ValueError, KeyError) as error:
            message = f"complete status failed artifact validation: {error}"
            return "FAIL", message, status_path, diagnostic, message
        return "PASS", validation_detail, status_path, diagnostic, None
    if status_value.startswith("failed"):
        message = f"queue entered terminal failure state: {status_value}"
        return "FAIL", message, status_path, diagnostic, message
    if status_value in WAITING_STATUSES:
        mapped = "QUEUED"
    elif status_value in RUNNING_STATUSES or status_value.startswith("running_"):
        mapped = "RUNNING"
    else:
        message = f"unknown queue status: {status_value}"
        return "FAIL", message, status_path, diagnostic, message
    if pid is None:
        message = f"{status_value} but queue PID is missing"
        return "FAIL", message, status_path, diagnostic, message
    alive = _pid_alive(pid)
    diagnostic["pid_alive"] = alive
    if alive is not True:
        message = f"{status_value} but queue PID {pid} is not alive"
        return "FAIL", message, status_path, diagnostic, message
    detail = f"queue_status={status_value}; pid={pid} alive"
    if diagnostic["last_idle_sample"]:
        detail += f"; last_idle_sample={diagnostic['last_idle_sample']}"
    return mapped, detail, status_path, diagnostic, None


def _validate_oasis_complete(run_dir: Path) -> str:
    run, _ = parse_oasis(run_dir / "result.json")
    if run["scope"] != "formal_50x5":
        raise ValueError(f"OASIS result is not formal 50x5: {run['scope']}")
    return (
        f"strict formal result passed: {run['quality_correct']}/"
        f"{run['quality_scored']} correct"
    )


def _validate_causalmem_complete(run_dir: Path) -> str:
    run, _ = parse_causalmem(run_dir / "official" / "metrics.json")
    if run["scope"] != "formal_50x5":
        raise ValueError(f"CausalMem result is not formal 50x5: {run['scope']}")
    return (
        f"strict formal result passed: {run['quality_correct']}/"
        f"{run['quality_scored']} correct"
    )


def _validate_stc_complete(run_dir: Path) -> str:
    observed = set()
    for relative in (Path("rekv/result.json"), Path("rekv_stc/result.json")):
        run, rows = parse_stc(run_dir / relative)
        observed.add(run["variant"])
        if len(rows) != 3:
            raise ValueError(f"unexpected STC stage count in {relative}: {len(rows)}")
    if observed != {"rekv", "stc"}:
        raise ValueError(f"STC result pair is incomplete: {sorted(observed)}")
    return "strict ReKV and ReKV+STC stage result pair passed"


def _validate_streamingtom_complete(run_dir: Path) -> str:
    observed = set()
    for relative in (
        Path("ctr/summary.json"),
        Path("oqm_write/summary.json"),
        Path("oqm_select/summary.json"),
    ):
        run, rows = parse_streamingtom(run_dir / relative)
        observed.add(run["variant"])
        if len(rows) != 2:
            raise ValueError(
                f"unexpected StreamingTOM timing-basis count in {relative}: {len(rows)}"
            )
    if observed != set(STREAMINGTOM_SPECS):
        raise ValueError(f"StreamingTOM core triplet is incomplete: {sorted(observed)}")
    return "strict CTR/OQM core summary triplet passed"


def _collect_oasis(
    run_dir: Path,
    metadata_path: Path,
) -> tuple[dict[str, str], dict[str, Any], str | None]:
    status, detail, source, diagnostic, error = _queue_snapshot(
        run_dir,
        pid_candidates=("queue_pid", "pid"),
        complete_validator=lambda: _validate_oasis_complete(run_dir),
    )
    output_path = run_dir / f"{metadata_path.stem}_output.json"
    if output_path.is_file():
        try:
            progress = summarize_official_output(
                output_path,
                metadata_path=metadata_path,
            )
        except (OSError, ValueError, KeyError) as progress_error:
            status = "FAIL"
            error = f"OASIS atomic-prefix validation failed: {progress_error}"
            detail = error
        else:
            diagnostic["progress"] = progress
            partial = progress["accuracy_on_scored"]
            partial_text = "n/a" if partial is None else f"{100.0 * partial:.2f}%"
            progress_detail = (
                f"{progress['completed_videos']}/{progress['expected_videos']} videos; "
                f"{progress['completed_questions']}/{progress['expected_questions']} "
                f"questions; {progress['correct']}/{progress['scored_questions']} "
                f"scored correct ({partial_text} partial, diagnostic only)"
            )
            if status in {"RUNNING", "QUEUED"}:
                detail = f"{progress_detail}; {detail}"
                source = output_path
    return (
        _record(
            method_id="oasis",
            stage="official_quality",
            status=status,
            detail=detail,
            source_path=source,
        ),
        diagnostic,
        error,
    )


def _collect_queue_method(
    *,
    method_id: str,
    stage: str,
    run_dir: Path,
    validator: Callable[[Path], str],
) -> tuple[dict[str, str], dict[str, Any], str | None]:
    status, detail, source, diagnostic, error = _queue_snapshot(
        run_dir,
        pid_candidates=("pid", "queue_pid"),
        complete_validator=lambda: validator(run_dir),
    )
    return (
        _record(
            method_id=method_id,
            stage=stage,
            status=status,
            detail=detail,
            source_path=source,
        ),
        diagnostic,
        error,
    )


def _validate_streamingtom_preflight(directory: Path) -> str:
    observed = set()
    for method, spec in STREAMINGTOM_SPECS.items():
        path = directory / f"{method}.json"
        payload = _load_object(path)
        source = payload.get("source")
        expected = {
            "format_version": 2,
            "evidence_tier": "official_core_gpu_microbenchmark",
            "method": method,
            "frames": spec["frames"],
            "layers": 28,
            "warmup": 20,
            "repeat": 200,
            "dtype": "float16",
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"StreamingTOM preflight mismatch in {path}: {mismatches}")
        if (
            not isinstance(source, dict)
            or source.get("name") != "streamingtom"
            or source.get("commit") != STREAMINGTOM_COMMIT
            or source.get("code_clean") is not True
        ):
            raise ValueError(f"StreamingTOM preflight source audit failed: {source}")
        observed.add(method)
    if observed != set(STREAMINGTOM_SPECS):
        raise ValueError(f"StreamingTOM preflight triplet is incomplete: {observed}")
    return "pinned CTR/OQM dry-run preflight triplet passed"


def collect_status(
    *,
    oasis_run: Path | None,
    oasis_metadata: Path | None,
    stc_run: Path | None,
    causalmem_run: Path | None,
    streamingtom_run: Path | None,
    streamingtom_preflight_dir: Path | None,
    observed_at: str | None,
) -> dict[str, Any]:
    if oasis_run is not None and oasis_metadata is None:
        raise ValueError("--oasis-metadata is required with --oasis-run")
    records = []
    diagnostics = {}
    validation_errors = []

    def append_result(
        method_id: str,
        result: tuple[dict[str, str], dict[str, Any], str | None],
    ) -> None:
        record, diagnostic, error = result
        records.append(record)
        diagnostics[method_id] = diagnostic
        if error is not None:
            validation_errors.append({"method_id": method_id, "error": error})

    if oasis_run is not None and oasis_metadata is not None:
        append_result("oasis", _collect_oasis(oasis_run, oasis_metadata))
    if stc_run is not None:
        append_result(
            "stc",
            _collect_queue_method(
                method_id="stc",
                stage="official_latency",
                run_dir=stc_run,
                validator=_validate_stc_complete,
            ),
        )
    if causalmem_run is not None:
        append_result(
            "causalmem",
            _collect_queue_method(
                method_id="causalmem",
                stage="official_quality",
                run_dir=causalmem_run,
                validator=_validate_causalmem_complete,
            ),
        )
    if streamingtom_run is not None:
        append_result(
            "streamingtom",
            _collect_queue_method(
                method_id="streamingtom",
                stage="official_latency",
                run_dir=streamingtom_run,
                validator=_validate_streamingtom_complete,
            ),
        )
    if streamingtom_preflight_dir is not None:
        try:
            detail = _validate_streamingtom_preflight(streamingtom_preflight_dir)
            status = "PASS"
            error = None
        except (OSError, ValueError, KeyError) as validation_error:
            status = "FAIL"
            error = f"StreamingTOM preflight validation failed: {validation_error}"
            detail = error
            validation_errors.append(
                {
                    "method_id": "streamingtom",
                    "stage": "runtime_preflight",
                    "error": error,
                }
            )
        records.append(
            _record(
                method_id="streamingtom",
                stage="runtime_preflight",
                status=status,
                detail=detail,
                source_path=streamingtom_preflight_dir,
            )
        )
        diagnostics["streamingtom_preflight"] = {
            "directory": str(streamingtom_preflight_dir.resolve()),
            "valid": error is None,
        }
    if not records:
        raise ValueError("at least one runtime run or preflight path is required")
    records.sort(key=lambda row: (row["method_id"], row["stage"]))
    return {
        "format_version": 1,
        "observed_at": _observed_at(observed_at),
        "records": records,
        "diagnostics": diagnostics,
        "validation_errors": validation_errors,
    }


def main() -> int:
    args = parse_args()
    payload = collect_status(
        oasis_run=args.oasis_run.resolve() if args.oasis_run is not None else None,
        oasis_metadata=(
            args.oasis_metadata.resolve() if args.oasis_metadata is not None else None
        ),
        stc_run=args.stc_run.resolve() if args.stc_run is not None else None,
        causalmem_run=(
            args.causalmem_run.resolve() if args.causalmem_run is not None else None
        ),
        streamingtom_run=(
            args.streamingtom_run.resolve()
            if args.streamingtom_run is not None
            else None
        ),
        streamingtom_preflight_dir=(
            args.streamingtom_preflight_dir.resolve()
            if args.streamingtom_preflight_dir is not None
            else None
        ),
        observed_at=args.observed_at,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
