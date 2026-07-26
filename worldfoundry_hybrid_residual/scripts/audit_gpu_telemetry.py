#!/usr/bin/env python3
"""Audit nvidia-smi telemetry for foreign GPU process overlap."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True)
class Snapshot:
    timestamp: str
    processes: tuple[ComputeProcess, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--used-gpu-uuid", action="append", default=[])
    parser.add_argument("--allowed-process-prefix", action="append", required=True)
    parser.add_argument(
        "--foreign-events",
        type=Path,
        help="Optional ancestry-based foreign-process events emitted by the runner.",
    )
    parser.add_argument("--require-exclusive", action="store_true")
    return parser.parse_args()


def parse_int(value: str) -> int:
    text = value.strip().removesuffix(" MiB")
    return int(text)


def read_snapshots(path: Path) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    timestamp: str | None = None
    processes: list[ComputeProcess] = []

    def flush() -> None:
        nonlocal timestamp, processes
        if timestamp is not None:
            snapshots.append(Snapshot(timestamp, tuple(processes)))
        timestamp = None
        processes = []

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("timestamp="):
            flush()
            timestamp = line.partition("=")[2].strip()
            continue
        fields = [field.strip() for field in line.split(",")]
        if timestamp is None or len(fields) < 4 or not fields[0].startswith("GPU-"):
            continue
        try:
            processes.append(
                ComputeProcess(
                    gpu_uuid=fields[0],
                    pid=parse_int(fields[1]),
                    process_name=fields[2],
                    used_memory_mib=parse_int(fields[3]),
                )
            )
        except ValueError:
            continue
    flush()
    if not snapshots:
        raise ValueError(f"no timestamped snapshots found in {path}")
    return snapshots


def estimate_interval_seconds(snapshots: list[Snapshot]) -> float | None:
    parsed: list[datetime] = []
    for snapshot in snapshots:
        try:
            parsed.append(datetime.fromisoformat(snapshot.timestamp))
        except ValueError:
            return None
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(parsed, parsed[1:])
        if right > left
    ]
    return statistics.median(deltas) if deltas else None


def audit_snapshots(
    snapshots: list[Snapshot],
    used_gpu_uuids: set[str],
    allowed_prefixes: tuple[str, ...],
) -> dict[str, object]:
    foreign_snapshots = 0
    own_snapshots = 0
    foreign_processes: dict[tuple[str, int, str], ComputeProcess] = {}
    own_processes: dict[tuple[str, int, str], ComputeProcess] = {}

    for snapshot in snapshots:
        selected = [
            process
            for process in snapshot.processes
            if not used_gpu_uuids or process.gpu_uuid in used_gpu_uuids
        ]
        own = [
            process
            for process in selected
            if any(process.process_name.startswith(prefix) for prefix in allowed_prefixes)
        ]
        foreign = [process for process in selected if process not in own]
        own_snapshots += bool(own)
        foreign_snapshots += bool(foreign)
        for process in own:
            own_processes[(process.gpu_uuid, process.pid, process.process_name)] = process
        for process in foreign:
            foreign_processes[(process.gpu_uuid, process.pid, process.process_name)] = process

    interval = estimate_interval_seconds(snapshots)
    timing_valid = foreign_snapshots == 0
    return {
        "timing_valid": timing_valid,
        "snapshots_total": len(snapshots),
        "snapshots_with_owned_process": own_snapshots,
        "snapshots_with_foreign_process": foreign_snapshots,
        "foreign_snapshot_fraction": foreign_snapshots / len(snapshots),
        "estimated_foreign_overlap_seconds": (
            foreign_snapshots * interval if interval is not None else None
        ),
        "median_monitor_interval_seconds": interval,
        "used_gpu_uuids": sorted(used_gpu_uuids),
        "allowed_process_prefixes": list(allowed_prefixes),
        "owned_processes": [
            process.__dict__ for process in sorted(
                own_processes.values(), key=lambda item: (item.gpu_uuid, item.pid)
            )
        ],
        "foreign_processes": [
            process.__dict__ for process in sorted(
                foreign_processes.values(), key=lambda item: (item.gpu_uuid, item.pid)
            )
        ],
    }


def main() -> None:
    args = parse_args()
    snapshots = read_snapshots(args.telemetry.resolve())
    audit = audit_snapshots(
        snapshots,
        set(args.used_gpu_uuid),
        tuple(args.allowed_process_prefix),
    )
    ancestry_events: list[str] = []
    if args.foreign_events is not None and args.foreign_events.is_file():
        ancestry_events = [
            line
            for line in args.foreign_events.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
    if ancestry_events:
        audit["timing_valid"] = False
    result = {
        "stage": args.stage,
        "telemetry": str(args.telemetry.resolve()),
        **audit,
        "ancestry_foreign_event_count": len(ancestry_events),
        "ancestry_foreign_events": ancestry_events[:100],
        "interpretation": (
            "Latency is admissible only when timing_valid is true. Correctness and "
            "quality artifacts may still be used when timing is contaminated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.require_exclusive and not bool(result["timing_valid"]):
        sys.exit(2)


if __name__ == "__main__":
    main()
