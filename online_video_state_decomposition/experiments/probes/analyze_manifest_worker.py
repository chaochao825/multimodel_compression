from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--analysis-name", default="analysis_flow")
    parser.add_argument("--layers", default="0,7,15,22,23")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64")
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--history-window", type=int, default=2)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-optical-flow", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "run_name" not in rows[0]:
        raise ValueError("manifest must contain run_name")
    return rows


def valid_existing(path: Path, *, optical_flow: bool) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("layers")) and (
            bool(data.get("optical_flow_enabled")) == optical_flow
        )
    except (OSError, json.JSONDecodeError):
        return False


def wait_for_export(
    run_dir: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[Path, Path]:
    npz_path = run_dir / "hidden.npz"
    metadata_path = run_dir / "metadata.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        # The exporter writes metadata only after np.savez_compressed returns.
        if npz_path.exists() and metadata_path.exists():
            return npz_path, metadata_path
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for export: {run_dir}")
        time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    rows = [
        row
        for index, row in enumerate(read_manifest(args.manifest))
        if index % args.num_workers == args.worker_index
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    analyzer = Path(__file__).with_name("analyze_hidden_sequence.py")
    optical_flow = not args.disable_optical_flow
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(args.threads)
    completed = []
    skipped = []
    failed = []
    for position, row in enumerate(rows, start=1):
        run_name = row["run_name"]
        run_dir = args.export_root / run_name
        analysis_dir = run_dir / args.analysis_name
        summary_path = analysis_dir / "analysis_summary.json"
        if not args.overwrite and valid_existing(
            summary_path,
            optical_flow=optical_flow,
        ):
            skipped.append(run_name)
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "skipped_existing",
                    }
                ),
                flush=True,
            )
            continue
        try:
            npz_path, metadata_path = wait_for_export(
                run_dir,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.wait_timeout_seconds,
            )
            command = [
                sys.executable,
                str(analyzer),
                "--npz",
                str(npz_path),
                "--metadata",
                str(metadata_path),
                "--out-dir",
                str(analysis_dir),
                "--layers",
                args.layers,
                "--ranks",
                args.ranks,
                "--max-shift",
                str(args.max_shift),
                "--local-radius",
                str(args.local_radius),
                "--block-size",
                str(args.block_size),
                "--history-window",
                str(args.history_window),
            ]
            if optical_flow:
                command.append("--use-optical-flow")
            started = time.monotonic()
            result = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path = run_dir / f"{args.analysis_name}_worker.log"
            log_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"analyzer exit {result.returncode}; see {log_path}"
                )
            completed.append(run_name)
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "completed",
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failed.append(
                {
                    "run": run_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "position": position,
                        "total": len(rows),
                        "run": run_name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
    summary = {
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "assigned": len(rows),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "optical_flow": optical_flow,
    }
    args.export_root.mkdir(parents=True, exist_ok=True)
    (
        args.export_root
        / f"{args.analysis_name}_worker_{args.worker_index}_summary.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
