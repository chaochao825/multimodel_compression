from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-compile", action="store_true")
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def audit_source(
    source: dict[str, object],
    *,
    external_root: Path,
    compile_source: bool,
) -> dict[str, object]:
    result = dict(source)
    local_directory = source.get("local_directory")
    expected_commit = source.get("commit")
    if not local_directory or not expected_commit:
        result.update(
            {
                "checkout_status": "not_available",
                "commit_matches": False,
                "worktree_clean": False,
                "required_paths_present": False,
                "compileall_status": "not_run",
                "audit_passed": False,
            }
        )
        return result

    checkout = external_root / str(local_directory)
    if not (checkout / ".git").is_dir():
        result.update(
            {
                "checkout_status": "missing",
                "commit_matches": False,
                "worktree_clean": False,
                "required_paths_present": False,
                "compileall_status": "not_run",
                "audit_passed": False,
            }
        )
        return result

    revision = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    observed_commit = revision.stdout.strip() if revision.returncode == 0 else ""
    required_paths = [str(path) for path in source.get("required_paths", [])]
    missing_paths = [path for path in required_paths if not (checkout / path).is_file()]
    compile_status = "skipped"
    compile_warnings = ""
    if compile_source:
        with tempfile.TemporaryDirectory(prefix="baseline-pycache-") as cache:
            compile_env = os.environ.copy()
            compile_env["PYTHONPYCACHEPREFIX"] = cache
            compiled = _run(
                [sys.executable, "-m", "compileall", "-q", str(checkout)],
                cwd=checkout,
                env=compile_env,
            )
        compile_status = "passed" if compiled.returncode == 0 else "failed"
        compile_warnings = (compiled.stdout + compiled.stderr)[-4000:].strip()

    status = _run(["git", "status", "--porcelain"], cwd=checkout)
    commit_matches = observed_commit == str(expected_commit)
    worktree_clean = status.returncode == 0 and not status.stdout.strip()
    required_present = not missing_paths
    audit_passed = (
        commit_matches
        and worktree_clean
        and required_present
        and compile_status in {"passed", "skipped"}
    )
    result.update(
        {
            "checkout_status": "present",
            "observed_commit": observed_commit,
            "commit_matches": commit_matches,
            "worktree_clean": worktree_clean,
            "required_paths_present": required_present,
            "missing_paths": missing_paths,
            "compileall_status": compile_status,
            "compileall_warnings": compile_warnings,
            "audit_passed": audit_passed,
        }
    )
    return result


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        audit_source(
            source,
            external_root=args.external_root,
            compile_source=not args.skip_compile,
        )
        for source in manifest["sources"]
    ]
    payload = {
        "format_version": 1,
        "manifest": str(args.manifest),
        "external_root": str(args.external_root),
        "sources": rows,
        "available_checkouts_passed": all(
            row["audit_passed"]
            for row in rows
            if row["checkout_status"] == "present"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["available_checkouts_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
