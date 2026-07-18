from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from audit_streaming_baseline_sources import audit_source  # noqa: E402
from smoke_external_baseline_modules import (  # noqa: E402
    _output_tail,
    run_probe,
)


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class StreamingBaselineSourceAuditTest(unittest.TestCase):
    def test_clean_pinned_checkout_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "demo"
            checkout.mkdir()
            _git("init", cwd=checkout)
            _git("config", "user.email", "audit@example.invalid", cwd=checkout)
            _git("config", "user.name", "Audit Test", cwd=checkout)
            (checkout / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git("add", "entry.py", cwd=checkout)
            _git("commit", "-m", "fixture", cwd=checkout)
            commit = _git("rev-parse", "HEAD", cwd=checkout)

            result = audit_source(
                {
                    "name": "demo",
                    "local_directory": "demo",
                    "commit": commit,
                    "required_paths": ["entry.py"],
                },
                external_root=root,
                compile_source=True,
            )

            self.assertTrue(result["audit_passed"])
            self.assertTrue(result["commit_matches"])
            self.assertTrue(result["worktree_clean"])
            self.assertEqual(result["compileall_status"], "passed")

    def test_unavailable_and_missing_sources_do_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unavailable = audit_source(
                {
                    "name": "paper-only",
                    "local_directory": None,
                    "commit": None,
                    "required_paths": [],
                },
                external_root=root,
                compile_source=False,
            )
            missing = audit_source(
                {
                    "name": "missing",
                    "local_directory": "missing",
                    "commit": "0" * 40,
                    "required_paths": ["entry.py"],
                },
                external_root=root,
                compile_source=False,
            )

            self.assertEqual(unavailable["checkout_status"], "not_available")
            self.assertFalse(unavailable["audit_passed"])
            self.assertEqual(missing["checkout_status"], "missing")
            self.assertFalse(missing["audit_passed"])

    def test_module_smoke_reports_missing_checkout_and_decodes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_probe(
                "stc",
                external_root=Path(directory),
                python=sys.executable,
                timeout_seconds=1.0,
            )
        self.assertEqual(result["status"], "missing_checkout")
        self.assertFalse(result["passed"])
        self.assertEqual(_output_tail(b"abc"), "abc")


if __name__ == "__main__":
    unittest.main()
