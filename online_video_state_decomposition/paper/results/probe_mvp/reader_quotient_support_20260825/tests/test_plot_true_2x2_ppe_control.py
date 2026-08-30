from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "figures" / "plot_true_2x2_ppe_control.py"
)
SPEC = importlib.util.spec_from_file_location("plot_true_2x2_ppe", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load true-2x2 PPE plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ppe_bootstrap_is_deterministic_and_paired() -> None:
    baseline = np.asarray([1.0, 2.0, 4.0, 8.0])
    candidate = np.asarray([0.5, 1.0, 2.0, 4.0])

    first = MODULE.bootstrap_paired(baseline, candidate, seed=23, draws=1_000)
    second = MODULE.bootstrap_paired(baseline, candidate, seed=23, draws=1_000)

    assert first == second
    assert first["mean_kl_ratio"] == 0.5
    assert first["mean_kl_delta"] == -1.875


def test_ppe_sign_test_reports_adverse_win_count() -> None:
    assert MODULE.one_sided_sign_pvalue(4, 4) == 0.0625
    assert MODULE.one_sided_sign_pvalue(0, 0) == 1.0
