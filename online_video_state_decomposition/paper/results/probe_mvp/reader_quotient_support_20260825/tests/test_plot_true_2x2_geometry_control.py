from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "plot_true_2x2_geometry_control.py"
)
SPEC = importlib.util.spec_from_file_location("plot_true_2x2_geometry", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load true-2x2 plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bootstrap_paired_is_deterministic_and_paired() -> None:
    flat = np.asarray([1.0, 2.0, 4.0, 8.0])
    spatial = np.asarray([0.5, 1.0, 2.0, 4.0])

    first = MODULE.bootstrap_paired(flat, spatial, seed=17, draws=1_000)
    second = MODULE.bootstrap_paired(flat, spatial, seed=17, draws=1_000)

    assert first == second
    assert first["mean_kl_ratio"] == 0.5
    assert first["mean_kl_delta"] == -1.875


def test_one_sided_sign_pvalue_counts_only_non_ties() -> None:
    assert MODULE.one_sided_sign_pvalue(4, 4) == 0.0625
    assert MODULE.one_sided_sign_pvalue(0, 0) == 1.0
