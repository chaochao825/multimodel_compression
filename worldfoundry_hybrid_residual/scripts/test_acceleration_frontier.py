from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_acceleration_frontier import (  # noqa: E402
    amdahl_speedup,
    cfg_parallel_speedup,
    expected_speculative_progress,
    fit_step_cost,
    picard_speedup,
    speculative_speedup,
)


def test_amdahl_and_cfg_bounds() -> None:
    assert math.isclose(amdahl_speedup(0.5, 2.0), 4.0 / 3.0)
    speedup = cfg_parallel_speedup(1.0, 0.5, 20, communication_fraction=0.0)
    assert 1.0 < speedup < 2.0
    assert cfg_parallel_speedup(1.0, 0.5, 20, 0.1) < speedup


def test_step_fit_recovers_fixed_and_variable_cost() -> None:
    frame = pd.DataFrame(
        {
            "sampling_steps": [4, 8, 12, 20],
            "seconds_including_text_and_vae": [2.0, 3.0, 4.0, 6.0],
        }
    )
    fixed, per_step, r2 = fit_step_cost(frame)
    assert math.isclose(fixed, 1.0, rel_tol=1e-10)
    assert math.isclose(per_step, 0.25, rel_tol=1e-10)
    assert math.isclose(r2, 1.0, rel_tol=1e-10)


def test_linear_verification_cannot_accelerate_speculation() -> None:
    assert math.isclose(expected_speculative_progress(0.0, 4), 1.0)
    assert math.isclose(expected_speculative_progress(0.5, 2), 1.5)
    assert math.isclose(expected_speculative_progress(1.0, 4), 4.0)
    assert math.isclose(speculative_speedup(1.0, 4, verification_ratio=4.0), 1.0)
    assert speculative_speedup(0.95, 4, verification_ratio=4.0) < 1.0


def test_two_gpu_full_window_picard_needs_one_iteration_to_win() -> None:
    assert math.isclose(picard_speedup(20, 20, 2, 1), 2.0)
    assert math.isclose(picard_speedup(20, 20, 2, 2), 1.0)
    assert picard_speedup(20, 20, 2, 3) < 1.0


def test_rmt_edge_and_subspace_overlap() -> None:
    torch = pytest.importorskip("torch")
    from probe_defect_rmt import mp_upper_edge, subspace_overlap

    assert math.isclose(mp_upper_edge(1.0, features=100, samples=400), 2.25)
    identity = torch.eye(4)
    assert math.isclose(subspace_overlap(identity[:, :2], identity[:, :2]), 1.0)
    assert math.isclose(subspace_overlap(identity[:, :2], identity[:, 2:]), 0.0)
