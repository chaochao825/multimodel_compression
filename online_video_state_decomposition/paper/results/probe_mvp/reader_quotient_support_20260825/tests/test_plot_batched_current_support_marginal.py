from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "plot_batched_current_support_marginal.py"
)
SPEC = importlib.util.spec_from_file_location("plot_batched_current_support", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load current-support plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_batch_interaction_uses_current_support_benefits() -> None:
    path_rows = [
        {
            "mode": mode,
            "sample_id": "sample",
            "selected_group_count": str(group_count),
            "candidate_kl": str(1.0 - 0.1 * index),
        }
        for mode in MODULE.MODES
        for index, group_count in enumerate(MODULE.GROUP_COUNTS)
    ]
    marginal_rows = []
    for mode in MODULE.MODES:
        for current_count in MODULE.GROUP_COUNTS[:-1]:
            marginal_rows.extend(
                {
                    "mode": mode,
                    "sample_id": "sample",
                    "current_group_count": str(current_count),
                    "selected_next_batch": "1",
                    "conditional_kl_benefit": "0.001",
                }
                for _ in range(49)
            )

    rows = MODULE.build_batch_interactions(path_rows, marginal_rows)

    assert len(rows) == 8
    assert abs(float(rows[0]["selected_benefit_sum"]) - 0.049) < 1e-12
    assert abs(float(rows[0]["batch_interaction_residual"]) + 0.051) < 1e-12
