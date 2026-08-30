from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "figures" / "plot_query_fixed_headwise_support_ceiling.py"
ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "query_fixed_headwise_exposed_v1"
)
SPEC = importlib.util.spec_from_file_location("plot_query_fixed_headwise", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load headwise support plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_headwise_tables_reproduce_registered_result() -> None:
    rows = MODULE.load_rows(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows)
    curve = tables["curve"]
    selected = curve[
        curve["method"].eq("headwise_exact_local") & curve["exact_group_count"].eq(196)
    ].iloc[0]
    assert selected["visual_mean"] == pytest.approx(0.012220126528215284)
    assert selected["visual_p95"] == pytest.approx(0.024931606557220223)

    improvement = tables["improvement"].set_index("layer_index")
    assert improvement.loc[0, "relative_improvement"] == pytest.approx(
        0.6924363609806532
    )
    assert improvement.loc[27, "relative_improvement"] == pytest.approx(
        0.628149875540603
    )


def test_headwise_figure_renders_all_formats(tmp_path: Path) -> None:
    rows = MODULE.load_rows(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows)
    MODULE.write_tables(tables, tmp_path)
    MODULE.render(rows, tables, tmp_path)

    for suffix in ("png", "pdf", "svg"):
        output = tmp_path / f"query_fixed_headwise_support_ceiling.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 1_000
