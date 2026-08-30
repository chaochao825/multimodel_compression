from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "figures" / "plot_query_fixed_measure_remainder.py"
ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "query_fixed_measure_exposed_v2_repair1"
)
SPEC = importlib.util.spec_from_file_location("plot_query_fixed_measure", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load query-fixed measure plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_query_fixed_tables_reproduce_registered_aggregates() -> None:
    rows = MODULE.load_rows(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows)
    curve = tables["curve"]
    selected = curve[
        curve["method"].eq("exact_local_score") & curve["exact_group_count"].eq(196)
    ].iloc[0]
    assert selected["visual_mean"] == pytest.approx(0.03450376846982787)
    assert selected["full_mean"] == pytest.approx(0.004237773315657655)

    envelope = tables["envelope"]
    assert len(envelope) == 72
    assert envelope["visual_relative_l2"].mean() == pytest.approx(0.032518, abs=1e-6)


def test_query_fixed_figure_renders_all_formats(tmp_path: Path) -> None:
    rows = MODULE.load_rows(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows)
    MODULE.write_tables(tables, tmp_path)
    MODULE.render(rows, tables, tmp_path)

    for suffix in ("png", "pdf", "svg"):
        output = tmp_path / f"query_fixed_measure_remainder.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 1_000
