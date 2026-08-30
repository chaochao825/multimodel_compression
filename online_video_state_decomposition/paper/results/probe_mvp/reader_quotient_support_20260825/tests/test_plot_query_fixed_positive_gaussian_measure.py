from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "figures" / "plot_query_fixed_positive_gaussian_measure.py"
ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "query_fixed_positive_gaussian_exposed_v1_repair4"
)
SPEC = importlib.util.spec_from_file_location("plot_positive_gaussian", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load positive Gaussian plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_positive_gaussian_tables_reproduce_registered_result() -> None:
    rows, summaries = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries)
    best = tables["eligible"].iloc[0]
    assert best["rank"] == 0
    assert best["topology"] == "spatial_7x7"
    assert best["selector"] == "oracle_local"
    assert best["active_read_ratio"] == pytest.approx(3.69811320754717)
    assert best["visual_mean"] == pytest.approx(0.15504768242438635)
    assert best["visual_p95"] == pytest.approx(0.25276239067316053)

    layer = tables["layer"].set_index("layer_index")
    assert layer.loc[0, "visual_mean"] == pytest.approx(0.047098382686575214)
    assert layer.loc[27, "visual_worst"] == pytest.approx(0.2960835099220276)


def test_positive_gaussian_figure_renders_all_formats(tmp_path: Path) -> None:
    rows, summaries = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries)
    MODULE.write_tables(tables, tmp_path)
    MODULE.render(tables, tmp_path)

    for suffix in ("png", "pdf", "svg"):
        output = tmp_path / f"query_fixed_positive_gaussian_measure.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 1_000

    svg = (tmp_path / "query_fixed_positive_gaussian_measure.svg").read_text(
        encoding="utf-8"
    )
    assert all(line == line.rstrip() for line in svg.splitlines())
