from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "figures" / "plot_query_fixed_progressive_exact_pages.py"
ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "query_fixed_progressive_exact_pages_exposed_v1"
)
SPEC = importlib.util.spec_from_file_location("plot_progressive_exact", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load progressive exact-page plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_progressive_exact_tables_reproduce_registered_result() -> None:
    rows, summaries = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries)
    spatial = tables["spatial"]
    exact = spatial[
        spatial["selector"].eq("exact_mass") & spatial["exact_fraction"].eq(0.25)
    ].iloc[0]
    assert exact["selected_visual_mass_mean"] == pytest.approx(0.6180613992942704)
    assert exact["visual_mean"] == pytest.approx(0.22268590610474348)
    assert exact["visual_p95"] == pytest.approx(0.3598748862743378)

    layer = tables["layer"].set_index(["selector", "layer_index"])
    assert layer.loc[("quest_box_bound", 27), "visual_mean"] == pytest.approx(
        0.5971849571168423
    )
    assert layer.loc[("exact_mass", 27), "bound_log10_looseness"] == pytest.approx(
        16.336284637451172
    )


def test_progressive_exact_figure_renders_all_formats(tmp_path: Path) -> None:
    rows, summaries = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries)
    MODULE.write_tables(tables, tmp_path)
    MODULE.render(tables, tmp_path)

    for suffix in ("png", "pdf", "svg"):
        output = tmp_path / f"query_fixed_progressive_exact_pages.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 1_000

    svg = (tmp_path / "query_fixed_progressive_exact_pages.svg").read_text(
        encoding="utf-8"
    )
    assert all(line == line.rstrip() for line in svg.splitlines())
