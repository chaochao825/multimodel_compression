from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "figures" / "plot_query_fixed_prototype_mixture.py"
ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "query_fixed_prototype_mixture_exposed_v1"
)
SPEC = importlib.util.spec_from_file_location("plot_prototype_mixture", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load prototype-mixture plotting module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prototype_tables_reproduce_registered_result() -> None:
    rows, summaries, diagnostics = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries, diagnostics)
    best = tables["frontier"].iloc[0]
    assert best["cluster_family"] == "key"
    assert best["prototype_count"] == 128
    assert best["selector"] == "oracle_local"
    assert best["active_read_ratio"] == pytest.approx(4.002343131436242)
    assert best["visual_mean"] == pytest.approx(0.07544140450449453)
    assert best["visual_p95"] == pytest.approx(0.13058750703930855)
    assert best["visual_worst"] == pytest.approx(0.13833704590797424)

    selector_gap = tables["selector_gap"].set_index(
        ["cluster_family", "prototype_count"]
    )
    assert selector_gap.loc[("key", 128), "visual_mean_penalty_ratio"] == pytest.approx(
        1.8350390218191777
    )
    sensitivity = tables["support_sensitivity"].set_index("selector")
    assert sensitivity.loc["oracle_residual_greedy", "visual_mean"] == pytest.approx(
        0.12170979952336186
    )
    assert sensitivity.loc["oracle_reverse_greedy", "visual_mean"] == pytest.approx(
        0.06163252276989321
    )


def test_prototype_figure_renders_all_formats(tmp_path: Path) -> None:
    rows, summaries, diagnostics = MODULE.load_data(ANALYSIS_DIR)
    tables = MODULE.build_tables(rows, summaries, diagnostics)
    MODULE.write_tables(tables, tmp_path)
    MODULE.render(tables, tmp_path)

    for suffix in ("png", "pdf", "svg"):
        output = tmp_path / f"query_fixed_prototype_mixture.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 1_000

    svg = (tmp_path / "query_fixed_prototype_mixture.svg").read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in svg.splitlines())
