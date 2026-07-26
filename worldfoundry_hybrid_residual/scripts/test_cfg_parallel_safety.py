#!/usr/bin/env python3
"""Static regressions for correctness-critical CFG generation invariants."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("generate_wan_cfg_parallel.py")


def function_node(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def attribute_calls(node: ast.AST) -> set[str]:
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def test_cfg_path_never_mutates_scheduler_input_in_place() -> None:
    node = function_node("generate_cfg_parallel")
    assert "copy_" not in attribute_calls(node)


def test_sequential_reference_avoids_pipeline_generate_collectives() -> None:
    node = function_node("generate_sequential_reference")
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        assert child.func.attr != "generate"


def test_both_paths_rebind_scheduler_output() -> None:
    for name in ("generate_cfg_parallel", "generate_sequential_reference"):
        node = function_node(name)
        assignments = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "latent" for target in child.targets)
        ]
        assert any(
            isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Attribute)
            and item.value.func.attr == "squeeze"
            for item in assignments
        )
