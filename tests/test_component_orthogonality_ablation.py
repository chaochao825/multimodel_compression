from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import component_orthogonality_ablation as coa
from hybrid_attention_decomposition import decompose


def test_frobenius_cosine_distinguishes_disjoint_and_overlapping_support() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 0.0]])
    disjoint = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    overlapping = np.asarray([[2.0, 0.0], [0.0, 0.0]])
    assert coa.frobenius_cosine(left, disjoint) == 0.0
    assert np.isclose(coa.frobenius_cosine(left, overlapping), 1.0)
    assert coa.support_jaccard(left, disjoint, 1e-8) == 0.0
    assert coa.support_jaccard(left, overlapping, 1e-8) == 1.0


def test_factorial_shapley_values_satisfy_efficiency() -> None:
    attention = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    components = {
        "sink": np.asarray([[0.7, 0.0], [0.0, 0.0]]),
        "global_svd": np.asarray([[0.0, 0.3], [0.0, 0.0]]),
        "local_cyclic": np.asarray([[0.0, 0.0], [0.2, 0.0]]),
        "sparse_routing": np.asarray([[0.0, 0.0], [0.0, 0.8]]),
    }
    _rows, utilities, shapley, interactions = coa.factorial_ablation(attention, components)
    full_mask = (1 << len(coa.COMPONENT_NAMES)) - 1
    assert np.isclose(sum(shapley.values()), utilities[full_mask] - utilities[0], atol=1e-12)
    assert len(interactions) == 24
    for left in range(len(coa.COMPONENT_NAMES)):
        for right in range(left + 1, len(coa.COMPONENT_NAMES)):
            pair = [
                row
                for row in interactions
                if row["left"] == coa.COMPONENT_NAMES[left]
                and row["right"] == coa.COMPONENT_NAMES[right]
            ]
            assert len(pair) == 4
            assert {row["background_size"] for row in pair} == {0, 1, 2}


def test_current_order_reproduces_saved_balanced_components() -> None:
    meta_path = ROOT / "remote_logs" / "hybrid_attention_decomposition_20260704.json"
    npz_path = ROOT / "remote_logs" / "hybrid_attention_decomposition_20260704.npz"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    for item in metadata["items"]:
        balanced = coa.load_balanced_config(item)
        key = item["key"]
        attention = arrays[f"{key}_attention"].astype(np.float64)
        result = decompose(
            attention,
            item["grid_shape"],
            int(balanced["sink_k"]),
            int(balanced["rank"]),
            int(balanced["radius"]),
            int(balanced["sparse_k"]),
        )
        error, components, approx = coa.sequential_fit(
            attention,
            item["grid_shape"],
            np.asarray(result["sink_cols"], dtype=np.int64),
            int(balanced["rank"]),
            int(balanced["radius"]),
            int(balanced["sparse_k"]),
            coa.CURRENT_ORDER,
        )
        assert np.isclose(error, float(result["relative_fro_error"]), atol=1e-12)
        for name in coa.COMPONENT_NAMES:
            assert np.allclose(components[name], result[name], atol=1e-12)
        assert np.allclose(approx, result["hybrid"], atol=1e-12)
        assert np.allclose(
            components["sink"] + components["global_svd"],
            arrays[f"{key}_sink_global_svd"],
            atol=1e-7,
        )
        assert np.allclose(components["local_cyclic"], arrays[f"{key}_local_cyclic"], atol=1e-7)
        assert np.allclose(components["sparse_routing"], arrays[f"{key}_sparse_routing"], atol=1e-7)
