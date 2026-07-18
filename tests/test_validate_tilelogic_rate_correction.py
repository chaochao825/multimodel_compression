from __future__ import annotations

from scripts.validate_tilelogic_rate_correction import (
    ALLOWED_CHANGED_COMPONENTS,
    _component_index,
    _non_rate_variant,
)


def test_rate_correction_signature_excludes_only_rate_payload() -> None:
    variant = {
        "method": "base_vq",
        "retention_rate": 0.125,
        "feature_nmse": 0.5,
        "rate": {"effective_bits": 10.0},
        "rate_components": [
            {"name": "base_codebook", "scope": "shared", "bits": 32}
        ],
    }
    assert _non_rate_variant(variant) == {
        "method": "base_vq",
        "retention_rate": 0.125,
        "feature_nmse": 0.5,
    }
    assert _component_index(variant) == {("base_codebook", "shared"): 32}
    assert "base_codebook" in ALLOWED_CHANGED_COMPONENTS
    assert "feature_nmse" not in ALLOWED_CHANGED_COMPONENTS
