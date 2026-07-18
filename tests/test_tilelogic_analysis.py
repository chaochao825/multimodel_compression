from __future__ import annotations

import math
import unittest

from scripts.review_tilelogic_rvq import _decision_boolean_audit
from tilespec_ex.tilelogic_analysis import (
    quality_guardrail,
    relative_increase,
    status_from_evidence,
    strict_frontier_extension,
    topk_recall,
)


class TileLogicAnalysisTest(unittest.TestCase):
    def test_strict_frontier_extension_requires_lower_envelope_improvement(self) -> None:
        baseline = [
            {"rate": 0.5, "error": 0.2},
            {"rate": 1.0, "error": 0.1},
        ]
        self.assertTrue(
            strict_frontier_extension(
                {"rate": 0.25, "error": 0.8},
                baseline,
                rate_key="rate",
                metric_key="error",
            )
        )
        self.assertTrue(
            strict_frontier_extension(
                {"rate": 0.75, "error": 0.15},
                baseline,
                rate_key="rate",
                metric_key="error",
            )
        )
        self.assertFalse(
            strict_frontier_extension(
                {"rate": 0.75, "error": 0.25},
                baseline,
                rate_key="rate",
                metric_key="error",
            )
        )
        self.assertFalse(
            strict_frontier_extension(
                {"rate": 1.0, "error": 0.1},
                baseline,
                rate_key="rate",
                metric_key="error",
            )
        )

    def test_topk_recall_is_deterministic_under_ties(self) -> None:
        self.assertEqual(topk_recall([3, 3, 1, 0], [3, 2, 4, 0], 2), 0.5)

    def test_quality_guardrail_applies_all_three_limits(self) -> None:
        baseline = {"gqa": 0.5, "textvqa": 0.4, "chartqa": 0.3}
        passing = quality_guardrail(
            {"gqa": 0.496, "textvqa": 0.4, "chartqa": 0.3},
            baseline,
            candidate_nll=1.009,
            baseline_nll=1.0,
            candidate_nmse=0.101,
            baseline_nmse=0.1,
        )
        self.assertTrue(passing["pass"])
        failing = quality_guardrail(
            {"gqa": 0.494, "textvqa": 0.4, "chartqa": 0.3},
            baseline,
            candidate_nll=1.0,
            baseline_nll=1.0,
            candidate_nmse=0.1,
            baseline_nmse=0.1,
        )
        self.assertFalse(failing["score_pass"])
        self.assertFalse(failing["pass"])

    def test_relative_increase_and_status_handle_boundaries(self) -> None:
        self.assertAlmostEqual(relative_increase(1.01, 1.0), 0.01)
        self.assertEqual(relative_increase(0.0, 0.0), 0.0)
        self.assertTrue(math.isinf(relative_increase(1.0, 0.0)))
        self.assertEqual(status_from_evidence(available=False, passed=True), "INCONCLUSIVE")
        self.assertEqual(status_from_evidence(available=True, passed=False), "FAIL")

    def test_decision_audit_rejects_wrong_question_3_guardrail_baseline(self) -> None:
        def guardrail(candidate: str, baseline: str) -> dict[str, object]:
            return {"candidate": candidate, "baseline": baseline, "pass": True}

        decisions = {
            "questions": [
                {
                    "id": 1,
                    "status": "FAIL",
                    "pass": False,
                    "evidence": {
                        "feature_frontier_extension_by_rate": {"0.125": False},
                        "nll_frontier_extension_by_rate": {"0.125": False},
                        "guardrails": [guardrail("base_vq", "base_scalar_quant")],
                    },
                },
                {
                    "id": 2,
                    "status": "FAIL",
                    "pass": False,
                    "evidence": {
                        "by_rate": [
                            {
                                "pass": False,
                                "guardrail_vs_base": guardrail(
                                    "base_vq_residual_rvq", "base_vq"
                                ),
                                "guardrail_vs_unweighted": guardrail(
                                    "base_vq_residual_rvq",
                                    "base_vq_residual_rvq_unweighted",
                                ),
                            }
                        ]
                    },
                },
                {
                    "id": 3,
                    "status": "PASS",
                    "pass": True,
                    "evidence": {
                        "by_rate": [
                            {
                                "spearman_delta": 0.03,
                                "topk_recall_delta": 0.03,
                                "guardrail": guardrail(
                                    "base_vq_mlp_router", "tile_energy_exception"
                                ),
                            }
                            for _ in range(2)
                        ]
                    },
                },
                {
                    "id": 4,
                    "status": "PASS",
                    "pass": True,
                    "evidence": {
                        "by_rate": [
                            {
                                "spearman_improvement_retained": 0.9,
                                "topk_improvement_retained": 0.9,
                                "guardrail": guardrail(
                                    "base_vq_logic_router", "base_vq_mlp_router"
                                ),
                            }
                            for _ in range(2)
                        ]
                    },
                },
                {
                    "id": 5,
                    "status": "PASS",
                    "pass": True,
                    "evidence": {
                        "by_rate": [
                            {
                                "all_paired_medians_lower": True,
                                "guardrail": guardrail(
                                    "logic_router_fixed_slots",
                                    "base_vq_logic_router",
                                ),
                            }
                            for _ in range(2)
                        ]
                    },
                },
                {
                    "id": 6,
                    "status": "PASS",
                    "pass": True,
                    "evidence": {
                        "by_rate": [
                            {
                                "feature_frontier_extension": True,
                                "nll_frontier_extension": True,
                                "material_improvement": True,
                                "guardrail": guardrail(
                                    "logic_router_fixed_slots_exact_fallback",
                                    "logic_router_fixed_slots",
                                ),
                            }
                        ]
                    },
                },
            ],
            "all_questions_pass": False,
            "aggregate_positive_claim_allowed": False,
        }
        errors = _decision_boolean_audit(decisions)
        self.assertTrue(
            any(
                "question 3 guardrail baseline" in error
                and "base_vq_residual_rvq" in error
                for error in errors
            )
        )


if __name__ == "__main__":
    unittest.main()
