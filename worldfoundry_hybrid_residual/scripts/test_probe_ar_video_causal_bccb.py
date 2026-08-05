import unittest

from probe_ar_video_causal_bccb import decide_gate, make_variants, select_variants


def protocol_fixture():
    return {
        "methods": {
            "primary_candidate": "primary",
            "primary_low_rank_residual_rank": 16,
            "variants": [
                {
                    "name": "primary",
                    "kernel_source": "record_qk_projection",
                    "spatial_structure": "periodic_bccb",
                    "query_groups": "capture_tiles",
                    "temporal_sharing": "frame_pair",
                    "exact_policy": "sink_recent",
                    "event_fraction": 0.05,
                }
            ],
        },
        "evaluation": {
            "dense_reference_parity_gate": 0.005,
            "aggregate_oracle_gate": 0.005,
            "worst_oracle_gate": 0.01,
            "aggregate_transfer_gate": 0.01,
            "worst_transfer_gate": 0.02,
            "minimum_primary_arithmetic_reduction": 1.5,
        },
    }


def summary(correction, aggregate, worst, reduction=1.6):
    return {
        "scope": "held_out",
        "method": "primary",
        "correction": correction,
        "rank": 16,
        "aggregate_relative_av_l2": aggregate,
        "worst_head_relative_av_l2": worst,
        "minimum_arithmetic_reduction": reduction,
    }


class VariantTest(unittest.TestCase):
    def test_variant_is_fully_bound(self):
        variants = make_variants(protocol_fixture())
        self.assertEqual(len(variants), 1)
        self.assertTrue(variants[0].periodic)
        self.assertEqual(variants[0].query_groups, "capture_tiles")
        self.assertEqual(select_variants(variants, ["primary"]), variants)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            select_variants(make_variants(protocol_fixture()), ["missing"])


class GateTest(unittest.TestCase):
    def test_incomplete_capture_cannot_pass(self):
        gate = decide_gate(
            protocol_fixture(),
            [],
            {"complete": False, "worst_dense_reference_parity": 0.0},
        )
        self.assertEqual(gate["classification"], "incomplete")

    def test_capacity_failure_stops_recurrent_bccb(self):
        summaries = [
            summary("adaptive_rank_oracle", 0.02, 0.04),
            summary("frozen_calibration_basis_oracle_coefficients", 0.03, 0.05),
        ]
        gate = decide_gate(
            protocol_fixture(),
            summaries,
            {"complete": True, "worst_dense_reference_parity": 0.001},
        )
        self.assertEqual(gate["classification"], "null")
        self.assertEqual(gate["action"], "stop_recurrent_bccb_and_do_not_build_kernel")

    def test_representation_pass_opens_residual_width_gate(self):
        summaries = [
            summary("adaptive_rank_oracle", 0.004, 0.009),
            summary("frozen_calibration_basis_oracle_coefficients", 0.009, 0.019),
        ]
        gate = decide_gate(
            protocol_fixture(),
            summaries,
            {"complete": True, "worst_dense_reference_parity": 0.001},
        )
        self.assertEqual(gate["classification"], "representation_pass")


if __name__ == "__main__":
    unittest.main()
