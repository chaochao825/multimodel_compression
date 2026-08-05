import json
from pathlib import Path
import tempfile
import unittest

import torch

from ar_video_residual_memory_core import dense_attention
from probe_ar_video_residual_memory import (
    decide_gate,
    evaluate_capture,
    make_variants,
    primary_method,
    summarize_rows,
    validate_manifests,
)


def protocol_fixture():
    return {
        "protocol_id": "unit-protocol",
        "capture": {
            "frame_seq_len": 4,
            "local_attention_frames": 4,
            "query_tile_size": 2,
            "query_tiles_per_record": 1,
            "heads": [0, 1],
        },
        "methods": {
            "exact_sink_frames": 1,
            "exact_recent_frames": 1,
            "primary_candidate": {
                "summary_key_mode": "post_rope",
                "summary_group_count": 1,
                "event_tile_fraction": 0.0,
                "low_rank_residual_rank": 2,
            },
            "summary_key_modes": ["post_rope"],
            "summary_group_counts": [1],
            "event_tile_size": 2,
            "event_tile_fractions": [0.0],
            "low_rank_residual_ranks": [0, 2],
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


class ProbeEvaluatorTests(unittest.TestCase):
    def test_manifest_is_bound_to_protocol_and_declared_artifact(self):
        protocol = {
            "protocol_id": "unit-protocol",
            "model": {
                "code_commit": "abc",
                "generator_sha256": "generator-hash",
                "lora_sha256": "lora-hash",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "record.pt").write_bytes(b"record")
            manifest = {
                "protocol_id": "unit-protocol",
                "source_commit": "abc",
                "protocol_sha256": "protocol-hash",
                "runtime_config_sha256": "runtime-hash",
                "generator_sha256": "generator-hash",
                "lora_sha256": "lora-hash",
                "captures": ["record.pt"],
            }
            (root / "capture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            manifests, artifacts = validate_manifests(
                root, protocol, "protocol-hash"
            )
            self.assertEqual(len(manifests), 1)
            self.assertEqual(artifacts, [root / "record.pt"])
            with self.assertRaisesRegex(ValueError, "protocol_sha256 mismatch"):
                validate_manifests(root, protocol, "different-protocol-hash")

    def test_temporal_summary_is_exact_for_identical_framewise_kv(self):
        torch.manual_seed(4)
        protocol = protocol_fixture()
        spatial, frames, heads, dim = 4, 4, 2, 3
        per_position_key = torch.randn(spatial, heads, dim)
        per_position_value = torch.randn(spatial, heads, dim)
        key = per_position_key.repeat(frames, 1, 1)
        value = per_position_value.repeat(frames, 1, 1)
        query = torch.randn(2, heads, dim)
        target = dense_attention(query, key, value)
        payload = {
            "metadata": {
                "protocol_id": "unit-protocol",
                "prompt_id": "calib",
                "prompt_split": "calibration",
                "seed": 0,
                "layer": 0,
                "current_start_frame": 3,
                "denoising_call_index": 1,
                "denoising_timestep": 901,
                "head_indices": [0, 1],
            },
            "query": query,
            "key": key,
            "value": value,
            "dense_output": target,
        }
        rows = []
        results = evaluate_capture(
            payload,
            protocol,
            make_variants(protocol),
            torch.device("cpu"),
            query_chunk_size=2,
            rows=rows,
        )
        primary = [item for item in results if item.variant.method == primary_method(protocol)]
        self.assertEqual(len(primary), 1)
        self.assertLess(float(primary[0].defect.float().norm()), 1e-5)
        self.assertTrue(any(row["correction"] == "adaptive_rank_oracle" for row in rows))

    def test_gate_requires_complete_capture(self):
        decision = decide_gate(
            protocol_fixture(),
            summaries=[],
            capture_validation={
                "complete": False,
                "worst_dense_reference_parity": 0.0,
            },
        )
        self.assertEqual(decision["classification"], "incomplete")

    def test_summary_uses_energy_weighted_aggregate(self):
        base = {
            "split": "validation",
            "method": "m",
            "correction": "none",
            "rank": 0,
            "arithmetic_reduction": 2.0,
        }
        rows = [
            {**base, "relative_av_l2": 0.1, "numerator_sq": 1.0, "denominator_sq": 100.0},
            {**base, "relative_av_l2": 1.0, "numerator_sq": 1.0, "denominator_sq": 1.0},
        ]
        summary = [item for item in summarize_rows(rows) if item["scope"] == "validation"][0]
        self.assertAlmostEqual(summary["aggregate_relative_av_l2"], (2.0 / 101.0) ** 0.5)
        self.assertEqual(summary["worst_head_relative_av_l2"], 1.0)


if __name__ == "__main__":
    unittest.main()
