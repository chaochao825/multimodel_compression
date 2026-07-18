> Public provenance note: hashes in the machine-audit findings describe the private source run. Hashes for sanitized bundled files are recorded in `PUBLICATION_MANIFEST.json` and `public_bundle_provenance`.

# TileLogic-RVQ Result Audit

## Status

**PASS**

## Major Issues

None.

## Minor Issues

None recorded by the machine audit.

## Checks

- **PASS** `required_outputs`: `{"missing": []}`
- **PASS** `cache_split_contract`: `{"counts": {"('chartqa', 'calibration')": 80, "('chartqa', 'evaluation')": 120, "('gqa', 'calibration')": 80, "('gqa', 'evaluation')": 120, "('textvqa', 'calibration')": 80, "('textvqa', 'evaluation')": 120}}`
- **PASS** `per_entry_cache_source_model_and_dtype_provenance`: `{"errors": [], "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3", "records": 600, "source_manifest_sha256": "2dd118b849e920b94b6dffe859dd7cfe464ebbcc52589e53b0e001325c858d5a"}`
- **PASS** `cache_provenance_migration_is_hash_linked_and_payload_preserving`: `{"format": "tilelogic_cache_provenance_backfill_v1", "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3", "new_manifest_sha256": "9858d3cf60c2c60be69fab27e13e7011009b41c5355fd2d746ba953c8ccc9e31", "old_manifest_sha256": "6cb6174ec20dfc505b9f96fb05e8a9c7441d54085099308ca128f9a1a7c5a391", "payload_tensors_unchanged": true, "records": 600, "source_manifest_sha256": "2dd118b849e920b94b6dffe859dd7cfe464ebbcc52589e53b0e001325c858d5a", "tensor_dtype_fields": ["thumbnail", "crops", "query", "crop_gradient"], "training_summary_sha256": "739a56c907546364c16b3aa55d3c762d3f06d0c1d74da09e3690ca0de5019a12"}`
- **PASS** `calibration_only_training_and_artifact_hashes`: `{"artifact_errors": [], "evaluation_entries_loaded": 0, "training_entries": 240}`
- **PASS** `router_training_and_fixed_slots_are_calibration_only`: `{"errors": []}`
- **PASS** `cache_training_feature_quality_hash_chain`: `{"cache_manifest_sha256": "9858d3cf60c2c60be69fab27e13e7011009b41c5355fd2d746ba953c8ccc9e31", "feature_summary_sha256": "ebb23a1ed61a6d16714f294c99520b7f85f5fd5b83013605931a45bbf10f4dfb", "training_summary_sha256": "739a56c907546364c16b3aa55d3c762d3f06d0c1d74da09e3690ca0de5019a12"}`
- **PASS** `rate_correction_preserves_all_non_rate_feature_semantics`: `{"allowed_changed_component_names": ["base_codebook", "base_scalar_scales", "logic_router", "mlp_router_normalizer", "mlp_router_parameters", "residual_codebooks", "router_curvature_prior"], "changed_component_names": ["base_codebook", "base_scalar_scales", "logic_router", "mlp_router_normalizer", "mlp_router_parameters", "residual_codebooks", "router_curvature_prior"], "changed_variants": 5760, "compared_variants": 8280, "errors": [], "format": "tilelogic_rate_precision_correction_validation_v1", "new_feature_samples_sha256": "ab50762d1b8324248c438d45364b94980423ea4b7fdb5d9e8325427679271105", "non_rate_semantics_identical": true, "old_feature_samples_sha256": "1d5dcdaac4d45f3267b529ad937820b993efc11e397d424b42074aa5b3c183e9", "records": 360}`
- **PASS** `complete_paired_360_evaluation_samples`: `{"calibration_records_loaded": 0, "counts": {"chartqa": 120, "gqa": 120, "textvqa": 120}, "feature_records": 360, "quality_records": 360}`
- **PASS** `quality_path_claim_boundary`: `{"answer_score_uses_all_manifest_answers": true, "cache_manifest_sha256": "9858d3cf60c2c60be69fab27e13e7011009b41c5355fd2d746ba953c8ccc9e31", "evaluation_samples": 360, "feature_eval_summary_sha256": "ebb23a1ed61a6d16714f294c99520b7f85f5fd5b83013605931a45bbf10f4dfb", "format": "tilelogic_quality_evaluation_v1", "gpu": "NVIDIA A800 80GB PCIe", "manifest_sha256": "2dd118b849e920b94b6dffe859dd7cfe464ebbcc52589e53b0e001325c858d5a", "model_dir": "external://private/66285546d2b821cf421d4f5eb2576359d3770cd3", "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31", "python": "3.10.0 | packaged by conda-forge | (default, Nov 20 2021, 02:24:10) [GCC 9.4.0]", "quality_path_is_latency_evidence": false, "quality_path_keeps_original_visual_token_length": true, "teacher_forced_target_policy": "first_manifest_answer_verbatim", "torch": "2.6.0+cu124", "transformers": "4.51.0"}`
- **PASS** `same_23_variant_matrix_for_feature_and_quality`: `{"errors": [], "expected_variants": 23}`
- **PASS** `exact_stream_shared_and_effective_rate_accounting`: `{"errors": []}`
- **PASS** `rate_precision_matches_executed_serialized_or_roundtripped_payloads`: `{"base_codebook_bits": 8454400, "errors": [], "legacy_logic_tree_values_require_exact_fp32_roundtrip": true, "logic_tree_formats": ["tilespec_logic_regression_tree_v1"], "policy": {"base_scalar_scale_bits": 32, "curvature_prior_bits": 32, "exact_fallback_value_bits": 16, "logic_leaf_bits": 32, "logic_threshold_bits": 16, "mlp_normalizer_bits": 32, "mlp_parameter_bits": 32, "vq_codeword_bits": 16, "vq_metric_weight_bits": 32, "vq_scale_table_bits": 16}, "residual_fisher_bits": 67371264, "residual_unweighted_bits": 67371264}`
- **PASS** `exact_fallback_fp16_payload_fully_charged`: `{"errors": []}`
- **PASS** `finite_feature_and_teacher_nll_metrics`: `{"errors": []}`
- **PASS** `answer_scores_and_agreement_recomputed`: `{"errors": []}`
- **PASS** `evaluation_oracle_router_arrays`: `{"errors": [], "oracle_samples": 48}`
- **PASS** `latency_components_memory_and_inclusion_flags`: `{"errors": [], "missing": [], "rows": 192}`
- **PASS** `latency_gpu_co_residency_provenance`: `{"file": "external://private/gpu_co_residency_during_run.log", "recomputed_sha256": "0f5a8c85ce76188af49c33093299ab66e96a0a158599853bcebb62ecc96513c5", "recorded": {"bytes": 21133, "file": "gpu_co_residency_during_run.log", "sha256": "0f5a8c85ce76188af49c33093299ab66e96a0a158599853bcebb62ecc96513c5"}}`
- **PASS** `router_mode_and_depth_usage`: `{"errors": [], "rows": 92}`
- **PASS** `six_predeclared_decisions_recomputed`: `{"errors": [], "statuses": {"1": "FAIL", "2": "FAIL", "3": "FAIL", "4": "INCONCLUSIVE", "5": "FAIL", "6": "FAIL"}}`
- **PASS** `report_claim_boundaries`: `{"aggregate_negative_needed": true, "required_phrases": ["not native compact-prefill latency evidence", "No PPA, kernel-fusion, or physical-hardware claim is made", "does not support an aggregate positive claim"]}`

## Recommended Next Step

Use exactly one independent Review Agent to inspect the implementation, raw evidence,
decision rules, and claim boundaries. Resolve every major finding before publication.
