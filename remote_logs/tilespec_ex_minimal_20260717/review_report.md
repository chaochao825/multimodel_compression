# TileSpec-Ex Independent Review Report

## Overall: PASS

Review PASS means evidence/accounting consistency, not that a scientific gate is positive.

## Checks

- **PASS** `required_outputs`: missing=[]
- **PASS** `three_dataset_200_sample_contract`: counts={'gqa': 200, 'textvqa': 200, 'chartqa': 200}
- **PASS** `quality_run_completed`: {"complete": true, "elapsed_seconds": 912.2497722610133, "new_oracle_records_by_dataset": {"chartqa": 16, "gqa": 16, "textvqa": 16}, "oracle_records": 48, "quality_records": 600}
- **PASS** `six_main_methods_two_rates`: errors=[], expected_variants=15
- **PASS** `five_image_multitile_grid`: errors=[]
- **PASS** `answer_scores_recomputed`: errors=[]
- **PASS** `answer_derived_fields_recomputed`: errors=[]
- **PASS** `exact_equal_budget_accounting`: errors=[]
- **PASS** `oracle_16_samples_per_dataset`: counts={'gqa': 16, 'textvqa': 16, 'chartqa': 16}
- **PASS** `oracle_arrays_and_risk_product`: errors=[]
- **PASS** `oracle_correlations_recomputed`: rows=96, errors=[]
- **PASS** `quality_and_oracle_gate_inputs_recomputed_from_raw`: errors=[]
- **PASS** `latency_matrix_complete`: missing=[], extra=[], invalid=0
- **PASS** `aligned_latency_budget_and_claim_boundary`: {"budget_contract": [{"base_tokens": 96, "exception_blocks": 8, "exception_tokens": 32, "retained_crop_tokens": 128, "retention_rate": 0.125}, {"base_tokens": 192, "exception_blocks": 16, "exception_tokens": 64, "retained_crop_tokens": 256, "retention_rate": 0.25}], "structured_gate_validated": false}
- **PASS** `structured_latency_recomputed_from_raw`: errors=[]
- **PASS** `structured_diagnostic_and_status_recomputed`: errors=[]
- **PASS** `gate_boole_recomputed`: errors=[]
- **PASS** `quality_latency_claim_boundary`: {"keeps_length": true, "latency_evidence": false}
- **PASS** `no_unsupported_system_claim`: matches=[], structured_inconclusive=INCONCLUSIVE

## Major Issues

- None.

## Minor Issues

- None.

## Recommended Next Step

Do not start fused-kernel work. Tile and risk gates failed; the structured gate needs a native compact multimodal TTFT path before it can be decided.
