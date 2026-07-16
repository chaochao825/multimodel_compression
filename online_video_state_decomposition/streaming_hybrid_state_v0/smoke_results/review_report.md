# Streaming Hybrid State V0 Review Report

## Overall: PASS

This report is produced by an independent result-audit script. The reviewer does not train models or select operating points.

## Checks

| Check | Status | Severity | Detail |
|---|---|---|---|
| required_outputs_present | PASS | major | missing files: [] |
| clip_split_disjoint | PASS | major | overlapping runs: [] |
| stratified_3_1_2_split | PASS | major | {"camera_motion": {"test": 2, "train": 3, "val": 1}, "high_change": {"test": 2, "train": 3, "val": 1}, "object_motion": {"test": 2, "train": 3, "val": 1}, "scene_cut": {"test": 2, "train": 3, "val": 1}, "static": {"test": 2, "train": 3, "val": 1}} |
| predictor_metrics_finite | PASS | major | checked 14 predictor rows |
| predictor_ablation_complete | PASS | major | expected predictors: ['ema_025', 'ema_050', 'ema_075', 'fourier_h4_k1', 'fourier_h8_k2', 'linear', 'previous'] |
| temporal_entropy_not_degenerate | PASS | minor | at least one residual time-series entropy must be nonzero |
| vq_metrics_finite | PASS | major | checked 46 VQ/scalar rows |
| vq_baselines_complete | PASS | major | observed methods: ['raw_pq', 'residual_pq', 'scalar_quant'] |
| int4_baseline_present | PASS | major | INT4 test layers: [22] |
| multi_bit_pq_sweep_present | PASS | major | raw PQ nominal points: [0.5, 1.0, 1.5, 2.0, 4.0] |
| vq_static_and_metadata_accounted | PASS | major | effective_bps must include nonzero codebook static bits |
| codebook_hash_stable_across_val_test | PASS | major | checked 10 method/codec/layer codebooks |
| controller_ablation_complete | PASS | major | observed learned controllers: ['decision_tree', 'threshold'] |
| controller_budget_sweep_complete | PASS | major | observed budgets: [0.5, 1.0, 1.58, 2.0, 4.0] |
| fixed_policy_baselines_present | PASS | major | observed policies: ['always_innovation', 'always_int4_refresh', 'always_predict', 'always_reuse', 'decision_tree', 'threshold'] |
| combined_action_rates_sum_to_one | PASS | major | checked 30 combined rows |
| combined_static_and_action_bits_accounted | PASS | major | effective_bps must be no smaller than stream payload_bps |
| combined_layers_complete | PASS | major | combined layers: [22] |
| controller_uses_causal_rgb_features | PASS | major | source and regression test use current/previous RGB summaries |
| codebook_fit_uses_train_split | PASS | major | codebook samples are selected from train clips |
| combined_codec_is_open_loop | PASS | major | predictor history is reconstructed state, not target state |
| claim_boundary_explicit | PASS | major | summary separates representation quality from task/PPA claims |
| no_unsupported_positive_claim | PASS | major | unsupported phrases: [] |

## Major Issues

- None.

## Minor Issues

- None.

## Claim Boundary

- PASS means the probe is internally consistent and auditable.
- It does not imply end-to-end Video-LLM quality, encoder speedup, or hardware PPA improvement.
- Component and combined scientific verdicts must still follow the measured held-out rows and stated kill criteria.

## Recommended Next Step

Promote only components that beat their matched simple baseline. Any promising combined point should next be tested on a task-level streaming benchmark before RTL or PPA claims are added.
