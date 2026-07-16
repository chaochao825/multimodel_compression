# Streaming Hybrid State V0 Review Report

## Overall: PASS

This report is produced by an independent result-audit script. The reviewer does not train models or select operating points.

## Checks

| Check | Status | Severity | Detail |
|---|---|---|---|
| required_outputs_present | PASS | major | missing files: [] |
| clip_split_disjoint | PASS | major | overlapping runs: [] |
| stratified_3_1_2_split | PASS | major | {"camera_motion": {"test": 2, "train": 3, "val": 1}, "high_change": {"test": 2, "train": 3, "val": 1}, "object_motion": {"test": 2, "train": 3, "val": 1}, "scene_cut": {"test": 2, "train": 3, "val": 1}, "static": {"test": 2, "train": 3, "val": 1}} |
| predictor_metrics_finite | PASS | major | checked 28 predictor rows |
| predictor_ablation_complete | PASS | major | expected predictors: ['ema_025', 'ema_050', 'ema_075', 'fourier_h4_k1', 'fourier_h8_k2', 'linear', 'previous'] |
| temporal_entropy_not_degenerate | PASS | minor | at least one residual time-series entropy must be nonzero |
| vq_metrics_finite | PASS | major | checked 92 VQ/scalar rows |
| vq_baselines_complete | PASS | major | observed methods: ['raw_pq', 'residual_pq', 'scalar_quant'] |
| int4_baseline_present | PASS | major | INT4 test layers: [15, 22] |
| multi_bit_pq_sweep_present | PASS | major | raw PQ nominal points: [0.5, 1.0, 1.5, 2.0, 4.0] |
| vq_static_and_metadata_accounted | PASS | major | effective_bps must include nonzero codebook static bits |
| codebook_hash_stable_across_val_test | PASS | major | checked 20 method/codec/layer codebooks |
| controller_ablation_complete | PASS | major | observed learned controllers: ['decision_tree', 'dlgn', 'mlp', 'threshold'] |
| controller_budget_sweep_complete | PASS | major | observed budgets: [0.5, 1.0, 1.58, 2.0, 4.0] |
| dlgn_hard_metrics_reported | PASS | major | checked 10 hardened DLGN rows |
| mlp_normalizer_state_accounted | PASS | major | checked 10 MLP rows |
| tree_topology_bits_accounted | PASS | major | checked 10 decision-tree rows |
| fixed_policy_baselines_present | PASS | major | observed policies: ['always_innovation', 'always_int4_refresh', 'always_predict', 'always_reuse', 'decision_tree', 'dlgn', 'mlp', 'threshold'] |
| combined_action_rates_sum_to_one | PASS | major | checked 80 combined rows |
| combined_static_and_action_bits_accounted | PASS | major | effective_bps must be no smaller than stream payload_bps |
| combined_effective_bps_recomputes | PASS | major | effective_bps exactly amortizes codebook and controller state |
| encoder_required_rate_contract | PASS | major | innovation+refresh requires current hidden state; reuse+predict is the encoder-skip fraction |
| combined_layers_complete | PASS | major | combined layers: [15, 22] |
| component_verdicts_complete | PASS | major | observed verdict keys: ['combined_memory', 'conditional_compute', 'end_to_end_task', 'logic_controller', 'predictor', 'residual_vq'] |
| memory_verdict_matches_candidate_rows | PASS | major | candidate layers=[22], verdict=Mixed |
| compute_verdict_matches_encoder_gate | PASS | major | candidate layers=[], verdict=Negative |
| derived_encoder_rates_exact | PASS | major | checked 40 learned policy rows |
| controller_uses_causal_rgb_features | PASS | major | source and regression test use current/previous RGB summaries |
| codebook_fit_uses_train_split | PASS | major | codebook samples are selected from train clips |
| fp16_cost_matches_parameter_precision | PASS | major | codebooks, scalar scales, tree thresholds, and DLGN thresholds are numerically rounded to the precision used in bit accounting |
| controller_static_description_complete | PASS | major | tree topology and MLP normalization state are counted |
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
