# Online-Video Evidence Completion Matrix

## Claim Decision

The frozen routed codec now passes an independent fixed-state preservation gate. No complete hybrid method has yet passed the combined end-to-end streaming quality, latency, and SLO gates; comparability groups remain separate.

## Our Component Gates

| Component | Status | Evidence tier | Primary metric |
|---|---:|---|---|
| query_conditioned_selector | FAIL | project_native_preregistered_proxy | accuracy_gain=-0.005 fraction |
| query_conditioned_native_memory | OPEN | project_native_model_level_confirmation | accuracy=0.51 fraction |
| dual_timescale_spectral_trigger | FAIL | controlled_synthetic_trigger | event_recall=0.8 fraction |
| low_rank_sparse_codec | FAIL | project_native_model_level_confirmation | accuracy=0.505 fraction |
| routed_low_rank_spatial_sparse_codec | OPEN | project_native_posthoc_same_set | prediction_agreement_rate=0.985 fraction |
| routed_low_rank_spatial_sparse_codec | PASS | project_native_frozen_independent_replication | accuracy=0.45 fraction |
| query_conditioned_routed_memory | OPEN | project_native_frozen_independent_replication | accuracy_gain=0.02 fraction |
| low_rank_long_term_state | OPEN | project_native_representation_probe | rank32_energy_mean=0.76627 fraction |
| sparse_event_residual | FAIL | project_native_representation_probe | top10_residual_energy_mean=0.481661 fraction |
| bccb_transport | FAIL | project_native_representation_probe | bccb_gain_vs_identity_mean=0.0963924 fraction |

## Completion Status

NA=8, OPEN=4, PAPER_ONLY=4, PASS=35, PLACEHOLDER=1, PROXY_ONLY=6, UNAVAILABLE=6

Runtime snapshot: `2026-07-22T05:46:06.569105+00:00` with 5 overrides.

## Claim Boundary

The current evidence supports independent routed-state preservation on the frozen LLaVA MVBench reserve. It does not justify claims that BCCB replaces global video attention, that the error-oracle route is a cheap semantic event detector, or that proxy/stage latency is official TTFT or SLO latency. SelectStream and StateKV remain paper/proxy references until executable official code is available.

The matrix figure is `streaming_evidence_completion_matrix.png`/`.pdf`; raw rows are preserved in `completion_matrix.csv` and `evidence_metrics.csv`.
