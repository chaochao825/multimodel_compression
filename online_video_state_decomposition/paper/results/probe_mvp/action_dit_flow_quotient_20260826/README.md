# Action-DiT FlowQuotient probe

This package contains the first action-generating Diffusion Transformer probe
in the structured-state research program. It evaluates selective W4 FFN fake
quantization and action-horizon mean, affine, circular, non-periodic Toeplitz,
low-rank, and hybrid corrections on three independently trained PushT
checkpoints.

The fixed replication result is mixed: train-0 and train-2 are `BOUNDARY`,
while train-1 is `GO`. Local action-axis structure transfers, but the post-hoc
low-rank coefficient predictor and complete-sampling stability do not. This is
not an environment-success or integer-kernel speed result.

Start with `reports/ACTION_DIT_PI_WAM_STRUCTURED_ANALYSIS_20260826.zh-CN.md`.
The report links the experiment to QuantVLA, QVLA, LightDP, pi0-class action
flow models, and world action models, and develops the proposed Action-Flow
Quotient Error Shaping objective.

## Transported Quotient Cache gate

The receding-horizon cache-innovation follow-up is complete on all three PushT
checkpoints. Equal-budget horizon shifting reduces exact-suffix velocity risk
by 67.66% on average, and a radius-2 temporal correction removes another
42.16%. A calibration-fixed rank-8 basis captures 85.07% of held-out
innovation energy.

The cross-checkpoint decision remains `BOUNDARY`. The effect requires aligned
latent noise on overlapping action tokens; independent current-tick noise
reverses the shift advantage. The deployed `m=8` setting also reuses only 20%
of the horizon and has a 1.119x optimistic denoiser arithmetic ceiling. This is
a frozen teacher-forced mechanism result, not an environment or speed claim.

Read `reports/ACTION_DIT_TRANSPORTED_QUOTIENT_CACHE_20260826.zh-CN.md` for the
full analysis, negative control, novelty boundary, and staged sampler/rollout
plan. Reproduction uses `scripts/probe_action_dit_transport_cache.py`; the
frozen protocol is `protocols/action_dit_transport_cache_geometry_20260826.md`.

## Reproduction

Run `scripts/probe_action_dit_structured_correction.py` or
`scripts/probe_action_dit_transport_cache.py` with a compatible Diffusion
Policy checkpoint and dataset. Scientific settings and stop rules are frozen
under `protocols/`. Unit tests cover schedule bucketing, non-periodic
boundaries, local-plus-global defect recovery, horizon shifting, equal-budget
reuse, and bounded low-rank payload.

Raw CSV/JSON outputs for all three checkpoints are under `results/`. The
publication plots and their bound CSV data are under `figures/`.
