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

## Independent-noise state and sampler follow-up

The follow-up separates representation capacity from sampler deployability.
Across all three PushT checkpoints, a calibration-only state-plus-noise bridge
recovers 99.55%--99.92% of the independent-noise oracle gap at less than 1% of
the replaced FFN `linear2` arithmetic cost. It remains stable for one, two, and
four consecutive skips, although the excess error above the oracle floor grows
with the skip horizon.

The same mechanism does not pass the real DDPM sampler gate. With common random
numbers, exact refresh every five denoising steps, and all eight FFNs replaced,
executed-step relative error is 3.51% and first-approximation P95 error is
17.86%. A late-only schedule does not repair the failure. Layer attribution
shows that layer 0 dominates, but even the frozen `{1,2}` subset misses the
registered first-action P95 threshold on two untouched checkpoints. Closed-loop
PushT rollout and speed claims are therefore not authorized.

Read `reports/ACTION_DIT_TQC_NOISE_STATE_SAMPLER_20260826.zh-CN.md` for the full
mechanism/sampler analysis. Reproduction uses
`scripts/probe_action_dit_noise_response_bridge.py`,
`scripts/probe_action_dit_multiskip_state.py`, and
`scripts/probe_action_dit_independent_sampler.py`; the preregistered additions
are recorded in `protocols/action_dit_noise_response_bridge_20260826.md`.
Summary CSV data and PNG/PDF/SVG figures use the `figures/action_dit_tqc_*`
prefix.

## Reproduction

Run `scripts/probe_action_dit_structured_correction.py` or one of the TQC probe
scripts with a compatible Diffusion Policy checkpoint and dataset. Scientific
settings and stop rules are frozen under `protocols/`. Unit tests cover
schedule bucketing, non-periodic boundaries, local-plus-global defect recovery,
horizon shifting, equal-budget reuse, noise-state bridges, and bounded
low-rank payload.

Raw CSV/JSON outputs for all three checkpoints are under `results/`. The
publication plots and their bound CSV data are under `figures/`.
