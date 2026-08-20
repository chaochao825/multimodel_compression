# RESULT-EXP-043: Production-UniPC finite-jump development screen

- Status: complete
- Validity: valid registered execution with a major interpretation limitation
- Date: 2026-08-13
- Gate: G-022
- Claim: C-022
- Candidate: L-022

## Registered outcome

`capacity-null`. The target-exposed arm reached 11.115% held-out frozen-field
endpoint relative L2, missing the registered 0.5%/1% aggregate/worst capacity
gate. The deployable observable-motion arm reached 10.901%, missing the 1%/2%
transfer gate. Every trained arm was worse than the 9.451% zero-correction
endpoint.

The registered alignment comparison mechanically passed: observable motion
reduced displacement-weighted curvature error by 66.2% relative to clean
unaligned lifting. This is not a positive mechanism result because the clean
arm collapsed to 336.83% curvature error; observable motion remained at
113.83%, worse than the 100% zero-correction reference.

## Supports

- The exact frozen EXP-043 training protocol does not provide a useful
  production 20-to-4 correction and must not proceed to rollout or kernel work.
- Clean-prediction identity coordinates transfer poorly between the two
  development identities under the registered local lifting model.
- Observable motion prevents the extreme clean-coordinate collapse, but does
  not beat zero correction and only narrowly beats the equal-parameter
  pointwise control.

## Does not support

- It does not show that trainable or distilled Wan acceleration is ineffective;
  released few-step students already contradict that broader statement.
- It does not establish a mathematical capacity limit for a 28,160-parameter
  adapter. The target-exposed arm was still trained only on `dev00_cloth` and
  therefore mixes capacity, optimization, and identity transfer.
- It does not establish a stable benefit from motion conjugation. The 66.2%
  ratio is caused by a failed denominator arm.
- It does not predict closed-loop four-step video quality or H200 end-to-end
  speed.

## Integrity limitation

The frozen `delta_sigma^4` sampler was extremely imbalanced. Its four
probabilities were 0.0099%, 0.0764%, 1.2217%, and 98.6919%; the deterministic
2,000-step counts were 0, 1, 23, and 1,976. The run is valid for its registered
protocol, but the first three intervals were not meaningfully trained. This
prevents interpreting `capacity-null` as a general function-class failure.

## Decision

Refute `C-022` within the exact EXP-043 boundary and park `L-022`. Do not add
rank, experts, loss terms, identities, rollout, or kernels under this Gate. A
new, explicitly diagnostic Gate may reuse only the already exposed development
identities to separate sampler starvation, missing interval conditioning,
local capacity, and cross-identity transfer. Such post-outcome diagnostics
cannot support a publication claim without a later prospective identity split.

## Integrity

- Remote process exited 0 on one exclusive NVIDIA H200 NVL.
- Twelve registered unit tests passed and foreign-process overlap was zero.
- Capture, code, config, checkpoint, summary, decision, and local-download
  SHA-256 audits passed.
- The only pre-metric repair fixed a missing local save-helper import and did
  not change any tensor, teacher, arm, identity, threshold, or access order.

## Evidence

- `worldfoundry_hybrid_residual/results/production_finite_jump_exp043_v1/SUCCESS.json`
- `worldfoundry_hybrid_residual/results/production_finite_jump_exp043_v1/summary.json`
- `worldfoundry_hybrid_residual/results/production_finite_jump_exp043_v1/decision.json`
- `worldfoundry_hybrid_residual/results/WAN_TRAINABLE_PRODUCTION_DISTILLATION_EXP043_20260813.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/production_finite_jump_exp043_20260813/production_finite_jump_exp043.png`
