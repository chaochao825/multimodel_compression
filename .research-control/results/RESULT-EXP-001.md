# RESULT-EXP-001

- Experiment: EXP-001
- Date: 2026-08-07
- Code identity: evaluator `5076405568a4c2ba`
- Data identity: F81 artifact `be8fa1f525b66cf7`
- Configuration identity: `EXP-001-frozen-granularities`
- Evaluator identity: `probe_wan_grouped_state_granularity.py@50764055`
- Evidence tier: post-hoc audit
- Independent unit: the existing 4 calibration and 4 held-out prompt/seed identities
- Validity: valid within the 16 sampled token rows
- Outcome class: null / no-potential
- Protocol deviations: none

## Observations

- The audit compared scalar, channel widths 256/64/16, sampled token row, and
  sampled row x channel-64 AR(3)+current-innovation coefficients.
- Calibration-only fits never read held-out values. The per-sample oracle read
  each held-out target and is interpreted only as a function-class upper bound.
- On H2, static scalar sampled-output error was 7.016%; the best token-count
  independent static grouping was channel-16 at 6.774%, only 3.45% relative
  improvement.
- The strongest registered target-leaking row x channel-64 oracle reached
  5.382% aggregate and 19.002% worst sampled-output relative L2. It passed only
  6/192 cells, all at block 0, steps 9--11, across the two CFG branches.
- The row x channel-64 oracle used 1,536 coefficients per sampled call and is
  token-count-linear diagnostic evidence, not a deployable predictor.
- No full-tensor capture, rollout, video metric, kernel, or speed measurement was
  run because the registered H2 oracle stop rule fired.

## Validity checks

- Four production-environment synthetic tests passed: grouped transform inverse,
  recovery of distinct channel dynamics, held-out exclusion from static fits,
  and open-loop H2 exclusion of the intermediate dense residual.
- The evaluator verified the source artifact SHA-256, frozen protocol identity,
  complete 8-run manifest, sample plan, F81 scope, tensor shape, and six frozen
  granularities before evaluation.
- The initial local test attempt encountered a concurrent Windows pycache lock
  and a local interpreter without torch. Neither affected scientific execution;
  all tests and the sole audit ran in the recorded remote production environment.
- The 16 sampled rows are an optimistic diagnostic slice, not a full-tensor or
  population estimate. A positive result could not have promoted the claim,
  while failure of the strongest target-leaking oracle activates the frozen
  stop rule.

## Claim update

- `C-001`: refuted within its registered sampled-row audit. Finer coefficients
  materially reduce only part of the scalar error and remain far above 1%/2%.
- Supports: row/channel heterogeneity exists, because the strongest oracle
  improves over the scalar oracle by about 18.3% relatively.
- Does not support: grouped state as a high-fidelity skipped-block predictor,
  any whole-block or end-to-end speedup, or a full-tensor Wan conclusion.
- Unknown: whether a basis-changing content-conditioned subspace model can pass
  a new capacity Gate. That would be a different claim, not a refinement of
  grouped scalar AR.

## Gate recommendation

Close `G-001`, stop `L-001`, and do not deepen this function class. The only
passing subset is a target-leaking 3.125% cell slice and cannot justify a fresh
compute Gate or meaningful Amdahl gain.

## Artifacts

- Local report: `worldfoundry_hybrid_residual/results/WAN_GROUPED_TEMPORAL_STATE_EXP001_20260807.zh-CN.md`
- Local raw tables: `worldfoundry_hybrid_residual/results/wan_step_state_transition_exp001/`
- Local publication views: `worldfoundry_hybrid_residual/results/wan_step_state_transition_exp001/audit_v2/`
- Remote frozen output: `/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/results/wan_step_state_transition_exp001/grouped_state_audit_v1/`
- Decision JSON SHA-256: `998708a7315507bf`
- Summary CSV SHA-256: `37f21725ed1ffcef`
