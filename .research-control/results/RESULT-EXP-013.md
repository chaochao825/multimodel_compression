# RESULT-EXP-013: Rank-conditioned support-manifold cost shaping

- Status: complete
- Validity: valid exploratory cost boundary
- Date: 2026-08-12
- Gate: G-013
- Claim: C-013
- Candidate: L-013

## Registered outcome

`cost-boundary`. Rank-conditioned swaps reduced the validation-selected mean
work from 58.57% to 58.01%, but the frozen test result was 1.197% aggregate and
2.125% worst-head error. The registered test-exposed shaped allocation reached
0.977% / 1.858% only at 61.33% optimistic work, exactly the baseline posterior
cost tier.

## Supports

- The residual-rank objective can improve individual regular supports and move
  isolated heads to a cheaper rank action.
- The current 35% mass support is already a local optimum for most records:
  518 of 648 shaped records accepted zero swaps.
- The remaining rank-17--32 shoulder is broad rather than localized to a few
  replaceable contiguous-64 blocks.

## Does not support

- It does not support `C-013`, a deployable support selector, measured H200
  speedup, model-wide coverage, or rollout quality.
- It does not rule out content-generated support, irregular sparse kernels, or
  a lower-cost tail representation.
- Dense-target swaps, adaptive SVD, and posterior allocations remain oracle
  diagnostics and cannot be exposed to held-out deployment.

## Decision

Park `L-013`. Do not add swap rounds or nearby regular pattern families on this
cell. One bounded successor may change the tail cost structure by measuring
whether a content-shared output basis grows sublinearly with the number of
query tiles and can amortize basis construction below 55% optimistic work.

## Integrity

- Two new mechanism tests plus eight inherited tests passed remotely.
- Local/remote probe, rank core, allocation core, test, and config hashes match.
- `SUCCESS.json`, manifest, finite-value, fixed-density, monotone-trace, and
  work guards pass.
- Runtime was 158.79 seconds on exclusive physical H200 GPU3; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/support_manifold_cost_shaping_f81_l14s9_exp013_v1/decision.json`
- `worldfoundry_hybrid_residual/results/support_manifold_cost_shaping_f81_l14s9_exp013_v1/support_rank_records.csv`
- `worldfoundry_hybrid_residual/results/WAN_SUPPORT_MANIFOLD_COST_SHAPING_EXP013_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/support_manifold_cost_shaping_exp013_20260812/support_manifold_cost_shaping_exp013_diagnostics.png`
