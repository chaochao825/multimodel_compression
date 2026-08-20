# RESULT-EXP-012: Cost-constrained heterogeneous-rank frontier

- Status: complete
- Validity: valid exploratory transfer boundary
- Date: 2026-08-12
- Gate: G-012
- Claim: C-012
- Candidate: L-012

## Registered outcome

`transfer-boundary`. The validation minimum-work deployment allocation used
35% exact support, average rank 13.67, and 58.57% optimistic dense work. It
reached 0.991% / 1.802% on validation but 1.191% / 2.125% under the frozen
head allocation on test, missing both registered quality guards.

## Supports

- Intermediate ranks materially lower the cost of the EXP-011 rank boundary;
  head heterogeneity is useful rather than incidental.
- Static head identity is insufficient for minimum-cost rank certification.
  Head 3 moved from 1.006% to 2.125% under the same rank-12 action.
- A post-hoc test-exposed optimum still required 61.33% work, so transfer is
  not the only issue; the representation has too little runtime margin at 60%.

## Does not support

- It does not support `C-012`, a static head-rank deployment policy, a learned
  selector, or measured H200 speedup.
- It does not establish model-wide coverage or rollout quality.
- It does not justify treating adaptive SVD or inverse work as an executable
  attention implementation.

## Decision

Park `L-012` as a static heterogeneous-rank candidate. The only bounded revival
is to change the defect itself: one rank-conditioned, hardware-regular support
shaping oracle may test whether the 61.33% test lower bound can be pushed below
55%--60%. Do not train a router unless that oracle creates real overhead margin.

## Integrity

- Three new Pareto tests and eleven inherited mechanism tests passed remotely.
- Local/remote probe, core, test, and config hashes match.
- `SUCCESS.json`, decision, manifest, finite-value, and work-cap guards pass.
- Runtime was 5.15 seconds on exclusive physical H200 GPU2; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/heterogeneous_rank_frontier_f81_l14s9_exp012_v1/decision.json`
- `worldfoundry_hybrid_residual/results/heterogeneous_rank_frontier_f81_l14s9_exp012_v1/cap_frontier.csv`
- `worldfoundry_hybrid_residual/results/WAN_HETEROGENEOUS_RANK_FRONTIER_EXP012_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/heterogeneous_rank_frontier_exp012_20260812/heterogeneous_rank_frontier_exp012_diagnostics.png`
