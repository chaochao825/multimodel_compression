# RESULT-EXP-041: Temporal heterogeneous-rank frontier

- Status: complete
- Validity: valid prospective staged-transfer pass on the registered claim
- Date: 2026-08-12
- Gate: G-020
- Claim: C-020
- Candidate: L-020

## Registered outcome

`strong-temporal-frontier`. The validation-frozen per-head action table reached
the previously unopened `s03` at 0.486% aggregate and 0.874% worst head-step
relative L2, improved 65.8% over the matched static table, passed all three
targets, and used 46.02% local optimistic work.

## Supports

- Adjacent-step marginal-defect subspaces contain transferable current-content
  information beyond a cross-identity same-step calibration basis.
- Per-head heterogeneity is necessary at the validation boundary: heads 2 and 6
  use dense fallback; non-dense ranks range from 32 to 96.
- The temporal table transfers without changing actions after validation and
  approaches its matched adaptive rank-table floor on the one staged identity.

## Does not support

- Support and coefficients are target-exposed; this is not an executable
  attention approximation.
- A dense previous-step defect is required for every registered temporal basis.
  Source-refresh amortization is untested.
- The 2.17x number is a local optimistic arithmetic upper bound, not measured
  H200 latency or end-to-end speedup.
- One layer, three steps, one branch, and one test identity do not establish
  whole-model or population generalization.
- Mean non-dense rank is 75.2/128, so this is not evidence for an extremely
  low-rank tail.

## Decision

Support `C-020` within its exact staged-test boundary and integrate `L-020` as a
mechanism witness. Open no rollout or kernel work. A separate Gate may test
target-free coefficient transport and dense-anchor horizon with refresh-aware
cost; failure of either should park train-free temporal transport.

## Integrity

- Formal exit 0, foreign-PID overlap 0, dual flock, and six idle checks.
- Seven registered tests and all locked hashes passed before data access.
- `s03` was loaded only after validation passed.
- Local and remote 22-artifact digest:
  `93f05023f501193307fa9027c4552bb6b87e05e0b92fc2fbd5ccefe9416061d0`.

## Evidence

- `worldfoundry_hybrid_residual/results/temporal_heterogeneous_rank_frontier_f81_l14_exp041_v1/decision.json`
- `worldfoundry_hybrid_residual/results/WAN_TEMPORAL_HETEROGENEOUS_RANK_FRONTIER_EXP041_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/temporal_heterogeneous_rank_exp041_20260812/temporal_heterogeneous_rank_exp041.png`
