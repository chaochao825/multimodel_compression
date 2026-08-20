# RESULT-EXP-016: Equal-cost distributed micro-anchor basis

- Status: complete
- Validity: valid exploratory null
- Date: 2026-08-12
- Gate: G-016
- Claim: C-016
- Candidate: L-016

## Registered outcome

`distributed-null`. At rank 64, the target-exposed distributed greedy oracle
reached 2.013% aggregate and 4.557% worst-head relative L2 on validation. The
best legal Q-k-center selector reached 2.213% / 5.125%. Both use 64 exact rows
per M8 group and 43.32% optimistic work, and both miss the registered 1% / 2%
gate.

No legal validation candidate passed, so the test tensor was not loaded and
`frozen_test` is null. Test path metadata was fingerprinted without reading
tensor contents.

## Supports

- Equal-cost distributed rows improve rank-64 aggregate error over one
  contiguous tile from 2.713% to 2.213% for Q-k-center and 2.013% for the
  target-exposed oracle.
- The same unselected rows have an adaptive rank-64 floor of about 0.737%, so
  residual low-rank capacity remains; the failure is observation/subspace
  embedding rather than a universal low-rank impossibility.
- Failure is head-heavy: heads 2, 6, 7, and 9 dominate the rank-64 tail even
  under target-exposed row selection.

## Does not support

- Four regular 16-row observations are not a stable M8 residual-subspace
  embedding under the same 64-row budget.
- More fixed microblock selectors, larger rank, BCM/BCCB, or Butterfly bases on
  this unchanged observation primitive are not justified.
- The experiment does not test coefficient prediction, measured H200 speed,
  rollout quality, or full-model coverage.

## Decision

Refute `C-016` and park `L-016`. A separate bounded Gate may test a genuinely
different observation class using calibration priors, progressive row budgets,
per-head certification, and dense fallback. Such a Gate must preserve the
EXP-016 null and first establish validation-to-test function-class transfer
before any target-free coefficient predictor.

## Integrity

- Formal run used physical H200 GPU3 after the competing workload exited and
  three 20-second idle confirmations; exit code 0 and runtime 18.31 seconds.
- Twenty related tests passed; registered probe/core/test hashes matched.
- Local and remote artifact SHA-256 values match.
- The invalid resource-overlap attempt remains isolated under remote trash and
  is not cited as evidence.

## Evidence

- `worldfoundry_hybrid_residual/results/distributed_micro_anchor_basis_f81_l14s9_exp016_v1/decision.json`
- `worldfoundry_hybrid_residual/results/distributed_micro_anchor_basis_f81_l14s9_exp016_v1/basis_candidates.csv`
- `worldfoundry_hybrid_residual/results/WAN_DISTRIBUTED_MICRO_ANCHOR_BASIS_EXP016_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/distributed_micro_anchor_exp016_20260812/distributed_micro_anchor_exp016_diagnostics.png`
