# PLAN-018: Test anchor self-error before fresh-data certification

- Status: completed
- Owner: researcher and Agent
- Gate: G-018
- Experiment: EXP-039
- Claim: C-018
- Candidate: L-018
- Lane: explore
- Resource cap: three existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether leave-one-anchor-out residual predicts unseen-row basis risk
without reading unobserved rows or the prior EXP-038 test identity.

## Milestones

1. Freeze cross-fit priors, LOAO definition, calibration multiplier, gates, data
   access, and outcome map.
2. Implement pure metric/calibration utilities and test observed-row isolation,
   rank-deficit accounting, deterministic ties, and outcome classification.
3. Lock hashes, run once on exclusive H200, and visualize LOAO-vs-actual risk,
   action calibration, high-risk recall, and refinement deltas.
4. Close G-018 before any fresh capture, coefficient predictor, or controller.

## Outcome

Completed as `observability-null`. Pooled Spearman and high-risk recall passed
at 0.9804 and 88.89%, and false-safe count was zero, but certified coverage was
only 16.67% versus the registered 25% gate. Post-hoc decomposition further
showed that the signal mostly tracks static head/action risk rather than
content drift. Same-step anchor self-certification is parked.

## Stop rules

- Do not read `s03` or tune features and thresholds on `s02`.
- Stop after one terminal outcome or the one allowed pre-outcome repair.
- An observability null parks same-step anchor self-certification on this cell.
