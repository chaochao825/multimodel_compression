# PLAN-000: Stagewise temporal transition and protocol-mismatch audit

- Status: completed; mainline and bounded side probe closed
- Owner: researcher and Agent
- Gate: G-000
- Claims: C-000
- Candidate line: L-000
- Lane: prove
- Resource cap: two eight-identity captures on two H200 GPUs, one frozen
  analysis, and at most one boundary refinement

## Decision to unlock

Decide whether `C-000` should advance to a separate rollout/kernel Gate, be
narrowed to an F17 or oracle-only boundary, or be rejected.

## Context

`PROJECT.md` defines the north star. `RDR-0001` records the researcher's choice
to test denoising-step state transitions after static spatial and cross-frame
state failures. `EXP-000`, `EXP-001`, and `EXP-002` are closed. The bounded
target-semantics audit did not change the mainline decision.

## Closed side-probe Gate

`G-002` compares whole-block residual forecasting with TaylorSeer-style
SA/CA/FFN forecasting under recomputed target-step conditioning. It is capped at
two F81 identities, blocks 0/12/29, deterministic sampled rows, and one frozen
H1/H2 analysis. It completed with only 2.00% relative H2 improvement and a
22.114% Block-12 error, so it cannot promote a method or authorize
rollout/kernel work.

## Non-goals

- No full rollout, kernel implementation, or paper claim before the screen passes.
- No held-out cell selection or deployment coefficient fitting.
- No architecture expansion beyond the single preregistered boundary refinement.

## Milestones

1. Frozen protocol, identities, code semantics, and sampled module targets.
2. Validated wrapper restoration, dense parity, call order, and BF16 replay.
3. Captured two F81 identities and ran the frozen evaluator once after one
   preregistered pre-outcome engineering repair.
4. Classified `EXP-002` as genuine-dynamics-failure and closed `G-002`.
5. No rollout, training, or kernel Gate was opened.

## Stop and escalation rules

Stop on manifest incompleteness, call-order mismatch, non-positive Gram norms,
calibration/held-out identity overlap, or resource-cap exhaustion. Stop the
family if its per-sample bounded oracle fails. If only the oracle passes, permit
one shallow dynamic refinement and no rollout. Classify F17-only success as a
resolution boundary.

## Mainline closure

Closed by `RESULT-EXP-000`, `RESULT-EXP-001`, `RESULT-EXP-002`, and `RDR-0002`.
The scalar oracle, grouped audit, and target-semantics explanation failed. No
rollout or kernel work is authorized. A different function class requires a new
plan.
