# PLAN-052: Balanced finite-jump trainability decomposition

- Status: completed
- Owner: researcher and agent
- Gate: G-023
- Claim: C-023
- Candidate: L-023
- Experiment: EXP-044
- Lane: explore
- Resource scope: immutable EXP-043 captures, two already exposed development
  identities, one exclusive H200, no model inference or VAE decode

## Decision question

Was EXP-043 primarily limited by interval-sampling starvation, missing stage
conditioning, local function capacity, or cross-identity transfer?

## Ordered execution

1. Reuse and hash-check the immutable production-UniPC capture payloads.
2. Train with exactly balanced interval updates and target-energy-normalized
   crop loss; no screen-based checkpoint selection.
3. Compare shared, interval-FiLM, stage-specific, and wider stage-specific
   lifting in a fixed order.
4. Run cross-identity transfer and same-identity transductive diagnostics.
5. Classify the bottleneck once, write all arms including adverse results, and
   stop without rollout, new identities, or architecture search.

## Gates

- Local capacity: wide target-exposed transductive endpoint at most 1%.
- Structured capacity: stage-specific target-exposed transductive endpoint at
  most 1%.
- Observable transfer: interval-conditioned observable endpoint at most 2%.
- Sampling relevance: balanced shared observable improves at least 25% over the
  frozen EXP-043 observable endpoint of 10.901%.

## Guards

- Both identities are development-only and outcome-exposed.
- Results cannot update a publication claim or select a final architecture.
- All comparisons use the same fixed steps, crop stream, loss normalization,
  final checkpoint rule, and interval counts unless explicitly labeled as a
  capacity control.
- Stop on hash drift, non-finite values, foreign GPU overlap, or one valid
  result. No rank, feature, optimizer, or loss search follows the outcome.

## Decision effect

- Transductive null: reject local finite-jump adapters and use a released/full
  few-step student.
- Transductive pass plus transfer null: attribute the bottleneck to data or
  observability and require multi-identity training under a new decision.
- Interval-conditioned transfer signal: permit only a fresh prospective
  equal-budget adapter-versus-LoRA Gate.
- Engineering failure: repair only implementation faults without changing the
  frozen arms, data, steps, or thresholds.

## Outcome

EXP-044 completed with `local-function-null`. The widest privileged
same-identity control reached 8.190% endpoint relative L2, while interval-FiLM
observable transfer reached 43.728%. All registered checks failed. L-023 is
parked and no rollout, new identity, width, rank, loss, or kernel work is
authorized for this candidate.
