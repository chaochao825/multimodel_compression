# PLAN-007: Test a signed content-generated numerator tail

- Status: completed
- Owner: researcher and Agent
- Gate: G-007
- Claim: C-007
- Candidate: L-007
- Lane: explore
- Resource cap: existing four F81 captures, at most two H200-hours, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether the latest numerator bottleneck is caused by the positivity
and shared-normalization restrictions of prior linear tails, or whether even a
signed content-generated numerator basis lacks sufficient capacity or transfer.

## Milestones

1. Freeze split, exact support, ranks, objective, selection rule, outcome map,
   and arithmetic guard.
2. Implement the signed numerator and separate positive-partition branches
   without modifying prior experiment code or artifacts.
3. Unit-test exact-support recovery, denominator positivity, signed weights,
   adapter identity initialization, and exact-replacement algebra.
4. Train all six architectures on calibration only, select once on validation,
   and evaluate once on test.
5. Run the selected target-exposed capacity diagnostic, analyze the error
   decomposition, and close G-007.

## Stop rules

- Do not add ranks, supports, losses, query tiles, identities, or training steps
  after reading outcomes.
- Stop after one valid terminal outcome or one allowed pre-outcome repair.
- Do not infer H200 speed from arithmetic work or promote one-cell evidence to
  Wan-wide behavior.
- A transductive exact-partition miss above 1% / 2% stops this function class on
  the registered cell.

## Non-goals

- No QKV fine-tuning, distillation, rollout, VBench, kernel, or speed claim.
- No revival of fixed BCM/BCCB/Butterfly output reconstruction.
- No change to the stopped temporal-transition mainline.
