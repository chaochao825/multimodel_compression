# RDR-0005: Authorize one bounded CMAQ capacity probe

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the explicit request to evaluate whether the
  proposed value-aware exact-block and control-variate construction is elegant
  and promising, followed by a simple validation
- Supersedes: none

## Context

Fixed structured outputs, static low-rank tails, positive-linear tails, binary
support refinement, scalar temporal prediction, and the structure-only risk
scout have all failed their registered local transfer or cost gates. RCAR was
retained only as a deterministic hierarchical baseline. A new conceptual
candidate, Certified Multilevel Attention Quadrature (CMAQ), changes the tested
object: after a coarse full-support numerator/denominator estimate, exact block
corrections are sampled as control variates rather than deleting the remaining
attention mass or fitting it with a frozen output basis.

Four existing Wan F81 Layer 14, step 9, conditional Q/K/V captures on server 236
already have frozen prompt/seed identities. They are sufficient for a bounded
numerical capacity and transfer probe without model generation.

## Options

1. Keep CMAQ conceptual and perform no numerical check.
2. Start a full runtime, H200 kernel, or rollout program.
3. Run one split-frozen offline probe over the four existing captures, comparing
   deterministic coarse/top-k baselines with uniform, oracle, and
   calibration-frozen block control-variate sampling.

## Decision

Select option 3 as one bounded side probe. The probe may compute dense block
contributions on the registered captures to establish an oracle ceiling and may
fit a block proposal using calibration identities only. Held-out identities may
not change the proposal, estimator, densities, normalization, random seeds, or
thresholds.

This decision authorizes one numerical script, its unit tests, one complete pass
over the four existing captures, and one pre-outcome engineering repair. It does
not authorize new captures, model training, rollout, a fused kernel, a measured
speed claim, or a mainline change.

## Consequences

- RCAR remains the deterministic comparison baseline; fixed BCM/BCCB/Butterfly
  output fitting remains parked.
- A positive result means only that the new function class deserves a fresh
  proposal/certificate Gate. It does not establish deployability.
- Oracle success with frozen-proposal failure is a boundary result and may
  justify low-cost learned proposals only after another researcher decision.
- Oracle failure closes CMAQ without increasing sample count beyond the frozen
  50% diagnostic ceiling.
