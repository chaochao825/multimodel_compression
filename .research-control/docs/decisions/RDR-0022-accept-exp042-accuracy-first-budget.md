# RDR-0022: Accept EXP-042 with an accuracy-first staged budget

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit instruction to validate the path
  without a fixed resource limit and to incorporate prior experimental lessons
- Supersedes: only the fixed two-H200-hour and one-repair limits in PLAN-045 and
  proposed EXP-042; all scientific, leakage, fairness, and stop guards remain
  binding

## Context

PLAN-044 established that historical UniPC/QKV artifacts cannot define the
finite-horizon curvature target. EXP-042 therefore needs a minimal explicit-
Euler trajectory capture and a new evaluator. The original two-H200-hour cap
could confound an implementation repair or Euler convergence check with a
scientific null.

The researcher has explicitly prioritized a thorough validation over a fixed
compute-hour ceiling. This is a protected budget decision, not permission to
search outcomes, alter the test set, or grow the method after a miss.

## Decision

Accept EXP-042/G-021 and remove the fixed GPU-hour and repair-count ceilings.
Replace them with a bounded staged scope:

1. Freeze and test the capture/evaluator before GPU use.
2. Run development identities and the Euler-100/Euler-200 convergence control.
3. Run calibration and validation once under the frozen method family.
4. Open test only after every validation unlock guard passes.
5. Stop on the first valid registered outcome or protocol invalidation.

There is no fixed GPU-hour ceiling inside these stages. Reproducible engineering
repairs are allowed until the evaluator yields a valid development result, but
each repair and rerun must be logged and cannot change the scientific endpoint,
identity split, operator family, arm budget, or gate threshold.

## Storage and execution boundary

- Store tensor artifacts on `/opt/data`, not the nearly full root filesystem.
- Keep fresh payload below 5 GiB by storing only the registered coarse states
  and velocities; QKV, block activations, and VAE decode remain prohibited.
- Use the two idle H200s for exact conditional/unconditional CFG branch
  parallelism where implementation equivalence is verified.
- Unlimited-resource wording does not authorize unrelated sweeps, extra model
  families, post-outcome rank/expert growth, or premature rollout.

## Consequences

- PLAN-045 closes as accepted.
- PLAN-046 implements, tests, and freezes capture/evaluator identities.
- EXP-042 execution begins only after PLAN-046 validates the implementation and
  registers exact hashes.
- A valid null/adverse result still parks L-021 under the original outcome map.
