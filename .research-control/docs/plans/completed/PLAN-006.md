# PLAN-006: Hold the closed portfolio after EXP-006

- Status: completed
- Owner: researcher and Agent
- Open Gate: none
- Active candidate: none
- Resource cap: no new experiment, training, rollout, or GPU kernel work

## Purpose

Preserve the completed temporal-transition, structure-scout, CMAQ, and
comparative sampling evidence without implicitly reopening a failed function
class. This plan is a holding state, not authorization for another probe.

## Current decision surface

- Claims `C-000` through `C-006` are refuted within their registered scopes.
- Gates `G-000` through `G-006` are closed.
- Per-query proposal improves numerical error but does not pass the strict local
  oracle gate and inflates a 64x64 support union to 95.44%.
- Exact partition estimation does not rescue the estimator; vector numerator
  approximation is the latest localized bottleneck.

## Next authorized action

Continue engineering hygiene and evidence comparison without starting a new
scientific experiment. Any learned numerator tail, relaxed trajectory objective,
new data coverage, rollout, or kernel requires a fresh researcher decision,
claim, candidate, protocol, cost cap, stop rule, and Gate.

## Non-goals

- No post-hoc tuning or method addition to EXP-006.
- No proposal learner, CUDA/H200 kernel, rollout, or speed claim.
- No change to the stopped mainline or protected historical artifacts.
