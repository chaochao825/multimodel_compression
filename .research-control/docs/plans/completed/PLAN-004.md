# PLAN-004: Hold the closed portfolio after EXP-005

- Status: active
- Owner: researcher and Agent
- Open Gate: none
- Active candidate: none
- Resource cap: no new experiment or GPU work

## Purpose

Preserve the completed RCAR, structure-scout, and CMAQ evidence without
implicitly reopening a failed function class. This plan is a holding state,
not authorization for another probe.

## Current decision surface

- All claims `C-000` through `C-005` are refuted within their registered scope.
- All Gates `G-000` through `G-005` are closed.
- Fixed structured output reconstruction, scalar temporal forecasting, frozen
  structure routing, and tile-level centroid PPS-CV remain stopped.

## Next authorized action

Wait for an explicit researcher decision selecting a genuinely new estimator
family or objective. Any new work must open a fresh claim, candidate, protocol,
cost cap, stop rule, and Gate before execution.

## Non-goals

- No parameter tuning or post-hoc rescue of EXP-005.
- No proposal learner, rollout, CUDA/H200 kernel, or speed claim.
- No change to the stopped mainline or protected historical artifacts.
