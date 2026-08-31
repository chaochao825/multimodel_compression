# RDR-033: Accept a bounded support-state co-design side probe

- Status: accepted
- Date: 2026-09-01
- Decider: researcher through the explicit instruction to establish the
  theoretical scope and then fairly compare exact-only, state-only,
  independent support-state, and joint support-state
- Supersedes: none

## Context

The Wan mainline remains the released rCM four-step H200 Pareto under L-026.
The reader-memory experiments address a different problem: retaining an
unknown-future-query video memory under a bounded active state. They cannot
establish Wan denoising acceleration.

On exposed VSI calibration/development data, a width-32 additive N/Z state
reduced whole-measure error but remained far outside fidelity. Exact 25%
support plus a separately fitted tail reached 2.368% visual mean error, while
the tail retained 90--94% normalized entropy. This leaves one precise
uncertainty: can support be selected to remove the error that the state cannot
represent, rather than independently preserving attention mass?

## Decision

Accept C-029 / L-029 / EXP-050 / G-029 as one bounded side probe. It does not
change the project north star, C-026, L-026, EXP-047, or its resource budget.

The probe has three ordered stages:

1. A development-only target-visible regular-page capacity ceiling using the
   already trained mergeable state. It compares mass-selected corrections with
   residual-aware corrections at identical exact fraction and state width.
2. Only after a capacity pass, a calibration-trained page router and state are
   evaluated on development data in a four-arm factorial comparison.
3. Only after a deployable development pass, frozen task intervention and
   isolated-H200 matched-latency Pareto evaluation may run.

The untouched positions 97--120, official selection, formal roles, and new
streaming benchmarks remain closed until every preceding stage passes.

## Scientific guards

- `exact-only` and `state-only` are ablations, not equal-cost evidence of
  interaction. The interaction claim is joint versus independent at identical
  page density, state width, exact storage, and read path.
- Deployment value is judged on measured H200 latency frontiers, not parameter
  count, FLOPs, or analytic state size alone.
- Target-visible support is a capacity ceiling only. It cannot choose a router,
  open confirmation, or support a speed claim.
- A numerical replay, hard-budget, or exact-recovery failure is engineering
  invalidity and permits repair without changing the scientific method.
- A valid capacity miss closes this support-state function class. A capacity
  pass followed by router failure is an observability null, not a capacity
  null.
- Task agreement is insufficient by itself. Local visual-measure error remains
  a hard guard against downstream masking by nonvisual context.
- Writer, cold exact storage, page scoring, retrieval, state read, and fallback
  costs are all charged before any system claim.

## Consequences

If joint support-state fails to improve paired risk by at least 25% with a
positive lower confidence bound, or fails the local fidelity ceiling, stop
before training a router. If a deployable joint model does not dominate the
independent arm under equal H200 latency while meeting agreement, harmful-flip,
path-consistency, and compression guards, stop the post-hoc reader-memory line.

A positive result remains a video-understanding side result. It must later beat
recent-window and published streaming-memory baselines on long videos with
multiple questions before it can claim to solve a field-level problem.
