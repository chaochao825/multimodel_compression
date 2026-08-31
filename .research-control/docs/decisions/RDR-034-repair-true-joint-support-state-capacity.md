# RDR-034: Repair the support-state Gate to test actual joint adaptation

- Status: accepted
- Date: 2026-09-01
- Decider: researcher through the explicit requirement to compare independently
  trained and jointly trained support-state methods without letting engineering
  omissions hide potential
- Supersedes: only RDR-033's implication that a fixed whole-state plus adaptive
  support is an upper bound on joint support-state training

## Context

EXP-050 found a strong support interaction but missed local fidelity. Its
residual-aware support reduced risk by 59.414% versus mass support at identical
payload, yet visual mean remained 6.759%. The registered state was frozen from
whole-measure training. It never adapted its feature map to the residual left
by the selected exact pages.

This is a missing factorial cell, not a request for more width, density,
irregular support, data, or a different endpoint. A true joint test must let the
same width-32 state train under residual-aware support and compare it against a
state trained under mass support from an identical initialization and batch
schedule.

## Decision

Accept EXP-051 / G-030 under the existing C-029 / L-029 side probe. Use the
repaired exact-page capture, calibration positions 1--72, exposed development
positions 73--96, and the same 25% regular support and width-32 payload.

Add a 2x2 diagnostic cross:

- independent state with mass support;
- independent state with residual support;
- joint state with mass support;
- joint state with residual support.

The final joint arm must beat the independent baseline and both one-factor
changes. Otherwise the result cannot be called support-state synergy.

## Guards and consequences

- Both trainable arms start from the same frozen whole-state checkpoint and use
  identical calibration batches, optimizer steps, learning rate, and seed.
- Development residual support remains target-visible and is a capacity ceiling
  only.
- No development target updates model parameters.
- All exact-recovery and finite-gradient guards remain required.
- A valid G-030 miss closes the width-32, 25%-regular-page post-hoc capacity
  line before router, task, confirmation, or H200 work.
- A pass authorizes a separately frozen deployable-router protocol; it does not
  itself authorize later data or a system claim.
