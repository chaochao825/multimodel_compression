# RDR-0016: Authorize equal-cost distributed micro-anchor basis probe

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue theory-driven,
  high-information experiments after valid failures
- Supersedes: none

## Context

EXP-014 established an M8/rank64 shared residual-subspace witness, but EXP-015
showed that one contiguous 64-query dense anchor is not an adequate witness for
the other seven tiles. Validation oracle-best projection was 2.545% / 6.600%
while the non-anchor adaptive floor was 0.711% aggregate. Increasing rank made
the projection/adaptive ratio worse, localizing the failure to row-space
coverage rather than rank or routing.

Adding a second full anchor is not cost-compatible: at 35% support, two anchors
per M8 group cost 51.25% before low-rank lifting. The minimal repair is to keep
exactly 64 dense query rows but distribute them across four regular 16-row
microblocks from distinct parent tiles.

## Decision

Authorize one basis-transfer-only Gate comparing a single-tile baseline,
fixed-spread blocks, two Q-only diversity selectors, and a target-exposed greedy
oracle. The total dense rows, support, group size, ranks, data split, and cost
model remain unchanged from EXP-015.

Non-anchor coefficients remain oracle projections. No coefficient predictor is
run until a legal distributed selector passes 1% / 2%; this isolates whether
distributed row sampling repairs basis observability.

## Consequences

- A legal test pass authorizes a fresh Gate for anchor-row coefficient lifting.
- An oracle-only pass localizes the bottleneck to Q-only microblock selection.
- If the target-exposed distributed oracle fails, park training-free online
  residual-basis generation on this cell and do not add more fixed structures.
- No learned selector, rollout, fused kernel, or speed claim is authorized.

## Pre-outcome implementation repair

Before the valid H200 execution began, code audit found that the probe eagerly
loaded both validation and test capture files even though test metrics were
computed conditionally. The waiting job was stopped before it executed, and the
one permitted engineering repair changed loading order so the test capture is
first opened only after a legal validation pass. No selector, rank, geometry,
metric, cost, gate, seed, or numerical operation changed. The repaired code
locks are `probe:943a0b899d050782`, `core:eebafa1b0f6a6c4a`, and
`test:580c5a66e47953d1`; 20 related remote tests passed.
