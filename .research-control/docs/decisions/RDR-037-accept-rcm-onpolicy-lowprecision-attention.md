# RDR-037: Accept an rCM-on-policy low-precision dense-attention Gate

- Status: accepted
- Date: 2026-09-01
- Decider: researcher through the continuing objective to retain the
  quality-passing rCM NFE baseline and then add FP8 dense, fused attention, and
  exact system optimization
- Supersedes: none

## Context

EXP-052 established the exact resident rCM4 F81 request at `9.637995s`, with
`3.205365s` denoising. EXP-053 closed the first exact VAE schedule as an
exactness null. The next permitted family is therefore same-step fused
low-precision dense attention, measured against L-030 rather than against the
old teacher20 service boundary.

Historical H200 evidence is mixed. Installed Sage SM90 dense attention reached
about `1.5906x` over FA3 BF16 and one exposed cell passed the registered local
error thresholds. A prospective teacher20 atlas certified only `33.3%` of its
sampled cells, so static precision islands were insufficient there. Those
results cannot be copied to rCM because the four-step student has different
weights and an on-policy trajectory, but they close ordinary FA3 FP8, per-head
two-call islands, and custom post-hoc correction as default successors.

Using EXP-052 component shares and the historical Sage speed, optimistic full
self-attention coverage saves only `0.6413s`, producing a `1.0713x` request
ceiling. A `1.05x` request requires `71.57%` safe coverage before integration.

## Options

1. Test a fresh rCM4 on-policy whole-cell Sage/FA3 atlas under a prospective
   quality and materiality Gate.
2. Reuse the teacher20 atlas. This is invalid because weights, timesteps, and
   trajectory differ.
3. Reopen per-head precision islands or add low-rank/sparse corrections. Prior
   evidence shows extra calls and rotating defects erase their benefit.
4. Stop at the exact `9.637995s` incumbent.

## Decision

Accept option 1 as `C-032 / L-032 / EXP-054 / G-033`.

- A cell is one complete `rCM step x Wan layer` self-attention call.
- Its only actions are installed Sage SM90 dense attention or official FA3
  BF16. Cross-attention remains official.
- Calibration computes both outputs but always returns FA3, so the atlas is
  learned on the exact baseline trajectory.
- The frozen atlas is static and whole-cell. There is no runtime router, head
  split, low-rank correction, sparse tail, BCM, Butterfly, cache, or fallback
  decision beyond its registered FA3 cells.
- The atlas must have zero held-out false-safe cells and cover at least 87 of
  120 cells. It must also imply at least `1.05x` using the newly measured Sage
  speed, not only the historical value.
- Only a passing atlas may enter candidate rollout. Promotion requires paired
  quality guards and at least `1.05x` complete resident-request speedup over
  EXP-052.

## Consequences

- L-030 remains the immutable exact incumbent; L-032 is one bounded side probe.
- A local coverage or transfer miss stops before video generation and closes
  train-free static low-precision dense attention on the released rCM path.
- A quality pass with a speed miss is a speed boundary, not an algorithm pass.
- One isolated H200, four GPU-hours, 10 GiB artifacts, and at most two bounded
  implementation repairs are allowed.
- Reopening requires a changed rCM checkpoint/backend or a trainable
  quantization policy under a new decision.
