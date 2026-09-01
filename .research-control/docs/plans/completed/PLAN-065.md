# PLAN-065: Execute the rCM-on-policy low-precision attention Gate

- Status: completed
- Owner: researcher and Agent
- Gate: G-033
- Claims: C-032
- Candidate line: L-032
- Lane: integrate
- Resource cap: one isolated H200, four GPU-hours, 10 GiB, two repairs

## Decision to unlock

Determine whether the released rCM four-step trajectory changes low-precision
attention risk enough for a static whole-cell Sage/FA3 atlas to produce a
quality-preserving, material speed increment over the exact resident runtime.

## Context and source authority

- Incumbent: L-030 / EXP-052 / `9.637995s` resident rCM4.
- Decision and protocol: RDR-037 / EXP-054.
- Historical implementation evidence: DIAG-027--040 and completed PLAN-027--040.
- Official rCM source and checkpoint remain those frozen by EXP-047/052.

## Non-goals

No teacher20 atlas reuse, per-head split, runtime router, FA3 FP8, sparse tail,
cache, low-rank correction, BCM/BCCB, Butterfly, compile, CUDA Graph, VAE
change, or post-hoc threshold relaxation.

## Milestones

1. Implement and unit-test metric aggregation, atlas freezing, patch lifecycle,
   stage identities, and materiality calculations.
2. Pass S0 on one isolated H200, including reference-wrapper equivalence and
   complete Sage backend timing.
3. Run the disjoint calibration/evaluation S1 atlas and stop immediately on a
   coverage or transfer miss.
4. Only after S1 passes, run S2 F17 and S3 final F81 quality/timing.
5. Close G-033, update durable evidence, and push verified progress directly to
   the existing branch.

## Validation

- Local pytest, ruff, py_compile, frozen-config, and diff checks.
- Expected cell and identity counts, no duplicate keys, finite metrics.
- H200 isolation and exact source/checkpoint checks at every measured stage.
- Scientific and system thresholds are exactly those in EXP-054.

## Stop and escalation rules

Stop at the first valid terminal outcome, after two repairs, at four GPU-hours,
10 GiB, isolation loss, or user interruption. Any new action family, threshold,
identity, or training requires a new decision.

## Closure

Write RESULT-EXP-054, update claim/candidate/experiment state, PLAN-065,
PROJECT.md, and the Chinese report. Preserve all failed attempts and leave
L-030 as the incumbent unless every promotion guard passes.

## Outcome

S0 passed patch equivalence and measured `1.586377x` full-shape Sage speed.
S1 completed all 960 records but selected `0/120` cells under the frozen
calibration thresholds, so projected request speed remained `1.000x`. The Gate
closed as `coverage-null` before S2/S3; C-032 was refuted in its registered
class, L-032 was parked, and L-030 remained the incumbent. See
`RESULT-EXP-054.md`.
