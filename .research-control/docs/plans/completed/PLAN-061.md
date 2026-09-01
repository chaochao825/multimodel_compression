# PLAN-061: Establish the exact rCM warm-service floor

- Status: completed
- Owner: researcher
- Mainline: L-030
- Experiment/Gate: EXP-052 / G-031
- Resource scope: one isolated H200, at most four GPU-hours and 10 GiB

## Objective

Determine whether the released quality-passing rCM endpoint reaches the
registered `2.5x` end-to-end target after removing only repeated UMT5
construction and fixed-negative encoding, with every distinct positive prompt
still fully encoded and all generated values exact.

## Steps

1. Implement the separate resident-text harness without changing EXP-047.
2. Run the four-distinct-prompt text-only exactness and latency screen.
3. If S0 passes, run all three methods through the F17 exact-generation smoke.
4. If S1 passes, measure the three-method four-prompt F81 resident Pareto.
5. Close G-031 and select exactly one next action from the measured bottleneck.

## Outcome

EXP-052 passed. The resident runtime was bitwise exact, saved at least
`15.208s/request` in the text screen, and reached `4.031x` resident F81 warm
speedup with rCM4. The exact `9.638s` rCM endpoint is now the mandatory
baseline. Its largest measured component is VAE decode (`4.308s`), followed by
the denoiser (`3.205s`) and serialization (`1.796s`); no approximate kernel was
added inside this plan.

## Stop rule

Stop on one valid Gate outcome, exactness or isolation failure, less than
`5.5s/request` text saving, two bounded repair attempts, four H200-hours,
10 GiB artifacts, or user interruption. No approximate kernel may be folded
into this Gate.
