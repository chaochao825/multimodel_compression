# PLAN-067: Await the exact VAE CUDA Graph decision

- Status: active
- Owner: researcher and Agent
- Mainline: L-030
- Side probes: none
- Lane: protected decision pending
- Resource cap: read-only review; no GPU job, implementation, or model run

## Decision to unlock

Accept, revise, or reject proposed RDR-038. Acceptance would authorize the
bounded `C-033 / L-033 / EXP-055 / G-034` exact full-F81 VAE CUDA Graph Gate.

## Evidence boundary

- Incumbent: exact resident rCM4 at `9.637995s` from EXP-052.
- Closed candidates: temporal VAE regrouping was an F81 exactness null in
  EXP-053; train-free static low-precision attention selected `0/120` cells in
  EXP-054.
- PLAN-066: VAE is 44.7% of the request; `1.119x` VAE yields `1.05x` request,
  while measured full-coverage Sage implies only `1.070935x` request.
- Proposed action: CUDA Graph capture/replay of the unchanged 21-frame official
  VAE decode, with static input-copy and output-handoff costs included.

## Allowed work

Only answer questions about the proposal, inspect existing evidence, or revise
RDR-038 without changing its scientific action family. Do not create C-033,
L-033, EXP-055, or G-034 registry rows and do not implement or execute the
candidate before explicit acceptance.

## Stop rule

Stop at the protected researcher decision. If accepted, replace this waiting
plan with one bounded execution plan tied to RDR-038; if rejected, return to a
new candidate-selection decision without silently opening serialization or
trainable quantization.
