# PLAN-069: Preserve the exact rCM incumbent after EXP-055

- Status: active
- Owner: researcher and Agent
- Mainline: L-030
- Lane: integrate
- Resource cap: no experiment or GPU allocation

## Decision to unlock

Select the next bounded successor to the exact `9.637995s` resident rCM4
baseline without promoting the component-only EXP-055 result or reopening a
closed approximation family.

## Current evidence

- EXP-052 remains the exact resident-service incumbent.
- EXP-055 preserves bitwise output and accelerates the complete VAE by
  `1.1367x`, but its `9.3257s` request is only `1.0335x` faster than the frozen
  incumbent and therefore misses the registered `1.05x` gate.
- No new scientific claim, Gate, experiment, or GPU run is authorized by this
  administrative plan.

## Milestones

1. Preserve and publish the valid EXP-055 speed-boundary result.
2. Compare the remaining exact transfer/serialization cost with the separately
   motivated denoiser and NFE candidates using existing evidence only.
3. Open at most one successor after a separate accepted decision fixes its
   claim, protocol, identities, cost accounting, and stop rule.

## Stop rules

Do not run a new experiment, combine EXP-055 with an unregistered optimization,
or alter the incumbent threshold under this plan. Complete or supersede this
plan only after a separate decision selects the next bounded candidate.
