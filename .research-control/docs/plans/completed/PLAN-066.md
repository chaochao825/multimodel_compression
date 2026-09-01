# PLAN-066: Select the next exact resident-rCM system successor

- Status: completed
- Owner: researcher and Agent
- Mainline: L-030
- Side probes: none
- Lane: candidate selection
- Resource cap: read-only evidence synthesis; no GPU job or new model run

## Decision to unlock

Select one next candidate that can materially improve the exact `9.637995s`
resident rCM4 request after the exact VAE schedule and train-free static
low-precision attention candidates both closed as valid nulls.

## Evidence boundary

- EXP-052 component shares: VAE `44.7%`, denoiser `33.3%`, serialization
  `18.6%`, text `0.7%`.
- EXP-053: the frozen exact temporal-grouping VAE candidate failed F81 equality
  and both registered speed guards.
- EXP-054: Sage SM90 reached `1.586377x` local speed, but the static whole-cell
  atlas selected `0/120` cells under the frozen safety margins.
- L-030 remains the only incumbent; no successor is accepted by this plan.

## Candidate surface

1. Exact serialization/transfer overlap or codec pipeline optimization.
2. Exact official-VAE kernel/backend optimization without temporal state reuse.
3. Separately accepted trainable low-precision rCM attention or checkpoint.

## Steps

1. Reconcile existing component traces and Amdahl ceilings against L-030.
2. Compare expected decision value, implementation readiness, exactness risk,
   and required H200 budget for the three candidate families.
3. Produce one successor recommendation and one proposed RDR for researcher
   acceptance before any implementation or GPU execution.

## Stop rule

Stop after one decision surface and proposed successor. Do not open an
experiment, modify model numerics, relax EXP-053/054 thresholds, or start a GPU
job without a separately accepted RDR.

## Outcome

The component/source audit selected exact full-F81 VAE CUDA Graph replay as the
single successor. VAE is 44.7% of the resident request and needs only `1.119x`
local speed for a `1.05x` request; measured Sage at optimistic full coverage
would yield only `1.070935x`, while train-free safe coverage was `0/120`.
Transfer/serialization remains second because its exact single-request contract
and overlap semantics are unresolved. The decision surface and bound data are
in `WAN_RCM_SUCCESSOR_SELECTION_PLAN066_20260901.zh-CN.md`; RDR-038 is proposed
and no GPU/model run was opened.
