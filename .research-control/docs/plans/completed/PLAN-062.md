# PLAN-062: Select the first optimization above the exact rCM runtime

- Status: completed
- Owner: researcher
- Mainline: L-030
- Experiment/Gate: none; successor selection only
- Resource scope: repository analysis and existing measurements; no new GPU run

## Objective

Use the completed EXP-052 component profile to select exactly one bounded
successor Gate against the `9.638s` exact resident rCM baseline. Compare exact
VAE/transfer/serialization work with the existing FP8 dense-attention evidence
using measured H200 shares and complete candidate overhead.

## Steps

1. Preserve EXP-052 as the mandatory baseline and separate cold-start from
   persistent-service claims.
2. Compute the end-to-end ceiling for each existing exact and approximate
   candidate from the resident component shares.
3. Reject candidates whose zero-overhead ceiling cannot materially improve the
   endpoint or whose quality evidence belongs to a stopped function class.
4. Present one successor claim, cost cap, stop rule, and comparison protocol for
   researcher acceptance before any new approximation or H200 run.

## Selection outcome

The measured Amdahl comparison selects one proposed successor: exact Wan VAE
temporal scheduling under `RDR-036 / EXP-053 / G-032`. VAE occupies 44.7% of
the resident endpoint and has a `1.808x` elimination ceiling; a `2x` VAE would
yield about `1.288x` endpoint improvement. Historical `1.51x` self-attention,
even with optimistic full coverage, yields only about `1.064x` endpoint
improvement and carries unresolved quality coverage.

RDR-036 is accepted under the researcher's continuing exact-system objective.
The candidate implementation and local tests exist. Detailed evidence is in
`worldfoundry_hybrid_residual/results/WAN_RCM_SUCCESSOR_SELECTION_PLAN062_20260901.zh-CN.md`.

## Outcome

PLAN-062 selected exact VAE temporal scheduling as the only successor Gate and
closed. EXP-053 starts with the F17 bitwise screen and may advance only through
its frozen staged guards.

## Stop rule

Stop after one evidence-backed successor proposal or a documented decision that
no measured candidate justifies a new Gate. Do not alter rCM, run new kernels,
or reopen BCM/Butterfly/whole-block residual families inside this plan.
