# PLAN-064: Select the first material kernel stack above resident rCM

- Status: completed
- Owner: researcher
- Mainline: L-030
- Experiment/Gate: none; evidence recovery and successor selection only
- Resource scope: existing artifacts and repository analysis; no GPU run

## Objective

After the exact VAE temporal schedule closed as an exactness null, select one
prospective successor against the `9.637995s` resident rCM4 baseline. The
selection must distinguish local operator speed, on-policy coverage, complete
candidate overhead, endpoint materiality, and generation-quality risk.

## Steps

1. Recover the strongest existing H200 evidence for dense FP8/FlashAttention,
   precision islands, exact serialization, transfer, and fusion.
2. Recompute each candidate's endpoint ceiling using EXP-052 component shares;
   reject any candidate whose zero-overhead ceiling is immaterial.
3. Verify whether attention evidence was measured on the rCM four-step
   trajectory and includes Q/K/V conversion, scaling, output conversion,
   fallback, and routing costs.
4. Select exactly one candidate and freeze its local quality, whole-attention,
   endpoint, memory, and H200 isolation guards before implementation or timing.

## Stop rule

Stop after one evidence-backed successor proposal or a documented decision
that no current candidate justifies a new Gate. Do not reopen fixed BCM/BCCB,
Butterfly, static low-rank, sparse-tail, cache, or whole-block residual families
that failed their registered defects. Do not run a GPU job inside this plan.

## Outcome

The selected successor is a calibration-frozen `rCM step x layer` atlas that
chooses one whole-cell implementation: installed Sage SM90 dense attention or
the official FA3 BF16 path. It does not split heads or add a router, residual,
cache, or structural expert. Historical Sage latency is `1.5906x` versus FA3,
but teacher20 static certification covered only `33.3%` of sampled cells.

Against EXP-052, optimistic full coverage saves only `0.6413s` and yields
`1.0713x` resident-request speedup. Reaching `1.05x` therefore requires at
least `71.57%` coverage at the historical kernel speed. This makes a fresh
rCM4 on-policy atlas scientifically necessary and gives it a hard materiality
stop. The candidate is frozen in RDR-037 / EXP-054 / G-033; no GPU run occurred
inside PLAN-064.
