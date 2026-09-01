# PLAN-064: Select the first material kernel stack above resident rCM

- Status: active
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
