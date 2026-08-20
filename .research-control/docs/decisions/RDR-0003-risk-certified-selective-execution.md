# RDR-0003: Open a bounded risk-certified selective-execution probe

- Status: accepted
- Date: 2026-08-10
- Decider: researcher through the instruction to pursue the most reasonable,
  simple direction and validate it only after a positive screen
- Supersedes: none

## Context

The scalar, grouped-scalar, and module-target temporal-transition gates are
closed. Fixed BCM/BCCB/Toeplitz/Butterfly operators, static geometry masks, and
train-free sparse-linear tails also failed their registered transfer or cost
gates. A post-hoc EXP-002 diagnostic nevertheless found that 43.6% of sampled
horizon-1 calls were individually below 2%, with strong block and timestep
heterogeneity. That observation is not a deployable policy because it used only
three blocks and selected calls after seeing each identity.

## Options

1. Continue enlarging a failed structured operator or predictor family.
2. Immediately implement a learned controller or fused kernel.
3. Run one full-model, calibration-frozen screen that only approximates locally
   certified SA/FFN calls and keeps every other call dense.

## Decision

Select option 3 as a bounded side probe. Test only one-step reuse and first-order
Taylor prediction of raw SA/FFN outputs with current-step gates recomputed. The
schedule is selected from calibration identities over all 30 blocks and 20
steps. Reuse requires the preceding output to be dense; Taylor requires the two
preceding outputs to be dense. Cross-attention and all uncertified calls remain
dense.

This decision does not reopen the stopped scalar mainline and does not authorize
kernel or rollout work before the local transfer gate passes.

## Consequences

- A positive local screen opens one untouched paired rollout gate; it is not
  evidence of final-video quality or H200 speedup.
- A capacity-oracle failure closes this function class without schedule tuning.
- A calibration-policy failure with a positive oracle localizes the bottleneck
  to certification transfer rather than predictor capacity.
- Fixed structured-operator expansion remains parked.
