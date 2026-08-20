# PLAN-051: Production-UniPC finite-jump training feasibility Gate

- Status: completed
- Owner: researcher and agent
- Gate: G-022
- Claim: C-022
- Candidate: L-022
- Experiment: EXP-043
- Lane: explore
- Resource scope: two development identities, one exclusive H200, no VAE
  decode, no calibration/validation/test access

## Decision question

Does a 28,160-parameter frozen-backbone finite-jump adapter learn a transferable
20-to-4 production-UniPC correction on one held-out development identity, and
does current-observable motion alignment materially outperform the identical
unaligned structure?

## Ordered execution

1. Implement and unit-test production-UniPC anchor capture and payload audits.
2. Capture only `dev00_cloth` and `dev01_turntable` at 20 steps, retaining five
   FP32 states and four FP32 guided predictions per identity.
3. Train every registered arm on `dev00_cloth` for a fixed 2,000 steps without
   reading `dev01_turntable` targets for fitting or checkpoint selection.
4. Evaluate `dev01_turntable` once, audit hashes and finite values, and classify
   the screen before any additional arm, rank, loss, or identity is introduced.
5. Produce a compact result report and plot. No rollout or external quality
   claim is authorized by this plan.

## Frozen arms

- zero first-order correction;
- raw-noisy unaligned lifting;
- clean-prediction unaligned lifting;
- current-observable motion-conjugated lifting;
- terminal-target-exposed motion capacity diagnostic;
- exactly parameter-matched pointwise local control.

## Development screen

- Capacity signal: target-exposed endpoint aggregate/worst at most 0.5%/1%.
- Transfer signal: observable endpoint aggregate/worst at most 1%/2%.
- Structural signal: observable motion alignment improves displacement-weighted
  residual error by at least 25% over identical clean unaligned lifting.
- Sanity signal: observable motion is not worse than the same-budget pointwise
  control on both aggregate and worst endpoint error.

The screen is `promising` only if all four signals pass. Any miss is a
development null for this tiny adapter and stops rank, loss, expert, and warp
growth under EXP-043.

## Guards

- UniPC is identified as a production trajectory teacher, not an ODE truth.
- The screen identity may be loaded only after every final checkpoint is saved.
- Target-exposed motion cannot select or modify a deployable arm.
- No published video metric, measured end-to-end speedup, or SOTA claim follows
  from open-loop teacher-state replay.
- Stop on foreign GPU overlap, non-finite payload, hash drift, identity leakage,
  or any post-outcome method-family change.

## Decision effect

- `promising`: open a new fresh-identity Gate comparing the tiny adapter, equal
  LoRA, and released rCM/TurboWan under closed-loop 4-step rollout.
- `capacity-null`: park the tiny adapter and use a released full student.
- `alignment-null`: retain generic finite-jump distillation but remove motion
  conjugation from the main claim.
- `transfer-null`: require a larger trainable student only under a new RDR; do
  not add ranks to this Gate.
- `engineering-failure`: repair only capture/schema defects without changing
  identities, arms, thresholds, or teacher semantics.

## Outcome

EXP-043 completed with `capacity-null`. Target-exposed and observable-motion
endpoint errors were 11.115% and 10.901%; every trained arm was worse than the
9.451% zero correction. L-022 is parked. The deterministic sampler assigned
1,976 of 2,000 updates to the final interval, so this closes the registered
protocol without establishing a general adapter-capacity limit.
