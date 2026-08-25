# Action-DiT independent-noise response bridge gate

Date: 2026-08-26

## Decision question

Can a low-cost response to the known cross-control noise mismatch recover the
low-dimensional innovation capacity of horizon-transported FFN caches without
changing the policy's independent-noise source distribution?

This is Gate B0/B0.5. It precedes any sampler, closed-loop, scheduler,
quantization, or latency claim.

## Prior evidence and fixed scope

- The previous transported-cache experiment used three frozen PushT
  Transformer Diffusion Policy checkpoints.
- At the deployed control offset `m=8`, horizon shift reduced aligned-noise
  exact-suffix velocity risk by 67.66%, but only 20% of action tokens overlap.
- Existing independent-noise geometry shows that shift-only does not beat raw
  reuse at any of the ten observed flow points. This result was known before
  registering B0.5 and is not a fresh endpoint.
- A calibration-fixed rank-8 oracle still improves independent-noise geometry,
  motivating a bounded noise-response test.
- Calibration uses training episodes. All reported bridge metrics use the
  validation episodes and independent random noise. No validation sample may
  fit a basis, coefficient map, bucket, threshold, or method choice.
- Only the deployed offset `m=8`, all eight decoder FFNs, ten fixed flow
  points, and the existing 96/48 calibration/evaluation transition split are
  used.

## Fixed predictors

For the transported cache innovation

\[
e_{\ell,n,k}=h_{\ell,n,k}-P_mh_{\ell,n,k-1},
\]

a rank-8 basis is fit on independent-noise calibration innovations. The
following coefficient predictors are fixed before evaluation:

1. `feature`: the previous cheap FFN-input/noisy-action/condition features.
2. `noise`: known \(\Delta\xi=\xi_k-P_m\xi_{k-1}\) and timestep only.
3. `noisy_delta`: known \(x_{k,n}-P_mx_{k-1,n}\) and timestep.
4. `state`: coefficients measured at the immediately preceding exact flow
   anchor. This is causal for one skipped flow step but not an open-loop result.
5. `state_noise`: previous-flow coefficients plus \(\Delta\xi\).
6. `state_noisy_condition`: previous-flow coefficients, noisy-action delta,
   observation-condition delta, and timestep.
7. `oracle`: held-out projection onto the calibration-fixed basis. It is a
   capacity ceiling and never a runnable method.

The equal-budget controls are raw reuse, horizon-shift reuse, and a
calibration-fit radius-2 depthwise local correction. The current tail tokens
remain exact for every method.

## Metrics

- Independent-noise activation relative L2 by layer and flow point.
- Exact-suffix velocity aggregate relative L2, sample mean, and P95.
- Coefficient prediction \(R^2\) for each bridge.
- Oracle-gap recovery

\[
R_{gap}=\frac{D_{shift}-D_{candidate}}
{D_{shift}-D_{oracle}}.
\]

- All-flow and final-three-flow summaries.
- Stored basis/map parameters and an arithmetic correction-cost proxy.

## Registered decisions

`LATE_SHIFT_GO` requires, on every checkpoint, the independent-noise local
correction over the final three flow points to reduce exact-suffix velocity
risk by at least 20% versus raw reuse without worsening P95.

`NOISE_BRIDGE_GO` requires, on every checkpoint, the noise-only bridge over the
final three flow points to:

- recover at least 50% of the shift-to-oracle gap;
- reduce exact-suffix risk by at least 20% versus the local correction;
- not worsen P95 versus the local correction.

`STATE_BRIDGE_BOUNDARY` requires the state-plus-noise bridge to recover at least
80% of the shift-to-oracle gap, reduce late exact-suffix risk by at least 20%
versus the local correction, and not worsen P95. This opens only a
multi-skip-state experiment because the current state consumes an immediately
preceding exact-flow coefficient.

If only the oracle passes, or a predictor works on fewer than all three
checkpoints, the result remains `BOUNDARY` or `NO_GO`. No result from this gate
authorizes aligned-noise sampling, learned scheduling, environment claims, or
GPU speed claims.

## Registered multi-skip extension

The observed `STATE_BRIDGE_BOUNDARY` may be followed by one teacher-forced
open-loop test. It must preserve the calibration/evaluation split, frozen
rank-8 bases, ridge maps, independent-noise source, `m=8` transport, and exact
tail semantics above. No basis, map, bucket, skip length, or threshold may be
selected on validation data.

For each held-out endpoint, an exact coefficient state is taken at one, two,
or four immediately preceding scheduler steps. The registered state map is
then applied recursively. After the anchor, no exact current-window FFN
innovation may enter the coefficient recurrence. A change of timestep bucket
uses only the calibration-fixed affine coordinate transform between its two
rank-8 bases. The previous-control-window cache at the target timestep remains
exact, so this experiment tests denoising-time state recurrence, not repeated
control-cycle cache drift or a sampler rollout.

The one-step recurrent result must numerically match the existing teacher-state
path. On every checkpoint, `MULTISKIP_2_STABLE` requires the two-step
state-plus-noise path over the final three endpoints to recover at least 70% of
the shift-to-oracle gap and not worsen P95 versus the local correction.
`MULTISKIP_4_STABLE` additionally requires the four-step path to recover at
least 50% of that gap, not worsen P95 versus the local correction, and have
aggregate risk no greater than 1.5 times the one-step recurrent risk.

Passing this extension permits only a later triangular sampler experiment with
an exact refresh interval no longer than the longest passing skip. It does not
authorize a closed-loop PushT score, latency, or deployment claim.

## Registered B1a independent-sampler gate

If all three checkpoints pass `MULTISKIP_4_STABLE`, one distribution-preserving
sampler test is allowed before aligned-noise or warm-start experiments. The
previous and current control windows use independent Gaussian initial noise.
For each current-window comparison, full and TQC sampling must use the same
initial latent and identically seeded DDPM scheduler randomness. Sampler bases
and state maps are refit on full training trajectories only; validation
trajectories may not alter them or the schedules below.

Two schedules are fixed:

1. `all_interval5`: one exact refresh followed by at most four recurrent TQC
   steps over the complete DDPM trajectory.
2. `late20_interval5`: exact execution for the first 80% of scheduler calls,
   followed by the same one-exact/four-recurrent cycle.

At a recurrent step, all eight FFN `linear2` outputs use transported cache plus
the rank-8 state-noise innovation on the two overlapping action tokens. The
eight non-overlapping tail tokens are computed exactly. An exact refresh updates
the coefficient state from the current approximate sampler trajectory. The
simulation may compute the replaced outputs for fidelity instrumentation, so it
cannot support a speed claim.

For each checkpoint and schedule, report physical-action aggregate relative L2,
sample P95, RMSE, the executed eight-action chunk, and the first executed
action. `SAMPLER_ALL_GO` requires `all_interval5` to achieve aggregate relative
L2 at most 1% and sample P95 at most 2% for both the full horizon and executed
chunk, with first-action P95 at most 2%, on every checkpoint.
`SAMPLER_LATE_BOUNDARY` uses the same thresholds for `late20_interval5` if the
all-step schedule fails. Only `SAMPLER_ALL_GO` may proceed directly to a paired
closed-loop PushT non-inferiority experiment. A late-only pass remains a
mechanism boundary because its PushT arithmetic ceiling is negligible.

If the all-layer sampler fails, train-0 may be used once for explanatory
layer attribution. Before inspecting train-1 or train-2 layer results, two
static sets are frozen: layers `{1,2}` as the discovery-safe set and layers
`{1,...,7}` as the layer-0 exclusion set. Their train-1/train-2 results are
transfer diagnostics, not a replacement for the failed all-layer endpoint.
Even a successful `{1,2}` transfer cannot authorize a system claim because it
touches only one quarter of layers and 20% of action tokens.
