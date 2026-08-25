# Action-DiT transported-cache innovation gate

Date: 2026-08-26

## Decision question

Does receding-horizon alignment turn the FFN residual reused across control ticks
into an innovation that is materially easier to represent with a cheap
non-periodic temporal operator or a calibration-fixed low-rank basis?

This gate tests a necessary statistical mechanism. It does not test environment
success, a learned cache scheduler, low-bit kernels, or end-to-end speed.

## Frozen scope

- Model family: the three existing PushT low-dimensional Transformer Diffusion
  Policy checkpoints used by the 2026-08-26 structured-correction study.
- Frozen modules: the policy, normalization, scheduler, attention, FFNs, and
  output head are never updated.
- Target: each decoder layer's `linear2` FFN residual, because selected FFNs
  account for about 55.31% of the measured model MAC proxy while attention-score
  computation accounts for about 0.68%.
- Data separation: all operators and bases are fit only on transitions from
  training episodes. Every reported transfer metric uses validation episodes.
- Control offsets: `m=8`, the checkpoint's deployed `n_action_steps`, is the
  primary setting. `m=1` is a preregistered frequent-replanning diagnostic and
  cannot rescue a failed `m=8` result.
- Reused support: both raw and shifted reuse replace exactly `H-m` tokens and
  retain the exact current FFN residual for the `m` new tail tokens. Their
  arithmetic budget is therefore identical.
- Noise: the primary capacity test aligns noise on physically overlapping
  action tokens. Independent current-tick noise is a negative control. This is
  a teacher-forced geometry test, not a claim about the existing stochastic
  sampler.
- Flow points: ten fixed locations from the checkpoint's 100-step schedule,
  paired with the immediately preceding denoising step.
- Layer coverage: all eight decoder layers.

## Frozen method family

1. `raw_reuse`: same-position previous-tick FFN residual on the reusable support.
2. `shift_reuse`: previous-tick residual shifted by the control offset.
3. `shift_toeplitz_r2`: shifted reuse plus a calibration-only, per-channel,
   non-periodic radius-2 temporal map from the current-versus-shifted FFN-input
   drift to the cache innovation.
4. `shift_circular_r2`: equal-parameter periodic control.
5. `shift_rank8_feature`: calibration-fixed rank-8 innovation basis with
   coefficients predicted from current cheap drift features.
6. `shift_rank8_prev_flow`: the same basis with a one-step coefficient-memory
   ceiling that may consume the oracle coefficient at the preceding flow step.
7. `shift_rank8_oracle`: held-out coefficients projected onto the
   calibration-fixed basis. This is capacity only and is never deployable.
8. `flow_reuse`, `flow_toeplitz_r2`, and `flow_rank8_oracle`: same-control-tick
   adjacent-flow diagnostics using the immediately preceding exact denoising
   step. These establish the standard cache reference but are not a novelty
   claim.

Each candidate is injected at one FFN at a time. All upstream computation and
the complete downstream suffix remain exact. The primary risk metric is the
resulting velocity-output error, not hidden-state reconstruction alone.

## Metrics

- FFN residual relative L2 on reused tokens.
- Exact-suffix velocity-output aggregate relative L2, sample mean, and P95.
- Improvement of shifted reuse over equal-budget raw reuse.
- Improvement of deployable Toeplitz repair over shifted reuse.
- Calibration-fixed rank-4/rank-8 held-out innovation energy.
- Rank-8 coefficient prediction R2 for cheap current features and prior-flow
  coefficient memory.
- Aligned-noise versus independent-noise sensitivity.
- Arithmetic ceiling from the actual overlap `(H-m)/H` and the measured FFN
  MAC fraction.

## Decision rule

`MECHANISM_GO` requires all three checkpoints at `m=8` to satisfy:

- shifted exact-suffix risk is at least 20% below raw-reuse risk;
- non-periodic radius-2 repair removes at least 25% of shifted-reuse risk and
  does not worsen P95;
- calibration-fixed rank-8 captures at least 70% of held-out cache-innovation
  energy.

`BOUNDARY` is assigned when the effect exists only for `m=1`, only for oracle
coefficients, or is statistically positive but the actual overlap gives less
than a 1.2x denoiser arithmetic ceiling. `NO_GO` is assigned when physical shift
does not beat equal-budget raw reuse or rank-8 captures less than 50% on any
checkpoint.

No outcome from this gate authorizes a cache scheduler or precision claim.
Only `MECHANISM_GO` may open a fixed-schedule cache-by-precision experiment.
