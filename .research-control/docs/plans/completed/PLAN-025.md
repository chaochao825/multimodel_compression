# PLAN-025: Audit the static FP8 attention coverage ceiling

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed evidence synthesis
- Claims: none
- Candidate line: exact dense mixed-precision system anchor
- Lane: explore
- Resource cap: one read-only analysis script, one report/figure, no new GPU run

## Decision to unlock

Determine whether the already measured FA3 FP8/BF16 attention path leaves
enough static-schedule headroom to justify a new F81 rollout campaign. Avoid
repeating existing kernel and one-step generation experiments.

## Frozen evidence

1. `results/acceleration_frontier_v1/attention.csv`: F81 self-attention share,
   FA3 FP8 kernel speed, and local dense-relative output error.
2. `results/attention_lowrank_audit_v1/raw/f81_repeat_generation_runs.csv`:
   five order-alternated F81 paired runs of BF16 attention versus one middle
   denoising step of FP8 attention.
3. `results/acceleration_frontier_h200_v1/H200_MEASURED_RESULTS_20260726.zh-CN.md`:
   execution and interpretation boundary for the H200 measurements.

All inputs are outcome-exposed historical evidence. This plan may synthesize
them but cannot create a new generalization, rollout-quality, or speed claim.

## Registered analysis

For attention share `a`, FP8 kernel speed `s`, and fraction `f` of attention
calls routed to FP8, compute the optimistic denoiser bound:

\[
S(f;s)=\frac{1}{1-a+a[(1-f)+f/s]}.
\]

Report:

1. required FP8 coverage for denoiser upper bounds `1.05x`, `1.10x`, `1.20x`,
   and `1.30x` at measured `s=1.512x` and hypothetical `2x/3x/4x` kernels;
2. observed paired speedup for the existing one-of-20-step FP8 schedule;
3. local output error versus the existing `1% / 2%` screening boundary;
4. the difference between denoiser upper bound and full end-to-end speed.

No quality model may infer SSIM from local L2. The one-step videos have hashes
but no registered paired quality table, so their quality remains unknown.

## Outcome map

- `static-rollout-headroom`: measured-kernel required coverage for `1.20x` is at
  most 80%, local error is at most 2%, and existing one-step timing is positive.
  Open a bounded multi-prompt static-schedule rollout Gate.
- `static-mixed-precision-boundary`: required coverage exceeds 80%, local error
  exceeds 2%, or the one-step schedule has no positive timing signal. Do not
  sweep more static schedules; retain FP8 as a kernel component for a trained or
  fused heterogeneous method.
- Standard invalid or contradictory outcomes retain precedence.

## Stop rules

Stop after the deterministic synthesis. Do not run Wan, tune a schedule, change
quality thresholds, or introduce a learned controller in this plan.

## Closure

Write the evidence table, coverage plot, bound report, and update `STATUS.md`.
Keep claim, candidate, experiment, and RDR registries unchanged.

Closed 2026-08-12 with outcome `static-mixed-precision-boundary`.

- The measured `1.512478x` FA3 FP8 attention kernel requires `91.291%`
  attention-call coverage to reach an optimistic `1.20x` denoiser bound,
  exceeding the registered `80%` boundary.
- Full-FP8 local attention output relative L2 was `13.7629%`, above the
  registered `2%` screen. This local metric does not imply SSIM or VBench.
- The one-of-20-step schedule has a `1.009212x` denoiser upper bound. Five
  paired full-pipeline repeats reached `1.002925x` geometric-mean speedup
  (`0.982740x`--`1.009812x`); paired video quality remains unknown because no
  registered quality table exists.
- Decision SHA-256:
  `787784b1e7113bd09d5d0eaea15a8cc2eb5adff6be5ef23e242b4023710af1a6`.
- Manifest SHA-256:
  `8f95aaf9162911f87acd9d716983afd7b3b27b34072da231f7bd981458bb7499`.
- Analysis SHA-256:
  `e6000cbcd77508a8b03d72987075d225013716d1b6fd7fbc15f513f544731488`.
- Evidence and report:
  `worldfoundry_hybrid_residual/results/fp8_static_coverage_ceiling_diag025_v1/`.
- Required action: `DO_NOT_SWEEP_STATIC_FP8_SCHEDULES`; preserve FP8 as a
  component of a semantics-preserving equalization or trained fused operator.
