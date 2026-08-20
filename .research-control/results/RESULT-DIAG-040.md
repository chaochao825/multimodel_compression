# RESULT-DIAG-040

- Experiment: DIAG-040 / PLAN-042
- Date: 2026-08-12
- Candidate: pinned Sparse-VideoGen stateful SAP/EAR Wan operators
- Candidate identity: commit
  `f89aedaf169ac2ae5b186bda674e53c3dc08c476`, tree
  `44b97a3df39700cc02ed6dc5511f9251db8c2b07`
- Protocol SHA256:
  `8605fbaca642fbdfb9a635012861c8ddfc6506e9b2a3e9a56fd511b01bc9dcfa`
- Probe/core/runner SHA256:
  `668c6cca3fa707935500ca9b33f224f6f8a176aec57b36f8228eca10fbc7bb3c`,
  `794518bf3da7e68b5913a40d16e922ab4ee013f6ee670649b4db59cfc9515c8e`,
  `40e0a3d36f9cbf316f2f024fd76797231aa06cd89fc51fe7fc7b08e2fa637fa8`
- Data: globally exposed `s00_p00_seed20260850`, Layer 14, conditional
  branch, F81 steps 7--10; all four payload hashes matched the protocol
- Environment: Python 3.11.15, Torch 2.5.1+cu121, FlashInfer 0.2.10,
  cuVS 25.06.01, CUDA 12.9 JIT toolchain, H200 NVL SM90
- Smoke artifact manifest SHA256:
  `aedd400af4d050c1a7c72fb941934752880697d13033e181c6d27938452f3d7b`
- Native artifact manifest SHA256:
  `c3ecd714aaa3341d1151a444185e948edb609a22f949b63402ff2f66a3c202c2`
- Evidence tier: exposed-data development quality/cost diagnostic
- Validity: valid
- Outcome class: `development-null`

## Observations

The same-run FlashInfer BF16 control passed every step and the pooled summary
at `0.1670--0.1675%` aggregate relative L2. Stateful SAP failed every local
quality guard: pooled aggregate/worst-head/worst-tile errors were
`8.0689% / 10.0064% / 10.4729%`. Stateful EAR materially reduced those values
to `3.9080% / 5.5455% / 5.6396%`, relative reductions of
`51.6% / 44.6% / 46.2%`, but still failed every `1% / 2% / 2%` guard.

The failure persisted after the shifted initialization step. EAR aggregate
error was `4.1940%`, `3.7313%`, and `3.6436%` at steps 8, 9, and 10, so the
missing true pre-step-7 centroid history does not explain the development
decision. SAP remained at `7.7106--8.4236%` over those steps.

At the frozen step-9 timing cell, same-run FlashInfer dense, SAP, and EAR
median complete wall times were `18.5836`, `14.6354`, and `23.6853 ms`.
Their speedups were `1.0000x`, `1.2698x`, and `0.7846x`; all CVs were below
`1.5%`. SAP exceeded the `12.3891 ms` time required for `1.5x` by `18.1%`.
EAR was `27.5%` slower than dense.

Step-9 weighted exact token-pair density was `16.8102%` for SAP and
`16.8328%` for EAR, with per-head ranges of `10.1187--43.5326%` and
`10.1728--43.3442%`. Thus EAR's quality improvement came from its centroid
compensation rather than denser exact support, but the compensation eliminated
the sparse path's speed benefit. Peak allocated/reserved memory was about
`6.17 / 6.49 GB`.

The shifted step-7 initialization used all 50 query iterations and 40 key
iterations. Q/K K-means alone took `40.563 s`; known lower bounds excluding
the untimed shared permutation were `40.970 s` for SAP and `45.126 s` for EAR.
Subsequent two-iteration Q/K updates took `4.66--5.16 ms`.

## Integrity and deviations

Six pure metric/state/density/outcome assertions passed before GPU execution.
The frozen environment lacks pytest's optional `pygments` dependency, so the
same assertions were invoked directly without installing packages. Smoke then
verified two-step state carry, exact Q/K/V permutation inversion, finite BF16
outputs, and four actual FlashInfer `fa2` sparse dispatches.

Both accepted runs used physical GPU3 under the registered lock after dual idle
checks. One unrelated short-lived CUDA process exited before lock acquisition;
there was zero foreign-PID overlap during either run. Total smoke plus native
runner wall time was about `171 s`, below `0.05 H200-hour` and the `0.5`-hour
cap. Formal timing excluded observer hooks, inverse-permutation diagnostics,
density reductions, finite scans, and experimental centroid-state cloning.

The first smoke manifest accidentally included its temporary output path. It
failed self-verification, is preserved as `artifact_sha256.unsealed.txt`, and
was replaced by a separately staged valid manifest without changing any run
artifact. The native manifest was separately staged correctly on its first
sealing attempt.

## Interpretation

This result supports three narrow statements. The released `50 -> 2 -> 2 -> 2`
centroid recurrence executes correctly at local Wan F81 shape; EAR compensation
does recover a substantial portion of SAP output error at nearly identical
exact density; and complete public preprocessing is too expensive to infer
speed from sparse arithmetic alone.

It does not support a plausible joint strict-quality/`1.5x` frontier for fixed
released SAP or EAR on this exposed Layer-14 trajectory. It also does not
support fresh capture, rollout, VBench, endpoint quality, end-to-end speed, or
paper-reproduction work for this local released function class.

Unknowns remain the official Wan-14B 720p 50-step setting, a trajectory with
true earlier centroid state, other layers/prompts/seeds, learned value-aware
routing, source-level fusion or amortization, and trained sparse-linear
baselines. This one exposed development null cannot refute those broader
families.

## Gate recommendation

Close PLAN-042 as `development-null` and park the local fixed released SAP/EAR
quality-cost line. Do not tune clusters, top-p, density, EAR ratio, residuals,
or fallback on this identity, and do not spend fresh capture or rollout data.
Revival requires a materially different cost or function class, such as an
official trained configuration, a fused/amortized router, or a separately
accepted prospective baseline-reproduction protocol.

No claim or experiment registry row is added because every evaluated identity
was globally exposed and the Gate was explicitly development-only.

## Artifacts

- `worldfoundry_hybrid_residual/results/WAN_STATEFUL_SPARSE_VIDEOGEN_QUALITY_COST_DIAG040_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/results/svg2_stateful_quality_cost_plan042_smoke_v1/`
- `worldfoundry_hybrid_residual/results/svg2_stateful_quality_cost_plan042_native_v1/`
- `worldfoundry_hybrid_residual/figures/stateful_sparse_videogen_diag040_20260812/`
