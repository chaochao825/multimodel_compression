# RESULT-EXP-006: Comparative row-aware attention sampling

## Status

- Experiment: `EXP-006`
- Claim: `C-006`
- Candidate: `L-006`
- Gate: `G-006`
- Terminal outcome: `null`
- Decision: stop the registered train-free block-sampling family under the
  strict local `AV` objective.

## Validity

- Used exactly the two registered held-out Wan F81 Q/K/V captures, three fixed
  query tiles per identity, query-group sizes, densities, estimators, and 12
  repetitions.
- Six estimator tests passed remotely with PyTorch 2.10.0+cu128.
- No pre-outcome repair, method addition, density change, or threshold change
  was used.
- The valid run used an idle RTX 4090 and completed in 289.695 seconds. It made
  no H200 timing, rollout, final-video, or paper-faithful reproduction claim.
- All outputs were finite and all realized denominators were positive.

## Primary evidence

At the frozen 50% diagnostic ceiling, the strongest target-leaking full-joint
mechanism was per-query joint PPS-CV:

| Query group | P95 aggregate | P95 worst record | Worst head | 64x64 union |
|---:|---:|---:|---:|---:|
| 64 | 3.474% | 4.918% | 9.449% | 36.24% |
| 16 | 2.904% | 4.211% | 7.958% | 62.32% |
| 1 | 1.666% | 2.129% | 5.873% | 95.44% |

No target-leaking `joint` or `exact-Z` method reached the registered 0.5%/1%
oracle gate, so the outcome is `null` by the frozen precedence map. Per-query
proposal reduced aggregate error by about 52% relative to 64-query sharing,
but erased regular-block sparsity through support-union inflation.

The registered normalization factorial localized the numerical bottleneck. For
per-query joint PPS-CV at 50% work:

| Mode | P95 aggregate | P95 worst record |
|---|---:|---:|
| Joint ratio | 1.666% | 2.129% |
| Exact partition `Z` | 1.661% | 2.124% |
| Exact numerator `N` | 0.164% | 0.208% |

The full-joint result is effectively unchanged by an exact partition, while
denominator-only error is already below the gate. Vector numerator estimation,
not scalar partition estimation, is therefore the dominant registered error.

At 37.5% work and per-query granularity, target-leaking joint PPS-CV reached
2.038%/2.617%. Joint top-k was 3.703%/4.828%, mass top-k sparse was
5.451%/7.227%, moment stratified HT-CV was 5.820%/7.632%, and moment PPS-CV was
8.517%/10.364%. No-replacement stratification did not rescue the function
class, and deployable moment proposals remained well behind the oracle.

## Interpretation boundary

### Supports

- Query-specific proposal granularity matters materially.
- The current strict local error is dominated by the value-weighted numerator.
- Simple with-replacement, no-replacement, uniform, moment-PPS, and top-k
  variants do not produce a high-fidelity, regular-block-sparse operating point.

### Does not support

- A proposal learner, H200 kernel, rollout, or controller for this estimator
  family.
- A claim that all approximate, sparse, or Monte Carlo attention is invalid.
- A direct comparison between these mechanism proxies and published system
  speedups or final-video metrics.

### Unknown

- Whether a learned sparse-linear numerator tail changes the function class
  enough to pass held-out gates.
- Whether a relaxed trajectory-risk or final-video objective tolerates the
  observed local error.
- Whether other layer, step, branch, or model cells have a more concentrated
  residual bulk.

## Decision effect

Refute `C-006`, park `L-006`, and close `G-006`. Keep EXP-006 as a no-go and
comparison baseline. A learned numerator-tail family or relaxed objective
requires a fresh researcher decision, claim, protocol, and Gate.

## Evidence artifacts

- Scientific report:
  `worldfoundry_hybrid_residual/results/WAN_ATTENTION_SAMPLING_MECHANISM_COMPARE_EXP006_20260811.zh-CN.md`
- Figure:
  `worldfoundry_hybrid_residual/figures/attention_sampling_mechanism_compare_exp006_20260811/attention_sampling_exp006_diagnostics.png`
- Scientific report SHA256: `8897b97f92ebe084627f01e16a1ab3d2ac5370847221f2ce3c9c57c01098fff1`
- Plot script SHA256: `5b4f5dc7ed248e88b0e91ab861206f0f03a19d5345e9f385bc2072f8175c4d7e`
- Plot data SHA256: `8ebfe6dd5e807206e0ffc7cd163050c2dea37f86c9c14b0c4ea47835026a7edf`
- PNG SHA256: `bdaf1d5798920568ab13ee011b96a068ab3e762acf407972994c199d247195ca`
- PDF SHA256: `869032b8dfbe9e01971467aaaf37224e92a9196145f1f589a9d377d2e939406c`
- Manifest SHA256: `2dc7ceb0cb5f35e9db720896d6b879a868c593001edd871b8c3a9dc3a47cf5fe`
- Summary SHA256: `f2cd09bc46676cc398db143002696d4b5436dfb9a61081771dda6ffaca1f7f3a`
- Record metrics SHA256: `6874cf849148b7133647117feb3e3930d66acce4af3d4256b460396864662808`
- Decision SHA256: `a23b400862c50968a0fbd9cded8a18a3381ae0185e74b618ccb9ce748224750a`
