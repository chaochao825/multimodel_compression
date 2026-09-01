# RESULT-EXP-054: rCM-on-policy low-precision dense attention

- Status: complete
- Validity: valid prospective H200 operator and local-atlas result
- Date: 2026-09-01
- Gate: G-033
- Claim: C-032
- Candidate: L-032
- Outcome: coverage-null

## Registered outcome

The installed SageAttention SM90 dense operator passed the S0 implementation
and speed screen, but no complete `rCM step x Wan layer` cell passed every
calibration identity under the frozen `0.8% / 1.6% / 1.6%` aggregate, head,
and query-tile thresholds. The frozen atlas therefore selected `0/120` cells,
below the required `87/120`, and projected no request-level improvement over
the `9.637995s` EXP-052 resident rCM4 incumbent.

Per protocol, EXP-054 stopped after S1. S2/S3 generation, video quality,
diversity, and resident-request timing were not run.

## S0 operator result

- reference-wrapper F17 latent: bitwise equal;
- network calls: four before and after patching;
- real F81 Q/K/V shape: `[1, 32760, 12, 128]`;
- FA3 BF16 median: `14.590048ms`;
- Sage SM90 median including installed conversions: `9.197088ms`;
- local dense-attention speedup: `1.586377x`;
- engineering-cell aggregate/head/tile relative L2:
  `2.1561% / 3.1473% / 2.5319%`.

## S1 atlas result

The run completed all eight frozen identities and all 960 expected scalar
records. Dual evaluation always returned FA3, so calibration and evaluation
remained on the exact baseline trajectory.

| Split | Aggregate mean / P95 / max | Worst-head mean / P95 / max | Worst-tile mean / P95 / max |
|---|---:|---:|---:|
| Calibration | `1.761% / 3.392% / 10.570%` | `2.643% / 4.864% / 22.423%` | `2.515% / 9.446% / 21.287%` |
| Evaluation | `1.790% / 3.251% / 12.280%` | `2.761% / 4.705% / 27.369%` | `2.560% / 9.558% / 21.326%` |

The calibration threshold was missed first by aggregate error: `0/120` cells
passed its `0.8%` limit, although 13 passed the head limit and 32 passed the
tile limit individually. The best cell was step 0, layer 10, with calibration
maxima `0.887% / 1.177% / 0.981%`; its normalized worst threshold ratio was
`1.108`.

The calibration and evaluation per-cell threshold scores had Pearson
correlation `0.981887`, and seven of their ten safest cells overlapped. Ten
cells would independently pass the looser evaluation thresholds, concentrated
at layers 10, 13, 15, and 16. This is descriptive only: none was eligible after
the frozen calibration rule, and it cannot justify a post-hoc threshold or
selection change.

## Interpretation

### Supports

- The installed Sage SM90 dense operator provides a reproducible `1.586x`
  full-shape attention speedup on the isolated H200.
- Low-precision risk has a stable layer topology across disjoint identities;
  the failure is not primarily random calibration-to-evaluation rank drift.
- Aggregate output error, rather than the registered head or tile maxima, is
  the first active safety bottleneck for the safest cells.

### Does not support

- Released rCM4 does not naturally expose a train-free static whole-cell
  Sage-safe atlas under the registered fidelity margin.
- Few-step flow-map distillation does not imply low-precision-safe internal
  attention, just as EXP-048 showed that it does not imply a rank-64 late-block
  state closure.
- The candidate cannot provide the required `1.05x` request increment because
  its eligible coverage is zero.

### Unknown

- A separately trained quantization policy, QAT checkpoint, changed attention
  backend, or changed rCM checkpoint was not tested.
- Candidate video quality and end-to-end latency are unknown because the
  registered S1 stop rule correctly prevented S2/S3.
- This result does not refute exact VAE, transfer, serialization, or other
  same-step kernel optimization.

## Engineering attempts

The first S0 preflight imported the FlashAttention interface before Torch and
failed before model or GPU execution; importing Torch first was bounded repair
1. S1 then produced all 960 valid records but the runner rejected legal zero
coverage during final projection; allowing zero coverage and deterministically
finalizing the complete scalar records was bounded repair 2. Neither repair
changed the operator, identities, thresholds, outputs, or selection rule.

## Decision

Refute C-032 in its registered class, close G-033 as `coverage-null`, and park
L-032. Keep L-030 and its `9.637995s` exact resident request as the incumbent.
Reopening this candidate requires a changed rCM checkpoint/backend or a
separately accepted trainable quantization policy.

## Evidence

- `worldfoundry_hybrid_residual/results/wan_rcm_onpolicy_attention_exp054_20260901/s0-smoke/manifest.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_onpolicy_attention_exp054_20260901/s1-atlas/atlas.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_onpolicy_attention_exp054_20260901/s1-atlas/cell_metrics.partial.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_onpolicy_attention_exp054_20260901/analysis_v1/summary.json`
- `worldfoundry_hybrid_residual/results/WAN_RCM_ONPOLICY_LOWPREC_ATTENTION_EXP054_20260901.zh-CN.md`
