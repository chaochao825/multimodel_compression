# RESULT-EXP-047: Released rCM four-step H200 Pareto Gate

- Status: complete
- Validity: valid prospective speed-boundary result
- Date: 2026-09-01
- Gate: G-026
- Claim: C-026
- Candidate: L-026

## Registered outcome

`speed-boundary`. The official BF16 rCM four-step release passed every frozen
quality, diversity, denoiser-speed, and output-completeness guard, but its
`2.181x` warm end-to-end speedup was below the registered `2.5x` threshold.
The conjunctive claim C-026 is therefore refuted even though rCM establishes a
strong quality-preserving component-local Pareto point.

## H200 timing

| Method | Denoiser (s) | Per forward (s) | Text (s) | VAE (s) | Serialization (s) | Warm E2E (s) |
|---|---:|---:|---:|---:|---:|---:|
| teacher20 | 32.202 | 0.805 | 17.386 | 4.181 | 1.928 | 56.126 |
| native4 | 6.357 | 0.795 | 14.332 | 4.221 | 2.377 | 27.791 |
| rcm4 | 3.177 | 0.794 | 16.119 | 4.214 | 1.827 | 25.729 |

Relative to `teacher20`, `rcm4` reached `10.135x` denoiser speedup and
`2.181x` warm end-to-end speedup. The denoiser guard passed by a wide margin;
the end-to-end guard missed by 12.7%. Median per-forward time was nearly
identical across methods, so the denoiser gain comes from the registered call
count: 40 teacher network calls, eight native-four-step calls, and four
official rCM calls.

## Quality and diversity

Across the eight frozen VBench dimensions, `rcm4` reached a mean
teacher-normalized score of `0.996875`; its lowest dimension was aesthetic
quality at `0.970645`. It passed both `0.90` mean and `0.80` minimum guards.
By contrast, `native4` reached `0.866922` mean and `0.500000` minimum, failing
both guards.

| Dimension | native4 / teacher | rcm4 / teacher |
|---|---:|---:|
| subject consistency | 1.0366 | 1.0062 |
| background consistency | 1.0094 | 0.9880 |
| temporal flickering | 1.0215 | 0.9887 |
| motion smoothness | 1.0078 | 0.9939 |
| dynamic degree | 0.5000 | 1.0000 |
| aesthetic quality | 0.8283 | 0.9706 |
| imaging quality | 0.7245 | 0.9820 |
| overall consistency | 0.8073 | 1.0455 |

The rCM inter-seed diversity ratios were `1.572`, `1.120`, `1.113`, and
`1.362` across the four prompts. All four exceeded `0.70`, and the minimum was
well above the `0.50` collapse guard. Values above one demonstrate absence of
collapse under the registered composite; they are not interpreted as generally
better diversity.

All 24 method/prompt/seed videos completed with 81 frames and no fallback,
NaN, OOM, or truncation. Paired SSIM/PSNR remain diagnostics only because rCM
changes the generated trajectory and are not used to override the registered
distributional quality decision.

## Mechanism interpretation

The valid positive mechanism is training-native finite-time flow-map
compression, not post-hoc low-rank closure. EXP-048 showed that the same rCM
weights did not make late whole-block residuals rank-64 Markov states, while
EXP-047 shows that the released model still reaches teacher-level endpoint
quality in four steps. Endpoint map compressibility and internal-state linear
compressibility are therefore distinct properties.

The timing decomposition also changes the engineering target. After NFE and
CFG-call reduction, text encoding, VAE decode, serialization, and residual
runtime overhead account for most rCM wall-clock. Additional denoiser-only
matrix structure cannot by itself close the registered end-to-end gap.

## Decision

Refute the conjunctive C-026 speed claim, close G-026 as `speed-boundary`, and
retain L-026 as the decided measured high-quality incumbent. Any successor should retain
the rCM endpoint and first target exact overheads such as prompt-embedding
reuse, VAE/serialization overlap, and runtime fusion before reopening a new
approximate hidden-state architecture.

## Evidence

- `worldfoundry_hybrid_residual/results/wan_rcm_baseline_exp047_20260901/quality/evaluation_v1/quality_gate_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_baseline_exp047_20260901/quality/evaluation_v1/vbench_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_baseline_exp047_20260901/quality/evaluation_v1/diagnostics.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_baseline_exp047_20260901/timing/outputs/timing_f81/*/generation_manifest.json`
- `worldfoundry_hybrid_residual/results/WAN_RCM_BASELINE_EXP047_20260901.zh-CN.md`
