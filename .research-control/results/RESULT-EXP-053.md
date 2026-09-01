# RESULT-EXP-053: Exact Wan VAE temporal scheduling

- Status: complete
- Validity: valid prospective exactness and component-speed result
- Date: 2026-09-01
- Gate: G-032
- Claim: C-031
- Candidate: L-031
- Outcome: exactness-null

## Registered outcome

The frozen first-frame-preserving temporal-grouping candidate did not preserve
the official BF16 Wan VAE output on the four F81 rCM latents. It also missed
the registered complete-VAE and projected-request speed guards. The full
resident endpoint was therefore not run.

This is a valid null for the selected `chunk_size=4` schedule under the
registered frame geometry and backend. It is not a claim that every possible
Wan VAE implementation, chunk size, or approximate decoder is ineffective.

## F17 selection

The official F17 VAE median was `0.894316s`. The frozen candidate screen gave:

| Chunk | Bitwise equal | VAE median (s) | Speedup |
|---:|:---:|---:|---:|
| 1 | yes | 0.887906 | 1.007x |
| 2 | no | 0.792581 | 1.128x |
| 4 | yes | 0.753553 | 1.187x |
| 8 | yes | 0.753578 | 1.187x |

The preregistered rule selected chunk 4 as the fastest bitwise-exact F17
candidate. Chunk 2 was retained as an exactness failure and was not eligible.

## F81 confirmation

The selected chunk-4 schedule was tested on all four frozen prompts with the
same rCM4 latents and seed used by the exact resident-runtime program.

| Prompt | Bitwise equal | Max abs | Relative L2 | VAE speedup |
|---:|:---:|---:|---:|---:|
| 0 | no | 0.111328 | 0.002202 | 1.198x |
| 1 | no | 0.164062 | 0.002772 | 1.189x |
| 2 | no | 0.160156 | 0.004129 | 1.196x |
| 3 | no | 0.139648 | 0.002460 | 1.198x |

The aggregate complete-VAE speedup was `1.196837x`, below the `1.26x` guard.
Applied to the `9.637995s` resident rCM4 baseline, the projected request time
was `8.772973s`, or `1.098601x`, below the `1.10x` guard. Network call counts
were unchanged and the selected H200 remained isolated.

## Interpretation

F17 contains only one post-sentinel chunk at chunk size 4, whereas F81 requires
multiple later chunks that reuse the candidate decoder's internal cache state.
The result therefore exposes a long-horizon state-closure boundary: equality
of a short decoded output does not certify equality of the hidden cache state
used by later chunks. The experiment does not identify whether chunk-shape
CUDA arithmetic, cache-state evolution, or both cause the divergence.

The speed result is independently material. Even if exactness were ignored,
the measured candidate misses both frozen speed thresholds, leaving no margin
for a full endpoint promotion.

## Decision

Refute C-031 in its registered class, close G-032 as `exactness-null`, and park
L-031. Do not run the skipped endpoint stage or rescue the observed result by
post-hoc selection of chunk 1 or 8. L-030 remains the mandatory exact resident
rCM baseline. The next successor may examine fused FP8/dense attention or a
different exact system bottleneck under a new prospective Gate.

## Evidence

- `worldfoundry_hybrid_residual/results/wan_rcm_vae_schedule_exp053_20260901/evaluation_v1/gate_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_vae_schedule_exp053_20260901/evaluation_v1/f17_candidates.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_vae_schedule_exp053_20260901/evaluation_v1/f81_prompt_rows.csv`
- `worldfoundry_hybrid_residual/results/WAN_RCM_VAE_SCHEDULE_EXP053_20260901.zh-CN.md`
