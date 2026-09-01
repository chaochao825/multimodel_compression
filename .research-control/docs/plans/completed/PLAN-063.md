# PLAN-063: Execute the exact Wan VAE temporal-scheduling Gate

- Status: completed
- Owner: researcher
- Mainline: L-030
- Side probe: L-031
- Experiment/Gate: EXP-053 / G-032
- Resource scope: one isolated H200, at most two GPU-hours and 5 GiB

## Objective

Determine whether the official framewise Wan VAE contains exact H200 scheduling
redundancy that can improve the `9.637995s` resident rCM endpoint without any
model, dtype, scheduler, attention, or output change.

## Steps

1. Validate the external schedule runner and frozen configuration locally.
2. On F17, compare official decode with chunk sizes `1/2/4/8`; select only the
   fastest bitwise-exact candidate.
3. If S1 passes, confirm that candidate on all four frozen F81 prompt latents,
   including complete VAE CUDA-event time and peak memory.
4. If S2 reaches VAE `>=1.26x` and projected endpoint `>=1.10x`, run the full
   resident rCM request boundary with serialization.
5. Close G-032 and retain all null, exactness-failure, adverse, or boundary
   evidence before selecting any FP8/fused-attention successor.

## Stop rule

Stop on no bitwise-exact F17 candidate, failed S2 materiality, one valid G-032
outcome, two repairs, two H200-hours, 5 GiB artifacts, GPU isolation loss, or
user interruption. Do not add compile, CUDA Graph, FP8, attention, codec, new
prompt, new seed, or tolerance changes.

## Outcome

The frozen chunk-4 candidate was exact on F17 but not on any of the four F81
prompts. Complete-VAE speedup was `1.196837x` and projected resident-request
speedup was `1.098601x`, both below their registered guards. Stage 4 was not
run. G-032 closed as `exactness-null`; see `RESULT-EXP-053.md`.
