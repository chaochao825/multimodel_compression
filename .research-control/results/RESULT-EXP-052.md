# RESULT-EXP-052: Exact resident-text rCM warm runtime

- Status: complete
- Validity: valid prospective exact-runtime result
- Date: 2026-09-01
- Gate: G-031
- Claim: C-030
- Candidate: L-030
- Outcome: pass

## Registered outcome

The exact persistent warm-service runtime passed every registered guard. It
kept UMT5 resident, fully re-encoded every distinct positive prompt, reused
only the globally fixed negative prompt where native classifier-free guidance
required it, and changed no Wan, rCM, scheduler, VAE, attention, dtype, or
serialization operation.

Resident `rcm4` reached `4.031x` median F81 warm end-to-end speedup over the
identically optimized resident `teacher20`, exceeding the registered `2.5x`
threshold. The corresponding denoiser speedup was `10.076x`.

## Exactness and text screen

The four-distinct-prompt text screen saved at least `15.208 s/request`, versus
the registered `5.5 s` minimum. All positive embeddings were bitwise equal,
all four positive prompts were encoded independently, and positive cache hits
remained zero. Under native CFG, the fixed negative prompt was encoded once
and reused exactly.

The F17 smoke then produced bitwise-equal decoded CPU video tensors under the
clear-after-request and resident policies for all three methods. Network call
counts were unchanged: 40 for `teacher20`, 8 for `native4`, and 4 for `rcm4`.
Peak reserved memory remained below 39 GiB.

## H200 F81 timing

Each method processed the same four distinct prompts once at seed 2026082701.
Model initialization is excluded from the warm-request latency.

| Method | Text (s) | Denoiser (s) | VAE (s) | CPU transfer (s) | Serialization (s) | Warm request (s) |
|---|---:|---:|---:|---:|---:|---:|
| teacher20 | 0.068 | 32.296 | 4.377 | 0.236 | 1.775 | 38.846 |
| native4 | 0.066 | 6.401 | 4.298 | 0.257 | 1.947 | 12.975 |
| rcm4 | 0.064 | 3.205 | 4.308 | 0.254 | 1.796 | 9.638 |

All twelve requests completed. Positive model calls were four per method and
positive cache hits were zero. The selected devices were H200 NVL GPUs; each
method retained the same model and numerical configuration. The physical H200
index changed only after the original device became occupied by an unrelated
job, and no scientific run overlapped a foreign process on its selected GPU.

## Interpretation

EXP-047's `2.181x` endpoint was not the true persistent-service Pareto. Its
harness destroyed and reconstructed the official global UMT5 encoder after
every request, making text setup consume roughly 16 seconds even after rCM had
reduced the denoiser to about 3 seconds. Removing only that exact repeated work
raises the measured rCM endpoint to `4.031x` without changing generated values.

The bottleneck has now moved again. In resident `rcm4`, VAE decode is the
largest measured component (`4.308 s`, 44.7% of request time), followed by the
denoiser (`3.205 s`, 33.3%) and serialization (`1.796 s`, 18.6%). Text encoding
is only 0.7%. The previously measured `1.51x` local FP8 attention result can
therefore provide only a small end-to-end increment after rCM; any next Gate
must compare it against exact VAE, transfer, and serialization optimizations on
this resident baseline.

This is a systems result, not a new rCM algorithm or quality experiment.
EXP-047's quality and diversity evidence transfers because F17 proves the
runtime is bitwise exact. The formal quality evidence remains four prompts by
two seeds and eight VBench dimensions; EXP-052 adds four-prompt one-seed timing
only.

## Decision

Support C-030, close G-031 as `pass`, and integrate L-030 as the mandatory
exact warm-service baseline for future approximate kernels. Do not compare a
future FP8, sparse, cache, or fused candidate against the clear-after-request
EXP-047 runtime.

## Evidence

- `worldfoundry_hybrid_residual/results/wan_rcm_exact_runtime_exp052_20260901/outputs_exp052/evaluation_v1/gate_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_exact_runtime_exp052_20260901/local_evaluation_v1/gate_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_exact_runtime_exp052_20260901/figures/wan_rcm_exact_runtime_pareto.png`
- `worldfoundry_hybrid_residual/results/WAN_RCM_EXACT_RUNTIME_EXP052_20260901.zh-CN.md`
