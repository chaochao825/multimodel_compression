# RESULT-EXP-055: Exact full-F81 Wan VAE CUDA Graph replay

- Status: complete
- Validity: valid prospective exactness and runtime-boundary result
- Date: 2026-09-02
- Gate: G-034
- Claim: C-033
- Candidate: L-033
- Outcome: speed-boundary

## Registered outcome

One fixed-shape `torch.cuda.CUDAGraph` preserved every tested official BF16 Wan
VAE output bit across three F17 rCM latents, four F81 prompts, alternating
replay order, and complete resident requests. It passed the registered F81 VAE
component, projected-request, call-count, ownership, stale-state, and memory
guards. It did not pass the absolute complete-request speed guard.

The candidate is therefore a valid exact component optimization but not a
standalone replacement for the L-030 incumbent.

## Engineering and F17 exactness

The first launch used a stale interpreter path and stopped during import before
model loading or CUDA Graph capture. The retained second attempt used the same
Python `3.11.15`, PyTorch `2.9.1+cu128`, CUDA `12.8`, and FA3-capable runtime as
EXP-052/053.

The F17 screen generated three on-policy rCM4 latents with seeds `2026082700`,
`2026082703`, and `2026082704`. Replays followed `0,1,2,1,0`. All graph/eager
GPU tensors, raw CPU tensors, and endpoint-normalized CPU tensors were bitwise
equal. Repeated inputs and eager decodes after all graph replays were also
equal. Every comparison had max absolute and relative L2 error `0`.

The complete copy/replay/owned-output handoff took `0.795--0.808s`; graph
capture peak reserved memory was `32270 MiB`.

## F81 component result

Four frozen EXP-052 prompts were evaluated in forward and reverse order for
eight paired complete decodes.

| Metric | Result | Guard | Status |
|---|---:|---:|:---:|
| Bitwise exact | 8/8 | 8/8 | pass |
| Eager VAE median | 4.231673 s | diagnostic | - |
| Graph VAE median | 3.722834 s | diagnostic | - |
| Complete-VAE speedup | 1.136681x | >=1.12x | pass |
| Projected request | 9.052529 s | diagnostic | - |
| Projected request speedup | 1.064674x | >=1.05x | pass |
| Peak reserved memory | 32292 MiB | <=59948 MiB | pass |

The candidate timing includes static-input `copy_`, graph replay,
synchronization, and an independently owned output clone. Bare replay latency
is not used.

## Complete resident request

The complete endpoint retained positive-prompt re-encoding, four rCM network
calls, VAE copy/replay/handoff, D2H transfer, and MP4 serialization.

| Prompt | Eager request (s) | Graph request (s) | Paired speedup | CPU equal |
|---:|---:|---:|---:|:---:|
| 0 | 9.577006 | 9.017864 | 1.062004x | yes |
| 1 | 10.009927 | 9.264448 | 1.080467x | yes |
| 2 | 10.341052 | 9.465897 | 1.092454x | yes |
| 3 | 10.294319 | 9.387025 | 1.096654x | yes |

Median graph request latency was `9.325737s`, or `1.033483x` versus the frozen
`9.637995s` L-030 incumbent. This misses the `9.179043s` (`1.05x`) guard.
Peak reserved memory was `45870 MiB`, all four CPU videos were bitwise equal,
and every request retained four network calls.

Within this paired run, median request speedup was `1.086460x`. That diagnostic
does not replace the prospective absolute guard: paired eager requests were
slower than the immutable EXP-052 incumbent. Relative to the incumbent,
non-VAE text, denoiser, transfer, and serialization time consumed enough of the
VAE gain to leave only `0.312s` absolute request improvement.

## Interpretation

This result isolates exact execution-graph redundancy. The official VAE's
Python cache lists are not themselves replayed, but the captured fixed-shape
tensor dependency graph overwrites all graph-owned cache state deterministically
for distinct inputs. That is a positive systems mechanism and is different
from assuming that video features, denoising residuals, or attention maps share
a fixed low-rank/BCCB representation.

The result also confirms bottleneck migration after NFE reduction. rCM makes
the denoiser cheap enough that VAE launch amortization is measurable, but a
`1.1367x` VAE gain is not robustly sufficient for a `1.05x` service gain once
D2H and serialization remain unchanged. Component projection was directionally
correct but lacked runtime-variance margin.

## Decision

Refute C-033 in its registered conjunctive form and close G-034 as
`speed-boundary`. Park L-033 rather than integrate it as a standalone mainline.
Its exact graph wrapper may be reused only inside a separately accepted bundle
that also measures the full resident request; it cannot lend its component
speedup to approximate attention, state-cache, or structure claims.

L-030 remains the mandatory exact incumbent. A successor should target a
larger exact bundle (VAE plus D2H/serialization scheduling) or a genuinely
material denoiser kernel, while retaining this graph as an optional exact
component rather than reopening graph-variant selection post hoc.

## Evidence

- `worldfoundry_hybrid_residual/results/wan_rcm_vae_cudagraph_exp055_20260902/evaluation_v1/gate_summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_vae_cudagraph_exp055_20260902/evaluation_v1/f81_component_rows.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_vae_cudagraph_exp055_20260902/evaluation_v1/f81_request_rows.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_vae_cudagraph_exp055_20260902/figures/wan_rcm_vae_cudagraph_exp055.pdf`
- `worldfoundry_hybrid_residual/results/WAN_RCM_VAE_CUDAGRAPH_EXP055_20260902.zh-CN.md`
