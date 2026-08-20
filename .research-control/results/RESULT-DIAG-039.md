# RESULT-DIAG-039

- Experiment: DIAG-039 / PLAN-041
- Date: 2026-08-12
- Candidate: pinned Sparse-VideoGen SAP/EAR Wan operator
- Candidate identity: commit
  `f89aedaf169ac2ae5b186bda674e53c3dc08c476`, tree
  `44b97a3df39700cc02ed6dc5511f9251db8c2b07`
- Harness identity: SHA256
  `f206a13187c4fc9519d1b5f8e14b14d378fc5ecd31d4d83e480fd39af1a6e676`
- Native data identity: immutable F81 QKV SHA256
  `d682537c68fe887c76cea25a779112ebc43e2c91ca80134e93fb069304cf272d`
- Environment: Python 3.11.15, Torch 2.5.1+cu121, FlashInfer 0.2.10,
  cuVS 25.06.01, CUDA 12.9 JIT toolchain, H200 NVL SM90
- Released artifact manifest SHA256:
  `c217e28f935693c580dfa3441c854b8f1f19d3fdab985f0bc3f7db3972231e3f`
- Native artifact manifest SHA256:
  `56ce273c28911055555c0980041d896c647dc823dbf3350a629793401b2a8442`
- Evidence tier: engineering-only operator readiness; no quality or timing
- Validity: valid
- Outcome class: pass (`operator-ready`)

## Observations

Released variable-block BF16 correctness passed at sequence lengths 256,
4096, and 8192. Relative L2 errors were `0.1706%`, `0.3550%`, and `0.3496%`,
with maximum absolute errors `0.00390625`, `0.001953125`, and `0.0009765625`.
Every case materialized the FlashInfer `fa2` paged-run path, retained the
expected shape and dtype, and produced finite output. Semantic permutation was
exactly reversible; both Triton and cuVS K-means smokes covered every token.

On the immutable Wan F81 payload, both SAP and EAR executed at BHSD shape
`[1,12,32760,128]`. Each produced one observed custom sparse call, a finite
BF16 output with unchanged shape, and a `[1,12,300,1000]` dynamic block map.
Peak allocated and reserved memory were `6,239,912,448` and `6,771,703,808`
bytes. No dense reference, quality table, or latency measurement was read or
computed.

## Integrity and deviations

The candidate checkout remained clean and all registered source, harness,
payload, and result hashes passed. Both accepted runs used physical GPU3 after
dual idle checks with zero foreign-PID overlap. The first isolated dependency
attempt stopped before operator execution on a CUDA-library ABI conflict. The
second coherent RAPIDS-25.6.1 environment passed. One preserved invocation
failed before JIT because its runner omitted the already-installed overlay
Ninja directory from `PATH`; the rerun changed only `PATH` and passed. These
failures update engineering provenance, not method belief.

## Interpretation

The result supports one narrow statement: the pinned public SAP and EAR
operators, including semantic reordering, dynamic variable blocks, and EAR
compensation, can be reconstructed and executed at the local Wan F81 native
shape on H200 without candidate-source modification or fallback.

It does not support any quality, sparsity, transfer, latency, end-to-end,
rollout, or paper-reproduction claim. In particular, the publication recipe
targets Wan-14B, 720p, 50 steps and stateful centroids; this local audit uses
Wan2.1-1.3B, 480x832, 20 steps and checks only one immutable operator payload.

## Gate recommendation

Close PLAN-041 as `operator-ready`. The next bounded diagnostic must preserve
the released first-50/subsequent-2 centroid evolution across adjacent steps,
evaluate SAP and EAR against the same dense reference, and include the entire
public preprocessing path in H200 latency. Do not spend fresh rollout data
until that development diagnostic establishes a plausible local quality/cost
frontier.

## Artifacts

- `worldfoundry_hybrid_residual/results/WAN_SPARSE_VIDEOGEN_OPERATOR_READINESS_DIAG039_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/results/svg2_operator_readiness_plan041_stage_b_v3/`
- `worldfoundry_hybrid_residual/results/svg2_operator_readiness_plan041_native_v1/`
