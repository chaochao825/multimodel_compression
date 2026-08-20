# PLAN-041: Sparse-VideoGen2/EAR H200 operator-readiness gate

- Status: completed
- Owner: researcher and Agent
- Gate: faithful public dynamic-sparse operator readiness
- Claims: none; this Gate checks artifact reconstruction and execution only
- Candidate: pinned `racoonykc/Sparse-VideoGen` SAP/EAR Wan operator
- Lane: explore
- Resource cap: one isolated environment of at most 20 GiB, at most two build
  attempts, at most one H200-hour on physical GPU3, no model download, no fresh
  capture, and no candidate-source edit

## Decision to unlock

Determine whether the exact pinned release can execute its semantic
permutation, variable-block sparse Attention, and EAR centroid compensation on
H200, first under its released correctness tests and then at the immutable Wan
F81 native QKV shape. This is a prerequisite for a later quality/latency Gate;
it is not a paper reproduction or a speed/quality result.

## Frozen candidate identity

- Repository: `/opt/data/wangmeiqi/baselines/Sparse-VideoGen`
- Commit: `f89aedaf169ac2ae5b186bda674e53c3dc08c476`
- Tree: `44b97a3df39700cc02ed6dc5511f9251db8c2b07`
- Candidate checkout must remain clean and read-only.
- Wan Attention SHA256:
  `b3899c3f06df58822899ecc4f83335df1adf8f7c2983661295c3ba5182d11c9c`
- K-means utility SHA256:
  `7169fd4db968a55c2ffe67bcf19aeebdc3ca0620a9cf18a12c93365a67b07e4d`
- Released dynamic-block test SHA256:
  `329023a20eb5c120e9b70bed31bff26d2d5772270a54a02a8d97ebe3cd1e7423`
- CMake SHA256:
  `05aa8207dbd43fac0c21310cb1527d2b009fccee805e636baf9791d204b82f93`
- Pinned gitlinks: CUTLASS
  `81a43e6d92cdd8c20d22392f9579604ed5f710a1`, FlashInfer
  `2f62643a8b73b5dd81dc8ca40f92e6e82dbf7c6a`, pybind11
  `945e251a6ce7273058b36214d94415e9c9530b8e`.

## Stage A: exact artifact reconstruction

- Audit existing environments before installing anything. Reuse only an
  artifact that matches the pinned source and ABI; otherwise create isolated
  `/opt/data/wangmeiqi/envs/svg2_h200_20260812`.
- Reconstruct exact gitlink revisions in a separate build workspace. Do not
  initialize, modify, or build inside the candidate checkout.
- Record Python, PyTorch, CUDA, compiler, cuVS, FlashInfer, CUTLASS, pybind11,
  built-extension, source, command, and environment hashes.
- No source patch, substituted implementation, silent fallback, or unpinned
  dependency is allowed as decision evidence.

## Stage B: released correctness

- Run the released dynamic-block sparse Attention correctness path in BF16,
  head dimension 128, first at sequence lengths 256 and 4096, then 8192.
- Require actual custom-kernel dispatch, finite outputs, matching shapes/dtypes,
  and the released `atol=rtol=1e-2` comparison against its dense/custom-mask
  reference.
- Separately smoke semantic permutation/inverse-permutation and GPU K-means.
- Stage B failure stops the Gate before native-shape work.

## Stage C: immutable Wan native-shape execution

- Use only the PLAN-040 payload SHA256
  `d682537c68fe887c76cea25a779112ebc43e2c91ca80134e93fb069304cf272d`.
- Convert BTHD to the candidate's required layout without changing values.
- Execute both SAP and EAR at sequence length 32760, 12 heads, head dimension
  128. Include K-means, semantic permutation, dynamic-map construction,
  variable-block sparse operator, EAR compensation, and inverse permutation.
- Record dispatch, shape, dtype, finite status, peak memory, and component
  availability only. Do not read dense output errors, search sparsity, or time
  a speed comparison in this Gate.

## Correctness and fairness guards

- Use physical GPU3 under `/tmp/codex_gpu3.lock`, with dual idle checks, PID
  audit, pre/post clocks/P-state/power/temperature, and candidate cleanliness.
- Record candidate, submodule, harness, environment, payload, command, and
  result hashes. Keep all harnesses and build outputs outside candidate source.
- Do not label any prior DIAG output, use exposed quality identities for method
  choice, or change algorithm semantics to make the build pass.
- This Gate is not a reproduction of the paper's Wan-14B, 720p, 50-step,
  H100 end-to-end result and must not be reported as one.

## Outcome mapping

- `operator-ready`: exact artifacts, released correctness, SAP native shape,
  and EAR native shape all pass. Authorize drafting, not running, a separate
  prospective quality/latency Gate.
- `sap-only-boundary`: released correctness and native SAP pass, but EAR's
  released compensation path cannot execute without semantic changes.
- `native-shape-boundary`: released correctness passes but the exact operator
  fails only at Wan F81 native shape or resource limits.
- `artifact-boundary`: exact pinned gitlinks or extension cannot be rebuilt
  within two isolated attempts, without source changes.
- `environment-boundary`: required pinned dependencies cannot coexist within
  the 20 GiB isolated environment; make no method claim.
- `invalid`: candidate/source/input identity, custom dispatch, GPU isolation,
  or frozen procedure differs.

## Stop rules

- Stop on unpinned dependency, candidate source change, fallback, nonfinite
  output, wrong payload, foreign GPU overlap, or resource-cap breach.
- Stop after the registered smokes; do not tune clustering, sparsity, block
  sizes, thresholds, compilation flags for speed, or model parameters.
- Do not collect quality or latency evidence in this Gate. A successful
  operator audit changes only implementation readiness.

## Closure

- Closed date: `2026-08-12`.
- Outcome: `operator-ready`.
- Exact source reconstruction passed with candidate commit
  `f89aedaf169ac2ae5b186bda674e53c3dc08c476`, tree
  `44b97a3df39700cc02ed6dc5511f9251db8c2b07`, and a clean candidate
  checkout. The isolated environment occupied about `1.6 GiB`.
- Released BF16 variable-block correctness passed at sequence lengths
  `256 / 4096 / 8192`. Relative L2 errors were respectively
  `0.1706% / 0.3550% / 0.3496%`; maximum absolute errors were
  `0.00390625 / 0.001953125 / 0.0009765625`. All three calls materialized
  the FlashInfer `fa2` paged-run module.
- Semantic permutation had an exact inverse. Triton and cuVS K-means smokes
  produced finite centroids and complete token assignments.
- The immutable Wan F81 payload executed both SAP and EAR at
  `[1,12,32760,128]` BF16 BHSD. Each method produced one observed custom
  sparse call, finite output of the same shape/dtype, and a dynamic map of
  `[1,12,300,1000]`. Peak allocated/reserved memory was about
  `6.24 / 6.77 GB`.
- Stage-B and native artifact manifests verify locally and remotely. Physical
  GPU3 had dual idle checks and zero foreign-PID overlap in both accepted
  runs.
- Environment attempt 1 stopped before operator execution because cuVS 26.6
  mixed CUDA-12.9 cuSOLVER with Torch CUDA-12.1 cuBLAS. Attempt 2 used a
  coherent RAPIDS 25.6.1 stack and passed. A missing overlay `PATH` entry for
  an already installed Ninja binary caused one preserved runner failure; the
  methods-identical rerun changed only `PATH` and passed.
- Boundary: no dense F81 reference, quality record, sparsity search, latency
  comparison, rollout, or model-level execution was performed. This result
  supports implementation readiness only and is not a reproduction of the
  paper's Wan-14B 720p 50-step result.
- Action: close the artifact/build uncertainty. Before fresh capture or
  rollout, run one separately frozen stateful quality/cost diagnostic that
  preserves the released first-50/subsequent-2 centroid update semantics.
- Evidence:
  `worldfoundry_hybrid_residual/results/WAN_SPARSE_VIDEOGEN_OPERATOR_READINESS_DIAG039_20260812.zh-CN.md`.
