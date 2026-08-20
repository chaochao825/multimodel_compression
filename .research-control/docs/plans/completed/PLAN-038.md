# PLAN-038: Public sparse-attention baseline reproducibility audit

- Status: active
- Owner: researcher and Agent
- Gate: source and environment feasibility; no scientific result
- Claims: none
- Candidate line: one faithful Wan sparse-attention baseline
- Lane: explore
- Resource cap: no GPU jobs, no model downloads, no kernel build, at most three
  LFS-disabled shallow source clones under `/opt/data/wangmeiqi/baselines`

## Decision to unlock

Select one exact public baseline whose released implementation can distinguish
whether the observed gap comes from the train-free/custom function classes or
from the Wan/H200 evaluator itself.

## Frozen candidates

1. NVIDIA Sol-Attn on the `NVlabs/Sana` `sol-engine` branch: training-free,
   online threshold routing plus skipped-block proxy-score reuse, released
   Hopper kernels, and an official Wan2.1 path.
2. Sparse-VideoGen2 or SVG-EAR from `svg-project/Sparse-VideoGen`: training-free
   semantic permutation or error-aware centroid compensation, public Wan2.1
   scripts, and customized sparse kernels.
3. SLA/SLA2 from `thu-ml/SLA`: trained sparse-linear routing/QAT and public
   fused operator code; use only if an end-to-end Wan adaptation recipe or
   compatible released checkpoint is present.

## Audit fields

For each candidate record:

- repository URL, exact commit, branch, license, and source-tree hash;
- supported Wan model size, resolution, frame count, sampler, and GPU target;
- whether checkpoint, training recipe, dataset, and evaluation prompts exist;
- Python/PyTorch/CUDA/CuTe/Triton requirements and compatibility with H200;
- whether the released kernel exposes real timing rather than FLOP estimates;
- changes required to use the existing Wan2.1-1.3B F81 evaluator;
- data/download/storage/build cost and the smallest faithful smoke test;
- fairness risks, including model-size, sampler, resolution, warmup, caching,
  compilation, and quality-metric differences.

## Outcome mapping

- `baseline-ready`: one candidate has public source, a pinned commit, a
  compatible Wan path, runnable H200 kernel, reproducible quality protocol, and
  a bounded smoke test. Open one implementation/reproduction gate.
- `model-boundary`: the strongest candidate is released only for an
  incompatible Wan size or pipeline. Preserve it as literature evidence and
  do not claim evaluator failure.
- `artifact-boundary`: paper claims are not backed by sufficient checkpoint,
  recipe, kernel, or evaluation artifacts for faithful reproduction.
- `environment-boundary`: artifacts are complete but cannot be isolated on the
  current server without exceeding the declared storage/build envelope.
- `invalid`: wrong repository/commit, incomplete provenance, or post-selection
  changes to candidates or audit fields.

## Selection rule

Prefer, in order:

1. exact Wan2.1-1.3B/F81 compatibility;
2. complete released kernel and end-to-end evaluator;
3. closest mechanism to the failed custom line;
4. lowest adaptation and training cost;
5. newest claimed speed.

Do not select solely by reported paper speedup.

## Stop rules

- Do not initialize git submodules, download LFS assets or model weights, build
  kernels, install packages, run GPUs, or modify candidate source.
- Stop after the three frozen candidates are audited or one candidate is
  strictly dominant under the selection rule.
- Archive one Chinese audit report and the exact source identities before
  opening any reproduction gate.

## Closure

- Closed date: `2026-08-12`.
- Outcome: `baseline-ready`.
- Selected baseline: Sol-Attn from `NVlabs/Sana`, branch `sol-engine`, commit
  `5dd502af9938d924be206c332ad1e911b4a925a1`, tree
  `730cef4dc0fe6be2e4b17997fec295f730afa541`.
- Reason: it is the only frozen candidate combining an official
  Wan2.1-T2V-1.3B configuration, released SM90 kernel, direct operator API,
  and a bounded smoke test that reuses immutable QKV captures.
- Boundary: the published high-sparsity operating point does not satisfy this
  project's strict dense-relative fidelity target. No quality, H200 speed, or
  rollout claim follows from source readiness.
- Evidence:
  `worldfoundry_hybrid_residual/results/PUBLIC_SPARSE_ATTENTION_BASELINE_AUDIT_PLAN038_20260812.zh-CN.md`.
- Action: open PLAN-039 for one pinned released-kernel Pareto test before any
  model download or end-to-end integration.
