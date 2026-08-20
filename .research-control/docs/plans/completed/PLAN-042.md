# PLAN-042: Stateful Sparse-VideoGen SAP/EAR development quality-cost gate

- Status: completed
- Owner: researcher and Agent
- Gate: released stateful sparse-attention local quality/cost
- Claims: none; all QKV identities are globally exposed development data
- Candidate: pinned Sparse-VideoGen Wan SAP and EAR operators
- Lane: explore
- Resource cap: one harness implementation, at most one pre-output harness
  correction, at most 0.5 H200-hour on physical GPU3, no environment rebuild,
  no model download, no fresh capture, and no candidate-source edit

## Decision to unlock

Determine whether the released stateful semantic-permutation function class,
especially EAR compensation, has a plausible joint local quality and H200-cost
frontier on Wan2.1-1.3B F81 before spending fresh prompt/seed or rollout data.
This is a development diagnostic, not a prospective transfer or paper
reproduction.

## Frozen candidate and environment

- Candidate commit/tree and source hashes are identical to PLAN-041 / DIAG-039.
- Candidate checkout remains clean and read-only. Use the accepted isolated
  `/opt/data/wangmeiqi/envs/svg2_h200_20260812` environment and exact build
  workspace; no dependency or compilation-flag change is allowed.
- Require the accepted PLAN-041 released and native artifact manifests before
  execution. Actual FlashInfer custom dispatch is mandatory.

## Frozen harness identities

- Protocol config SHA256:
  `8605fbaca642fbdfb9a635012861c8ddfc6506e9b2a3e9a56fd511b01bc9dcfa`.
- Pure core SHA256:
  `794518bf3da7e68b5913a40d16e922ab4ee013f6ee670649b4db59cfc9515c8e`.
- Pure test SHA256:
  `445119e1079bf3f886c674918b5cf574d8e0294846ac1c8de08c2717dd092263`.
- GPU probe SHA256:
  `668c6cca3fa707935500ca9b33f224f6f8a176aec57b36f8228eca10fbc7bb3c`.
- Remote execution runner SHA256:
  `40e0a3d36f9cbf316f2f024fd76797231aa06cd89fc51fe7fc7b08e2fa637fa8`.
- Six pure metric/state/density/outcome assertions passed in the accepted
  environment before the gate opened. They were invoked directly because the
  frozen environment lacks pytest's optional `pygments` dependency; no package
  was installed and no candidate output was produced.
- Quality and warmup paths validate Q/K/V permutation reversibility, finite
  outputs, density, and actual custom dispatch. Formal timing disables those
  diagnostic scans and observer hooks, and restores a cloned frozen incoming
  centroid state before the timing window, so the primary wall boundary
  contains only the operators listed below.

## Frozen development trajectory

- Use only sample `s00_p00_seed20260850`, Layer 14, conditional branch, F81,
  steps `[7,8,9,10]`, sequence length `32760`, 12 heads, head dimension 128,
  BF16 BTHD Q/K/V.
- Exact payload SHA256 values are frozen in
  `worldfoundry_hybrid_residual/configs/stateful_sparse_videogen_quality_cost_diag040_v1.json`.
- Every identity is globally exposed. No generalization, transfer, or new-data
  claim is permitted.
- The available trajectory starts after the released 20-step schedule's true
  first sparse step. Step 7 is therefore a *shifted initialization anchor*:
  initialize centroids from current step-7 Q/K and do not claim exact rollout
  state reproduction.

## Frozen methods

- Dense control/baseline: same-environment FlashInfer BF16 dense prefill.
- SAP: released semantic permutation and variable-block sparse Attention.
- EAR: identical semantic permutation plus released error-aware centroid
  compensation.
- Query clusters `300`, key clusters `1000`, top-p `0.9`, SAP minimum key
  cluster ratio `0.10`, and EAR source-hardcoded minimum ratio `0.05`.
- Seed the step-7 random centroid initialization once with `20260812`.
  Run at most 50 K-means iterations at step 7, then carry the returned Q/K
  centroids through steps 8--10 with at most two iterations per step.
- Quality execution may share the method-independent clustering/permutation
  state between SAP and EAR. Latency execution must independently include that
  preprocessing for each candidate from the same frozen incoming centroids.
- No alternate cluster count, top-p, minimum ratio, initialization, layout,
  residual, fallback, or threshold is allowed.

## Evaluator and guards

- FP32 reference: the existing deterministic 16 query tiles of 64 rows, evenly
  spanning the native token order, with all 32760 keys/values.
- For every method and step, record aggregate, worst-head, and worst-tile
  output relative L2. Also record the pooled four-step summary.
- FlashInfer BF16 dense must remain at or below `0.3%` aggregate error for
  every step.
- A candidate quality pass requires every step and its pooled summary to meet
  `1% / 2% / 2%` aggregate/head/tile guards.
- Record weighted exact token-pair density, dynamic-map shape, K-means
  iterations, finite/shape/dtype, exact inverse permutation, dispatch, and peak
  memory. No target value may influence routing or state.

## H200 cost protocol

- Use step 9 with incoming centroids frozen after steps 7--8.
- Time `flashinfer_dense_bf16`, stateful SAP, and stateful EAR in alternating
  forward/reverse order, with one warmup and five synchronized repetitions.
- SAP/EAR primary wall time includes two-iteration Q/K clustering, dynamic-map
  construction, semantic permutation, workspace allocation, FlashInfer plan,
  sparse execution, EAR compensation when applicable, inverse permutation,
  and synchronization. It excludes payload loading and model QKV projection.
- Restore identical incoming centroids before every candidate repetition.
- Require finite outputs, actual custom dispatch, zero foreign GPU overlap,
  timing CV `<=5%`, and measured whole-attention speedup `>=1.5x` versus the
  same-run FlashInfer BF16 dense baseline.
- Report first-50 initialization cost separately; do not hide or directly add
  it to the steady-state speedup.

## Outcome mapping

- `development-joint-pass`: SAP or EAR passes every quality guard and reaches
  `>=1.5x` steady-state speed. Authorize drafting, not running, one fresh
  continuous-trajectory prospective Gate.
- `quality-only-boundary`: at least one candidate passes quality but no passing
  candidate reaches `1.5x`. Preserve the quality mechanism and park fresh
  rollout pending a lower-cost implementation.
- `speed-only-boundary`: at least one candidate reaches `1.5x`, but neither
  passes quality. Preserve the runtime evidence and do not capture fresh data.
- `development-null`: neither candidate passes quality nor reaches `1.5x`.
  Park this local released-function-class line before fresh capture.
- `shifted-initialization-boundary`: execution succeeds but the missing true
  earlier centroid state prevents a decision because step-7 initialization
  dominates or is numerically unstable. Preserve diagnostics; make no method
  claim.
- `engineering-failure`: the exact accepted environment cannot complete the
  frozen evaluator within the resource cap. Preserve diagnostics; make no
  method claim.
- `invalid`: source/input identity, evaluator, method, dispatch, isolation,
  state reset, or timing boundary differs.

## Stop rules

- Unit-test pure metric, state-reset, density, and outcome logic before GPU
  execution. Run a small-shape state-carry smoke before loading native QKV.
- Stop on modified source, wrong payload, fallback, nonfinite output, failed
  dense control, foreign GPU overlap, timing CV failure, or resource breach.
- Do not inspect other identities, tune any public parameter, add fallback,
  compare a post-hoc residual, capture QKV, run a model, or start rollout.
- Close immediately after the fixed four-step quality table and one step-9
  latency table.

## Closure

- Closed date: `2026-08-12`.
- Validity: valid; all frozen source, harness, config, payload, prerequisite,
  and result hashes passed locally and remotely. Physical GPU3 had dual idle
  checks and zero foreign-PID overlap.
- Outcome: `development-null`.
- Dense BF16 control passed every step at about `0.167%` aggregate error.
  SAP pooled aggregate/head/tile errors were
  `8.0689% / 10.0064% / 10.4729%`; EAR improved them to
  `3.9080% / 5.5455% / 5.6396%`, but neither passed any registered quality
  summary.
- Complete step-9 median wall times were `18.5836 / 14.6354 / 23.6853 ms`
  for dense/SAP/EAR, giving `1.0000x / 1.2698x / 0.7846x`. Every timing CV
  was below `1.5%`, but neither candidate reached `1.5x`.
- Step-9 exact density was about `16.8%` for both candidates. EAR therefore
  demonstrated meaningful centroid-error compensation at nearly unchanged
  support, but its extra path was slower than dense.
- The shifted step-7 Q/K initialization took `40.563 s`; subsequent
  two-iteration state updates took `4.66--5.16 ms`. Steps 8--10 still missed
  quality by wide margins, so missing true pre-step-7 state does not make the
  outcome indeterminate.
- Action: park the local fixed released SAP/EAR quality-cost line. Do not tune
  exposed settings or spend fresh capture/rollout data. Revival requires a
  materially different cost/function class under a separately accepted Gate.
- Evidence:
  `.research-control/results/RESULT-DIAG-040.md` and
  `worldfoundry_hybrid_residual/results/WAN_STATEFUL_SPARSE_VIDEOGEN_QUALITY_COST_DIAG040_20260812.zh-CN.md`.
