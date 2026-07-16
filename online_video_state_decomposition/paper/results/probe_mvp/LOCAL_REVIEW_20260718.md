# Local Review Record

## Review Mode

The development instructions require an independent subagent review for code
changes. No subagent tool was available in this task's tool surface, so this
record documents the required local substitute review rather than silently
skipping it.

## Scope

Reviewed:

- `feature_memory_codec.py`;
- calibration feature extraction and PCA fitting;
- compressed native-memory evaluation;
- aggregation, exact preservation gate, and matched-state selector analysis;
- formal validation and split auditing;
- rank-sweep and GPU runner scripts;
- the parallel `streaming_hybrid_state_v0` predictor, residual-PQ, controller,
  analysis, and result-audit path;
- result plots, protocol documents, and publishing exclusions.

## Findings Addressed

### High: confirmation split was checked against the wrong set

The first validator required the 200 confirmation IDs to equal the original
200-example evaluation split. The confirmation selection manifest actually
contains a frozen 200-example subset of the 500-example reserve pool. The data
were correctly disjoint, but the validator's semantic check was wrong.

Resolution:

- replaced equality-to-original-evaluation with subset-of-reserve;
- added explicit disjointness checks against calibration, original evaluation,
  and prior-formal sets;
- added expected sample count, task-balance, and outside-reserve checks;
- added positive reserve-subset and negative leakage unit tests;
- reran formal validation successfully.

### Medium: compression report did not directly test retained selector gain

The initial aggregate compared each compressed memory only with its own full
cache. That did not directly answer whether query-conditioned selection still
outperformed exact recent under the same compressed state.

Resolution:

- added `selector_gain_by_variant.csv`;
- paired selectors within each memory variant;
- added a dedicated report table and unit test.

### Low: preservation-plot labels were too wide

Resolution:

- shortened labels to selector family plus residual count;
- rerendered smoke and formal figures;
- visually inspected all formal PNG outputs.

### Low: remote-only parallel probe needed release scoping

The 210 working directory contained a completed
`streaming_hybrid_state_v0` subtree that was not present in the local snapshot.
It predates the publishing sync and includes disjoint splits, unit tests, an
independent result-audit script, and explicit negative/mixed verdicts.

Resolution:

- reviewed its package README, formal analysis, result summary, and audit;
- reran all 13 regression tests from the staged repository;
- retained it as current implementation with an explicit
  representation-level claim boundary;
- documented that its Fourier, logic-controller, and conditional-compute gates
  fail, while selected residual-PQ points are mixed-positive.

## Correctness Checks

- Calibration payloads contain visual tokens and indexing metadata but no
  question, candidates, answer, or answer label.
- PCA fitting requires exactly 100 token files, one extraction fingerprint,
  and one native feature shape.
- The rank-256 codec was fitted only from the frozen calibration IDs.
- FP16 latent values, FP16 residual vectors, and int16 residual indices are all
  included in per-stream bytes.
- Shared codec parameters are excluded from steady-state per-stream state and
  included once in cold-start state.
- All policies receive the same provisioned selector budget.
- Query-time source-video replay is disabled and the visual state is counted.
- Selected frame indices exactly match the frozen manifest.
- The formal full-cache branch matches the prior native run on 400/400 paired
  predictions and correctness outcomes.
- The non-inferiority gate uses a one-sided 95% Clopper-Pearson upper bound on
  full-correct/compressed-wrong events; a degenerate bootstrap interval cannot
  promote a configuration.

## Verification

- Remote unit tests: 54/54 passed in the `Qwen3` environment.
- Streaming hybrid regression tests: 13/13 passed from the staged repository.
- Local unit tests: 54 discovered, 46 passed and 8 torch-dependent tests
  skipped because torch is not installed locally.
- Remote `compileall`: passed for `experiments/` and `figures/`.
- Bash syntax: all `experiments/scripts/*.sh` passed `bash -n`.
- Staged Git whitespace check, large-file scan, and credential scan: passed.
- Formal validation: all checks passed for 200 checkpoints and 1,200
  predictions.
- CSV/JSON structure checks: passed.
- Figure review: all four formal PNG figures are readable and consistent with
  the source CSV files.

## Residual Risks

- Results cover one LLaVA encoder and one five-task MVBench subset.
- The dense visual pool exists transiently during writing before compression;
  peak writer memory has not been optimized.
- Codec timings are unfused Python/CUDA measurements, not production kernels.
- Sparse residual selection is reconstruction-driven rather than task-aware.
- Rank selection used a five-sample smoke gate and needs cross-encoder
  replication.
- Rank-256 plus four residuals misses the strict 2% non-inferiority gate:
  one loss in 200 gives a 2.35% one-sided upper bound.
- The parallel controller never selects the cheap predictor action and no
  reported learned candidate reaches a 30% encoder-required-rate gate; it does
  not support a visual-compute reduction claim.

## Release Decision

No unresolved code defect blocks publishing this research snapshot. Publish
the result as mixed evidence: strong compression and retained query-memory
signal, but no claim of lossless equivalence, method novelty, or BCCB
superiority.
