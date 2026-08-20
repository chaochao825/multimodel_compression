# RESULT-EXP-003

- Experiment: EXP-003
- Date: 2026-08-10
- Code identity: capture `4ca3cb4ddb0058a1`; analyzer `19ef2d00daef8b85`;
  runtime `34c405273a44c6ed`
- Configuration identity: `6265dc08ded1bc2d`
- Data identities: `e584a3f604177ebd`, `11f73afc597d189e9`,
  `b0eb096e72f0344b`, `3010b52e9266ac08`
- Decision identity: `a49284c5eaa9af18`
- Evidence tier: prospective full-model local-transfer screen
- Independent unit: two calibration and two held-out prompt/seed identities
- Validity: valid within eight deterministic sampled token rows
- Outcome class: null
- Protocol deviations: none

## Registered result

The calibration-frozen policy selected two self-attention actions and no FFN
actions out of 1,200 calls per module. On held-out identities its combined
aggregate/worst sampled block-output relative L2 was 0.916%/1.611%, but it
removed only 0.0898% of denoiser work. Its profile-based ideal denoiser speedup
was 1.0009x.

The held-out post-hoc non-chained oracle selected 22 self-attention and nine FFN
actions. Its aggregate/worst error was 0.848%/1.228%, but it removed only 1.081%
of denoiser work, for an ideal upper bound of 1.0109x. This is far below the
registered 16.67% removed-work requirement.

## Validity checks

- All four captures covered 30 blocks, 20 steps, and both CFG branches.
- Every full-tensor production-order module decomposition had relative error 0.
- Prompt/seed identities and splits matched the frozen config, with no overlap.
- The selector read calibration rows only; held-out rows were loaded only for
  evaluation and the explicitly labelled post-hoc oracle.
- Non-chaining history constraints were checked for every selected stream.
- Core selector tests (4) and existing module-runtime regression tests (3)
  passed in the frozen remote environment.
- No NaN/Inf, incomplete cell, hash mismatch, retry, or output overwrite occurred.

## Claim update

- `C-003`: refuted within its registered Wan/F81, sampled-row, method, and
  threshold scope.
- Supports: strict non-chaining and fallback can keep selected local errors
  below 1%/2%.
- Does not support: enough transferable coverage for rollout, measurable H200
  acceleration, or final-video quality.
- Unknown: whether the earlier whole-module H1 signal was produced by stable
  cross-module cancellation, and whether a learned content-conditioned policy
  can obtain a different quality-coverage frontier.

## Decision

Close `G-003` as null. Do not implement selective rollout or kernels for this
candidate. Preserve a post-hoc failure diagnostic as non-decision-bearing
evidence; it cannot change this registered outcome.

## Post-hoc failure diagnostic

The diagnostic confirmed that cross-module cancellation was real but not
deployably sufficient. Across held-out cells, the mean/median cancellation
fraction was 50.23%/50.29%. Under the same strict 1%/2% whole-block guard, the
held-out post-hoc oracle selected only 7 of 1,200 calls, removed 0.5833% of
denoiser work, and had an ideal speedup of 1.0059x. Reaching 19.25% removed work
required relaxing the observed aggregate/worst local error to 3.808%/6.928%.

This resolves the prior uncertainty: the historical 43.6% H1 signal reflected
a restricted three-block post-hoc population plus substantial module-error
cancellation. It does not transfer to the full 30-block strict deployment
population. These diagnostics are explanatory only and do not alter the
registered null outcome.

## Artifacts

- Local analysis:
  `worldfoundry_hybrid_residual/results/wan_selective_execution_atlas_exp003/analysis_v1/`
- Local post-hoc diagnostics:
  `worldfoundry_hybrid_residual/results/wan_selective_execution_atlas_exp003/posthoc_diagnostics_v1/`
- Remote captures and analysis:
  `/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/results/wan_selective_execution_atlas_exp003/`
