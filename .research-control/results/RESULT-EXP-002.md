# RESULT-EXP-002

- Experiment: EXP-002
- Date: 2026-08-09
- Code identity: analyzer `4e7570f660e010cd`; capture `6e660a486eefde50`; runtime `34c405273a44c6ed`
- Data identity: artifacts `ce689c0e62933e5a` and `c826af13503544db`
- Configuration identity: `0c17b32eef20f457`
- Evidence tier: bounded diagnostic
- Independent unit: two frozen prompt/seed identities
- Validity: valid within 16 deterministic sampled token rows
- Outcome class: genuine-dynamics-failure / null
- Protocol deviations: one preregistered engineering retry after invalid BF16/FP32 replay; no outcome was inspected before repair

## Observations

- Recomputing target-step conditioning and forecasting raw SA/CA/FFN outputs
  separately reduced H2 sampled block-output error from 14.305% to 14.018%,
  only 2.00% relatively versus the preregistered 50% requirement.
- Module-level Taylor H2 errors were 4.392%, 22.114%, and 12.545% for blocks
  0, 12, and 29. No block met the 2% condition for both identities.
- Block-12 H2 SA, CA, and FFN component errors were 17.398%, 3.333%, and
  23.680%. The dominant error is therefore inside SA/FFN dynamics rather than
  omitted current-step conditioning.
- H2 error was U-shaped over denoising steps: 38.269% at target step 3, roughly
  3.762%--6.754% in the central region, and 16.654% at step 19.
- Under an individual-call 2% diagnostic guard, H1 passed 89/204 calls (43.6%)
  but H2 passed only 21/204 (10.3%). Block 12 passed 0/68 calls at both
  horizons.
- No rollout, final-video metric, kernel, or speed measurement was performed.

## Validity checks

- Three production-environment synthetic tests passed for module observation,
  wrapper restoration, call order, and dense return parity.
- The first captures were classified invalid before forecasting because FP32
  gate reconstruction did not match the production BF16/autocast path. The
  invalid artifacts remain preserved under the remote `_v1` directories.
- The sole allowed repair used exact production-dtype gates and sequential BF16
  residual replay. Both valid captures then achieved exact replay error 0.
- The frozen evaluator used identical calls, rows, horizons, and denominators
  for whole-block and module-level methods. Target gates are a labelled
  conditioning oracle, not deployment evidence.
- Two identities, three blocks, and sampled rows are sufficient to falsify the
  registered 50%/2% claim but not to characterize all Wan blocks or video DiTs.

## Claim update

- `C-002`: refuted. Target semantics explain only 2.00% of aggregate H2 error,
  and Block 12 remains at 22.114%.
- Supports: temporal risk is strongly nonuniform by step and block; a selective
  H1 policy may have local headroom.
- Does not support: uniform module Taylor prediction, strict rollout quality,
  any H200 acceleration, or a general failure of temporal caching.
- Unknown: whether full-model calibration-only sensitivity routing, global
  Chebyshev forecasting, or a learned stage/subspace predictor can retain
  enough certified calls under the project's 1%/2% and SSIM 0.98 guards.

## Gate recommendation

Close `G-002` and stop the target-semantics explanation. Do not extend this Gate
into rollout or kernel work. Any selective controller or basis-changing
predictor requires a new claim, full 30-block coverage, untouched identities,
and a paper-faithful novelty/fairness audit.

## Artifacts

- Local report: `worldfoundry_hybrid_residual/results/WAN_MODULE_TARGET_MISMATCH_EXP002_20260809.zh-CN.md`
- Local analysis: `worldfoundry_hybrid_residual/results/wan_module_target_audit_exp002/analysis_v1/`
- Local post-hoc diagnostics: `worldfoundry_hybrid_residual/results/wan_module_target_audit_exp002/analysis_v1/posthoc_diagnostics_v1/`
- Remote valid captures: `/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/results/wan_module_target_audit_exp002/p00_seed202608070_v2/` and `p01_seed202608071_v2/`
- Remote invalid captures: same root under the corresponding `_v1` directories
