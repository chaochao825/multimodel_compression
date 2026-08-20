# RESULT-EXP-000

- Experiment: EXP-000
- Date: 2026-08-07
- Protocol identity: `EXP-000-wan-step-state-transition-v1`
- Code identity: runtime `e4274fb4`, capture `75e048f9`, evaluator `1b92d3cb`
- Data identity: F17 artifact `5c8c4e90`; F81 artifact `be8fa1f5`
- Configuration identity: `fd2fbab1`
- Evaluator identity: `analyze_wan_step_state_transition.py@1b92d3cb`
- Artifact identity/checksum: manifests and complete hashes under the registered result path
- Evidence tier: prospective
- Independent unit: prompt/seed identity (4 calibration, 4 held-out)
- Validity: valid
- Outcome class: null
- Protocol deviations: none

## Observations

- Both F17 and F81 captures completed 8/8 identities, six blocks, both CFG
  branches, and all 20 denoising steps. Calibration and held-out identities did
  not overlap.
- No calibration-only method certified any cell under the 0.75%/1.5%
  calibration safety margin.
- F81 full-scope horizon-2 aggregate output errors were 12.686% for reuse,
  10.163% for L2P ridge, 10.316% for stage AR(3), 8.589% for stage
  AR(3)+innovation, and 8.344% for the transductive per-sample scalar oracle.
- The strongest transductive F81 horizon-2 oracle passed 0/192 block/branch/step
  cells under the 1% aggregate and 2% worst guard. Horizon 1 passed only 16/204
  cells, all concentrated in block 0.
- Direction-normalized temporal rank99 averaged 8.31 on F81, while median
  adjacent-residual cosine was 0.9698. Low offline trajectory rank therefore did
  not imply stable causal AR(3) extrapolation.
- No rollout, video-quality, whole-block timing, or end-to-end speed result was
  produced because the representation/transfer stop rule fired first.

## Validity checks

- Synthetic tests passed for dense-output parity, wrapper restoration, exact
  Gram reconstruction, causal fitting, nonuniform Taylor interpolation,
  open-loop recursion, and held-out exclusion from fitting.
- Capture manifests passed hash, identity, schema, shape, energy, and completeness
  checks. The frozen evaluator ran once after both captures completed.
- The per-sample oracle used held-out targets and is interpreted only as a
  function-class upper bound, never as deployment evidence.
- The four held-out identities are sufficient for the deterministic worst-case
  Gate but not for a broad population claim about Wan or all video DiTs.

## Claim update

- `C-000`: oppose. The bounded scalar oracle itself failed the preregistered F81
  quality/coverage threshold, so calibration-only stagewise scalar state cannot
  support the claim.
- Narrow boundary: one-step prediction is feasible for a small subset of block-0
  cells, but 7.84% sampled-cell coverage and one block out of 30 cannot imply the
  required whole-model speedup.
- Unknown: channel/tile-conditioned or learned subspace state was not tested by
  this protocol. Its offline plausibility does not revive `C-000`.

## Gate recommendation

Close `G-000` and stop the scalar temporal-transition family. Do not open the
rollout/H200 Gate. Any feature-subspace or content-conditioned state must be a
new claim with a new untouched protocol and a novelty audit against SVD-Cache,
L2P, LESA, CG-Taylor, and BWCache.

## Artifacts

- Local report: `worldfoundry_hybrid_residual/results/WAN_STEP_STATE_TRANSITION_EXP000_20260807.zh-CN.md`
- Local analysis: `worldfoundry_hybrid_residual/results/wan_step_state_transition_exp000/analysis_v1/`
- Local diagnostics: `worldfoundry_hybrid_residual/results/wan_step_state_transition_exp000/audit_v1/`
- Remote raw sufficient statistics: `/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/results/wan_step_state_transition_exp000/`
