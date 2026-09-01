# Research status

Updated: 2026-09-01

## Current decision surface

- North star: faithful, materially faster Wan video generation through a
  scientifically defensible training-and-system co-design.
- Latest primary result: C-030 is supported. Exact resident UMT5 lifetime and
  fixed-negative embedding reuse close the rCM warm-service speed boundary
  while every distinct positive prompt is fully re-encoded.
- Mainline incumbent: L-030 is integrated at `9.638s` median F81 warm latency
  and `4.031x` speedup versus resident teacher20. L-026 remains the immutable
  released-rCM quality evidence inherited by this bitwise-exact runtime.
- EXP-052 / G-031 are complete with a valid `pass`; PLAN-061 is completed. No
  FP8 or approximate kernel was folded into this Gate.
- PLAN-062 is the sole active plan and performs successor selection only. It
  authorizes no new GPU run or approximation before one measured Gate is
  presented for researcher acceptance.
- EXP-047 / G-026 remain complete with a valid `speed-boundary` result.
  PLAN-056 is completed and superseded only for successor selection by RDR-035.
- PLAN-057 / EXP-048 / G-027 are complete with a valid `null/adverse` result.
  L-027 is parked, C-027 is refuted, and no side probe remains active. EXP-047
  remains the unchanged sole mainline.
- L-024 remains parked; no additional rank, shift, router, rollout, or kernel
  work is authorized inside EXP-045.
- Locked data: the EXP-046 final four identities remain unopened and cannot be
  repurposed without a new accepted protocol.

## Latest belief-changing evidence

- EXP-052's resident policy was bitwise exact at F17 and saved at least
  `15.208s/request` in the four-distinct-prompt text screen. Positive prompt
  cache hits remained zero and method network calls were unchanged.
- Resident teacher20/native4/rcm4 F81 medians were `38.846s`, `12.975s`, and
  `9.638s`. rCM therefore reached `4.031x` warm end-to-end and `10.076x`
  denoiser speedup, passing every exactness, memory, completeness, and timing
  guard.
- The rCM bottleneck is now VAE `4.308s`, denoiser `3.205s`, and serialization
  `1.796s`; text is `0.064s`. Prior `1.51x` local FP8 attention remains useful
  but is secondary to this measured exact floor and cannot be evaluated against
  the obsolete clear-after-request runtime.

- EXP-047 completed all 24 F81 formal videos and the frozen eight-dimension
  VBench plus inter-seed diversity suite. rCM reached `0.996875` mean and
  `0.970645` minimum teacher-normalized quality; all four prompt diversity
  ratios exceeded `1.11`.
- rCM denoiser timing was `3.177s` versus `32.202s` for teacher20 (`10.135x`),
  but warm end-to-end was `25.729s` versus `56.126s` (`2.181x`), below the
  registered `2.5x` guard. Text encoding, VAE, and serialization now dominate.
- Native4 reached only `0.866922` mean and `0.500000` minimum quality. The
  positive rCM result therefore comes from training the four-step finite-time
  map, not from merely reducing the unchanged solver's NFE.
- EXP-050/051 closed the video-understanding side probe. Residual-aware support
  improved fixed-state risk by `59.414%`, but true joint support-state training
  missed `0.5%/1%/2%` capacity and was `66.536%` worse than the best independent
  arm. Query/task-conditioned structure remains plausible; the fixed width-32
  post-hoc family is closed.

- EXP-048 completed the frozen 2x2 teacher/rCM weight by native4/rCM4 trajectory
  cross over four calibration and four untouched selection F17 identities.
- rCM/rCM4 rank-64 capacity and H1 output L2 were 22.460% and 30.843%, with
  0/10 layers passing both absolute gates. H2/H3 rose to 33.096%/35.734%.
- At fixed native4 and rCM4 inputs, rCM weights worsened H1 by 55.10% and
  34.93%; no layer improved at least 25% on both trajectories. The rCM4
  on-policy trajectory compensated only 2.86% of rCM H1 error.
- Two-lag history improved only 0.137% and a globally shared basis changed H1
  by only 0.049%. The null is therefore not explained by one missing lag or a
  simple teacher/rCM hidden gauge. Ordinary few-step distillation did not
  create the registered low-rate internal closure.

- EXP-046 completed four fresh selection identities with 2,880 finite rows.
  Target-visible rank 64 reached 4.833% aggregate and 11.093% worst
  block-output L2, with 0/60 cells passing; diagnostic rank 96 remained at
  4.345%/10.210% and also passed 0/60.
- Rank-64 H1 alone remained at 3.595%/4.104% for steps 4/6, so the null is not
  caused only by H3 open-loop instability. The best rank-96 cell still missed
  both quality limits at 1.189% aggregate and 1.716% worst.
- The successful smoke preserved the exact dense final latent with zero
  relative L2. G-025 is therefore a valid representation-capacity null, not an
  engineering failure.

- EXP-045 completed 4 calibration and 4 selection F17 identities over layers
  20--29, steps 4/6, and both CFG branches with 9,760 finite metric rows.
- DPLR-16 recovered 86.2%/85.9% of the target-visible one-step oracle gap, but
  passed only L21/L24/L25 at both steps and had no registered H2/H3 evidence.
- Broyden-2 was the strongest complete candidate: it passed only L24/L25,
  recovered 45.4%/57.5%, and reached 1.530x AR(2) risk in open loop.
- Q/K-selected nonperiodic transport added cost without improving the diagonal
  predictor. Even the target-visible oracle retained 6.86%/5.53% block-output
  relative L2 at steps 4/6.
- The result closes low-cost current-input observability as a broad late-layer
  skip mechanism, while preserving the narrower evidence that three layers
  contain a useful one-step current-input low-rank Jacobian component.

- EXP-043 returned `capacity-null`; every trained local finite-jump arm was
  worse than zero correction, and its registered interval sampler was severely
  starved at `[0, 1, 23, 1976]` updates.
- EXP-044 removed that sampler defect and separately tested interval labels,
  stage-specific weights, privileged motion, same-identity fitting, and four
  times the local width.
- EXP-044 returned `local-function-null`: the widest target-exposed
  transductive control remained at 8.190% endpoint relative L2 and 58%--96%
  per-interval curvature error. Four times the parameters improved endpoint
  error by only 1.6% relative.
- Public rCM, Causal-rCM, T2V-Turbo, VideoLCM, DOLLAR, Data-Forcing, SLA/SLA2,
  SALAD, and TurboDiffusion results still provide strong evidence that
  full-observability training-based acceleration is viable.

## Closed interpretation

The failed object is a local finite-jump correction predicted from latent,
guided velocity, and local motion. Training cannot recover prompt/CFG, global
hidden-state, attention, and UniPC-history information absent from that input.
This does not refute full students, hidden-state adapters, consistency/DMD
training, or trainable sparse attention.

Fixed BCM/BCCB/Butterfly attention, local finite-jump width/rank growth,
post-hoc whole-block rank-state growth, and rollout/kernel work for
L-022/L-023/L-024/L-025/L-027 remain closed. Released-rCM deployment evidence
continues through L-026/EXP-047 and the exact L-030/EXP-052 runtime.

## Closed primary Gate

EXP-047 compared native 20-step UniPC, unchanged native four-step UniPC, and the
official unquantized four-step rCM release on one H200. It freezes four prompts,
two seeds, F81 generation, component timing, eight VBench dimensions, and
inter-seed diversity. It closed as `speed-boundary`: quality, diversity,
completeness, and denoiser speed passed, while `2.181x` warm end-to-end missed
the `2.5x` guard. EXP-052 subsequently closed that systems boundary at `4.031x`
without changing generated values. Physical-time long-video transport remains
separate.

## EXP-049 Boundary

EXP-049 compared one conditional innovation family at the self-attention, FFN,
and whole-block boundaries on the exposed EXP-003 split. No target passed the
local promotion conjunction. At the frozen 1% policy, attention covered 5.926%
of calls at 1.127%/1.314% deployable error and only a 1.033x zero-renderer
end-to-end ceiling; FFN covered 1.759%; whole block covered none. The closest
attention quality point covered only 0.741% of calls. G-028 therefore closed at
its registered early stop without fresh suffix capture, training, or H200
timing. EXP-047 remains unchanged.
