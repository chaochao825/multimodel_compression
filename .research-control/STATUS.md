# Research status

Updated: 2026-08-27

## Current decision surface

- North star: faithful, materially faster Wan video generation through a
  scientifically defensible training-and-system co-design.
- Primary claim: C-026, the released unquantized rCM four-step H200
  quality/latency baseline.
- Mainline: L-026 is active under accepted RDR-030; L-025 remains parked.
- Active plan: PLAN-056. EXP-047 / G-026 are active; PLAN-055 is complete.
- PLAN-057 / EXP-048 / G-027 are complete with a valid `null/adverse` result.
  L-027 is parked, C-027 is refuted, and no side probe remains active. EXP-047
  remains the unchanged sole mainline.
- L-024 remains parked; no additional rank, shift, router, rollout, or kernel
  work is authorized inside EXP-045.
- Locked data: the EXP-046 final four identities remain unopened and cannot be
  repurposed without a new accepted protocol.

## Latest belief-changing evidence

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
continues only through L-026/EXP-047.

## Active Gate

EXP-047 compares native 20-step UniPC, unchanged native four-step UniPC, and the
official unquantized four-step rCM release on one H200. It freezes four prompts,
two seeds, F81 generation, component timing, eight VBench dimensions, and
inter-seed diversity. A valid pass requires at least 3.5x denoiser and 2.5x
end-to-end speedup while retaining the registered teacher-normalized quality
and diversity guards. Physical-time long-video transport remains separate.
