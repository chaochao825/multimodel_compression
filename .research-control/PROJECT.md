# multimodel_compression_block_butterfly_20260802

Updated: 2026-09-01

## North star

Determine whether a pretrained Wan video generator can be accelerated with
negligible quality loss through a defensible combination of training,
structured computation, and H200 system optimization, with claims separated by
NFE reduction, per-step acceleration, and end-to-end wall-clock.

## Latest primary result

C-030 is supported. The released quality-passing rCM endpoint closes its H200
end-to-end speed boundary under an exact persistent warm-service runtime. The
runtime keeps UMT5 resident, reuses only the globally fixed negative embedding,
and fully re-encodes every distinct positive prompt. EXP-052 preserved bitwise
embeddings and decoded videos and reached `4.031x` median F81 warm end-to-end
speedup versus an identically optimized teacher20.

## Scientific value

The closed finite-jump program tested whether physical-time motion provides a
data-efficient coordinate system for low-cost few-step adaptation. EXP-043 and
EXP-044 showed that its local observable state is insufficient even after
balanced sampling, stage conditioning, same-identity fitting, privileged
motion, and four times the width. This negative boundary narrows future work to
full-observability students or hidden-state adapters rather than more local
matrix structure.

EXP-045 then tested the remaining low-cost observability hypothesis on fresh
F17 identities. Current block-input drift exposed useful one-step structure in
three late layers, but no complete causal method covered six contiguous layers
while remaining stable over two and three skipped steps. This closes the
post-hoc low-cost denoising-time predictor family without making a claim about
physical-time video dynamics.

EXP-046 separated representation from observability using the best
target-visible rank-state correction after the causal diagonal renderer. Rank
64 and diagnostic rank 96 both passed 0/60 cells, so the program stopped before
learning state coordinates, opening final identities, rollout, or kernel work.

RDR-030 therefore selects a released full-observability few-step student before
another state architecture. EXP-047 isolates NFE reduction with unquantized
rCM, native 20-step, and native four-step controls; sparse attention,
quantization, cache, and custom kernels remain disabled in this first Gate.

EXP-047 subsequently established rCM as a high-quality component-local
incumbent: `0.996875` mean teacher-normalized quality and `10.135x` denoiser
speedup. Its `2.181x` warm end-to-end result missed the `2.5x` guard because
text encoding, VAE decode, and serialization dominate after NFE reduction.
EXP-048 independently showed that this endpoint success does not induce a
post-hoc rank-64 late-block state closure.

RDR-035 therefore opened an exact runtime Gate before approximate kernels. The
EXP-047 harness destroys the official global UMT5 encoder after every request;
EXP-052 showed that a resident encoder and exact fixed-negative embedding reuse
remove at least `15.208s/request` of repeated text work. Positive prompt
embeddings remained uncached, all three methods used the same lifetime policy,
and rCM4 reached `9.638s` versus resident teacher20 at `38.846s`. The measured
runtime floor is now dominated by VAE decode, denoising, and serialization.

EXP-053 then tested whether the official BF16 VAE had an exact temporal-grouping
schedule. Chunk 4 was bitwise equal on F17 but failed equality on all four F81
prompts, reached only `1.1968x` complete-VAE speedup, and projected to
`1.0986x` request speedup. The Gate closed as an exactness null before endpoint
timing. This preserves L-030 and shows that short-horizon output equality does
not certify long-horizon decoder-cache closure.

EXP-054 then tested the next permitted same-step operator on the released rCM
trajectory. Installed Sage SM90 dense attention reached `1.5864x` over FA3
BF16 on the real F81 shape, but the calibration-frozen whole-cell atlas selected
`0/120` cells under the registered local error margins. Calibration and
evaluation risk topology was stable (`r=0.9819`), so the result is an absolute
fidelity/coverage null rather than evidence of random transfer collapse. The
Gate stopped before candidate rollout; L-030 remains the incumbent.

PLAN-066 subsequently compared the remaining exact-system and trainable
successors on the same `9.637995s` baseline. VAE remains the largest component
at 44.7% and needs only `1.119x` local speed to produce a `1.05x` request,
whereas the measured Sage speed under optimistic full attention coverage would
produce only `1.070935x`. RDR-038 therefore proposes an exact full-F81 VAE CUDA
Graph replay Gate that preserves the official 21-frame schedule and cache
semantics. The decision is pending; no candidate, experiment, or GPU run has
been opened.

## Success envelope

- Development feasibility: any successor must first establish a capacity
  oracle using the exact information available to its deployable student.
- Confirmatory evidence: a later fresh identity-disjoint Gate must freeze
  quality, diversity, latency, and fallback thresholds before training, then
  pass closed-loop rollout on multiple prompts and seeds.
- Fairness: all trainable comparisons must use identical teacher trajectories,
  teacher-call budget, trainable-parameter budget, optimizer steps, and NFE.
- Deployment evidence: claims require comparison to released
  4-step rCM/TurboWan on the same prompts, seeds, H200 software stack, video
  metrics, and wall-clock.
- Reproducibility: prompt, seed, scheduler, anchor, model, code, config,
  checkpoint, and artifact hashes are frozen before one-shot screen or test
  access.

## Non-goals

- Do not claim first few-step or motion-aware video distillation. Flow Map
  Matching, Shortcut Models, MeanFlow, rCM/Causal-rCM, T2V-Turbo, VideoLCM,
  Motion Consistency Model, and related work occupy those broad claims.
- Do not describe production UniPC anchor states as a converged probability-flow
  ODE solution. They are black-box production trajectory targets.
- Do not use screen/test targets for checkpoint selection, basis fitting,
  routing, hyperparameter selection, or fallback.
- Do not revive fixed BCM/BCCB/Butterfly attention, static low-rank attention,
  sparse attention, quantization, or cache components on prior failed defects.
- Do not claim SOTA quality or speed from local replay, arithmetic estimates,
  published third-party numbers, or a single prompt/seed.

## Preserved prior boundary

- C-021 remains untested. EXP-042 is preserved as an engineering failure caused
  by the nonconverged continuous teacher implementation.
- C-000--C-013 and C-015--C-019 remain refuted in their registered classes.
- C-014 and C-020 remain narrow target-exposed representation witnesses.
- C-022 and C-023 are refuted; L-022 and L-023 are parked after EXP-043/044.
- C-024 is refuted; L-024 is the closed incumbent after EXP-045 while protected
  successor selection remained pending.
- C-025 is refuted and L-025 is parked after the valid EXP-046 capacity null.
- L-025 is parked after its capacity null. L-026 remains the immutable quality
  incumbent after EXP-047; L-030 is the integrated exact-runtime incumbent
  after the valid EXP-052 pass.
- L-031 and L-032 are parked after valid exact-VAE and static low-precision
  attention nulls. Sage's local operator speed remains evidence, but neither
  candidate changed the L-030 endpoint.
- Exact dual-H200 CFG branch parallelism remains a valid independent system
  optimization; it is not part of C-022's algorithmic claim.

## Protected human decisions

The researcher owns changes to the north star, primary claim, mainline,
training/data budget, protected architecture, canonical repository, and external
release. Selecting a full-student rCM/DMD integration or a new hidden-state
adapter requires a new accepted decision rather than silent scope growth.

## Repository boundary

- Project root: repository containing `.research-control.json`.
- Control root: `.research-control`.
- Canonical implementation: `worldfoundry_hybrid_residual/`; exact Git and file
  hashes are written by each run manifest.
- Canonical model: local Wan2.1-T2V-1.3B checkpoint on server 236.
