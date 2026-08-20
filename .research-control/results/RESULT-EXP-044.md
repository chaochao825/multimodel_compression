# RESULT-EXP-044: Balanced finite-jump trainability decomposition

- Status: complete
- Validity: valid post-outcome development diagnostic
- Date: 2026-08-13
- Gate: G-023
- Claim: C-023
- Candidate: L-023

## Registered outcome

`local-function-null`. All four frozen checks failed:

- wide target-exposed transductive endpoint: 8.190% versus 1% gate;
- stage target-exposed transductive endpoint: 8.327% versus 1% gate;
- interval-FiLM observable transfer endpoint: 43.728% versus 2% gate;
- balanced shared observable transfer endpoint: 20.648%, 89.4% worse than
  the frozen EXP-043 observable reference rather than 25% better.

## Supports

- EXP-043 was not rescued by exact balanced interval counts.
- Adding an interval label, independent stage parameters, privileged terminal
  motion, same-identity fitting, or four times the local width does not make
  the tested local finite-jump function class high fidelity.
- Stage-specific parameters help relative to shared/FiLM transfer, but the
  result remains worse than zero correction and far outside the gate.
- The width-128 transductive control improves the width-32 target control only
  from 8.327% to 8.190%, a 1.6% relative reduction after four times the
  parameters.

## Does not support

- It does not refute full-model consistency, rCM, DMD, reward, or
  distribution-matching distillation.
- It does not establish an information-theoretic impossibility. Only two
  development identities, one architecture family, one optimizer, and one
  fixed training budget were tested.
- It does not compare generated-video quality or H200 end-to-end speed.
- It cannot support a publication claim because both identities were exposed
  before this diagnostic was registered.

## Mechanism interpretation

The progressive controls weaken sampler starvation, timestep ambiguity,
parameter sharing, identity transfer, motion estimation, and local-width
limitations. Remaining per-interval target errors are still approximately
58%--96% even for the widest same-identity privileged control. The most
consistent explanation is observability/function-object mismatch: a local
latent, guided-velocity, and motion stack does not contain the prompt, CFG,
global hidden-state, attention, and UniPC history needed to predict the finite
jump field.

## Decision

Refute C-023 within its development boundary and park L-023. Stop local
finite-jump correction, BCM/BCCB/Butterfly growth, and rollout/kernel work for
this candidate. Training-based acceleration remains viable only through a new
prospective decision that chooses either a released/full few-step student or a
hidden-state adapter with complete conditioning and rollout/distribution-level
training.

## Integrity

- Remote process exited 0 on an initially idle NVIDIA H200 NVL. The next
  foreign process started more than five minutes after `SUCCESS.json`, so there
  was no GPU-process overlap during the run.
- Twelve relevant unit tests passed before execution.
- Every run used exactly 8,000 updates and `[2000, 2000, 2000, 2000]` interval
  counts with fixed final-step checkpoint selection.
- Config, code, payload, checkpoint, summary, decision, and downloaded artifact
  hashes passed.

## Evidence

- `worldfoundry_hybrid_residual/results/production_finite_jump_trainability_exp044_v1/SUCCESS.json`
- `worldfoundry_hybrid_residual/results/production_finite_jump_trainability_exp044_v1/summary.json`
- `worldfoundry_hybrid_residual/results/WAN_TRAINING_POTENTIAL_AND_EXP044_20260813.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/production_finite_jump_trainability_exp044_20260813/production_finite_jump_trainability_exp044.png`
