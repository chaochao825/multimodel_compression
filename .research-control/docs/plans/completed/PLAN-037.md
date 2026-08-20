# PLAN-037: Candidate selection after train-free operand closure

- Status: completed
- Owner: researcher and Agent
- Gate: portfolio decision only; no experiment is authorized
- Claims: none
- Candidate line: unresolved
- Lane: decide
- Resource cap: zero GPU hours and no implementation changes

## Decision to unlock

Choose whether the next bounded research line should reproduce a strong trained
sparse/quantized Wan attention baseline under the existing H200 evaluator, or
whether the custom strict-quality low-precision attention line should close.

This plan exists to keep the control plane mechanically valid while the
protected portfolio choice is pending. It is not an experiment protocol.

## Evidence requiring the decision

- DIAG-029 is an exposed implementation witness: 11-head Sage plus one BF16
  head reached `1.4931x` H200 attention speed and passed its eight exposed
  cells.
- DIAG-030 is the prospective boundary: only `4/12` layer-step cells were
  certifiable, producing `1.1172x` equal-cell atlas speed.
- DIAG-034/035/036 validly rejected the registered shared V rotation, 5%
  regular V support, and fine Q/K/V scale-granularity candidates.
- These nulls close the tested train-free operand function classes, but do not
  test trained QAT, learned sparse-linear routing, other number formats, or a
  faithful external baseline implementation.

## Options

1. **Replicate a trained baseline.** First verify public code, compatible Wan
   checkpoint support, training/calibration cost, kernel availability, license,
   and reported quality protocol. Then preregister one fair H200 reproduction
   with the existing prompts, seeds, quality guards, and dense baseline.
2. **Close this attention line.** Preserve the implementation witness and all
   nulls, and return to a separately accepted project-level candidate rather
   than opening another exposed QKV proxy.

## Recommendation

Prefer Option 1 only as a bounded replication, not as a new custom-method
claim. It is the lowest-cost way to determine whether the remaining gap is
caused by the train-free constraint or by the Wan/H200 evaluator itself.

## Stop rules

- Do not run GPU jobs, edit kernels, train adapters, or capture new data until
  the researcher selects an option.
- Do not reopen scale group-size, clipping, rotation, BCM/BCCB, sparse V-block,
  or post-hoc low-rank searches on the existing exposed atlas.
- A replication option must name one exact repository/commit, checkpoint,
  evaluator, cost cap, quality gates, and outcome mapping before execution.

## Closure condition

Close this plan only after the researcher selects an option and either a
bounded successor protocol is accepted or the attention line is explicitly
parked.

## Closure

- Closed date: `2026-08-12`.
- Outcome: Option 1 selected by the researcher's repeated instruction to
  continue high-information exploration and compare against strong same-class
  methods rather than close the line.
- Scope: selection authorizes a source and reproducibility audit only. It does
  not authorize training, kernel compilation, H200 timing, fresh capture, or
  rollout.
- Action: open PLAN-038 to select one exact public baseline and commit under a
  frozen feasibility protocol.
