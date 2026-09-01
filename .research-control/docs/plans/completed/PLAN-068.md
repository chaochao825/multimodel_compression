# PLAN-068: Execute exact full-F81 VAE CUDA Graph Gate

- Status: completed
- Owner: researcher and Agent
- Gate: G-034
- Claim: C-033
- Candidate line: L-033
- Lane: integrate
- Resource cap: one isolated H200, two GPU-hours, 5 GiB, two repairs

## Decision to unlock

Determine whether launch amortization of the unchanged official full-F81 VAE
is bitwise exact and materially improves the exact `9.637995s` resident rCM4
request.

## Authority and incumbent

- Accepted decision: RDR-038.
- Frozen protocol: EXP-055.
- Incumbent: L-030 / EXP-052.
- Closed neighboring families: EXP-053 temporal grouping exactness-null and
  EXP-054 train-free low-precision attention coverage-null.

## Milestones

1. Implement the minimum reusable graph wrapper, evaluator, and high-value
   exactness/stale-state tests without editing official rCM source.
2. Pass S0 on one isolated H200 with three distinct F17 inputs.
3. Only after S0 passes, run S1 on four F81 inputs and enforce exactness,
   complete-VAE speed, projected request, and memory guards.
4. Only after S1 passes, run the complete resident S2 endpoint.
5. Close G-034, update durable evidence and visualizations, then commit and push
   verified progress to the existing branch.

## Validation

- Local pytest, ruff, format, py_compile, config and diff checks.
- Remote tests in the frozen rCM environment before the first model run.
- Exactly one visible isolated H200 for every measured stage.
- Distinct-input equality, repeated-input stale-state equality, output
  ownership, peak memory, and full timing-boundary checks.

## Stop rules

Stop at the first valid protocol outcome, after two repairs, at two GPU-hours,
5 GiB, isolation loss, or user interruption. Do not turn a null into temporal
regrouping, compile, kernel fusion, quantization, or a custom VAE backend.

## Outcome

Completed as a valid `speed-boundary`. S0 and S1 passed exactness, stale-state,
memory, `1.1367x` VAE, and `1.0647x` projected-request guards. S2 retained
bitwise-equal CPU videos, four network calls, and `45870 MiB` peak reserved
memory, but measured `9.3257s` or `1.0335x` versus the `1.05x` absolute gate.
No graph variants, threshold changes, approximate kernels, or extra scientific
candidate were introduced.
