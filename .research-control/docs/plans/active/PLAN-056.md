# PLAN-056: Measure the released rCM H200 Pareto

- Status: active (Gate complete; successor decision pending)
- Owner: researcher
- Mainline: L-026
- Completed experiment/Gate: EXP-047 / G-026 (`speed-boundary`)
- Resource scope: one H200, at most 12 GPU-hours and 40 GiB project storage

## Objective

Integrate the pinned official unquantized rCM Wan2.1-T2V-1.3B release and
measure its four-step F81 quality/latency Pareto against native 20-step and
native four-step generation on frozen prompt-seed identities.

## Steps

1. Pin the official repository/checkpoint and establish an isolated environment.
2. Pass import, checkpoint-load, and F17 engineering smoke checks.
3. Run the three-method F81 timing smoke with complete component accounting.
4. If valid, generate the frozen formal outputs and run the fixed quality suite.
5. Close G-026 and update the successor decision without reviving EXP-045/046.

## Stop rule

Stop on one valid G-026 outcome, two bounded integration failures, identity or
GPU-isolation invalidation, storage/time exhaustion, or user interruption.

## Outcome

EXP-047 closed validly on 2026-09-01. rCM passed quality, diversity,
output-completeness, and denoiser-speed guards, but reached only `2.181x` warm
end-to-end speedup against the registered `2.5x` threshold. L-026 remains the
measured high-quality incumbent. This plan remains active only to preserve
that incumbent while the researcher chooses a successor; it authorizes no new
experiment or resource use.
