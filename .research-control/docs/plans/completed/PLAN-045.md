# PLAN-045: EXP-042 acceptance and implementation-identity freeze

- Status: complete
- Owner: researcher and agent
- Gate: none; EXP-042/G-021 remains proposed and non-executable
- Claims: C-021
- Candidate: L-021
- Lane: explore
- Resource cap: zero GPU-hours, zero fresh captures, zero training, zero rollout,
  zero held-out payload loading, and zero remote environment changes

## Decision question

Should the researcher accept, revise, or reject the proposed EXP-042/G-021
protocol for testing motion-conjugated finite-horizon flow curvature?

## Authorized work

1. Review the artifact audit, related-work boundary, equations, identity split,
   equal-budget controls, metrics, thresholds, and outcome mappings.
2. Apply researcher-requested protocol corrections without opening any data
   payload or changing the registered scientific endpoint silently.
3. If the protocol is explicitly accepted, implement and test the minimal Euler
   capture and common evaluator, then freeze their hashes in a successor plan
   before requesting execution authorization.

## Prohibited work

- Do not run Wan, allocate GPUs, capture trajectories, train adapters, decode
  videos, load validation/test tensor payloads, or change the remote environment.
- Do not infer acceptance from continued discussion, implementation detail, or
  available compute.
- Do not weaken the identity isolation, capacity/transfer/cost gates, or stop
  mappings after seeing data.

## Exit mapping

- Accepted: close PLAN-045 and open one bounded implementation-freeze plan; this
  still does not authorize GPU execution until capture/evaluator hashes exist.
- Revised: update the proposed protocol and repeat the zero-compute audit.
- Rejected: close PLAN-045 and park L-021 through a protected decision record.

## Completion outcome

The researcher explicitly accepted EXP-042/G-021 and removed the fixed
GPU-hour limit while retaining staged access and scientific stop guards.
RDR-0022 records this protected decision. No GPU execution or tensor-payload
access occurred under PLAN-045.
