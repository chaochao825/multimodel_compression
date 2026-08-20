# RESULT-EXP-042: Motion-conjugated finite-horizon flow curvature

- Status: completed; teacher engineering failure
- Current stage: closed before scientific method evaluation
- Valid scientific outcome: none yet
- Implementation freeze:
  `worldfoundry_hybrid_residual/results/motion_conjugated_flow_curvature_exp042_implementation_freeze.json`

PLAN-047 produced valid minimal tensors but the shifted-base Euler-100/200
teacher comparison failed at `29.0653%` aggregate and `40.0219%` worst endpoint
relative L2. Initial states and first velocities match exactly, so this is
classified as an engineering failure before method evaluation, not evidence for
or against motion-conjugated curvature.

RDR-0023/PLAN-048 now test a uniform flow-sigma integration grid using only the
same development identities. No curvature-method result is recorded until a
teacher passes the engineering guard and later stages satisfy their protocol.

The uniform-sigma diagnostic also failed: Euler-100 versus Euler-200 is
`50.4400%` aggregate and `55.4407%` worst. Matched initial states and first
velocities are exact, but the worst identity reaches `16.5999%` state error by
the first quarter point. Uniform spacing therefore does not repair the teacher;
it removes high-sigma resolution that the shifted schedule appears to need.
PLAN-048 now measures one 200/400 contraction pair per grid before deciding
whether explicit Euler resolution can be repaired economically.

The uncontaminated shifted-grid refinement gives `24.8149%` at `200 -> 400` on
`dev01_turntable`, a contraction ratio of about `0.620` relative to its
`40.0219%` `100 -> 200` error. This is contraction but remains far above the
teacher guard. The uniform refinement is pending because an unrelated GPU2 job
started between capture phases; the overlapping attempt was quarantined before
it wrote a payload and is not evidence.

The clean uniform refinement subsequently completed through a bitwise-verified
single-H200 CFG path and gives `23.1600%` at `200 -> 400`. Its empirical order is
`1.2593`, versus `0.6896` for shifted Euler, but both remain far above the
teacher guard. Euler is therefore rejected as the teacher implementation; this
does not support or refute C-021 because no curvature arm has been evaluated.
RDR-0025/PLAN-049 now run one bounded midpoint-RK2 teacher Gate.

PLAN-049 completed all four registered midpoint captures without contamination.
Shifted midpoint RK2-100/200 gives `21.1244%` aggregate and `29.0273%`
worst endpoint relative L2; uniform gives `36.6326%` and `42.0954%`.
Both fail the unchanged `0.5%` refined-teacher guard. The final artifact audit
passes for eight payloads, 72 finite FP32 tensors, and `603,870,592` tensor
bytes. Initial state and first velocity match bitwise for every 100/200 pair.

This closes EXP-042 as a teacher engineering failure under RDR-0025. It does
not support or refute C-021: capacity, transfer, alignment gain, adapter cost,
calibration, validation, and test were never evaluated. A new float-time or
production-scheduler teacher definition would require a separate prospective
protocol rather than an unregistered continuation of this Gate.

Canonical evidence:

- `worldfoundry_hybrid_residual/results/motion_conjugated_flow_curvature_exp042_development/midpoint_rk2/teacher_resolution.json`
- `worldfoundry_hybrid_residual/results/motion_conjugated_flow_curvature_exp042_development/midpoint_rk2/final_capture_audit.json`
- `worldfoundry_hybrid_residual/results/motion_conjugated_flow_curvature_exp042_development/midpoint_teacher_diagnostics.json`
- `worldfoundry_hybrid_residual/results/WAN_MOTION_CONJUGATED_FLOW_CURVATURE_EXP042_TEACHER_DIAGNOSTIC_20260813.zh-CN.md`
