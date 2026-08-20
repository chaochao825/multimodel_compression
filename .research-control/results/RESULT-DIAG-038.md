# RESULT-DIAG-038

- Experiment: DIAG-038 / PLAN-040
- Date: 2026-08-12
- Candidate: public Sol `cute_sm90` plus FA3 BF16, two-call static head groups
- Data identity: immutable DIAG-030 Layer-14 step-12 conditional latency QKV,
  SHA256 `d682537c68fe887c76cea25a779112ebc43e2c91ca80134e93fb069304cf272d`
- Configuration identity: global Morton3D `(21,30,52)`, Sol `tau=-1.5`,
  `diag`, `kv_splits=1`, no sink, dense-head counts `[1,2,3,4]`
- Artifact identity: formal manifest SHA256
  `233a96af2e77a2d0ba8b2aecbe5c33e385a1f1c3e173f61e6f197d6230e67dac`
- Evidence tier: engineering-only released-operator timing; globally exposed input
- Validity: valid after preregistered provenance-only repeat
- Outcome class: null (`grouped-runtime-null`)
- Protocol deviations: first complete execution omitted clock context and remains
  diagnostic only; the pre-result-authorized exact-method repeat added only
  pre/post GPU clock, P-state, power, and temperature records

## Observations

The formal all-head FA3 BF16 median was `15.6741 ms`. Sol/FA3 groups `11+1`,
`10+2`, `9+3`, and `8+4` reached respectively `1.3786x`, `1.3464x`, `1.2798x`,
and `1.2419x`. The registered `9+3` decision point was therefore below the
`1.5x` gate by `1.7976 ms`. Every output was finite, slice-consistency checks
passed, every CV was below 2%, and no foreign GPU process overlapped.

The earlier diagnostic run independently produced the same ordering and the
same null outcome. The formal repeat ran on the same H200 NVL in P0 at a 600 W
limit without hardware or thermal slowdown.

## Interpretation

The result supports a narrow implementation boundary: composing the released
Sol and FA3 APIs as two head-group calls loses enough head parallelism and adds
enough preprocessing, launch, synchronization, and concatenation cost that no
registered split reaches `1.5x`. It does not reject a single fused heterogeneous
kernel, content-aware sparse routing, trained methods, rollout quality, or
end-to-end acceleration.

## Claim update

No registered claim is updated. The payload is globally exposed and the Gate
contains no quality endpoint. This is an engineering null, not model evidence.

## Gate recommendation

Close PLAN-040 as `grouped-runtime-null`. Park the public two-call Sol/FA3 path
and open one bounded operator-readiness audit of a pinned, natively fused dynamic
sparse baseline before any fresh quality capture.

## Artifacts

- `worldfoundry_hybrid_residual/results/WAN_STATIC_HEAD_GROUP_SOL_FA3_DIAG038_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/results/sol_fa3_static_head_group_diag038_v1_rerun1/`
- `worldfoundry_hybrid_residual/results/sol_fa3_static_head_group_diag038_v1/`
- `worldfoundry_hybrid_residual/figures/static_head_group_sol_fa3_diag038_v1/`
