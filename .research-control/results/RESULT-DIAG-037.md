# RESULT-DIAG-037

- Experiment: DIAG-037 / PLAN-039
- Date: 2026-08-12
- Protocol identity: `sol_attn_released_kernel_pareto_diag037_v3.json`, SHA256 `6dd3102d67dd0d782d5bb209c6837624c3d3821d4185acd1790eb7062725c5ad`
- Code identity: probe `e7f57f93a39d36b0942f56d3f9ac00e8abc6544fe5790f420e1335ecfe614556`; candidate commit `5dd502af9938d924be206c332ad1e911b4a925a1`
- Data identity: immutable DIAG-030 96-capture evaluation QKV; index `78e1af997f583d2c8f5611bf12629d736fb2c26471bbbdc79a4a177d473ab676`
- Configuration identity: global Morton3D `(21,30,52)`, tau `[-2,-1.5,-1,-0.5,0,0.5,1]`, `diag`, `kv_splits=1`, no sink
- Evaluator identity: FP32 tiled dense softmax; actual FA3 BF16 latency baseline
- Artifact identity/checksum: result manifest `a8fef89fd8626c6e50effad7fd5fbe5dc9d2ff8e6f52399cbcc71c1c898cb907`
- Evidence tier: development released-operator reproduction; all identities globally exposed
- Independent unit: one capture = sample × layer × step × CFG branch; 96 captures
- Validity: valid
- Outcome class: boundary (`quality-boundary`)
- Protocol deviations: none after pre-result revision 3; revisions 1/2 produced no candidate result

## Observations

FA3 BF16 passed the frozen control with maximum capture aggregate relative L2
`0.1798%`. The first Sol point above `1.5x`, tau `-1.5`, reached `1.5423x` but
passed only 34/96 captures at global aggregate/head/tile errors
`1.279% / 2.425% / 8.042%`. The nearest quality point, tau `-2.0`, reached
`1.3542x`, passed 40/96 captures, and had global errors
`0.826% / 1.526% / 7.296%`.

At tau `-2.0`, Layer 29 passed 32/32, Layer 0 passed 8/32, and Layer 14 passed
0/32. At tau `-1.5`, the counts were 32/32, 2/32, and 0/32. Timing had finite
outputs, zero foreign GPU PIDs, and wall-time CV below 2.7% for every method.

## Validity checks

All 96 QKV payload hashes, candidate source hashes, protocol/code hashes, FA3
ABI artifact hashes, strict `cute_sm90` dispatch, dual idle checks, and no-overlap
guards passed. Sol and FA3 received identical preordered QKV; Sol timing included
public preprocessing. This evidence can answer the released-operator local
quality-speed question, but not prospective transfer or rollout quality.

## Claim update

No registered claim is updated because DIAG-037 uses globally exposed data.
The result supports a narrow engineering boundary: the released fused kernel
has real H200 speed, but no frozen global tau meets this project's joint local
quality and speed guards. It does not reject heterogeneous or trained sparse
attention methods.

## Gate recommendation

Close PLAN-039 as `quality-boundary`; preserve the public Pareto and park further
tau/dynamic-fallback tuning on the exposed atlas.

## Artifacts

- `worldfoundry_hybrid_residual/results/WAN_SOL_ATTN_RELEASED_KERNEL_PARETO_DIAG037_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/results/sol_attn_released_kernel_pareto_diag037_v3/`
- `worldfoundry_hybrid_residual/figures/sol_attn_released_kernel_pareto_diag037_v3/`
- Remote raw result: `/opt/data/wangmeiqi/sol_attn_released_kernel_pareto_diag037_v3/result`
