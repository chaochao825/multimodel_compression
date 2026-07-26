# F81 几何稀疏与冻结低秩基底 H200 Pilot

日期：2026-07-26

## 结论

当前候选 **fixed geometry sparse attention + frozen low-rank tail** 为 **NO-GO**。不应为这一固定策略继续开发融合 kernel，也不应扩展到完整 72-cell 网格。

该结论仅针对 Wan2.1-T2V-1.3B、F81、layer 0、sampling step 0 的 Q/K/V replay。它不是端到端视频质量或 H200 时延结论，但已经足以否决当前候选：即使使用 held-out dense defect 求低秩系数的 oracle 上界，跨独立 seed 的最大输出相对误差仍为 18.69%，远高于 2% 严格门和 5% 宽松门。

## 实验契约

- nominal calibration：`s00_p00_seed20260740`
- nominal validation：`s01_p01_seed20260740`
- nominal test：`s02_p00_seed20260741`、`s03_p02_seed20260741`
- cell：layer 0、sampling step 0、cond/uncond 两个 CFG branch
- 每个 cell：12 heads、128 个固定 query
- 静态几何 mask 只允许 validation 选择，test 不参与选择
- 冻结 basis 只由 calibration 构造；ridge 特征与正则系数只由 validation 选择；test 仅评估一次

## 独立性审计

8 个 nominal replay 实际只有 2 个 bit-level 不同的 Q/K/V content group，content independence ratio 为 0.25。两个 nominal test sample 也只有 1 个独立 group，并检测到 4 对跨 split 的 bit-exact duplicate。

原因是 layer 0、step 0 的 self-attention 位于 cross-attention 之前：相同 seed 下，不同 prompt 和 cond/uncond branch 对这一位置的 Q/K/V 不可见。直接张量比较确认：

- `s00` 与 `s01` 同 seed、不同 prompt，Q/K/V bit-exact；
- 同一 sample 的 cond 与 uncond，Q/K/V bit-exact；
- `s02` 与 `s03` 同 seed、不同 prompt，Q/K/V bit-exact；
- seed `20260740` 与 `20260741` 的 Q/K/V 不同。

因此本 pilot 只能提供 **seed transfer** 证据，不能声称已经验证 prompt 或 CFG-branch generalization。若未来研究 sample-adaptive router，应在 text 已注入的更深层或更晚 step 重采样。

## 固定几何稀疏

校准后冻结的静态策略在 validation 上对 6 种 mask 都只能选择 0/12 sparse heads，所有 head 均回退 dense：

| Mask family | Validation sparse heads | Execution density | Gate |
|---|---:|---:|---|
| `s3` / `s5` | 0/12 | 100% | FAIL |
| `s3_temporal_pm2` | 0/12 | 100% | FAIL |
| `s3_tfull` / `s5_tfull` | 0/12 | 100% | FAIL |
| `s3_tfull_anchor12` | 0/12 | 100% | FAIL |

全 dense 回退的输出误差为 0，但其加速收益也为 0。`geometry_generalization_holdout.png` 中的零误差点因此不能作为稀疏策略有效的证据；应以 `geometry_pilot_decision.png` 为准。

## 冻结低秩尾部

Rank-16 的 test 汇总如下。coefficient oracle 使用 held-out dense defect 求系数，是不可部署的乐观上界；ridge 才是低成本预测系数的近似。

| Mask | Oracle mean / max error | Ridge mean / max error | Basis energy p05 | Overlap p05 |
|---|---:|---:|---:|---:|
| `s3_temporal_pm2` | 8.63% / 18.69% | 17.64% / 27.66% | 75.69% | 66.56% |
| `s3_tfull` | 13.18% / 23.09% | 30.39% / 55.57% | 77.33% | 66.45% |
| `s3_tfull_anchor12` | 12.18% / 22.56% | 29.55% / 55.63% | 77.07% | 65.93% |

即便 frozen basis 保留了约 76%--77% 的低分位能量，且子空间 overlap 约 66%，剩余误差仍显著超出高保真约束。说明局部 defect 在单 replay 上看似低秩，并不意味着该 basis 能跨 seed 固定；系数可预测性又进一步恶化结果。

## 决策与后续路线

1. 停止当前 fixed geometry + frozen basis 候选，不投入融合 sparse/low-rank kernel。
2. 保留 head-class 异质性这一发现，但只把它作为 sample-adaptive mixed-head router 的候选；必须先在更深 layer、多个 timestep、独立 seed 上证明 head class 稳定。
3. F81 主线继续完成独占 H200 的 exact CFG-parallel 基准，并优先研究能够改变实际 attention kernel 工作量的机制。
4. F17 主线继续 exact FFN/pointwise/kernel-fusion 基准，先量化可兑现的系统冗余。
5. 所有 H200 latency 结果必须通过中途 GPU ancestry/telemetry 独占审计；本 pilot 不产生 latency claim。

## 可复核产物

- 总决策：`geometry_generalization_h200_v2_compact/decision/geometry_pilot_decision.json`
- 独立性审计：`geometry_generalization_h200_v2_compact/independence_audit/qkv_capture_independence_summary.json`
- 固定几何策略：`geometry_generalization_h200_v2_compact/geometry_analysis/geometry_generalization_summary.json`
- 冻结 basis：`geometry_generalization_h200_v2_compact/basis_transfer/geometry_basis_transfer_summary.json`
- 主可视化：`geometry_generalization_h200_v2_compact/decision/geometry_pilot_decision.png`
