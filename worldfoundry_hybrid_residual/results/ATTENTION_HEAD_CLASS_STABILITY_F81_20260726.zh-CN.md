# F81 Attention Head Class 跨 Seed 稳定性

日期：2026-07-26

## 结论

Wan2.1-T2V-1.3B 的 F81 layer-0/step-0 self-attention 在两个独立 seed 上表现出极稳定的 head 统计角色：entropy、temporal-tile geometry mass、top-64 mass 和 participation support 的跨 seed Pearson 相关系数都超过 `0.99998`；12/12 head 的规则类别一致，localized head 集合 Jaccard 为 `1.0`。

因此，**固定 head role + sample-adaptive block routing** 获得 provisional GO；但固定 token mask 与 frozen low-rank basis 仍是 NO-GO。该实验没有验证 prompt、深层 layer、晚期 timestep、端到端质量或 H200 kernel 加速。

## 实验范围

- 模型/形状：Wan2.1-T2V-1.3B，F81，32,760 tokens，12 heads，head dim 128。
- cell：layer 0、sampling step 0、conditional branch。
- 独立变量：seed `20260740` 与 `20260741`；Q/K/V 张量经 bit-level 审计确认不同。
- 每个 head 使用 128 个分层 query。
- geometry feature：`s3_temporal_pm2` attention probability mass。
- top-k feature：top-64、top-256、top-1024 mass。
- 计算仅使用 CPU，每个进程限制 4 线程、低调度优先级；不产生 H200 latency claim。

## 结果

| 指标 | 跨 seed 结果 |
|---|---:|
| normalized entropy correlation | 0.999995 |
| temporal-tile mass correlation | 0.999988 |
| top-64 mass correlation | 0.999996 |
| participation correlation | 0.999996 |
| rule-based class agreement | 12/12 = 100% |
| localized-head Jaccard | 1.0 |

规则类别在两个 seed 上完全一致：

- localized：heads `4, 5, 11`；entropy 约 0.34--0.50，geometry mass 约 0.94。
- transitional：heads `1, 6, 7`；entropy 约 0.62--0.73，geometry mass 约 0.65--0.73。
- diffuse：heads `0, 2, 3, 8, 9, 10`；大多数 entropy 接近 0.88--1.00，geometry mass 较低。

## 与冻结低秩失败的关系

两个结论并不矛盾：

- head role 是 entropy、support 和 geometry mass 等低维统计，跨 seed 很稳定；
- frozen correction basis 需要具体 defect 方向稳定，但最佳 rank-16 coefficient oracle 的 held-out 最大误差仍为 18.69%。

也就是说，**可以稳定预测“哪个 head 更局部”，但不能用一个固定低秩子空间准确预测“该 head 的缺陷向哪个方向修正”**。这支持低成本策略选择器，不支持静态 correction basis。

固定几何策略在 validation 上仍选择 0/12 sparse heads。即便 localized head 在目标 tile 中集中了约 94% 概率质量，剩余质量、softmax 归一化与 value 方向仍可能造成超过严格输出误差门的缺陷。因此不能从 head-class 稳定性直接推导 kernel GO。

## 后续实验门

1. 在 layer `0/14/29`、step `0/9/19` 上采集 compact head statistics，而不是全量 QKV。
2. 使用 prompt/seed factorial split，避免 layer-0/step-0 中 prompt 尚不可见的问题。
3. localized head 只测试 geometry sparse + shared normalization；transitional head 测试 coarse sparse + cache/dense tail；diffuse head 保持 dense 或仅做 cache 决策。
4. 必须同时满足跨 prompt/layer/step 类别稳定、pre-projection 输出误差、最终视频质量和 fused H200 kernel cost gate，才进入实现。
5. 若 deeper/later cells 的 class agreement 低于 75% 或 entropy/geometry correlation 低于 0.9，停止固定 head-role 路线，改为完全 sample-adaptive router。

## 产物

- 每 seed head 指标：`head_class_seed_stability_cpu_v1/seed*/attention_rmt_entropy_heads.csv`
- pair/head 明细：`head_class_seed_stability_cpu_v1/summary/head_stability_*.csv`
- 机器可读结论：`head_class_seed_stability_cpu_v1/summary/head_stability_summary.json`
- 可视化：`head_class_seed_stability_cpu_v1/summary/head_class_stability.png` 与 `.pdf`
