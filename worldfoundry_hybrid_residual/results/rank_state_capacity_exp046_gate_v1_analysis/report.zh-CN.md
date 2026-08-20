# EXP-046 whole-block rank-state capacity 分析

- 数据 split：`selection`
- identity 数：`4`
- G-025 决策：`FAIL`
- 这是 target-visible 表示容量上界，不是可部署 observer、rollout 或速度结果。
- 状态因子没有保存；held-out target 只用于本次 SVD 容量测量。

## Rank-64 总体结果

- 聚合 block-output L2：`4.833%`
- 最坏 identity/branch L2：`11.093%`
- 通过 cell：`0/60`
- 最大 state-render / estimated exact MAC：`0.323%`

## 覆盖 Gate

| rank | step | horizon | passing layers | Gate |
|---:|---:|---:|---:|---|
| 64 | 4 | H1 | 0/10 | FAIL |
| 64 | 4 | H2 | 0/10 | FAIL |
| 64 | 4 | H3 | 0/10 | FAIL |
| 64 | 6 | H1 | 0/10 | FAIL |
| 64 | 6 | H2 | 0/10 | FAIL |
| 64 | 6 | H3 | 0/10 | FAIL |
| 96 | 4 | H1 | 0/10 | FAIL |
| 96 | 4 | H2 | 0/10 | FAIL |
| 96 | 4 | H3 | 0/10 | FAIL |
| 96 | 6 | H1 | 0/10 | FAIL |
| 96 | 6 | H2 | 0/10 | FAIL |
| 96 | 6 | H3 | 0/10 | FAIL |

## Rank frontier

| rank | aggregate output L2 | worst output L2 | remaining defect | pass cells | render/exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 42.957% | 392.024% | 100.000% | 0/60 | 0.000% |
| 8 | 7.455% | 15.279% | 31.521% | 0/60 | 0.040% |
| 16 | 6.541% | 13.504% | 25.384% | 0/60 | 0.081% |
| 32 | 5.678% | 12.424% | 19.749% | 0/60 | 0.161% |
| 64 | 4.833% | 11.093% | 14.589% | 0/60 | 0.323% |
| 96 | 4.345% | 10.210% | 11.898% | 0/60 | 0.484% |

在当前 renderer 下停止 rank-64 whole-block state 路线；该结论不否定训练原生状态或完整 few-step student。
