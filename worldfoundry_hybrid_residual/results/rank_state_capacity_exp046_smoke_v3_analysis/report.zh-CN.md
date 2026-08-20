# EXP-046 whole-block rank-state capacity 分析

- 数据 split：`calibration`
- identity 数：`1`
- G-025 决策：`INCOMPLETE`
- 这是 target-visible 表示容量上界，不是可部署 observer、rollout 或速度结果。
- 状态因子没有保存；held-out target 只用于本次 SVD 容量测量。

## Rank-64 总体结果

- 聚合 block-output L2：`6.214%`
- 最坏 identity/branch L2：`7.045%`
- 通过 cell：`0/3`
- 最大 state-render / estimated exact MAC：`0.323%`

## 覆盖 Gate

| rank | step | horizon | passing layers | Gate |
|---:|---:|---:|---:|---|
| 64 | 4 | H1 | 0/1 | FAIL |
| 64 | 4 | H2 | 0/1 | FAIL |
| 64 | 4 | H3 | 0/1 | FAIL |
| 96 | 4 | H1 | 0/1 | FAIL |
| 96 | 4 | H2 | 0/1 | FAIL |
| 96 | 4 | H3 | 0/1 | FAIL |

## Rank frontier

| rank | aggregate output L2 | worst output L2 | remaining defect | pass cells | render/exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 15.419% | 18.559% | 100.000% | 0/3 | 0.000% |
| 8 | 8.370% | 9.553% | 30.375% | 0/3 | 0.040% |
| 16 | 7.811% | 8.888% | 26.554% | 0/3 | 0.081% |
| 32 | 7.097% | 8.061% | 22.023% | 0/3 | 0.161% |
| 64 | 6.214% | 7.045% | 16.956% | 0/3 | 0.323% |
| 96 | 5.651% | 6.401% | 14.057% | 0/3 | 0.484% |

当前 split 不构成正式 G-025 结论。
