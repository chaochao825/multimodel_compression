# EXP-045 current-input observability 分析

- 数据 split：`selection`
- identity 数：`4`
- G-024 决策：`FAIL`
- 当前最强因果方法：`dplr16`
- 本报告只讨论 Wan 去噪采样步之间的 block residual 可观测性，不讨论视频物理时间预测。

## 最强方法 Gate 分解

- step 4 通过层数：`3/10`
- step 6 通过层数：`3/10`
- step 4 oracle 恢复率：`0.862`
- step 6 oracle 恢复率：`0.859`
- 最大 CFG branch harm ratio：`0.559`
- 最大 H2/H3 risk ratio：`inf`

Oracle 恢复率严格使用 `(R_AR2-R_method)/(R_AR2-R_oracle)`；target-visible oracle 仅是上界，不能成为运行时方法。

## 方法总表

| 方法 | L@step4 | L@step6 | recovery4 | recovery6 | output L2 | MACs | open-loop | G-024 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| broyden2 | 2 | 2 | 0.454 | 0.575 | 10.222% | 0.132G | 1.530 | FAIL |
| broyden4 | 2 | 2 | 0.452 | 0.573 | 10.236% | 0.204G | 1.530 | FAIL |
| diagonal | 2 | 2 | 0.450 | 0.510 | 10.415% | 0.048G | 6.769 | FAIL |
| dplr16 | 3 | 3 | 0.862 | 0.859 | 7.348% | 0.407G | inf | FAIL |
| dplr8 | 3 | 3 | 0.845 | 0.841 | 7.483% | 0.216G | inf | FAIL |
| transport2_history | 2 | 2 | 0.445 | 0.502 | 10.461% | 0.129G | 14.117 | FAIL |
| transport2_qk | 2 | 2 | 0.449 | 0.504 | 10.437% | 0.321G | 7.631 | FAIL |
| transport4_history | 2 | 2 | 0.443 | 0.501 | 10.481% | 0.201G | 14.188 | FAIL |
| transport4_qk | 2 | 2 | 0.446 | 0.501 | 10.463% | 0.393G | 7.925 | FAIL |
