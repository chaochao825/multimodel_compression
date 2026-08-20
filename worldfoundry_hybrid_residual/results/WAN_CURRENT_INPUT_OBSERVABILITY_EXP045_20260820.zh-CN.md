# Wan 当前输入可观测性实验：EXP-045 结果与机制分析

日期：2026-08-20
模型：Wan2.1-T2V-1.3B
决策：G-024 `FAIL`

## 1. 科学问题与边界

EXP-004 已说明：只从过去 block residual 的范数、余弦和差分，无法恢复
Wan 晚层随去噪采样步变化的动态坐标。EXP-045 进一步问：当前 block input
`h_k`、当前 AdaLN 和极廉价 Q/K sketch 是否补足了这部分可观测性。

它测试的是 **denoising sampler time**，不是视频第一秒到第二秒的物理状态
传播，也不是自回归长视频 KV cache。

主候选为：

\[
\hat r_k = D(z_k)r_{k-1}
+\sum_m g_m(z_k)P_mr_{k-1}
+B_k(h_k-h_{k-1}),
\]

其中 `P_m` 只作用于 `5 x 30 x 52` 时空 token 网格，采用非周期零边界，
不对无几何意义的 hidden-channel 序号做平移。

## 2. 公平实验设置

- F17、480x832、20-step UniPC、CFG scale 5.0。
- 连续 blocks 20--29，目标 sampler steps 4/6，conditional/unconditional 双分支。
- 4 个 calibration prompt/seed 只拟合 matched scalar AR(2)；4 个 selection
  prompt/seed 只用于 G-024；4 个 final identity 全部未打开。
- Broyden/secant 只读取最后一次 exact anchor 之前的 residual；H2/H3 中间
  residual 始终使用预测值，不允许回读 exact tensor。
- Q/K sketch 只投影各 8 个固定行，并将 `2*N*D*8` 计入成本。
- Oracle 恢复率严格使用
  `(R_AR2 - R_method) / (R_AR2 - R_oracle)`。

正式运行得到 9,760 条有限指标，无 selection-to-calibration 泄漏。

## 3. 核心结果

| 方法 | step 4 residual risk | step 6 residual risk | 聚合 block-output L2 | 2x 通过层数 | H2/H3 |
|---|---:|---:|---:|---:|---|
| frozen AR(2) | 35.082% | 27.040% | 14.124% | 基线 | 基线 |
| current-input diagonal | 26.194% | 19.489% | 10.415% | 2/10、2/10 | 最坏 6.77x |
| Broyden-2 | 26.121% | 18.526% | 10.222% | 2/10、2/10 | 最坏 1.53x |
| DPLR-8 | 18.394% | 14.588% | 7.483% | 3/10、3/10 | 未注册 |
| DPLR-16 | 18.062% | 14.324% | 7.348% | 3/10、3/10 | 未注册 |
| target-visible 75-shift oracle | 15.342% | 12.239% | 6.86% / 5.53% | 上界 | 不可部署 |

DPLR-16 的 oracle-gap recovery 为 step4 `86.2%`、step6 `85.9%`，但只有
L21、L24、L25 通过 2x 门槛。L26--L29 只有约 `1.02x--1.22x`，不存在
“越晚层越容易复用”的单调规律。

## 4. 新增 insight

### 4.1 当前输入确实包含信息，但只暴露部分 Jacobian

diagonal 方法相对 frozen AR(2) 有稳定改善，说明：

\[
\Delta r_k \approx J_h(h_k-h_{k-1})
\]

中的 channel-diagonal 成分是真实存在的。DPLR-8/16 进一步下降，说明三个
层还存在低秩 channel-coupling 成分。因此 EXP-004 的失败不能解释成“晚层
完全没有时间复用”，更准确的结论是过去 residual 不足，而当前 `h_k` 只
恢复了局部动态坐标。

### 4.2 容量增加已开始饱和

Broyden-2 到 Broyden-4 几乎不变；DPLR-8 到 DPLR-16 的聚合 block-output
L2 只从 `7.483%` 降到 `7.348%`。问题不再是简单缺 rank，而是正确映射随
layer、step 和内容变化，且当前历史中的 secant span 不能稳定外推。

### 4.3 规则 transport 没找到目标坐标

history/QK 选择的 2/4 个非周期 token shift 都与 diagonal 近似持平。
QK-2 的平均成本约 `0.321G` MAC，高于 history-2 的 `0.129G`，质量却没有
改善。这否定了“少量固定行 Q/K 就足以选择正确 transport expert”的命题，
而不是否定物理时间中的运动传输。

### 4.4 单步收益不能转化为连续跳步

Broyden-2 在 H1 有收益，但 H3 最坏达到 matched AR(2) 的 `1.53x`；diagonal
和 transport 的 H3 放大更严重。secant 在 anchor 附近可作为局部线性化，
离开 anchor 后 Jacobian 变化与预测状态漂移共同累积，固定局部算子失稳。

### 4.5 当前 oracle 本身不具备高保真替换能力

target-visible oracle 的 block-output L2 仍为 `6.86%/5.53%`，远高于
`1%--2%` 的组件替换目标。因此失败发生在 router 之前：即使看到目标，
当前 temporal/75-shift 函数族也不能生成足够准确的 residual。训练一个
width-32 gate 只会选择已有错误算子，不能修复函数类上限。

## 5. 决策

不继续增加 DPLR rank、secant 数、Q/K 行数或 shift expert；不训练 router，
不打开 final split，不做 approximate rollout 或 kernel benchmark。

该结果只关闭“低成本当前观测 + post-hoc sampler-step residual predictor”
作为连续晚层通用跳步机制。下一条有科学依据的主线需要显式选择：

1. 复现并集成一个 full-observability 的 released few-step student，建立真实
   H200 质量/速度基线；或
2. 训练原生 hidden-state/state-render 模型，让动态状态在训练时成为可观测
   变量，而不是事后从局部 residual 猜测。

物理时间长视频传输仍应使用独立数据对象和 claim，不能由本实验外推。

## 6. 可视化与原始证据

- `current_input_observability_exp045_gate_v1_analysis/risk_reduction_by_layer.png`
- `current_input_observability_exp045_gate_v1_analysis/open_loop_stability.png`
- `current_input_observability_exp045_gate_v1_analysis/quality_cost.png`
- `current_input_observability_exp045_gate_v1_analysis/method_gate.csv`
- `current_input_observability_exp045_gate_v1_analysis/layer_gate.csv`
- `current_input_observability_exp045_gate_v1_analysis/source_all_cell_metrics.csv`
