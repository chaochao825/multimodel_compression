# Support-state 协同设计最终判决：理论一致性、失败机制与研究边界

日期：2026-09-01
实验：`EXP-050`、`EXP-051`

## 核心结论

实验与此前的条件冗余理论一致，但它给出了比“结构化拟合失败”更具体的边界：

> frozen reader 中确实存在可由 exact innovation 修正的条件冗余；但是在 width-32 mergeable state、25% 规则 page 和当前正值 feature map 下，残差驱动的 support 与 state 联合适配没有形成稳定互补，深层反而出现负交互。

因此，当前失败不是“视频模型没有冗余”，也不是 control-variate 代数、训练发散或 page 预算错误。它否定的是一个明确的 post-hoc 函数类：

\[
\widehat y=
\frac{\widehat N_\theta+
\sum_{g\in\Omega_\theta(x)}(N_g-\widehat N_{\theta,g})}
{\widehat Z_\theta+
\sum_{g\in\Omega_\theta(x)}(Z_g-\widehat Z_{\theta,g})},
\]

其中 \(\widehat M_\theta\) 是 width-32 正值可加状态，\(\Omega\) 限定为 25% 四 token 规则 page。

## 1. 结果为什么与理论一致

条件风险分解为

\[
R(\widehat Y)=
\mathbb E\operatorname{tr}\operatorname{Cov}(Y\mid Z,H)
+\mathbb E\|\mathbb E[Y\mid Z,H]-\widehat Y(Z,H)\|^2.
\]

`EXP-050` 固定 bulk state 后，target-visible residual support 相对 mass support 将风险降低 `59.414%`。这证明 support 选择确实可以降低第二项，即已选函数类的逼近误差；它支持“exact evidence 应修正 state 的缺陷，而非只保留大 attention mass”这一窄机制。

`EXP-051` 允许 state 在同一 support 下继续适配后，最佳方法却变为 mass-trained state + mass support，视觉风险 `0.002095`。最终 joint residual 风险为 `0.003489`，相对最佳独立 arm 恶化 `66.536%`，95% bootstrap 区间 `[-142.261%,-0.227%]`。

这说明此前 positive interaction 主要是**固定 state 的条件效应**，不是稳定的 joint synergy。state 一旦可以吸收 support 后的可预测 bulk，残差的定义本身随 \(\theta\) 改变：

\[
D_{\theta,\Omega}=Y-\widehat Y_{\theta,\Omega}.
\]

于是 selector 与 state 构成非平滑双层问题：

\[
\min_\theta R(\theta,\Omega^*(\theta)),\qquad
\Omega^*(\theta)=\arg\min_{|\Omega|=B}R(\theta,\Omega).
\]

离散 support 每次变化都会旋转剩余缺陷，state 更新又会改变下一轮 support。统计冗余存在，并不保证这个 moving residual 可以由低带宽、跨样本稳定的规则 page 接口表达。

## 2. 公平四组比较给出的因果信息

| State 训练目标 | Mass support 风险 | Residual support 风险 |
|---|---:|---:|
| frozen whole-state | 0.014068 | **0.005710** |
| mass-support trained | **0.002095** | 0.007298 |
| residual-support trained | 0.005088 | 0.003489 |

三个结论可以同时成立：

1. 固定 state 时，residual support 有价值；
2. 固定 residual support 时，joint state 相对 mass-trained state 改善 `52.198%`，说明 state 适配不是无效；
3. 但 residual support 相对同一 joint state 的增益区间跨 0，且最终 joint 仍显著输给 mass-trained + mass。

所以问题不是某一个组件完全无用，而是两者没有在注册预算下形成可迁移的正协同。这正是 factorial comparison 比简单 `Q+S+L` 叠加更有价值的地方。

## 3. 为什么深层决定了 no-go

| Layer | 独立 arm visual mean | Joint visual mean | Joint 风险变化 |
|---:|---:|---:|---:|
| 0 | 2.064% | 1.941% | `+7.6%` |
| 13 | 3.168% | 2.649% | `+32.1%` |
| 27 | **6.825%** | **8.251%** | `-95.3%` |

Layer 0/13 仍保留部分协同，Layer 27 却出现重尾：joint P95/worst 达到 `19.604%/21.254%`。这与此前 Wan/reader 多条负结果的共同模式一致：越接近语义输出，动态 mode、value leverage 和任务条件越强，固定低带宽状态越难稳定覆盖。

这不是单纯增大 rank 就能解释的优化不足。训练曲线有限且整体下降；432 个评价行和 66 个训练记录均为 finite。只读重放还验证了：

- 144 个 trained-state cell 全部通过；
- 每个 head 严格读取 98/392 pages；
- state replay 最大误差 `4.77e-6`；
- all-page dense recovery 最大绝对/相对误差 `3.58e-6/2.36e-7`。

因此没有证据把 no-go 归因于数值实现。

## 4. 与最初 BCM/low-rank 动机的关系

最初目标是“规则 bulk + 小 residual”同时获得表达力和硬件效率。Support-state 保留了这个抽象：

- mergeable N/Z state 对应 bulk；
- exact pages 对应 sparse innovation；
- shared normalization 对应统一测度；
- 规则 page 对应 GPU 可执行结构。

但它没有恢复原始 Wan/DiT 加速主线，也没有证明 BCM/BCCB。对象已从固定权重/attention 矩阵变为未知未来查询下的 reader memory。`EXP-051` 表明即使在这个更合理对象上，post-hoc 规则分解也无法自动把条件创新变成足够低带宽的接口。

这进一步支持此前更一般的判断：

\[
\text{统计冗余}
\not\Rightarrow
\text{固定低秩/循环结构}
\not\Rightarrow
\text{低成本可观测接口}
\not\Rightarrow
\text{系统加速}.
\]

## 5. 是否解决真实核心问题

没有。当前只是 8 帧、单次 reader attention、三个 layer 的 exposed-development capacity screen；没有多问题长流、答案一致性、recent-window、StreamMem/StreamKV、merge writer、冷 K/V 搬运或 H200 latency。

更重要的是，本轮连本地 0.5%/1%/2% capacity ceiling 都未通过，所以继续训练 router 或测 H200 只会把函数类误差包装成系统实验。按预注册 stop rule，应停止这一 post-hoc width-32/25%-regular-page 方向。

## 6. 仍保留什么核心 insight

1. **共享 N/Z control variate 是正确组合方式。** all-page 精确恢复已验证，今后任何 sparse + state 方案都应保留这一代数。
2. **support 的价值必须相对于 state residual 定义。** `EXP-050` 证明该诊断有信息量，但 `EXP-051` 说明它不等于可训练 synergy。
3. **深层条件创新是主要边界。** 未来 train-native memory 必须显式看到多查询任务、语义 mode 和长期状态，而不能只压缩 post-hoc attention measure。
4. **capacity、observability、task、system 必须分层关闭。** 本轮在 capacity 就停止，避免无效 router 和 H200 工程。

若未来重新打开，应是独立的新主张：训练原生多查询 streaming state 与 differentiable regular router，在长视频 benchmark 上直接比较 recent-window 和现有 memory，并把 reader 质量与真实 latency 纳入同一目标。它不能作为本轮失败后的调参修复。

## 最终决策

- `G-030`: `NO_JOINT_SUPPORT_STATE_CAPACITY`；
- `C-029`: 在注册函数类内 refuted；
- `L-029`: parked；
- positions 97--120 保持未读；
- 不进入 router、task、confirmation 或 H200；
- Wan 加速主线仍为独立的 released rCM `L-026/EXP-047`。

可视化与原始数据：

- `figures/support_state_factorial_20260901.pdf`
- `figures/support_state_factorial_20260901_data.csv`
- `analysis/.../control_variate_support_state_capacity_dev_v5_valid/`
- `analysis/.../joint_control_variate_support_state_capacity_dev_v1/`
