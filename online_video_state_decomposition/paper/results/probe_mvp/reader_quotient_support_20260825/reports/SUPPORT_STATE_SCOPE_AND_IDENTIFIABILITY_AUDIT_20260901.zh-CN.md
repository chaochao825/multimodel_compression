# Support-state 协同设计：问题边界、理论可识别性与实验协议

日期：2026-09-01

## 直接结论

四组比较是合理的，但它不会直接恢复最初的 Wan/DiT 加速主线。它检验的是另一项真实问题：

> 在不知道未来问题的长视频流中，能否用一个可合并 bulk state 保存广泛背景，并只为无法被该状态表达的条件创新保留规则 exact evidence？

这与最初“高秩结构主体 + 低成本残差”的抽象动机一致，但结构对象已经改变：固定 BCM/低秩矩阵被替换为任务条件下的充分统计量与 exact control variate。若实验成功，它首先是 streaming video understanding memory，而不是视频生成加速。

当前 8 帧 VSI 实验只能验证机制，不能证明解决领域核心困难。领域级结论至少还需要长视频、多时点、多问题、recent-window 强基线和真实内存/延迟。StreamingBench 每段视频提供五个不同时点的问题，正好暴露未知未来查询和状态复用；StreamMem 已直接研究 query-agnostic 固定大小 KV memory。[StreamingBench](https://arxiv.org/abs/2411.03628)、[StreamMem](https://arxiv.org/abs/2508.15717)

## 一、为什么 joint support-state 可能有本质收益

对未来查询 \(q\)，dense visual measure 为

\[
N(q)=\sum_i e^{q^T k_i}v_i,\qquad Z(q)=\sum_i e^{q^T k_i}.
\]

可加状态给出 \(\widehat N_{all},\widehat Z_{all}\)。对规则 page 集合 \(\Omega\) 精确读取后，正确组合不是两条归一化输出相加，而是控制变量：

\[
\widehat N_\Omega=\widehat N_{all}+
\sum_{g\in\Omega}(N_g-\widehat N_g),
\]

\[
\widehat Z_\Omega=\widehat Z_{all}+
\sum_{g\in\Omega}(Z_g-\widehat Z_g),\qquad
\widehat y_\Omega=\widehat N_\Omega/\widehat Z_\Omega.
\]

它有三个重要性质：

1. state 可以跨片段严格相加；
2. exact correction 与 state 使用同一测度和归一化；
3. 取回所有 page 时严格恢复 dense visual measure。

独立 support 最大化的是 attention mass 或 page criticality。联合 support 最大化的是“该 page 被精确修正后，state 剩余风险下降多少”。只有后者能主动降低残余条件创新：

\[
\Omega^*=\arg\min_{|\Omega|\le B}
R\left(\widehat M+sum_{g\in\Omega}(M_g-\widehat M_g)\right).
\]

因此 joint 相对 independent 的提升是一个可证伪的 interaction，而不是 Q+S+L 的组件叠加。

## 二、为什么它也很可能失败

当前证据给出的先验并不乐观：

- whole-state visual mean 为 31.51%，说明 width-32 指数核函数类严重失配；
- 25% exact + separately fitted tail 为 2.368%，仍超过 1%/2% 门槛；
- tail entropy 为 90%--94%，ESS 覆盖 44%--58% tail tokens，错误不是少量遗漏 outlier；
- mass 与 value-effect support Jaccard 为 87%--94%，简单 value-aware 排序几乎没有新 support。

joint 仍有一次合理机会，因为此前 state 是在固定 support 后拟合，support 并没有以 state residual 为目标。但若 target-visible residual support 都不能通过，就没有理由把失败归因于 router、优化器或工程实现。

## 三、怎样公平比较四组

必须分开“机制公平”和“部署公平”。

### 机制公平

- exact-only、state-only 是 factorial ablation；它们不用于证明交互。
- independent 与 joint 必须使用完全相同的 25% exact 比例、width-32 state、规则 page、共享 N/Z、训练数据和读取路径。
- 唯一变量是 support 是否针对 state residual 优化。
- 先跑 target-visible regular-page ceiling，消除 router 工程能力的混淆。

### 部署公平

参数量、FLOPs 和状态字节都不是时延。最终应为每种方法扫描 density/width，比较同一 H200 wall-clock 下的 Pareto 点，并计入：

- writer feature projection；
- page summary 和 query-router；
- cold exact K/V storage 与搬运；
- state read；
- exact correction 与共享归一化；
- fallback；
- 单问题和五问题时的摊销成本。

Quest 已说明 page criticality 依赖 query，并使用 page-level K min/max 进行低成本选择；这应是 deployable support 的最低强基线，而不是 tokenwise dense QK oracle。[Quest](https://arxiv.org/abs/2406.10774)

## 四、怎样避免工程失败掩盖潜力

实验按三阶段关闭不同不确定性：

1. **Capacity**：同一训练好 state，比较 mass correction 与 target-visible residual correction。失败就是函数类 no-go。
2. **Observability**：capacity 通过后，训练只读取 query 和 page summary 的 router。失败说明动态坐标不可低成本观测，不是否定容量。
3. **Task/system**：只有 router 通过才注入完整 reader、测 logits/accuracy，再测 H200。失败分别归类为任务迁移或系统成本边界。

每阶段在判决前检查 dense replay、page 数、全 page exact recovery、merge order、hard mask、finite rows 和梯度流。这样可以修复实现错误，但不能在看结果后改变 scientific identity。

## 五、它是否能达到最初预期

需要分两层回答：

- **不能直接达到**“Wan 视频生成在 H200 上高保真明显加速”的最初系统目标。正式主线仍是 released rCM 四步基线，support-state 不能替代其结果。
- **可能达到**最初更抽象的研究目标：以一个规则、可合并的 bulk 结构承担广泛冗余，以少量 exact innovation 保留重尾条件信息，并让二者在同一任务风险下协同分配预算。

若最终只能在 8 帧、单问题、解析 active-state ratio 上成立，它没有领域意义。若在 StreamingBench 等多问题长流上，以固定内存显著胜过 recent-window、StreamMem、StreamKV，并在 H200 上兑现延迟，才是在解决实际核心问题。近期 SimpleStream 甚至表明四帧 recent-window 可以匹配或超过许多复杂 streaming 方法，因此该基线必须纳入，不能以复杂 memory 本身作为进步证据。[SimpleStream](https://arxiv.org/abs/2604.02317)、[StreamKV](https://arxiv.org/abs/2511.07278)

## 六、最终 Gate

development 只有在 joint 相对 equal-cost independent 同时满足以下条件时才可提议 confirmation：

- paired task-risk 改善至少 25%，bootstrap 下界大于 0；
- dense answer agreement 至少 98%；
- harmful flip 为 0；
- merge/path consistency 小于 1%；
- active-state compression 至少 3x；
- local visual mean/P95/worst 不超过 1%/2%/5%；
- measured H200 latency 不高于 matched independent。

Stage 0 先使用更严格的 0.5%/1%/2% visual capacity ceiling。若它失败，停止 post-hoc support-state 路线，不增加 width、非规则 token mask、启发式 fallback 或 confirmation 使用。

## 七、执行结果

`EXP-050` 在 frozen whole-state 下观察到 residual-aware support 相对 mass
support 的 `59.414%` 风险改善，但 visual mean 仍为 `6.759%`。为排除
“state 未按 residual support 适配”的遗漏，`EXP-051` 在同一初始化、数据、
batch schedule、width 和 density 下完成了真正的 2x2 joint comparison。

最终最佳 arm 是 mass-support-trained state + mass support，visual
mean/P95/worst 为 `4.019%/7.651%/10.536%`。Joint residual arm 为
`4.281%/13.642%/21.254%`，相对最佳独立 arm 的风险恶化 `66.536%`，
95% bootstrap 区间 `[-142.261%,-0.227%]`。Layer 0/13 有局部正收益，
但 Layer 27 风险恶化 `95.3%`。

144-cell 只读重放通过 page budget、finite checkpoint、state replay 和
all-page dense recovery，因此 no-go 不能归因于 control-variate 实现或训练
发散。按冻结 stop rule，`C-029` 在注册函数类内 refuted，`L-029` parked，
positions 97--120 继续封存，不进入 router、task 或 H200。

完整判决见
`SUPPORT_STATE_EXP050_EXP051_VERDICT_20260901.zh-CN.md`，图见
`figures/support_state_factorial_20260901.pdf`。
