# 从结构化冗余到条件化计算：视频理解与 Wan 的联合判决

日期：2026-09-01（补入 `EXP-052` exact resident H200 结果）

## 总结

当前结果不能用一个“成功/失败”概括。Wan 已经达到注册范围内的高质量、真实 H200
端到端明显加速预期；视频理解只达到 reader-state 表示压缩的容量预期，尚未达到
可交换、长流和真实系统部署预期。两条路线都没有验证“统一 BCM/BCCB、低秩远场
和稀疏事件分支本身就是主加速器”，但共同得到一个更准确的机制结论：

\[
\boxed{
\text{视频冗余真实存在，但其可压缩坐标由当前查询、任务读出或有限时间 flow map 决定；}
\quad
\text{raw hidden/residual 上的固定共享结构通常不是正确接口。}
}
\]

| 研究线 | 原始目标 | 当前判决 | 达到预期的部分 |
|---|---|---|---|
| 视频理解 | exact local/event support + mergeable far-field state | 表示容量 `BOUNDARY`，post-hoc joint state `NO-GO` | `7.86x` reader-state 缩减可基本保持任务准确率；reader-aware support 与 exact correction 有真实容量 |
| Wan post-hoc state | 用低秩/结构化状态跨去噪步跳过模块 | 未达到 | 定位到 current-state observability，而非 rank 容量不足 |
| Wan rCM + exact runtime | 训练原生 4-step flow map并兑现端到端速度 | **达到注册范围内的质量和速度预期** | 8 维质量 `0.9969`、多样性通过、denoiser `10.08x`、warm E2E `4.031x` |

## 起初理论哪些成立

起初的两类结构可以抽象为：

\[
M=B_{\mathrm{near}}+H_{\mathrm{far}}+S_{\mathrm{event}},
\qquad
\Delta W=W_{\mathrm{BCM}}+UV^\top.
\]

它们包含三个合理直觉：视频有局部连续性，少量事件具有高杠杆，远场信息可能
用低带宽状态表示。这些直觉没有被否定。EXP-050 中，在同一 frozen width-32
状态下，用 residual-aware exact pages 替代 mass support，使 paired visual risk
改善 `59.414%`；早期 reader-quotient 实验也显示，用 native reader Fisher metric
选择 exact support，候选 KL 可改善约 `72%--76%`。因此“结构化 exact support 应
围绕剩余误差和任务风险设计”是成立的。

失败的是更强的外推：存在局部/远场冗余，并不保证一个跨样本共享的固定 Fourier
特征向量、PCA basis 或低带宽 state 能承载它。最初分解更准确的条件化版本应写为：

\[
M(q,x)=B_{\mathrm{recent}}^{\mathrm{exact}}(q,x)
+H_{\mathrm{far}}(s(x);q)
+S_{\mathrm{innovation}}(q,x),
\]

其中 query/reader 在视频理解中决定坐标，有限时间 flow map 与条件输入在 Wan 中决定
坐标。固定结构只能是执行基或候选生成器，不能替代这些条件变量。对应的条件风险分解是：

\[
R(\hat Y)=
\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(Y\mid Z,H)
+
\mathbb E\|\mathbb E[Y\mid Z,H]-\hat Y(Z,H)\|^2.
\]

增加 BCM block、rank 或 expert 只能降低第二项；如果当前 query、mode、reader 或
flow coordinate 没有进入 `Z`，它无法降低第一项。EXP-005 将信息集从历史 residual
扩展到 current-input channel field 后，风险从 `0.141061` 降到 `0.072832`，达到
`1.937x` 并恢复 `0.877` 的 oracle gap；但只有 L21/L24/L25 三层满足 breadth gate。
这正是“条件变量有用但接口覆盖不足”的实证。

## 视频理解：reader 商空间存在，长期记忆接口未闭合

视频理解首先得到一个比 support-state 更稳定的表示信号。早期匹配状态实验中，
query-conditioned native memory 相对 recent baseline 得到 `+3.5pp` 趋势；
rank-256 + sparse residual 将每流状态从 `8.024 MiB` 降到 `1.024 MiB`，达到
`99%` prediction agreement，但未通过保守的 2% non-inferiority Gate。

更强的 OneVision 证据进一步确认了 reader quotient：`PCA-r456` 在 500 个完全未参与
选择的 MVBench 样本上把 tensor payload 从 `21.44 MiB` 降到 `2.73 MiB`
（`7.86x`），准确率从 `54.2%` 到 `55.2%`。但 prediction agreement 只有
`96.8%`，低于冻结的 `98%` 门槛。跨域 Video-MME 的 600 个样本又得到
`55.17% -> 54.17%`、agreement `95.83%` 和 harmful upper bound `2.603%`。
因此可以主张共享 semantic bulk 和 reader-null 方向真实存在，不能主张该状态与原生
状态逐样本可交换，也尚无 token-count、TTFT 或 H200 加速结论。

EXP-050 的 fixed-state 结果是一个真实正发现：在 25% 规则 exact pages 和同一
width-32 N/Z state 下，residual-aware support 将 visual mean/P95/worst 从
`11.124%/19.106%/21.330%` 降到 `6.759%/12.566%/19.383%`，并保留约
`3.842x` active-state 压缩。这证明 exact support 不应只追逐 attention mass，
而应主动塑造剩余 measure 的可压缩性。

但 EXP-051 完成真正 joint training 后，最佳 independent mass arm 已达到
`4.019%/7.651%/10.536%`，joint residual arm 却为
`4.281%/13.642%/21.254%`，paired risk 反而恶化 `66.536%`。所有 arm 都远高于
预注册 `0.5%/1%/2%` 门槛，因此没有打开 task transfer、sealed confirmation、
router 或 H200 timing。

新的 insight 不是“support-state 没有 interaction”，而是：

1. **interaction 非单调。** 固定 state 下最优的离散 support，会改变 state 的训练
   分布和 shared numerator/denominator 条件数；局部收益不能直接外推到联合最优。
2. **任务 metric 决定 support。** Fisher/readout-aware support 与欧氏 support 的
   overlap 只有约 `42.6%`，且 feature L2 与 reader KL 可以相反。
3. **有效 support 会随 query/content 旋转。** 静态跨任务 Fisher prior 曾使 KL
   恶化 `123%--191%`；OneVision 复现也只有 3/5 tasks 正向。
4. **正确代数不等于足够带宽。** N/Z control-variate 共享归一化是正确的，但
   width-32 state 仍无法满足严格视觉误差。
5. **视频理解比 Wan 更允许丢弃状态，但更受未知未来 query 限制。** reader-null
   方向可以安全丢弃而不会产生去噪轨迹累计误差；另一方面，open-query writer 不知道
   未来问题，固定小状态必须同时覆盖很大的 query family，深层条件创新因此成为瓶颈。

因此视频理解最有潜力的后续不是继续增加固定 rank/BCM，而是 query/readout-conditioned
小 router 或训练原生 multi-query memory：规则 page kernel 保留，support 由当前
问题与 reader sensitivity 选择，state 的训练目标直接使用任务风险。只有新的
capacity oracle 先通过，才值得做 sealed task 与 H200。

## Wan：post-hoc 结构失败，训练原生 flow map 与 exact runtime 成功

Wan 的历史结果系统性排除了 raw residual 上的简单后处理：

- EXP-046 的 target-visible rank-64/rank-96 仍为 `4.833%/4.345%` aggregate，
  0/60 cells 通过；
- EXP-048 的 rCM/rCM4 rank-64 capacity/H1 为 `22.460%/30.843%`；
- EXP-049 在 1% policy 下 attention 只覆盖 `5.926%` calls，零 renderer 的
  end-to-end ceiling 仅 `1.033x`。

这与物理时间 `tau` 和去噪时间 `lambda` 不可交换的分析一致。真实视频中的移动
坐标可使某个对象局部平稳，但不能推出跨去噪步 raw residual 存在固定 Toeplitz、
BCCB 或低秩算子。RoPE、AdaLN、CFG、attention routing 和遮挡都会使条件坐标变化。

EXP-047 则给出正向边界：官方 rCM 四步在 8 维冻结项目集上达到 teacher-normalized
均值 `0.9969`、最低 `0.9706`，四个 prompt 的多样性最低 `1.113`；denoiser
`10.135x`。未经训练的 native4 只有 `0.8669` 均值和 `0.50` 最低值。这说明：

\[
\boxed{
\text{Wan 最可压缩的对象是条件 finite-time flow map，而非任意内部 hidden residual。}
}
\]

rCM 与 EXP-048 不矛盾。学生网络可以把复杂、满秩、内容条件化的四步映射摊销到
权重中，却不需要让每个 late-block state 变成共享 rank-64 Markov state。

`EXP-047` 最初仍是 `speed-boundary`：旧 harness 下 warm E2E 只有 `2.181x`，低于
`2.5x`，因为它在每个请求后销毁并重建 UMT5。`EXP-052` 没有改 Wan、rCM、scheduler、
VAE、attention、dtype 或序列化，只把服务生命周期改成 resident，并且仍对每个不同的
positive prompt 完整执行 tokenizer 和 UMT5；仅全局固定 negative embedding 被精确复用。
F17 的 embedding 与 decoded CPU video 均逐 bit 相等。

在相同 resident policy 下，F81 median warm request 为：

| 方法 | Warm request | Denoiser | 相对 teacher20 E2E |
|---|---:|---:|---:|
| teacher20 | `38.846s` | `32.296s` | `1.000x` |
| native4 | `12.975s` | `6.401s` | `2.994x` |
| rCM4 | **`9.638s`** | **`3.205s`** | **`4.031x`** |

因此 Wan 已经通过原注册的质量和 `2.5x` 端到端速度门槛。这个 PASS 的科学对象是
“训练原生 finite-time flow-map + exact warm-service runtime”，不是 BCM、低秩或
FP8 的功劳。证据范围仍是四 prompt × 两 seed 的质量与四 prompt × 一 seed 的 timing，
不能外推成通用 SOTA 声明。

瓶颈也再次移动：resident rCM 中 VAE 为 `4.308s`（44.7%），denoiser 为 `3.205s`
（33.3%），serialization 为 `1.796s`（18.6%），text 仅 0.7%。所以后续应先在
`9.638s` 基线上比较 exact VAE/serialization 优化与 fused same-step attention。
历史 `1.51x` 局部 attention 加速即使覆盖全部 self-attention，乐观也只节省约
`0.58s`，对应 rCM endpoint 约 `1.06x` 增量；它不再是决定性杠杆。

## 联合新认识

1. **统计冗余不等于低秩，不等于廉价算子，也不等于端到端加速。** 每一步都需要
   独立 Gate。
2. **“捕获 80%--95% 能量”远不足以支持 0.5%--1% 输出误差。** 严格 endpoint
   要求几乎完整地恢复高杠杆尾部。
3. **结构应服务于条件，而不是替代条件。** BCM/Toeplitz/规则 block 可以继续作为
   kernel-friendly expert 或 candidate generator，但不能再充当全局语义模型。
4. **训练可以改变可压缩对象。** rCM 的正结果说明把有限时间映射直接蒸馏进学生，
   比对一个冻结模型做 post-hoc residual fitting 更有效；这叫 map compressibility，
   不等于内部 state 变成低秩。
5. **exact 系统语义也是算法 Pareto 的组成部分。** 同一数值计算仅改变 UMT5 生命周期，
   就把 rCM warm E2E 从 `2.181x` 提高到 `4.031x`。旧结果的低加速不是 rCM 算法上限，
   而是错误的服务生命周期测量。
6. **优化会移动瓶颈。** 当 NFE 从 20 降到 4 且 CFG 调用减少后，系统研究必须从
   denoiser 内部转向 VAE、序列化和整条生成链。

## 最终判决

视频理解没有达到部署预期，但达到了一部分表示预期：冻结 reader 的功能商空间可以
以约 `7.86x` 状态缩减保留总体准确率；同时获得了“task-conditioned exact support
与 state 必须协同、且固定共享低带宽 state 不能替代未来 query”的新机制证据。

Wan 的 post-hoc 结构化状态没有达到预期，但 rCM 已经同时达到质量、denoiser 和
resident warm-E2E 预期，形成明确的 finite-time flow-map 正结果。当前问题已从
“能否明显加速”转为“在 `9.638s` 基线上，VAE、serialization 与 same-step fused
kernel 各还能兑现多少增量”，而不是再证明一个固定矩阵结构。

因此最初方向并非完全失败，而是被重新定位为：

> **在条件化坐标中学习或选择可压缩接口，并把规则结构限制在可融合的执行层；
> 视频理解由 query/readout 决定坐标，视频生成由有限时间 flow map 决定坐标。**
