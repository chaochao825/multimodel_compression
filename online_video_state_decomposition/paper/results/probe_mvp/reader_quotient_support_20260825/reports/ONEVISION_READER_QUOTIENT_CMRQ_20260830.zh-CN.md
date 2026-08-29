# 三谱一耦合审计与 Progressive CMRQ 校准结果

日期：2026-08-30

对象：冻结 LLaVA-OneVision Qwen2-7B 视觉状态

证据层级：fresh calibration / cross-fit mechanism probe

判决：`CALIBRATION_CONDITIONAL_GO`，尚未进入 selection 或 formal

## 1. 直接结论

附件提出的“feature energy、reader sensitivity、margin 三谱及其耦合”与已有结果
和本轮新实验整体一致，而且确实帮助找到了比继续旋转固定 PCA/BCM 更合理的核心：

> 用多域 pooled PCA 表示稳定的高能 bulk，用少量 calibration-only reader-risk
> 原子保护决策边界，再由**压缩结果自身的 top-1 margin**触发 exact state。

这条路线可称为 **Progressive CMRQ**。它不是新的大函数族，而是把统计压缩、功能
风险与有限样本安全性分成三个接口：

\[
\underbrace{U_{\rm bulk}}_{C:\text{稳定能量}}
+
\underbrace{U_{\rm boundary}}_{R:\text{读出风险}}
+
\underbrace{\mathbf 1[\widehat M\le\tau]\,X}_{M:\text{精确回退}}.
\]

72 个严格 leave-one-fold-out calibration endpoint 给出的关键结果是：

- 静态 `risk-32` 的 mean KL 为 `0.001497`，比 pooled PCA 低 `25.45%`，
  agreement 为 `98.61%`，`1` 个 mismatch、`0` harmful；
- 静态 `boundary mix (g=32, w=0.3)` 的 mean/P95 KL 为
  `0.001414/0.003218`，比 pooled PCA 的 mean KL 低 `29.56%`，比同权重
  permuted-risk null 低 `32.05%`，但仍有 `2` 个 mismatch、`1` harmful；
- 使用**压缩态** margin `<=0` 触发 exact 后，boundary mix 回退 `7/72`
  (`9.72%`)，唯一 harmful 被捕获，剩余 `1` mismatch、`0` harmful；
- 按“先传压缩态、再请求完整态”的保守成本，状态传输仍缩小 `4.46x`；若能在
  发送前完成路由，理想摊销为 `4.72x`；
- margin `<=0.125` 可消除所有 mismatch/harmful，但回退升至 `31.94%`，
  保守传输比降至 `2.24x`。

因此当前不是 formal positive。准确判决是：**静态同秩 CMRQ 没有同时支配连续风险
与离散安全；progressive exact tier 在校准端点给出值得进入一次 fresh selection 的
条件性信号。**

![三谱错位、跨折风险旋转、同秩 CMRQ 与 progressive fallback](../figures/onevision_reader_quotient_cmrq_stage_b.png)

## 2. 实验边界与公平性

本轮没有复用已经观察完的 MVBench task 作为新 endpoint，而是冻结 VSI-Bench 的
scene-level 划分：120 calibration、60 selection、63 formal scenes。当前只读取了
calibration；selection/formal 均未运行。

- source、target、VSI 三域各使用 120 个视频估计 covariance，消除了此前
  source-20/target-120 的主要样本量混杂；
- rank 固定为 `456`，每个状态 coefficient payload 固定为 `2,860,032 bytes`；
- dense native state 固定为 `22,478,848 bytes`；
- reader、frame policy、candidate token、BF16 数值路径均未改变；
- 三个 reader-risk fold 每个使用 24 个 calibration scenes；
- 每次 exact evaluation 都使用另外的 24 scenes；48-scene risk fit 只由另两折构成；
- risk/null/boundary-mix 全部同 rank、同 payload；
- benchmark answer 不参与 basis 构造，teacher prediction 仅用于行为保持诊断；
- progressive router 使用近似 reader 自己产生的 margin，不使用 exact teacher margin。

第一次 merged-risk 汇总和第一次 identity-count 执行失败均原样保留，后续修复使用新
输出目录，没有覆盖失败证据。本轮数值汇总来自完整、身份匹配的 `_v2` 与三折 LOO 目录。

## 3. 三个谱分别告诉了什么

### 3.1 Feature spectrum：容量低维，但 rank 边界不稳定

同预算 rank-456 的 feature explained energy 为：

| Domain | Explained energy | Tail | Relative eigengap at 456 |
|---|---:|---:|---:|
| source | `97.503%` | `2.497%` | `0.074%` |
| target | `96.357%` | `3.643%` | `0.088%` |
| VSI | `97.182%` | `2.818%` | `0.132%` |

这确认“统计状态容量低维”，但三个 eigengap 都很小。source-target rank-456 overlap
为 `0.8618`，source-VSI/target-VSI 为 `0.8499/0.8504`；最弱 principal
direction 的 cos² 接近零。bootstrap 又显示 cross-domain overlap 随每域样本数从
20 到 120 明显上升，说明过去的一部分“domain rotation”确实是有限样本 covariance
估计误差，但完整 120-video 结果仍保留真实的旋转 tail。

结论不是“PCA 不可用”，而是：

\[
\boxed{U_{1:424}\text{ 一类的 bulk 较稳定，rank 边界附近不能仅按方差决定。}}
\]

### 3.2 Reader-risk spectrum：更重尾，而且随 query/scene 旋转

三个独立 24-scene risk fold 的 rank-456 尾能量为
`9.96% / 12.25% / 6.23%`，显著高于 feature tail。把两个 fold 合并为 48 scenes后，
尾能量为 `10.04%–14.72%`，没有因更多数据自动变成低秩；相反，多样 query 的 active
directions 联合起来后，静态 union rank 会增长。

更关键的是迁移稳定性：

- risk-matrix cosine 只有 `0.230–0.274`；
- top-32 risk subspace overlap 只有 `0.179–0.212`；
- top-456 overlap 也只有 `0.324–0.367`；
- 一折 top-32 通常只能捕获另一折约 `10%–28%` 的 risk trace。

所以 reader risk 确实存在，但不能被解释成一个跨 query 固定的 Fisher basis。这与
此前静态 Fisher prior 反向迁移、Wan 中静态 low-rank/BCM correction 失效的机制一致：
**不是没有敏感结构，而是敏感坐标随条件旋转。**

### 3.3 Margin spectrum：有限样本 flip 由联合尾部决定

fold-0 的一阶 margin surrogate 对 source PCA 的 exact shift Pearson 为 `0.877`，
但 relative L2 仍为 `0.501`；对 VSI PCA 只有 `0.640/0.815`。这说明 gradient
可以提出候选方向，却不足以替代 exact reader 选择，尤其不能把平均线性相关当成
flip certificate。

本轮最有信息量的样本是 `vsi_question_1077`：boundary mix 把一个原本只有
`0.125` 的 teacher margin推成严格 tie，造成唯一 harmful flip。压缩态 top-1 margin
恰为 0，因此无需 exact teacher 信息即可触发回退。这正符合

\[
\Pr(\text{flip})
\le
\Pr(M\le\tau)+\Pr(A\ge\tau),
\]

以及“小平均 KL 不可能提供 margin-independent top-1 保证”的理论。

## 4. 耦合：为什么 PCA-only 与 risk-only 都不对

fold-0 的同秩诊断给出了非常清楚的 Pareto 几何：

| Basis | Feature capture | Reader-risk capture | Exact mean KL |
|---|---:|---:|---:|
| pooled PCA-r456 | `96.747%` | `24.780%` | `0.001384` |
| risk-only-r456 | `26.776%` | `90.037%` | `0.020477` |
| bulk + risk-16 | `96.639%` | `67.863%` | `0.001370` |
| bulk + risk-32 | `96.523%` | `72.312%` | `0.000967` |
| bulk + risk-64 | `96.270%` | `76.926%` | `0.001406` |
| bulk + risk-96 | `95.981%` | `79.731%` | `0.001636` |

feature-risk commutator ratio 在三折为 `0.073–0.096`。它不是“很大即可证明无共同
基”的绝对阈值，但结合仅 `24.8%` 的 PCA risk capture，足以否定“高 feature
variance 等价于高 reader leverage”。同样，risk-only 虽覆盖 90% 风险，却删除了
约 73% feature energy，exact KL 反而扩大一个数量级。

因此正确对象不是简单

\[
\max \operatorname{tr}(P C)
\quad\text{或}\quad
\max \operatorname{tr}(P R),
\]

而是固定 rank 下的耦合约束：

\[
\max_{P=P^2=P^\top,\;\operatorname{tr}P=456}
\operatorname{tr}(PC)+\lambda\operatorname{tr}(PR),
\]

并且最终必须由 exact finite reader behavior 与 margin tail 判定。风险原子的作用是
提出一个小 union span，而不是把 average Fisher 当作最终 projector。

## 5. 同秩 CMRQ 的实际结果

三折 48-risk-fit → 24-exact-eval 的 72 个端点汇总如下：

| Method | Mean KL | P95 KL | Feature L2 | Agreement | Mismatch | Harmful |
|---|---:|---:|---:|---:|---:|---:|
| pooled PCA | `0.002008` | `0.006754` | `0.1456` | `97.22%` | 2 | 1 |
| VSI PCA | `0.001645` | `0.005796` | `0.1395` | `95.83%` | 3 | 1 |
| risk-32 | `0.001497` | `0.005186` | `0.1505` | `98.61%` | 1 | 0 |
| feature-only null-32 | `0.001928` | `0.005969` | `0.1450` | `97.22%` | 2 | 1 |
| random null-32 | `0.001671` | `0.004394` | `0.1506` | `95.83%` | 3 | 2 |
| permuted-risk null-32 | `0.001734` | `0.004881` | `0.1506` | `98.61%` | 1 | 0 |
| boundary mix w=0.3 | **`0.001414`** | **`0.003218`** | `0.1505` | `97.22%` | 2 | 1 |
| permuted mix w=0.3 | `0.002082` | `0.005811` | `0.1506` | `97.22%` | 2 | 1 |

配对 bootstrap 显示：

- risk-32 比 feature-only null 改善 `22.38%`，95% paired-delta CI 不跨零；
- risk-32 比 pooled/VSI/random/permuted-risk 的点估计改善为
  `25.45%/9.02%/10.43%/13.66%`，但对应 CI 均接触或跨零；
- boundary mix 比 pooled 改善 `29.56%`，paired-delta CI 为负；
- boundary mix 比同权重 permuted mix 改善 `32.05%`，CI 为负；
- boundary mix 相对 VSI 的 `14.04%` 与相对 risk-32 的 `5.51%` 尚不显著。

这证明 risk 信息不是“多换 32 个方向”或随机 union span 的伪增益，但也证明样本数
不足以宣称静态 risk basis 普遍优于所有强 null。低权重 `0.0003–0.01` 的 boundary
mix 没有形成可靠 Pareto；较强 `w=0.3` 才改善连续尾部，却暴露了 harmful tie。

## 6. Progressive CMRQ 为什么是更合理的核心

固定 basis 仍然面对 query-conditioned union rank。最小改进不是继续增加 risk atom，
而是承认存在不可压缩边界，并使用当前近似读出的可观测不确定性：

\[
\widehat X=\mu+ZU^\top,
\qquad
\widehat M=\widehat z_{(1)}-\widehat z_{(2)},
\]

\[
X_{\rm delivered}=
\begin{cases}
\widehat X,&\widehat M>\tau,\\
X,&\widehat M\le\tau.
\end{cases}
\]

压缩态 margin 的结果为：

| Method / threshold | Fallback | Remaining M/H | Effective mean/P95 KL | Conservative transfer |
|---|---:|---:|---:|---:|
| mix, `0` | `7/72` | `1/0` | `0.001282/0.003218` | **`4.46x`** |
| mix, `0.125` | `23/72` | `0/0` | `0.001043/0.003218` | `2.24x` |
| risk-32, `0` | `8/72` | `1/0` | `0.001360/0.005186` | `4.20x` |
| pooled PCA, `0` | `7/72` | `1/0` | `0.001865/0.006754` | `4.46x` |
| VSI PCA, `0` | `9/72` | `2/0` | `0.001432/0.004835` | `3.96x` |

`tau=0` 并不是经 teacher margin 搜索出来的任意超参数，它表示近似 reader 无法给出
严格 top-1 顺序时回退。它捕获了本轮唯一 harmful 事件，但 72 个 calibration 样本
不足以把 `0 harmful` 解释成概率保证。`tau=0.125` 是诊断曲线，不应在观察这些结果
后直接冻结为部署阈值。

这条路线的创新边界也必须写清：selective prediction、early exit、progressive coding
和 uncertainty fallback 都有大量先例。这里可能有价值的组合是：

> 在严格同秩、同字节的视频 native-state codec 中，用多域 feature bulk 与冻结
> reader-risk boundary atoms共同塑造 fast tier，再用 fast-tier 自身的 candidate
> margin 给出 exact finite refinement。

若 fresh selection 不能复现 risk-aware mix 相对 VSI/permuted null 的连续优势，并在
低 fallback 下保持行为，则这个组合也应停止。

## 7. 与 Wan 热力学审计是否一致

### 7.1 一致的核心

Wan 的 `EXP-003/004/005` 与本轮 OneVision 都支持：

\[
\boxed{
\text{统计冗余}
\ne
\text{功能冗余}
\ne
\text{低成本可部署冗余}
}
\]

- `EXP-004` 的 past-residual full-rank predictor 仅约 `1.001x`，说明增加函数容量
  不能补回缺失 current mode；
- `EXP-005` 的 current-input diagonal field 达到 `1.937x`，说明条件可观测性比固定
  结构更关键，但只在 3/10 层通过，尚无 suffix/速度结论；
- OneVision 的 pooled feature bulk 明显低维，但 reader-risk top-32 跨折 overlap 仅
  `0.179–0.212`；压缩态 margin 却能以极低状态量指出危险 tie。

三者都指向同一设计原则：**昂贵 exact operator 是测量/anchor；cheap tier 必须携带
当前条件；累计或边界不确定性高时精确刷新。**

### 7.2 不能混为一个数学证明的部分

[Score-SDE](https://arxiv.org/abs/2011.13456) 建立的是 forward/reverse diffusion 与
time-dependent score；漂移近似的路径 KL、Fisher/thermodynamic length 因而适用于
Wan 的去噪轨迹风险。随机热力学讨论的是定义良好的非平衡 ensemble 与轨迹 entropy
production，[Seifert 综述](https://arxiv.org/abs/1205.4176)不能直接赋予任意 Transformer
内部 feature 一个物理温度。2026 年关于 score matching 与 time-asymmetry entropy
production 的工作仍是新近理论结果，也不构成冻结 VLM codec 的证明
([preprint](https://arxiv.org/abs/2606.17252))。

OneVision 本轮是静态 frozen-reader 的 rate-distortion/decision-boundary 问题，没有
扩散时间、概率流或 suffix path measure。两条路线共享的是条件风险与 exact refresh，
不是同一个热力学定理。报告中不应把 `reader-risk trace` 称为 entropy production，
也不能用本轮结果反证或证明 Wan cache。

## 8. 是否能产生实际计算收益

当前只证明 state/storage/transfer，不证明 reader FLOPs 或 GPU latency。现 codec 会恢复
完整 `[3136,3584]` token state，token 数和后续 LLM 层都未减少。

但 orthogonal codec 对 OneVision 的第一层 RMSNorm + linear readout 有一个值得单独
验证的 exact folding。对单 token

\[
x=\mu+Uz,\qquad U^\top U=I,
\]

有

\[
\|x\|^2
=
\|z+U^\top\mu\|^2
+
\|(I-UU^\top)\mu\|^2.
\]

因此 RMSNorm 的 norm 可在 rank-456 坐标中精确计算；对第一层带 channel scale 的线性
读出，可离线折叠

\[
U^\top\operatorname{diag}(\gamma)W,
\qquad
\mu^\top\operatorname{diag}(\gamma)W.
\]

这可能省去一次完整 hidden-width reconstruction 与首个 projection，但它不会减少
token 数，也不会自动压缩后续 nonlinear Transformer。只有组件级数值等价测试通过后，
才能测 prefill/TTFT；当前不能把 `4.46x` state transfer 写成计算加速。

## 9. 下一步唯一建议 Gate

如果继续，固定以下候选后只运行一次现有 60-scene fresh selection：

1. pooled PCA-r456；
2. VSI PCA-r456；
3. hard risk-32；
4. boundary mix `g=32,w=0.3`；
5. same-weight permuted-risk mix；
6. progressive boundary mix，主阈值只使用语义明确的 `tau=0`。

selection 只在以下条件同时满足时进入 63-scene formal：

- risk-aware mix 相对 same-weight permuted null 保持明确的 paired KL/P95 改善；
- 不弱于 VSI PCA 的连续风险，而不是只胜 pooled baseline；
- progressive 路径 `harmful=0`、agreement 至少 98%；
- exact fallback 不高于 15%，保守传输仍至少 `4x`；
- 不使用 benchmark answer、exact teacher margin 或 selection gradient做路由。

若失败，应关闭 static/progressive fixed-basis quotient，转向 query-conditioned retrieval
或 native compact readout；不要继续调 atom count、risk weight、margin threshold 或
BCM basis。若通过，formal 仍只支持冻结 reader 的 state transfer，计算收益必须由后续
独立 kernel/TTFT gate 验证。

## 10. 可复现材料

- 聚合结果：
  [`cmrq_analysis/`](../analysis/onevision_reader_quotient_stage_a_20260830/cmrq_analysis/)
- 三域谱与 reader-risk 原始小表：
  [`onevision_reader_quotient_stage_a_20260830/`](../analysis/onevision_reader_quotient_stage_a_20260830/)
- 分析脚本：
  [`analyze_onevision_reader_quotient_cmrq.py`](../scripts/analyze_onevision_reader_quotient_cmrq.py)
- 绘图脚本：
  [`plot_onevision_reader_quotient_cmrq.py`](../scripts/plot_onevision_reader_quotient_cmrq.py)
- 图：
  [`onevision_reader_quotient_cmrq_stage_b.png`](../figures/onevision_reader_quotient_cmrq_stage_b.png)

本轮结论的最短表述是：

> 理论方向成立，但 universal low-rank quotient 不成立。真正可保留的核心是
> **stable energy bulk + rotating reader-risk boundary + observable exact fallback**；
> 当前只获得 calibration-level conditional GO，尚无正式泛化或速度结论。
