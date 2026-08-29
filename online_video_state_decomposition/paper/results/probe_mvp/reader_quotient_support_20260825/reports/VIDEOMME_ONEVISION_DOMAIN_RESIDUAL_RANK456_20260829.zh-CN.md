# OneVision 同秩领域残差：理论判决、选择结果与下一方法边界

日期：2026-08-29
协议：`videomme_onevision_domain_residual_rank456_20260829_v1`
选择判决：`CAPACITY_ONLY`
正式保留集：未运行，未授权

## 1. 直接结论

这轮实验部分达到理论预期，但没有达到方法门槛。

达到预期的部分是：Video-MME 中确实存在一套比 MVBench source PCA 更合适的
同维子空间。用 120 个平衡 calibration 视频重新拟合 target PCA-r456，在相同
`2,860,032` bytes 状态下，将 180 个 selection 视频的平均 KL、P95 KL 和 feature
relative L2 分别降到 source codec 的 `0.521x / 0.605x / 0.852x`。这证明此前
跨域退化并不是 rank-456 容量绝对不足，也不是 Video-MME 状态本身不可压缩。

没有达到预期的部分是：更低的连续失真没有变成更稳定的冻结 reader 行为。
target PCA 的 prediction mismatch 从 source 的 `6` 增至 `8`，只得到
`CAPACITY_ONLY`，不能进入 255 视频正式保留集。小规模 residual swap 同样没有
同时通过容量和离散行为门槛。因此当前可以支持的结论是：

> 共享低维 bulk 存在，但 feature-only PCA 找到的高方差方向不等于冻结 reader
> 的低风险 quotient；领域适配必须直接处理少量高 leverage 的决策边界方向。

## 2. 实验身份与无泄漏边界

- 模型为冻结 LLaVA-OneVision Qwen2-7B；模型参数未修改；
- source codec 是此前仅由 20 个 MVBench calibration 视频拟合的 PCA-r456；
- calibration 使用此前 600 视频跨域实验中的 120 个视频，只读取视觉 feature；
- selection 使用与 calibration 视频零重叠的 180 个已观察域视频；
- calibration 不读取问题、答案、selection feature 或 selection logits；
- 255 个此前从未执行过的 Video-MME 正式视频在 manifest 中冻结；
- 冻结协议规定只有 selection 为 `GO` 才可运行正式保留集，本轮为
  `CAPACITY_ONLY`，因此这些视频保持未见；
- 所有候选的 rank、latent shape、state bytes、reader 路径和 feature injection
  完全相同；180/180 样本完成，失败为 0，最大 injection 差异为 0。

这是一项 selection/capacity gate，不是新的正式 benchmark endpoint。不能把
selection accuracy 或 flip 数量当成独立泛化结论。

## 3. 同秩候选结果

| Candidate | KL ratio | P95 ratio | L2 ratio | Mismatch | Harmful | Correct | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| Source r456 | 1.000 | 1.000 | 1.000 | 6 | 1 | 102 | BASELINE |
| Target mean + source | 0.986 | 1.215 | 0.996 | 8 | 2 | 101 | NO_GO |
| Residual swap r16 | 0.862 | 0.996 | 0.977 | 5 | 0 | 103 | NO_GO |
| Residual swap r32 | 1.024 | 1.572 | 0.968 | 5 | 0 | 103 | NO_GO |
| Residual swap r64 | 0.898 | 1.413 | 0.954 | 8 | 2 | 102 | NO_GO |
| Residual swap r96 | 0.782 | 1.100 | 0.943 | 5 | 1 | 101 | NO_GO |
| Residual swap r128 | 0.840 | 1.109 | 0.934 | 4 | 0 | 102 | NO_GO |
| Target PCA r456 | **0.521** | **0.605** | **0.852** | 8 | 1 | 102 | CAPACITY_ONLY |

Target PCA 是唯一同时通过三项连续容量阈值和 safety 条件的候选，但没有将 mismatch
从 6 至少降到 4。Residual swap r128 正好达到 mismatch 门槛，且没有 harmful
或 accuracy 损失，却未通过 KL/P95/L2 容量门槛。两者形成了清楚的 Pareto 分裂。

![同秩领域适配的连续风险与离散 reader 行为](../figures/videomme_onevision_domain_residual_rank456.png)

## 4. Flip 集合揭示的机制

只读 flip 诊断没有改变任何选择规则：

| Candidate | Source overlap | 修复 Source mismatch | 新增 mismatch | KL 优于 Source 的样本 |
|---|---:|---:|---:|---:|
| Swap r16 | 5 | 1 | 0 | 89/180 |
| Swap r128 | 4 | 2 | 0 | 99/180 |
| Target PCA | 4 | 2 | 4 | 115/180 |

Target PCA 并非简单保留了原有错误。它修复了 source 的 2 个 mismatch，但在另外
4 个视频上产生了新 mismatch。Swap r128 也修复 2 个，却没有产生新 mismatch。
这说明完整 target PCA 在追逐目标域 feature 能量时，替换了少量跨域共享但具有高
reader leverage 的 source 方向；Swap r128 保留 source top-328 后更保守，所以
离散行为更稳，但未能充分降低连续失真尾部。

Target PCA 的 source-subspace overlap 只有 `0.7984`，Swap r128 为 `0.8628`。
这个观察支持“需要共享 bulk + reader-risk residual”的方向，但不支持把某个固定
swap rank 事后选为方法。各 swap 的 P95 KL 强烈非单调，说明 source eigenvalue
顺序和 target residual eigenvalue 顺序都不是 reader-risk 顺序。

## 5. 与理论是否一致

### 5.1 共享容量与领域残差

令领域 (d) 的视觉 feature 协方差为

\[
C_d=C_{shared}+\Delta_d.
\]

Source 与 target PCA-r456 的子空间 overlap 为 `0.7984`，说明存在较大的共享
bulk，也存在不可忽略的领域残差。Target PCA 将 calibration feature L2 从 source
的 `0.1921` 降至 `0.1594`，selection 上也从 `0.1946` 降至 `0.1658`，因此领域
残差不是 calibration 过拟合产生的纯假象。

### 5.2 PCA 风险与 reader 风险不等价

PCA 求解

\[
\min_{U^TU=I}\mathbb E\|X-XUU^T\|_F^2,
\]

但冻结 reader 对问题 (q) 的局部功能风险近似为

\[
D_q(X,\hat X)
\simeq
\operatorname{vec}(X-\hat X)^T
G_{\theta,q}
\operatorname{vec}(X-\hat X).
\]

只要 (C_d) 与 (G_{\theta,q}) 的主方向不交换，高 variance 方向就不必是高
reader leverage 方向。离散 top-1 还会在候选 margin 很小时放大很小的连续扰动。
这解释了为什么 target PCA 的 KL 均值下降 `47.86%`，prediction mismatch 却上升。

### 5.3 与此前生成和理解实验的统一

这与此前“条件冗余不等于固定共享算子”的结论一致：

- DiT 中固定 BCM/BCCB 或静态 low-rank residual 失败，是因为敏感坐标随内容、
  step 和条件旋转；
- 视频理解中的状态 codec 更容易获得正结果，因为它只需保留 reader 的功能商
  空间，不必重建完整物理状态或去噪轨迹；
- 但商空间仍随 reader、query 和领域改变，因此纯 source PCA 跨域只能部分迁移；
- 本轮 target PCA capacity positive 证明“仍有冗余”，离散 gate failure 则证明
  “低成本函数度量仍未编码进 codec”。

因此它不为 BCM/BCCB 主路径提供新证据，也不否定低秩。当前 gate 否定的是本轮
feature-only、静态、按方差排序的同秩领域替换候选，不能外推到所有 reader-aware
同秩适配。

## 6. 下一方法：多域 reader-risk quotient

下一轮只值得验证一个简洁假设：保持一套静态 rank-456 basis，但让 basis 直接最小化
多域 reader margin 的尾部风险，而不是 feature MSE。

令完整 reader 的候选 logits 为 (z(X,q))，参考 top-1 为 (y^*\)，候选 margin 为

\[
m_j=z_{y^*}-z_j,
\qquad
J_j=\nabla_X(z_{y^*}-z_j).
\]

对投影误差 (E_U=X-XUU^T)，一阶 margin 风险为

\[
r_{j}(U)=
\frac{\langle J_j,E_U\rangle^2}
{m_j^2+\epsilon}.
\]

建议求解固定字节的多域目标：

\[
\min_{U^TU=I,\;\dim U=456}
\max_{d\in\mathcal D}
\left[
\alpha\,\mathbb E_d\|E_U\|_F^2
+(1-\alpha)\,\operatorname{CVaR}_{0.95,d}
\left(\max_{j\ne y^*}r_j(U)\right)
\right].
\]

这里使用完整 reader 的预测而不是 benchmark label，所以目标是保持原模型行为，不是
利用 calibration label 提高 benchmark accuracy。CVaR 和领域最大值分别针对本轮的
重尾 mismatch 与 source/target 风险迁移。

### 6.1 最小实现

1. 使用多域 calibration，冻结模型、frame policy、rank 和状态字节；
2. 以 pooled multi-domain PCA-r456 初始化，不再单独使用 source 或 target PCA；
3. 只对 candidate-margin logits 做 calibration VJP，形成离线 reader-risk；
4. 在 Grassmann/Stiefel 约束下优化同一个 (3584\times456) basis；
5. 每步 QR retraction，禁止增加 sparse residual、query router 或动态 basis；
6. 部署时仍只执行静态投影，所有梯度成本仅发生在离线 calibration；
7. 与 Source、Swap-128、Target-PCA 和 pooled PCA 在相同 2,860,032 bytes 下比较。

若完整 Grassmann 优化过重，可先测试一个受限版本：保留 pooled PCA 的
`456-r_g` 个 bulk 方向，再从 margin-gradient-weighted residual covariance 中加入
`r_g in {32,64,96}` 个 risk atoms。这个候选比任意 swap 更符合当前机制，而且运行
时成本完全相同。

### 6.2 数据和 gate

当前 180 selection 已被观察，不能再作为正式端点；255 个旧协议 formal 视频因
`CAPACITY_ONLY` 未获授权，也不能改作新方法调参。新实验必须使用新的 benchmark
或不同 reader，重新冻结 calibration、selection 和 formal 三段。

选择门槛继续保持：相对 Source-r456，平均 KL `<=0.70x`、P95 KL `<=0.80x`、
feature L2 `<=0.90x`、mismatch 至少减少 2、harmful 不增加、correct 不下降、
各领域/时长无重尾退化、state bytes 完全相同。只有全部通过才进入正式集。

建议停止条件：

- pooled PCA 或 risk-atoms 若不能同时支配 Swap-128 与 Target-PCA，停止静态同秩适配；
- selection 只有连续误差改善仍记为 `CAPACITY_ONLY`，不得运行正式集；
- 正式跨域若 prediction agreement 仍低于 98%，停止声称通用 quotient；
- 即使质量通过，在没有 compact-state native reader 前也只主张状态/传输压缩，
  不主张 LLM prefill 或端到端加速。

## 7. 潜力与创新边界

潜力是有的，但边界需要清楚：

- 低秩 PCA、domain PCA、subspace alignment 和 task-aware compression 都是已有
  技术，不能作为创新点；
- 本轮只证明 reader-aware robust objective 值得一次新 gate，没有证明其有效；
- 单独的 margin-weighted PCA 仍偏增量；
- 更有论文价值的完整主线是“固定大小的多域 reader-risk quotient state + 有害
  事件证书 + compact-state native readout”，并证明它在在线视频多查询场景降低
  state/传输和实际 reader latency；
- 如果最终仍需 decode 回完整 `[16,196,3584]` token，本方法的直接收益主要是
  7.86x persistent payload，而不是 7.86x 计算加速。

相关边界包括：[Subspace Alignment](https://openaccess.thecvf.com/content_iccv_2013/html/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.html)
已覆盖经典跨域子空间对齐；[machine-oriented compression](https://arxiv.org/abs/2112.08168)
已覆盖按任务而非视觉 MSE 优化；[CausalMem](https://arxiv.org/abs/2606.25658)
已覆盖 training-free online semantic basis。因此新贡献必须来自原生 reader 风险、
跨域最坏风险、固定状态预算和端到端系统兑现的联合，而不是“PCA 加一个领域残差”。

## 8. 最终判决

本轮是一个可信的 `CAPACITY_ONLY`：

1. 理论中的共享低维 bulk 和领域残差得到支持；
2. feature-only 同秩适配没有通过 reader 行为门槛；
3. Swap-128 与 Target-PCA 的 Pareto 分裂揭示了可操作的新信息，即应保留共享
   高-leverage 方向，同时用 reader-risk atoms 修复领域 residual；
4. 下一步只授权设计一项多域 reader-risk quotient gate，不恢复 BCM/BCCB、动态
   Fisher support 或更多 PCA rank sweep；
5. 在新独立数据可用前，不运行旧 formal reserve，也不宣称泛化或加速。

## 9. 证据位置

- 冻结协议：`protocols/videomme_onevision_domain_residual_rank456_20260829.md`；
- 冻结角色 manifest：`configs/videomme/onevision_domain_residual_rank456_20260829.json`；
- 选择汇总与 flip 诊断：
  `analysis/videomme_onevision_domain_residual_rank456_selection/`；
- fitting 与运行身份：`metadata/videomme_onevision_domain_residual_rank456_v1/`；
- 可视化及原始绘图数据：
  `figures/videomme_onevision_domain_residual_rank456.{png,pdf,svg}` 与
  `figures/videomme_onevision_domain_residual_rank456_data.csv`。
