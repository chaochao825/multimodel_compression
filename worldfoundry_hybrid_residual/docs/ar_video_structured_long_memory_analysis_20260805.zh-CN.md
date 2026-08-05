# BCCB/BCM 与残差结构用于自回归长视频记忆的原理分析

日期：2026-08-05

状态：理论分析与下一轮候选定义。本文区分已验证结果、公开论文结果和待验证假设；不新增端到端质量或 H200 加速声明。

## 1. 直接结论

BCCB、BCM 和残差结构有机会以较小代价支持更长的视频上下文，但合理目标不是用一个固定结构近似完整 causal attention，而是构造异构、可回退的多层记忆：

1. recent window、条件 token、scene anchor 和高风险 event 保持精确；
2. 低风险历史 K/V 使用 K/V 分离的低比特残差编码；
3. 空间平稳或运动对齐后的背景使用因果结构化递归状态；
4. 全局身份和语义使用训练得到的低维 latent memory；
5. 当前 pre-RoPE query 只召回少量相关 scene、tile 和 residual refinement；
6. 不能通过 calibration certificate 的 layer/head 保持 FP8/BF16 recent-window 路径。

其中，BCCB/BCM 最可能成立的位置是**结构化递归状态或低成本候选 router**，而不是完整 attention map；最有价值的 residual 是**精确事件残差、分级量化残差和训练式语义 latent**，而不是跨样本固定的 post-hoc rank-16 输出修正。

## 2. 长视频记忆真正需要保留什么

对当前 query \(q\)，完整历史的 attention 输出为

\[
y(q)=\frac{N(q)}{Z(q)},\qquad
N(q)=\sum_i e^{q^\top k_i}v_i,\qquad
Z(q)=\sum_i e^{q^\top k_i}.
\]

一个固定大小的记忆若要对任意未来 query 精确恢复输出，就必须同时保存函数 \(N(q)\) 和 \(Z(q)\)。一般情况下，有限个均值或低阶矩不能唯一表示任意数量、任意位置的指数核原子。因此，固定大小的训练免费 summary 不可能对任意未来 query 都无损。小记忆必须至少依赖以下一种假设：

- 未来 query 分布受到限制且可预测；
- 历史 K/V 可被少量内容簇描述；
- attention importance 具有强 heavy-tail，可安全删除多数 token；
- 时空关系在对齐后近似平稳；
- 模型经过训练，主动把信息写入受限 latent state；
- 高风险样本可被证书识别并回退到精确路径。

[Future Forcing](https://arxiv.org/abs/2605.30083) 观察到 canonical pre-RoPE query 分布具有近似稳定性，为第一条假设提供了公开证据；[Quant VideoGen](https://arxiv.org/abs/2602.02958) 的内容自适应 centroid/residual 支持第二条；[PackCache](https://arxiv.org/abs/2601.04359) 和 [Forcing-KV](https://arxiv.org/abs/2605.09681) 分别利用时间衰减、anchor 与 head specialization；[VideoMLA](https://arxiv.org/abs/2605.30351) 则说明训练式 latent bottleneck 可以成功，即使原始预训练 attention 并不低秩。

## 3. BCCB/BCM 在不同轴上的含义

| 放置位置 | 原理判断 | 长记忆价值 |
|---|---|---|
| hidden-channel BCM | channel 没有天然周期邻接，固定 Fourier basis 容易错配 | 最多压缩 QKV/FFN 权重，不解决随视频长度增长的 KV cache |
| 完整时间 BCCB | 引入首尾周期连接，违反因果性、scene cut 和新旧不对称 | 不适合作为 causal memory 语义模型 |
| 原始空间 BCCB | 假设所有位置共享相对位移核，不能处理遮挡、物体独立运动和边界 | 只能覆盖少数 localized/stationary heads |
| 非周期空间 Toeplitz/shift | 保留平移结构但不 wrap-around | 比 BCCB 更适合作为局部候选生成器 |
| motion/semantic 对齐后的 BCCB | 先用 permutation/warp 对齐物体，再应用局部平稳核 | 有中等潜力，但 router 与对齐成本必须受限 |
| 结构化递归状态中的 BCM | 用循环/Toeplitz算子更新固定大小时空状态 | 是最值得验证的新位置 |
| QVG assignment/delta 编码 | 只压缩索引或小 metadata | 当前 QVG state 中 cluster ID 仅占约 2.6%，总收益很小 |

[Circulant Attention](https://arxiv.org/abs/2512.21542) 在视觉 Transformer 中报告了近似 BCCB 的 attention 模式和 \(O(N\log N)\) 算法，但这不能直接推出 causal AR 视频的全局 attention 也满足相同平稳性。其正面证据更适合支持“空间局部 expert 值得 probe”，而不是恢复固定全局 BCCB 主路径。

## 4. 为什么此前固定 BCCB/BCM 失败

### 4.1 固定 Fourier eigenvector 错配

任意 BCCB 矩阵都可写为

\[
C=F^H\operatorname{diag}(\lambda)F.
\]

若多个 BCCB expert 只做标量混合，

\[
\sum_m g_m(x)C_m
=F^H\operatorname{diag}\!\left(\sum_m g_m(x)\lambda_m\right)F,
\]

它们仍共享同一组 Fourier eigenvectors，只改变 eigenvalues。我们此前把每 head 参数从 2,184 增加到约 80,250，再到 530,070，平均输出误差仅从 57.20% 降到 50.41% 后平台化。这支持“basis mismatch”，而不是“容量不足”。

### 4.2 因果时间不是循环时间

AR memory 的时间结构是 lower-triangular：未来不能影响过去，recent、sink、旧 scene 和 prompt switch 的角色不同。把时间做成 circulant 会制造尾帧到首帧的伪连接。即使使用 FFT，也应把 causal Toeplitz 嵌入更大的 circulant 矩阵仅作为计算手段，不能把周期边界当成模型语义。

### 4.3 内容运动破坏空间平稳性

若所有像素执行统一平移，attention 可以接近 shift/circulant；真实视频包含多物体运动、形变、遮挡、镜头切换和重新出现。对应关系更接近内容相关的 sparse warp/permutation，通常稀疏但满秩。原始 raster 坐标上的单一 BCCB 无法表达这种非平稳关系。

### 4.4 生成动态 BCCB kernel 也可能不省计算

如果动态 kernel 必须先读取完整 QK 才能生成，就已经支付了 dense attention 的主要成本。可部署 BCCB 只能由 pooled Q/K、motion proxy、scene ID 或少量 anchor 生成，并映射到少量规则 tile；否则它只是 oracle 拟合器。

## 5. BCCB 更合理的形式：因果结构化递归记忆

对局部或背景 head，可先假设每个 temporal lag 的空间作用近似为非周期 Toeplitz/BCCB kernel：

\[
y_t\approx\sum_{\tau\ge0}\sum_{r=1}^{R}
c_r\lambda_r^\tau C(g_r)v_{t-\tau}.
\]

定义固定大小状态

\[
m_{r,t}=\lambda_r m_{r,t-1}+v_t,
\qquad
y_t\approx\sum_{r=1}^{R}c_r C(g_r)m_{r,t}.
\]

这样历史长度从 \(T\) 增长时，状态大小保持 \(O(RPd)\)，而不是 \(O(TPd)\)；空间卷积可用 FFT 或融合 shift kernel 计算。若 \(R\) 很小，这比每次扫描全部历史更符合“用很小代价维持长上下文”的目标。

但固定递归状态只能保存平滑、稳定的 modes。更现实的内容条件化形式是

\[
m_{r,t}=\lambda_r(x_t)W_{t\leftarrow t-1}m_{r,t-1}+B_r(x_t),
\]

其中 \(W\) 是受限 motion/semantic warp，\(\lambda_r\) 负责 scene-aware decay，当前 query gate 决定读取哪些 modes。该结构已经更接近受限状态空间模型，而不是 post-hoc BCCB attention，因此大概率需要低成本训练。

可将整体 causal operator 写为

\[
A_{t,s}\approx
P_t^T T_{t-s}P_s
+S_{t,s}
+L_{t,s},
\]

其中：

- \(P_t\)：内容条件化 permutation/warp，把同一对象或 scene 对齐；
- \(T_{t-s}\)：非周期 Toeplitz/BCCB 局部结构；
- \(S_{t,s}\)：精确 sparse event、遮挡边界和 scene recall；
- \(L_{t,s}\)：训练得到的低维全局语义/身份状态。

这三个分量分别处理几何平稳、稀有高熵事件和分布式语义，功能上比“BCCB + 一个固定 LoRA tail”更完备。

## 6. 此前残差方法的实际边界

| 方法 | 已验证结果或原理边界 | 是否值得保留 |
|---|---|---|
| 固定多分辨率时间 summary | 1.638x 算术 reduction 下，adaptive rank-16 为 11.464%/34.131% | 不作为统一主路径 |
| 更保守 summary + event 10% | 仅 1.162x reduction，仍为 1.585%/9.722% | 说明质量与成本前沿过陡 |
| calibration-frozen rank-16 | 主候选恶化至 16.439%/47.025% | 静态 post-hoc basis 停止 |
| per-sample adaptive low-rank | 明显优于无修正，但需要 dense defect，且子空间随内容旋转 | 只作为容量诊断 |
| 精确 sparse event residual | 可保护遮挡、scene cut、身份重现和高 leverage V | 高潜力，需未来 query-aware 选择 |
| progressive quantization residual | QVG INT2 显著优于 RTN，但当前严格 AV gate 仍失败 | 高内存潜力，不直接减少 attention FLOPs |
| K/V 同位宽量化 | INT2 calibration 中 K-only 7.77%，V-only 27.65% | 停止统一位宽，V 需要更高精度 |
| differential temporal coding | 相邻帧 residual 可能低熵，但 scene cut 会失效，长链随机访问昂贵 | 使用短 GOP、anchor 和 event reset 后再测 |
| cache-aware refresh | 能把高风险 token 恢复为精确表示 | 高潜力，与量化和 sparse residual 联合 |
| Butterfly 时间 merge | 仅编码时间跨度，不知道对象对应或未来重要性 | 只保留为多尺度候选拓扑 |

当前 36 个 layer-head 的认证结果也支持异构而非统一 memory：质量候选中 adaptive 通过 22/36，frozen 通过 17/36，但 Layer 14 为 0/12。风险排序在 calibration 与 held-out 间的 Pearson/Spearman 约为 0.99，说明静态 head certificate 可用很小开销缩小动态决策空间。

## 7. 最有潜力的多层记忆候选

候选可暂称为 **Certified Anchor-Track-Residual Memory**，但在实验通过前不作为正式方法命名。

### Tier 0：精确短期与语义锚点

- text/condition tokens；
- 少量 frame sink；
- recent window；
- prompt switch、scene cut、主体出现/重现与高 value-leverage tiles。

这些 token 使用 BF16/FP8，并作为稳定 fallback。

### Tier 1：异构低比特历史

- K 优先 INT4/FP8；
- V 优先 FP8/BF16；
- centroid、coarse residual、fine residual、exact correction 构成嵌套 bitstream；
- 当前 query 只为高风险 scene/tile 解码更深 residual。

该层继承 QVG 的内容自适应聚类，但率失真目标应从未加权 K-means 改为 attention/trajectory-weighted AV 缺陷。

### Tier 2：对齐后的结构化背景状态

- pooled Q/K 或低成本 motion proxy 选择 2--4 个 warp/shift expert；
- 时间使用 causal decay/Toeplitz state；
- 空间使用 nonperiodic Toeplitz，只有周期边界误差确实很小时才用 BCCB；
- 状态以低分辨率 spatial map 或少量 spectral modes 保存。

### Tier 3：低维全局语义状态

- 共享 KV down/up projection 或 latent memory；
- 只训练 write/read projection、gate、normalization 和可选 Q/K LoRA rank 4/8；
- 不要求预训练 K/V 本身 post-hoc 低秩。

### 读取过程

1. pre-RoPE query 选择 scene、head policy 和候选 tiles；
2. 读取精确 recent/event；
3. 对候选旧 token 渐进解码 residual；
4. 读取少量结构化背景 modes 和语义 latent；
5. 所有正值 attention 分支共享 numerator/denominator；
6. 风险证书不足时回退 FP8/BF16。

## 8. 成本模型

BF16 dense cache 的主要存储近似为

\[
M_{\rm dense}=2TPHd\cdot2\ \text{bytes},
\]

其中 2 对应 K/V，\(T\) 是历史帧数，\(P\) 是每帧 token 数。

若保留 \(W\) 帧精确历史，其余 token 中比例 \(e\) 精确、其余使用 \(q\)-bit 编码，忽略小 metadata 时旧历史的相对字节约为

\[
\rho_M=e+(1-e)\frac{q}{16}.
\]

例如 \(e=10\%\)：

- INT4：\(\rho_M=0.325\)，旧历史理论约 3.08x 压缩；
- INT2：\(\rho_M=0.2125\)，旧历史理论约 4.71x 压缩。

这些只是位宽上界，不包含 centroid、scale、索引和 alignment state。当前 QVG 实测逻辑压缩更适合作为真实基线。

若当前 query 只读取旧历史比例 \(\rho_T\)，attention 主体的长历史成本近似缩放为 \(\rho_T\)，但还需加 router、dequant、warp、latent 和 fallback 成本。QVG 若完整解压全部 K/V，\(\rho_T=1\)，只能节省容量/带宽，不能减少 QK/AV FLOPs。

结构化递归状态的意义是让历史扫描成本不再随 \(T\) 线性增长；但只有当 mode 数 \(R\)、warp 数和 fallback rate 都很小时，它才优于简单 compacted KV。

## 9. 最小且高信息量的验证顺序

1. 扩展到 48/96 帧真实 LongLive rollout capture，避免 12-frame window 低估长期收益和 scene recall 风险。
2. 固定相同 memory/compute budget，对比 raw-coordinate BCCB、nonperiodic Toeplitz、motion-aligned Toeplitz 与 unconstrained dynamic oracle。
3. 对每种 structured state 重新选择 5%/10%/20% exact event residual，不沿用旧 fixed recent mask。
4. 分别报告 K、V、normalization 和 scene-recall 缺陷；测试 K-INT4/FP8 与 V-FP8/BF16。
5. 先测试 structured recurrent oracle 是否能让 Layer 14 的 adaptive residual 达到 0.5%/1%；不能达到则停止 BCCB recurrent 路线。
6. oracle 通过后，只训练 write/read gate、warp、latent projection 和 branch normalization；主干 QKV 保持冻结。
7. 最后实现 fused packed-KV/sparse/structured attention，并测 H200 wall-clock。

建议 gate：

- calibration-only router 在 held-out 上聚合 AV 误差 <=1%、最坏 <=2%；
- 48+ frame identity/scene recall 不低于 dense/recent-window incumbent；
- old-history memory 至少 4x 压缩；
- 实际读取旧 token 比例 <=25%；
- router 与状态更新开销 < dense attention 的 5%；
- H200 Whole-Attention >=1.5x，端到端收益达到可测量水平。

## 10. 最终判断

固定 BCCB/BCM、固定时间 summary 和固定低秩 residual 不能以很小代价可靠维持任意长上下文，现有负结果已经足以停止这些简单形式。

仍有潜力的是：**把 BCCB 从 attention 近似器改成 motion/scene 对齐后的因果递归状态；把低秩从 post-hoc correction 改成训练式语义 latent；把 residual 从统一小扰动改成 value-aware exact event 与渐进低比特 payload。**

这种系统可以把常见平滑背景压入常数大小状态，把稀有但关键的信息留在有界 episodic memory，并用当前 query 触发精确 recall。它在原理上能获得随视频长度增长越来越明显的内存和计算收益，但必须通过长 rollout、Layer 14 hard-cell、future-query leakage guard 与 H200 fused-kernel 四个门槛后，才能形成可靠结论。

## 11. 本地证据

- `results/AUTOREGRESSIVE_VIDEO_RESIDUAL_MEMORY_LONGLIVE_20260805.zh-CN.md`
- `results/QUANTVIDEOGEN_AR_VIDEO_COMPARATIVE_AUDIT_20260805.zh-CN.md`
- `docs/block_butterfly_recovery_analysis_20260802.zh-CN.md`
- `results/20260805_full_v1/summary.json`
