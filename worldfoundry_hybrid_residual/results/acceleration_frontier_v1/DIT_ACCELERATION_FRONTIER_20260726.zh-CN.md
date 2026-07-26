# 视频 DiT 严格高保真加速前沿与 World Foundry 验证计划

日期：2026-07-26
对象：Wan2.1-T2V-1.3B / World Foundry / H200 NVL
约束：优先要求同模型、同 sampler、同 seed 的 paired SSIM >= 0.98；不能用仅分布指标接近替代同轨迹高保真。

## 1. 核心结论

当前最合理的优先级不是继续做全局低秩或 BCM，而是：

1. **双 H200 精确 CFG branch parallel**。条件分支和无条件分支在每一步读取同一 latent，彼此无依赖，可并行执行。这是当前唯一同时满足零模型近似、理论收益大、实现路径清晰的方向。
2. **F81 fused structured sparse attention**。F81 中 self-attention 占 53.88%，2x attention kernel 对 denoiser 的 Amdahl 上限约 1.369x，3x 时约 1.561x。必须使用空间/时间 head-aware 的静态可编译 block pattern、少量动态 refresh 和融合 kernel。
3. **F17 whole-block fusion / graph capture**。F17 self-attention 仅占 21.81%，elementwise/memory 占 47.76%。即使 attention 达到 4x，denoiser 上限也只有 1.196x，主要矛盾是 AdaLN、RoPE、残差、cast、launch 和 allocator 碎片。
4. **timestep-bucketed FP8/W8A8**。保留，但必须做 timestep/channel outlier smoothing、静态 scale 和融合 GEMM。当前动态 FP8 路径的 cast/scale/launch 已经吞掉理论收益。
5. **sample-adaptive cache/forecast**。可作为近似路线继续研究，但现有 Wan 20-step 严格 paired 证据尚未通过 SSIM 0.98，不能直接引用 FLUX/HunyuanVideo 的平均感知指标作为本模型结论。
6. **DiT speculative / parallel-time sampling**。在 deterministic UniPC、20 步、只有 2 张 H200 时不应作为主线。先测完整 30-block target 对 2/4 个候选状态的 batch verification 扩展率；若接近线性，传统 LLM 式推测执行没有硬件收益。attention/QKV microbenchmark 只能作前置筛查，不能代替完整模型结论。

全局 weight low-rank、静态 row-block sparse、BCM/CM 和 FFN 逐行/逐列 FFT 暂停作为主线。它们在当前模型上没有同时满足低误差和真实 H200 加速。

## 2. 三种“无损”不能混淆

| 层级 | 定义 | 适用方法 | 本项目要求 |
|---|---|---|---|
| 同轨迹精确 | 相同 seed 下 latent 递推与 dense 基线等价 | CFG parallel、精确 K/V cache、算子融合 | 最强，优先 |
| 目标分布精确 | 采样分布不变，但单个 seed 的轨迹可以变化 | 连续空间 speculative sampling | 可作系统吞吐研究，不能代替 paired SSIM |
| 感知质量近似 | VBench/FID/ImageReward 接近，逐像素轨迹不同 | sparse attention、cache、量化、少步 sampler | 必须额外通过 paired SSIM 0.98 gate |

因此，“speculative diffusion 生成 exact samples”通常表示目标分布精确，不表示与原 sampler 相同 seed 的视频逐帧相同。

## 3. 已有证据合并

### 3.1 运行时结构

| 场景 | 主要占比 | 直接含义 |
|---|---:|---|
| F17 elementwise/memory | 47.76% | 先做 whole-block fusion / graph，不应只改 GEMM |
| F17 self-attention | 21.81% | sparse attention 的端到端上限有限 |
| F81 self-attention | 53.88% | attention 是第一主线 |
| F81 elementwise | 27.99% | attention 后仍需 fusion |
| F81 linear | 14.29% | weight-only 压缩的 Amdahl 空间较小 |

由 4/6/8/12/20 步实测时延拟合：

\[
T(N)=0.605819+0.269639N\ \mathrm{s},\qquad R^2=0.998659.
\]

这说明 20 步路径中大部分时间仍随 denoising step 线性增长，但存在约 0.606 秒文本/VAE/初始化固定成本。若乐观地把全部 per-step slope 都视为两个等长 CFG branch，通信占单步成本 0%、2.5%、5%、10% 时，端到端 envelope 分别约 1.817x、1.745x、1.679x、1.562x。scheduler、Python 和其他串行 step 工作尚未拆出，因此这些数是待实测校准的乐观包络，不是速度预测或保证。

### 3.2 低秩、稀疏和量化

- eager rank-16 correction 实测约 0.46x，额外读写和 kernel launch 大于被替代的工作。
- F81 attention 的 top-128 sparse critical component + rank-16 tail，局部 output rel-L2 为 6.55%、cosine 为 0.9969；F17 为 4.91%、0.9982。它说明低秩只适合做已筛出关键高秩结构后的 marginal tail，不支持全局低秩替换。
- 全局 quantization/cache activation defect 的 rank-16 能量分别只有约 34.2%/76.1%，并且跨 layer、step、CFG branch 高度异质。
- 真实 FFN activation 上，FP8 output error 约 1.54%，INT4 约 7.53%，spectral-r16 + INT4 约 6.89%。rank-16 只挽回约 9.4% 的 INT4 误差，仍远离严格无损。
- Wan FFN 权重存在大量 MP outlier spikes，但 held-out activation error 并不随 spike 数简单下降。谱 spike 表示存在结构，不等于 bulk 可删除。

### 3.3 step 轴

现有粗减步已经否决：UniPC 20 步基线 6.045 秒；12 步 1.602x，但 paired SSIM 仅 0.3137；8/6/4 步也只有 0.2043/0.1737/0.2142。DPM++ 12/8 步同样只有约 0.304/0.185。

这不否决所有 step 加速，但否决了“直接减少现有 solver 的 NFE”在严格同轨迹约束下的可行性。后续只能研究：

- 更高阶、误差控制的 solver，但现有 20-step UniPC 已经很短；
- fixed-point/Picard 时间并行，用更多计算换 wall time；
- speculative target-distribution sampling；
- feature forecast + dense anchor + sample-adaptive fallback。

## 4. 冗余空间的几何分解

令一步 sampler 为：

\[
z_{t-1}=G_t\left(z_t,F_\theta(z_t,t,c)\right).
\]

DiT 冗余不是单一的“矩阵低秩”，而分布在五个不同坐标轴。

### 4.1 CFG branch 轴：精确冗余

\[
F_{\mathrm{cfg}}=F_u+s(F_c-F_u).
\]

给定相同 \(z_t,t\)，\(F_c\) 与 \(F_u\) 没有数据依赖。两张 GPU 各持有一份模型，分别计算两个 branch，再传输一个 noise prediction 并按原顺序做 CFG 和 scheduler，即可保持模型与 sampler 不变。它消除的是顺序执行冗余，不是数值近似。

### 4.2 token/head 轴：F81 最大的结构冗余

视频 attention 不是文本 attention 的一维局部性，而是 T x H x W 几何。合理分解是：

\[
A V=A_{\mathcal S}V+U_r C V+R V,
\]

其中 \(A_{\mathcal S}\) 是空间邻域、同位置跨帧和少量全局 anchor 构成的高秩稀疏主项；\(U_r C\) 只拟合遗漏的低秩 marginal tail；\(R\) 必须由误差 gate 控制。不能把整个 \(A\) 直接压成低秩，因为运动边界、局部纹理和跨帧对应关系本身可以是高秩的。

已有工作也指向这个机制：Sparse VideoGen 将 head 动态分为 spatial/temporal pattern，并依赖 layout transformation 和定制 kernel 才兑现 2.28x 左右端到端收益；HASTE 进一步做 head-wise budget 和 mask refresh；LVSA 使用 rotating global anchors 避免固定窗口的长视频偏置。

### 4.3 step 轴：平滑但累积敏感

相邻 step 的 feature 可以高度相关，但误差通过 sampler 递推累积。对近似动作 \(a\) 定义局部缺陷：

\[
d^a_{\ell,t}=F^a_{\ell,t}(h_{\ell,t})-F^D_{\ell,t}(h_{\ell,t}),
\]

真正相关的不是 \(\|d\|_2\)，而是传播风险：

\[
\rho^a_{\ell,t}=d^{a\top}_{\ell,t}J^\top_{(\ell,t)\rightarrow z_0}
J_{(\ell,t)\rightarrow z_0}d^a_{\ell,t}.
\]

这解释了为什么局部 feature cosine 很高仍可能破坏最终视频，也解释了固定 refresh schedule 在多 prompt、多 seed 下不稳定。

### 4.4 channel/weight 轴：需要功能加权

矩阵近似 \(\widehat W\) 的正确一阶指标是：

\[
\mathbb E\|x(W-\widehat W)^\top\|_2^2
=\mathrm{tr}\left((W-\widehat W)\Sigma_x(W-\widehat W)^\top\right),
\]

而不是 \(\|W-\widehat W\|_F^2\)。DiT 的 \(\Sigma_x\) 又随 timestep、AdaLN 条件、CFG branch 和 motion 改变，所以静态 SVD、静态 BCM、静态 per-tensor quant scale 很容易在真实激活上失效。

### 4.5 runtime 轴：数学省 FLOPs 不等于 H200 加速

当前主要低效包括：

- 条件/无条件 branch 顺序执行；
- AdaLN、norm、RoPE、bias、GELU、residual、cast 被拆成大量 pointwise kernel；
- 动态量化每次执行 scale、cast 和 Python dispatch；
- eager low-rank/sparse residual 额外读取 activation、写中间 tensor、再做 add；
- cross-attention text K/V 在每一步重复投影；
- Python 循环、allocator 和可能的 CPU sync 阻碍 CUDA Graph；
- 稀疏 pattern 若不转换成连续 block layout，索引和访存会吞掉 FLOP 节省。

## 5. 随机矩阵理论能做什么

### 5.1 可用判据

对 layer x timestep bucket 收集真实 activation defect：

\[
D_{\ell,b}=[d_1,d_2,\ldots,d_n]\in\mathbb R^{d\times n}.
\]

在适当中心化、按 activation covariance whitening 后，对 \(DD^\top/n\) 检验 spiked covariance。若纯噪声近似 i.i.d.、aspect ratio \(\gamma=d/n\)，Marchenko-Pastur 上边界为：

\[
\lambda_+=\sigma^2(1+\sqrt\gamma)^2.
\]

超过 bulk edge 且跨 prompt/seed 稳定的 eigenvectors，可作为需要保留或补偿的 coherent defect subspace；bulk 只能说明剩余误差高维、近随机，**不能直接说明它对输出无害**。

最终保留判据应是：

\[
\text{score}_i=(\lambda_i-\lambda_+)_{+}
\cdot \mathrm{stability}_i
\cdot \mathrm{trajectory\ sensitivity}_i
\,/\,\mathrm{H200\ cost}_i.
\]

只有 score 高且可以融合到主 kernel 的 spike 才值得作为 low-rank correction。

### 5.2 与已有研究的边界

LLM 中已有 Heavy-Tailed Self-Regularization 和 AlphaPruning 一类工作用经验谱密度决定层级 pruning/质量；它们提供诊断思路，但没有证明 DiT 的 MP bulk 可无损删除。HiCache 也使用 CLT/RMT 解释 feature finite difference 的近 Gaussian 性，并明确承认没有“神经特征必然 Gaussian”的完整第一性原理证明。

本项目应避免写成“RMT 证明 DiT 可低秩压缩”。更严谨的定位是：

> 使用 RMT 构造显著性检验，把稳定、传播敏感的 coherent defect spikes 与高维 bulk 分开；随后用 paired rollout 和实测 H200 kernel cost 决定是否补偿。

实现上使用两条阈值而不是把相关 token 行强行视为 i.i.d.：一条是 MP 理论边界，另一条是对每个 channel 独立 circular row-shift 得到的经验 null 最大特征值 95% 分位数，最终取二者较大值。跨 prompt/seed 的 top-r projection overlap 再检验子空间是否稳定。这样仍然只是保守的候选筛查，不能代替 trajectory-weighted rollout。

## 6. 为什么 LLM speculative decoding 不能直接迁移

LLM draft 出 \(k\) 个 token 后，target 可以在一次 teacher-forced causal forward 中同时验证这些位置，因为每个候选位置的 token prefix 已知，attention mask 保证因果关系。

扩散的未来状态满足：

\[
z_{t-j-1}=G_{t-j}(z_{t-j},F_\theta(z_{t-j},t-j)).
\]

如果第一个 draft state 被拒绝，后续所有 draft state 的输入都来自错误轨迹，不能继续作为原轨迹的合法 target 状态。对 deterministic sampler，target transition 和 draft transition 分别是 Dirac measure：

\[
K_t^D(z,\cdot)=\delta_{G_t^D(z)},\qquad
K_t^Q(z,\cdot)=\delta_{G_t^Q(z)}.
\]

其 maximal-coupling 接受概率为

\[
1-\operatorname{TV}(K_t^D,K_t^Q)
=\mathbf 1\{G_t^D(z)=G_t^Q(z)\}.
\]

也就是说，draft 只要不是数值完全相同，严格 target-law 接受率就是零。现有连续扩散 speculative 方法依赖随机转移、residual sampling 或 exchangeability；把 deterministic UniPC 改成 stochastic DDPM 可以研究分布精确，但会改变相同 seed 的基线轨迹，不能满足本项目最强的 paired fidelity 定义。虽然可以把多个 draft latent 堆成 batch 并行调用 target，但：

1. target 对长视频 DiT 主要是大 GEMM/attention，batch 2/4 很可能近线性增加成本；
2. 连续空间 acceptance/rejection 需要高效 residual distribution sampling，比离散 token 困难；
3. 目标分布精确不等于相同 seed 轨迹精确；
4. 只有 2 张 H200 时，设备已经可用于收益更确定的 CFG branch parallel。

连续 diffusion speculative sampling 已有论文证明可以保持目标分布，并报告约减半 target NFE；Auto-Speculative Decoding 则利用 stochastic DDPM 的 exchangeability 构造无额外 draft model 的渐近精确方法。它们回答的是目标分布而非 deterministic same-seed trajectory。2026 年的 block verification 工作仍指出连续 residual sampling 是核心难点，其 Free Drafter 相对已有 speculative 方法只再提升最多 6.3%。这说明它是有效研究方向，但不是当前 Wan 20-step、双卡、严格 paired fidelity 的第一选择。

### 6.1 两卡时间并行上限

对 \(T=20\)、window \(w\)、设备数 \(g\)、每窗 fixed-point 迭代 \(R\)，忽略通信的理想上限近似：

\[
S_{\mathrm{Picard}}\leq
\frac{T}{\lceil T/w\rceil\lceil w/g\rceil R}.
\]

当 \(g=2,w=20\) 时，\(S\leq 2/R\)。只要需要两轮迭代就不超过 1x；需要三轮则约 0.667x。ParaDiGMS/ParaTAA 在 100/1000 步和更多设备上很有意义，但 20 步、2 卡的并行深度太小。

因此本项目对 speculative 的第一项实验不是端到端实现，而是 H200 target batch verification probe：先测 batch 1/2/4 的 attention、QKV、FFN latency，再以相同初始 latent 和 timestep 测完整 30-block Wan denoiser。只有完整模型 batch-2 成本显著小于 2 倍、stochastic sampler 被明确接受、且离线 acceptance 足够高，才进入完整 sampler。

## 7. 从 LLM 压缩路线得到的正确迁移

LLM 压缩真正成功的路线不是只看权重谱，而是逐步转向：

1. activation-aware / Hessian-aware error；
2. outlier channel 单独保护；
3. group-wise scale 和静态可融合格式；
4. kernel-realizable structured sparsity；
5. serving-level KV cache、continuous batching、speculative decoding。

DiT 应做对应但不同的迁移：

- activation-aware 必须再加 timestep、CFG branch 和 motion bucket；
- outlier smoothing 必须围绕 AdaLN 后的 channel-specific、time-dependent outlier；
- LLM token sparsity不能直接复用，视频要利用 T x H x W 几何和 head specialization；
- DiT batch=1 但 token 很长，权重在一个 step 内复用充分，weight-only memory compression 的 latency 收益弱于 LLM autoregressive decode；
- DiT 没有跨请求增长的 autoregressive KV cache，主要冗余在跨 step feature、静态文本 K/V 和 CFG branch；
- 任何 residual path 必须融合，否则额外 HBM traffic 会比数学节省更贵。

## 8. 实验矩阵与停止条件

### E0：精确双卡 CFG

- F17：1-step smoke 后跑 20-step，交替执行顺序，至少 2 repeats。
- F81：F17 通过后再跑。
- 指标：E2E、denoiser、通信比例、peak memory、逐帧 SSIM/PSNR、exact pixel fraction。
- 数值等价先看 final latent max-abs、relative L2 和 exact fraction，再看有损编码视频；没有 latent 证据时不得声明同轨迹精确。
- Go：paired SSIM >= 0.999 且 F17 >= 1.45x；若 <1.30x，优先查通信/模型副本/同步，不进入更多近似实验。

### E1：speculative batch economics

- 对已捕获 F17/F81 QKV，测 batch 1/2/4 的 FA3 attention、QKV GEMM、FFN。
- 对完整 30-block Wan denoiser 在 F17/F81 测 batch 1/2/4；只有 `operation=full_wan_model` 的结果可进入 speculative 收益公式。
- 计算 verification ratio \(r_b=T_{batch=b}/T_{batch=1}\)。
- 用 acceptance \(p\) 的离线曲线计算收益边界，并计入 draft cost。
- No-go：deterministic same-seed 路线直接停止；若 stochastic target-law 路线的完整模型 \(r_2\geq1.8\)，在两卡 CFG 可用时停止完整 speculative sampler；若 \(r_2<1.4\)，再做 latent forecast acceptance probe。

### E2：F81 sparse-high-rank attention

- head x step x branch 记录 top-p support、spatial/temporal pattern、Jaccard drift、motion boundary。
- pattern：spatial window、same-position temporal stripe、rotating/global anchors、dense fallback heads。
- tail：仅对遗漏 output defect 做 rank 4/8/16，先做谱 gate；不得 eager 单独启动 GEMM。
- kernel：FlashInfer/Triton block-sparse，mask reuse，QK norm/RoPE/layout 融合。
- Go：局部 output rel-L2 <= 2%，paired SSIM >= 0.98，F81 attention >= 2x，E2E >= 1.30x。

### E3：RMT defect spike probe

- layer：0/6/12/18/24/29；step：0/5/10/15/19；CFG branch 分开；多 prompt/multi-seed。
- 比较 raw、centered、whitened defect ESD；bootstrap MP edge 和 eigenvector stability。
- 同时测 rank energy 与 trajectory-weighted energy，不只测 Frobenius energy。
- No-go：rank-16 trajectory-weighted energy <80% 或 subspace stability <0.8 时，不做 low-rank correction。

### E4：forecast/Hermite 仅作严格 gate 复验

- 同一 feature stream 比较 reuse、linear、Taylor-1/2、scaled Hermite-1/2。
- motion boundary 和 trajectory curvature 高时强制 dense anchor。
- 必须多 prompt、多 seed；报告 paired SSIM，而不是只报告 VBench/ImageReward。
- No-go：per-sample oracle 在 SSIM 0.98 下也 <1.2x，停止免训练严格高保真 cache 主线。

### E5：timestep-grouped quantization

- 对 AdaLN 后的 qkv、FFN-up、FFN-down 分别做 per-channel outlier profile。
- 分桶静态 scale，W8A8/FP8 优先，不再尝试全局 INT4。
- 重参数化 shift/scale 到相邻层，使用 fused GEMM，禁止 eager correction。
- Go：局部 output rel-L2 <=1%，paired SSIM >=0.98，kernel >=1.3x；否则只作为 sparse attention 内部低精度实现。

### E6：exact runtime cleanup

- text cross-attention K/V cache；
- torch.compile / CUDA Graph；
- 预分配 scheduler buffer；
- 移除 `.item()` 和隐式 CPU sync；
- fusion：AdaLN + projection 前处理，RoPE + QK norm，bias + GELU + residual。
- 所有结果与 eager BF16 按同 seed 配对。

## 9. 当前执行状态

已完成：

- 前沿静态分析与九面板 dashboard；
- NFE fixed/per-step cost 拟合；
- CFG parallel Amdahl/communication sweep；
- speculative acceptance/cost 与 Picard device/window 理论边界；
- RMT spike 与真实 activation rescue 的功能对照；
- H200 attention/QKV/FFN microbenchmark 与完整 30-block Wan target-batch benchmark；
- 真实 activation defect 的 MP/null/stability probe 及独立 dashboard 脚本；
- 双 H200 exact CFG generation、配对视频汇总和安全 runner。

待 H200 空闲后自动执行顺序：

1. F17/F81 attention/QKV/FFN microbenchmark；
2. F17/F81 完整 Wan target-batch benchmark；
3. 真实 quantization/cache defect RMT probe；
4. dual-H200 CFG 1-step smoke；
5. F17 20-step sequential vs CFG-parallel paired run；
6. 通过后扩展 F81 和更多 prompt/seed。

当前 H200 2/3 被其他独立训练作业持续占用，不能用竞争状态下的数据作为论文时延。runner 会要求连续多个轮询无 compute process 后再开始。

## 10. 最终路线判断

在“质量几乎不下降”约束下，Wan/World Foundry 的可利用空间按可信度排序为：

1. **精确并行和 exact system optimization**：CFG parallel、cross K/V cache、graph/fusion。
2. **长序列结构稀疏**：F81 head-aware spatial/temporal sparse attention + anchors + refresh。
3. **时变低精度**：timestep/channel grouped FP8/W8A8，且必须融合。
4. **轨迹预测**：sample-adaptive forecast，仅在 strict oracle 证明后保留。
5. **低秩**：只保留为 attention marginal tail 或显著 RMT defect spike correction。
6. **speculative/Picard**：多 GPU、多 step 情景有潜力；当前 2-H200、20-step 不是主线。
7. **BCM/FFT/静态 FFN 稀疏**：除非出现稳定几何频谱优势并有融合 kernel，否则停止。

这也解释了当前加速比“异常低”的根因：此前主要优化的是 F17 中占比不高的 GEMM/attention 子项，并额外引入了 unfused residual、动态 scale 和 kernel launch；数学 FLOP 减少没有落到 dominant kernel，也没有减少 HBM traffic。F81 attention 与双分支 CFG 才是能够形成明显端到端收益的主矛盾。

## 11. 主要文献

- [xDiT: SP、PipeFusion 与 CFG parallel](https://arxiv.org/abs/2411.01738)
- [PipeFusion](https://arxiv.org/abs/2405.14430)
- [Parallel Sampling of Diffusion Models / ParaDiGMS](https://arxiv.org/abs/2305.16317)
- [Accelerating Parallel Sampling / ParaTAA](https://arxiv.org/abs/2402.09970)
- [Speculative Sampling for Diffusion](https://arxiv.org/abs/2501.05370)
- [Speculative Sampling for Diffusion Models, ICML 2025](https://proceedings.mlr.press/v267/de-bortoli25a.html)
- [Auto-Speculative Decoding for Diffusion Models, ICML 2025](https://proceedings.mlr.press/v267/hu25d.html)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Speculative Diffusion Block Verification](https://arxiv.org/abs/2606.13426)
- [Sparse VideoGen](https://arxiv.org/abs/2502.01776)
- [SVG2: training-free sparse attention for video diffusion](https://arxiv.org/abs/2505.18875)
- [HASTE](https://arxiv.org/abs/2605.14513)
- [LVSA](https://arxiv.org/abs/2605.31057)
- [TaylorSeer](https://arxiv.org/abs/2503.06923)
- [HiCache](https://arxiv.org/abs/2508.16984)
- [Hierarchical Timestep Grouping PTQ](https://arxiv.org/abs/2503.06930)
- [AlphaPruning / RMT-guided layer sparsity](https://arxiv.org/abs/2410.10912)
- [Traditional and Heavy-Tailed Self-Regularization](https://arxiv.org/abs/1901.08276)
- [UniPC](https://arxiv.org/abs/2302.04867)
- [DPM-Solver](https://arxiv.org/abs/2206.00927)
