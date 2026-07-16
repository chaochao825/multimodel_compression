# 在线视频理解研究现状与本项目执行边界

证据截止日期：2026-07-16。本文只总结论文原文和当前仓库已验证结果，不把论文自报结果写成本项目结论。

## 一、当前研究已经做到什么

### 1. 固定预算记忆已经非常拥挤

Flash-VStream、VideoChat-Online、ReKV、LiveVLM、StreamKV、HERMES、DSCache、SAVEMem、MuKV、StateKV、SelectStream、CausalMem 和 FOLIO 已经覆盖了层次记忆、KV 压缩、语义分段、固定容量状态、在线语义基、实体记忆、检索缓存和优先级分配。因而“做一个固定大小的 memory bank”不能作为核心创新。

最新工作给出两个重要结论：

1. 记忆越来越按实体、事件和多尺度语义组织，而不是按原始 token 时间顺序保存。
2. SelectStream 强调过多旧历史可能稀释当前场景感知，因此 recent-window/instant-cache 不是弱基线，而是必须保留的强基线。

### 2. 空间持续状态也已有直接工作

Tensor Memory 已经使用固定大小的三维 recurrent tensor，通过连续空间写入、读取和门控更新保存视频状态。我们的工作不能声称首次提出“固定空间状态”或“空间 memory tensor”。

### 3. 事件触发和事件档案已经建立

Event-VStream、EventMemAgent 和 StreamForest 已经覆盖语义事件边界、事件触发更新、长期事件档案和局部时空窗口。我们的稀疏事件项只有在以下条件同时成立时才有贡献：

1. 残差能量确实在少量 block 上集中；
2. 因果 router 不依赖完整 dense attention；
3. router 加档案的总开销小于被避免的计算；
4. 它在 rare-event、OCR、数字、场景切换或事件关系切片上产生机制特异收益。

### 4. 系统瓶颈同时存在于视觉前端和 LLM 后端

StreamingTOM 和 STC 表明，重复视觉编码、视觉 token prefill、LLM KV 增长和查询检索都可能成为瓶颈。只报告 attention FLOPs 或 adapter 参数量不足以证明在线视频系统加速。

V-Rex 已经明确进入动态 KV 检索的算法—硬件协同，因此本项目不能声称首次进行在线视频算法—硬件协同。

### 5. BCCB、低秩和稀疏组合都不是新的数学原语

Circulant Attention、CirCNN、Toeplitz Neural Network、Structured Transforms、Low Displacement Rank、Kaleidoscope、Monarch Mixer 和 FlashFFTConv 已经覆盖块循环、Toeplitz、低位移秩、多结构组合和硬件友好 FFT kernel。在线 RPCA、DMD、OLSTEC 又覆盖了低维加稀疏的在线动态分解。

因此，本项目不能写：

> 我们首次提出块循环、低秩和稀疏的组合。

更稳妥的表述是：

> 我们以 probe-first 方式检验在线视频隐藏状态是否存在可分离的局部传输、固定容量持续状态和事件创新机制，并在统一内存与平均计算预算下进行因果归因和端到端系统测量。

## 二、已有本地证据真正说明什么

现有 Qwen3-VL 探针只测量每个 temporal slice 内二维空间 attention 的 BCCB 近似：

- 三个 Video-MME 视频上的平均 BCCB \(R^2\) 约为 `0.0738`、`0.0875` 和 `0.0791`；
- 对应平均相对 Frobenius 误差约为 `0.8632`、`0.8844` 和 `0.8945`。

这说明纯 BCCB 不是可靠的通用 attention 替换，但它不回答跨帧隐藏状态是否低维、是否可由局部传输预测，也不回答事件残差是否稀疏。

Wan2.2 的已有结果只支持“部分 head/layer/timestep 存在条件性的坐标对齐 cyclic 成分”，不能扩展为全局固定结构。

DRE-BCM 的 low-rank residual 属于参数空间权重残差，不能作为 temporal memory 低秩性的证据。

## 三、本项目仍然可能成立的研究缺口

当前最可辩护的缺口不是新 memory 结构，而是以下实验方法和系统合同的组合：

1. 在提交完整架构前，先分别证伪或验证三个 latent-dynamics 机制；
2. 对 identity、recent window、flow、BTTB/convolution、BCCB、在线子空间、语义记忆和事件记忆做公平对照；
3. 用路径删除、预算交换和事件反事实证明机制差异，而不是只看总准确率；
4. 固定总状态字节和平均每帧计算，不允许把 CPU/disk offload、检索索引和 archive metadata 排除；
5. 报告 P50/P95/P99、视觉编码、prefill、router、检索、archive 和 structured kernel 的分项时间。

## 四、三个 Probe 的修订版决策门

### Probe 1：时间状态低维性

必须同时报告未对齐、identity 对齐、低成本运动对齐和 region/object pooling。比较 truncated SVD、在线 PCA/subspace tracker、fixed window、reservoir、StateKV-like recurrent state 和 Tensor-Memory-like spatial state。

只有在至少两个模型或数据域上，\(r\le 32\) 达到预注册能量或延迟查询保真门槛，才能保留“低维持续状态”表述。

### Probe 2：事件创新稀疏性

必须在 identity、flow、convolution/BTTB、local BCCB 和 memory predictor 后分别计算残差，避免把未建模的相机运动误称为事件。需要同时报告 oracle top-k、motion heuristic、uncertainty heuristic 和因果学习 router。

### Probe 3：结构化空间传输

必须比较 identity/cache、flow warp、depthwise convolution、BTTB、global BCCB、masked/local BCCB、mixture-of-bases 和低秩映射。所有循环结构需处理边界 wraparound；FFT 版本必须实测而不是只报复杂度。

如果 local BCCB 在稳定片段上不能稳定优于 convolution/BTTB 或其 gate 使用率接近零，则从标题和主贡献中删除 BCCB，只保留为失败分析或可选 kernel。

## 五、最近的执行优先级

1. 用合成平移、缓慢漂移、遮挡、场景切换和稀有事件序列验证 Probe 代码的可识别性。
2. 在 210 上复用 Qwen3-VL-30B-A3B-FP8 视觉塔和三个现成 Video-MME 视频跑开发探针。
3. 用 210 上的 LLaVA-1.5-7B 视觉塔作为第二编码器，避免等待 34/35 上不完整的 Qwen2.5-VL 权重。
4. 扩展到 16–32 个分层 Video-MME 视频，并获取 StreamingBench/OVO-Bench 的许可、split 和数据路径。
5. 只有三个 Probe 通过对应门槛后，才实现完整三路径模型。

## 六、当前论文定位

建议暂定标题：

> Probe-First State Decomposition for Budgeted Online Video Understanding

在 BCCB 传输路径通过 Probe 3 前，不建议把 `circulant` 或 `BCCB` 放进主标题。若最终只有持续状态与事件创新通过，则应主动收缩为双路径工作。
