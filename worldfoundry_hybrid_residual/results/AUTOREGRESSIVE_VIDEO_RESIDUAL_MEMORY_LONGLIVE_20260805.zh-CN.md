# 自回归视频多分辨率残差内存：LongLive-1.3B 调研、实验与创新性审计

日期：2026-08-05
协议：`ar-video-residual-memory-longlive-v1`
模型：LongLive-1.3B v1.0，官方源码 commit `e52d9ef6865d843282a6b5e9d46d03b35f88929d`
硬件：NVIDIA H200；本轮 H200 用于官方模型 capture 与数值评估，不构成 kernel 或端到端加速声明。

## 1. 结论摘要

本轮把此前的“规则块/Butterfly + 精确稀疏残差 + 低秩尾部”思路迁移到帧级自回归视频模型，构造了以下训练免费内存近似：

1. 精确保留 frame sink 与 recent frames；
2. 将更早的因果 KV 历史按多分辨率时间组压缩为空间对齐代表；
3. 额外保留 0%、5% 或 10% 的 event residual tokens；
4. 比较 post-RoPE 直接汇总和 inverse-RoPE 后再按组中心重编码；
5. 对剩余 \(AV\) 缺陷测试 adaptive rank-4/8/16 上界和 calibration-frozen rank-4/8/16 basis；
6. 仅当表示门槛通过后，才允许继续开发 predictor、融合 kernel 和端到端 rollout。

预注册主候选的 held-out 结果为：

| 方法 | 算术 key reduction | rank-16 聚合 \(AV\) 误差 | 最坏 head | 判定 |
|---|---:|---:|---:|---|
| 自适应 oracle | 1.638x | 11.464% | 34.131% | FAIL |
| 冻结 calibration basis，oracle coefficients | 1.638x | 16.439% | 47.025% | FAIL |

因此，**固定多分辨率时间汇总 + 小 event residual + rank-16 输出修正不能作为 LongLive 的统一 attention/KV 主路径**。按预注册停止规则，本轮没有开发 predictor、融合 kernel，也没有声称实际 H200 加速。

这不是“低秩在自回归视频中无效”的结论。更准确的结论是：

- 固定训练免费汇总规则产生的缺陷，在关键中后层并不具有足够小、足够稳定的 rank-16 子空间；
- 自适应 rank-16 明显优于无修正，说明缺陷含有低维成分；
- adaptive 到 frozen 的迁移差距说明这些子空间随 prompt、seed、step、layer 和 head 旋转；
- 部分 layer-head 可通过严格局部门槛，异构认证与 dense fallback 仍有研究价值；
- 训练得到的低秩 KV bottleneck 仍可能有效，[VideoMLA](https://arxiv.org/abs/2605.30351) 正是直接反例。

## 2. 方法与理论对象

对当前 query \(q_i\)，将历史键值划分为精确集合 \(\mathcal E_i\)、压缩历史组 \(\{\mathcal G_m\}\) 和 event residual 集合 \(\mathcal R_i\)。精确分支计算真实 logits；每个历史组使用空间位置对齐的代表 \((\bar k_m,\bar v_m)\) 和组大小 \(|\mathcal G_m|\)。所有分支共享 softmax numerator/denominator：

\[
\hat y_i=
\frac{
\sum_{j\in\mathcal E_i\cup\mathcal R_i}e^{q_i^\top k_j}v_j
+\sum_m |\mathcal G_m|e^{q_i^\top\bar k_m}\bar v_m
}{
\sum_{j\in\mathcal E_i\cup\mathcal R_i}e^{q_i^\top k_j}
+\sum_m |\mathcal G_m|e^{q_i^\top\bar k_m}
}.
\]

这避免了“各分支分别 softmax 后再相加”的归一化错误。phase-aligned 版本先对历史 key 做 inverse 3D RoPE，再按代表帧中心重新编码；它只校正位置相位，不会自动对齐物体、遮挡或 value 内容。

定义真实输出缺陷：

\[
D=Y_{\rm dense}-\hat Y_{\rm memory}.
\]

adaptive oracle 对每个 capture/head 直接计算最优 rank-\(r\) 近似 \(D_r\)。frozen transfer 仅从 calibration 构造输出 basis，held-out 只允许求 oracle coefficients。后者仍不是可部署方法，但比 per-sample SVD 更严格地测试了子空间迁移性。

Butterfly 在本轮只表示**平衡的多尺度 merge schedule**，不是 softmax attention 算子的 Butterfly 因子乘积。这样避免把不可交换、带逐行归一化的 softmax 分支错误解释为线性 Butterfly 变换。

## 3. 相似工作与创新边界

相关方向在 2026 年已经非常拥挤，不能主张“首次分层 KV”“首次块稀疏 + 低秩”或“首次 head 异构”。

| 工作 | 已覆盖能力 | 与本轮关系 |
|---|---|---|
| [LongLive](https://arxiv.org/abs/2509.22622) / [官方仓库](https://github.com/NVlabs/LongLive) | frame sink、短窗口、KV recache；2.0 又加入 NVFP4 和 TriAttention | 本轮必须以官方 sink/recent 语义为基线，不能把它们算作创新 |
| [PackCache](https://arxiv.org/abs/2601.04359) | 训练免费 anchor、跨帧衰减、空间保持压缩；报告 1.7--2.2x 端到端 | 已覆盖“训练免费按时间压缩历史”的大方向 |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | anchor/compressed/recent 分层内存、relative RoPE、difference-aware decay | 与“分层时间内存 + 相位处理”高度重叠 |
| [Future Forcing](https://arxiv.org/abs/2605.30083) | 利用 canonical pre-RoPE query 稳定性，做未来感知评分与 affine merge | 说明 inverse-RoPE 可有用，但需要 query-aware 选择，而非固定组均值 |
| [Light Forcing](https://arxiv.org/abs/2602.04789) / [Sparse Forcing](https://arxiv.org/abs/2604.21221) | chunk-aware、层次 frame/block 稀疏，后者采用训练式持久/局部稀疏 | 已覆盖 AR 视频中的分层块稀疏主线 |
| [Forcing-KV](https://arxiv.org/abs/2605.09681) | static/dynamic head 异构压缩；H200 上报告 29+ FPS 和最高 1.35x LongLive 加速 | 已覆盖 head role 与混合 KV policy；本轮的 head 认证不能声称首次 |
| [Head Forcing](https://arxiv.org/abs/2605.14487) | local/anchor/memory head 和分层 episodic memory | 进一步覆盖 head-specific hierarchical memory |
| [VideoMLA](https://arxiv.org/abs/2605.30351) | 训练得到 shared latent KV 和 decoupled 3D-RoPE key；92.7% KV 内存下降、B200 1.23x | 证明“预训练谱不低秩”不妨碍训练后 latent bottleneck 成功 |
| [Pixelated Butterfly](https://openreview.net/forum?id=Nfl-iXa-y7R) | flat block Butterfly + low-rank 的通用硬件友好稀疏结构 | 覆盖一般 Butterfly + low-rank 组合 |
| [Dimension Mixer](https://proceedings.mlr.press/v280/sapkota25a.html) | 多级 group mixing 与 Butterfly attention | 覆盖一般层次跨块 mixing 思想 |

因此仍可能站得住的窄主线不是结构堆叠，而是：

> **Residual-certificate-driven heterogeneous AR memory：在真实 H200 成本约束下，选择 exact support 的目标不是只保留最大 attention mass，而是主动降低剩余 \(AV\) 缺陷的内在维数；随后用 calibration-only 证书决定每个 layer × step × head 使用压缩、训练式低秩尾部或 dense fallback。**

这一主张目前只完成了诊断基础，尚未形成成功算法。特别是，本轮 exact support 仍由固定 sink/recent/event 规则产生，还没有真正优化“残差宽度”。

## 4. 实验完整性与公平性

- 8 个 prompt/seed records：4 calibration、2 validation、2 test；prompt 类型覆盖 rigid、fluid、multi-object 和 non-rigid。
- 3 个 Transformer 层：0、14、29。
- 2 个生成起点：frame 15、18。
- 2 个实际 denoising calls：warped timestep 1000、833。
- 每个 capture 使用 4 个 64-query tiles、全部 12 个 heads。
- 共预期 96 captures，实际 96，缺失 0。
- evaluator 只读取 manifest 显式声明的 `.pt`，拒绝 stray/stale artifacts。
- 官方源码与两份权重均有固定 commit/hash；capture 目录拒绝覆盖和非空复用。
- calibration basis 只由 calibration 构造；validation/test 不参与 basis 或候选选择。
- 报告中的“速度候选”和“质量候选”均只按 calibration 选择，再固定到 held-out 评估。
- dense-reference parity：聚合 0.0982%，最坏 0.1928%，低于 0.5% capture 有效门槛。

这里的 `arithmetic_reduction` 只是代表 key/token 数量下降，不是实测 kernel speedup。adaptive/frozen correction 均读取 dense defect，因此即使通过也只允许进入 predictor 开发，不能直接声称部署或加速。

## 5. 全量结果

### 5.1 三个无泄漏候选

| 候选 | 选择依据 | 方法 | reduction | adaptive 聚合/最坏 | frozen 聚合/最坏 |
|---|---|---|---:|---:|---:|
| 预注册主候选 | protocol 固定 | phase-aligned, g1, event 5% | 1.638x | 11.464% / 34.131% | 16.439% / 47.025% |
| 速度候选 | calibration 中最优且 reduction >=1.5x | post-RoPE, g2, event 0% | 1.500x | 2.809% / 16.840% | 3.838% / 20.896% |
| 质量候选 | calibration 中聚合误差最小 | post-RoPE, g4, event 10% | 1.162x | 1.585% / 9.722% | 2.177% / 11.968% |

没有任何候选达到 adaptive `0.5%/1%`，更没有达到 frozen `1%/2%`。增加 summary groups 和 event payload 的确降低误差，但即使 arithmetic reduction 只剩 1.162x，最坏 head 仍接近 10%。这是明显的质量--成本前沿瓶颈，而不是 rank 选择的小问题。

### 5.2 主候选 rank 趋势

| 修正 | rank | held-out 聚合 | 最坏 head |
|---|---:|---:|---:|
| 无修正 | 0 | 21.143% | 58.375% |
| adaptive | 4 | 16.653% | 48.101% |
| adaptive | 8 | 14.417% | 42.147% |
| adaptive | 16 | 11.464% | 34.131% |
| frozen | 4 | 19.126% | 53.893% |
| frozen | 8 | 18.141% | 51.259% |
| frozen | 16 | 16.439% | 47.025% |

rank 增加持续有效，说明存在低维可修复成分；但 rank-16 离门槛仍有数量级差距，而且 frozen 改善远小于 adaptive，说明主要困难是内容条件化和子空间迁移，而不是简单增加 rank。

### 5.3 分层瓶颈

主候选 adaptive rank-16 在 held-out 上：

| Layer | 聚合误差 | 最坏 head |
|---:|---:|---:|
| 0 | 1.864% | 3.588% |
| 14 | 8.366% | 18.921% |
| 29 | 13.669% | 34.131% |

冻结 basis 后分别变为 2.637%/5.109%、12.118%/23.143%、19.577%/47.025%。固定时间汇总在中后层丢失的不是一个可跨样本共享的小输出子空间。

### 5.4 Head 异构性

使用“聚合 <=0.5% 且最坏 <=1%”认证每个 layer-head：

| 候选 | adaptive 通过 | frozen 通过 | Layer 14 |
|---|---:|---:|---:|
| 主候选 | 10/36 | 2/36 | 0/12 |
| 速度候选 | 16/36 | 9/36 | 0/12 |
| 质量候选 | 22/36 | 17/36 | 0/12 |

这说明统一策略失败，但异构策略有真实信号。Layer 0 和部分 Layer 29 heads 可压缩；Layer 14 在三种预算下均无 head 通过，应直接 dense fallback，或采用训练式 content-adaptive sparse/latent memory。由于 frozen 仍使用 oracle coefficients，这些通过数也只是 predictor 工作的上限，不是部署通过数。

### 5.5 3D RoPE phase alignment

在所有配对设置中，phase alignment 相对 post-RoPE 的 rank-16 held-out 聚合误差平均变化为 **-0.81%**（负值表示变差），中位数 -0.23%，最好改善仅 0.73%，最差变差 4.88%。

因此不能从“canonical pre-RoPE query 可能稳定”直接推出“固定历史组 key 在 inverse-RoPE 后均值更准确”。RoPE 只处理位置载波；内容相位、运动对应、遮挡、value leverage 和 softmax 非线性仍未解决。

## 6. 为什么当前块/低秩结构失败

### 6.1 在 softmax 前合并，不保留充分统计量

组代表希望用：

\[
|\mathcal G|e^{q^\top\bar k}\bar v
\approx
\sum_{j\in\mathcal G}e^{q^\top k_j}v_j.
\]

该等式只在组内 score 和 value 对当前 query 都近似齐次时成立。视频中的运动边界、遮挡、对象重现和 prompt 切换使组内异质性很强。均值同时破坏指数核的高阶矩和高杠杆 value，之后的固定低秩输出 basis 无法恢复 query-specific 丢失信息。

### 6.2 时间接近不等于内容可合并

balanced-recency/Butterfly merge tree只编码时间跨度。它没有回答“哪些空间 token 表示同一对象”“哪些历史帧会被未来 query 重新关注”。AR 视频需要内容对应和未来重要性，而不是仅按 XOR/树层级连接远距离块。

### 6.3 RoPE 对齐不等于物体对齐

inverse-RoPE 可移除确定性位置旋转，但不能移除 \(q,k,v\) 中的内容变化。将不同运动轨迹的 token 重编码到同一代表帧，仍会混合多物体、形变和遮挡。因此 phase alignment 没有形成稳定收益并不异常。

### 6.4 低秩修正的子空间随内容旋转

adaptive rank-16 明显优于 frozen rank-16，尤其在 Layer 29。形式上，若每个样本的最佳缺陷子空间为 \(U_{x,\ell,t,h}\)，固定 \(U_{\ell,t,h}\) 只有在 principal angles 足够小时才有效。本轮结果表明固定 basis 的误差重新上升，跨样本共享假设不成立。

### 6.5 中层承担动态内容整合

Layer 14 的所有 heads 都未通过局部门槛，说明其历史依赖不是简单局部/锚点二分。它更可能同时处理当前噪声状态、跨帧运动和语义绑定；固定时间 summaries 在这里产生重尾缺陷。

## 7. 有效性与创新性判断

### 已验证

- 官方 LongLive-1.3B 上的 capture/evaluator 链路完整，96/96 captures 可复现。
- fixed recency summaries、event residual、phase alignment、adaptive/frozen low-rank 的表示边界已被较严格测量。
- 当前预注册主候选全局失败，且失败不是单个 seed 或单个 head 造成。
- head/layer 异构性存在；固定统一压缩不是合理部署形式。

### 未验证

- 没有 predictor，oracle correction 不可部署。
- 没有 fused sparse/summary/low-rank kernel，算术 reduction 不等于 H200 speedup。
- 没有完整 rollout/VBench/长时一致性结果。
- 仅测 3/30 层、两个 frame starts 和两个 denoising calls，不能外推整个 LongLive 状态图。

### 创新性判断

当前实现本身不足以成为新方法：其结构部件分别已被 LongLive、PackCache、Echo-Forcing、Future Forcing、Forcing-KV、VideoMLA 和 Pixelated Butterfly 覆盖，且主候选数值失败。

仍有中等潜力的是**支持集--残差维数协同设计与认证运行时**。创新必须同时满足：

1. exact support 根据“移除后缺陷的 tail width”选择，而非只按 attention mass；
2. support 是内容/运动/未来 query 条件化的，但映射到少量规则 GPU tiles；
3. low-rank tail 由当前 Q/K/V 生成，或通过极低成本适配主动塑形；
4. calibration-only certificate 决定 head/cell 路径和 fallback；
5. 真实 H200 kernel cost 进入目标，并与已有 AR KV 方法正面对比。

## 8. 最合理的下一轮路线

### Stage A：扩大认证图谱

- 扩展到全部 30 层、更多 denoising calls、更多 frame starts 和长 rollout。
- 固定 calibration/validation/test，不根据 held-out 调方法。
- 加入 LongLive short-window、PackCache、Forcing-KV/Head-Forcing 风格 policy 作为可复现 baseline。
- 输出每个 cell 的 raw error、adaptive capacity、frozen transfer、router error 和成本。

### Stage B：只攻击 Layer 14 的函数类瓶颈

按最小训练成本依次测试：

1. frozen QKV，仅训练 stage gate/router；
2. content/value-aware exact tiles，使目标直接最小化 residual rank-16 tail energy；
3. 训练 summary weights、branch normalization 和 content-generated tail；
4. 若仍不够，加入 Q/K LoRA rank 4/8，让 attention mass 主动进入硬件规则块；
5. Layer 14 过不了 adaptive `0.5%/1%` 时立即停止，不再增加 Butterfly stage 或固定 BCM block。

Butterfly 可保留为候选 tile 的多尺度连接先验；它不应再定义完整 attention 输出。长跨度连接应由 query-aware/history-aware router 选择，而不是仅由 XOR stride 决定。

### Stage C：严格 kernel gate

只有 Stage B 达到以下门槛才进入 H200 kernel：

- adaptive capacity：聚合 <=0.5%，最坏 <=1%；
- frozen/router：聚合 <=1%，最坏 <=2%；
- router overhead < dense attention 的 5%；
- Whole-Attention 实测 >=1.5x；
- 再做多 prompt/seed 长 rollout、VBench-Long 和真实 wall-clock。

## 9. 最终判断

将此前方法直接迁移到 AR 视频的简单版本**没有达到有效性门槛，也不具备独立创新性**。这次实验的价值是排除了一个很容易被误判为有效的方案：更多时间组、更强 RoPE 对齐和 rank-16 oracle 虽能逐步降低局部误差，却无法同时满足高保真与 1.5x 算术预算。

研究不必回到“更大的固定 block/Butterfly”。更合理的延续是：利用本轮发现的 head/layer 异构性，把块结构降为硬件规则候选集，用 content-aware support 主动塑造剩余缺陷，再用小规模训练生成 tail 和 gate。若这一更强函数类在 Layer 14 的 adaptive oracle 仍失败，就应转向 LongLive 2.0 的低精度 dense、已有动态稀疏 KV 或训练式 VideoMLA，而不是继续增加结构复杂度。

## 10. Artifact 索引

- 协议：`configs/ar_video_residual_memory_longlive_v1.json`
- 方法核心：`scripts/ar_video_residual_memory_core.py`
- capture：`scripts/capture_longlive_causal_qkv.py`
- evaluator：`scripts/probe_ar_video_residual_memory.py`
- 通用绘图：`scripts/plot_ar_video_residual_memory.py`
- head 认证绘图：`figures/ar_video_head_certification_plot.py`
- gate：`results/20260805_full_v1/gate_decision.json`
- 全量 summary：`results/20260805_full_v1/summary.json`
- 全量逐 head metrics（确定性 gzip）：`results/20260805_full_v1/metrics.csv.gz`，SHA256 `60ef36bb9ed5f8dbafbb060b2b69abdf7d4d346db624e79f359b684a75a20326`
- 图和绑定 CSV：`figures/20260805_full_v1/`
- 远端原始 60 MB metrics：`/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/results/20260805_full_v1/metrics.csv`，SHA256 `a7350cfbede22c2c40e957aaa3a231732ccc3a9e7ffe8dd7a25e257dc3371d85`
- 远端 capture：`/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805/captures/`
