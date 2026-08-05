# 自回归视频中的分块、低秩与稀疏残差：文献和创新边界审计

日期：2026-08-05

## 结论先行

把此前方法直接搬成“全局 token Butterfly attention”不合理。Wan F81
实验已经表明，固定 XOR 拓扑、跨 stage 复用同一组 Q/K、逐 stage softmax
会产生严重函数族失配。在自回归视频中真正新增的结构不是“attention 更像
Butterfly”，而是**因果帧/块和持续增长的 KV 历史具有真实时间语义**。

因此第一轮候选被限定为：

> 保留 sink 与最近帧的精确 K/V；将较旧帧按空间位置对齐后，通过平衡二叉
> reduction schedule 形成多分辨率时间摘要；保留少量高残差事件 tile，并检验
> 剩余 attention-output defect 是否可由小 rank、跨样本可迁移的输出子空间修复。

这里 Butterfly 仅描述在线合并调度，不再是多个局部 softmax 算子的乘积。
所有分支共享同一个 softmax numerator/denominator；任何先分别归一化再混合的
实现都不属于该候选。

## 最接近的已有工作

| 工作 | 已覆盖内容 | 对当前候选的约束 |
|---|---|---|
| [CausVid](https://arxiv.org/abs/2412.07772) | 将双向 Wan 蒸馏成因果少步生成器并使用 KV cache | 提供 AR Wan 基线，不是压缩创新 |
| [LongLive](https://arxiv.org/abs/2509.22622) | 帧级 AR、短窗、frame sink、KV-recache | 当前真实载体；sink/window 必须作为 incumbent |
| [Light Forcing](https://arxiv.org/abs/2602.04789) | AR 视频中的层次化 frame/block 稀疏 attention | 不能主张首次层次化块稀疏 |
| [Sparse Forcing](https://arxiv.org/abs/2604.21221) | persistent block 与局部动态 block-sparse window | 不能把 persistent sparse history 作为主要新意 |
| [PackCache](https://arxiv.org/abs/2601.04359) | 免训练 anchor、时间衰减和紧凑 KV cache | 必须比较直接 token 选择/衰减策略 |
| [FAST-AR](https://arxiv.org/abs/2602.01801) | 时间对应关系合并 KV、ANN self/cross attention | 均值 K/V 加 log multiplicity 是直接 baseline |
| [FlowCache](https://arxiv.org/abs/2602.10825) | chunk-specific denoising cache 与 KV 压缩 | 不能把 chunk-aware cache 当作新意 |
| [Future Forcing](https://arxiv.org/abs/2605.30083) | 未来 query proxy 与 affine-subspace token merge | 是 query-aware merge 的最接近冲突 |
| [Head Forcing](https://arxiv.org/abs/2605.14487) | head 角色与 fast/episodic hierarchical memory | 不能主张首次 head-wise hierarchical memory |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | anchor/compressed/recent 三层记忆和 spatial recall frame | 空间对齐历史摘要本身已不新颖 |
| [Forcing-KV](https://arxiv.org/abs/2605.09681) | 静态/动态 head 的混合 KV 压缩及 H200 实测 | head 异构和真实 kernel 是必要 baseline |
| [Quant VideoGen](https://arxiv.org/abs/2602.02958) | 2-bit KV 和 progressive residual quantization | 低比特 residual 不能作为独立新意 |
| [TreeKV](https://www.ijcai.org/proceedings/2025/899) | LLM 中的树状平滑 KV 压缩 | 树结构本身没有新颖性 |
| [Hierarchical Self-Attention](https://arxiv.org/abs/2509.15448) | 多尺度层次 attention 与动态规划 | 不能主张首次多尺度/树 attention |
| [LongLive 2.0](https://github.com/NVlabs/LongLive) | NVFP4、融合 RoPE/AdaLN、量化 KV 与并行推理基础设施 | v1.0 probe 只验证表示；最终系统必须比较 2.0 的精度和 H200 吞吐 |
| [TriAttention](https://arxiv.org/abs/2604.04921) | 用 pre-RoPE Q/K 集中性和三角级数距离偏好选择 KV；已接入 LongLive | 仅使用位置频率或 Q/K 中心的粗路由不是独立创新 |
| [DySink](https://arxiv.org/abs/2605.21028) | 动态检索 frame sink 与 sink-collapse gate | 动态 anchor/sink 选择和 RoPE 异常检测已被覆盖 |
| [VideoMLA](https://arxiv.org/abs/2605.30351) | 训练式共享低秩 latent KV、解耦 3D-RoPE，报告 92.7% KV 内存下降 | 不能主张首次低秩 AR-video KV；必须区分训练形成的瓶颈与 post-hoc 谱近似 |

此外，Pixelated Butterfly 已覆盖一般 block-Butterfly + low-rank
结构；它不是视频 KV cache 方法，但排除了“首次 Butterfly + low-rank”的表述：
[Pixelated Butterfly](https://openreview.net/forum?id=Nfl-iXa-y7R)。

截至 2026-08 的检索意味着，`多尺度块缓存 + 动态 token 选择 + head
异构 + 低精度` 已是拥挤组合。当前候选只有在以下命题同时成立时才有独立性：

- 展开准则来自共享 softmax 分子/分母的可校准残差证书，而非时间距离、
  attention mass、Q/K norm 或相似度的普通排序；
- exact event support 的目标显式包含剩余 \(AV\) defect 的 rank-width，证明
  support 与 residual manifold 联合设计优于等预算质量贪心；
- 节点摘要可组合且 payload 有硬上界，validation/test 不允许后验更新 basis；
- 未通过证书的 cell 回退到 LongLive 2.0 FP8/NVFP4 或 BF16，并在 H200 上把
  router、展开、fallback 和数据移动全部计入 wall-clock。

若本轮 primary oracle 不通过，结果只能支持“该固定 summary/event 函数族不够”，
不能否定所有 AR 视频 KV 压缩；若 oracle 通过但 frozen basis 失败，则最合理的下一步
是小型 content-conditioned residual predictor，而不是增加静态 BCM/Butterfly 容量。
VideoMLA 还表明预训练 video attention 的 99%-energy rank 可以很高，但训练后的
latent bottleneck 仍可工作；因此 frozen-basis 负结果不能被写成“低秩对 AR 视频无效”。

## 仍可能成立的创新边界

普通“分层 memory”已经过于拥挤。只有以下联合命题仍有清晰差异：

1. **残差证书驱动的展开**：不是按时间、attention mass 或 novelty 单独保留
   token，而是根据摘要对 softmax 分子和分母的误差上界决定是否展开子节点。
2. **support-compressibility co-design**：稀疏事件 tile 的目标不仅是保留大
   attention，还要让剩余 AV 缺陷的 rank-8/16 谱更快衰减。
3. **可组合的低秩加稀疏节点残差**：父节点由子节点统计量合并得到，payload
   受固定 rank、tile density 和树宽约束；不允许保存任意后验系数。
4. **认证式异构回退**：不能通过门槛的 layer/head/chunk 保持 LongLive 原始
   sink + exact window，且 speedup 必须包含 router、展开和 fallback。

若没有第 1 至 3 点，候选会退化成 Head/Echo/Future Forcing 或通用 TreeKV 的
组合，创新性不足。

## 数学对象

对一个历史节点 \(S\)，不能只平均 K/V 后把输出相加。需要同时近似：

\[
Z_S(q)=\sum_{j\in S}\exp(q^\top k_j/\sqrt d),\qquad
N_S(q)=\sum_{j\in S}\exp(q^\top k_j/\sqrt d)v_j.
\]

最终输出必须是：

\[
\hat y(q)=\frac{N_{\rm exact}(q)+\sum_S\hat N_S(q)}
{Z_{\rm exact}(q)+\sum_S\hat Z_S(q)}.
\]

第一版使用空间位置对齐的组均值 \(\bar k_{S,p},\bar v_{S,p}\)，并为每个
代表 token 加 \(\log |S|\) multiplicity bias。这在组内 K 完全相同时精确，
也是 FAST-AR 合并引理的自然 baseline。随后保留 residual 最大的规则空间
tile，并测量剩余输出缺陷：

\[
D=Y_{\rm dense}-Y_{\rm summary+sparse}.
\]

只有 adaptive rank-16 先达到 `0.5%/1%`，才允许测试 calibration-frozen
输出 basis；只有 frozen basis 达到 `1%/2%`，才值得训练 Q/K/V-conditioned
系数预测器。这样能够区分表示失败、迁移失败和路由失败。

## 有效性风险

- LongLive 本身只有 `3` 帧 sink 加 `12` 帧滑窗，历史已被强截断；可压缩空间
  比全局 Wan 小，Amdahl 上限必须实测。
- post-RoPE K 的跨帧平均可能发生相位抵消。若数值上成立，部署仍需决定存储
  post-RoPE summary，或为 relative-RoPE 维护相位统计。
- adaptive low-rank correction 使用 dense defect，只是 oracle，不是部署方法。
- 不同物体运动会破坏同一空间位置的时间对应；稀疏 event tile 或 flow-aware
  对齐只有在 oracle 通过后才值得增加。
- 当前 LongLive v1 实现会 clone 完整临时 K/V cache；任何 kernel timing 必须
  先移除这项框架开销，否则无法归因于 attention 方法。

## 决策门槛

冻结协议见
[`../configs/ar_video_residual_memory_longlive_v1.json`](../configs/ar_video_residual_memory_longlive_v1.json)。
第一轮只回答一个问题：在 LongLive 的真实 causal Q/K/V 上，相同 cache 预算下，
多分辨率摘要加稀疏事件是否能把剩余 AV 缺陷塑造成小 rank、可迁移的结构。
oracle 失败即停止，不进入训练和 kernel；oracle 通过但 frozen basis 失败，则只
考虑小型内容条件化适配，不声称免训练成功。
