# 在线视频状态分解：研究命题、论文提纲与实验路线图

## 正式研究命题

本研究检验的不是“视频 attention 是否能被 BCCB 替换”，而是一个受预算约束的在线动力学假设：

\[
H_t=
\underbrace{\mathcal T_{\theta_t}(H_{t-1})}_{\text{结构化空间传输}}
+
\underbrace{U_tZ_{t-1}}_{\text{低维持续状态}}
+
\underbrace{S_t}_{\text{稀疏事件创新}},
\qquad
r\le R,\quad
\mathbb E|\Omega_t|\le K,\quad
\mathbb E C_t\le B .
\]

- 结构化空间路径负责平稳背景、局部纹理和近似平移，候选实现为 local BTTB、masked/window BCCB 或 mixture-of-BCCB。
- 固定容量低维状态负责对象身份、场景阶段和延迟查询所需的持续语义。
- 块稀疏事件路径负责 scene cut、新对象、OCR/数字和罕见异常，并保留少量高保真事件证据。

核心主张是：在相同状态字节和相同平均每帧计算预算下，三路径模型应比匹配预算的单路径、双路径和现有流式记忆方法形成更好的任务质量、稀有事件召回和尾延迟 Pareto 前沿。若任一路径没有独立贡献，论文必须删除该路径，而不是维持预设叙事。

## 证据边界与创新位置

现有本地结果已经支持“从纯结构替换转向动态混合”的问题重构：Qwen3-VL visual 的平均 BCCB \(R^2\) 约为 `0.0814`；零训练 BCCB 替换误差较高；Wan2.2 只在部分 head/layer/timestep 呈现几何对齐的 cyclic 成分；oracle `global/sink + local-cyclic + sparse` 拟合有效，但 support 跨样本迁移失败；SCTM 的高排名稀疏 route 具有任务级因果作用。

这些结果不能证明时间状态天然低秩，也不能把 DRE-BCM 的权重残差当作长期记忆证据。低秩加稀疏分解、固定容量 memory、事件 memory 和结构化 attention 均已有先例。预期创新应限定为：

1. 在架构承诺之前，用跨模型 probe 检验 latent dynamics 分解；
2. 在统一预算下，通过路径干预区分传输、持续状态和事件创新的机制职责；
3. 计入 router、archive、cache 和 kernel 开销，报告真实 P50/P95/P99 延迟与有界内存。

## 论文提纲

1. **Introduction**：在线因果处理、历史累积、稀有事件保留和尾延迟问题；提出可证伪的状态分解假设。
2. **Evidence and Reframing**：报告纯 BCCB 的负面边界、动态 route 证据及参数低秩与时间低秩的区别。
3. **Related Work**：结构化 attention、固定预算流式 memory、事件记忆、状态空间模型和 online RPCA。
4. **Method**：token 对齐、结构化传输、固定 slot memory、事件 archive、因果 router 与预算 gate。
5. **Probes**：时间低维性、事件创新稀疏性、前一状态到当前状态的结构化传输。
6. **System and Evaluation**：匹配预算主比较、路径消融、跨编码器复现、内存增长和尾延迟。
7. **Limitations**：失败 probe、结构路径适用范围、rare-event 丢失风险和生成侧迁移边界。

## 实验路线图与决策门

| 阶段 | 关键实验 | 进入下一阶段的证据 |
|---|---|---|
| A0 可控验证 | 合成平移、持续 latent、稀疏事件序列 | 能恢复已知 transport、rank 和 event support |
| A1 三个 probe | 对齐时间谱；预测残差集中度；\(H_t\approx\mathcal T(H_{t-1})\) | `P1/P2/P3` 至少两项通过，否则删除失败组件 |
| A2 最小系统 | 冻结视觉编码器和 LLM，在 projector 前加入三路径模块 | 一个在线基准三种子可复现，完整单/双/三路径消融 |
| A3 公平比较 | fixed window、reservoir、StateKV、CausalMem、事件 memory 方法 | 相同 state bytes 和平均计算下改善 Pareto 前沿 |
| A4 系统验证 | batched structured kernel、低维 router、全部缓存与 archive 计账 | 实测 P95/P99 获益，且总保留状态不随视频时长增长 |
| A5 扩展 | 第二视觉编码器；可选 Wan 生成迁移 | 主机制跨编码器成立；生成不作为首篇论文完成条件 |

预注册门槛均为 `planned`：`P1` 要求 \(r\le32\) 的对齐状态达到至少 `70%` 时间能量解释率，或在同状态字节下改善延迟查询；`P2` 要求 top `10%` blocks 捕获至少 `70%` 残差能量并保持至少 `80%` 关键事件召回；`P3` 要求 local BTTB/BCCB mixture 在平稳片段相对 identity/cache 降低至少 `10%` 激活预测误差，并产生真实 kernel 成本收益。

第一篇论文的推荐标题为 **Structured State Decomposition for Budgeted Online Video Understanding**。在线视频生成仅作为后续迁移方向，优先复用时间差分 kernel、head/timestep gate 和事件触发更新。

证据与执行索引：[现有结果边界](../ANALYSIS_REPORT.md)、[详细研究合同](RESEARCH_CONTRACT.md)、[贡献与证伪条件](paper/brief/contribution-map.yaml)、[证据矩阵](paper/brief/evidence-matrix.csv)、[实验矩阵](paper/notes/design/experiment-matrix.csv)。
