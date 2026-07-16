# 在线视频状态分解：研究命题与执行合同

面向立项讨论的压缩版本见
[ONE_PAGE_RESEARCH_BRIEF.zh-CN.md](ONE_PAGE_RESEARCH_BRIEF.zh-CN.md)。

## 核心命题

本项目不再假设视频 attention 本身可被一个 BCCB 矩阵替换，而检验一个更强、也更容易被证伪的在线动力学假设：

\[
H_t =
\underbrace{\mathcal{T}_{\theta_t}(H_{t-1})}_{\text{结构化空间传输}}
+
\underbrace{U_t Z_{t-1}}_{\text{低维持续状态}}
+
\underbrace{S_t}_{\text{稀疏事件创新}},
\]

其中 \(H_t\in\mathbb{R}^{N_t\times d}\)，\(Z_t\in\mathbb{R}^{r\times d}\)，\(r\ll N_{\text{history}}\)，且事件支持集
\(\Omega_t=\operatorname{supp}_{\text{block}}(S_t)\) 满足 \(|\Omega_t|\ll N_t\)。系统同时约束固定状态容量
\(r\le R\)、平均事件预算 \(\mathbb{E}|\Omega_t|\le K\) 和每帧计算预算 \(\mathbb{E}C_t\le B\)。

三项的职责必须通过干预实验区分，而不是只靠拟合误差解释：

| 路径 | 负责内容 | 首选实现 | 失败时的收缩方向 |
|---|---|---|---|
| 结构化空间传输 | 平稳背景、局部纹理、近似平移 | local BTTB、masked/window BCCB、mixture-of-BCCB | 若收益弱，从标题和主贡献中移除，只保留为可选低成本路径 |
| 低维持续状态 | 对象身份、场景阶段、长期语义 | 固定 \(16/32/64\) slot 的 recurrent latent memory | 若时间谱不衰减，改为字典式或分层 episodic memory |
| 稀疏事件创新 | 场景切换、新对象、OCR/数字、罕见异常 | 低维 block router + 精确稀疏读写 | 若残差能量不集中，取消稀疏假设，改为自适应密集更新 |

## 已有证据边界

当前仓库已经验证：

- Qwen3-VL visual 的平均 BCCB \(R^2\) 约为 `0.0814`，ViT/Qwen 的零训练 grid-BCCB 替换误差很高，纯 BCCB 不是可靠主假设。
- Wan2.2 的部分 layer/head/timestep 存在与真实三维坐标对齐的 cyclic 成分，但它不是全局、恒定的结构。
- oracle `sink/global + local-cyclic + sparse` 分解能显著降低代表矩阵误差，但跨样本 support/template transfer 失败，非局部路由必须动态预测。
- ViT/SCTM 中排名靠前的稀疏 route 具有任务级因果作用。
- DRE-BCM 的低秩项属于参数空间权重残差，不能作为时间低秩 memory 的证据。

权威来源分别为 [ANALYSIS_REPORT.md](../ANALYSIS_REPORT.md)、
[ATTENTION_PATTERN_MECHANISM_STUDY.md](../ATTENTION_PATTERN_MECHANISM_STUDY.md) 和
`E:/Codex_work/ssh_experiment/diff_bitnet/dre_bcm/results/report.md`。

## 三个先行 Probe

1. **时间状态低维性**：对齐连续帧 token 后，测量 effective rank、stable rank、前 \(r\) 维能量、在线子空间预测误差和下游 logit 保真度。必须同时报告未对齐、光流对齐和对象/区域汇聚结果，避免把 token 漂移误判成高秩。
2. **事件创新稀疏性**：用 identity、flow warp、BTTB/BCCB 和 memory predictor 得到 \(\hat H_t\)，分析 \(R_t=H_t-\hat H_t\) 的 block 能量集中度、top-k 事件召回、scene-cut/OCR/rare-event 对齐和 router 额外开销。
3. **结构化空间传输**：直接拟合 \(H_t\approx\mathcal{T}_{\theta_t}(H_{t-1})\)，比较 cache、flow warp、depthwise convolution/BTTB、global BCCB、local BCCB、mixture-of-BCCB、low-rank 和完整三项模型。

预注册的 MVP 进入条件：

- `P1`：在至少两个模型/数据域上，\(r\le32\) 的对齐状态达到 `>=70%` 时间能量解释率，或在同等状态字节下优于 reservoir/fixed-window 的延迟查询保真度。
- `P2`：top `10%` blocks 在至少两个真实数据域上捕获 `>=70%` 预测残差能量，并保持 `>=80%` 标注关键事件召回。
- `P3`：local BTTB/BCCB mixture 在平稳片段上相对 identity/cache 降低 `>=10%` 激活预测误差，且实际 kernel 成本低于 dense mixing。

这些阈值是 planned 决策门槛，不是已有结果。若仅两项通过，论文应按证据删去失败组件，而不是保留三路径叙事。

## 最小系统与论文贡献

第一版冻结视觉编码器和语言模型，仅在视觉塔输出与 multimodal projector 之间加入：

`local structured transport -> fixed-slot memory -> block-sparse event archive -> budgeted gated fusion`

拟定贡献为：

1. 提出并跨模型检验“结构化传输 + 低维状态 + 稀疏创新”的在线视频 latent-dynamics 假设。
2. 给出不依赖完整 dense attention 的因果三路径模型，并用路径删除、预算交换和事件反事实证明互补性。
3. 在固定 memory bytes 和固定平均计算预算下，报告准确率、rare-event recall、P50/P95/P99 延迟、峰值显存和 router 开销的 Pareto 前沿。

在线视频生成仅作为后续迁移：优先复用时间差分 kernel、head/timestep gate 和事件触发更新，不把它作为首篇论文的主任务。

## 执行顺序

`合成可控序列单元测试 -> 现有 Video-MME/Qwen 开发集 -> 第二视觉编码器复现 -> StreamingBench/OVBench 在线任务 -> 最小三路径系统 -> 真实 kernel 与尾延迟 -> 可选 Wan 生成迁移`

详细贡献边界、证据状态、基线和实验矩阵位于 `paper/brief/`、`paper/notes/design/` 和 `paper/plan/`。
