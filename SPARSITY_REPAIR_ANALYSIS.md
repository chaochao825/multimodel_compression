# 稀疏/剪枝的小参数保护与误差修补分析

## 结论

从参数利用率看，最有效的不是继续追加少量坐标，而是保存或利用低维的“充分统计量”：行质量、block 质量、共享 tail prior、以及 sparse path 相对 backbone 的 gain。

1. **Post-softmax top-k attention 的首选是质量守恒修补。** 对精确保留值，缺失质量可直接由 `1-sum(kept)` 得到，不需要额外参数。10% row-top-k 在强制 renormalization 后的平均 NRMSE 为 `0.4259`；加入零参数 uniform-tail 后降为 `0.03746`。
2. **一个全局 tail gate 就很有效。** 10% top-k 只增加 1 个 FP16 标量，payload 从 `14.400%` 增至 `14.411%`，NRMSE 降到 `0.06941`。它说明大部分失真来自保留质量被错误放大，而不是保留值本身不准。
3. **一个共享 column prior 比等 bit 的额外 nnz 更有效。** 增加约 `0.905%` dense bits 后，column-prior tail 达到 `0.02359`；把同样预算用于额外稀疏坐标仍为 `0.38384`。对于这些 attention map，pruned tail 更像“低维、扩散背景”，而不是少量遗漏尖峰。
4. **量化稀疏需要保存 mass/scale。** 10% row-top-k、4-bit shape code 只用 `6.980%` payload，但 NRMSE 为 `0.44362`。增加每行一个 FP16 kept-mass 后：
   - 不 renorm 的 raw path NRMSE 降到 `0.07292`；
   - 若把缺失质量放到 uniform tail，NRMSE 为 `0.04345`，payload `7.885%`；
   - 再加一个 FP16 column prior，NRMSE 为 `0.02959`，payload `8.790%`。
5. **结构化 block pruning 也适合 mass summary。** 10% block keep 的 baseline 为 `0.75157 @ 11.29%`；每个 pruned block 只存一个 FP16 mass 后为 `0.07451 @ 16.85%`。同等增量 bit 用于保留更多完整 block 仍为 `0.44165`。
6. **对“结构 backbone + sparse outlier”模型，最有效的是先保护 outlier，再拟合 tail。** 在统一约 25% dense-FP16 cap 下：
   - FP16 exact sparse：backbone-first `0.15135`，component-first `0.02475`；
   - q4 + query-block scale，component-first：`0.02300`（max scale）/ `0.02061`（loss-aware scale）；
   - q4 + per-row scale，component-first：`0.01289` / `0.00837`；
   - q4 query-block、4-stage sequential error feedback：`0.01100 @ 23.58%`。
7. **Scale 只能修幅度，不能恢复被剪方向。** 只缩放 retained support 会使 10% top-k 从 `0.4259` 变差到 `0.5577`；整行共同 scale 在 row normalization 后严格抵消。有效的 scale 必须控制 sparse branch 相对 backbone/tail branch 的质量，或者与显式 tail template 结合。

因此建议的方法优先级是：

1. attention probability：质量守恒 uniform tail；
2. 低 bit sparse probability：row mass + uniform/global-prior tail；
3. hybrid compression：component-first q4 sparse outlier + query-block scale + 2–4 阶 error feedback；
4. 极简部署：每 layer/head 一个校准 sparse gain；
5. 只有当 tail 本身稀疏时，才优先增加 nnz。

## 1. 为什么 scale 能保护量化，却不能直接修好剪枝

量化主要产生幅度/径向误差：方向和 support 大致还在，只要给一组低 bit codes 配一个 scale，就能恢复动态范围。

剪枝同时产生两类误差：

\[
e=e_{\text{amplitude}}+e_{\text{support}}.
\]

- `e_amplitude`：保留值整体偏大/偏小，scale 可以修；
- `e_support`：被删除坐标方向完全不存在，scale 无法恢复。

对于 attention probability，还会多一项 normalization error。设保留 support 为 `Ω_i`：

\[
m_i=\sum_{j\in\Omega_i}A_{ij},\qquad
p_{ij}=\frac{A_{ij}}{m_i},\quad j\in\Omega_i.
\]

直接把 `p_i` 当 sparse attention 相当于把保留质量从 `m_i` 错误放大到 1。一个更合理的 codec 是：

\[
\hat A_i=m_i\hat p_i+(1-m_i)t_i,
\]

其中 `t_i` 是 pruned support 上的 uniform 或共享 column-prior template。

- 精确 post-softmax kept values：`m_i=sum(decoded kept)`，无需额外存储；
- 只有 q-bit shape codes：存一个 FP16 `m_i`；
- 共享 column prior：全矩阵再存 `n` 个 FP16 值。

对应的 ideal-packed payload 为：

\[
B_{\text{sparse-shape}}=nk(q+\ell),\qquad
B_{\text{mass}}=B_{\text{sparse-shape}}+16n,
\]

\[
B_{\text{mass+prior}}=B_{\text{sparse-shape}}+32n,
\]

其中 `q` 为 code bits，`ℓ=ceil(log2 n)` 为列索引位宽。

这就是剪枝中最接近 quantization scale 的机制：**用极少数幅值/质量状态保护大量低 bit 或被删除坐标的总体贡献。**

## 2. Row-top-k 参数效率

以下均值来自 8 张 hand-picked attention map、4 个 source input。

| 10% row-top-k 方法 | Mean NRMSE | Payload / dense FP16 | 增量 payload |
|---|---:|---:|---:|
| Top-k + renorm | 0.42592 | 14.400% | — |
| + 1 global uniform gate | 0.06941 | 14.411% | 0.011% |
| + query-block gates | 0.04985 | 14.627% | 0.226% |
| + derived mass / uniform tail | 0.03746 | 14.400% | 0 |
| + one column prior | **0.02359** | 15.305% | 0.905% |
| + equal-prior-bit extra nnz | 0.38384 | 15.296% | 0.896% |
| Scale retained column groups only | 0.55767 | 14.627% | 0.226% |
| + rank-1 tail | 0.02242 | 16.221% | 1.820% |

为什么 column prior 优于追加坐标：一个长度 `n` 的 prior 同时给每一行的全部 pruned support 提供方向；同等 bit 的 COO extra-nnz 只能修几十个单点。当前 tail 是扩散型 background/sink prior 时，低维模板的复用次数更高。

`rank-1 tail` 数值也较好，但 signed factor 经 ReLU 后通常不再保持严格 rank-1 执行，不能只按 factor bits 声称加速。

## 3. 低 bit sparse shape 的 mass protection

固定 10% row-top-k：

| Codec | NRMSE | Raw NRMSE | Payload |
|---|---:|---:|---:|
| q2 shape only | 0.81923 | 0.81923 | 5.744% |
| q2 + row mass only | 0.81923 | 0.18576 | 6.649% |
| q2 + mass/uniform tail | 0.15630 | 0.15630 | 6.649% |
| q3 shape only | 0.50486 | 0.50486 | 6.362% |
| q3 + mass/uniform tail | 0.06253 | 0.06253 | 7.267% |
| q4 shape only | 0.44362 | 0.44362 | 6.980% |
| q4 + row mass only | 0.44362 | 0.07292 | 7.885% |
| q4 + mass/uniform tail | 0.04345 | 0.04345 | 7.885% |
| q4 + mass/column-prior tail | **0.02959** | 0.02959 | 8.790% |

“row mass only”的 normalized NRMSE 不变，是一个必要负对照：若随后再次 row-normalize，共同 mass 被抵消。但在不 renorm 的 post-softmax `A@V` raw path 上，row mass 恢复了输出幅度。

因此实现必须先决定执行语义：

- strict stochastic attention：需要把 `(1-m_i)` 分配到 tail template；
- raw sparse `A@V`：可以只保存 `m_i`，但输出不再严格 row-stochastic；
- sparse branch + dense/structured branch：scale 控制两条 branch 的相对贡献，不会被共同 normalization 抵消。

## 4. Block pruning 的小参数修补

设 block 大小为 `b=4`。每个 pruned block：

- block-mass：存 1 个 FP16 scalar，重构为 block 内 uniform；
- row-mass：存 `b` 个 FP16 scalar，每一行只在 block columns 内 uniform。

| 10% block keep | Mean NRMSE | Payload |
|---|---:|---:|
| Kept blocks only | 0.75157 | 11.29% |
| + one mass / pruned block | **0.07451** | 16.85% |
| + b row masses / pruned block | 0.05371 | 33.52% |
| + equal block-mass bits as full blocks | 0.44165 | 16.74% |

Block-mass 的参数效率明显优于多保留几个完整 block；row-mass 更准但参数增加过快。更合理的硬件折中是 block-mass 或少量 query-row-group mass，而不是给每个 pruned block 恢复完整 dense 值。

## 5. Scale-protected sparse outlier repair

使用 `b=4` block-scale rank-1 backbone：

\[
\hat A=\operatorname{RowNorm}\left(B_{\text{structured}}+
g_G Q_q(P_\Omega R)\right).
\]

名义 payload：

\[
B=16(P_{blocks}+b)+nk(q+\ell)+16G,
\]

其中 `G=1/n/b/n` 分别对应 global/query-block/per-row scale。

### 同 25% cap

| Codec | Backbone→sparse | Sparse→backbone | Mean k |
|---|---:|---:|---:|
| FP16 exact | 0.15135 | 0.02475 | 18.0 |
| FP16 + loss-aware folded global gain | 0.06545 | 0.02361 | 18.0 |
| q4 global max scale | 0.16901 | 0.04018 | 37.1 |
| q4 global loss-aware | 0.10133 | 0.03973 | 37.1 |
| q4 query-block max scale | 0.14874 | 0.02300 | 36.8 |
| q4 query-block loss-aware | 0.06293 | 0.02061 | 36.8 |
| q4 per-row max scale | 0.14221 | 0.01289 | 35.8 |
| q4 per-row loss-aware | **0.04852** | **0.00837** | 35.8 |
| q4 query-block, T=4 sequential error feedback | — | 0.01100 | 8.4/stage |

低 bit 的价值不只是在同一个 `k` 上逼近 FP16，而是把每个 sparse value 从 16 bit 降到 4 bit 后，在同总预算里把平均 `k` 从 18 提高到约 36。对于当前 outlier-heavy residual，多覆盖一倍 support 比保留 FP16 精度更有效。

`per-row` 最低误差，但动态 scale 数量为 `n`；`query-block` 只需 `n/b` 个 scale。4-stage query-block sequential error feedback 在 `23.58%` payload 下达到 `0.01100`，是更平衡的工程候选。

Loss-aware scale 与 max-based quant scale 使用相同 bits。它把 scale/gain 直接针对最终 row-normalized loss 优化，而不是仅最小化 sparse-value MSE。代价是 raw NRMSE 可能很差：25% cap 的 backbone-first q4 per-row loss-aware raw NRMSE 为 `1.47`。因此它只适用于确定会执行相同 normalization 的路径，不能直接当作幅度忠实的 raw `A@V` codec。

实现中 gain 会真正折叠进原有 FP16 value 或 group scale，并在乘法后重新 round 到 FP16，因此没有借用未计费的 float64 精度。多阶段版本采用顺序坐标更新，且候选网格包含 gain `1`；所以每一阶段的 scale 更新在相同 payload 下都不会劣于更新前的 row-normalized loss。

这里的 loss-aware 结果是 `gain in [0, 3]`、步长 `0.05` 的**有界网格候选**，不是无约束最优。954 条 loss-aware 结果中有 276 条的 `gain_max` 触及上界；25% cap、component-first q4 per-row 的 8 张图中有 4 张至少一个 row 触及上界。这不会制造虚假提升，但意味着当前网格可能低估继续增大 gain 的潜在收益，也提示实际部署需要正则化或扩大后重新校准范围。

### 一个标量的极简版本

Backbone-first FP16 sparse 在 25% cap 下为 `0.15135`。逐 target 选一个 global gain 可降至 `0.06545`；只对 gain 做 4-source-input leave-one-source-out calibration，平均 gain `2.45`，NRMSE 为 `0.08516`。

这说明每 layer/head 一个固定 gain 可能有价值，但当前 sparse support 仍由目标 `A` 选择。LOSO 只验证 gain transfer，没有验证 router/support transfer。

## 6. 哪些机制有效、哪些无效

### 有效

- **守恒统计量**：row mass、block mass；
- **跨大量 pruned 坐标复用的模板**：uniform tail、column prior、低秩 background；
- **outlier-first**：先保护高价值方向，再用 cheap backbone 拟合 tail；
- **相对 branch gain**：控制 sparse 与 backbone 的相对幅度；
- **loss-aware scale**：在执行目标确定时直接优化最终 loss；
- **multi-stage error feedback**：用多次小 sparse update 补不同残差方向。

### 无效或风险高

- **整行共同 scale + row normalization**：数学上为 no-op；当前 FP16/浮点实现的最大 loss 差小于 `6.4e-8`；
- **只缩放 retained support**：无法恢复 support error，实验中反而恶化；
- **独立 kernel 后再加自由 scale**：可吸收到 kernel，是冗余参数；
- **错误 support + 更强 scale**：scale 不能创造被删方向；已有 transfer probe 的 sparse-route Jaccard 约 `0.009`，router 仍是主要瓶颈；
- **只看 parameter bits**：implicit tail、ReLU low-rank、multi-stage sparse 的 decoder/MAC/latency 不同，必须另做系统 Pareto。

## 7. 与静态模型权重剪枝的关系

本轮没有持久化模型权重和 calibration activations，因此没有直接验证 weight pruning。可迁移的设计原则是：

- 对 output channel/block 保存 FP16 norm、mean、second moment 或 activation-aware gain；
- 用保留权重的 low-bit shape + channel/block scale 重构幅度；
- 用少量 outlier FP16、低秩 residual 或 error-feedback block 修方向；
- 使用 input covariance/Hessian-aware compensation，而不是只按 weight magnitude；
- scale 要作用于 pruned branch 与 residual/backbone 的相对贡献，不能放在会被 normalization 抵消的位置。

仓库已有 task-level SCTM 证据也支持 importance-aware pruning：drop strongest selected route 的 loss 约 `+0.214`，drop weakest route 仅约 `+0.001`。这说明优先保护重要 outlier route 比均匀增加精度更值得。

## 8. 适用边界与下一步

当前结果仍有以下边界：

- 8 张 hand-picked attention map，只来自 4 个 source input；
- support、mass、prior、per-target scale 都直接看目标 `A`；
- 没有新方法对应的 `V`、logits、CE/NLL、accuracy、PPL 或 denoising loss；
- bit 为 ideal-packed dynamic representation estimate；未计 router、top-k selection、decoder、MAC、访存和 latency；
- loss-aware scale 的好结果依赖 row normalization，raw path 不一定忠实；
- loss-aware gain 搜索限制在 `[0,3]`，部分 group 触及边界；当前值是有界候选而非无约束最优。

下一轮应优先：

1. 在 calibration split 学固定 layer/head gain、column prior；test 冻结；
2. 训练不生成 dense `A` 的 support/scale router；
3. 保存 `V` 和逐样本 logits，测 `A@V`、task loss 与 output drift；
4. 对 q4 query-block T=2/4 做 fused kernel 和实测 latency；
5. 用真实 weight + activation calibration 对 channel/block mass protection 做静态 pruning 实验；
6. 同时报 serialized bytes、index overhead、nonzero/MAC density 和 latency。

## 9. 复现

```powershell
python scripts/sparsity_repair_probe.py
python figures/sparsity_repair_efficiency_plot.py
python -m pytest -q tests/test_sparsity_repair_probe.py
```

产物：

- `remote_logs/sparsity_repair_probe_20260712.json/csv`
- `remote_logs/sparsity_repair_pareto_20260712.csv`
- `figures/fig24_sparsity_repair_parameter_efficiency.pdf/png`

Probe 共生成 2,610 行候选、135 个逐 target empirical Pareto 点；所有结果均保存脚本、依赖和 6 个输入文件的 SHA-256。
