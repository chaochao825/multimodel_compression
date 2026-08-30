# Risk-Observable Writer 前瞻性审计

日期：2026-08-30

状态：有效 `NO_GO`；关闭当前 post-hoc quotient + query-independent 32D writer + exact-group retrieval 家族

## 直接结论

本轮主动改变了存储接口，而不是继续扩大失败的 metadata controller，但仍未恢复随问题旋转的 task risk：

> 32D learned writer key 能小幅降低 soft KL，却几乎没有提高高风险 group 的可观测性；joint writer-controller 也没有显示组合协同。更关键的是，同预算 target-gradient oracle 自身只有 `91.67%` agreement 和 `2` harmful，因此当前 43.75% support 与一阶 teacher 的容量上限也未达到严格保真。

prospective 上：

- fixed controller recall：`30.23%`；
- learned writer + dot recall：`30.02%`；
- joint writer-controller recall：`30.70%`；
- joint 相对 fixed 只增加 `0.47 pp`，相对 writer-only 只增加 `0.68 pp`；
- joint risk mass 为 `31.93%`，低于注册的 `50%`；
- joint reader agreement 为 `70.83%`，低于 `91.67%` Gate；
- 判决为 `NO_GO`。

因此不读取保留的 calibration positions 97--120，不进入风险证书、selection、formal 或速度测试，也不继续调 key width、mode width、rank、group 数或 fallback threshold。

## 协议与执行有效性

数据在 prospective 读取前冻结为：

- positions 1--48：train；
- positions 49--72：validation；
- positions 73--96：一次性 prospective；
- positions 97--120：保留；
- selection/formal：未读。

共同预算为 8 帧、1,568 visual tokens、392 个连续 4-token groups；每组一个 quotient mean，恰好 98 个 groups 恢复 exact tokens，无 fallback 时 retention 为 `43.75%`。

writer 使用固定 4x4 Hadamard 将每组分为一个 quotient mode 和四个 signed innovation modes，再经过：

\[
H\xrightarrow{W_h}8,
\qquad
5\times 8\xrightarrow{W_m}32
\]

生成 query-independent 32D key。问题侧使用 (H\to32) projection。三种 learned 对照在相同 48/24 开发数据、top-98 labels、AdamW、100 epochs 和 earliest-best validation recall 下训练。

第一次正式启动在 prospective 读取前因日志变量覆盖失败：JSON 尝试序列化 position Tensor。失败日志和 `v1` 路径保留；唯一修复只重命名日志局部变量，不改变数据、模型或 Gate。`v2` 从头运行，完成 train/validation capture、writer input、模型冻结后才读取 prospective，顺序有效。

## 主要结果

### 1. Writer interface 几乎没有恢复 oracle gap

| selector | validation recall | prospective recall | prospective risk mass |
|---|---:|---:|---:|
| question cosine | `28.23%` | `27.89%` | `27.62%` |
| residual energy | `28.49%` | `29.17%` | `29.46%` |
| fixed writer + controller | `31.46%` | `30.23%` | `30.22%` |
| learned writer + dot | `31.85%` | `30.02%` | `31.61%` |
| joint writer-controller | **`32.10%`** | **`30.70%`** | **`31.93%`** |
| target-gradient oracle | `100%` | `100%` | `80.95%` |

joint 只恢复了 fixed-to-oracle recall gap 的 `0.67%`，以及 risk-mass gap 的 `3.38%`。validation 到 prospective 的轻微下降说明结果不是灾难性过拟合；主要问题仍是函数可观测性与目标容量。

### 2. Soft KL 改善没有转成决策保持

| prospective selector | mean KL | agreement | harmful | candidate accuracy |
|---|---:|---:|---:|---:|
| residual energy | **`0.03861`** | `75.00%` | `3` | `12.50%` |
| question cosine | `0.06116` | `75.00%` | `2` | `25.00%` |
| fixed controller | `0.05126` | `75.00%` | `2` | `25.00%` |
| writer + dot | `0.06823` | `70.83%` | `2` | `25.00%` |
| joint writer-controller | `0.04325` | `70.83%` | `2` | `25.00%` |
| target-gradient oracle | **`0.02957`** | **`91.67%`** | `2` | `16.67%` |

joint 相对 fixed 将 KL 降低 `15.64%`，恢复约 `36.95%` 的 fixed-to-oracle KL gap，却多产生一个 mismatch。这再次证明平均 distribution distortion、top-1 agreement 和 harmful flips 不是同一个目标。

baseline full-reader accuracy 在该 24-question slice 上仅 `25%`，因此不能从 aggregate accuracy 不降推断方法安全。joint 与 full reader 都是 `25%`，但 joint 仍有 `2` 个 harmful；oracle accuracy 甚至降到 `16.67%`。

### 3. 容量 ceiling 本身未通过

target-gradient risk 知道 held-out reader Jacobian，并直接选择 top-98 groups，仍只有：

- agreement `91.67%`；
- mismatch `2/24`；
- harmful `2/24`；
- mean KL `0.02957`。

因此即使开发一个完美 predictor 复现当前一阶 teacher，也不能达到 `98%/0 harmful` 的最终目标。剩余误差来自至少三部分：

1. 98-group budget 不足；
2. group effects 在 Transformer/softmax 中非加性，一阶独立排序忽略交互；
3. top-1 margin teacher 与 ground-truth task loss 不等价。

## 为什么改变 writer 仍失败

当前函数族为：

\[
k_g=E_\phi(X_g),
\qquad
\hat r_g=f_\psi(q,k_g,b_g,p_g).
\]

真实 teacher 则是：

\[
r_g^*=F\left(
\nabla_{X_g}m(X_1,\ldots,X_G,q),
\delta X_g,
m
\right).
\]

梯度依赖完整视觉上下文、所有 group 的竞争、深层 cross-modal routing 和当前答案边界。query-independent local key 即使保留局部方向，也没有观测全局 reader state。这个差异可写为仍然很大的条件创新：

\[
\mathbb E\operatorname{Var}
\left(r_g^*\mid q,k_g,b_g,p_g\right).
\]

32D key 不是唯一可能的表示，但本轮已排除“固定随机 sketch 太弱”这一简单解释：使用完整 group 的可训练 Hadamard-mode writer 后，recall 仍停在约 `30%`。继续把 32 改成 64/128 更可能增加 payload，而没有证据改变缺失的 global reader state。

## 与条件冗余/热力学理论的一致性

理论判断仍成立，但它只指出正确 metric，不自动提供廉价 observer。

- 稳定 quotient 说明视觉状态存在统计冗余；
- target-gradient oracle 说明 reader-induced metric 比欧氏 proxy 更接近任务价值；
- tiny controller 和 learned writer 均失败，说明该 metric 不能由当前低带宽局部状态稳定观测；
- oracle 自身未过门槛，说明一阶局部风险也不是完整 endpoint certificate。

在 diffusion 中，这对应“知道 drift-error metric”不等于能从廉价 feature 估计整条路径 KL；在视频理解中则对应：

\[
G_q=J_{X\to\ell(q)}^\top H_\ell J_{X\to\ell(q)}
\]

是合理度量，但计算或预测 (G_q) 的足够统计量仍接近 reader 本身。不能把这一点包装成 entropy-production 的物理验证。

## 对最初结构化动机的最终边界

保留下来的部分：

- quotient/低秩 bulk 适合做存储、索引和 amortized memory；
- exact innovation 适合规则 group/tile 冷存；
- risk metric 应由任务边界而非 raw L2 定义。

被当前证据关闭的部分：

- 固定 reader-risk basis；
- frame-level exact retrieval；
- query/residual static proxy；
- scalar compressed-margin certificate；
- fixed metadata width-32 controller；
- query-independent Hadamard low-rank 32D writer；
- 当前一阶 top-98 risk teacher作为严格证书。

BCM/BCCB/Butterfly 不能解决这里的核心缺失变量。它们可以改变 writer 编码成本，却不能从 local (X_g) 推导依赖完整 reader trajectory 的 Jacobian。

## 是否还值得继续

在“冻结 reader、post-hoc 写入、43.75% retention、近乎无损”约束下，潜力已经很低，应 `park` 而不是继续修饰组件。

若未来重新打开，只剩一个本质不同的方向：训练原生 memory tokenizer 与 reader，使压缩码在训练时成为任务充分接口，并用 differentiable rate/coverage loss 联合优化 reader。此时必须公平比较：

- learned tokenizer only；
- learned reader routing only；
- joint tokenizer-reader；
- LongVU、FrameFusion、StreamingTOM、FlexMem 类方法。

它将是一个新的训练项目，不再是当前 post-hoc quotient 的自然下一版。创新也不能声称 query-aware memory 或 progressive retrieval；必须证明 adverse-risk training 在相同 memory/latency 下带来独立收益，并完成真实任务与系统评估。

## 最终判决

\[
\boxed{
\text{bulk 冗余存在}
\;\land\;
\text{task metric 有价值}
\;\land\;
\text{局部低带宽 observer 不充分}
\;\land\;
\text{当前 teacher/budget ceiling 也未过门槛}
}
\]

这不是“所有视频理解压缩失败”，而是对当前研究问题的清晰止损：如果不联合训练 reader 的表示与读取机制，就没有证据继续扩大 post-hoc writer/controller 家族。
