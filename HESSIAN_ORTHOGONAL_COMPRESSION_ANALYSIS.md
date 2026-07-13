# Hessian 正交组合压缩：条件、舒适区与同预算优势

## 结论

组合压缩可以在相同压缩率下优于单一压缩，但“正交”不是自动成立，也不是充分条件。需要同时满足：

1. 各压缩扰动在同一个 PSD Hessian/Fisher 度量下低相关；
2. 所有扰动仍处于局部二次近似的 trust region；
3. 单方法离开舒适区后，单位压缩率的边际 loss 明显上升；
4. bit 预算包含 mask/index、scale、codebook 和修补参数；
5. calibration 上的正交与组合优势能迁移到 held-out task loss。

本轮在 8 张 attention map、4 个 source input 上得到以下结果：

- 对压缩误差 `delta=C(A)-A`，Frobenius（等价于 squared-NRMSE 的常数倍二次度量）下 structured–pruning、pruning–quantization、structured–quantization 的平均 cosine 分别为 `0.095 / 0.088 / 0.797`。structured 与 quantization 明显重复破坏同一方向。
- 换成局部 KL Hessian 后分别为 `0.298 / 0.135 / 0.659`。因此正交性依赖目标 loss；普通 Frobenius 正交不能替代任务曲率正交。
- 在 softmax-logit Fisher 模型中，完整 OBS survivor correction 将 pruning residual 与任意 retained-only quantization 扰动的平均 `|rho_H|` 降到 `4.38e-17`，最大值为 `1.53e-15`，验证了固定二次模型中的数值精度级正交。
- 受边界约束的 cross-null row scale 将平均 `|rho_H|` 降到 `9.97e-4`，但最大值仍为 `0.0607`，且 `498/206,272=0.241%` 的 scales 发生裁剪。因此它是低成本近似投影，不是严格正交构造。
- 严格码率比较要求单方法家族中存在一个使用组合预算至少 `99%` 的可实现 codec，并且报告 Taylor 舒适区内结果。在实际 payload `24.03%`（约 `4.16x` 压缩）处，组合在 8/8 maps、4/4 sources 上获胜：逐 map 相对 Hessian/KL gain 的等权平均为 `49.2%/44.0%`；先在 source 内平均逐-map gain、再对 sources 等权的结果为 `45.6%/41.2%`。KL 固定使用同一对 Hessian-selected codecs。
- 更极端的 `19.34%` payload 点只有 6/8 maps、3/4 sources 同时通过 Taylor 检查；在这 6 个样本上 Hessian/KL 收益为 `68.4%/63.4%`，只能作为条件性证据。到 `28.44%` payload，逐-map Hessian relative gain 的等权平均仅 `+1.0%`，endpoint KL 为 `-8.3%`，source-balanced 两者均为负；从 `34.61%` payload 起最佳单方法明确胜出。

因此，正确结论不是“组合一定更好”，而是：**把每种方法限制在各自低边际曲率的舒适区，并把总压缩需求分配到互补的 Hessian 子空间，才能扩展低损失压缩率。**

## 1. 什么是 Hessian 正交

设基准参数为 `theta`，第 `i` 种压缩产生固定扰动 `d_i`。局部展开为

\[
\Delta L\left(\sum_i d_i\right)
=g^\top\sum_i d_i
+\frac12\sum_i d_i^\top H d_i
+\sum_{i<j}d_i^\top H d_j
+R_3.
\]

定义

\[
\langle d_i,d_j\rangle_H=d_i^\top H d_j,
\qquad
\rho^H_{ij}=\frac{d_i^\top H d_j}
{\sqrt{d_i^\top H d_i}\sqrt{d_j^\top H d_j}}.
\]

在固定的精确二次模型中，pairwise `d_i^T H d_j=0` 是任意子集 loss 二阶可加的充分必要条件。如果只要求一个完整组合的交叉项总和为零，正负 interaction 可以偶然抵消，但这种结果通常不稳定。

实际达到可用的“正交效果”还要求：

- `H` 是 PSD。推荐 damped GGN/Fisher，而不是直接把含负特征值的 full Hessian 当内积；
- `g^T d_i` 很小或被单独报告。非驻点上的一阶项可能大于二阶项；
- quantization codes、pruning mask、router support 固定后再做 `epsilon` 路径；否则离散切换处 Hessian 不存在；
- `G_ii` 高于数值噪声，`|rho_H|` 小且 bootstrap 区间仍在阈值内；
- `epsilon=0.02/0.05/0.1` 和 `alpha:0→1` 路径稳定，实际 mixed loss interaction 与 `d_i^T H d_j` 一致；
- 两种组合顺序得到接近的 endpoint loss；
- calibration Hessian 与 held-out Hessian 的主要子空间稳定。

若 `H=B^T B`，Hessian 正交等价于白化后的扰动 `B d_i` 普通欧氏正交。若 `H≈J^T F J`，它表示两种压缩导致的 `F^(1/2)Jd_i` 输出变化正交，而不是参数 support 简单不重叠。

## 2. 剪枝与量化可以构造二阶精确正交

把参数分为被剪集合 `P` 和保留集合 `R`。naive pruning 固定产生

\[
d_P=-\theta_P.
\]

允许 retained weights 做补偿，最小化局部二阶 loss：

\[
d_R^*=\arg\min_{d_R}
\frac12
\begin{bmatrix}d_P\\d_R\end{bmatrix}^{\!T}
H
\begin{bmatrix}d_P\\d_R\end{bmatrix}
=H_{RR}^{-1}H_{RP}\theta_P.
\]

于是

\[
H_{RP}d_P+H_{RR}d_R^*=0.
\]

后续 quantization 只修改 retained weights，记为 `q=[0,q_R]`，则

\[
(d_{\text{prune}}^*)^T Hq
=q_R^T(H_{RP}d_P+H_{RR}d_R^*)=0.
\]

这就是 `OBS/Hessian-compensated pruning + retained-weight quantization` 的精确二阶正交条件。补偿后的 pruning loss 由 Schur complement 给出：

\[
\frac12\theta_P^T
\left(H_{PP}-H_{PR}H_{RR}^{-1}H_{RP}\right)\theta_P,
\]

不高于 naive pruning 的 `0.5 theta_P^T H_PP theta_P`。

本轮用每行 softmax Fisher

\[
H=\operatorname{diag}(p)-pp^T+\lambda I
\]

验证该性质。`H_RR` 是 diagonal-minus-rank-one，可以用 Sherman–Morrison 计算，无需显式构造完整 inverse。OBS 修补是 encoder/calibration 步骤；最终修补值直接折叠进 retained FP16 values 或 quant codes/scales，不增加 decoder payload。

## 3. 少量 scale 能做到多少

若修补只允许位于低维 basis `U` 中：

\[
d^*=d_0+U\alpha^*,\qquad
\alpha^*=-(U^THU)^\dagger U^THd_0,
\]

则只能保证

\[
U^THd^*=0.
\]

所以一个全局或 block scale 只能消掉其 scale tangent direction 上的交叉项；它不能自动与任意逐元素 quantization error 正交。要消掉 `k` 个独立压缩方向，修补 basis 至少需要相应的有效秩，并满足 `D^T H U` 对目标约束满秩。

当前每种组合变体各有 1,408 组配置，均值如下：

| 修补方式 | Mean `|rho_H|` | Mean damped-H loss | Mean exact KL | 解释 |
|---|---:|---:|---:|---|
| Naive pruning + max scale | `4.00e-3` | `0.10116` | `0.09491` | 基线 |
| 1-scale pruning repair | `6.16e-3` | `0.09392` | `0.09380` | loss 略降，但未对准任意 quant error |
| Bounded cross-null folded scale | `9.97e-4` | `0.09331` | `0.10335` | 交叉项更小，但 endpoint KL 反而变差；0.241% scales 触边 |
| Loss-optimal folded scale | `0.12424` | **`0.04739`** | **`0.07469`** | 不追求正交，而是允许有利 interaction |
| Full OBS + max scale | **`4.38e-17`** | `0.08568` | `0.09087` | 对所有 retained-only quant noise 二阶正交 |

这张表说明：

- scale 必须改变误差方向，而不是只把同一扰动乘一个 scalar；
- bounded cross-null scale 可以替换已存在的 FP16 quant scale，不增加 bits，但裁剪和 FP16 rounding 会留下 residual；
- 最小 total loss 不一定出现在 `rho_H=0`。负/非零 interaction 可以有利，但更依赖 calibration，跨数据迁移风险更高；
- full OBS 最稳健，但 encoder 侧需要更丰富的 retained-space correction；低秩、block-diagonal、KFAC 或 group basis 是实际近似方向。

## 4. 同压缩率下组合为什么可能更好

令 `x_i=log C_i` 表示方法 `i` 提供的对数压缩率。组合压缩近似满足

\[
\log C_{\text{total}}\simeq\sum_i x_i.
\]

若误差 Hessian-orthogonal，局部 rate–distortion 分配为

\[
D_{\text{combo}}^*(X)
=\min_{\sum_i x_i=X}\sum_i\phi_i(x_i).
\]

最优解对所有激活方法满足

\[
\phi_i'(x_i)=\lambda,
\]

即每多节省一 bit 的边际 Hessian loss 相等。因为“把全部预算交给一种方法”也是可行解，允许组合后的最优值不会高于单方法；严格优势出现于：

- 至少两种方法的误差方向互补；
- `phi_i` 离开舒适区后严格凸，边际 loss 快速增加；
- 最优分配实际激活多个方法；
- cross term 和 metadata 开销不足以吃掉收益。

若两个方法的舒适区曲线都是 `phi(x)=a x^2`，平均分配时损失从 `aX^2` 降为 `aX^2/2`。若 Hessian correlation 为 `rho`，组合/单方法的近似比值为 `(1+rho)/2`。`rho=0` 时二阶损失减半；`rho≈1` 时两种方法重复破坏同一方向，没有优势。

## 5. 同 bit 实验结果

实验将每行 attention 转成中心化 FP16 logits，以 softmax Fisher + damping 作为 PSD Hessian。pruning mask 按 diagonal-Hessian sensitivity 选择；量化使用 signed codes 和 FP16 row scale。bit 计费包含：

- retained codes/FP16 values；
- 每行 bitmap 或 fixed-row COO 中较小的 support 编码；
- 每行一个 FP16 quant scale；
- 修补直接折叠进 retained values/codes，不额外计 metadata scalar。

单方法候选包含整数 `2..12` bit dense quantization，以及真正编码 row-bitwidth bitmap 的逐行 `q/(q+1)` mixed precision；后者用最多 64 档 row count 填充码率。剪枝率以 5% 为间隔覆盖 `5%..80%`。每个组合 winner 只与同一实际 bit budget 下的最佳单方法 envelope 比较，并要求该 envelope 中至少一个可实现 codec 使用该预算的 `99%`。下表进一步要求双方 endpoint KL 与 Fisher quadratic 的比值都在 `[0.8,1.25]`；KL 列固定使用 Hessian 选中的同一对 codec，而不是重新按 KL 选 winner。

| Payload cap | Mean actual payload | Maps / sources | Hessian gain map/src | Same-codec KL gain map/src | H wins | KL wins |
|---:|---:|---:|---:|---:|---:|---:|
| 20% | 19.34% | 6 / 3 | `+68.4% / +66.9%` | `+63.4% / +61.9%` | 6/6 | 6/6 |
| 25% | 24.03% | 8 / 4 | `+49.2% / +45.6%` | `+44.0% / +41.2%` | 8/8 | 8/8 |
| 30% | 28.44% | 8 / 4 | `+1.0% / -22.4%` | `-8.3% / -36.1%` | 6/8 | 5/8 |
| 35% | 34.61% | 8 / 4 | `-74.8% / -89.9%` | `-85.3% / -100.2%` | 1/8 | 0/8 |
| 40% | 37.43% | 8 / 4 | `-112.8% / -139.0%` | `-121.4% / -148.8%` | 0/8 | 0/8 |
| 50% | 48.00% | 8 / 4 | `-193.7% / -204.2%` | `-196.7% / -207.4%` | 0/8 | 0/8 |

`24.03%` 是当前离散、target-fitted grid 中未经 held-out 选择校正的最完整组合优势点。此时最佳单方法在全部 maps 上都是 dense mixed `q3/q4`；最佳组合则剪掉 `35%..70%` 的低敏感度 logits，再给 survivors 使用 `q4..q9`。参数利用率优势来自把“不重要坐标的 value bits”转移给“重要 survivors 的更高精度”，并用 OBS 或 folded scale 修补敏感误差方向。这正是稀疏版的“scale 保护”：小量结构/scale 信息不是直接增加容量，而是保护高 Hessian-energy 子空间。

`20%` 点若不做 Taylor 过滤可覆盖 8/8 maps，但不能据此下 endpoint 结论；`30%` 点已经对 map/source 聚合方式敏感；进入更高 payload 后，dense quantization 可以给所有坐标更高 bit，而组合仍承担 pruning error floor，因此单方法更优。这里不存在普适的 25% knee，只有当前 target-fitted codec grid 上的条件性舒适区。

## 6. 为什么“普通正交”还不够

当前 representative codec 的 error Gram 为：

| Error pair | Frobenius / squared-NRMSE cosine | Local-KL-H cosine (`floor=1e-8`) |
|---|---:|---:|
| structured–pruning | `0.095` | `0.298` |
| pruning–quantization | `0.088` | `0.135` |
| structured–quantization | `0.797` | `0.659` |

因此：

- structured + quantization 不应默认组合，它们在两个指标下都高度冗余；
- structured + pruning 在 NRMSE 下近似正交，但在 KL 下不满足 `|rho_H|<0.1`；
- pruning + quantization 最有潜力，并可通过 OBS correction 构造精确二阶正交；
- 评价指标必须与最终 task loss 一致。选择恒等 Hessian 会把任务敏感方向全部等权处理。

以上 Gram 是 8 张 maps 等权平均，五种 repair variant 是所有 map/configuration 等权平均；source 分布为 `2/1/1/4`，因此它们不是 source-balanced 统计。headline matched-rate 结果同时给出逐-map 与 source-balanced gain，避免让单一 source 的四张 maps 占一半权重。

## 7. 下一步任务级验证

当前仍是 8 张 target-fitted attention map、4 个 source input 的受控实验。尚未验证模型权重、held-out task loss、accuracy、真实 serialized bytes 或 latency。

任务级验证建议在 ViT/SCTM checkpoint 上压缩位于 route selection 之后的 `v_proj/out_proj`，避免 top-k support 跳变。固定构造 `d_quant/d_prune/d_lowrank`，用 projected GGN：

\[
G_{ij}=E\left[v_i^T(\operatorname{diag}p-pp^T)v_j\right],
\qquad
v_i=\frac{z(\theta+\epsilon d_i)-z(\theta-\epsilon d_i)}{2\epsilon}.
\]

同时用 CE mixed finite difference 检查 true-H interaction，报告一阶项、`epsilon` 稳定性、双顺序 endpoint loss、held-out accuracy 和 whole-model packed rate。三种扰动的 GGN 只需约 7 组 forward；完整 pairwise central-difference 校验约 19 组。

## 8. 复现

```powershell
python scripts/hessian_orthogonal_compression_probe.py
python figures/hessian_orthogonal_compression_plot.py
python -m pytest -q tests/test_hessian_orthogonal_compression_probe.py
```

产物：

- `remote_logs/hessian_orthogonal_compression_20260712.json/csv`
- `remote_logs/hessian_compression_error_gram_20260712.csv`
- `remote_logs/hessian_matched_rate_20260712.csv`
- `figures/fig25_hessian_orthogonal_compression.pdf/png`

Probe 生成 17,680 条唯一 codec 结果、96 条 error-Gram 结果和 284 条严格码率 matched-rate 结果，并保存脚本、依赖与 6 个输入文件的 SHA-256。
