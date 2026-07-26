# World Foundry Phase-2 执行契约

日期：2026-07-26

硬件：NVIDIA H200 NVL 143 GB

模型：Wan2.1-T2V-1.3B，480x832，UniPC 20 steps

## 1. 本轮目标

本轮只保留两条能够被真实硬件证据证伪的路线：

1. F81 主线：`geometry sparse high-rank attention + optional marginal correction + cache-aware refresh`。
2. F17 主线：exact cross-attention K/V cache 与整段 pointwise/kernel fusion。

CM/BCM、静态 weight-error low rank、静态 row-block FFN residual 和 hidden-channel FFT 不再作为主线。它们只有在 held-out trajectory 上显示稳定结构优势，并且存在 fused kernel 时，才重新进入候选集。

## 2. 已确认的出发证据

| 证据 | 结果 | 当前含义 |
|---|---:|---|
| F17 exact two-H200 CFG | 平均 1.7743x，final latent 与像素 exact | 真实可兑现的 exact system baseline |
| F81 profile self-attention 占比 | 53.88% | F81 应优先优化 attention，而不是 FFN residual |
| F17 elementwise/memory 占比 | 47.76% | F17 应优先做整段融合和 launch/allocator 优化 |
| runtime defect rank-16 energy | 11.37% 至 14.38% | 单一全局低秩 defect basis 不成立 |
| 跨运行 top-16 subspace overlap | 0.220 至 0.356 | 静态 low-rank residual 难以跨 prompt/step 泛化 |
| 完整 Wan speculative batch ratio | F17 B2=1.952，F81 B2=1.990 | 当前确定性 DiT 不适合传统并行候选验证式 speculative decoding |

以上数字是已完成实验的测量结果。本轮排队中的 F81 严格 CFG、geometry holdout 和 F17 cache/FFN 结果在完成前不作结论。

## 3. F81 attention 的理论边界

对单个 query，dense attention 写成：

```text
p_i = exp(s_i - m)
z = sum_i p_i
n = sum_i p_i v_i
o = n / z
```

将 key support 分成几何稀疏集合 `S` 与省略集合 `R`：

```text
z = z_S + z_R
n = n_S + n_R
o = (n_S + n_R) / (z_S + z_R)
```

因此只逼近输出 residual 并不完备。任何可执行 sparse + marginal 方案都必须同时处理：

1. 分子缺陷 `n_R`；
2. partition 缺陷 `z_R`；
3. sparse 与 marginal 分支共享的稳定 max/LSE；
4. mask/router 的构建代价；
5. 不规则 gather、padding 和低 occupancy 的执行损失。

本轮 geometry probe 使用 selected-key softmax 重新归一化，是真实 sparse attention 语义；rank-8/16 correction 是对已经观察到的当前 replay activation defect 做 SVD，只能作为 oracle 上界，不能视为部署算法。

## 4. 密度口径

每个候选同时报告三个密度，避免把理论稀疏率误当 H200 工作量：

1. `logical_density`：有效 token 对中被 mask 选中的比例。
2. `padded_block_density`：只计 K 侧 block padding 后的比例。
3. `execution_density`：同时计 Q/K 两侧补齐到完整 64x64 tile 后，相对有效 dense `N^2` 的实际 tile 工作量。

F81 `(T,H,W)=(21,30,52)` 不能被 64-token tile 完整整除，当前 smoke test 的 padding overhead 为 14.8718%。后续性能判断必须使用 `execution_density`。

## 5. Geometry mask 候选

当前候选全部由 token 几何和 layer phase 决定，不读取 dense QK 或 holdout defect：

| 候选 | F81 execution density | 作用 |
|---|---:|---|
| spatial 3x3 tiles | 1.5228% | 最小空间局部路径 |
| spatial 5x5 tiles | 3.2540% | 更宽空间上下文 |
| spatial 3x3 + same spatial tile across all time | 6.0111% | 长时同位置路径 |
| spatial 5x5 + same spatial tile across all time | 7.7423% | 宽空间与全时路径 |
| spatial 3x3 + temporal +/-2 | 7.1790% | 局部运动路径 |
| spatial 3x3 + full-time + 12 phased anchors | 8.5896% | 固定远程 escape tiles |

这些比例只证明 kernel 形状预算合理，不证明 attention 误差或视频质量满足要求。

## 6. 四级证据门

### Gate A：单 replay 数值可行性

每个 layer/step/branch/head 报告：

- query NRMSE p95/p99；
- query cosine p05；
- absolute LSE error p95；
- numerator relative error；
- multi-head pre-output-projection relative L2；
- dense fallback head 数量。

静态 sparse-head 局部门槛：

```text
p95 NRMSE <= 3%
p99 NRMSE <= 7.5%
p05 cosine >= 0.995
p95 |LSE error| <= 0.10
```

### Gate B：跨 trajectory 泛化

只在 calibration sample 冻结 `layer x step x branch x mask x head` 的 sparse/dense 策略，在 validation sample 上选择候选 mask，最后在不参与选择的 test samples 上原样评估。

每个 holdout cell 的联合门槛：

```text
pre-o-projection energy error <= 2%
dense fallback heads <= 3 / 12
effective execution density <= 12.5%
```

任何只在当前 replay 上按 defect energy 选择 fallback 的曲线，必须标记为 `oracle_current_replay_defect_energy`。

### Gate C：H200 fused-kernel 可兑现性

候选必须使用完整 64x64 block，并与相同输入、相同 dtype 的 FA3 dense baseline 比较。最低门槛：

```text
sparse attention kernel speedup >= 2.0x
```

路由、mask、LSE 合并、fallback 和 correction 的全部时间必须计入。若只有 FLOPs proxy 或 PyTorch gather 实现，则不得通过 Gate C。

### Gate D：端到端质量与速度

最终在多 prompt、多 seed、交替 method order 下比较 dense：

```text
SSIM >= 0.98
F81 end-to-end speedup >= 1.20x
```

同时报告 VBench/FVD 或可用的运动与时序指标，避免 SSIM 对感知和运动退化不敏感。若 Gate C 通过但 Gate D 失败，应优先检查 trajectory propagation 与 refresh，而不是继续增加 residual 分支。

## 7. Cache-aware refresh 的定位

cache 不是与 sparse/FP8 无条件叠加的第四种误差。外层 executor 采用互斥动作：

```text
D = dense BF16 anchor
Q = sparse/fused recompute
C = cache reuse or forecast
```

`D` 只能刷新当前 feature reference、cache age 和风险观测，不能恢复已经偏离的 latent trajectory。refresh 触发应至少考虑 timestep、motion boundary、CFG residual、cache age 与上次 anchor 后的累计风险。

本轮 exact text cross-attention K/V cache 不引入近似，主要用于测量 exact 系统冗余。由于 text K/V 在 F81 总时间中占比极小，预期只作为基础设施优化，不作为论文主创新。

## 8. F17 exact kernel 路线

既有测量表明，孤立 Triton bias+GELU 比 eager pointwise 快，但接回 GEMM 后反而更慢。原因是额外 launch、读写以及失去原 GEMM bias epilogue 优势。

本轮只保留完整 FFN 或更大范围路径：

1. eager 完整 FFN；
2. `torch.compile` 完整 FFN，记录首次编译与稳态；
3. 固定 shape CUDA Graph，记录 capture 成本与稳态；
4. 后续 CUTLASS/CuTe GEMM epilogue 或 LayerNorm/AdaLN/gate/residual 融合。

所有 exact 路径必须报告 bitwise fraction、max-abs、relative L2；若编译改变浮点归约顺序，则只能标记 numerically equivalent，不能称 bit exact。

F17 microbenchmark 的 go 条件是完整 FFN 稳态至少 1.10x，且预计映射到 profile 后有可见端到端收益。只优化约 1.8% 的 GELU 不足以进入端到端实验。

## 9. 正在执行的实验矩阵

1. F17 exact cross-attention K/V cache：2 prompts x 2 repeats，交替顺序。
2. F81 exact two-H200 CFG：4 prompts x 2 seed groups x 2 repeats，共 16 paired runs。
3. F81 geometry smoke：现有 F17/F81 replay，6 masks，rank 0/4/8/16 oracle tail。
4. F81 trajectory pilot：4 个析因 sample，step 0，layer 0，cond/uncond，共 8 QKV cells；最后一个目标 step 后立即停止 trajectory。
5. F81 frozen-policy 泛化：prompt 0/seed A calibration；prompt 1/seed A validation；prompt 0/seed B 与 prompt 2/seed B independent test。
6. 只有 pilot 的 frozen defect basis 通过 validation/test 后，才扩展到 steps 0/9/19 与 layers 0/14/29 的 72-cell 矩阵。
7. F17/F81 full FFN exact-path benchmark：代表层、eager/compile/CUDA Graph。

GPU 队列使用同一 `flock` 串行化，并在 H200 2/3 连续三个采样点均空闲后启动。这样等待外部作业属于正常排队，不应被误判为卡死。

## 10. 停止条件

出现任一条件即停止对应分支：

1. 所有 geometry mask 在 holdout 上都需要超过 3/12 dense heads 才达到 2% error。
2. 满足误差的候选 `execution_density > 12.5%`，或 fused kernel 对 FA3 小于 2x。
3. F81 端到端在 SSIM 0.98 下小于 1.20x。
4. rank-8/16 defect basis 在跨 prompt/step 上无稳定能量集中或子空间重合。
5. 完整 FFN compile/graph 稳态小于 1.10x，或收益被 capture/shape guard 抵消。

若 geometry oracle 可行但固定策略失败，研究重点转向低成本 sample-adaptive router；若固定策略与 kernel 均可行但端到端失败，研究重点转向 trajectory-aware refresh；若 oracle 本身失败，则不再用更复杂的 low-rank/CM residual 强行修补。
