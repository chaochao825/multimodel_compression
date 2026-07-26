# Wan FFN 精确执行路径 H200 审计

日期：2026-07-26

## 结论

在独占 NVIDIA H200 NVL 上，Wan2.1-T2V-1.3B 的完整 checkpoint FFN 没有从通用 `torch.compile` 或单 FFN CUDA Graph 获得可用的严格保真加速。四条候选路径在 F17/F81、layer 0/14/29 共 24 个 cell 中均为 NO-GO。

- `torch.compile` 三种模式的 steady-state 中位加速均低于 `1.0x`，最差为 `0.868x`；同时改变了 BF16 数值路径，relative L2 为 `0.209%--0.389%`。
- 静态地址 CUDA Graph 是唯一 6/6 cell 逐位一致的路径。F17 三层中位加速为 `1.022x/1.070x/1.035x`，F81 为 `0.997x/1.010x/1.005x`，没有达到预注册的 `1.10x` 门。
- CUDA Graph capture 成本为约 `7.6--14.5 ms`。按每层 40 次调用摊销，F17/F81 最差加速仅 `0.761x/0.895x`；可达到正收益的 cell 需要约 154--892 次 replay，超过一次 20-step CFG 生成的同层调用预算。

因此停止 standalone FFN compile/graph 路线。F17 只保留两种工程候选：跨更大 residual/attention/FFN 区段的整图捕获，或真正把 bias/GELU/residual 融入 GEMM epilogue 的 kernel；两者都必须重新做端到端 Amdahl 门。F81 资源继续优先投向 attention。

## 实验协议

- 模型：Wan2.1-T2V-1.3B checkpoint 的真实 `ffn` 模块，hidden dim 1536、FFN dim 8960。
- 形状：F17 为 7,800 tokens，F81 为 32,760 tokens。
- 层：0、14、29。
- 路径：eager reference、静态地址 eager CUDA Graph、`torch.compile` default/reduce-overhead/max-autotune。
- 时延：10 次 warmup、50 次 replay，报告 median 与 P95；setup 按 40 次同层调用摊销。
- 门：逐位一致、median speedup >= 1.10、P95 speedup >= 1.00、40-call amortized speedup >= 1.00、额外显存 <= 4 GiB。
- 环境：PyTorch 2.9.1+cu128、CUDA 12.8、H200 NVL，完整 benchmark 约 52 秒。

GPU telemetry 的 ancestry 审计为 `timing_valid=true`：12 个快照中 foreign process 为 0，估计外部重叠 0 秒。因此这组时延可以用于性能结论。

## 关键数据

| 路径 | bitwise exact | 六 cell 中位 speedup 调和均值 | 最差 median | 最差 P95 | 最差 40-call amortized | 结论 |
|---|---:|---:|---:|---:|---:|---|
| CUDA Graph eager static | 6/6 | 1.022x | 0.997x | 0.969x | 0.761x | NO-GO |
| compile default | 0/6 | 0.948x | 0.911x | 0.868x | 0.009x | NO-GO |
| compile reduce-overhead | 0/6 | 0.924x | 0.874x | 0.867x | 0.077x | NO-GO |
| compile max-autotune | 0/6 | 0.922x | 0.868x | 0.859x | 0.002x | NO-GO |

`torch.compile` 的误差并不意味着视频质量一定明显下降；它说明编译器选择了不同的 BF16 GEMM/归约数值路径，不能再称为 same-trajectory exact。由于其 steady-state 本身也更慢，没有必要为它追加昂贵的视频质量实验。

## 为什么收益这么低

完整 FFN 由两个大 GEMM 主导。CUDA Graph 只能消除 launch/CPU 调度，不能减少 GEMM FLOPs 或中间张量的主要 HBM 流量，所以 F81 的局部收益接近零。F17 的 token 数较小，launch 占比稍高，因此个别层出现约 7% 局部提升；但 capture setup 相对一次短生成过大。

这与 source-bound fusion ceiling 一致：即使理想移除已测得的 hidden intermediate traffic，估计端到端上限也只有 F17 `1.019x`、F81 `1.011x`。通用 compile/graph 还没有真正实现跨 GEMM 的 epilogue 融合，因而不可能兑现这个理想上限。

## 后续门

1. 不再测试更多 `torch.compile` mode 或 standalone bias/GELU kernel。
2. 若实现 whole-block CUDA Graph，必须包含 norm、attention、FFN、residual 和固定地址输入更新，并直接测完整 denoise step；单 FFN microbenchmark 不可外推。
3. 若实现 GEMM epilogue fusion，先要求完整 FFN local speedup >= 1.10，再按 F17/F81 runtime share 计算端到端上限。
4. F81 主线继续做 head-role-aware、sample-adaptive coarse block router 与 fused sparse-high-rank attention；固定 token mask 与 frozen low-rank basis 保持 NO-GO。

## 产物

- 原始路径结果：`ffn_exact_h200_v3_first_idle/attempt_01/wan_ffn_exact_paths.csv`
- 完整逐 cell summary：`ffn_exact_h200_v3_first_idle/attempt_01/wan_ffn_exact_summary.csv`
- GPU 独占审计：`ffn_exact_h200_v3_first_idle/attempt_01/gpu_exclusivity_audit.json`
- 可视化与聚合数据：`ffn_exact_h200_v3_first_idle/summary/ffn_exact_paths.{png,pdf}`、`ffn_exact_plot_data.csv`
