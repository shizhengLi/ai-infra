# 实验058：联合 Prefill/Decode 压力门控

## 实验问题

实验057证明，只根据 pending prefill token 切换到更小 budget，会把相同工作拆成更多 batch，
既没有降低 8472-token 队列峰值，也没有改善 TTFT SLO。本实验增加第二个信号：活跃 Decode
请求数。只有 pending token 和 active decode 两个条件同时满足，才启用 overload budget。

直观上，长 Prompt 正在排队不一定意味着 Decode 需要保护；只有许多请求同时逐 token 生成时，
Prefill 才会直接挤占 Decode 的执行机会。联合门控用于减少低 Decode 压力下的误限流，并在压力
下降后自动恢复普通 1024-token budget。

## 实现

- `PrefillBudgetPolicy.select()` 新增 `active_decode_requests` 输入。
- 新增默认关闭的 `--decode-overload-active-threshold`。值为 0 时保留实验057的队列-only
  行为，因此不改变现有默认配置。
- overload 条件变为：Decode 可运行、pending token 达到阈值、active decode 达到阈值。
- Scheduler 直接使用 `len(decode_manager.running_reqs)`，不增加 GPU 同步或热路径张量操作。
- benchmark 报告记录 active threshold，保证结果可以复现。
- 对负数、孤立参数和联合门控边界增加单元测试。

## 固定条件

- Qwen3-32B，FP8 W8A8，BF16 runtime/KV cache，TP4，L20 GPU 0--3。
- FlashInfer、CUDA Graph 最大 batch 32、Direct PyNCCL、`num_pages=16384`。
- 最大序列 8192、普通 Prefill 4096、`max_prefill_streak=1`、Decode-active budget 1024。
- 输入 256/1024/4096 token 均衡混合，输出 128 token，每个 trace 12 个请求。
- 到达率 0.5/1.0/1.5 req/s，各 3 次，seed `5600042`。
- TTFT SLO <= 2000 ms，单请求最大 TPOT SLO <= 1000 ms。
- 复用实验056 baseline 和实验057 queue-only 4096/512 结果；两个新候选共 18 个 run、
  216 个请求，均完整返回。

## 参数矩阵

| 配置 | overload budget | pending threshold | active decode threshold | 目的 |
| --- | ---: | ---: | ---: | --- |
| baseline（实验056） | 0 | 0 | 0 | 当前 FP8 推荐配置 |
| queue-only（实验057） | 512 | 4096 | 0 | 单信号对照 |
| joint-768-a4 | 768 | 4096 | 4 | 温和限制，较早保护 |
| joint-512-a6 | 512 | 4096 | 6 | 强限制，只在高 Decode 压力触发 |

验收条件：1.5 req/s TTFT SLO 至少提高 5 个百分点或 P90 TTFT 降低至少 15%，吞吐损失不
超过 5%，TPOT SLO 不低于 99%。

## 结果

| 配置 | 速率 | tok/s | P90 TTFT ms | TTFT SLO | P99.9 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5 | 70.32 | 1071.98 | 100.0% | 240.52 |
| baseline | 1.0 | 98.64 | 1005.68 | 100.0% | 237.51 |
| baseline | 1.5 | 153.81 | 2489.72 | 80.6% | 244.49 |
| queue-only 512 | 0.5 | 70.30 | 1055.08 | 100.0% | 273.85 |
| queue-only 512 | 1.0 | 98.68 | 1004.46 | 100.0% | 226.17 |
| queue-only 512 | 1.5 | 151.68 | 2608.29 | 80.6% | 231.09 |
| joint-768-a4 | 0.5 | 70.31 | 1082.06 | 100.0% | 240.87 |
| joint-768-a4 | 1.0 | 98.61 | 994.84 | 100.0% | 230.41 |
| joint-768-a4 | 1.5 | 152.97 | 2505.13 | 80.6% | 243.92 |
| joint-512-a6 | 0.5 | 70.30 | 1084.28 | 100.0% | 241.67 |
| joint-512-a6 | 1.0 | 98.64 | 1005.02 | 100.0% | 231.01 |
| joint-512-a6 | 1.5 | 153.47 | 2496.96 | 80.6% | 235.16 |

联合门控成功消除了 queue-only 方案的大部分吞吐回退：在 1.5 req/s，joint-512-a6 相对
baseline 仅下降 0.22%，而 queue-only 下降 1.38%。P99.9 TPOT 由 baseline 的 244.49 ms
降到 235.16 ms，但 P90 TTFT 仍略高 0.29%，TTFT SLO 仍是 80.6%，未通过主验收门槛。

## Telemetry 与原理解释

| 配置 | prefill batches | limited batches | budget selections | max pending tokens | max pending requests |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 232 | 149 | 1024:432, 4096:33 | 8472 | 5 |
| queue-only 512 | 255 | 172 | 512:295, 1024:151, 4096:33 | 8472 | 5 |
| joint-768-a4 | 236 | 153 | 768:258, 1024:176, 4096:33 | 8472 | 5 |
| joint-512-a6 | 237 | 154 | 512:240, 1024:190, 4096:33 | 8472 | 5 |

联合门控确实执行：采样的 768-budget batch 对应 4--8 个活跃 Decode 请求。它把 Prefill
batch 数从 queue-only 的 255 降到 236/237，解释了吞吐恢复；但所有配置的最大 pending
token 都是 8472，说明门控只能减少误限流，不能从根本上消除长 Prompt 到达形成的队列。

## 决策

- 保留 `--decode-overload-active-threshold` 作为默认关闭的实验/调优能力；默认 0 完全兼容
  原行为。
- 不把任何联合门控候选设为 L20 默认配置：吞吐和 TPOT 风险已受控，但 TTFT 主指标没有提升。
- 停止继续扫固定阈值。下一个调度实验应改变服务机制而不是继续调数字，例如 admission
  control、按等待时间/剩余 Prefill 排序，或 Prefill/Decode 资源隔离。
- 面试表述重点：用两个运行时信号降低策略误触发，并用 telemetry 解释吞吐恢复；同时依据
  SLO 拒绝一个“实现正确但业务指标不增益”的优化。

## 产物

- `learning/results/058_joint_4096_768_active4_fp8_poisson.{json,md}`
- `learning/results/058_joint_4096_512_active6_fp8_poisson.{json,md}`
- `learning/results/058_joint_*_prefill_telemetry.jsonl`
- 基线：`learning/results/056_baseline_fp8_poisson.json`
