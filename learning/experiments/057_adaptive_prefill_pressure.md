# 实验057：基于 Pending Pressure 的动态 Prefill Budget

## 实验问题

实验056表明静态修改 `max_prefill_length`、`decode_active_prefill_length` 和
`max_prefill_streak` 无法改善混合长 prompt 的 TTFT 尾延迟。当前 scheduler 已支持一个
压力感知分支：当 pending prefill token 达到阈值时，将 decode-active prefill budget 切换
到更小的 overload budget。本实验验证这条机制是否能在高压力时保护 Decode。

## 方法

固定实验056 baseline：Qwen3-32B FP8 TP4、FlashInfer、CUDA Graph max batch 32、Direct
PyNCCL、`num_pages=16384`、最大 Prefill 4096、`max_prefill_streak=1`、普通 decode-active
budget 1024。只改变 overload budget 和 pending token threshold。

Workload 使用 seed `5600042` 的相同开放环 trace：输入长度 256/1024/4096，输出 128，
到达率 0.5/1.0/1.5 req/s，每个配置每个速率 3 次。TTFT SLO 为 2000 ms，TPOT SLO 为
1000 ms。每个配置有 9 个 run，三个配置共 27 个 run、324 个请求，均完整返回。

| 配置 | overload budget | threshold | 目的 |
| --- | ---: | ---: | --- |
| baseline（实验056） | 0 | 0 | 普通 active budget 1024 |
| pressure4096_budget512 | 512 | 4096 | 队列超过约一个长请求时降半 |
| pressure2048_budget512 | 512 | 2048 | 更早触发保护 |
| pressure4096_budget256 | 256 | 4096 | 更激进地保护 Decode |

## 结果

| 配置 | 速率 | tok/s +/- std | P90 TTFT ms | TTFT SLO | avg TPOT ms | P99.9 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5 | 70.32 +/- 14.69 | 1071.98 | 100.0% | 20.89 | 240.52 |
| baseline | 1.0 | 98.64 +/- 8.78 | 1005.68 | 100.0% | 21.30 | 237.51 |
| baseline | 1.5 | 153.81 +/- 8.59 | 2489.72 | 80.6% | 30.18 | 244.49 |
| pressure4096_budget512 | 0.5 | 70.30 +/- 14.68 | 1055.08 | 100.0% | 21.02 | 273.85 |
| pressure4096_budget512 | 1.0 | 98.68 +/- 8.77 | 1005.46 | 100.0% | 21.32 | 226.17 |
| pressure4096_budget512 | 1.5 | 151.68 +/- 7.63 | 2608.29 | 80.6% | 30.42 | 231.09 |
| pressure2048_budget512 | 0.5 | 70.26 +/- 14.65 | 1112.07 | 100.0% | 21.13 | 272.63 |
| pressure2048_budget512 | 1.0 | 98.37 +/- 8.74 | 1036.98 | 100.0% | 21.10 | 222.13 |
| pressure2048_budget512 | 1.5 | 149.74 +/- 7.48 | 2687.47 | 80.6% | 30.75 | 225.93 |
| pressure4096_budget256 | 0.5 | 70.30 +/- 14.68 | 1031.55 | 100.0% | 21.12 | 286.95 |
| pressure4096_budget256 | 1.0 | 98.52 +/- 8.71 | 997.39 | 100.0% | 21.39 | 224.02 |
| pressure4096_budget256 | 1.5 | 148.02 +/- 6.38 | 2884.95 | 80.6% | 30.35 | 229.77 |

相对 baseline，所有动态配置在 1.5 req/s 的 TTFT SLO 仍为 80.6%，没有达到“提升至少 5
个百分点”的验收门槛。4096/512 的 P90 TTFT 还上升到 2608 ms，吞吐下降约 1.4%；更早
触发的 2048/512 和更小的 4096/256 分别把吞吐降低约 2.6% 和 3.8%，同时 TTFT 更差。

## Telemetry

| 配置 | prefill batches | budget-limited batches | budget selections | max pending tokens | max pending requests |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 232 | 149 | 1024:432, 4096:33 | 8472 | 5 |
| pressure4096_budget512 | 255 | 172 | 512:295, 1024:151, 4096:33 | 8472 | 5 |
| pressure2048_budget512 | 311 | 228 | 512:409, 1024:95, 4096:33 | 8472 | 6 |
| pressure4096_budget256 | 309 | 225 | 256:343, 1024:155, 4096:33 | 8472 | 6 |

动态分支不是没有执行：4096/512 确实有 295 次选择 512 budget，2048/512 有 409 次，
4096/256 有 343 次。但最大 pending token 仍然是 8472，说明只缩短当前 prefill batch 没有
降低队列峰值；它只是把同一批工作拆成更多次执行，增加了调度和 kernel 边界。

## 验收与决策

- **拒绝将现有 overload 参数作为默认优化**：没有提高 TTFT SLO，且高压力吞吐下降。
- 该结果排除了“单一 pending token 阈值 + 固定更小 budget”这一简单策略，但验证了压力
  感知入口本身可工作。
- P99.9 TPOT 的局部改善不足以抵消 TTFT 和吞吐回退；服务优化必须同时看 TTFT、TPOT、
  吞吐和队列压力。
- 下一步应研究更完整的策略：根据 pending token、active decode request 数和请求剩余
  prefill 长度联合决策，并允许压力下降后恢复大 budget；或者把 Prefill 与 Decode 做真正的
  资源隔离。下一次实验应先用短 trace/telemetry 验证策略，再做完整 3-repeat 矩阵。

## 产物

- `learning/results/057_*_fp8_poisson.json` / `.md`
- `learning/results/057_*_prefill_telemetry.jsonl`
- 基线：`learning/results/056_baseline_fp8_poisson.json`
