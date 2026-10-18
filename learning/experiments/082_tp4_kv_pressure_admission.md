# 实验 082：TP4 KV cache pressure 与 admission backpressure

## 实验目的

实验 081 的请求数较少，虽然输出长度为 128 token，但没有把 KV cache 推到明显的排队
边界。本实验固定 `page_size=16`，把请求数提高到 16，使单轮潜在需求约为
`16 * (17 + 128) = 2320 token`，明显超过 cache 容量，观察：

1. 到达率从 20 提高到 80 req/s 后，吞吐和 TTFT tail 如何变化；
2. 极小 cache 是否触发 `decode_capacity_trimmed/deferred`；
3. scheduler 是通过 decode 裁剪，还是通过 prefill admission 和排队保护 allocator；
4. 小 cache 下 decode batch、GPU 利用率和请求完成率的变化。

## 实验矩阵

| 场景 | page size | num pages | 总 KV 容量 | 到达率 | 请求数 | 输出长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal pressure | 16 | 64 | 1024 token | 20/80 req/s | 16 | 128 |
| severe pressure | 16 | 16 | 256 token | 80 req/s | 16 | 128 |

其他条件固定：L20 GPU 2/3/6/7、Qwen3-32B-AWQ、TP4、`float16`、默认 PyNCCL、token
reservation、normal prefill budget=64、decode-active budget=32、最大 prefill streak=1、
CUDA Graph batch shapes `[1,2,4,8]`。每个场景重复 3 次，TTFT/TPOT SLO 设为 120 s。

完整 benchmark 结果：

- [`082_tp4_pressure.md`](../results/082_tp4_pressure.md)
- [`082_tp4_pressure256.md`](../results/082_tp4_pressure256.md)

## 1024-token cache 结果

| Arrival rate | 吞吐 tok/s | 标准差 | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P90 E2E s | GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 req/s | 380.87 | 8.24 | 3261.43 | 3276.32 | 42.47 | 4.762 | 84.48% |
| 80 req/s | 392.95 | 0.67 | 3471.03 | 3506.16 | 36.91 | 4.974 | 87.67% |

从 20 提高到 80 req/s 后，吞吐没有下降，说明该 workload 下 GPU 仍有处理余量；但 P90
TTFT 增加约 6.4%，P90 E2E 增加约 4.5%，排队压力首先体现为首 token 等待，而不是 decode
速度崩溃。两组都是 48/48 请求完成，TTFT/TPOT SLO 均为 100%。

## 256-token cache 结果

| Arrival rate | 吞吐 tok/s | 标准差 | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P90 E2E s | GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 req/s | 89.96 | 0.59 | 19769.62 | 21188.83 | 12.04 | 21.147 | 96.54% |

256-token cache 下 48/48 请求仍然完成，但吞吐只有 89.96 tok/s，P90 TTFT 达到 19.77 秒，
P99 TTFT 达到 21.19 秒，GPU 利用率接近 96.5%。这说明容量不足时系统没有丢请求或破坏
allocator 不变量，而是把压力转化成串行化和排队延迟。

## Telemetry 与调度解释

1024-token 场景共记录 7 个 trace（含 warmup）：

- `decode_capacity_trimmed=0`、`decode_capacity_deferred=0`、retraction=0；
- 最大 pending prefill 为 11 个请求、254 token；
- 7 个 trace 合计 118 次 budget-limited batch；
- decode batch 主要为 4/6，说明仍能维持一定批处理并行度。

256-token 场景共记录 4 个 trace（含 warmup）：

- `decode_capacity_trimmed=0`、`decode_capacity_deferred=0`、retraction=0；
- 最大 pending prefill 为 15 个请求、375 token；
- 所有 6160 个 selected decode batch 都是 batch size=1；
- 没有 capacity trim 事件，但请求被 admission 排队，导致 TTFT tail 显著上升。

因此，本实验没有证明容量裁剪分支的线上收益，也不能把 telemetry 为 0 理解成“没有
容量压力”。恰恰相反，结果表明当前 scheduler 在进入 `_fit_decode_batch` 前就限制了活跃
请求集合；在极小 cache 下，这种保护策略把压力表现为等待时间和 batch 并行度下降。

## 结论

1. TP4 PyNCCL 路径在 1024/256-token KV cache 下都保持正确性，所有请求完成，没有
   allocator assertion 或请求丢失。
2. 1024-token cache 可承受 20/80 req/s 的短请求流，主要代价是 TTFT tail 小幅增加；
   256-token cache 则明显退化到 batch=1，P90 TTFT 接近 20 秒。
3. admission backpressure 是当前实现的主要容量保护机制，decode capacity telemetry 在
   真实压力下仍未命中。后续若要评估 073-076 的裁剪公平性，需要构造能让请求进入
   `_fit_decode_batch` 但未来 page demand 超限的确定性 trace。
4. 暂不修改默认 scheduler 策略。下一步适合做“固定 256-token cache、token 与 page-aware
   reservation A/B”，同时记录 UID 等待轮数和请求完成顺序，判断安全保护是否带来饥饿。

## 面试表达要点

可以这样解释：我把总 KV 容量压到 256 token，同时提交 16 个总需求约 2320 token 的请求，
系统没有通过粗暴拒绝请求来解决内存不足，而是在 admission 层把活跃 decode batch 限制为
可容纳范围，剩余请求排队。结果是正确性保持不变，但 P90 TTFT 从约 3 秒上升到约 20 秒，
并行 decode 从多请求 batch 降为 batch=1。这说明性能优化不能只看 GPU 利用率，还要区分
吞吐、排队延迟、KV allocator 安全不变量和公平性。
