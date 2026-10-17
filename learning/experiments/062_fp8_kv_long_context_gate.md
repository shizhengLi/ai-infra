# 实验 062：FP8 KV 长上下文质量与性能 gate

日期：2026-10-17  
状态：完成；容量和单请求输出一致性通过，当前 eager 路径性能 gate 未通过

## 目标

实验 061 已证明 FP8 KV 可以把 Qwen3-32B TP4 的 KV capacity 从 413,820 提高到
827,641 tokens，但 capacity 不等于端到端收益。本实验固定 Qwen3-32B FP8 W8A8 权重、L20
GPU 2/3/6/7、FlashInfer、TP4 和单请求 workload，只切换 KV cache dtype，比较 4096/8192/16384
输入长度下的 TTFT、TPOT、端到端时延和输出一致性。

## 实验方法

两组服务均使用 `--cuda-graph-max-bs 0`，避免 CUDA Graph 的 dummy batch 干扰首批校准，也
避免把图捕获差异混入 dtype 对照。每个长度使用并发 1、重复 2 次、输出 32 tokens；prompt
由 `benchmark/online/bench_l20.py` 按相同 seed 生成，禁用 EOS。

BF16 对照：

```text
--kv-cache-dtype bfloat16 --cuda-graph-max-bs 0 --max-prefill-length 20000
```

FP8 实验：

```text
--kv-cache-dtype float8_e4m3fn --kv-cache-scale 0.25 \
--kv-cache-calibration first_batch --cuda-graph-max-bs 0 \
--max-prefill-length 20000
```

完整原始结果：

- [BF16 4096](/data2/lszlsz/code/mini-sglang/learning/results/062_bf16_4096.md)
- [BF16 8192](/data2/lszlsz/code/mini-sglang/learning/results/062_bf16_8192.md)
- [BF16 16384](/data2/lszlsz/code/mini-sglang/learning/results/062_bf16_16384.md)
- [FP8 4096](/data2/lszlsz/code/mini-sglang/learning/results/062_fp8_4096.md)
- [FP8 8192](/data2/lszlsz/code/mini-sglang/learning/results/062_fp8_8192.md)
- [FP8 16384](/data2/lszlsz/code/mini-sglang/learning/results/062_fp8_16384.md)

## 结果

| 输入 tokens | dtype | 输出 tok/s | Avg TTFT ms | Avg TPOT ms | P90 TPOT ms | P90 E2E s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | BF16 KV | 14.89 | 1,072.22 | 34.74 | 36.68 | 2.149 |
| 4,096 | FP8 KV | 10.54 | 1,221.92 | 58.82 | 67.55 | 3.045 |
| 8,192 | BF16 KV | 13.79 | 1,170.17 | 37.15 | 42.68 | 2.322 |
| 8,192 | FP8 KV | 10.66 | 1,240.92 | 56.82 | 60.01 | 3.002 |
| 16,384 | BF16 KV | 8.00 | 2,749.22 | 41.26 | 55.37 | 4.028 |
| 16,384 | FP8 KV | 7.18 | 2,711.40 | 56.42 | 61.41 | 4.460 |

相对 BF16，FP8 的变化为：

| 输入 tokens | TTFT | TPOT | 输出吞吐 | E2E |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | +13.96% | +69.34% | -29.22% | +41.71% |
| 8,192 | +6.05% | +52.96% | -22.66% | +29.32% |
| 16,384 | -1.38% | +36.73% | -10.26% | +10.72% |

四个 rank 的 061d telemetry 在这三次长上下文请求累计统计 4,347,265,024 个写入元素，
其中 36,639 个达到饱和阈值，加权饱和率为 **0.000843%**。scale 范围为 0.01909--0.28195，
中位数为 0.05287。服务在三种长度下均返回 HTTP 200，未出现 FlashInfer 或 cache layout
错误。

## 输出一致性 smoke

使用同一个随机生成的 4096-token prompt（prompt SHA-256 为
`4bb87d5cb5f3ef7f3fa8c980f2e41c5a6b0d4aaf740372b56b02516b38bfc9ee`）、
`temperature=0`、`top_k=1`、32 output tokens 分别运行 BF16 和 FP8。两次输出的 SHA-256
均为 `f39a4c722ade4221678da1f09d32c3a776716a9dd51cf334913ebe7ba480565a`，该样本通过
exact-output smoke。它不是 perplexity 或 held-out quality gate，只说明本次确定性短样本没有
发生 token 分叉。

## 结论

1. **容量 gate 通过**：FP8 KV 仍提供约 2x KV capacity，长上下文请求可正常服务。
2. **校准/饱和 gate 通过**：长上下文累计饱和率低于 0.001%，未观察到明显极值截断。
3. **正确性 smoke 通过**：同 prompt、贪心采样下 4096-token 输出完全一致。
4. **当前性能 gate 不通过**：在 eager、并发 1 的实现中，FP8 decode 每 token 仍比 BF16
   慢 36.7%--69.3%；16K TTFT 已接近 BF16，但总吞吐仍低 10.3%。因此不能宣传 FP8 KV
   当前已经带来端到端加速，主要确定性收益仍是 capacity。

最可能的优化方向是把 BF16-to-FP8 cast 与 page store 合并，并检查 FlashInfer FP8 KV decode
kernel 的 load/dequant 路径；完成后需要重跑同一矩阵。下一实验进入 fused cast/store 的
CUDA kernel 原型和写入开销测量。
