# 实验 081：长 decode 下 TP4 page size 敏感性

## 实验目的

实验 080 只生成 16 个 token，page 粒度对持续 decode 和 KV cache 增长的影响还不明显。
本实验把输出长度提高到 128 token，并把总 KV cache 容量固定为 1024 token，重新比较
`page_size=8/16/32`。对应配置为：

| page size | num pages | 总容量 |
| ---: | ---: | ---: |
| 8 | 128 | 1024 token |
| 16 | 64 | 1024 token |
| 32 | 32 | 1024 token |

固定容量是关键：如果 page 数和 page size 同时变化，就无法判断结果来自 cache 容量还是
page 粒度。本实验只改变 page 粒度，模型、TP、请求到达序列和 prefill/decode 策略保持一致。

## 配置

- GPU：L20，`CUDA_VISIBLE_DEVICES=2,3,6,7`
- 模型：Qwen3-32B-AWQ；TP4；dtype `float16`
- 通信：默认 PyNCCL；Gloo 只负责 CPU rank/NCCL UID bootstrap
- `max_seq_len_override=640`，可以容纳 17-token 输入和 128-token 输出
- 总 KV cache：1024 token；`memory_ratio=0.9`
- prefill：normal budget=64，decode-active budget=32，最大 prefill streak=1
- reservation：`token`
- CUDA Graph：coarse policy，batch shapes `[1,2,4,8]`
- workload：每轮 8 个请求，输入 17 token，输出 128 token，Poisson arrival rate=20 req/s
- 重复次数：3，每组共 24 个请求；TTFT/TPOT SLO 均为 120 s
- benchmark seed：8100071（后续重复使用 8110071、8120071）

原始结果：

- [`081_tp4_p8.md`](../results/081_tp4_p8.md)
- [`081_tp4_p16.md`](../results/081_tp4_p16.md)
- [`081_tp4_p32.md`](../results/081_tp4_p32.md)

## 结果

| page size | 吞吐 tok/s | 标准差 | P90 TTFT ms | P99 TPOT ms | P90 E2E s | GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 305.82 | 3.72 | 1439.01 | 38.81 | 2.915 | 82.82% |
| 16 | 294.03 | 5.55 | 1576.01 | 44.97 | 3.064 | 81.91% |
| 32 | 297.83 | 9.71 | 1520.06 | 33.98 | 2.992 | 81.77% |

相对 `page_size=16`：

- `page_size=8` 吞吐高 4.01%，P90 TTFT 低 8.69%，P99 TPOT 低 13.68%；
- `page_size=32` 吞吐高 1.29%，P90 TTFT 低 3.55%，P99 TPOT 低 24.42%；
- 三组均达到 TTFT/TPOT SLO 100%，没有 token gap 超过 1 秒。

长 decode 的结论与实验 080 不同：实验 080 的 16-token workload 中，`page_size=32` 的
TPOT P99 明显较差；输出增长到 128 token 后，`page_size=32` 反而取得最低的 TPOT P99。
这说明 page size 的收益取决于 decode batch 形态和请求持续时间，不能用短请求 benchmark
直接决定默认值。

## Telemetry 观察

| page size | trace 数 | decode capacity trimmed | deferred | retraction | budget-limited batches | decode batch size 分布 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 4 | 0 | 0 | 0 | 0 | 1/2/3/4/5/6 均出现 |
| 16 | 4 | 0 | 0 | 0 | 11 | 1/2/3/4/5/6 均出现 |
| 32 | 4 | 0 | 0 | 0 | 0 | 1/2/3/4/5 均出现 |

三组都没有命中 `decode_capacity_trimmed`、`decode_capacity_deferred` 或 retraction，说明
当前 1024-token 容量足以让 admission 先控制活跃请求集合。`page_size=16` 的 11 次
budget-limited 说明它在该请求轨迹中更频繁地受到 prefill budget 约束，但没有造成请求失败。
page size 改变了每个请求增长时的 page 对齐和调度批次组成，这可能解释了长 decode 下的 tail
差异；还需要更高并发和更长输出才能把这一推断与真正的 allocator 碎片区分开。

## 结论

1. 1024-token 固定容量下，三种 page size 都能稳定完成长 decode，24/24 请求成功，未出现
   allocator assertion 或容量裁剪。
2. 当前 workload 中 `page_size=8` 吞吐最高，`page_size=32` 的 TPOT P99 最低；但差异只有
   单一输入长度、单一 arrival rate 和三次重复，不能据此修改默认 page size。
3. 长输出改变了 page size 的相对排序，证明 page 粒度优化必须同时覆盖短请求和长 decode，
   并报告 TTFT、TPOT tail、吞吐和 batch/telemetry，而不是只看平均 tok/s。
4. 暂不修改默认值。下一步应增加输出 512 token 或更高到达率，使用更大总容量，观察真正
   命中 page pressure 时的内部碎片、等待轮数和 P99.9 延迟。

## 面试表达要点

这个实验可以说明两个工程方法：第一，比较内存管理参数时固定总 token 容量，保证变量可归因；
第二，短请求结论不能外推到长 decode。实验 080 中大 page 的尾延迟较差，但实验 081 在
128-token 输出下大 page 的 TPOT P99 反而最好，说明 scheduler batch shape、page 对齐和
持续 KV 增长共同决定性能。最终工程决策不是“选择一个看起来最快的 page”，而是保留基线，
补充压力矩阵和 telemetry 后再做默认参数变更。
