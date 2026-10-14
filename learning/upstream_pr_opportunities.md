# 基于上游 Issue/PR 的功能改进路线

更新时间：2026-10-14

## 结论先行

当前不建议继续做“换一个 prefill threshold”的实验。实验056--058已经说明，固定 token
budget 只能改变 batch 切分，不能解决 L20 上长 prompt 到达造成的队列峰值。下一阶段应把
调度从“估算 token 数”推进到“按真实 KV page admission”，同时补一个低风险的请求 token
buffer 优化作为可独立提交的 PR。

推荐顺序：

1. **PR-A：Page-aware admission + safe decode retraction**，主性能 PR，直接对应上游
   #97 的 better estimation policy，但针对本地 L20/FP8/overlap 配置重新设计和验证。
2. **PR-B：预分配 Req token buffer**，对应上游 #143 的 O(n) `torch.cat` 问题，改动小、
   容易证明正确，适合先提交。
3. **PR-C：L20 FP8 KV cache v2**，参考上游 #132，加入 L20 SM89 的 FlashInfer gate、
   calibrated scale 和 fused cast/store，目标是提高 KV 容量和 decode 带宽。
4. **PR-D：FlashInfer page size > 1 + page-size autotune**，参考上游 #110，解决当前
   `fi.py` 只支持 page size 1 的限制。
5. **PR-E：L20 CUDA Graph shape policy**，参考上游 #56/#160，做 memory-aware 的细粒度
   graph shape 选择和 padding telemetry，而不是只复制固定 shape 列表。
6. **PR-F：批量 TokenizeManager**，参考上游 #117/#55，在高请求速率下减少 CPU tokenizer
   开销；它是性能辅助 PR，不应替代 GPU 调度主线。

## 上游状态梳理

上游当前同时存在打开和关闭的 PR。打开的性能/功能候选包括 #97 better estimation policy、
#110 FlashInfer page size > 1、#117 batch tokenization、#132 FP8 KV cache、#134 vanilla
speculative decoding、#143 `Req.append_host`、#160 CUDA Graph 上限修复，以及 #154 radix
page-table 修复。上游 PR 列表显示这些改动仍处于开放状态，而 #113（TP rank decode order）、
#125（stream=false）、#130（benchmark credential）等已经合并关闭；#103（FlashInfer race）、
#88（GPU memory leak）、#98（LM head reshape）等关闭的修复说明上游很重视 correctness 和
request lifecycle，而不是只看 tokens/s。

上游未合并或标记为 WIP 的功能不能直接当作完成方案：例如 #82 hierarchical cache 明确写有
“可能永远不会合并”，#76 cache-aware prefill 仍是 WIP，#134 speculative decoding 是
standalone/offline 方向。我们的 PR 应保留清晰的默认关闭边界，并给出 L20 真实 workload 的
对照结果。

## PR-A：Page-aware admission + safe retraction

### 问题

本地 `PrefillAdder._try_allocate_one()` 用
`extend_len + output_len` 做粗略 reserve。它没有直接表达 page allocation，也无法根据
实际 decode KV 增长动态修正。实验057/058中即使缩小 prefill budget，最大 pending 仍是
8472 tokens，说明瓶颈在 admission，而不是 batch 大小。

### 设计

- 新增独立 `ReserveEstimator`，按 `page_size` 计算新请求需要的 KV pages。
- 维护 `new_token_ratio`：用历史请求实际新增 token/page 更新估计，空闲时衰减到初始值。
- Prefill admission 只接受不会超过可用 page 的请求；decode batch 组装前再次检查剩余 page。
- 极端情况下撤回最晚加入的 decode request，安全地重新放回 pending queue。
- overlap scheduling 中给 request 增加 `retracted`/generation 标记，避免延迟 GPU 结果覆盖已
  重排的请求。
- 默认参数保持旧行为；先用 `--schedule-estimate-policy page-aware` opt-in。

### L20 验收

| 指标 | 目标 |
| --- | --- |
| 1.5 req/s 混合 256/1024/4096 prompt P90 TTFT | 降低至少 15% |
| 输出吞吐 | 不低于 baseline 的 98% |
| TPOT SLO | >= 99% |
| 最大 pending prefill tokens | 降低至少 20% |
| correctness | greedy 输出与 baseline 一致；无 stale page/table、无 illegal memory access |

实验必须包含短 prompt、长 prompt、shared-prefix、unique-prefix 和长输出五类 workload。面试
时可以讲清楚：不是“把 budget 调小”，而是把 admission 从 token heuristic 改成 page-aware
closed-loop controller，并用 decode retraction 保证内存安全。

上游 #97 已公开了相同方向的动机和结果：它把 admission 改为 page-aware，引入
`new_token_ratio` 和 decode retraction，并在 Qwen3-0.6B WildChat replay 报告了 15.71% 的
offline throughput 提升；但它的硬件是 RTX 4090，不能直接当作 L20 结果，必须重新测量。

## PR-B：预分配 Req token buffer

### 问题与实现

当前 `Req.append_host()` 每生成一个 token 都执行 `torch.cat`，长输出会反复分配和复制已有
CPU tensor，复杂度接近 O(output_len^2)。利用 `max_device_len` 已知这一事实：初始化时分配
一次 `torch.empty(max_device_len)`，把 prompt 写入前缀；append 只在下一个位置赋值，并返回
当前长度 view。需要检查所有 `input_ids` 切片、cache insert 和 tokenizer 输出路径，避免把
capacity view 当成有效长度。

### 验收

- 单请求 4096/8192 output 的 host append CPU 时间降低至少 50%。
- Qwen3-32B FP8 TP4 长输出 workload 总吞吐提高 3--8%，或在无吞吐变化时降低 scheduler
  CPU 占用。
- 生成 token 与 baseline 完全一致，`torch.cat` 在 decode 热路径中计数为 0。
- 对 `Req` 增加 capacity/length 单元测试和 aborted request 测试。

这是最适合作为第一份独立 PR 的低风险改动。上游 #143 已指出同一个 O(n) re-allocation
问题，但我们可以补充 L20 长输出 benchmark、分配次数 telemetry 和严格的 view 边界测试。

## PR-C：L20 FP8 KV cache v2

### 设计

参考上游 #132 的 opt-in FP8 KV pool，但不直接复制其 scale=1 的 v1：

- `kv_dtype=float8_e4m3fn` 只允许 FlashInfer；L20 是 SM89，FlashAttention 的 FP8 KV
  路径需要 SM90+，必须在启动时明确报错而不是静默产生错误结果。
- 每层或每 head 保存 `k_scale/v_scale`，使用 calibrated max/percentile scale，避免统一
  clamp 带来的质量损失。
- 融合 clamp + cast + scatter，消除每层 K/V 两次额外 kernel launch。
- decode 读取时把 scale 传入 FlashInfer；embedding、LM head、collective 仍保持 BF16。

### 量化目标

KV storage 理论上从 BF16 变为 FP8，单 token 字节数减半。L20 Qwen3-32B FP8 TP4 当前约
16.5 GiB/card 的采样显存，应重点测：可用 pages、最大 context、并发上限和 decode ITL，
而不是宣称整卡显存直接减半。

### 验收

- 可用 KV pages/最大并发提升至少 70%。
- 长上下文 decode P99 TPOT 改善 5--20%，短 prompt TTFT 回退不超过 5%。
- 8k/16k context greedy token agreement、perplexity 或至少固定问答集质量不回退超过 1%。
- BF16 KV 默认路径 zero regression；unsupported backend 有明确错误。

上游 #132 报告了 2x cache capacity 和 5--27% median ITL 改善，但也诚实报告了 4--12%
TTFT regression、未校准 scale 和 SM90 FlashAttention 限制；这正好给我们的 L20 v2 提供了
清晰的改进点。

## PR-D：FlashInfer page size > 1

当前本地 `attention/fi.py` 仍断言 `page_size == 1`，虽然 radix cache、MHA pool 和其他
attention backend 已经有 page-size 抽象。参考上游 #110：移除 flatten，按 page stride 构造
block ids，并在 CUDA Graph capture 中保存最后一页有效长度。

L20 实验建议 `page_size=16/32`，保持总 KV token capacity 不变，比较：

- page table host-to-device bytes；
- scheduler metadata CPU time；
- radix match/eviction fragmentation；
- prefill/decode throughput、P90 TTFT、P99 TPOT；
- page size 与 FP8 KV 的组合效果。

这项 PR 的价值不是保证 tokens/s 一定上涨，而是减少 page-table 元数据和 allocator 操作，
并为以后 FP8 KV/HiCache 铺平接口。验收首要条件是多请求、chunked prefill、eviction、CUDA
Graph capture 下 greedy 输出完全一致。

## PR-E：L20 CUDA Graph shape policy

当前 graph shape 默认是 `[1, 2, 4, 8, 16, 24, 32]`（max batch 32）。上游 #56 提出小 batch
细粒度 capture、二分查找和 padding waste 指标；#160 又修复了不应超过
`--cuda-graph-max-bs` 的 capture 行为。

我们的新意应是 L20 memory-aware policy，而不是简单复制列表：

- `--cuda-graph-shape-policy coarse|l20|explicit`；L20 profile 默认捕获
  `[1..16] + [20,24,28,32]`，超出显存预算自动回退 coarse。
- 记录 `requested_bs/padded_bs/padding_ratio/graph_hit`。
- 用 request batch histogram 决定是否值得增加某个 shape；每增加一个 graph 都记录显存成本。

### 验收

- 平均 padding waste 降低至少 30%，小 batch 3/5/7 的 waste 为 0%。
- Graph capture 额外显存不超过 1 GiB/card。
- Qwen3-32B FP8 TP4 mixed workload throughput 提高 3% 以上，或 P90 TTFT 不回退。
- capture/replay 结果与 graph-disabled baseline 一致。

## PR-F：批量 TokenizeManager

本地 `tokenizer/tokenize.py` 明确留下 `TODO: batch tokenization`，每条消息单独调用
`encode()` 或 `apply_chat_template()`。参考上游 #117/#55，按 plain text 和 chat template
分组批量 tokenize，去除 padding，并按 uid 恢复原顺序。

验收不只看 tokenizer microbenchmark：

- 32 条混合请求的 tokenizer wall time 降低至少 2x；
- 1.5/2.5 req/s 时 TTFT P90 降低至少 5%；
- `num_tokenizer=0/2/4` 下输出顺序、abort 和 streaming 行为一致。

## 不建议现在做的方向

- **Speculative decoding (#134)**：收益可能很大，但需要 draft model、accept/reject、KV
  回滚和 API 设计，先不放进当前 L20 主线；可以作为单独 offline demo。
- **Expert parallelism (#96)**：对 Qwen3-32B dense 主模型没有直接收益，只有切换到
  Qwen3 MoE/30B-A3B 时才值得做。
- **Hierarchical cache (#82)**：上游自己标记为 PoC/可能不合并，且需要 host/NVMe cache
  控制面；先完成 page-aware admission 和 page size 抽象再评估。
- **继续调固定压力阈值**：实验058已证明这类改动不足以降低队列峰值。

## 面试叙事

建议最终形成四个互相独立的故事：

1. **Admission**：从 token heuristic 到 page-aware estimator + safe retraction，解决 TTFT
   tail 和 KV OOM 风险。
2. **Memory**：FP8 weight + FP8 KV，让 L20 在相同显存下承载更长上下文/更高并发，并用
   quality gate 证明不是单纯牺牲精度。
3. **GPU execution**：L20-specific graph shape policy 减少 padding waste，解释显存成本和
   replay hit rate。
4. **CPU path**：预分配 token buffer + batch tokenization，降低长输出和高到达率下的 CPU
   开销。

每个故事都必须同时报告 throughput、P90/P99 TTFT、P99 TPOT、显存、正确性和失败的候选，
这样面试时讲的是完整的 bottleneck -> hypothesis -> implementation -> measurement ->
decision，而不是“调参后某个数字变好”。

## 参考上游

- [mini-sglang Issues](https://github.com/sgl-project/mini-sglang/issues)
- [mini-sglang Pull Requests](https://github.com/sgl-project/mini-sglang/pulls)
- [#56 CUDA Graph batch size/padding](https://github.com/sgl-project/mini-sglang/pull/56)
- [#82 hierarchical cache](https://github.com/sgl-project/mini-sglang/pull/82)
- [#97 better estimation policy](https://github.com/sgl-project/mini-sglang/pull/97)
- [#110 FlashInfer page size > 1](https://github.com/sgl-project/mini-sglang/pull/110)
- [#117 batch tokenization](https://github.com/sgl-project/mini-sglang/pull/117)
- [#132 FP8 KV cache](https://github.com/sgl-project/mini-sglang/pull/132)
- [#134 speculative decoding](https://github.com/sgl-project/mini-sglang/pull/134)
- [#143 amortized Req token append](https://github.com/sgl-project/mini-sglang/pull/143)
- [#160 CUDA Graph max-batch fix](https://github.com/sgl-project/mini-sglang/pull/160)
