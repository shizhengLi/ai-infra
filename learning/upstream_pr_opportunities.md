# 基于上游 Issue/PR 的功能改进路线

更新时间：2026-10-18

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

## 2026-10 本地实现后的重新排序

上面的 PR 方向是调研时的候选，不代表当前已经具备 upstream-ready 证据。结合实验 069-084，
现在更适合提交的是“边界清楚、默认行为不变、测试可独立运行”的小 PR，而不是一次性提交
page、FP8 KV、Graph 和 tokenizer 的大合集。

### 已经有本地代码和证据的方向

| 方向 | 本地状态 | 证据 | upstream readiness |
| --- | --- | --- | --- |
| PyNCCL wheel runtime linker | 已实现绝对路径 + rpath | TP4 Qwen3-32B-AWQ 在 L20 初始化、Graph capture、API ready | 高；适合先提交 build/load regression test |
| decode admission safety guard | 已实现未来 page demand guard | 实验 071/072 无 allocator assertion；默认策略不激进 | 中高；补齐 scheduler focused tests 后可提交 |
| capacity round-robin / UID cursor | 已实现公平选择和动态 UID 维护 | 实验 073/074/075/076/084；动态集合 trace 通过 | 中高；真实 L20 pressure 仍需 085 |
| wait-round telemetry | 已实现字段和 trace 输出 | 实验 076/084 命中 12 trim + 1 deferred；077/082/083 保留 admission 负证据 | 高；属于低风险 observability PR |
| Req preallocated token buffer | 本地已验证 allocation/copy 明显下降 | 实验 060 无 token 错误，但 E2E 吞吐无收益 | 高；应诚实定位为 CPU overhead PR |
| page-aware reservation | 已实现 opt-in | 实验 078 有 trim 安全收益，079/083 无稳定性能收益，084 验证公平基础设施 | 中低；仍需 085 的真实策略 A/B |
| L20 Graph shape policy | 已实现 opt-in l20 policy | 实验 066 padding 15.65% -> 1.48%，低负载 tail 回退 | 中；需要更广矩阵和显存预算测试 |
| FP8 KV cache | 有 opt-in pool/fused/native 原型 | 实验 061-064 capacity 约 2x，但长上下文 gate 未通过 | 低；先完成 087，不应直接 PR 默认路径 |

## 推荐的 PR 拆分

### PR-1：PyNCCL wheel library discovery and link regression

**问题**：很多 Python CUDA wheel 只安装版本化 `libnccl.so.2`，而 `-lnccl` 只寻找
unversioned `libnccl.so`。在本地环境中这会让 TP4 scheduler 在 AOT link 阶段直接失败，用户
误以为必须安装系统 NCCL 或禁用 PyNCCL。

**本地方案**：在 `_load_nccl_module()` 前搜索 wheel runtime，使用绝对库路径、rpath 和
清晰的错误信息；保留环境变量/系统库 fallback。不要把 Gloo bootstrap 和 GPU collective
混为一谈：当前 Gloo 负责 CPU rank/NCCL UID，PyNCCL 负责实际 GPU TP collective。

**PR 验收**：

- 只有 `libnccl.so.2` 时 AOT module 可以编译并 `ffi.Module` 加载；
- 没有 NCCL runtime 时错误信息包含查找路径和安装建议；
- `tests/kernel/test_pynccl_link.py` 在系统库、wheel 库和缺失库三种场景下覆盖；
- 不改变 PyNCCL API 和 Gloo bootstrap；
- 附 TP4 L20 ready 日志，但不把单机日志当作性能提升。

这是最适合先投 upstream 的 PR：改动边界小、回归明确，并且解决的是实际安装阻塞。

### PR-2：capacity telemetry and decode wait-round observability

**问题**：没有 telemetry 时，用户无法区分“容量裁剪没收益”和“ admission 早已把 batch
限制住”。实验 077、082、083 中 capacity trim 都为 0，但 pending queue 和 batch size 已
经显示严重压力；如果只看吞吐，很容易得出错误结论。

**本地方案**：为 selected/deferred/trimmed 事件记录 request UID、requested/selected batch、
pending tokens/requests、wait rounds、Graph hit、budget-limited 和 retraction；schema 保持
向后兼容，默认只写轻量 trace，路径通过 CLI opt-in。

**PR 验收**：focused tests 覆盖 UID 完成/取消/新请求、字段默认值、JSONL 可读性和多进程
rank-0 输出；不改变 scheduler selection。实验 084 已证明 deterministic trim/deferred trace 能
填充这些字段，实验 085 还需补充真实 L20 pressure 的线上样本。

**面试价值**：这是一个很好的“可观测性先于优化”的故事，能解释为什么 082/083 没有被包装成
page-aware 性能收益。

### PR-3：amortized `Req` token append

对应上游 #143。实验 060 已表明预分配 buffer 可以消除 decode 热路径的重复 allocation 和
大量 CPU copy，但端到端吞吐没有明显变化。PR 说明应聚焦 CPU overhead 和长输出可扩展性：

- 初始化分配 capacity，append 只写入下一个位置；
- 所有消费者使用有效长度 view，不能把 capacity 当作有效 token；
- 覆盖 4096/8192 output、abort、截断、tokenizer 返回和 streaming 顺序；
- benchmark 同时报告 append wall-time、allocation 次数、scheduler CPU 和 E2E tok/s。

如果 E2E 没有收益，PR 描述必须写“减少 CPU 复制和分配”，不能写“提升 serving 吞吐”。

### PR-4：page-aware admission and capacity fairness

对应上游 #97 better estimation policy 和本地实验 071/073-076/078/083。这是最有面试价值、
但也最容易被质疑的主 PR。

**必须先完成的证据**：

1. 实验 084 已命中 `decode_capacity_trimmed/deferred`，并证明 wait-round/fairness cursor
   在动态 membership 下工作；
2. 实验 085 仍需在不同容量、输出长度和 arrival rate 下重复 token/page-aware；
3. page-aware 至少在一个真实压力区间改善等待公平性或 P99 TTFT，同时吞吐不低于 98%；
4. 其他区间没有安全回归，并且默认 token 路径完全不变。

PR 不能只带实验 078 的单个吞吐下降/TPOT 改善数字，也不能用 082/083 的 admission-serialized
结果证明公平性。需要把 page demand、retraction、UID cursor 和 allocator invariant 拆成
可审查的小提交。

### PR-5：FlashInfer page size > 1

对应上游 #110。当前本地已在 scheduler、Radix cache 和 metadata 层使用 page size 抽象，但
FlashInfer attention 路径仍需要验证 page stride、last-page length、page table 和 Graph
capture。实验 080/081 说明 page size 的性能排序与输出长度有关，因此不能只测单个 C=1 微基准。

建议 PR 先提交 metadata/正确性支持，再提交性能 policy：

- page size=16/32 的 block ids 与最后一页有效长度正确；
- BF16/FP8 KV、chunked prefill、eviction、Graph capture 和多请求 greedy 输出一致；
- 记录 page table H2D bytes、allocator calls、fragmentation、P99 TTFT/TPOT；
- 默认仍为 page size=1，直到跨 workload gate 通过。

### PR-6：L20 Graph shape policy

对应上游 #56/#160。实验 066 已有本地 l20 policy，但低负载 P99 TPOT 回退，因此 upstream
PR 应优先提交 shape selection、padding telemetry 和 `--cuda-graph-max-bs` 上限修复，而不是
声称 L20 一定更快。

验收包括 requested/padded batch、padding ratio、graph hit、capture 显存成本；需要检查
capture shape 不超过用户上限，并在 C=1/8/32、output16/128、arrival20/80 矩阵上不产生
低负载 tail 回退。

### PR-7：batch TokenizeManager

对应上游 #117/#55。实验 067 已证明 tokenizer CPU median 有价值，但 1 ms wait 没有稳定低
速率 TTFT 收益。PR 应聚焦批量 encode、chat template 分组和 UID 顺序恢复，不强制增加 wait
window。必须覆盖 `num_tokenizer=0/2/4`、streaming、abort、不同模板和高到达率。

### PR-8：FP8 KV cache v2

对应上游 #132，但目前不适合直接提交默认开启版本。L20 是 SM89，不能假设 SM90 的
FlashAttention FP8 KV 路径可用；还需要 calibrated scale、fused cast/store 和 long-context
quality gate。实验 061-064 的 capacity 结果足以证明方向值得研究，但性能退化和 scale 精度
仍是 blocker。实验 087 通过后，再拆成 dtype validation、scale metadata、kernel、quality
benchmark 四个 PR。

## 明确不作为近期 PR 的方向

- speculative decoding：需要 draft model、accept/reject、KV rollback 和 API 设计，当前
  主线没有质量/吞吐基线；
- hierarchical cache：上游本身标记为 PoC/WIP，控制面和 eviction 复杂度太高；
- expert parallelism：当前主模型是 dense Qwen3-32B，不是 MoE workload；
- 继续微调固定 prefill threshold：实验 056-058 已证明只改变 batch 切分，不能解决 queue
  pressure；
- 直接重写 GEMM/attention kernel：环境缺少稳定 ncu/CUTLASS 组合，尚无可归因 profile。

## upstream 提交顺序

1. PyNCCL linker + regression test；
2. telemetry/wait-round observability；
3. Req preallocated buffer；
4. page size > 1 metadata/attention correctness；
5. 实验 084/085 通过后提交 page-aware admission/fairness；
6. Graph shape policy、batch tokenizer、FP8 KV 按各自 gate 单独提交。

每个 PR 都应附：问题复现命令、默认行为、focused tests、至少三次 benchmark、Markdown/JSON
原始结果、失败候选和回滚方式。这样面试中可以清楚回答“为什么做、改了什么、收益在哪里、
为什么没有把所有实验结果都写成性能提升”。
