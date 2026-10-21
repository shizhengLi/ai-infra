# DeepInfer 特性在 Mini-SGLang/L20（SM89）上的迁移分析与实验规划

日期：2026-10-21

## 1. 结论先行

DeepInfer 的 NVIDIA 完整运行时当前不能直接安装到 L20，也不能通过把编译参数从
`sm_100` 改成 `sm_89` 来完成迁移。更准确地说：

- DeepInfer 当前 NVIDIA 支持矩阵只有 SM100，依赖 CUDA 13、PyTorch CUDA 13.2、
  FlashAttention 4，以及 FlashMLA、DeepGEMM、DeepEP、DeepSelect 等配套库；
- 当前开源版只支持 DeepSeek-V4.1-Flash，不是可直接装载 Qwen3 的通用 engine；
- 开源版明确没有提供完整在线服务所需的 KV cache system、P/D disaggregation glue 和
  跨副本 load balancer；
- L20 是 SM89，本机每卡约 45 GiB，共 4 卡约 180 GiB 可见显存。DeepInfer 上游 SM90
  RFC 给出的 V4.1 FP4 checkpoint 约为 499 GB，单是权重就无法放进当前 4 卡 L20；
- SM89 支持 FP8 Tensor Core，但不支持 Blackwell 的 FP4 Tensor Core，也没有 SM100 的
  架构专用指令、共享内存规模和 kernel 生态。

因此，正确路线不是“移植 DeepInfer”，而是“借鉴 DeepInfer 的系统设计，在 Mini-SGLang
已有 Qwen3、FlashInfer、paged KV cache、PyNCCL 和 L20 kernel 基础上重新实现”。

最值得迁移的能力按优先级排序如下：

| 优先级 | DeepInfer 能力 | SM89 可行性 | 对 Mini-SGLang 的建议 |
| --- | --- | --- | --- |
| P0 | 投机解码的验证状态机与指标体系 | 高 | 先实现 Qwen3 双模型、greedy、固定 draft length |
| P0 | KV cache 的 speculative reserve/commit/rollback | 高，但工程量大 | 作为投机解码正确性的核心，不允许跳过 |
| P1 | 请求级确定性采样、batch invariant 验证 | 高 | 先加入 request seed/counter，再扩展随机投机采样 |
| P1 | Structured output | 高 | 用 xgrammar 编译 grammar，SM89 上重写/复用 logits mask |
| P1 | 异步调度方法论 | 中高 | Mini-SGLang 已有 overlap loop，先 profile 再补空洞 |
| P2 | confidence-scheduled variable verification | 中 | 固定长度获益后再做；不能直接使用 V4.1 Confidence Head |
| P3 | SWA bounded replay | 低 | Qwen3-32B 是全注意力主线，不能照搬模型专用近似 |
| P3 | Attention DP、Wide-EP、EPLB | 低 | 当前是 4 卡 PCIe TP4/Qwen3 dense，不匹配部署前提 |
| 拒绝 | NVFP4/FlashMLA/DeepGEMM SM100 kernel 直拷贝 | 不可行 | 保留现有 FP8/AWQ/FlashInfer 路线，按 SM89 重写必要算子 |

当前最有面试价值且能量化的主实验，是“双模型投机解码”：用 Qwen3-0.6B 作为 draft，
Qwen3-32B FP8 TP4 作为 target，在完全一致的 greedy 输出约束下，测 acceptance length、
target forward 次数、TPOT、吞吐和额外显存。DeepInfer 的 confidence scheduling 应作为第二阶段，
而不是第一版就照搬。

## 2. 调研范围与事实依据

本分析基于 2026-10-21 拉取的 DeepInfer `main`，commit：
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。

主要上游依据：

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)：支持模型、功能、硬件和依赖；
- [DeepInfer inference deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)：
  DSpark、异步调度、结构化输出、SWA replay 和大规模部署原理；
- [DeepInfer SM90 RFC #47](https://github.com/deepseek-ai/DeepInfer/pull/47)：连 SM90 都需要重新解决
  FP4 GEMM 和 attention kernel，当前仍是 Draft RFC；
- `deepinfer/model_executor/layers/sampler.py`：draft verification、residual sampling、
  structured-output mask 的组合流程；
- `deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py`：置信度累乘、
  全局排序和验证预算分配；
- `deepinfer/structure_output/`：xgrammar 编译、matcher 和 bitmask；
- [NVIDIA CUDA Compute Capability 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：
  SM89/SM100 的 Tensor Core 类型、TMA 和共享内存差异；
- [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/archive/13.0.0/pdf/Blackwell_Compatibility_Guide.pdf)：
  架构专用 target 不能向旧架构反向兼容。

本机实测环境：8 张 NVIDIA L20，compute capability 8.9，每卡 46,068 MiB；当前 CUDA
toolkit 为 12.8。本文只规划 4 卡 L20 的 Mini-SGLang 实验，不假设有 SM100。

## 3. 为什么 DeepInfer 本体不能直接迁到 SM89

### 3.1 软件支持矩阵不同

DeepInfer 的 NVIDIA prerequisites 是 CUDA 13，`requirements-nvidia.txt` 固定 PyTorch
CUDA 13.2 和 FlashAttention 4。Mini-SGLang 当前环境是 CUDA 12.8、PyTorch 2.9.1+cu128，
attention 主路径是 FlashInfer。

这不是简单的 Python 依赖版本冲突。DeepInfer 的模型执行依赖一组共同设计的低精度格式、
attention、MoE 和通信内核。即使 C++ scheduler 能在 x86 上编译，也不代表模型 forward
可以在 L20 上运行。

### 3.2 硬件能力不同

| 能力 | L20 / SM89 | DeepInfer NVIDIA 目标 / SM100 | 迁移影响 |
| --- | --- | --- | --- |
| FP8 Tensor Core | 支持 | 支持 | FP8 算法可保留，但 kernel 必须重新选择 |
| FP4 Tensor Core | 不支持 | 支持 | 不能运行原生 NVFP4/FP4 GEMM |
| TMA / thread-block cluster | 不支持 | 支持 | 依赖这些能力的流水和数据搬运必须重写 |
| 每 SM 最大 shared memory | 约 100 KB | 约 228 KB | SM100 tile/block 配置不能原样使用 |
| CUDA 架构专用代码 | `sm_89` | 可能为 `sm_100a`/SM100 专用 | 不能向旧 GPU JIT 或二进制兼容 |

DeepInfer 的部分 TileLang 小算子显式关闭了 TMA/warp-specialized lowering，说明其算法未必
天然依赖 Blackwell；但上游没有 SM89 的 build、correctness 或 benchmark 支持，而且
MTP variable-length kernel 中出现了 `T.pdl_sync()`。结论只能是“可以参考并重写”，不能说
“这些 kernel 已经兼容 L20”。

### 3.3 模型和容量前提不同

DeepInfer 当前只支持 DeepSeek-V4.1-Flash，许多特性与该模型的 DFlash draft head、
Confidence Head、MLA、SWA/YOCO、CED 和 MoE 结构绑定。我们的主模型是 Qwen3-32B dense：

- 没有 DeepInfer 所需的 DFlash/MTP/Confidence Head checkpoint；
- attention 是标准 Qwen3 GQA，不是 V4.1 的 MLA/indexer/SWA 组合；
- 4 卡 L20 无法容纳约 499 GB 的 V4.1 FP4 权重；
- 4 卡 PCIe 拓扑也不是 DeepInfer 推荐的 EP64--128 scale-up domain。

所以“让 DeepInfer 支持 L20”与“给 Mini-SGLang 增加 DeepInfer 风格能力”是两件完全不同的事。
前者需要模型、kernel、并行和容量的系统性重做；后者可以按功能逐项落地。

## 4. 第一优先级：投机解码

### 4.1 原理

普通自回归 decode 每次 target model 只产生一个 token。投机解码先让便宜的 draft model
连续提出 `K` 个候选，再让 target model 用一次多 token forward 并行验证。若平均每次能
接受多个 draft token，就能减少昂贵 target decode step 的数量。

```text
当前已确认前缀
    |
    +--> draft model 提议 d1, d2, ..., dK
    |
    +--> target model 一次验证 K 个位置
             |
             +--> 接受最长合法前缀 d1...dA
             +--> 在首个拒绝位置输出 target token
             +--> commit 已接受 KV，rollback 被拒绝 KV
             +--> 进入下一轮
```

对于 greedy decoding，验证规则可以先做成严格且容易测试的版本：从第一个位置开始，draft
token 等于 target argmax 就接受；遇到第一个不相等的位置，输出 target argmax 并停止本轮；
如果 `K` 个 draft 全部接受，还可以从最后一个 target logits 额外输出一个 token。

对于 temperature/top-k/top-p sampling，不能继续用 argmax 比较。要保持 target 分布不变，
标准 speculative sampling 对 draft token `x` 以
`min(1, q(x) / p(x))` 的概率接受；拒绝后从归一化的 `max(q-p, 0)` 残差分布采样。
DeepInfer 的 sampler 也保存 draft logprob、计算 accept/reject，并在拒绝位置做 residual sampling。

### 4.2 DeepInfer 的增量：confidence-scheduled verification

DeepInfer/DSpark 不只是固定 `K=5`。Confidence Head 预测每个 draft token 被 target 接受的
概率，再根据硬件 profile 得到的吞吐曲线，把有限验证 token budget 分给边际收益最高的
位置。

设请求 `i` 的第 `j` 个 draft token 的条件接受概率为 `a[i,j]`，验证到该位置能产生有效
前缀的近似概率为：

```text
prefix_prob[i,j] = product(a[i,0:j+1])
```

DeepInfer 的 GPU 调度 kernel 先做 cumulative product，再对所有请求/位置的 prefix probability
排序，在全局 verification budget 内选 top-k，最终得到每个请求不同的 `seqlen_q`。

这个设计的关键不是“永远多验证”，而是只验证预期收益大于计算成本的 token。DeepInfer
公布的数据中，动态长度把平均验证长度从 5 降到 3.2--4.4，平均接受长度基本不变；在其
EP64/DP64、约 60 sequences/rank 的生产点，估算吞吐提升约 4%--7%。这个数字不能直接套到
L20 TP4，因为模型、batch、互连和 kernel 完全不同。

### 4.3 Mini-SGLang 第一版应采用双模型，而不是 DSpark Head

当前 Qwen3-32B checkpoint 没有可直接加载的 DSpark draft/confidence head。最务实的实验是：

- target：Qwen3-32B，FP8 W8A8，TP4；
- draft：Qwen3-0.6B，相同 tokenizer，BF16 或 FP8；
- 第一阶段只支持 greedy 和固定 `K`；
- verification 先走 eager/ragged prefill 路径，不立刻做 CUDA Graph；
- 所有能力 default-disabled，baseline 路径不得改变。

draft 部署候选：

| 方案 | 优点 | 缺点 | 决策 |
| --- | --- | --- | --- |
| 每个 TP rank 完整复制 0.6B draft | 无 draft collective，提议 token 天然在各 rank 一致 | 每卡增加约 1--2 GiB 权重和独立 KV；需支持 local TP=1 model context | 首选最终方案 |
| draft 也做 TP4 | 权重占用最低，复用现有 TP loader | 小模型每层都走 PCIe collective，draft 很可能比收益更贵 | 仅作对照，不作为默认 |
| 只在 rank 0 放 draft，再广播 token | 实现概念简单 | rank 0 显存/计算不均衡，其他卡等待 | 可作早期原型 |
| 单独第 5 张 GPU 跑 draft | target 侧改动少 | 多用一张 GPU，资源公平性差，服务链路更复杂 | 只作功能验证 |
| CPU draft | 不占 GPU | 延迟通常无法隐藏 | 拒绝作为性能方案 |

每 rank 复制 draft 不是当前代码可以零改动完成的：Mini-SGLang 使用全局 TP state 和单例
`Context`，Qwen3 layer 构建时会按 TP4 分片。因此需要增加独立 `DraftRunner` 和局部 model
context，不能直接在现有 `Engine` 里再调用一次 `create_model()`。

### 4.4 Mini-SGLang 的具体改造点

#### 请求与 batch 状态

`python/minisgl/core.py`：

- `SamplingParams` 增加 speculative mode、draft length、draft model 等显式配置；
- `Req` 增加 `draft_token_ids`、draft logprobs、accepted count 和 speculative generation id；
- `Batch.phase` 增加 `verify`，或新增不会和普通 prefill 混淆的 `VerifyBatch`；
- `complete_one()`/`append_host()` 从固定单 token 扩展为 commit 多 token；
- EOS 出现在已接受 block 中间时，必须截断后续 token 和 KV。

#### Draft runner

建议新增 `python/minisgl/speculative/draft_runner.py`：

- 管理独立 draft model、draft KV cache 和 tokenizer compatibility check；
- 第一版顺序生成固定 `K` 个 token；
- 验证 target/draft 的 vocab size、special token ids、chat template 和 tokenizer hash；
- draft failure 可以回退普通 target decode，但不得丢失或重复 token。

#### Target verification

`python/minisgl/engine/engine.py` 和 `python/minisgl/engine/sample.py`：

- target 一次返回每个 verification position 的 logits，而不是每个请求只取一个 logits；
- `ForwardOutput` 返回每请求变长的 accepted tokens、accepted count 和 rollback length；
- greedy 先做 longest matching prefix；
- 随机版本保存 draft/target 概率并实现严格 residual sampling；
- request-scoped RNG 使用 `seq_seed + logical_sample_position`，不能依赖 batch 中的 row index。

#### FlashInfer attention

`python/minisgl/attention/fi.py` 已支持 prefill 的 ragged `seqlens_q`，可以作为 eager verification
的起点；现有 decode CUDA Graph wrapper 假设 `max_seqlen_q == 1`，不能直接用于多 token
verification。第一版应显式走 eager prefill wrapper，正确性和收益成立后再按 `(batch bucket,
draft length bucket)` 捕获 Graph。

#### KV cache 事务

`python/minisgl/scheduler/cache.py`、`table.py` 和 radix cache 是最关键的正确性边界：

1. reserve：为本轮最多 `K+1` 个 target 位置预留 page/token slot；
2. write：verification forward 可以写入临时 KV；
3. commit：只推进已接受前缀和最终 target token 对应的 logical length；
4. rollback：释放未接受尾部的新 page，清理/截断 page table；
5. publish：只有 commit 后的前缀才能进入 radix cache；
6. abort/retraction：请求在 in-flight verification 中取消时，事务必须只释放一次。

物理 KV 中保留不可见的旧字节不是问题，问题是 page table、sequence length 或 radix node
不能再引用这些字节。必须增加 transaction/fault-injection 单测，不能依靠“下一轮会覆盖”。

#### Scheduler 与 streaming

`python/minisgl/scheduler/scheduler.py` 当前 `_process_last_data()` 每请求只处理一个 token，
`DetokenizeMsg` 也只携带一个 token。投机解码需要：

- 一轮按顺序发出 1 到 `K+1` 个 token；
- 在第一个 EOS/stop token 处停止；
- 处理客户端 abort、backpressure 和半轮发送失败；
- telemetry 记录 drafted/verified/accepted/emitted tokens；
- 普通 decode、spec verify 和 prefill 的 admission 不能让某一类请求饥饿。

第一版可以连续发送多个现有 `DetokenizeMsg`，降低协议修改范围；性能稳定后再考虑 batch
token message。

### 4.5 为什么 L20 可能受益，也可能回退

L20 上小 batch decode 常受权重读取/显存带宽约束。target 一次验证多个 token，有机会用
更多计算换更少权重扫描，提升 arithmetic intensity，这是投机解码最有利的区域。

但以下情况会回退：

- draft acceptance 低，target 验证了大量最终被丢弃的 token；
- batch 已经很大，target GEMM 已接近计算饱和，额外 verification 直接增加成本；
- Qwen3-0.6B draft 自己的多步串行生成和 KV 管理太慢；
- TP4 verification 的 ragged metadata、collective 或 Graph miss 开销过高；
- draft model 占用显存，导致 target KV capacity/并发下降；
- rollback 频繁跨 page 边界，引入 allocator 和 radix cache 开销。

所以必须同时报告“每张卡有效输出 tok/s”和“总 GPU 数”。使用第 5 张卡的方案即使更快，
也不能和 4 卡 baseline 直接宣称等资源收益。

## 5. 第二阶段：置信度调度能否迁移

可以迁移算法，但不能直接迁移 DeepInfer 的 head 和 kernel。

### 5.1 缺失的模型能力

DeepInfer 的 Confidence Head 是 DeepSeek-V4.1 DFlash checkpoint 的一部分。Qwen3-0.6B
不会自动给出“将被 Qwen3-32B 接受的概率”。我们有三种选择：

1. 固定 `K`：最简单，也是第一阶段唯一应该采用的生产逻辑；
2. heuristic：用 draft top-1 probability、top-1/top-2 margin、entropy 和 token position
   预测 acceptance，必须用真实 target 结果做校准；
3. 训练 confidence head：采集 `(draft hidden, target accept)` 数据，训练轻量分类头，并在
   held-out prompt/domain 上验证 calibration error。

heuristic 只能叫“acceptance estimator”，不能在文档中包装成 DSpark Confidence Head。

### 5.2 SM89 kernel 路线

DeepInfer 的 variable-length kernel 使用 TileLang、cumulative product、全局 rank counting 和
`T.pdl_sync()`。SM89 版建议分三步：

1. CPU reference：纯 PyTorch/NumPy 计算每请求长度，验证 tie-break 和 budget invariant；
2. GPU prototype：用普通 PyTorch/CUDA 或 Triton 实现 cumprod + top-k，不依赖 PDL/TMA；
3. 只有 profiler 证明它在 critical path，再写 SM89 CUDA fused kernel。

在 L20 的常见 batch 1/8/32 下，CPU top-k 可能足够快，但 GPU 到 CPU 的 confidence 同步
可能破坏异步流水。因此先 profile，不能先假设 fused GPU scheduler 一定更好。

### 5.3 L20 自己的收益函数

不能使用 DeepInfer EP64 的 throughput curve。我们要离线拟合：

```text
target_cost(batch_size, total_verified_tokens, context_bucket)
draft_cost(batch_size, draft_length)
expected_emitted = 1 + sum(prefix_accept_probability)
objective = expected_emitted / (draft_cost + target_cost)
```

每个 context bucket、batch bucket 都可能选出不同验证预算。只有 adaptive policy 在 held-out
流量上优于最好的 fixed-K，才算 confidence scheduling 成功。

## 6. 其他值得迁移的特性

### 6.1 Structured output：高可行、适合独立 PR

DeepInfer 使用 xgrammar，把 JSON schema、regex 或 EBNF 编译为 grammar automaton；运行时
每个 decode step 根据 automaton 生成合法 token bitmask，再在 sampling 前屏蔽非法 logits。

这个设计与 SM100 无关。需要适配 SM89 的只是 mask apply kernel，可以先用 PyTorch
`masked_fill`/index 操作实现 correctness reference，再用 Triton、FlashInfer 已有能力或小型
CUDA kernel 优化。建议注意：

- grammar 编译必须异步，不能阻塞 scheduler；
- schema cache 必须有大小上限，避免不受控内存增长；
- matcher 状态属于 request，abort/retraction 后必须释放；
- 与 speculative decoding 组合时，要为每个 draft position 推演 grammar 状态，首个非法
  draft token 必须拒绝，不能只给最后一个位置加 mask；
- 测 JSON parse success、schema validation、普通请求 zero regression 和 mask 开销。

它对面试很有价值，因为同时涉及 automaton、异步编译、GPU logits mask、流式状态机和安全边界，
但它主要是功能/正确性能力，不应强行宣称提升普通文本吞吐。

### 6.2 请求级确定性采样与 batch invariance：高可行

当前 Mini-SGLang greedy 路径容易做确定性比较，但随机采样没有显式 request seed/counter。
可以借鉴 DeepInfer，把随机数定义为请求属性：

```text
random_key = hash(seq_seed, logical_sample_position, rejection_draw_index)
```

这样相同输入和 sampling params 在 batch composition、Graph padding 或调度顺序变化时仍可复现。
它也是 stochastic speculative decoding 的前置条件，否则 baseline 与 speculative 路径消耗
随机数的顺序不同，无法做严格分布/回归测试。

需要区分两个目标：

- deterministic sampling：固定 seed 时可复现；
- batch-invariant model result：不同 batch shape 下 logits/token 完全一致。

后者还受 GEMM reduction、attention backend、CUDA Graph 和浮点非结合性影响，难度更高，
不能因为 RNG 可复现就宣称 batch invariant。

### 6.3 异步调度：先验证现有实现，不重复造轮子

DeepInfer 把 `schedule(N+1)`、`forward/sample(N)` 和 `post-process(N-1)` 重叠。Mini-SGLang
现有 `Scheduler.overlap_loop()` 已经使用独立 stream，并保存上一轮 `ForwardData`，所以方向上
已经具备类似流水。

真正需要做的是用现有 pipeline telemetry/NVTX 回答：

- `copy_done.synchronize()` 前后 GPU 是否有空洞；
- 多 token verification 后 CPU post-process 是否重新进入 critical path；
- draft proposal 能否和上一轮 target verification 重叠；
- pinned D2H 只复制 accepted count/token，还是复制完整 verification tensor；
- Graph replay、metadata prepare 和 PyNCCL collective 是否破坏 overlap。

没有 timeline 证据前，不新增线程、stream 或 barrier。

### 6.4 SWA bounded replay：当前不迁移

DeepInfer 的 bounded replay 依赖 V4.1 的 SWA/YOCO/CED 语义，用约 128-token bounded suffix
近似恢复特定 state cache。Qwen3-32B 主线是 full attention，截断到最近窗口会直接改变模型
语义和输出质量。除非未来选择原生 sliding-window 模型，并证明 checkpoint 的 attention
语义匹配，否则该能力不进入实验路线。

### 6.5 Wide-EP/EPLB、Attention DP/CP：只借鉴 telemetry

DeepInfer 的收益建立在 DeepSeek MoE、EP64--128、scale-up domain 和高速互连上。当前主线是
4 卡 PCIe、Qwen3-32B dense、TP4；直接实现 Wide-EP 没有对象，通信成本也不同。

未来测试 Qwen3-MoE 时可以借鉴：每 expert token count、最慢 rank/平均 rank、热 expert
replica 和重排周期等 telemetry。但 DeepEP kernel 与 symm_mem 路线不应作为当前 L20 PR。

### 6.6 FP4 和 DeepInfer kernel 库：当前明确拒绝

SM89 没有 FP4 Tensor Core。软件解包 FP4 后转 FP8/BF16 GEMM 可能节省权重容量，但会增加
decode 热路径的解量化成本；这需要独立 kernel 和质量研究，不等价于 DeepInfer/Blackwell
原生 FP4。我们已经有 FP8 W8A8 和 AWQ-Marlin 结果，近期应继续使用已有 SM89 优化栈。

FlashMLA 面向 MLA，Qwen3 使用 GQA；DeepGEMM/DeepEP 也与当前模型和拓扑不匹配。因此不做
vendor submodule 移植，不把能否编译某个 microbenchmark 当成端到端迁移成功。

## 7. 推荐实验路线

实验编号接在当前 087 之后；原计划中的 088 upstream PR 拆分不取消，DeepInfer 路线从 089
开始，避免覆盖已有计划。

### 实验 089：speculative decoding CPU reference 与契约测试

目标：不加载 GPU 模型，先锁定 accept/reject、EOS、rollback length 和 streaming 顺序。

实现：

- greedy longest-prefix reference；
- stochastic `min(1, q/p)` 与 residual distribution reference；
- `SpeculativeResult` 数据结构；
- fixed seed property tests；
- EOS 位于第 1/中间/最后位置、全部接受、首 token 拒绝、abort 的边界测试。

验收：greedy 与逐 token target 100% token-identical；随机版本在小 vocab 枚举中保持 target
分布，概率和误差低于 `1e-6`。

### 实验 090：eager greedy 双模型最小闭环

目标：Qwen3-0.6B draft + Qwen3-32B target，`K=2`，单请求，无 Graph、无 radix publish。

验收：

- 128/512-token 输出与普通 target greedy 完全一致；
- 首拒绝、全接受和 EOS 都不出现重复/漏 token；
- KV logical length、page table 和输出长度完全一致；
- acceptance/rollback telemetry 可解释。

### 实验 091：L20 fixed-K 性能矩阵

固定 target 为当前 Qwen3-32B FP8 TP4 profile，比较 baseline 与 `K=1/2/4/6`：

| 维度 | 取值 |
| --- | --- |
| context | 256 / 4096 / 16384 |
| output | 128 / 512 |
| concurrency | 1 / 8 / 32 |
| repeats | 至少 3 个不同 seed |

必须报告：

- output tok/s、TTFT、平均/P90/P99 TPOT、E2E；
- drafted、verified、accepted、emitted tokens；
- average accepted length 和 acceptance rate；
- target forward steps/request；
- draft/target/verification/rollback 分段 GPU 时间；
- 每卡 peak memory、KV capacity 和 Graph hit rate；
- 4 卡等资源结果，不能混入第 5 张 draft GPU。

初始通过门槛：C=1 或 C=8 至少一个主 workload 吞吐提升 10% 或平均 TPOT 降低 10%；
C=32 吞吐不回退超过 3%；所有 greedy 输出完全一致；每卡 peak memory 不超过安全水位。

### 实验 092：KV transaction、radix cache 和取消恢复

目标：把 speculative reserve/commit/rollback 接入 page allocator 和 radix cache。

覆盖：page size 1/16/32、跨页部分接受、prefix hit 后验证、abort、retraction、OOM reserve
失败、重复 prefix request。验收是不泄漏 page、不发布 rejected prefix、不出现 stale result，
并通过 cache integrity check。

### 实验 093：Graph shape 与异步流水

在 091 证明 eager 有收益后再做：

- verification length bucket 1/2/4；
- batch bucket 与现有 L20 Graph policy 联合；
- draft(N+1)、target verify(N)、post-process(N-1) timeline；
- 比较完整 logits D2H 与只回传 accepted result。

如果 Graph shape 爆炸导致显存或 capture 时间不可接受，保留 verification eager，不为追求
Graph 覆盖率牺牲整体吞吐。

### 实验 094：adaptive verification

先用真实运行记录训练/校准 acceptance estimator，再比较：

- best fixed-K；
- per-request threshold；
- global token budget；
- context/batch-aware cost model。

验收必须在 held-out prompts 上优于实验 091 的 best fixed-K，且 calibration error、额外
scheduler 时间和 GPU/CPU 同步成本全部可见。否则固定 K 是更好的工程决策。

### 实验 095：structured output 独立 PR

实现 xgrammar async compile、request matcher、GPU/torch mask 和 OpenAI JSON-schema/regex/EBNF
入口。先与普通 decode 集成，再验证 speculative draft 的逐位置 grammar state。

验收：schema validation 100%，无 grammar 请求 zero regression，非法 schema 有明确错误，
cache 有容量上限，abort 后状态释放。

## 8. 必须避免的错误结论

1. 不能说“DeepInfer 支持 L20”。准确表述是“借鉴算法，在 Mini-SGLang 上为 SM89 重写”。
2. 不能把 `sm_100` 改成 `sm_89` 编译成功当作迁移成功；还需要模型正确性和端到端性能。
3. 不能把 DeepInfer 的 EP64 4%--7% 收益直接预测为 L20 TP4 收益。
4. 不能只报告 acceptance rate。acceptance 高，但 draft 或 verification 更贵，仍可能变慢。
5. 不能只比较 target step 数。应比较等 GPU 资源下的有效输出 tok/s 和 tail latency。
6. 不能在拒绝 draft 后保留可见 KV/radix prefix，否则会产生静默错误。
7. 不能在 stochastic sampling 中用简单 token equality 代替概率校正。
8. 不能把 heuristic confidence 称为模型 Confidence Head。
9. 不能为 Qwen3 开启 SWA bounded replay，除非模型语义和质量 gate 已被单独证明。
10. 不能直接复制 Apache-2.0 文件而删除原许可证和 attribution。优先按公开算法重新实现；
    如果复用代码，必须保留上游版权/许可证说明并审查依赖许可证。

## 9. 面试时如何讲

推荐表述：

> 我没有直接 fork 一个只支持 SM100 的框架并修改架构号，而是先拆分算法依赖和硬件依赖。
> DeepInfer 的 FP4、FlashMLA、DFlash head 和 Wide-EP 与 Blackwell/DeepSeek-V4.1 强绑定，
> 不能用于 L20/Qwen3；但 speculative verification、KV transaction、request-scoped RNG、
> confidence budget 和 async scheduling 是可迁移的系统设计。我们先在 SM89 上用 0.6B draft
> 加 32B target 做 lossless greedy fixed-K，要求 token 完全一致，并同时测 acceptance、target
> step、TPOT、吞吐、显存和 rollback；只有 fixed-K 端到端通过后，才训练 confidence estimator
> 和实现 adaptive verification。这样每层优化都有独立 correctness/performance gate。

可以继续追问的技术点：

- 为什么投机解码在 memory-bound 小 batch 更容易获益，在 compute-bound 大 batch 可能回退；
- greedy verification 与 stochastic residual sampling 的区别；
- 为什么 rejected token 已写入物理 KV 不是问题，但它仍被 page table/radix cache 引用是问题；
- 为什么 request-scoped counter RNG 是 batch invariance 和随机投机解码的前置条件；
- 为什么变长 verification 会造成 CUDA Graph shape explosion，如何用长度 bucket 控制；
- 为什么 confidence scheduling 要基于本机 throughput curve，而不是只看模型概率；
- 如何用等 GPU 数、held-out prompt 和 tail latency 避免不公平 benchmark。

## 10. 最终建议

近期不尝试编译或运行 DeepInfer 本体，也不迁移 Blackwell kernel。执行顺序建议为：

1. 完成原计划实验 088 的 upstream PR 拆分；
2. 实验 089 先提交投机验证 CPU reference、数据结构和边界测试；
3. 实验 090 实现 greedy、fixed-K、eager、单请求的 Qwen3 双模型闭环；
4. 只有 token/KV 正确性通过后，才上 4 卡 L20 做实验 091；
5. fixed-K 有真实收益后再做 KV/radix 完整集成、Graph 和 confidence scheduling；
6. structured output 可以作为与投机解码并行、风险更低的独立功能 PR。

这条路线既保留 DeepInfer 最有价值的思想，也尊重 L20/SM89、Qwen3 和 Mini-SGLang 当前
代码结构的真实边界。
