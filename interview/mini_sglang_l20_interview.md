# Mini-SGLang + NVIDIA L20 面试准备材料

这份材料用于系统性回答“你做了什么、为什么做、怎么证明、哪些失败了、下一步怎么做”。所有数字都来自仓库中的实验记录；回答时应同时说明实验条件，不要只报一个孤立的提升百分比。

## 一、30 秒项目介绍

我基于 Mini-SGLang 做了一个面向 NVIDIA L20 的大模型推理优化项目，主要模型是 Qwen3-32B。我把请求路径拆成 Tokenizer、Scheduler、Prefill/Decode、Attention、KV Cache、Tensor Parallel 通信和采样几个阶段，用固定 seed、开放环请求和 CUDA/Nsight telemetry 做瓶颈归因。

在结果上，我验证了 L20 PCIe 环境下 TP4 比 TP8 更适合当前服务效率，FP8 W8A8 是比 BF16 更有价值的低精度候选；FP8 KV Cache 把 KV capacity 提高到约 2 倍，但早期 decode 性能有回退，所以继续做了 fused cast/store 和 SM89 原生 FP8 转换。最终我把方案分成三类：通过性能和质量 gate 的候选、只适合显式 opt-in 的实验路径，以及因为尾延迟或数值风险应当拒绝默认启用的方案。

## 二、3 分钟完整回答模板

### 1. 背景

LLM 在线推理不是单一 GEMM benchmark。用户体验取决于首 token 时间 TTFT 和生成阶段每个 token 的间隔 TPOT；服务容量还取决于 KV Cache、权重显存和多卡通信。L20 有 48 GiB 显存、SM89 FP8 能力，但多卡主要通过 PCIe，因此适合观察计算、显存和通信的真实权衡。

### 2. 方法

我先建立 Qwen3-32B TP4 baseline，然后一次只改变一个主要因素。每组实验固定模型、GPU、TP、dtype、attention backend、输入输出长度、seed 和重复次数，并记录平均值、标准差、P90/P99.9、GPU 利用率和峰值显存。对于调度问题，固定并发不够，我进一步使用开放环到达率；对于 kernel 问题，则使用 CUDA event 和 Nsight 做局部归因。

### 3. 主要结果

- TP scaling 显示 TP4 是当前 L20 配置的效率平衡点，TP8 虽然吞吐更高，但 PCIe collective 让并行效率下降。
- FP8 W8A8 让 Qwen3-32B TP4 吞吐提高约 47%–55%，显存/card 下降约 15.48%，MMLU 下降约 0.70 分，是当前低精度服务候选。
- INT4 AWQ-Marlin 降显存和 GPU 数量的效果明显，但质量和长 prompt TTFT gate 没通过，不能只因为“能塞进一张卡”就宣布它更好。
- FP8 KV Cache 把 capacity 从 413,820 提高到 827,641 tokens，但初版 eager decode TPOT 比 BF16 慢 36.73%–69.34%。因此我没有把容量收益误写成端到端加速。
- fused cast/store 在同代码对照中把 4K/8K/16K TPOT 降低 19.47%/14.58%/29.02%，但纯 CUDA event 写入 kernel 仍慢 13.6%–60.7%，所以保持 opt-in。
- native pair conversion 使用 SM89 原生 FP8 转换，部分 row size 变快，但真实 scale 先舍入为 BF16/FP16，BF16 byte mismatch 约 3.65%，因此不能默认启用。

### 4. 工程结论

最重要的结论不是“某次跑得快”，而是建立了部署决策：FP8 TP4 可以作为当前研究 profile；fused store 可以作为需要进一步验证的 opt-in；native pair 必须先解决 scale 精度问题；所有通用默认值保持保守，避免把只在某个 seed、某个 batch 或某个 microbenchmark 下有效的方案带入生产。

## 三、系统原理

### 3.1 Prefill 和 Decode 的差异

Prefill 一次处理整段输入，矩阵乘和 attention 的并行度高，但会产生大量 KV；Decode 每次通常只处理一个新 token，计算量小，却必须读取历史 KV，容易受显存带宽、kernel launch 和多卡同步影响。两者共用 GPU 资源时，长 Prefill 可能阻塞 Decode，表现为平均吞吐正常但 TPOT P99.9 突然变大。

因此实验不能只看平均 tokens/s：

- TTFT：请求提交到第一个输出 token 的时间，反映排队和 Prefill；
- TPOT：输出 token 之间的时间，反映 Decode 稳定性；
- P90/P99.9：尾部用户体验，能暴露调度阻塞；
- E2E：TTFT 加生成时间，反映完整请求体验。

### 3.2 Continuous Batching

Scheduler 在每一轮把处于 Prefill 或 Decode 状态的请求组成 batch。Decode 请求希望频繁执行，长 Prefill 则希望一次处理更多 token 来提高吞吐。两者冲突时，不能只把 Prefill batch 调小：这可能降低单批效率、增加调度次数，甚至让吞吐下降而尾延迟没有改善。

本项目的经验是：先用 pending token pressure、active decode 数、Prefill streak 和请求到达率解释问题，再设计动态策略。实验 026–030 说明没有稳定的请求身份/热度信息时，过于激进的通用 eviction heuristic 会丢失前缀命中，不能仅凭减少 eviction 次数就接受。

### 3.3 Radix Cache

Radix Cache 按 token 前缀组织 KV 节点。相同系统 prompt、多轮对话或 RAG 前缀可以复用已有 KV，减少 Prefill 计算。命中率必须按 workload 解释：唯一 prompt、共享 system prompt 和真实多轮对话的命中行为完全不同。

淘汰策略要同时考虑：释放多少 token、是否误删热前缀、一次淘汰的放大倍数、淘汰调用频率和命中后的 TTFT。实验 021–030 中，简单 LRU、年龄、最近命中和衰减热度都不能作为无条件通用策略；page-aligned partial eviction 在固定 profile 有价值，但也保持 opt-in。

### 3.4 CUDA Graph

CUDA Graph 可以把固定形状的 GPU 操作预捕获，减少每轮 Python/C++ launch overhead。问题是真实 batch size 往往不是图形状，系统需要 padding 或选择相邻图形状。padding 会浪费计算和显存，图捕获又会消耗 workspace。

因此 graph 优化要同时记录：图命中率、逻辑 batch、实际 padded batch、capture 显存、TTFT、TPOT 和尾延迟。只看 Graph enabled 的平均吞吐可能掩盖 padded 32 的尾延迟问题；本项目多次验证了局部 barrier 或 eager fallback 可能改善某个转场，却使 C=32 P90 TPOT 变差。

## 四、Tensor Parallel 与 L20 PCIe 通信

### 4.1 为什么 TP 不是卡越多越快

Tensor Parallel 把层内矩阵或 attention head 分到多张 GPU。每层计算后通常需要 all-reduce 或 all-gather 同步 partial result。增加 TP 后单卡 GEMM 变小，但 collective 的消息、同步和 launch 成本不会按相同比例下降。

在 NVLink 机器上，这个代价可能相对较低；L20 多卡主要走 PCIe，通信更容易成为一阶瓶颈。因此应该报告并行效率：

```text
并行效率(TP=n) = 吞吐(TP=n) / (n × 吞吐(TP=1参考))
```

实际比较时还要说明显存配置是否完全一致、TP1 是否能容纳模型，以及请求并发是否改变。

### 4.2 PyNCCL 实验怎么回答

我先用 NVTX/Nsight 确认 collective 在 GPU kernel 时间中的占比，再比较 direct PyNCCL、buffer 配置和 Graph 状态。实验 032–034 证明 graph-disabled 的 direct path 在 TP4 有约 9%–12% 的在线吞吐收益，TP8 也有收益，但 Graph 模式的严格尾延迟 gate 没通过。

面试时要强调：

1. 不是看到 NCCL 就直接改通信参数；
2. 先确认通信确实占时间，并区分通信时间和调度等待；
3. 只接受在目标 workload 上同时通过吞吐、P90/P99 和正确性 gate 的配置；
4. `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` 是当前机器和 profile 的显式配置，不是所有环境的默认答案。

### 4.3 为什么 TP4 是推荐点

实验 047 显示 TP4 在当前 Qwen3-32B L20 profile 上有较好的效率；TP8 的绝对吞吐可以继续提高，但并行效率下降，说明每增加一张卡，节省的计算时间越来越难抵消 PCIe collective。实际部署应根据成本、吞吐目标和单请求延迟选择 TP4 或 TP8，而不是简单追求卡数最大。

## 五、量化：FP8 与 INT4

### 5.1 FP8 W8A8

W8A8 表示权重和 activation 使用 8-bit 浮点表示，通常需要 scale 把原始值映射到 FP8 范围。Ada SM89 有 FP8 Tensor Core，能让 GEMM 走高吞吐低精度路径。

优化时必须把三个误差分开：

1. 量化误差：权重或 activation 从 BF16/FP16 映射到 FP8 后的数值误差；
2. kernel 错误：scale、layout、dtype metadata 或反量化实现错误；
3. 系统开销：动态量化、scale 加载、转换和通信是否抵消 GEMM 收益。

实验 053 的结果是 TP4 吞吐提升约 47%–55%、显存/card 下降约 15.48%，MMLU 下降约 0.70 分。这个结论必须完整表述为“在指定模型、TP、workload 和质量评测下的候选”，不能说 FP8 永远比 BF16 好。

### 5.2 INT4 AWQ-Marlin

AWQ 是权重-only 量化，常见形式是 W4A16：权重为 INT4，activation 仍保持 BF16/FP16，kernel 中按 group scale 反量化再计算。它通常比 FP8 更省权重显存，适合容量优先场景，但不会自动降低 BF16 activation 或 collective 的消息大小。

实验 054 说明 INT4 能让 Qwen3-32B 减少 GPU 数量，但质量和长 prompt TTFT gate 未通过。正确的面试回答是：AWQ 的优点是容量和成本，FP8 的优点是 Ada Tensor Core 上的计算吞吐；二者目标不同，不能只比较一个 microbenchmark。

## 六、FP8 KV Cache 深入回答

### 6.1 为什么 KV Cache 占显存

Transformer 每层会保存历史 token 的 K/V。粗略地说，KV 显存随以下因素线性增长：

```text
tokens × layers × 2(K/V) × local_kv_heads × head_dim × dtype_bytes
```

Qwen3-32B 的长上下文服务中，KV Cache 可能比单次 activation 更长期地占用显存。将 BF16 KV 存为 FP8 可以把每个元素从 2 bytes 降到 1 byte，理论上接近 2 倍容量，但 attention 读取时需要使用 scale 反量化，因此 capacity 收益不等于端到端速度收益。

### 6.2 scale 和 calibration

本项目支持固定 scale 和 `first_batch` per-layer calibration。对于 E4M3，最大有限值约为 448；校准时用首个 eager Prefill 的 K/V max-abs 估计每层 scale，并记录 saturation telemetry。

需要注意：

- scale 太小会让大值饱和，损失信息；
- scale 太大则浪费 FP8 的有效动态范围；
- 每层分布不同，单个全局 scale 可能不如 per-layer scale；
- CUDA Graph capture 阶段不能执行会破坏 capture 的 `.item()`，所以校准应放在 eager 路径；
- FlashInfer 的 query dtype、KV dtype、K/V scale metadata 必须匹配。

实验 061/062 的长上下文累计 saturation fraction 为 0.000843%，但 FP8 eager decode 相对 BF16 TPOT 仍慢 36.73%–69.34%。这说明低显存表示和低延迟表示是两个独立目标。

### 6.3 fused cast/store

legacy 路径大致是：

```text
BF16/FP16 K,V -> float 除以 scale -> clamp 到 FP8 范围
             -> PyTorch cast 到 FP8 -> indexed page store
```

实验 063 把 scale、clamp、FP8 conversion 和 indexed store 放入一个 CUDA kernel，每个 warp 处理一个 token row，每个 lane 处理 4 个连续元素。它减少了中间 tensor 和额外 kernel，但 fused kernel 自己承担了转换和写入成本。

因此出现了一个很有面试价值的现象：端到端 TPOT 有改善，但纯 CUDA event microbenchmark 仍比 legacy 慢。原因可能包括 legacy PyTorch kernel 已经高度向量化、fused kernel 的 row 映射和 occupancy 不理想，以及端到端收益来自减少中间 tensor/同步而不只是写入 kernel。

### 6.4 native pair conversion 的数值风险

实验 064 使用 `__half2`/`__nv_bfloat162` 做 pairwise scale，再调用 `__nv_fp8x4_e4m3`。当 scale 是 0.125、0.03125 这类可被 BF16/FP16 精确表示的值时，输出 byte 与 float fused 一致；但 `scale=0.12345` 时，native 先把 scale 舍入为 BF16/FP16，导致 BF16 K 输出约 3.65% byte mismatch。

这不是“测试差一点”这么简单：KV cache 的 byte 差异可能改变 attention logits，进而改变长上下文生成。即使一次短请求 output hash 相同，也不能替代长上下文质量 gate。因此当前 native pair 只保留为显式研究开关，后续要么保留 FP32 scale 做 pair conversion，要么增加 scale 可表示性和误差门禁。

## 七、失败实验如何回答

### 问：为什么不直接把 CUDA Graph batch size 调到最大？

答：更大的图形状可能减少 launch overhead，但会增加 capture workspace、padding 计算和显存压力。telemetry 显示 padded 24/32 形状集中出现较长 model interval；某些 eager fallback 或 barrier 虽然改变了转场顺序，却使 C=32 P90 TPOT 回退。因此 Graph 需要按真实 batch 分布选择形状，不能只看理论最大 batch。

### 问：为什么动态 Prefill budget 没有直接设成越小越好？

答：Prefill batch 越小，Decode 被阻塞的时间可能下降，但 kernel 利用率和总吞吐也会下降。实验中固定较小 budget 没有同时改善 TTFT SLO 和吞吐；后续基于 pending token pressure 的策略在特定开放环负载有帮助，但没有可靠的通用阈值，所以保持 profile-specific opt-in。

### 问：为什么减少 eviction 次数的 cache 策略也被拒绝？

答：eviction 次数不是唯一目标。一次淘汰可能释放很多 token，但也可能删掉马上复用的热前缀。实验 026–030 中 raw adaptive、age-aware、recent-match 和 decayed-hotness 都至少丢失一个 paired survivor prefix。没有 workload identity 或可靠 admission 信号时，不能把启发式策略设为通用默认。

### 问：native pair 明明更快，为什么不默认？

答：它的速度收益依赖 row size，而且真实 calibration scale 下 BF16 byte mismatch 约 3.65%。推理系统的 correctness gate 高于局部 kernel 速度；要默认启用，必须先解决 scale 精度，完成长上下文输出和质量验证。

## 八、常见追问与标准回答

### 问：你怎么证明优化不是 benchmark 噪声？

答：固定 GPU、模型、seed、输入输出长度和服务参数；至少重复多次；报告均值、标准差和 P90/P99.9；区分 warmup、JIT、Graph capture 与 steady state；必要时用 held-out seed 重跑。对端到端结果不直接归因，先用 NVTX/Nsight 或 CUDA event 做 attribution。

### 问：为什么平均值不够？

答：在线服务通常有 SLO，用户感受到的是尾部请求。长 Prefill 可能只影响少数请求，平均吞吐不变但 P99 TPOT 变差，所以必须同时报告 TTFT、TPOT 和尾延迟。

### 问：Telemetry 在项目中具体做什么？

答：Telemetry 是运行时记录的结构化观测。例如 FP8 KV summary 会按 rank、layer 记录 scale、写入元素和饱和元素；调度 telemetry 会记录 pending token、active decode、Prefill batch 和 completion 间隔。它把“感觉像排队”转成可验证的因果证据，但 telemetry 本身不是优化，不能把相关性直接当成因果性。

### 问：L20 支持量化吗？

答：支持。SM89 支持 FP8 Tensor Core，适合 FP8 W8A8；INT4 可以通过 AWQ-Marlin 等 weight-only kernel 运行。区别在于 FP8 更偏计算吞吐，INT4 更偏权重容量；两者都必须单独验证质量、scale、kernel layout、显存和在线尾延迟。

### 问：为什么 FP8 KV capacity 2 倍，但速度不一定更快？

答：存储字节数减少了，但 Decode 仍有 FP8 KV 的读取、反量化、scale metadata 和 attention kernel 路径。若转换或 decode kernel 不是高效实现，节省显存不会自动变成节省时间。实验 062 正好证明了 capacity gate 通过、TPOT gate 失败。

### 问：如果让你继续做，你会做什么？

答：第一，改进 native pair，使 scale 保持 FP32 精度并重新做 byte/质量 gate；第二，进入 FlashInfer page size >1，减少 page table 和 metadata 开销；第三，分析 Graph shape/padding 和 attention decode kernel 的真实 batch 分布；第四，在固定的 TP4 profile 上做并发和开放环的 held-out 确认，而不是继续无目的扫参数。

## 九、代码定位

| 模块 | 位置 | 说明 |
| --- | --- | --- |
| 参数解析 | `python/minisgl/server/args.py` | dtype、量化、KV Cache 和 Graph 参数 |
| Engine 配置 | `python/minisgl/engine/config.py` | 将 CLI 参数传入 engine/kvcache |
| Scheduler | `python/minisgl/scheduler/` | batch 组装、Prefill/Decode 和请求状态 |
| KV pool | `python/minisgl/kvcache/mha_pool.py` | FP8 storage、scale calibration、telemetry |
| FlashInfer attention | `python/minisgl/attention/fi.py` | metadata、plan、decode/prefill 调用 |
| fused store Python wrapper | `python/minisgl/kernel/store_fp8.py` | JIT 参数和 kernel dispatch |
| fused store CUDA | `python/minisgl/kernel/csrc/jit/store_fp8.cu` | warp、pair load、FP8 packed store |
| 量化 dispatch | `python/minisgl/quantization/` | FP8/AWQ 配置和 QuantLinear |
| benchmark | `benchmark/online/` | 固定并发、开放环、GPU telemetry |
| 实验记录 | `learning/experiments/` | 每个假设的原理、gate 和结论 |

## 十、面试中必须主动说明的边界

1. 绝对 tokens/s 只属于指定 L20、模型、TP、prompt、output 和服务配置，不能泛化到 A100、H100 或 NVLink 集群。
2. FP8 W8A8 的质量结果来自指定评测集；不能只报吞吐而省略 MMLU 或其他 quality gate。
3. FP8 KV Cache 的 2 倍 capacity 是容量结果，不等价于 2 倍端到端性能。
4. 实验 063/064 的 fused/native 路径仍是 opt-in；默认关闭是因为正确性和普适性能门禁尚未完全通过。
5. 失败实验不是无效工作：它们说明了哪些启发式、Graph barrier、量化或通信配置不能在当前 workload 上安全推广。

## 十一、最后的项目总结

这个项目最值得展示的能力不是记住某个命令，而是能把一个推理系统问题拆成可验证的工程问题：先描述现象，再定位瓶颈，设计最小改动，建立公平 baseline，测量端到端和局部指标，最后根据质量和尾延迟 gate 做接受或拒绝。对于 AI Infra 面试，这比“把 batch size 调大后吞吐提高了”更能说明对推理引擎、GPU 硬件和生产约束的理解。
