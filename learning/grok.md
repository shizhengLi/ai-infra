# Mini-SGLang L20 优化项目总览

更新时间：2026-10-18

## 这是什么项目

Mini-SGLang 是一个规模较小、但保留现代 LLM serving engine 核心路径的推理引擎。它包含
请求生命周期、continuous batching、Radix KV cache、chunked prefill、Paged KV cache、
CUDA Graph、FlashAttention/FlashInfer、Tensor Parallelism、PyNCCL 和 OpenAI 兼容 API。

本项目不是只做一个模型 demo，而是把推理系统拆成可测量的瓶颈：

```text
HTTP 请求
  -> tokenizer / prompt 构造
  -> scheduler admission
  -> prefill/decode batch 组装
  -> KV cache page 分配与回收
  -> attention backend + CUDA Graph
  -> TP collective / PyNCCL
  -> streaming response
```

每个优化都要求有代码、实验、telemetry、回归测试和明确的保留/拒绝结论。实验结果统一
记录在 `learning/experiments/`、`learning/results/`、`learning/progress.md` 和
`learning/l20_optimization_plan.md`。

## 为什么选择 NVIDIA L20

L20 是 Ada Lovelace 数据中心卡，Compute Capability 为 SM89，单卡 48 GB GDDR6。它适合
做推理系统优化，但也有几个必须正面处理的约束：

- 没有 NVLink，多卡 TP 主要走 PCIe + NCCL，通信和同步更容易成为瓶颈；
- SM89 与 SM90 的 FP8、FlashAttention、CUTLASS DSL 能力不同，不能把 H100 的结论直接
  复制到 L20；
- 48 GB 显存足以承载 Qwen3-32B 的量化权重和大 KV cache，但在 BF16/长上下文/TP 配置下
  仍需要精确的 page admission；
- L2 较大，decode 既受显存带宽影响，也受 kernel launch、CUDA Graph padding 和 PCIe
  collective 影响。

当前实验主要使用 GPU 2/3/6/7 做 TP4，模型为 Qwen3-32B-AWQ，dtype 为 float16。这个
组合能稳定启动真实 API 服务，且保留了量化、KV cache、调度和多卡通信几个面试主题。

## 已经实现的能力

### 运行时和通信

1. 修复了 PyNCCL AOT 模块对 wheel 内版本化 `libnccl.so.2` 的链接。现在链接绝对库路径
   并设置 rpath，即使系统 `/usr/local/cuda-12.8/lib64` 没有 unversioned `libnccl.so`，
   PyNCCL 仍可以加载。
2. TP4 Qwen3-32B-AWQ 已在 L20 GPU 2/3/6/7 完成权重加载、CUDA Graph capture 和 API
   ready。Gloo 仅用于 CPU rank group 和 NCCL UID bootstrap，GPU TP collective 走
   PyNCCL/NCCL；因此目前不需要为了“去掉 Gloo”而重写 bootstrap。
3. 保留了 `fi`、`fa`、CUDA Graph shape policy、PyNCCL buffer 等显式控制，便于做可复现
   A/B，而不是让自动策略掩盖实际变量。

### 低精度推理

- FP8 W8A8 权重路径已经可以在 L20 SM89 上完成加载、TP sharding、Graph capture 和真实
  Qwen3-32B TP4 serving。实验 053/055 选择 FP8 作为当前 Graph/TP4 性能候选；BF16 仍是
  参考实现。
- AWQ/INT4-Marlin 已能在 TP1/TP2 运行，显著降低单副本显存和 GPU 数，但 MMLU 质量 gate
  在实验 054 中失败，因此 AWQ 是容量优先 profile，不是无条件的默认性能优化。
- FP8 KV cache、fused cast/store 和 native pair conversion 已有 opt-in 原型。它们能把 KV
  token capacity 理论上扩大约 2 倍，但实验 062/063/064 暴露了长上下文 TPOT、写 kernel
  和 scale 表示问题，当前保持 default-disabled。

### KV cache 与调度安全

- 实验 071 增加了 decode admission guard，按当前可见 free/evictable 容量裁剪未来 page
  demand，修复 overlap pipeline 下的 allocator assertion 风险。
- 实验 073/075/076 实现 decode capacity round-robin、动态 UID cursor 和 wait-round
  telemetry，解决固定前缀选择造成的饥饿和请求集合变化导致的索引漂移。
- `token` 与 `page-aware` prefill reservation 都可通过 CLI 选择；page-aware 作为安全/尾
  延迟优先的 opt-in，不改变默认 token 路径。
- telemetry 已覆盖 prefill budget、pending token/request、decode batch、Graph hit、
  capacity trim/deferred、retraction 和 wait rounds，使“优化分支是否真的执行”可以被验证。

## 已完成实验的主线结论

### 早期主线：量化、Graph、通信和 CPU 路径

- 实验 032-034：direct PyNCCL 在 graph-disabled TP4/TP8 上有约 9%-12% 的在线吞吐收益，
  但 Graph 模式的尾延迟 gate 未通过；因此 direct PyNCCL 保留为显式 profile，不修改通用默认。
- 实验 051-055：FP8 W8A8 是当前 L20 TP4 的性能候选；AWQ/INT4 更偏向容量和 GPU 数量优化，
  质量 gate 风险必须单独说明。
- 实验 060：预分配 `Req` token buffer 去掉了 decode 热路径中的反复 `torch.cat`，append
  allocation 降为 0、复制 token 大幅下降，但端到端吞吐没有收益，CPU wall-time -50% gate
  未通过。因此这是可独立提交的 CPU correctness/优化 PR，不应包装成 serving 吞吐提升。
- 实验 066：L20 memory-aware Graph shape 将 padding ratio 从 15.65% 降至 1.48%，增加约
  0.33 GiB/card，吞吐提高 0.67%-1.00%；低负载 P99 TPOT 仍有回退，默认保持 coarse。
- 实验 067：tokenizer micro-batch 的 CPU median 提高约 2.26x，但 1 ms wait 未通过低速率
  TTFT -5% gate；batch tokenizer 是高并发 CPU 候选，等待窗口保持 0 ms。

### 新近主线：page、长 decode 和容量压力

- 实验 078/079：page-aware reservation 在真正命中容量裁剪的 4-page trace 中消除了 trim，
  但吞吐下降约 12.84%；在 TP4 Qwen3-32B 的 8/16-token page 压力中 admission 已提前限流，
  token 与 page-aware 几乎相同。因此 page-aware 不能仅凭局部安全收益改成默认。
- 实验 080：总 KV 容量固定为 128 token、输出 16，比较 page size 8/16/32。page size=16
  吞吐最高且 TPOT tail 最低，但样本很短，不能外推到长 decode。
- 实验 081：总容量固定为 1024 token、输出 128。page size=8 吞吐最高（305.82 tok/s），
  page size=32 的 P99 TPOT 最低（33.98 ms）。它与实验 080 的排序不同，证明 page 粒度必须
  和输出长度、decode batch shape 一起评价。
- 实验 082：16 个请求制造 KV pressure。1024-token cache 在 20/80 req/s 下保持约
  381/393 tok/s，但 P90 TTFT 约 3.26/3.47 秒；256-token cache 吞吐降至 89.96 tok/s，
  P90 TTFT 升至 19.77 秒，decode batch 全部为 1。两种容量都没有 allocator assertion，
  说明当前主要是 admission backpressure，而不是 decode trim。
- 实验 083：在 256-token cache 下 A/B token/page-aware。吞吐 88.99/90.14 tok/s，差异约
  1.30%；两组 telemetry 完全相同，6160 个 decode batch 全部为 1，未命中公平性 cursor。
  这是中性结果，证明 page-aware 在 admission 已串行化时没有可见额外收益。
- 实验 084：用确定性多候选 scheduler trace 直接命中 12 次 trim 和 1 次 deferred，覆盖
  UID 完成、取消、新 UID 加入与容量恢复；UID 0/1/2 在 9 轮中各服务 6 次。该结果接受为
  fairness/telemetry 集成正确性证据，但因为没有运行模型，不宣称 GPU 吞吐收益。

## 当前默认决策

| 能力 | 当前状态 | 为什么 |
| --- | --- | --- |
| BF16 | 保留为 reference | 质量和兼容性基线 |
| FP8 W8A8 | Graph/TP4 显式候选 | 性能收益已测量，需保持质量 gate |
| AWQ/INT4 | capacity-first opt-in | 省显存和 GPU，但质量/延迟不是默认最优 |
| PyNCCL wheel linker | 已接受 | 修复真实部署阻塞，TP4 已验证 |
| direct PyNCCL buffer=0 | 显式 profile | Graph 模式 tail gate 未通过 |
| page-aware reservation | opt-in | 只在特定 trim trace 有安全收益，吞吐和 TP4 pressure 证据不足 |
| page size 8/16/32 | 不改默认 | 结果依赖输出长度和 batch shape |
| L20 Graph shape policy | coarse 默认，l20 opt-in | padding 明显下降但低负载 tail 仍有风险 |
| tokenizer wait | 0 ms 默认 | CPU microbenchmark 有收益，低速率 TTFT gate 未通过 |
| FP8 KV cache | default-disabled | capacity 有潜力，但长上下文和 kernel gate 未完成 |

## 面试时如何讲

建议用“瓶颈、假设、实现、证据、决策”五步，而不是只报一个最快数字：

1. **定位瓶颈**：L20 无 NVLink，所以区分 PCIe collective、Graph padding、KV cache admission
   和 CPU scheduler；用 telemetry/Nsight 证明瓶颈在哪。
2. **提出假设**：例如 page size 太大可能产生内部碎片，或 prefill admission 只按 token
   估算会破坏 page allocator 不变量。
3. **小范围实现**：增加 guard、cursor、wait telemetry、page-aware policy 或 PyNCCL linker，
   每次只改变一个主要变量。
4. **做压力和 held-out 验证**：至少三次重复，报告 throughput、P90/P99 TTFT、P99 TPOT、
   GPU util、容量事件和正确性；明确哪些实验没有命中目标路径。
5. **做工程决策**：只有通过 correctness、tail 和回归 gate 的改动才进入默认；其他结果保留
   为 opt-in 或拒绝，并说明下一步需要什么证据。

最适合简历的表述是：

> 在 L20 PCIe TP4 上构建可复现的 Mini-SGLang serving 优化实验链路，完成 PyNCCL wheel
> runtime 修复、KV admission safety guard、容量公平性 telemetry、FP8/AWQ profile 和
> page-size/长 decode/极小 KV cache pressure 矩阵；用真实 HTTP workload 和三次重复区分
> throughput、TTFT tail、TPOT tail、GPU 利用率和 allocator correctness，保留明确的
> default/opt-in 决策。

## 下一阶段路线

1. 实验 085：在 256/512/1024-token cache 上做 output 128/512 与 arrival rate 20/80 的
   pressure matrix，寻找 page-aware 的有效边界，而不是继续做无压力 A/B。
2. 实验 086：把 `Req` 预分配 buffer 和 append allocation telemetry 整理为独立上游 PR，
   用长输出 CPU benchmark 证明收益边界。
3. 实验 087：完成 FP8 KV fused cast/store 的长上下文 quality/ITL gate；若仍无端到端收益，
   保留为研究原型，不修改默认。
4. 实验 088：将 telemetry、PyNCCL linker、page-aware admission 和 Graph policy 的接口
   拆成可审查的 upstream PR，附最小复现、focused tests 和 L20 benchmark，而不是提交大杂烩。
