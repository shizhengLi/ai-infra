<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang：L20 推理优化实验

这是一个基于 [Mini-SGLang](https://github.com/sgl-project/mini-sglang) 的大语言模型推理
系统学习与优化项目。Mini-SGLang 是一个规模较小、结构清晰的 SGLang 参考实现，包含了
现代 LLM serving 中最重要的组成部分：请求调度、Continuous Batching、KV Cache、Radix
Cache、Chunked Prefill、CUDA Graph、Tensor Parallelism，以及 FlashAttention/FlashInfer
等高性能 CUDA kernel。

本仓库的重点不是只把模型“跑起来”，而是回答一个更实际的问题：

> 在 4 卡或 8 卡 NVIDIA L20 上，如何通过可解释、可复现、可量化的系统优化，让大模型服务
> 在吞吐、首 token 延迟、decode 延迟、显存占用和质量之间取得更好的平衡？

## 我们在做什么

我们以 Qwen3-32B 为主要实验模型，以 L20 为目标硬件，沿着真实推理请求的生命周期分析
系统瓶颈：

```text
客户端请求
    -> Tokenizer
    -> Scheduler
       - Prefill / Decode 调度
       - Continuous Batching
       - Radix KV Cache
       - Chunked Prefill
    -> Engine
       - FlashInfer Attention
       - CUDA Graph
       - FP8/INT4 GEMM
       - Tensor Parallel 通信
    -> Detokenizer / OpenAI-compatible API
```

当前优化方向包括：

1. **多卡通信与 Tensor Parallelism**：分析 L20 PCIe 通信成本，比较 TP2、TP4、TP8，并评估
   PyNCCL buffer、通信协议和计算/通信重叠。
2. **调度与排队**：研究 Prefill 和 Decode 之间的资源竞争，观察混合短 prompt/长 prompt 负载
   下的 TTFT 尾延迟，进一步尝试动态 Prefill budget。
3. **低精度推理**：在 L20 的 Ada Tensor Core 上比较 BF16、FP8 W8A8 和 INT4 AWQ-Marlin，
   同时检查吞吐、显存、CUDA Graph 兼容性和模型质量。
4. **KV Cache 与请求复用**：评估 Radix Cache 的共享前缀命中、显存分页和长上下文行为。
5. **性能分析方法**：用固定 seed、重复实验、开放环到达模型、Telemetry 和必要的 Nsight
   证据，把“参数看起来更快”转化为可靠的系统结论。

## 为什么选择 NVIDIA L20

L20 是面向推理和数据中心部署的 Ada Lovelace GPU，单卡通常提供 48 GiB 显存，并支持
BF16/FP16 Tensor Core 计算以及 FP8 等低精度路径。它非常适合做大模型推理优化，但又不会
像带 NVLink 的高端训练卡那样掩盖通信问题。

对本项目最重要的硬件特性是：

- **显存较大**：48 GiB 使 Qwen3-14B 可以作为单卡参考，也可以让 Qwen3-32B 在量化后尝试
  TP1/TP2。
- **支持 FP8**：Ada Tensor Core 可以执行 FP8 W8A8 GEMM，因此可以研究权重低精度与动态
  activation quantization 的端到端收益。
- **支持 INT4 推理 kernel**：Marlin W4A16 路径可以显著降低权重显存，但量化误差、校准
  数据和服务吞吐必须分别验证，不能只看 kernel microbenchmark。
- **多卡主要依赖 PCIe**：L20 的多卡通信不像 NVLink 系统那样便宜。Tensor Parallel 的
  all-reduce 会直接影响扩展效率，这正是本项目的真实优化点。
- **计算与通信存在权衡**：提高 TP 数可能让单卡计算压力下降，但通信和同步增加；最优 TP
  配置取决于模型、batch、prompt 长度和服务目标，而不是固定的“卡越多越快”。

因此，L20 既有足够的硬件能力支撑大模型实验，又能暴露调度、通信和低精度实现中的真实
瓶颈，非常适合作为推理系统优化和 AI Infra 面试项目的实验平台。

## 当前实验结论

我们已经完成了从通信、服务负载到量化的连续实验。当前结论如下：

### FP8 是当前在线服务候选

在 Qwen3-32B、TP4、FlashInfer、CUDA Graph max batch 32 的配置下，FP8 相比 BF16：

- 混合开放环负载下 P90 TTFT 更低；
- 平均 TPOT 更低；
- 峰值显存从约 19109 MiB/card 降到约 16453 MiB/card；
- held-out MMLU 没有观察到明显质量回归。

因此当前推荐把 FP8 作为 L20 TP4 的服务候选，把 BF16 保留为质量参考和回退配置。

### AWQ 是容量优先方案，不是当前默认方案

INT4 AWQ-Marlin 可以让 Qwen3-32B 在单卡或双卡上运行，显著减少 GPU 数量，但在当前 TP1
调度配置下，混合长 prompt 的 TTFT SLO 明显失败；实验054的旧 seed 也暴露了 AWQ 质量风险。
所以 AWQ 的正确定位是“低显存/容量优先候选”，不是直接替换 FP8 的默认服务路径。

### 静态 Prefill 参数没有解决排队瓶颈

实验056比较了 `max_prefill_length`、`max_prefill_streak` 和
`decode_active_prefill_length` 的多个组合。减小 chunk 没有改善 TTFT，减小 decode-active
budget 降低了吞吐，连续处理更多 prefill batch 则放大了 TPOT 尾部。所有配置的最大 pending
prefill pressure 都约为 8472 tokens。

这说明下一步应该做 **基于 pending token pressure 的动态 Prefill budget**，而不是继续扩大
静态参数扫查。

详细结果和每次实验的原理、配置、数据、验收标准及结论都记录在 [`learning/`](./learning/)
目录中，实验按编号递增。

## 如何运行

### 环境准备

项目面向 Linux + NVIDIA CUDA 环境。建议使用 Python 3.12 和仓库已有的 `.venv`：

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

Mini-SGLang 会在运行时编译部分 CUDA/C++ 扩展。必须保证 PyTorch、CUDA Toolkit、NVIDIA
驱动和 FlashInfer 的版本匹配。本实验环境使用 CUDA 12.8；如果机器上存在多个 `nvcc`，
应在实验进程中显式选择对应版本：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

### 单卡快速启动

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

服务默认提供 OpenAI-compatible API，例如：

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话解释KV Cache"}],
    "max_tokens": 32
  }'
```

### L20 上的 Qwen3-32B FP8 TP4

下面是当前实验使用的核心服务配置。实际实验还会根据 benchmark 选择端口、模型路径和
GPU 编号：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 \
python -m minisgl \
  --model /path/to/Qwen3-32B \
  --tp-size 4 \
  --attention-backend fi \
  --quantization fp8 \
  --dtype bfloat16 \
  --memory-ratio 0.9 \
  --cuda-graph-max-bs 32 \
  --num-pages 16384 \
  --max-seq-len-override 8192 \
  --max-prefill-length 4096 \
  --max-prefill-streak 1 \
  --decode-active-prefill-length 1024 \
  --port 1953
```

`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` 是当前 L20 TP4 实验中验证过的 Direct PyNCCL 配置；它
不是所有机器上的通用默认值，使用时应结合自己的 NCCL、拓扑和 benchmark 重新验证。

### 开放环服务 benchmark

开放环 benchmark 不等前一个请求完成，而是按照指数分布生成到达时间，更接近真实在线服务：

```bash
python benchmark/online/bench_poisson_l20.py \
  --base-url http://127.0.0.1:1953/v1 \
  --input-lens 256 1024 4096 \
  --output-len 128 \
  --request-count 12 \
  --arrival-rates 0.5 1.0 1.5 \
  --repeats 3 \
  --seed 5600042 \
  --gpu-indices 0,1,2,3 \
  --server-tp 4 \
  --server-quantization fp8 \
  --server-dtype bfloat16 \
  --server-max-extend-tokens 4096 \
  --server-max-prefill-streak 1 \
  --server-decode-active-prefill-tokens 1024
```

重点指标包括：

- **吞吐**：output tokens/s，以及多次重复的标准差；
- **TTFT**：Time To First Token，衡量用户开始看到回答的等待时间；
- **TPOT**：Time Per Output Token，衡量生成阶段的稳定性；
- **P90/P99/P99.9**：尾延迟，避免平均值掩盖长 prompt 排队；
- **SLO 达成率**：例如 TTFT <= 2 秒、最大 TPOT <= 1 秒；
- **显存、GPU 利用率和 prefill telemetry**：解释性能变化的原因，而不是只报告结果。

## 实验规范

每个实验都应该回答一个明确的问题，并遵循以下流程：

1. 固定模型、GPU、精度、请求轨迹和 seed，建立可复现 baseline；
2. 一次只改变一个主要因素，避免把多个收益混在一起；
3. 至少重复三次，同时记录均值、方差和尾延迟；
4. 同时检查性能、显存、正确性和模型质量；
5. 对候选方案设置明确的接受/拒绝门槛；
6. 将原理、命令、原始 JSON、摘要和结论写入 `learning/experiments/` 与
   `learning/results/`；
7. 只有通过验收的结果才进入推荐配置，失败实验同样保留，因为它说明了为什么某种优化
   不适合当前硬件或 workload。

这套方法的目标是形成一条完整的 AI Infra 优化链路：

```text
现象 -> 瓶颈假设 -> 最小改动 -> 可复现实验 -> 指标与 profiling -> 验收/拒绝 -> 下一步假设
```

## 目录说明

- `python/minisgl/`：推理引擎、scheduler、engine、KV Cache、attention 和量化实现；
- `benchmark/`：离线、在线、通信、Radix Cache 和量化 benchmark；
- `tests/`：核心逻辑、调度、KV Cache、kernel、通信和量化测试；
- `docs/structures.md`：系统结构和请求生命周期；
- `docs/features.md`：功能与命令行参数；
- `learning/grok.md`：项目背景、L20 特性和面试导向的优化方向；
- `learning/l20_optimization_plan.md`：编号实验规划、验收标准和阶段结论；
- `learning/progress.md`：实验进度总览；
- `learning/experiments/`：每个实验的原理、方法和详细分析；
- `learning/results/`：机器可读结果的摘要、benchmark Markdown 和 telemetry 记录。

## 面试中如何介绍这个项目

可以把项目概括为：

> 我基于 Mini-SGLang，在 4 卡 L20 上对 Qwen3-32B 做了面向在线服务的推理优化。先用 TP
> scaling、PyNCCL 和开放环 workload 定位 PCIe 通信与 Prefill 排队瓶颈，再实现并验证 FP8
> W8A8 和 INT4 AWQ-Marlin。最终选择 FP8 TP4 作为服务候选，并通过 TTFT、TPOT、吞吐、显存、
> SLO 和 held-out 质量测试说明哪些优化有效、哪些方案应当拒绝。当前继续研究基于 pending
> token pressure 的动态 Prefill/Decode 调度。

这比单纯描述“调了几个参数”更重要：它体现了从系统建模、瓶颈定位、kernel/量化实现到端到端
验收的完整推理基础设施工程方法。
