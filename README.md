# Mini-SGLang：面向 NVIDIA L20 的大模型推理优化实验

这是一个基于 [Mini-SGLang](https://github.com/sgl-project/mini-sglang) 的大语言模型推理系统学习、测量和优化项目。项目以 Qwen3-32B 为主要生产形状，以 NVIDIA L20 为目标硬件，围绕请求调度、KV Cache、低精度推理、Tensor Parallel 通信和 CUDA kernel 做可复现实验。

项目的重点不是“把模型启动起来”，而是建立一条完整的 AI Infra 优化链路：

```text
可观察现象 -> 瓶颈假设 -> 最小代码改动 -> 固定 workload 的 A/B 实验
           -> CUDA/Nsight/telemetry 归因 -> 性能、质量、显存和稳定性验收
           -> 接受、拒绝或保留为 opt-in
```

所有实验的原理、命令、原始结果、验收标准和结论都放在 [`learning/`](./learning/) 下；面试导向的详细回答材料放在 [`interview/`](./interview/) 下。

## 项目做什么

Mini-SGLang 保留了现代 LLM serving 的主要组成部分：

- OpenAI-compatible API、Tokenizer 和 Detokenizer；
- Scheduler、Continuous Batching、Prefill/Decode 调度和请求生命周期管理；
- Chunked Prefill、Radix Cache、分页 KV Cache 和显存容量管理；
- FlashInfer/FlashAttention attention backend；
- CUDA Graph、Tensor Parallel、PyNCCL/NCCL collective；
- FP8 W8A8、INT4 AWQ-Marlin 等低精度权重路径；
- 可扩展的 CUDA/C++ JIT kernel。

本项目围绕五个问题展开：

1. L20 的 PCIe Tensor Parallel 通信何时成为瓶颈，TP2/TP4/TP8 如何选择？
2. 长 Prefill 如何影响 Decode 的 TPOT 和尾延迟，怎样在吞吐与交互延迟之间做调度？
3. FP8/INT4 如何影响吞吐、显存、质量和服务稳定性，量化误差与 kernel 错误如何区分？
4. FP8 KV Cache 能否增加长上下文容量，如何处理 scale、饱和、FlashInfer metadata 和写入开销？
5. 一个 kernel 的局部 microbenchmark 变快，为什么不一定代表端到端服务变快？

## 请求执行路径

```text
HTTP 请求
  -> Tokenizer / chat template
  -> Scheduler
       -> Radix Cache 匹配
       -> KV admission / page reservation
       -> Prefill 与 Decode batch 组装
  -> Engine
       -> Model forward
       -> FlashInfer attention
       -> CUDA Graph 或 eager execution
       -> TP collective（PyNCCL/NCCL）
       -> KV Cache 写入和采样
  -> Detokenizer
  -> 流式 OpenAI 响应
```

TTFT 主要受到 Tokenizer、Prefill、队列等待和首个 Decode 的影响；TPOT 主要受到 Decode attention、KV Cache 读取、模型层计算、collective 和调度间隔影响；显存则由权重、activation、KV Cache、CUDA Graph workspace 和通信 buffer 共同决定。

## 为什么选择 L20

L20 是 Ada Lovelace 数据中心 GPU，计算能力为 SM89，单卡约 48 GiB GDDR6 显存，支持 BF16/FP16 Tensor Core 和 FP8 Tensor Core。它很适合推理系统实验：

- 48 GiB 显存可以运行较大的模型，并且能观察 KV Cache、量化和 workspace 的容量变化；
- SM89 支持 FP8，能够研究 W8A8、FP8 KV Cache 和原生 FP8 转换指令；
- L20 多卡主要依赖 PCIe，没有 NVLink 的低延迟互联，Tensor Parallel 的 all-reduce 成本不会被硬件互联隐藏；
- L20 既能展示低精度带来的收益，也会暴露通信、launch overhead、padding 和排队问题。

因此本项目不是只追求一个最高 tokens/s，而是研究在 L20 的计算、显存、PCIe 通信和在线 SLO 之间如何做工程取舍。

## 当前已经验证的结果

下表只记录已经完成并写入实验文档的结果。`opt-in` 表示代码存在，但默认配置不会自动启用。

| 方向 | 已验证结论 | 当前决策 |
| --- | --- | --- |
| TP2/TP4/TP8 | TP4 在当前 Qwen3-32B L20 配置下效率最好；TP8 有吞吐收益但 PCIe 通信导致效率明显下降 | 推荐 TP4；TP8 作为吞吐优先选项 |
| Direct PyNCCL | graph-disabled TP4/TP8 的通信路径有稳定收益；Graph 模式尾延迟 gate 未通过 | 保留为显式部署 profile，不改通用默认值 |
| 调度与 Prefill | 静态增大 Prefill 会放大 Decode 尾延迟；基于 active decode/pending pressure 的策略在特定 workload 有收益，但不能泛化 | 只保留经过验证的 opt-in 策略 |
| FP8 W8A8 | Qwen3-32B TP4 吞吐相对 BF16 提升约 47%–55%，显存/card 下降约 15.48%，MMLU 下降约 0.70 分 | 当前低精度权重的优先候选 |
| INT4 AWQ-Marlin | 能让大模型减少 GPU 数量，但质量和长 prompt TTFT gate 未通过 | 容量优先候选，不作为默认服务路径 |
| FP8 KV Cache | Qwen3-32B TP4 的 KV capacity 从 413,820 提升到 827,641，约 2.00 倍；长上下文饱和率 0.000843% | capacity 优化成立，性能需继续优化 |
| FP8 KV fused store | 4K/8K/16K 端到端 TPOT 相对同代码 legacy 下降 19.47%/14.58%/29.02%，但纯写入 kernel 仍慢 13.6%–60.7% | 显式 opt-in，默认关闭 |
| FP8 native pair conversion | 部分 row size 的写入 kernel 更快，但真实 scale 下 BF16 byte mismatch 约 3.65% | 仅研究开关，默认关闭；不能宣称 bit-exact |

完整数据见 [`learning/experiments/`](./learning/experiments/) 中的 047、053、054、061–064 实验记录。

## 环境与 CUDA 工具链

当前实验环境使用 Python 3.12、PyTorch 2.9.1+cu128、CUDA Toolkit 12.8、FlashInfer 和 NVIDIA L20。Mini-SGLang 会在运行时编译 CUDA/C++ 扩展，所以 `torch.version.cuda`、CUDA Toolkit、driver、FlashInfer 和 `nvcc` 必须匹配。

本机器的 `/usr/bin/nvcc` 是 CUDA 11.5，而正确工具链在 `/usr/local/cuda-12.8/bin/nvcc`。误用旧 nvcc 会出现 `Unknown option '--compress-mode=size'` 或 C++20 编译错误。运行前建议：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

## 安装与启动

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
uv pip install pytest pytest-cov
```

单卡快速启动：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

L20 上 Qwen3-32B FP8 W8A8 TP4 的 graph-disabled 研究配置：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 \
python -m minisgl \
  --model /path/to/Qwen3-32B \
  --tp-size 4 \
  --attention-backend fi \
  --dtype bfloat16 \
  --quantization fp8 \
  --memory-ratio 0.8 \
  --kv-cache-dtype float8_e4m3fn \
  --kv-cache-scale 0.25 \
  --kv-cache-calibration first_batch \
  --cuda-graph-max-bs 0 \
  --max-prefill-length 20000 \
  --port 1964
```

FP8 fused store 和 native pair 是研究开关，必须明确知道其正确性/性能限制后再使用：

```bash
  --kv-cache-fused-store \
  --kv-cache-native-pair
```

## 如何做可信实验

每个实验至少记录：

- 模型 checkpoint、GPU 编号、TP size、dtype、attention backend 和 memory ratio；
- 输入长度、输出长度、并发或到达率、重复次数和随机 seed；
- output tokens/s、TTFT、TPOT、P90/P99.9、E2E、GPU 利用率和峰值显存；
- 正确性或质量指标，例如 HTTP 成功率、token 一致性、MMLU、byte-level cache 对照；
- 失败原因、未通过的 gate 以及是否保留为 opt-in。

固定并发 benchmark 适合做局部 A/B；开放环 benchmark 更接近真实在线服务，因为它能观察排队、Prefill/Decode 竞争和负载 knee。局部 kernel 则需要 CUDA event 或 Nsight 归因，不能用端到端结果反推单个 kernel 的原因。

## 目录

- `python/minisgl/`：推理引擎、scheduler、engine、KV Cache、attention、量化和 kernel；
- `benchmark/`：离线、在线、开放环、通信、Radix Cache 和量化 benchmark；
- `tests/`：核心逻辑、调度、KV Cache、kernel、通信和量化测试；
- `learning/grok.md`：项目背景和优化方向；
- `learning/l20_optimization_plan.md`：实验编号、计划和验收标准；
- `learning/progress.md`：实验进度和决策；
- `learning/experiments/`：实验原理、方法、数据和结论；
- `learning/results/`：Markdown/JSON benchmark、telemetry 和原始结果；
- `interview/`：中文面试讲解材料和常见追问回答。

## 面试中的一句话介绍

> 我基于 Mini-SGLang，在 PCIe 互联的 4 卡 NVIDIA L20 上对 Qwen3-32B 做了可复现的推理优化。我先用 TP scaling、Nsight、开放环 workload 和 telemetry 定位通信、Prefill/Decode 排队以及 KV Cache 瓶颈，再实现 FP8 W8A8、FP8 KV Cache 和 fused CUDA store。最终我根据吞吐、TTFT、TPOT、尾延迟、显存和质量 gate，区分了可推荐的 FP8 TP4 profile、只能 opt-in 的 fused kernel，以及因为数值风险而拒绝默认启用的 native pair conversion。

更详细的技术回答见 [`interview/mini_sglang_l20_interview.md`](./interview/mini_sglang_l20_interview.md)。
