# 实验055：Held-out 量化服务对比

## 实验目的

实验053和054分别证明了 FP8 和 AWQ-Marlin 的局部/固定并发收益，但固定并发和单一旧
MMLU抽样不能直接回答部署选择：在开放环到达、混合 prompt 长度和长 prompt 排队下，哪种
配置更适合 L20 上的在线服务。本实验使用新的随机种子、相同到达轨迹和相同 prompt，比较
BF16、FP8 和 INT4 AWQ 的端到端行为。

## 假设与原则

1. FP8 W8A8 应在相同 TP4 服务拓扑下减少线性层计算和显存压力，从而改善 TTFT/TPOT。
2. AWQ-Marlin 应以更少 GPU 提供容量优势，但 TP1 的计算和调度余量可能不足以承受混合长
   prompt 的开放环负载。
3. held-out 质量集只用于发现明显回归；570 道题、单个新 seed 不足以证明几个百分点以内的
   微小精度差异，因此质量结论必须结合实验054的旧 seed 结果解释。

## 被测配置

所有配置均使用 Qwen3-32B、BF16 KV cache、FlashInfer attention、CUDA Graph 最大 batch
32、`num_pages=16384`、最大序列长度8192、最大 prefill 4096、最大 prefill streak 1、
decode-active prefill 1024。TP4 使用 Direct PyNCCL（`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`），
物理 GPU 0--3；AWQ 使用物理 GPU 0 的 TP1。三种配置如下：

| 配置 | 权重/运行时 | TP | 目的 |
| --- | --- | ---: | --- |
| BF16 | BF16 / BF16 | 4 | 精度和服务参考 |
| FP8 | FP8 W8A8 / BF16 | 4 | 低精度端到端候选 |
| AWQ | INT4 AWQ-Marlin / BF16 | 1 | 单卡容量候选 |

### 开放环 workload

使用 `benchmark/online/bench_poisson_l20.py`，独立指数间隔模拟 memoryless 请求流。每个
速率运行 3 次，每次 12 个请求；输入长度在 256/1024/4096 token 间平衡，输出 128 token，
随机种子为 `5500042`，到达速率为 0.5、1.0、1.5 req/s。TTFT SLO 为 2000 ms，最大单请求
TPOT SLO 为 1000 ms。基准脚本新增 `--server-quantization` 和 `--server-dtype`，并把它们
写入 Markdown/JSON，避免结果文件丢失运行时配置。

### held-out 质量

MMLU 每个 subject 抽取 10 题，共 570 题；并发 32、最大输出 64 token，使用同一新 seed
`5500042`。记录解析率、准确率和三组预测的一致性。

## 服务结果

以下为三次重复的均值；吞吐标准差保留，用来展示开放环固定轨迹下的波动。完整逐请求数据
分别保存在 `learning/results/055_*_poisson.json` 和对应 Markdown 中。

### BF16 TP4 vs FP8 TP4

| 速率 req/s | BF16 tok/s | FP8 tok/s | BF16 P90 TTFT ms | FP8 P90 TTFT ms | BF16 TTFT SLO | FP8 TTFT SLO | BF16 avg TPOT ms | FP8 avg TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 59.54 +/- 9.49 | 63.05 +/- 10.91 | 1534.47 | 1099.33 | 97.2% | 100.0% | 34.00 | 19.71 |
| 1.0 | 100.39 +/- 10.35 | 125.05 +/- 23.02 | 3355.19 | 1553.18 | 66.7% | 97.2% | 42.22 | 27.16 |
| 1.5 | 104.68 +/- 1.82 | 133.14 +/- 11.27 | 2792.49 | 1352.11 | 72.2% | 97.2% | 41.26 | 25.92 |

FP8 的吞吐提高约 5.9%/24.6%/27.2%，P90 TTFT 在三个速率都降低；TPOT SLO 两者均为
100%。采样的每卡峰值显存为 BF16 19109 MiB、FP8 16453 MiB，FP8 少约 2.6 GiB/card。
开放环吞吐标准差较大，尤其是 1.0 req/s，因此不把单次速率点解释为精确容量上限。

### AWQ TP1

| 速率 req/s | AWQ tok/s | P90 TTFT ms | P90 TTFT（4096 输入）ms | TTFT SLO | TPOT SLO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 56.44 +/- 7.68 | 3481.09 | 4807.42 | 61.1% | 94.4% |
| 1.0 | 67.20 +/- 3.83 | 9327.50 | 10370.65 | 27.8% | 100.0% |
| 1.5 | 68.33 +/- 1.88 | 9958.94 | 10555.36 | 22.2% | 100.0% |

AWQ 的单卡峰值显存约 24265 MiB（最高点 24969 MiB），但长 prompt 的 prefill 让 TTFT
迅速排队；这不是延迟敏感混合 workload 的可接受服务配置。它仍然保留“少 GPU/大容量”的
价值，适合低并发或容量优先场景，不能以本实验为依据把 AWQ 设为默认。

## Held-out MMLU 结果

| 配置 | 正确/总数 | 准确率 | 解析率 |
| --- | ---: | ---: | ---: |
| BF16 TP4 | 467/570 | 81.9298% | 100% |
| FP8 TP4 | 468/570 | 82.1053% | 100% |
| AWQ TP1 | 469/570 | 82.2807% | 100% |

BF16/FP8 预测一致 554/570，BF16/AWQ 和 FP8/AWQ 各一致 539/570，三者同时答对 452 题。
新 seed 上没有观察到明显质量回归，但 AWQ 不能因此被宣布“质量通过”：实验054的旧 seed
为 BF16 80.35%、FP8 79.65%、AWQ BF16 78.60%，AWQ 当时低于一百分点 gate。正确表述是
“本 held-out seed 未发现额外回归，旧 seed 质量风险仍存在”。

## 结论与决策

- **接受 FP8 作为当前 Graph/TP4 在线服务候选**：在相同模型和调度参数下，开放环 TTFT
  尾延迟、TPOT 和显存均优于 BF16，且 held-out 质量没有明显下降。
- **保留 BF16 作为质量参考和回退配置**：它的行为稳定、实现最简单，但服务效率较低。
- **AWQ-Marlin 保持默认关闭**：实现和容量/单卡结果有效，但当前 TP1 scheduler 在混合长
  prompt 下 TTFT SLO 明显失败；若要用于服务，应单独研究 AWQ TP2、prefill/decode 分离或
  更强的 AWQ 校准，而不是直接替换默认配置。
- 本实验不改变全局默认量化开关；结论只适用于记录的 Qwen3-32B、L20、Graph max batch 32
  和当前 prefill 调度 profile。

## 可复现实验产物

- `learning/results/055_bf16_tp4_poisson.json` / `.md`
- `learning/results/055_fp8_tp4_poisson.json` / `.md`
- `learning/results/055_awq_tp1_poisson.json` / `.md`
- `learning/results/055_bf16_mmlu.json`
- `learning/results/055_fp8_mmlu.json`
- `learning/results/055_awq_mmlu.json`

