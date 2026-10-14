# 实验056：FP8 Prefill/Decode 调度参数扫查

## 实验问题

实验055选择了 FP8 TP4 作为当前 L20 服务候选，但混合 256/1024/4096-token prompt 的开放环
workload 仍有 TTFT 尾延迟。本实验验证限制单次 prefill 工作量或调整 decode 可运行时的
prefill budget，能否在不牺牲吞吐和 TPOT 的情况下改善 TTFT。

## 固定条件

- Qwen3-32B，FP8 W8A8 权重，BF16 runtime，TP4，L20 物理 GPU 0--3。
- FlashInfer、CUDA Graph 最大 batch 32、Direct PyNCCL（`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`）。
- `num_pages=16384`、最大序列长度 8192、BF16 KV cache、memory ratio 0.9。
- 输入长度均衡为 256/1024/4096，输出 128 token，每次 12 个请求。
- 到达率 0.5/1.0/1.5 req/s；每个配置每个速率 3 次；seed `5600042`。
- TTFT SLO <= 2000 ms，单请求最大 TPOT SLO <= 1000 ms。
- `/usr/bin/nvcc` 为 CUDA 11.5，无法解析 `-std=c++20`；本实验进程显式使用
  `/usr/local/cuda-12.8/bin/nvcc`，未修改系统 CUDA 配置。

## 变量矩阵

| 配置 | `max_prefill_length` | `max_prefill_streak` | `decode_active_prefill_length` |
| --- | ---: | ---: | ---: |
| baseline | 4096 | 1 | 1024 |
| P1 chunk2048 | 2048 | 1 | 1024 |
| P2 active512 | 4096 | 1 | 512 |
| P3 chunk2048+active512 | 2048 | 1 | 512 |
| P4 streak2 | 4096 | 2 | 1024 |

每个配置包含 9 个 run，共 45 个 run、540 个请求；所有请求都返回完整输出，无请求错误，
TPOT SLO 均为 100%。同时开启 prefill telemetry，记录 budget selection、batch 数和最大
pending token。

## 端到端结果

| 配置 | 速率 | tok/s +/- std | P90 TTFT ms | TTFT SLO | avg TPOT ms | P99 TPOT ms | P99.9 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5 | 70.32 +/- 14.69 | 1071.98 | 100.0% | 20.89 | 216.34 | 240.52 |
| baseline | 1.0 | 98.64 +/- 8.78 | 1005.68 | 100.0% | 21.30 | 166.15 | 237.51 |
| baseline | 1.5 | 153.81 +/- 8.59 | 2489.72 | 80.6% | 30.18 | 230.46 | 244.49 |
| P1 chunk2048 | 0.5 | 70.31 +/- 14.69 | 1075.05 | 100.0% | 20.89 | 216.83 | 241.22 |
| P1 chunk2048 | 1.0 | 98.52 +/- 8.83 | 1012.65 | 100.0% | 21.25 | 168.32 | 231.17 |
| P1 chunk2048 | 1.5 | 153.76 +/- 8.57 | 2489.49 | 80.6% | 30.21 | 227.51 | 238.21 |
| P2 active512 | 0.5 | 70.15 +/- 14.58 | 1165.23 | 100.0% | 21.27 | 128.00 | 148.44 |
| P2 active512 | 1.0 | 98.22 +/- 8.79 | 1076.36 | 100.0% | 21.39 | 126.16 | 131.81 |
| P2 active512 | 1.5 | 148.62 +/- 7.71 | 2725.45 | 80.6% | 30.98 | 131.17 | 158.90 |
| P3 chunk2048+active512 | 0.5 | 70.15 +/- 14.57 | 1188.28 | 100.0% | 21.30 | 127.81 | 147.94 |
| P3 chunk2048+active512 | 1.0 | 98.19 +/- 8.85 | 1095.38 | 100.0% | 21.44 | 125.64 | 132.09 |
| P3 chunk2048+active512 | 1.5 | 148.70 +/- 7.69 | 2729.03 | 80.6% | 30.99 | 131.41 | 153.10 |
| P4 streak2 | 0.5 | 70.35 +/- 14.72 | 1064.00 | 100.0% | 20.90 | 176.60 | 445.66 |
| P4 streak2 | 1.0 | 98.83 +/- 8.78 | 982.60 | 100.0% | 21.31 | 164.62 | 435.63 |
| P4 streak2 | 1.5 | 154.68 +/- 9.10 | 2441.38 | 80.6% | 30.19 | 423.23 | 447.27 |

所有配置的采样峰值显存均为 16453 MiB/card。所有配置的 1.5 req/s TTFT SLO 都是 80.6%，
没有候选改善 TTFT 门槛。

## Telemetry 结果

| 配置 | prefill batches | budget-limited batches | budget selections | max pending tokens | max pending requests |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 232 | 149 | 1024:432, 4096:33 | 8472 | 5 |
| P1 chunk2048 | 240 | 157 | 1024:429, 2048:41 | 8472 | 5 |
| P2 active512 | 372 | 290 | 512:570, 4096:33 | 8472 | 6 |
| P3 chunk2048+active512 | 380 | 298 | 512:568, 2048:41 | 8472 | 6 |
| P4 streak2 | 236 | 149 | 1024:446, 4096:33 | 8472 | 5 |

P1 只增加了 prefill batch 数，最大 pending token 不变。P2/P3 把 active budget 减半后，batch
数从 232 增加到 372--380，吞吐下降约 3.3%，而 TTFT 没有改善。P4 的吞吐只略增，但
P99.9 TPOT 从约 240 ms 增至约 445 ms，说明更多连续 prefill 批次放大了 decode token gap。

## 决策

- **不接受 P1--P4 作为新默认配置**；实验055的 4096/1/1024 baseline 继续保留。
- 静态 budget 没有改变本 workload 的最大 pending queue；下一步应根据 pending token pressure
  动态改变 budget，或做真正的 prefill/decode 分离。
- P2/P3 的 TPOT P99/P99.9 改善不能单独作为成功标准，因为平均 TPOT、吞吐和 TTFT 没有
  达到验收门槛。
- 下一步测试 pending-pressure 自适应 prefill budget：低压力使用 1024/4096 保持吞吐，高
  压力时只允许较小 budget，并用相同 workload 检查 TTFT 尾部。

## 产物

- `learning/results/056_*_fp8_poisson.json` / `.md`
- `learning/results/056_*_prefill_telemetry.jsonl`
