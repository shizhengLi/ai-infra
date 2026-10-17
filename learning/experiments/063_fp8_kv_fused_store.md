# 实验 063：FP8 KV fused cast/store kernel

日期：2026-10-17  
状态：完成；正确性通过，端到端 TPOT 有改善，纯写入 kernel 性能目标未完全通过，默认关闭

## 动机

实验 062 说明 FP8 KV 的 capacity 和校准收益成立，但 eager 路径的 FP8 decode 仍有明显
TPOT 回退。原实现先执行 BF16/FP16 -> FP8 cast/clamp，再调用 indexed page store，至少
包含两段 GPU 工作。本实验加入一个显式 `--kv-cache-fused-store` 原型，把 scale、clamp、
FP8 E4M3 conversion 和 K/V page store 放进同一个 CUDA kernel。

## 实现

- 新增 `python/minisgl/kernel/csrc/jit/store_fp8.cu`，每个 warp 负责一个 token 行。
- 支持 BF16 和 FP16 输入；每个 lane 处理 4 个元素，使用 CUDA
  `__nv_fp8x4_e4m3` 打包写入。
- scale 通过每层 1-element CUDA tensor 传入，和 `first_batch` 校准保持同一 stream 顺序。
- 新增 `--kv-cache-fused-store`，默认 `false`；关闭时继续走原有 cast + `store_cache`，
  因此实验性 kernel 不改变默认服务行为。
- fused 输出只写入目标 `indices` 行；BF16/FP16 输入下与 legacy 路径写出的 FP8 bytes
  完全一致（GPU smoke，K/V 各 16 行、1024 列）。

## CUDA 微基准

L20 GPU 2、KV row width 1024、BF16 输入、FP8 E4M3 输出，重复 80 次，CUDA event 计时：

| Rows | Legacy cast + store ms | Fused ms | Fused 相对变化 |
| ---: | ---: | ---: | ---: |
| 256 | 0.0185 | 0.0210 | +13.64% |
| 1,024 | 0.0179 | 0.0210 | +17.01% |
| 4,096 | 0.0181 | 0.0211 | +16.24% |
| 16,384 | 0.0184 | 0.0295 | +60.70% |

成对加载和 `fp8x4` 打包已经比第一版逐元素 kernel 改善 16K 的开销，但纯写入路径仍未
达到“fused 比 legacy 快 20%”的目标。原因是 legacy 的 PyTorch cast 和现有 memcpy kernel
都使用高度优化的向量化实现，而 fused kernel 仍承担逐行 scale/clamp/conversion。

## Qwen3-32B TP4 端到端 A/B

同一 commit、L20 GPU 2/3/6/7、FP8 W8A8 权重、FlashInfer、CUDA Graph disabled、并发 1、
32-token 输出、重复 2 次。两组都启用 `first_batch` calibration 和 summary；唯一变量是
`--kv-cache-fused-store`。

| Input | Legacy tok/s | Fused tok/s | 吞吐变化 | Legacy TPOT ms | Fused TPOT ms | TPOT 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 9.67 | 11.08 | +14.56% | 69.07 | 55.62 | -19.47% |
| 8,192 | 10.38 | 11.44 | +10.21% | 58.36 | 49.85 | -14.58% |
| 16,384 | 6.33 | 7.38 | +16.50% | 76.09 | 54.01 | -29.02% |

Fused 的 TTFT 也略低于同一轮 legacy（约 0.7%--1.7%），但两组绝对值受单机 GPU 调度
抖动影响；因此主要采用 TPOT/吞吐方向判断。与实验 062 的 BF16 baseline 相比，fused
FP8 仍有 4K/8K 吞吐回退，16K 的回退收窄到约 8%，所以这不是“FP8 已全面加速”的结论。

两组 telemetry 的累计写入元素、scale 分布和饱和率一致：4,347,265,024 个元素中
36,639 个饱和，饱和率 0.000843%。这证明 fused store 没有改变校准策略或饱和行为。

原始 benchmark：

- [fused 4096](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_fused_4096.md)
- [fused 8192](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_fused_8192.md)
- [fused 16384](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_fused_16384.md)
- [legacy 4096](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_legacy_4096.md)
- [legacy 8192](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_legacy_8192.md)
- [legacy 16384](/data2/lszlsz/code/mini-sglang/learning/results/063_fp8_legacy_16384.md)

## 结论和后续

1. **正确性通过**：BF16/FP16 输入下，fused 与 legacy 对目标 cache 行生成完全相同的
   FP8 bytes；真实 Qwen3-32B 三种长上下文均 HTTP 200。
2. **端到端局部改善**：fused 在本轮 A/B 中把 TPOT 降低 14.6%--29.0%，吞吐提升
   10.2%--16.5%。
3. **纯 kernel 目标未通过**：fused 写入微基准仍比 legacy 慢 13.6%--60.7%，说明还不能
   将该 kernel 作为普适优化或默认策略。
4. **部署决策**：`--kv-cache-fused-store` 保持显式 opt-in；默认继续使用 legacy，直到
   完成更低级的 vectorized conversion/FlashInfer decode attribution，并在并发 workload
   下重复验证。
