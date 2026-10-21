# 实验 087：FP8 KV cache 最终长输出质量与性能 gate

日期：2026-10-21
状态：完成；容量通过，校准/饱和通过，长输出质量和性能 gate 失败，保持 default-disabled

## 1. 为什么还要做 087

实验 061-064 已经回答了几个局部问题：

- FP8 E4M3 KV 在 L20/FlashInfer 上可运行，KV token capacity 约为 BF16 的 2 倍；
- `first_batch` 逐层校准可以把饱和率控制在很低水平；
- fused cast/store 相对 FP8 legacy 路径能改善部分端到端 TPOT；
- native-pair conversion 对真实非二进制 scale 不是 byte-exact，不能进入最终默认候选。

但旧实验只做了 32-token 输出。短输出中的固定 prefill/TTFT 占比很大，也无法观察量化误差
是否在自回归生成中累积。本实验只比较最终候选：BF16 KV 与 fused FP8 KV，补齐
4096/16384 context 和 128/512 output 的长 decode gate。

## 2. 实验设计

### 固定配置

- GPU：L20 2/3/6/7，TP=4；模型：Qwen3-32B FP8 W8A8 权重。
- attention：FlashInfer；CUDA Graph disabled；memory ratio=0.8。
- max prefill length=20000；并发=1；每个性能点重复 2 次。
- BF16：`--kv-cache-dtype bfloat16`。
- FP8：`--kv-cache-dtype float8_e4m3fn --kv-cache-calibration first_batch
  --kv-cache-fused-store`。
- native pair 关闭：实验 064 已证明真实 calibration scale 下存在 byte mismatch。
- FP8 服务先运行固定 4K/512 质量请求完成逐层校准，后续性能点复用固定 scale。
- 每个性能点的 prompt seed 在 BF16/FP8 间完全一致。

### 矩阵

| context | output | seed | repeats |
| ---: | ---: | ---: | ---: |
| 4096 | 128 | 8701042 | 2 |
| 4096 | 512 | 8702042 | 2 |
| 16384 | 128 | 8703042 | 2 |
| 16384 | 512 | 8704042 | 2 |

首次 BF16 权重加载受共享存储影响耗时约 14 分钟；FP8 服务重启因页缓存缩短到约 3 分钟。
这属于环境 I/O，不计入 TTFT/TPOT，因为 benchmark 只在服务 ready 后开始。

## 3. 容量和校准结果

| KV dtype | KV tokens/card | K+V budget/card | 初始化后 free/card |
| --- | ---: | ---: | ---: |
| BF16 | 339,255 | 20.71 GiB | 8.76 GiB |
| FP8 E4M3 | 678,510 | 20.71 GiB | 8.76 GiB |

FP8 在相同显存预算下提供精确 2.00x token capacity。4 个 TP rank 共 256 层全部完成校准；
本轮所有质量/性能请求累计写入 11,759,648,768 个 K/V 元素，其中 1,395 个达到饱和阈值：

- 加权饱和率：0.0000119%；
- scale min/median/max：0.02438 / 0.05404 / 0.30780。

因此后续质量或性能失败不能简单归因于明显的 FP8 极值饱和。

## 4. 长输出确定性质量 gate

新增 `benchmark/online/check_kv_output_agreement.py`，固定 prompt seed、greedy sampling、
`ignore_eos=true`，保存 prompt/output SHA-256、stream fragments 和重分词 token IDs。

质量 workload：input=4096、output=512、seed=8700042。

| 指标 | BF16 KV | fused FP8 KV |
| --- | --- | --- |
| prompt SHA-256 | `f11be80f...e587` | `f11be80f...e587` |
| output SHA-256 | `f4bde5f6...7149` | `60349448...d010` |
| content fragments | 511 | 511 |
| retokenized output tokens | 511 | 511 |
| longest identical fragment/token prefix | 12 | 12 |

两组输入完全相同，但第 13 个内容 token 首次分叉，最终输出 hash 不同。协议流包含一个非内容
事件，因此 requested output=512 对应 511 个内容 fragment；两组计数一致，不影响比较。

这个结果否定了“32-token exact-output smoke 可以代表长输出质量”的假设。它仍不是完整的
perplexity/MMLU 评测，但对于“FP8 KV 是否可作为 BF16 的透明默认替换”已经是明确失败。

## 5. 端到端性能结果

表中 TPOT 为两次重复的 average TPOT；tail 为 P99.9 TPOT。

| context/output | KV | tok/s | avg TTFT ms | avg TPOT ms | P99.9 TPOT ms | E2E s | peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K/128 | BF16 | 19.11 | 1095.25 | 44.50 | 70.61 | 6.747 | 36,625 |
| 4K/128 | FP8 fused | 16.08 | 1144.20 | 53.79 | 77.91 | 7.975 | 36,625 |
| 4K/512 | BF16 | 24.01 | 1090.77 | 39.69 | 72.25 | 21.374 | 36,625 |
| 4K/512 | FP8 fused | 15.31 | 1145.05 | 63.55 | 106.78 | 33.622 | 36,625 |
| 16K/128 | BF16 | 12.48 | 4854.88 | 42.92 | 68.78 | 10.306 | 37,027 |
| 16K/128 | FP8 fused | 8.65 | 4954.95 | 77.44 | 113.14 | 14.790 | 37,027 |
| 16K/512 | BF16 | 18.62 | 4849.38 | 44.41 | 106.12 | 27.542 | 37,029 |
| 16K/512 | FP8 fused | 13.85 | 4937.78 | 63.10 | 109.64 | 37.182 | 37,027 |

FP8 fused 相对 BF16：

| context/output | throughput | avg TTFT | avg TPOT |
| --- | ---: | ---: | ---: |
| 4K/128 | -15.88% | +4.47% | +20.86% |
| 4K/512 | -36.23% | +4.98% | +60.11% |
| 16K/128 | -30.66% | +2.06% | +80.45% |
| 16K/512 | -25.63% | +1.82% | +42.09% |

四个点方向一致：FP8 TTFT 只小幅回退，但 decode TPOT 显著恶化，最终吞吐下降
15.9%--36.2%。实验 063 的“fused 相对 FP8 legacy 改善”仍然成立，但不能推出 fused FP8
已经追平 BF16；087 使用正确的 BF16 reference 后，最终性能 gate 明确失败。

峰值显存看起来相同，是因为 `memory_ratio=0.8` 会把相同预算尽量分配给 KV pool；FP8 的
收益表现为 token capacity 从 339,255 增到 678,510，而不是 benchmark 时少占一半显存。

## 6. 验收结论

| Gate | 状态 | 证据 |
| --- | --- | --- |
| 2x KV capacity | 通过 | 339,255 -> 678,510 tokens/card |
| calibration 完整 | 通过 | 4 ranks、256/256 layers calibrated |
| saturation < 0.001% | 通过 | 0.0000119% |
| 长输出 token agreement | 失败 | 4K/512 在第 13 token 分叉 |
| fused FP8 TPOT 不回退 | 失败 | avg TPOT +20.9%--+80.4% |
| throughput 不低于 BF16 98% | 失败 | -15.9%--36.2% |
| 默认启用 FP8 KV | 拒绝 | quality 和 performance 均未通过 |

最终决策：FP8 KV 只保留为 capacity-first、显式 opt-in 的研究能力。对于“BF16 KV 容量
不足、否则模型/上下文无法运行”的场景可以选择它，但必须接受输出可能分叉和 decode 变慢；
它不是当前 L20 通用性能优化，也不进入默认配置。

## 7. 验证和原始结果

- 新增工具 compile/help smoke 通过。
- `tests/core/test_kv_cache_dtype.py` 与 `tests/core/test_fp8_quantization.py`：20/20 通过。
- 初次测试命令误引用不存在的 `tests/kernel/test_store_fp8.py`，未运行任何测试；随后已用
  仓库实际测试文件纠正，不把失败命令计为通过。
- 原始结果：`learning/results/087_bf16_*.{json,md}`、
  `learning/results/087_fp8_fused_*.{json,md}`、两个 `087_*_quality_4096x512.json` 和
  `087_fp8_calibration.rank{0,1,2,3}.json`。
