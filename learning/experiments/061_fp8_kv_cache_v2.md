# 实验 061：L20 FP8 KV Cache v2（第一、二阶段）

日期：2026-10-17  
状态：第一阶段完成；第二阶段完成首批校准、真实服务 smoke 和可审计 telemetry，融合写入和质量/长上下文性能 gate 尚未完成

## 背景和目标

上游 mini-sglang #132 的方向是把 KV cache 从 BF16 改成 FP8，以降低 KV 存储字节数。
L20 是 Ada SM89：本实验选择 FlashInfer 路径，显式保持 query 为 BF16、KV 为
`float8_e4m3fn`。目标不是只看“显存少了”，而是把省下的显存重新换成更多可服务的 KV
tokens，并验证 attention kernel 能正确读回 FP8 cache。

第一阶段采用一个显式的全局标量 scale：写入时存储 `clamp(K / scale)`，FlashInfer
运行时传入相同的 `k_scale/v_scale`。这是为了先隔离 dtype、布局、页数计算和 kernel
兼容性；它不是最终的 per-layer/head calibrated v2 方案。

## 实现

- 新增 `--kv-cache-dtype {auto,float16,bfloat16,float8_e4m3fn}`，默认 `auto`，因此
  现有 BF16/FP16 行为不变。
- 新增 `--kv-cache-scale`，要求为正数；FP8 默认只作为显式实验开关使用。
- 新增 `--kv-cache-calibration {fixed,first_batch}`。`first_batch` 在每层第一次 eager
  写入时估计 `max(abs(K), abs(V)) / (448*0.95)`，随后固定该层 scale。
- `Context` 保存 model/query dtype，FlashInfer metadata 分离 `q_data_type` 与
  `kv_data_type`，不再把两者强制成同一个 dtype。
- `MHAKVCache.store_kv()` 在 FP8 模式执行有限值 clamp、cast 和现有 `store_cache`
  写入；当前 cast 与 store 仍是两个步骤，尚未实现 fused cast/store CUDA kernel。
- FP8 KV 仅允许 FlashInfer、`page_size=1`；其它 attention backend 会在启动时明确报错。
- 修正 `_determine_num_pages()`：页大小计算使用实际 KV dtype 的 `itemsize`。此前按模型
  BF16 计算，FP8 只能降低实际显存占用却不能增加可用 KV tokens，这是本实验中发现并
  修复的 capacity accounting bug。

`first_batch` 不在 CUDA Graph capture 阶段调用 `.item()`：capture 中保留 CLI fallback
scale，校准延迟到第一次 eager prefill。正式校准实验使用 `--cuda-graph-max-bs 0`，避免
dummy graph 激活污染统计。

## 离线和 CUDA 正确性

```text
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
.venv/bin/pytest -q tests/core
99 passed
```

额外 smoke 使用 L20（SM89）GPU 2：

- FP8 cache 的实际元素大小为 1 byte；`MHAKVCache.store_kv()` 写入后，反量化结果保持
  finite，测试样本 scale=0.25 的最大绝对误差为 K 0.0625、V 0.125。
- FlashInfer `BatchDecodeWithPagedKVCacheWrapper` 以
  `q_data_type=BF16`、`kv_data_type=float8_e4m3fn` 成功 plan/run，输出为 BF16 且
  `torch.isfinite(output).all() == True`。
- 新增 CLI、dtype、scale、cache allocation、首批校准和 summary 测试；核心测试合计 99 个通过。
- 独立 CUDA smoke 使用首批幅度 4.0，得到 `scale=0.0093984962`，第二批幅度 100.0
  不会改变已固定的 layer scale。

## L20 Qwen3-32B TP4 capacity 对照

硬件使用空闲 GPU 2/3/6/7，模型为 Qwen3-32B，权重为已有 FP8 W8A8 checkpoint，
FlashInfer、CUDA Graph max batch 1，`memory_ratio=0.9`。BF16 对照和 FP8 KV 使用相同的
模型与 graph 配置。

| 配置 | KV dtype | KV scale | KV tokens | K+V budget | 初始化后 free/card |
| --- | --- | ---: | ---: | ---: | ---: |
| 对照 | BF16 | 1.0 | 413,820 | 25.26 GiB | 4.40 GiB |
| 实验 | FP8 E4M3 | 0.25 | 827,641 | 25.26 GiB | 4.40 GiB |

FP8 capacity 为 BF16 的 2.000x（约 +100%，高于原计划的 +70% 目标）。两组 graph capture
后均约剩余 4.32 GiB/card，说明不是简单“少占显存”，而是按实际 dtype 重新分配了 KV
页数。FP8 服务在 `http://127.0.0.1:1962/v1/chat/completions` 返回 HTTP 200，rank-0
日志无 attention/kernel 异常。

## 061b：真实 prefill calibration smoke

在同一组 L20 GPU 2/3/6/7 上使用以下配置启动 graph-disabled 服务：

```text
--kv-cache-dtype float8_e4m3fn \
--kv-cache-scale 0.25 \
--kv-cache-calibration first_batch \
--cuda-graph-max-bs 0
```

Qwen3-32B FP8 W8A8 TP4 完成初始化，KV capacity 仍为 827,641 tokens，初始化后约剩
4.40 GiB/card；首个真实 prefill + decode 请求返回 HTTP 200，rank-0 scheduler 正常回到
idle。该结果证明 calibration 分支能和真实 FlashInfer 服务生命周期共存，但没有记录完整
各层 scale 分布，也没有证明输出质量或 ITL 改善。

## 结果解释和限制

本阶段接受以下结论：L20 SM89 + 当前 FlashInfer 版本支持 BF16 query/FP8 KV；cache 写入、
FlashInfer 读取、CUDA Graph 初始化和真实 API 请求均可工作；修正 page accounting 后，
KV capacity 约翻倍。

本阶段不接受以下部署结论：尚未完成 fused cast/store、长上下文 ITL/TTFT 矩阵和 held-out
quality gate。
因此 FP8 KV 仍保持显式 opt-in，不改变默认配置，也不能把本实验的 capacity 收益直接
等同为吞吐收益。

## 061c：FlashInfer attention 误差与 kernel 对照

新增 `benchmark/offline/bench_fp8_kv_attention.py`，在同一批 BF16 Q/K/V 上分别运行
BF16 KV 和 FP8 E4M3 KV，统计 attention 输出的 relative L2、cosine、饱和率和 decode
kernel 时间。参数为 L20 SM89、GQA 32/8、head_dim=128、`activation_std=0.2`、seed
`6100042`；这是 kernel/量化误差实验，不等同于模型真实激活质量。

| Context | Scale | Saturation | Relative L2 | Cosine | BF16 ms | FP8 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.25 | 0% | 2.760% | 0.999621 | 0.0462 | 0.0675 |
| 8,192 | 0.0025 | 0% | 2.810% | 0.999607 | 0.0302 | 0.0444 |
| 8,192 | 0.00125 | 0.503% | 4.275% | 0.999106 | 0.0302 | 0.0452 |
| 16,384 | 0.002588 calibrated | 0% | 2.825% | 0.999601 | 0.0408 | 0.0444 |

`scale=0.25` 在这组分布中没有饱和，但相对误差仍约 2.8%；calibrated scale 主要避免
极值饱和，不能消除 E4M3 固有的相对量化误差。FP8 decode 在 2k/8k 的小 batch kernel
上比 BF16 慢约 46%，到 16k context 差距缩小到约 9%。这说明 FP8 KV 的主要确定性收益
仍是容量，不应直接宣传为 ITL 加速；真实模型质量和端到端 ITL 仍需单独 gate。

## 061d：真实模型校准 telemetry

为了避免只凭服务 HTTP 200 判断校准是否有效，新增可选参数
`--kv-cache-calibration-summary /path/kv.json`。每个 TP rank 在 scheduler 优雅退出时写出
独立的 `kv.rank{rank}.json`，记录每层 scale、是否完成校准、写入元素数和饱和元素数。
该参数默认关闭；未指定时不执行饱和统计，避免普通服务路径增加同步和 `.item()` 开销。

在 L20 GPU 2/3/6/7 上对已有 Qwen3-32B FP8 W8A8 checkpoint 做了一次真实请求采集，配置为：

```text
--attention-backend fi --quantization fp8 \
--kv-cache-dtype float8_e4m3fn --kv-cache-scale 0.25 \
--kv-cache-calibration first_batch \
--kv-cache-calibration-summary /tmp/exp061d_kv_summary.zXmLiD/kv.json \
--cuda-graph-max-bs 0
```

服务返回 HTTP 200，4 个 rank 均生成 summary；每 rank 64 层全部完成首批校准。汇总结果如下：

| 指标 | 结果 |
| --- | ---: |
| TP ranks / layers | 4 / 256 |
| scale 最小值 / 最大值 | 0.01681 / 0.26668 |
| scale 中位数 | 0.05228 |
| 累计写入元素 | 6,422,528 |
| 累计饱和元素 | 145 |
| 加权饱和率 | 0.00226% |

各 rank 的饱和率分别为 0.00336%、0.00212%、0.00187% 和 0.00168%。这说明
`max(abs(K), abs(V)) / (448*0.95)` 的首批逐层校准在这次真实短请求中没有明显极值截断；但
该采集只覆盖一次短 prefill，不能替代 4096/8192/16384 长上下文、held-out perplexity 或
端到端 TTFT/ITL gate。summary 也只统计写入前的饱和候选，不能证明量化误差已经可接受。

## 下一步

1. 在真实 Qwen3 激活上增加 BF16 KV vs FP8 KV 的 held-out quality/error gate，并覆盖
   4096/8192/16384 上下文，避免把一次短请求 telemetry 当成部署结论。
2. 将 BF16-to-FP8 cast 与 page store 合并为 CUDA kernel，测量写入开销和 prefill TTFT。
3. 在短输出和长输出 workload 中做 BF16 KV vs FP8 KV 的
   ITL、TTFT、吞吐和质量对照；若 quality gate 通过，再进入默认策略评审。
