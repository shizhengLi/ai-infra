# 实验 059：Page-aware admission 第一阶段

日期：2026-10-14  
状态：阶段 1 和短在线验证完成；decode retraction 和完整 L20 压测待做

## 背景与假设

实验 056--058 表明，继续调小 prefill token budget 只会切分 batch，最大 pending prefill
pressure 仍然是 8472 tokens。当前 `PrefillAdder` 用 `extend_len + output_len` 估计新增
KV 需求，但底层 `CacheManager.allocate_paged()` 实际按 `page_size` 分配 page。这个单位不
一致会让 admission 逻辑难以解释，也为后续 safe decode retraction 留下错误的容量边界。

本阶段假设：把估算抽成纯策略，并显式按 page 向上取整，可以在不改变默认行为的前提下，为
后续闭环 admission 和 retraction 提供可测试的容量模型。

## 实现

- 新增 `PageAwareReserveEstimator`，输入 cached prefix、完整 prompt 和最大输出预算，按
  `ceil(final_sequence/page_size) - ceil(cached_prefix/page_size)` 返回新增 page 数；这会
  正确处理 prefix cache 末页未满的情况。
- 增加可选 `new_token_ratio` 和 EMA `observe_output()`，用于下一阶段用真实 decode 输出
  修正最大输出预算；默认比例为 1.0，最小比例默认为 0.5。
- `PrefillAdder` 在 opt-in 模式下的两次容量检查和 `reserved_size` 累计使用同一 page-aware
  估算，避免检查与累计采用不同单位。
- 新增 CLI：`--prefill-reserve-policy token|page-aware` 和
  `--prefill-new-token-ratio`。默认 `token`，因此现有部署行为不变。
- 本阶段没有实现 decode retraction，也没有启用在线自适应更新；`observe_output()` 仅作为
  可单测的策略接口，避免在未定义 request lifecycle 之前把 mutable 状态放进调度热路径。

## 离线验证

执行环境：仓库 `.venv`，Python 3.12.14，pytest 9.1.1。项目默认 pytest 配置引用了未安装
的 `pytest-cov`，因此使用 `-o addopts=''` 关闭项目级 coverage 参数运行测试。

```text
.venv/bin/pytest -o addopts='' -q tests/core
87 passed in 4.88s
```

新增策略和配置相关用例单独结果：

```text
.venv/bin/pytest -o addopts='' -q \
  tests/core/test_prefill_reserve.py \
  tests/core/test_schedule_policy.py \
  tests/core/test_engine_config.py
33 passed in 4.23s
```

覆盖的边界包括：

| 场景 | 结果 |
| --- | --- |
| `page_size=16` 的非整页需求 | 正确向上取整 |
| 已命中 prefix cache | 只估算未缓存输入加输出 |
| prefix cache 末页未满 | 按结束页差计算，不额外预留整页 |
| `page_size=1` | 与旧的 token 估算一致 |
| 输出比例 EMA 与最小下限 | 正确更新且不低于安全下限 |
| 非法 page/ratio/update 配置 | 正确拒绝 |
| CLI page-aware 参数 | 正确映射到 `SchedulerConfig` |
| 现有 cache/scheduler core tests | 87/87 通过 |

## 当前结论

阶段 1 接口和容量数学模型通过离线验证，默认路径零行为变化。由于 L20 上 GPU 0--7 当前
均有外部任务占用，本轮没有启动服务、没有抢占或终止外部进程，也没有伪造 TTFT/吞吐结果。
因此不能宣称实验 059 的 P90 TTFT -15%、pending peak -20% 目标已经达成。

## L20 短在线验证

随后 GPU 2/3/6/7 空闲，使用这四张卡进行了同一服务配置的 A/B 对照。GPU 0/1/4/5 的外部
任务保持不变。服务启动时显式设置 `CUDA_HOME=/usr/local/cuda-12.8` 和对应 `PATH`，因为
系统 `/usr/bin/nvcc` 是 CUDA 11.5，不能编译 PyNCCL 所需的 C++20；这只改变本次进程环境，
没有修改系统 CUDA 链接。

固定配置：Qwen3-32B、FP8 W8A8、BF16 runtime、TP4、FlashInfer、CUDA Graph max batch 32、
16384 KV tokens、`max_prefill_length=4096`、`max_prefill_streak=1`、decode-active prefill
1024、GPU 2/3/6/7。请求为 256/1024/4096 三种输入长度均衡混合，64 输出 token，1.5
req/s，8 请求，2 次重复，seed `5900042`。两组均为 16/16 请求完成，TTFT 和 TPOT SLO
达成率均为 100%。

| policy | throughput tok/s | avg TTFT ms | P90 TTFT ms | avg TPOT ms | peak MiB/card |
| --- | ---: | ---: | ---: | ---: | ---: |
| token | 74.10 +/- 8.48 | 697.40 | 1394.32 | 28.77 | 16413 |
| page-aware | 73.77 +/- 8.31 | 711.54 | 1422.28 | 28.80 | 16413 |
| page-aware vs token | -0.45% | +2.03% | +2.01% | +0.11% | 0 |

结果文件：

- `learning/results/059_token_poisson.json` / `.md`
- `learning/results/059_page_aware_poisson.json` / `.md`

本次服务的 `page_size` 仍为 1，因为当前 FlashInfer attention 路径还限制 page size >1。
在 `page_size=1` 且 `new_token_ratio=1.0` 时，page-aware 估算退化为旧的
`extend_len + output_len`，所以短在线 A/B 的预期是等价而非加速。观测到的 -0.45% 吞吐和
+2.01% P90 TTFT 在两次重复和短 trace 的噪声范围内，结论是 **无回归、未证明性能收益**。
真正能体现 page-aware admission 差异的实验需要先完成 FlashInfer page size >1，或接入
decode 实际输出比例和 retraction；不能把本次结果包装成主性能 PR 已通过。

## 下一步

1. 为 `Req`/`PendingReq` 增加 generation/retracted 生命周期标记，设计 decode retraction 的
   原子回放路径。
2. 在 scheduler offline replay 中先验证“撤回后不发送 stale result、不重复释放 page、不破坏
   page table”的 correctness gate。
3. GPU 空闲后使用 L20 FP8 TP4 的 256/1024/4096 prompt、shared/unique prefix 和长输出
   workload，比较 token policy 与 page-aware policy 的 TTFT、TPOT、pending peak、吞吐和
   显存。
