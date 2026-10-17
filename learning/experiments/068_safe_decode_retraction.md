# 实验 068：Page-aware admission 的 safe decode retraction

日期：2026-10-17  
状态：阶段 1（生命周期与离线 correctness）完成；L20 压力在线结果待执行

## 为什么做

实验 059 已经把 prefill admission 从 token 粗略估算抽成了 page-aware estimator，但当
decode 请求已经占据大量未来 KV 预算时，新的 prefill 只能等待。单纯缩小 prefill batch
只能改变切分，不能释放 decode 的保留容量。本实验实现上游 PR-A 中更有价值的部分：在
admission 被容量阻塞时，安全地暂时暂停一个 decode 请求，把未来 reservation 让给 prefill。

这里的“撤回”不是取消请求，也不是从原 prompt 重新生成。请求已经发送给客户端的输出必须
保留；请求不会被释放或重做 prefill，而是暂时保留现有 KV/table、移出 decode batch，等
prefill 压力消失后原地恢复。

## 实现原理

### 1. 候选选择

`DecodeManager.retract_candidate()` 只从 `running_reqs` 中选择可继续 decode 的请求，按
`ceil(remain_len / page_size)`、剩余 token 数和 uid 排序，优先释放未来 KV 需求最大的请求。
调度器把当前尚未处理的 GPU batch 中的请求作为 `protected_reqs`，候选选择永远跳过这些
请求。这样 retraction 发生在 CPU 调度边界，不会修改正在执行的 CUDA batch。

### 2. 请求生命周期

- `Req` 增加 `retracted` 和 `generation` 标记；回收前先使旧 scheduled incarnation 失效。
- `Req.prompt_len` 保存原始 prompt 边界，供生命周期调试和 stale-result 防护使用。
- 请求不释放 table slot 和现有 KV；只从 `running_reqs` 移到 `paused_reqs`，并为恢复时的下
  一个 page 保留一页安全余量。
- 新请求继续使用同一个 uid，因此客户端不会看到新的请求；已经发送的 token 不会重复。
- pending prefill 清空后，原 `Req` 通过 `mark_resumed()` 回到 decode 集合，不需要重做
  prompt prefill。
- `_process_last_data()` 保留 `retracted` 检查，作为延迟 GPU 结果的最后一道防线。

### 3. 默认边界

新增 `--prefill-safe-retraction`，且只有在 `--prefill-reserve-policy page-aware` 时才允许
启用。默认关闭，旧 token admission 和旧 overlap 行为不变。在线使用时建议同时打开
`--decode-result-before-prefill`，让上一个 decode 结果先完成，再进入可回收的调度边界。

示例：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
  .venv/bin/python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp 4 --quantization fp8 --prefill-reserve-policy page-aware \
  --prefill-safe-retraction --decode-result-before-prefill
```

## Correctness 验证

环境：仓库 `.venv`，Python 3.12.14，pytest 9.1.1。项目默认 coverage 插件未安装，因此
用 `-o addopts=''` 运行。

```text
.venv/bin/pytest -o addopts='' -q tests/core/test_safe_retraction.py \
  tests/core/test_prefill_reserve.py tests/core/test_schedule_policy.py \
  tests/core/test_req_buffer.py
41 passed in 4.84s

.venv/bin/pytest -o addopts='' -q tests/core
109 passed in 5.85s
```

覆盖点包括：

| 检查 | 结果 |
| --- | --- |
| 最大 page demand 的候选选择 | 通过 |
| 当前 GPU batch 保护 | 通过 |
| retraction generation 标记 | 通过 |
| 已生成历史不重复、剩余 `max_tokens` 正确 | 通过 |
| CLI 只能显式启用 safe retraction | 通过 |
| 原有 `tests/core` | 109/109 通过 |

## L20 在线结果

固定配置为 Qwen3-32B、FP8、BF16 runtime、TP4、L20 GPU 2/3/6/7、4096 KV tokens、page
size 1、max prefill 2048、decode-active 512、decode-result-before-prefill。请求为
256/1024/2048 三种输入均衡混合，64 输出 token，seed `6800042`，每档 1 次重复。

| rate (req/s) | control tok/s | pause tok/s | 吞吐变化 | control P90 TTFT | pause P90 TTFT | retractions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 66.91 | 66.78 | -0.18% | 1234.77 ms | 1505.49 ms | 143 |
| 2.5 | 72.30 | 71.63 | -0.93% | 3766.34 ms | 3852.18 ms | 706 |

两组都完成全部 12 个请求、每个请求返回 64 个 token，服务 idle 时 page/cache integrity
检查没有报错；TPOT-SLO 在 pause 版本仍为 100%。但是 retraction 次数很高，说明在当前
page size 1 和限页 workload 下存在 reservation thrashing，P90 TTFT 没有改善，未达到
15% gate。因此本实验的结论是 **correctness 通过、性能候选拒绝**，`safe-retraction` 继续
保持默认关闭。第一次“释放后重新 prefill”的实现结果保存在
`learning/results/068_treatment_poisson.*`，吞吐仅 15.97/10.42 tok/s，证明必须保留
现有 KV 而不能重算。

原始文件：

- `learning/results/068_control_poisson.json`
- `learning/results/068_treatment_pause_poisson.json`
- `learning/results/068_control_telemetry.jsonl`（被 gitignore 的本地 telemetry）
- `learning/results/068_treatment_pause_telemetry.jsonl`（被 gitignore 的本地 telemetry）

## 当前结论

阶段 1 和一次 L20 压力对照证明了 retraction 的状态转换、默认兼容性和无重复输出，但当前
策略没有性能收益。后续若继续，应增加 pause hysteresis/最小收益阈值，避免同一批 pending
工作反复暂停和恢复；在此之前不把它作为默认调度策略。

在线 gate：retraction 不能产生重复 token、stale result、double free 或 page-table
integrity 错误；在吞吐不低于 control 98% 的前提下，混合 256/1024/4096 prompt 的 P90
TTFT 目标降低至少 15%。未完成在线压测前，不把本实验标记为主性能 PR 通过。
