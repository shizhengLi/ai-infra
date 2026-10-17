# 实验 070：Retraction 滞回稳定性与 Overlap 容量边界

## 1. 实验问题

实验 069 只使用一次在线重复，虽然 retraction 次数显著下降，但还不能说明 cooldown=4、pending gate=2048 在不同随机到达轨迹下稳定。实验 070 做两件事：

1. 在关闭 overlap scheduling 的参考路径上重复 control 和 hysteresis，隔离 GPU/CPU overlap 对结果的影响。
2. 在开启 overlap 的限页压力下做更高负载验证，检查 safe retraction 是否能保证 KV admission 不触发 allocator 断言。

## 2. 配置

模型为 Qwen3-32B，FP8 W8A8，BF16 runtime，TP=4，L20 GPU 2/3/6/7，page size=1，4096 KV pages，max prefill 2048，decode-active prefill 512，max-prefill-streak=1，decode-result-before-prefill=true。

有效 A/B 使用 8 个请求、输入长度 256/1024/2048、输出 64 token、Poisson 到达率 1.5/2.5 req/s，每档 2 次重复，seed `7000042`：

- control：safe retraction cooldown=0、pending gate=0；
- treatment：cooldown=4、pending gate=2048；
- 两组均设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，确保容量压力不会被 overlap pipeline 的额外状态混入。

原始结果见 [`070_no_overlap_control_poisson.md`](../results/070_no_overlap_control_poisson.md) 和 [`070_no_overlap_treatment_poisson.md`](../results/070_no_overlap_treatment_poisson.md)。

## 3. 稳定性结果

| 到达率 | control 吞吐 | treatment 吞吐 | 变化 | control P90 TTFT | treatment P90 TTFT | 变化 | control TTFT SLO | treatment TTFT SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 req/s | 70.69 +/- 14.47 | 69.52 +/- 13.34 | -1.65% | 1322.42 ms | 1396.08 ms | +5.57% | 100% | 93.75% |
| 2.5 req/s | 84.08 +/- 3.79 | 84.21 +/- 3.79 | +0.16% | 3187.29 ms | 3185.09 ms | -0.07% | 68.75% | 68.75% |

遥测显示 retraction 从 control 的 2502 次、72104 token 降到 treatment 的 573 次、16429 token，分别下降 77.1% 和 77.2%。输出长度、HTTP 请求和 page integrity 均正常。

低负载点的 P90 TTFT 回退超过预注册的 5% gate，因此本实验不接受 hysteresis 作为独立性能优化；它只能作为降低 reservation thrashing 的条件性控制。高负载点没有性能回退，但没有足够证据抵消低负载回退。

## 4. Overlap 边界发现

随后在开启 overlap、12 个请求、相同 256/1024/2048 混合输入和 2.5 req/s 压力下运行 control。4096 pages 和 8192 pages 两种容量都出现 scheduler 子进程异常退出，错误为：

```text
AssertionError: Cannot satisfy 1/2/4 cache tokens with only 0 evictable
```

客户端因此无法完成完整 trace，未生成有效 benchmark JSON；这些运行不计入上面的 A/B 数值，也不能被包装成性能结果。关闭 overlap 后相同限页路径可完成请求，说明当前问题集中在 overlap pipeline 下的 admission/reservation 时序，而不是模型权重或 FlashInfer 初始化。

## 5. 结论与下一步

实验 070 的结论是：

- cooldown + pending gate 能稳定减少 retraction oscillation，但低负载有可见 TTFT 代价，默认继续关闭；
- safe retraction 在关闭 overlap 的窄容量路径上可以保持正确性；
- overlap 开启时存在需要优先修复的 KV admission 安全边界，当前不应继续扩大压力矩阵。

面试中可以说明：**性能优化必须先满足资源安全不变量。overlap 虽然隐藏 CPU/GPU 调度开销，但会让“已提交未完成 batch”的未来 KV 需求脱离普通 admission 视图；如果不追踪这部分 reservation，系统可能在 decode append 时才发现无可驱逐 page。** 下一实验应实现并验证 in-flight batch reservation 或 decode admission guard，再重新跑 overlap A/B。
