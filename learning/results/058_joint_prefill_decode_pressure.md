# 实验058结果摘要：联合 Prefill/Decode 压力门控

新增的 active-decode 门槛使 overload budget 只在 Prefill 队列和 Decode 并发同时较高时
触发。`joint-512-a6` 在 1.5 req/s 达到 153.47 tok/s、2496.96 ms P90 TTFT 和 80.6%
TTFT SLO；相比 baseline 吞吐仅下降 0.22%，相比实验057 queue-only 的 151.68 tok/s 明显
恢复。P99.9 TPOT 为 235.16 ms，所有请求 TPOT SLO 均通过。

但是联合策略没有提高 TTFT SLO，最大 pending prefill 仍为 8472 tokens，因此不升级为 L20
默认配置。保留默认关闭的联合门控作为可观测、可调优机制，下一步停止固定阈值扫查，转向
admission control、等待时间/剩余 Prefill 感知排序或 Prefill/Decode 资源隔离。
