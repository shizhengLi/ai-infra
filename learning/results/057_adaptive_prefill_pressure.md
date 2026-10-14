# 实验057结果摘要：动态 Prefill Budget

现有的 pending-pressure overload 分支可以正常触发，但当前三组参数都没有改善高负载 TTFT
SLO。baseline 在 1.5 req/s 为 153.81 tok/s、P90 TTFT 2489.72 ms、TTFT SLO 80.6%；
4096/512 动态配置为 151.68 tok/s、2608.29 ms、80.6%。更早触发或使用 256 budget 会继续
降低吞吐并提高 TTFT。

Telemetry 显示动态预算确实生效，但最大 pending prefill 仍为 8472 tokens，说明固定小
budget 只是拆分工作，没有降低队列峰值。拒绝将该配置设为默认；下一步应联合 active decode
数量和请求剩余 prefill 长度设计策略。
