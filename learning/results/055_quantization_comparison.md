# 实验055结果摘要：held-out 量化对比

## 结论

在 Qwen3-32B/L20 的相同开放环混合长度 workload 上，FP8 TP4 比 BF16 TP4 的 P90 TTFT、平均
TPOT 和显存更好；AWQ TP1 只适合容量优先场景，当前配置不适合延迟敏感在线服务。

## 核心数据

| 配置 | 0.5 req/s TTFT SLO | 1.0 req/s TTFT SLO | 1.5 req/s TTFT SLO | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| BF16 TP4 | 97.2% | 66.7% | 72.2% | 19109 MiB/card |
| FP8 TP4 | 100.0% | 97.2% | 97.2% | 16453 MiB/card |
| AWQ TP1 | 61.1% | 27.8% | 22.2% | 24265 MiB |

held-out MMLU（seed `5500042`，570 题）为 BF16 81.9298%、FP8 82.1053%、AWQ 82.2807%，三者
解析率均为 100%。由于实验054旧 seed 仍显示 AWQ 低于质量 gate，该结果只支持“未观察到新
回归”，不支持把 AWQ 设为默认。

