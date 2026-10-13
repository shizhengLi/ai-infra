# Experiment 049: Open-Loop Arrival Rates

## Question

Does the accepted TP4 Direct PyNCCL, graph-disabled profile maintain user-visible latency when
requests arrive over time instead of being released as a synchronized concurrency batch?

## Method

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3.
- Communication: `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.
- CUDA Graphs: disabled with `--cuda-graph-max-bs 0`.
- Scheduler: maximum prefill 2048, prefill streak 1, decode-active prefill budget 1024.
- Workload: 24 requests per trace, balanced input lengths 256/1024/4096, output length 128.
- Arrival process: seeded exponential inter-arrival offsets from `bench_poisson_l20.py`; clients do
  not wait for earlier responses.
- Arrival rates: 0.5, 0.9, 1.5, and 2.5 requests/s, three traces per rate.
- Seed: `4900042`.
- SLOs: TTFT <= 2000 ms and per-request maximum TPOT <= 1000 ms.

The open-loop trace mixes short and long prompts to expose queueing and prefill interference that a
closed synchronized batch cannot measure.

## Results

| Rate req/s | Output tok/s | P90 TTFT ms | TTFT SLO % | P99 TPOT ms | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 66.50 | 1793.48 | 94.4 | 295.28 | 100.0 | 87.64 |
| 0.9 | 89.07 | 1884.16 | 91.7 | 299.24 | 100.0 | 91.65 |
| 1.5 | 153.47 | 3489.45 | 59.7 | 324.79 | 100.0 | 93.92 |
| 2.5 | 171.03 | 4340.12 | 44.4 | 324.90 | 100.0 | 93.02 |

Average TTFT by input length across the same rate groups:

| Rate req/s | 256 tokens | 1024 tokens | 4096 tokens |
| ---: | ---: | ---: | ---: |
| 0.5 | 370 ms | 510 ms | 1483 ms |
| 0.9 | 346 ms | 889 ms | 1580 ms |
| 1.5 | 1449 ms | 1545 ms | 2436 ms |
| 2.5 | 2331 ms | 2089 ms | 2966 ms |

Per-request maximum TPOT stayed below the 1000 ms SLO for every request at every rate. The failure
is TTFT queueing: once the arrival rate reaches 1.5 requests/s, even short prompts wait behind
long-prefill work. At 2.5 requests/s, the server still produces more output tokens, but only 44.4%
of requests meet the TTFT SLO.

## Interpretation

The direct communication profile is stable under open-loop load at the token-generation level: no
trace developed a token gap over one second, and TPOT SLO attainment remained 100%. The limiting
resource is prefill admission and queueing, not decode progress. GPU utilization around 92-94% at
the high rates confirms that the workload is close to the available compute/PCIe service capacity.

The practical operating region for this mixed workload is at or below roughly 0.9 requests/s if a
2-second TTFT SLO is required. Increasing the rate to 1.5 or 2.5 requests/s raises output
throughput but violates the TTFT target. Closed-loop C=32 throughput alone is therefore
insufficient for a serving recommendation.

## Decision

- Keep the accepted Direct PyNCCL graph-disabled profile unchanged.
- Do not claim that its throughput result implies a higher sustainable request arrival rate.
- Treat 0.5-0.9 requests/s as the measured TTFT-SLO operating range for this mixed-length workload.
- Do not add a new scheduler policy from this experiment: this result identifies a capacity knee but
  does not isolate a safe prefill intervention.
- Proceed to Experiment 050, a fresh-seed held-out evaluation with longer output and both unique and
  shared-prefix requests.

## Artifacts

- `results/049_direct_seed4900042.json`
- `results/049_direct_seed4900042.md`
- `results/049_direct_telemetry.jsonl`
- Harness: `benchmark/online/bench_poisson_l20.py`
