# L20 optimization progress

## Current status

| Phase | Status | Result |
| --- | --- | --- |
| Environment | Complete | 8 x L20 detected; PyTorch 2.9.1+cu128; CUDA 12.8 JIT verified |
| Reproducible harness | Complete | Deterministic workload; Markdown/JSON output; 3-run statistics |
| CUDA Graph sizing | Complete | 23 -> 19 graphs; initialization 8.3615 -> 7.3553 s |
| CUDA Graph value | Complete | Enabled graph improves median throughput by 30.97% vs eager |
| Attention backend sweep | Complete | FA3 compatibility restored; `fa,fi` -0.17%, pure `fa` -15.00% vs `fi` median |
| KV/radix experiments | Partial eviction complete | Amplification 3.061x -> 1.000x; reclaimed tokens -32.35%; hot TTFT +16.23% |
| Online serving | Complete | TP=4 peaks at 503.35 token/s; latency knee between C=8 and C=32 |
| Prefill budget | Complete | 32768 cuts P99.9 TPOT 46.82% but regresses average TTFT 23.99% |
| TP 2/4/8 | Complete | TP=4 is efficiency optimum; TP=8 is 22% faster at 57% more total power |
| Scheduling fairness | Complete | Prefill streak 1 cuts worst decode stall 62% at 0.13% throughput cost |
| Adaptive prefill | Complete | Active budget 4096 halves worst TPOT with no measured throughput loss |
| Poisson budget sweep | Complete | 2048 reaches 100% TPOT SLO at 2.5 req/s with 0.12% throughput cost |
| Queue-adaptive prefill | Complete | 3 overload interventions restore 100% TPOT SLO with <0.01% throughput change |
| Mixed-length pressure | Complete | Static 2048 alone reaches 100% TPOT SLO; adaptive threshold 4096 triggers too late |
| Overload threshold 2048 | Complete | Restores 100% TPOT SLO but is behaviorally equivalent to static 2048 |
| Prefill stall-time model | Complete | 99 samples fit `11.19 + 0.31529 * tokens` ms with R-squared 0.99973 |
| Model-derived budget | Complete | 2304 keeps 100% TPOT SLO and improves average TTFT 6.10%/1.85% vs 2048 |
| Main scheduling experiment | Complete, criteria failed | 286/288 SLO passes; max TPOT 13.32s -> 1.34s, but light-load SLO and 2.5 req/s throughput criteria fail |
| Global prefill bound | Complete, rejected | Reproduces 1.34s gap under 2304 active budget; global 2304 does not improve 70/72 light-load SLO |
| Decode response visibility | Complete, accepted | Fresh held-out matrix passes 288/288 requests; max TPOT 1.12s -> 0.79s with at most 0.125% per-rate throughput cost |
| PyNCCL environment | Complete | Wheel `libnccl.so.2` linked with rpath; TP=4 API reached ready |

## Decisions

- Start with Qwen3-0.6B because it makes iteration cheap; do not generalize its absolute throughput
  results to production-size models.
- Use offline fixed-shape decode first to isolate engine behavior.
- Keep generated benchmark facts separate from interpretation notes.
- Do not accept parameter sweeps as source optimizations unless they lead to a robust automatic
  policy or a documented deployment profile.
- Keep SM89 automatic attention on `fi`: the repaired `fa,fi` path is effectively tied and pure
  `fa` is 15.00% slower on the fixed 256-input/128-output workload.
- Recommend `--max-prefill-streak 1 --decode-active-prefill-length 2304
  --decode-result-before-prefill` for the calibrated L20/Qwen3-32B TP=4 profile. Experiment 019's
  fresh matrix passed 288/288 requests with at most 0.125% per-rate throughput regression. Keep the
  generic default disabled because this policy is model, hardware, and workload dependent.
- Accept opt-in rank-0 Radix Cache telemetry. Shared-system reuse matches 776/1,036 tokens after the
  cold request and lowers post-first TTFT by 69.21% versus unique prompts. The baseline is far below
  cache capacity, so no eviction-policy claim is justified yet.
- Keep LRU ordering: under a controlled 4096-token pool it retained 21/21 expected useful prefixes
  across three seeds and evicted older C/D first. Target node granularity instead: pressure eviction
  reclaimed 3.061x the requested tokens because whole 532-token leaves were removed.
- Retain page-aligned partial-leaf eviction behind its default-disabled flag. It passed all primary
  experiment 022 rules: 21/21 hot survivors, 1.000x amplification, 32.35% less pressure reclamation,
  and pooled D/C matches increasing from 18 to 1,048 tokens. Do not recommend it for deployment yet:
  exact trimming caused 32 pressure eviction calls per repetition and regressed hot TTFT by 16.23%.

## Next experiment

Run experiment 023: add bounded eviction headroom to partial-leaf trimming and compare 16/32/64-page
reserves against experiments 021-022. The target is to retain most of D's partial prefix while
reducing pressure eviction calls from 32 to about two and restoring hot TTFT.
