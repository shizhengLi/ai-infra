# L20 optimization progress

## Current status

| Phase | Status | Result |
| --- | --- | --- |
| Environment | Complete | 8 x L20 detected; PyTorch 2.9.1+cu128; CUDA 12.8 JIT verified |
| Reproducible harness | Complete | Deterministic workload; Markdown/JSON output; 3-run statistics |
| CUDA Graph sizing | Complete | 23 -> 19 graphs; initialization 8.3615 -> 7.3553 s |
| CUDA Graph value | Complete | Enabled graph improves median throughput by 30.97% vs eager |
| Attention backend sweep | Complete | FA3 compatibility restored; `fa,fi` -0.17%, pure `fa` -15.00% vs `fi` median |
| KV/radix experiments | Planned | Requires instrumentation and larger-model workload |
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
| Decode response visibility | Complete, candidate accepted | Result-before-prefill cuts max TPOT 1.34s -> 0.78s; 144/144 boundary requests pass at 0.060% overload throughput cost |
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
- Keep `--decode-result-before-prefill` opt-in until a fresh held-out matrix confirms experiment
  018's 144/144 TPOT result. The paired boundary matrix justifies carrying the candidate forward,
  but not changing the default from the trace family used to derive it.

## Next experiment

Run experiment 019: compare active-only 2304 with and without `--decode-result-before-prefill` on
three fresh seeds at 0.5/0.9/1.5/2.5 requests/s. Promote the policy only if every request meets the
TPOT SLO and per-rate mean throughput regression remains within 1%.
