# L20 optimization progress

## Current status

| Phase | Status | Result |
| --- | --- | --- |
| Environment | Complete | 8 x L20 detected; PyTorch 2.9.1+cu128; CUDA 12.8 JIT verified |
| Reproducible harness | Complete | Deterministic workload; Markdown/JSON output; 3-run statistics |
| CUDA Graph sizing | Complete | 23 -> 19 graphs; initialization 8.3615 -> 7.3553 s |
| CUDA Graph value | Complete | Enabled graph improves median throughput by 30.97% vs eager |
| Attention backend sweep | Blocked | `sgl-kernel 0.3.21` incompatible with CUTLASS DSL 4.7.1 API |
| KV/radix experiments | Planned | Requires instrumentation and larger-model workload |
| Online serving | Complete | TP=4 peaks at 503.35 token/s; latency knee between C=8 and C=32 |
| TP 2/4/8 | Planned | Qwen3-32B TP=4 startup verified; performance profiling pending |
| PyNCCL environment | Complete | Wheel `libnccl.so.2` linked with rpath; TP=4 API reached ready |

## Decisions

- Start with Qwen3-0.6B because it makes iteration cheap; do not generalize its absolute throughput
  results to production-size models.
- Use offline fixed-shape decode first to isolate engine behavior.
- Keep generated benchmark facts separate from interpretation notes.
- Do not accept parameter sweeps as source optimizations unless they lead to a robust automatic
  policy or a documented deployment profile.

## Next experiment

Run experiment 006: increase `max_extend_tokens` from 8192 to 32768 at concurrency 32 and measure
TTFT plus P99.9/max TPOT stalls. Then use the selected prefill budget for TP=2/4/8 scaling.
