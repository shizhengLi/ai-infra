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
| Online serving | Planned | TTFT/TPOT/percentile workload |
| TP 2/4/8 | Planned | Requires topology and communication profiling |

## Decisions

- Start with Qwen3-0.6B because it makes iteration cheap; do not generalize its absolute throughput
  results to production-size models.
- Use offline fixed-shape decode first to isolate engine behavior.
- Keep generated benchmark facts separate from interpretation notes.
- Do not accept parameter sweeps as source optimizations unless they lead to a robust automatic
  policy or a documented deployment profile.

## Next experiment

Add online benchmark automation for TTFT/TPOT and changing batch sizes. Repair the FlashAttention
dependency stack separately so package changes do not contaminate the graph measurements.
