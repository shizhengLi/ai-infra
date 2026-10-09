# Experiment 027: Age-aware adaptive reserve

## Status

Complete, rejected for the tested open-loop trace. The design and acceptance rules below were fixed
before treatment data collection.

## Objective

Experiment 026 showed that the raw adaptive reserve, which sums every request's remaining output,
reduces eviction calls on open-loop arrivals but discards prefixes that are still useful. This
experiment tests whether discounting nearly finished requests can preserve survivors without giving
up the allocator-call reduction.

## Policy

The new opt-in `age-aware` mode uses each request's remaining-output fraction as its weight:

`weighted_pages = max(1, floor(ceil(remain_len / page_size) * remain_len / output_len))`

Each active request contributes at least one page while it still has output remaining. The weighted
sum is capped by the configured adaptive maximum. The existing `raw` mode and fixed 16-page
reserve are unchanged and are included as controls.

## Workload and environment

The trace is identical to experiment 026 so survivor matches can be compared directly:
`A16 B64 C128 D16 E64 F128`, refresh `A/B` at 0.75 s, pressure `G/H/I` at 1.50 s, survivor
probes `I/H/G/B/A/F/E` at 4.50 s, and evicted probes `D/C` at 6.00 s. Prompts are 512 tokens,
the pool is 4576 page-size-1 KV pages, and each cell uses a fresh Qwen3-32B BF16 TP=4 server on
L20 GPUs with CUDA Graph max batch size 64, prefill streak 1, active budget 2304, and
result-before-prefill enabled.

## Stage A screening

Seed 3100042 compares fixed 16, raw adaptive 64, age-aware adaptive 64, and age-aware adaptive
128. Stage B repeats fixed 16 and the best age-aware cap with seeds 3101042 and 3102042 only if
the treatment passes the survivor rule in Stage A.

## Acceptance rules

- all 20 outputs complete with no integrity, allocation, or server error;
- no survivor probe loses matched tokens versus paired fixed 16;
- complete-trace eviction calls decrease by at least 25% versus fixed 16;
- mean actual reclamation is no more than 5% above fixed 16;
- actual reclamation never exceeds target;
- open-loop output throughput regresses by no more than 3%;
- survivor-probe TTFT regresses by no more than 5%.

## Artifacts

- Harness: `benchmark/online/bench_radix_open_loop_l20.py`
- Raw reports: `learning/experiments/027_open_*_raw.md`
- Request JSON: `learning/results/027_open_*.json`
- Rank-0 telemetry: `learning/results/027_open_*_telemetry.jsonl`

## Stage A results

All four cells used seed 3100042 and fresh TP=4 servers. The fixed control was rerun because each
server starts from an empty cache and the open-loop timing is part of the treatment.

| Policy | Calls | Target / actual | Survivor matches in trace order | Survivor TTFT | Throughput |
| --- | ---: | ---: | --- | ---: | ---: |
| Fixed 16 | 58 | 2047 / 2047 | I519 H519 G441 B519 A439 F519 E519 | 217.92 ms | 139.883 tok/s |
| Raw adaptive 64 | 18 | 2089 / 2089 | I519 H519 G392 B519 A444 F519 E519 | 215.81 ms | 139.832 tok/s |
| Age-aware 64 | 20 | 2021 / 2021 | I519 H519 G519 B519 A369 F519 E519 | 207.35 ms | 139.834 tok/s |
| Age-aware 128 | 13 | 2079 / 2079 | I519 H519 G519 B457 A438 F519 E519 | 217.90 ms | 139.812 tok/s |

All 80 requests completed without errors and every eviction exactly matched its target. Age-aware 64
reduced eviction calls by 65.5%, but lost A's match (369 vs 439). Age-aware 128 reduced calls by
77.6%, but lost B's match (457 vs 519). Throughput stayed within 0.1% of fixed control and both
age-aware variants stayed within the TTFT limit, but the survivor-integrity rule failed for both.
No Stage B repetitions were run.

## Decision

Reject this age-only weighting for the tested open-loop trace. Discounting nearly finished requests
changes the eviction timing, but it does not identify which prefixes are safe to remove: at cap 64
it harms A, while at cap 128 it harms B. Keep the mode opt-in and do not broaden the deployment
recommendation. A useful next design needs cache-value or prefix-hotness information, not only
request age; a conservative candidate is to reserve based on request age while protecting recently
matched prefixes from the reserve target.
