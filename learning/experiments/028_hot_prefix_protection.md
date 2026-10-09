# Experiment 028: Hot-prefix protection for adaptive reserve

## Status

Complete, rejected for the tested open-loop trace. The design and acceptance rules below were fixed
before treatment data collection.

## Objective

Experiment 027 showed that age-aware weighting reduces allocator calls but still evicts a useful
survivor prefix. Experiment 028 adds a soft protection layer based on actual recent prefix matches.
Recently matched radix nodes are preferred over cold nodes during eviction; if cold nodes cannot
satisfy the request, the protection is relaxed so allocation correctness is preserved.

## Policy

The treatment uses raw adaptive reserve with a 64-page cap, plus a recent-match protection window.
The window stores the most recent distinct matched radix nodes. During eviction, unprotected leaves
are ordered first, then protected leaves by the existing LRU timestamp. The default remains zero,
which preserves the existing behavior.

## Workload and environment

The trace is identical to experiments 026 and 027: `A16 B64 C128 D16 E64 F128`, refresh `A/B` at
0.75 s, pressure `G/H/I` at 1.50 s, survivor probes `I/H/G/B/A/F/E` at 4.50 s, and evicted probes
`D/C` at 6.00 s. Prompts are 512 tokens, the pool is 4576 page-size-1 KV pages, and each cell uses
a fresh Qwen3-32B BF16 TP=4 server on L20 GPUs with CUDA Graph max batch size 64, prefill streak 1,
active budget 2304, and result-before-prefill enabled.

## Stage A screening

Seed 3100042 compares fixed 16, raw adaptive 64, hot protection with 4 recent matches, and hot
protection with 8 recent matches. Stage B repeats fixed 16 and the best passing treatment with
seeds 3101042 and 3102042 only if a treatment passes every Stage A rule.

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
- Raw reports: `learning/experiments/028_open_*_raw.md`
- Request JSON: `learning/results/028_open_*.json`
- Rank-0 telemetry: `learning/results/028_open_*_telemetry.jsonl`

## Stage A results

All four cells used seed 3100042 and fresh TP=4 servers. The fixed control and raw adaptive control
were rerun on the same implementation so the protection comparison uses paired fresh-cache cells.

| Policy | Calls | Target / actual | Survivor matches in trace order | Survivor TTFT | Throughput |
| --- | ---: | ---: | --- | ---: | ---: |
| Fixed 16 | 58 | 2037 / 2037 | I519 H519 G447 B519 A439 F519 E519 | 219.10 ms | 140.023 tok/s |
| Raw adaptive 64 | 18 | 1999 / 1999 | I519 H519 G519 B519 A398 F519 E519 | 321.35 ms | 139.661 tok/s |
| Hot protection 4 | 18 | 2089 / 2089 | I519 H519 G393 B519 A442 F519 E519 | 214.35 ms | 139.763 tok/s |
| Hot protection 8 | 18 | 2183 / 2183 | I519 H519 G519 B519 A383 F519 E519 | 197.45 ms | 139.234 tok/s |

All 80 requests completed without errors and every eviction exactly matched its target. Hot4 reduced
eviction calls by 69.0% but lost G's match (393 vs 447). Hot8 reduced calls by 69.0% but lost A's
match (383 vs 439). The protection window changed which prefix was discarded, but did not preserve
all future survivors. Raw adaptive 64 also showed a high survivor-TTFT outlier in this repeat;
throughput remained within 3% for every treatment. No Stage B repetitions were run.

## Decision

Reject the tested recent-match protection policy for this open-loop trace. A bounded list of recent
nodes is not a reliable proxy for future reuse because requests are admitted and matched while the
eviction loop is changing the tree; the window can rotate before the next probe arrives. Keep the
feature opt-in and disabled by default. The next design should protect cache entries using explicit
request-level reuse evidence or a measured hotness score with decay, rather than a fixed node count.
