# Experiment 029: Decayed request-level hotness

## Status

Complete, rejected for the tested open-loop trace. The design and acceptance rules below were fixed
before Stage A data collection.

## Objective

Experiments 027 and 028 showed that request age and a fixed recent-node window do not reliably
identify future cache reuse. Experiment 029 gives each matched radix node a reuse score. Every match
decays existing scores and adds one point to the matched node and its ancestors. Eviction orders
unprotected leaves by increasing score, then by the existing LRU timestamp.

## Policy

The treatment uses raw adaptive reserve with a 64-page cap and a positive per-match decay factor.
The decay is explicit and opt-in; zero preserves the previous behavior. A score is evidence of
repeated prefix reuse, not a hard lock, so cold entries eventually become eligible for eviction.

## Workload and environment

The trace is identical to experiments 026-028: `A16 B64 C128 D16 E64 F128`, refresh `A/B` at
0.75 s, pressure `G/H/I` at 1.50 s, survivor probes `I/H/G/B/A/F/E` at 4.50 s, and evicted probes
`D/C` at 6.00 s. Prompts are 512 tokens, the pool is 4576 page-size-1 KV pages, and each cell uses
a fresh Qwen3-32B BF16 TP=4 server on L20 GPUs with CUDA Graph max batch size 64, prefill streak 1,
active budget 2304, and result-before-prefill enabled.

## Stage A screening

Seed 3100042 compares fixed 16, raw adaptive 64, decayed hotness with decay 0.90, and decayed
hotness with decay 0.99. Stage B repeats fixed 16 and the best passing treatment with seeds 3101042
and 3102042 only if a treatment passes every Stage A rule.

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
- Raw reports: `learning/experiments/029_open_*_raw.md`
- Request JSON: `learning/results/029_open_*.json`
- Rank-0 telemetry: `learning/results/029_open_*_telemetry.jsonl`

## Stage A results

All four cells used seed 3100042 and fresh TP=4 servers.

| Policy | Calls | Target / actual | Survivor matches in trace order | Survivor TTFT | Throughput |
| --- | ---: | ---: | --- | ---: | ---: |
| Fixed 16 | 57 | 1990 / 1990 | I519 H519 G519 B519 A438 F519 E519 | 275.96 ms | 138.909 tok/s |
| Raw adaptive 64 | 20 | 2071 / 2071 | I519 H519 G519 B519 A376 F519 E519 | 208.57 ms | 139.899 tok/s |
| Decay 0.90 | 17 | 2001 / 2001 | I519 H519 G519 B519 A397 F519 E519 | 258.53 ms | 139.480 tok/s |
| Decay 0.99 | 19 | 2105 / 2105 | I519 H519 G391 B519 A444 F519 E519 | 206.46 ms | 139.335 tok/s |

All 80 requests completed without errors and every eviction exactly matched its target. Decay 0.90
reduced calls by 70.2% versus fixed control but lost A's match (397 vs 438). Decay 0.99 reduced
calls by 66.7% but lost G's match (391 vs 519). Both treatments stayed within the throughput and
survivor-TTFT limits, but neither passed survivor integrity. No Stage B repetitions were run.

## Decision

Reject this decayed hotness ranking for the tested open-loop trace. Request-level reuse frequency is
still not enough to predict the next probe: the slower decay protects older prefixes and harms G,
while the faster decay lets A be reclaimed. Keep the option opt-in and disabled by default. The
next design should use explicit admission or workload-level reuse identities, or stop pursuing
generic open-loop reserve adaptation and retain the phase-batched profile only.
