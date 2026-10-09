# Experiment 026: Open-loop arrivals and adaptive reserve cap

## Status

Complete, rejected for open-loop use. The arrival trace, cap sweep, metrics, seeds, and acceptance
rules below were fixed before treatment data was collected.

## Objective

Experiment 025 validated adaptive-max-128 with phase barriers. Experiment 026 removes those barriers
and sweeps the cap to determine whether the policy remains stable when short and long requests are
admitted while older requests are still decoding.

## Workload and trace

The workload keeps experiment 024/025's prefixes and output lengths:
`A16 B64 C128 D16 E64 F128 G16 H64 I128`, with 512-token prompts and 4576 KV pages. All 20
requests are scheduled on one open-loop trace; clients do not wait for earlier responses:

| Offset | Prefixes | Count |
| ---: | --- | ---: |
| 0.00 s | A-F | 6 |
| 0.75 s | refresh A/B | 2 |
| 1.50 s | pressure G-I | 3 |
| 4.50 s | survivor probes I/H/G/B/A/F/E | 7 |
| 6.00 s | evicted probes D/C | 2 |

The offsets are relative to the first request and are reused exactly for every policy and seed.
They intentionally overlap the 16/64/128-token generations while preserving the cache-pressure
order. A warmup request is sent before the trace and excluded from measured request counts.

## Policies and stages

The fixed control uses partial eviction with a 16-page reserve. Adaptive treatments use the same
allocator with maximum reserve caps 64, 128, and 256 pages. All modes are opt-in and mutually
exclusive.

Stage A uses seed 3100042 to screen the three adaptive caps against fixed 16. Select the lowest cap
that passes all acceptance criteria; ties prefer lower actual reclamation, then lower cap. Stage B
repeats fixed 16 and the selected cap with seeds 3101042 and 3102042.

## Fixed environment

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Page size: 1; KV pool: 4576 pages
- CUDA Graph maximum batch size: 64
- Scheduling: prefill streak 1, active budget 2304, result-before-prefill enabled
- One fresh server for every policy/seed run

## Acceptance rules

For each adaptive cap, and then across the selected-cap Stage B repetitions:

- all 20 outputs complete with no integrity, allocation, or server error;
- no survivor probe loses matched tokens versus paired fixed 16;
- complete-trace eviction calls decrease by at least 25%;
- mean actual reclamation is no more than 5% above fixed 16;
- actual reclamation never exceeds target (1.000x target fidelity);
- open-loop output throughput regresses by no more than 3%;
- survivor-probe TTFT regresses by no more than 5%.

If every cap fails reclamation or latency, retain the phase-batched result from experiment 025 but do
not broaden the adaptive recommendation. The generic defaults remain disabled regardless of result.

## Implementation

`benchmark/online/bench_radix_open_loop_l20.py` generates the fixed trace, launches all requests
with scheduled sleeps, records real server UIDs, and emits per-request Markdown/JSON. Rank-0 cache
telemetry supplies eviction and prefix-match metrics. Completion order is not used to infer prefix
identity; analysis joins records by server UID.

## Planned artifacts

- `learning/experiments/026_*_raw.md`
- `learning/results/026_*.json`
- `learning/results/026_*_telemetry.jsonl`

## Stage A results

All four cells used seed 3100042 and fresh TP=4 servers. The fixed control is the paired reference;
the three adaptive caps were screened before any Stage B seed was selected.

| Policy | Calls | Target / actual | Survivor matches in trace order | Survivor TTFT | Throughput |
| --- | ---: | ---: | --- | ---: | ---: |
| Fixed 16 | 60 | 2078 / 2078 | I519 H519 G440 B519 A438 F519 E501 | 217.31 ms | 139.942 tok/s |
| Adaptive 64 | 18 | 2176 / 2176 | I519 H519 G519 B519 A383 F519 E519 | 198.26 ms | 139.230 tok/s |
| Adaptive 128 | 11 | 2202 / 2202 | I519 H519 G329 B519 A411 F519 E519 | 227.18 ms | 139.813 tok/s |
| Adaptive 256 | 9 | 2226 / 2226 | I519 H519 G519 B519 A303 F519 E413 | 388.37 ms | 139.184 tok/s |

All 80 requests completed successfully and every eviction exactly matched its target. However,
adaptive 64 lost A (383 vs 438), adaptive 128 lost G (329 vs 440) and A (411 vs 438), and adaptive
256 lost A (303 vs 438) and E (413 vs 501). The lower cap reduced calls by 70.0%, the middle cap by
81.7%, and the largest cap by 85.0%, but none preserved every paired survivor. Adaptive 256 also
increased survivor TTFT by 78.8% versus fixed control. Because the first acceptance rule failed for
all treatments, Stage B was not run.

## Decision

Reject adaptive reserve for this open-loop trace. The phase-barrier result from experiment 025 does
not generalize: when requests arrive while earlier decodes are active, summing remaining demand
into a reserve target changes which prefixes are discarded. A larger cap amortizes allocator calls
but spends more of the finite cache headroom and can worsen latency. Keep adaptive reserve opt-in and
do not use it as a generic open-loop policy. The next design should make reserve aware of request
age/priority or use a smaller per-batch contribution rather than a raw sum.

## Artifacts

- Harness: `benchmark/online/bench_radix_open_loop_l20.py`
- Raw reports: `learning/experiments/026_open_*_raw.md`
- Request JSON: `learning/results/026_open_*.json`
- Rank-0 telemetry: `learning/results/026_open_*_telemetry.jsonl`
