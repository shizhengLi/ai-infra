# Experiment 025: Concurrent variable-length adaptive reserve

## Status

Complete, accepted for the next production-shaped validation stage. The workload, capacity,
policies, seeds, metrics, and acceptance rules below were fixed before treatment data was collected.

## Objective

Experiment 024 accepted the output-aware reserve on sequential mixed lengths, but its policy sums
remaining demand from every request admitted to a batch. This experiment tests that sum under actual
batched admission. The question is whether adaptive reserve reduces allocator re-entry without
over-reserving when several 16/64/128-token decodes overlap.

## Hypothesis and policy

For each paged allocation, adaptive reserve is:

```text
min(sum(ceil(req.remain_len / page_size) for req in admitted_batch), 128)
```

The fixed control uses 16 reserve pages. Partial-leaf eviction is enabled in both treatments. The
two reserve modes are mutually exclusive and both remain opt-in. The primary risk is not cache
correctness but excess reclamation caused by summing several requests that may soon finish.

## Workload

The nine prefixes use the same 512-token prompts and output mapping as experiment 024:
`A16 B64 C128 D16 E64 F128 G16 H64 I128`. Requests are submitted in concurrent groups, with a
barrier between groups:

1. Fill `A-F` concurrently (six requests).
2. Refresh `A/B` concurrently (two requests).
3. Pressure `G-I` concurrently (three requests).
4. Probe survivors `I/H/G/B/A/F/E` concurrently (seven requests).
5. Probe evicted `D/C` concurrently (two requests).

This gives 20 measured requests and concurrent allocation batches without changing the prefix
reuse order. A warmup request is sent before measurement and is excluded from telemetry analysis.

## Fixed environment

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- KV pool: 4576 pages, page size 1
- CUDA Graph maximum batch size: 64
- Scheduling: prefill streak 1, active budget 2304, result-before-prefill enabled
- Seeds: 3100042, 3101042, 3102042
- One fresh server per policy/seed pair

## Acceptance rules

Across the three paired repetitions, accept adaptive reserve for the next production-shaped test
only if:

- every run returns all 20 requested outputs with no integrity or server error;
- no survivor prefix loses matched tokens versus its paired fixed run;
- mean complete-sequence eviction calls decrease by at least 25%;
- mean actual reclamation is no more than 5% above fixed 16;
- actual reclamation never exceeds policy target (target overshoot remains 1.000x);
- concurrent output throughput regresses by no more than 3%;
- mean survivor-group TTFT regresses by no more than 5%.

If the adaptive policy meets the cache criteria but fails the latency or reclamation criterion,
retain it as a sequential-only option and reject a concurrency recommendation. The generic defaults
remain disabled in either case.

## Implementation

`benchmark/online/bench_radix_concurrent_l20.py` submits each workload group with
`asyncio.gather()`, records server UID, phase, prefix, output length, TTFT, TPOT, and group timing,
and emits Markdown plus JSON. Rank-0 cache telemetry remains the source of eviction calls,
requested/target/actual token counts, and per-prefix cache matches.

## Planned artifacts

- `learning/experiments/025_*_raw.md`
- `learning/results/025_*.json`
- `learning/results/025_*_telemetry.jsonl`
- focused benchmark validation tests, if the harness adds reusable logic

## Results

All six fresh-server runs returned all 20 requested outputs. Cache metrics come from rank-0
telemetry; benchmark metrics use the concurrent group barrier sequence. Prefix matches are joined by
the server UID because completion order inside a concurrent group is nondeterministic.

| Seed | Policy | Eviction calls | Requested | Target / actual | Survivor matches | Survivor TTFT | Throughput |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 3100042 | Fixed 16 | 42 | 1757 | 2429 / 2429 | 7/7 paired; E 415/519 | 133.17 ms | 78.245 tok/s |
| 3100042 | Adaptive 128 | 8 | 1597 | 2343 / 2343 | 7/7 paired; E 418/519 | 130.79 ms | 77.548 tok/s |
| 3101042 | Fixed 16 | 41 | 1636 | 2292 / 2292 | 7/7 paired; E 415/519 | 132.24 ms | 77.919 tok/s |
| 3101042 | Adaptive 128 | 9 | 1679 | 2439 / 2439 | 7/7 paired; E 418/519 | 132.80 ms | 77.606 tok/s |
| 3102042 | Fixed 16 | 42 | 1651 | 2323 / 2323 | 7/7 paired; E 415/519 | 133.78 ms | 77.483 tok/s |
| 3102042 | Adaptive 128 | 8 | 1283 | 2140 / 2140 | 7/7 paired; E 418/519 | 129.06 ms | 78.408 tok/s |
| Mean | Fixed 16 | 41.67 +/- 0.58 | 1681.3 | 2348.0 / 2348.0 | no paired loss | 133.06 ms | 77.882 tok/s |
| Mean | Adaptive 128 | 8.33 +/- 0.58 | 1519.7 | 2307.3 / 2307.3 | no paired loss | 130.88 ms | 77.854 tok/s |

Sample standard deviations for actual reclamation were 68.6 pages for fixed and 149.9 pages for
adaptive; throughput standard deviations were 0.384 and 0.460 token/s respectively. Adaptive
reduced complete-sequence eviction calls by 80.0% and actual reclamation by 1.73%. Throughput
changed by -0.036%, while survivor-group TTFT improved by 1.64%. Every eviction's actual count
matched its target exactly, so overshoot remained 1.000x.

## Acceptance decision

| Rule | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Complete outputs / integrity errors | 20 / 0 | 20 / 0 in all six runs | Pass |
| Paired survivor regressions | 0 | 0; all 21 pairs equal | Pass |
| Mean call reduction | >=25% | 80.0% | Pass |
| Mean reclamation increase | <=5% | -1.73% | Pass |
| Target overshoot | 1.000x | 1.000x | Pass |
| Concurrent throughput regression | <=3% | 0.036% | Pass |
| Survivor-group TTFT regression | <=5% | -1.64% | Pass |

The adaptive sum does not over-reserve in this batched workload. It preserves the same useful
prefixes as the fixed control while eliminating most repeated allocator entries. The E prefix is
partially retained at 418/519 in both policies, so that loss is caused by the shared capacity and
concurrent pressure pattern rather than adaptive treatment.

This validates adaptive reserve for the current production-shaped follow-up profile, but does not
make it a generic default. The test uses phase barriers and a fixed 128-page cap. Open-loop arrivals,
larger concurrent batches, and cap sensitivity remain future work. Keep both partial eviction and
adaptive reserve disabled by default.

## Artifacts

- Harness: `benchmark/online/bench_radix_concurrent_l20.py`
- Raw reports: `learning/experiments/025_concurrent_*_raw.md`
- Request JSON: `learning/results/025_concurrent_*.json`
- Rank-0 telemetry: `learning/results/025_concurrent_*_telemetry.jsonl`
