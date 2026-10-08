# Experiment 023: Amortized partial eviction with bounded reserve

## Status

Complete. The pre-registered screen selected a 16-page reserve, and all primary and latency rules
passed across three fresh-server repetitions. The setting is accepted for the fixed
Qwen3-32B/L20/16-output-token experiment profile while remaining opt-in and disabled by default.

## Objective

Retain experiment 022's partial-prefix benefit while avoiding repeated one-token eviction. When an
allocation first exhausts the free list, reclaim a small page-aligned reserve in addition to the
immediate deficit so following decode allocations can consume headroom without entering the radix
eviction path again.

## Principle

For an immediate deficit of `requested_tokens` and configured reserve of `reserve_pages`:

```text
target_tokens = min(
    requested_tokens + reserve_pages * page_size,
    currently_evictable_tokens,
)
```

The allocator must still receive at least the immediate deficit. Partial-leaf eviction should
reclaim exactly the policy target when page alignment permits. Telemetry must keep three quantities
separate:

1. `requested_tokens`: the immediate allocator deficit.
2. `target_tokens`: deficit plus bounded policy reserve.
3. `evicted_tokens`: actual radix reclamation.

`evicted / requested` measures total headroom cost, while `evicted / target` continues to expose
node-granularity overshoot. A reserve is valid only when partial-leaf eviction is enabled. Both
features remain disabled by default.

## Correctness requirements

Focused tests must cover:

1. Zero reserve preserves experiment 022 behavior.
2. A reserve leaves the configured number of free pages after satisfying the allocation.
3. The reserve is capped by currently evictable pages and can never make allocation fail.
4. Page sizes greater than one remain aligned.
5. Telemetry separately accumulates immediate requests, policy targets, and actual eviction.
6. Positive reserve is rejected unless radix partial eviction is enabled.
7. Cache integrity remains `free_pages + cached_pages == num_pages`.

## Fixed environment and workload

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- KV pool: 4096 pages, page size 1
- CUDA Graph maximum batch size: 64
- Scheduling: prefill streak 1, active budget 2304, result-before-prefill enabled
- Workload: experiment 021/022 pressure-revisit sequence
- Prompt/output: 512/16 tokens, greedy sampling
- Seeds: 3100042, 3101042, 3102042
- One fresh server for every treatment repetition

Experiment 022's three `reserve=0` repetitions are the exact zero-reserve control and will not be
rerun.

## Screening design

Run reserves 16, 32, and 64 pages once with seed 3100042. Select one reserve using this fixed
lexicographic rule:

1. Reject any treatment that loses a hot survivor or has an integrity failure.
2. Prefer the lowest pressure-phase eviction-call count.
3. Among ties, prefer the lowest pressure reclaimed-token total.
4. Among remaining ties, prefer the highest D/C matched-token total.
5. If still tied, prefer the smaller reserve.

Latency is reported during screening but cannot override cache correctness or the selection rule
after data is visible. The selected reserve is then repeated with seeds 3101042 and 3102042.

## Final acceptance rules

Accept the selected reserve for the experimental L20 profile only if all three repetitions:

- preserve all 21/21 pooled hot survivors at 519/519 tokens;
- reduce pressure eviction calls from 32 to at most 2 per repetition;
- keep mean pressure reclamation no more than 5% above experiment 022's 771.7 tokens;
- retain at least 310/519 mean matched tokens for D;
- complete without allocation, accounting, or output-correctness failures.

The latency goal is pooled hot-survivor TTFT no more than 5% above experiment 021's 44.10 ms. If
the cache criteria pass but latency does not, retain the mechanism as an experimental option and
investigate metadata-copy cost rather than claiming an end-to-end improvement.

## Planned artifacts

- CLI/config option `--radix-partial-eviction-reserve-pages`
- Extended eviction telemetry with policy target tokens
- Focused reserve and telemetry tests
- `learning/experiments/023_r{16,32,64}_seed3100042_raw.md`
- Two additional raw reports for the selected reserve
- Paired JSON and rank-0 telemetry JSONL files under `learning/results/`

## Implementation

`CacheManager` now accepts `radix_partial_eviction_reserve_pages`. On cache exhaustion it computes
the immediate deficit, adds the configured page reserve, caps the target at current evictable
capacity, and asks the radix cache to reclaim that target. Positive reserve is rejected unless
partial radix eviction is enabled; negative reserve is always rejected.

Cache telemetry adds cumulative and per-operation `target_tokens`. Existing
`eviction_requested_tokens` continues to mean the immediate allocation deficit, so old metrics do
not silently change meaning. The server and benchmark expose and record
`--radix-partial-eviction-reserve-pages`; its default is zero.

## Screening results

All screening runs used seed 3100042. Every treatment preserved 7/7 hot probes and completed
without integrity errors.

| Reserve pages | Pressure calls | Immediate deficit | Policy target | Actual reclaimed | D match | Hot TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 742 | 774 | 774 | 337/519 | 47.70 ms |
| 32 | 2 | 726 | 790 | 790 | 322/519 | 45.69 ms |
| 64 | 2 | 694 | 822 | 822 | 323/519 | 41.79 ms |

All three tied on the first two selection conditions: no correctness rejection and two pressure
calls. The next condition selects the lowest actual reclamation, so 16 pages wins with 774 tokens.
Its D retention was also highest. Latency did not participate in the choice, as pre-registered.

The immediate deficit decreases with a larger reserve because headroom from H remains available to
I. Policy target and actual reclamation are therefore the comparable cache-cost metrics, not the
immediate deficit by itself.

## Selected-candidate results

| Seed | Pressure calls | Immediate deficit | Target / reclaimed | Target overshoot | D match | C match | Hot survivors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3100042 | 2 | 742 | 774 / 774 | 1.000x | 337/519 | 3/519 | 7/7 |
| 3101042 | 2 | 740 | 772 / 772 | 1.000x | 354/519 | 3/519 | 7/7 |
| 3102042 | 2 | 740 | 772 / 772 | 1.000x | 354/519 | 3/519 | 7/7 |
| Mean / pooled | 2.0 | 740.7 | 772.7 / 772.7 | 1.000x | 348.3/519 | 3.0/519 | 21/21 |

Mean pressure reclamation was `772.7 +/- 1.2` tokens. This is only 0.13% above experiment 022's
exact-trim mean of 771.7, remains 32.26% below experiment 021's whole-leaf mean of 1,140.7, and is
well inside the pre-registered 5% allowance. Reclamation exactly matched the policy target in every
run, so bounded reserve introduced no additional node-granularity overshoot.

The selected reserve reduced pressure eviction calls from 32 to 2 per repetition (93.75%) while
preserving all hot prefixes. D averaged `348.3 +/- 9.8` matched tokens, exceeding both the 310-token
threshold and experiment 022's 346.3-token mean. Pooled D/C matches were 1,054 versus 1,048 for
exact trimming and 18 for whole-leaf control; the small difference from exact trimming is normal
cross-run allocation variation, not evidence that reserving pages creates cache capacity.

## Latency results

| Metric | Whole leaf (021) | Exact partial (022) | Reserve 16 | Versus 021 |
| --- | ---: | ---: | ---: | ---: |
| Hot-survivor TTFT | 44.10 ms | 51.25 ms | 45.62 +/- 5.13 ms | +3.47% |
| D probe TTFT | 185.68 ms | 79.97 ms | 80.22 +/- 2.62 ms | -56.80% |
| C probe TTFT | 182.21 ms | 186.57 ms | 186.81 +/- 0.63 ms | +2.52% |
| Sequential output throughput | 33.54 token/s | 33.83 token/s | 33.61 +/- 0.09 token/s | +0.22% |
| Average TPOT | 23.046 ms | 23.012 ms | 23.015 +/- 0.026 ms | -0.13% |

Hot TTFT is 3.47% above the experiment 021 control and passes the 5% latency limit. It improves
10.98% over exact trimming. D retains the partial-prefix latency benefit, while C remains fully
cold. Throughput and TPOT are effectively unchanged.

Across the complete 20-request sequence, reserve 16 needed only 5-6 eviction calls, compared with
101 calls for exact trimming in experiment 022 repetition 1. Pressure-phase call count is the
pre-registered primary metric because later D/C probes intentionally alter the cache.

## Acceptance evaluation

| Rule | Result |
| --- | --- |
| Hot survivor retention is 21/21 | Pass: 21/21 at 519/519 tokens |
| Pressure eviction calls are at most 2 per run | Pass: 2, 2, 2 |
| Mean reclamation is at most 5% above 771.7 | Pass: 772.7, +0.13% |
| Mean D match is at least 310/519 | Pass: 348.3/519 |
| No allocation/accounting/output failures | Pass |
| Hot TTFT is at most 5% above 44.10 ms | Pass: 45.62 ms, +3.47% |
| Full core tests pass | Pass: 41 tests |

## Interpretation

The measured problem in experiment 022 was lack of allocator hysteresis, not partial eviction
itself. Reserving 16 pages on the first miss converts fifteen following one-page allocations into
free-list operations. Because 16 matches this experiment's output length, it nearly eliminates
re-entry without materially increasing discarded prefix data.

The result is workload-specific. A fixed 16-page reserve is not automatically appropriate for
longer outputs, larger page sizes, concurrent batches, or a production-sized cache. The parameter
therefore remains explicit rather than becoming a generic default. A future adaptive policy should
derive headroom from expected near-term allocation and enforce an upper bound so batching cannot
discard excessive prefixes.

## Decision

Accept `--radix-partial-eviction --radix-partial-eviction-reserve-pages 16` for the calibrated
Qwen3-32B BF16, TP=4, 4096-page, 16-output-token L20 experiment profile. Keep both generic defaults
disabled. This setting resolves experiment 022's excessive eviction-call count and passes the
cache, latency, and correctness criteria without changing LRU ordering.

Experiment 024 should test generalization across output lengths 16/64/128 and mixed sequential
lengths. It should compare fixed reserve 16 with a bounded adaptive reserve derived from expected
near-term output allocation before any broader deployment recommendation.

## Verification

- Five fresh Qwen3-32B TP=4 servers completed: three screening treatments and two additional
  selected-candidate repetitions
- Every result has 20 measured requests and every telemetry file has warmup UID 0 plus UIDs 1-20
- Focused cache/telemetry/allocation tests: 24 passed; complete `tests/core`: 41 passed
- Python compilation, Black checks, CLI exposure, JSON/JSONL parsing, and `git diff --check`: passed
- Post-run cleanup: no listener on port 1919 and no GPU compute process
