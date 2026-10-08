# Experiment 022: Page-aligned partial-leaf eviction

## Status

Complete. The opt-in implementation passed every pre-registered primary rule in three fresh-server
repetitions. It is retained as an experimental primitive, but is not yet a deployment
recommendation because exact trimming increased hot-probe TTFT by 16.23%.

## Objective

Reduce the 3.061x pressure-eviction amplification measured in experiment 021 without changing LRU
ordering. Preserve the useful prefix of an old compressed leaf when only its tail pages are needed.

## Principle

The radix tree stores a compressed path segment in each node. Current eviction removes a complete
least-recently-used leaf even when the allocator needs only a small fraction of that leaf. For an
evictable leaf with no live handles, removing a page-aligned suffix is structurally valid:

1. The first page of the retained key does not change, so the parent's child lookup key is stable.
2. Key and KV-index tensors are shortened by the same aligned amount.
3. A later match can reuse the retained prefix and insert the missing suffix as a new child.
4. Protected nodes and internal nodes remain untouched.

If earlier whole leaves do not satisfy the request, the final selected leaf may be trimmed by:

```text
trim_tokens = ceil(remaining_tokens / page_size) * page_size
```

The feature must be opt-in and default disabled. Existing whole-leaf behavior remains the control.

## Correctness requirements

Focused tests must cover:

1. Default whole-leaf eviction remains unchanged.
2. Page-size-1 trimming returns exactly the requested tail indices and preserves prefix matching.
3. Page-size-4 trimming rounds to a page, returns aligned pages, and keeps accounting valid.
4. A trimmed prefix can be extended again without duplicate or leaked KV indices.
5. Partial eviction never trims a protected or non-leaf node.
6. `free_pages + cached_pages == num_pages` after allocate/evict/reinsert cycles.

## Fixed online comparison

Use the exact pressure-revisit sequence and seeds from experiment 021:

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- KV pool: 4096 pages, page size 1
- Workload: fill A-F, refresh A/B, pressure G-I, probe I/H/G/B/A/F/E, probe D/C
- Prompt/output: 512/16 tokens
- Seeds: 3100042, 3101042, 3102042
- CUDA Graph maximum batch size: 64
- Scheduling: prefill streak 1, active budget 2304, result-before-prefill enabled
- One fresh server per seed

Experiment 021 is the already collected control. Only partial-leaf eviction changes in treatment.

## Primary metrics

| Metric | Experiment 021 control | Treatment target |
| --- | ---: | ---: |
| Hot survivor retention | 21/21 | 21/21 |
| Pressure amplification | 3.061x | <=1.05x |
| Pressure reclaimed tokens | 1,140.7 mean | At least 25% lower |
| D/C matched tokens | 3/519 each | Pooled D/C matches materially higher |
| Cache integrity failures | 0 | 0 |

Because exact trimming leaves less free slack, later allocation requests may be larger. Therefore
amplification and total reclaimed tokens are both required; a lower ratio alone is insufficient.

## Secondary metrics

- D/C probe TTFT should decrease when a partial prefix survives.
- Hot-prefix TTFT, TPOT, and output correctness must not regress materially.
- Report three-run means and sample standard deviations.

## Decision rules

Accept the opt-in implementation only if all correctness tests pass, all three online repetitions
retain every hot survivor, mean pressure reclaimed tokens fall by at least 25%, and pooled D/C
matched tokens increase. Keep the feature default disabled regardless of the result because the
experiment uses an intentionally constrained cache.

Reject or redesign it if tree/accounting invariants fail, hot retention decreases, or lower
amplification merely shifts equal-or-greater reclamation to the next allocation without preserving
additional prefix tokens.

## Planned artifacts

- Radix/cache manager implementation and focused tests
- CLI flag `--radix-partial-eviction`
- `learning/experiments/022_rep{1,2,3}_raw.md`
- `learning/results/022_rep{1,2,3}.json`
- `learning/results/022_rep{1,2,3}_telemetry.jsonl`

## Implementation

`RadixTreeNode.trim_tail()` retains the aligned prefix of an unprotected leaf and returns only its
tail KV indices to the page allocator. `RadixPrefixCache.evict()` first removes complete older
leaves, then trims the final leaf only when that can satisfy the remaining request without deleting
the node. The cache manager exposes this behavior through `--radix-partial-eviction`; the flag is
false by default and is rejected for the naive cache.

The retained key and value tensors are cloned before the node is shortened. This prevents a small
retained view from pinning the full original tensor storage, but makes repeated one-token trims a
potential metadata-copy cost. The online results below show that this cost matters when the pool is
kept exactly full.

## Results

### Primary cache metrics

The pressure phase includes only H and I (UIDs 10-11), before survivor and miss probes mutate the
cache.

| Repetition | Requested tokens | Reclaimed tokens | Amplification | D matched | C matched | Hot survivors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 773 | 773 | 1.000x | 335/519 | 3/519 | 7/7 |
| 2 | 771 | 771 | 1.000x | 347/519 | 3/519 | 7/7 |
| 3 | 771 | 771 | 1.000x | 357/519 | 3/519 | 7/7 |
| Mean / pooled | 771.7 | 771.7 | 1.000x | 346.3/519 | 3.0/519 | 21/21 |

Pressure reclaimed-token mean/sample standard deviation was `771.7 +/- 1.2`, versus
`1,140.7 +/- 1.2` in experiment 021. This is a 32.35% reduction and exceeds the 25% target.
Amplification fell from `3.0608 +/- 0.0064x` to exactly `1.0000 +/- 0.0000x`.

Pooled D/C matched tokens increased from 18 in the control to 1,048 in treatment. C remained the
oldest fully removed leaf, while D retained 332-354 additional non-template tokens depending on
prompt token content and allocation timing. Every I/H/G/B/A/F/E probe remained a 519/519 hit, so
useful retention stayed at 21/21.

All repetitions reached exactly 4,096 resident tokens without allocation or integrity failures.
The higher treatment request total is expected: whole-leaf control eviction created excess free
space, whereas partial eviction returned only what subsequent allocations actually required.

### Latency and operational cost

| Metric | Experiment 021 control | Partial-leaf treatment | Change |
| --- | ---: | ---: | ---: |
| Hot-survivor TTFT | 44.10 +/- 3.35 ms | 51.25 +/- 9.23 ms | +16.23% |
| D probe TTFT | 185.68 +/- 1.05 ms | 79.97 +/- 1.78 ms | -56.93% |
| C probe TTFT | 182.21 +/- 3.44 ms | 186.57 +/- 0.96 ms | +2.39% |
| Sequential output throughput | 33.54 token/s | 33.83 +/- 0.48 token/s | +0.86% |
| Average TPOT | 23.046 ms | 23.012 +/- 0.005 ms | -0.15% |

TTFT values use the same probe groups in both experiments. The D improvement corroborates its
partial match, while C remains cold. UID 1 had fresh-shape TTFT outliers in repetitions 2 and 3 and
is not part of these probe comparisons.

Exact trimming generated 32 pressure-phase eviction calls per repetition: each H/I prefill caused
one substantial trim and each of its 15 subsequent decode allocations caused another one-token
trim. Similar one-token trims continued during later probes. This behavior explains why exact token
reclamation can improve retained-prefix value while still increasing hot-request TTFT: the pool has
no headroom and `trim_tail()` repeatedly copies metadata tensors on the request path.

## Acceptance evaluation

| Rule | Result |
| --- | --- |
| Correctness and accounting tests pass | Pass: 19 focused/integration tests; 36 full core tests |
| Hot survivor retention is 21/21 | Pass: 21/21 at 519/519 tokens |
| Mean pressure reclamation drops at least 25% | Pass: -32.35% |
| Pressure amplification is at most 1.05x | Pass: 1.000x |
| Pooled D/C matches materially increase | Pass: 18 -> 1,048 tokens |
| Default remains disabled | Pass: explicit opt-in CLI flag |

## Interpretation

Partial-leaf eviction validates the experiment 021 diagnosis: the excess reclamation came from
compressed-node granularity, not incorrect LRU ordering. A page-aligned tail can be removed without
damaging tree matching, cache accounting, or the useful survivor set, and the retained D prefix
turns a cold request into a partial hit.

However, minimizing reclaimed tokens is not the same as minimizing serving latency. Exact trimming
eliminates all allocator slack, so autoregressive one-page allocations repeatedly enter eviction.
The next policy needs bounded headroom: reclaim a small page-aligned reserve on the first eviction,
then amortize that operation across following decode tokens. A second candidate is an O(1) metadata
slice with a bounded-storage compaction rule, but it would not reduce heap/telemetry call count by
itself.

## Decision

Accept the implementation as a default-disabled experimental primitive because every
pre-registered primary rule passed. Do not add it to the L20 deployment profile yet because the
16.23% hot-probe TTFT regression violates the secondary latency goal.

Experiment 023 should compare small eviction reserves (for example 16, 32, and 64 pages) using this
same workload. It must preserve 21/21 hot matches and most of D's retained prefix while reducing
pressure eviction calls from 32 to approximately two per repetition and restoring hot TTFT to the
experiment 021 range.

## Verification

- Three fresh Qwen3-32B BF16 TP=4 servers completed with seeds 3100042, 3101042, and 3102042
- Each JSON records `server_radix_partial_eviction=true`; each JSONL contains warmup UID 0 and
  measured UIDs 1-20
- Focused cache tests: 19 passed; complete `tests/core`: 36 passed
- Benchmark and test formatting, Python compilation, JSON parsing, and `git diff --check`: passed
- Post-run cleanup: no server listener on port 1919 and no GPU compute process
