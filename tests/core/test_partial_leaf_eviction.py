from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.kvcache.radix_cache import RadixPrefixCache
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def make_radix_cache(page_size: int = 1) -> RadixPrefixCache:
    core.set_global_ctx(core.Context(page_size=page_size))
    return RadixPrefixCache(device=torch.device("cpu"))


def insert(cache: RadixPrefixCache, tokens: list[int], indices: list[int]):
    return cache.insert_prefix(
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(indices, dtype=torch.int32),
    ).handle


def test_default_eviction_still_removes_whole_leaf() -> None:
    cache = make_radix_cache()
    tokens = list(range(10, 18))
    insert(cache, tokens, list(range(100, 108)))

    evicted = cache.evict(3)

    assert evicted.tolist() == list(range(100, 108))
    assert cache.match_prefix(torch.tensor(tokens, dtype=torch.int32)).cuda_handle.cached_len == 0
    assert cache.size_info.evictable_size == 0


def test_partial_eviction_trims_tail_and_preserves_prefix() -> None:
    cache = make_radix_cache()
    tokens = list(range(10, 18))
    insert(cache, tokens, list(range(100, 108)))

    evicted = cache.evict(3, partial=True)
    match = cache.match_prefix(torch.tensor(tokens, dtype=torch.int32)).cuda_handle

    assert evicted.tolist() == [105, 106, 107]
    assert match.cached_len == 5
    assert match.get_matched_indices().tolist() == list(range(100, 105))
    assert cache.size_info.evictable_size == 5


def test_partial_eviction_rounds_to_page_boundary() -> None:
    cache = make_radix_cache(page_size=4)
    tokens = list(range(12))
    insert(cache, tokens, list(range(12)))

    evicted = cache.evict(5, partial=True)
    match = cache.match_prefix(torch.tensor(tokens, dtype=torch.int32)).cuda_handle

    assert evicted.tolist() == list(range(4, 12))
    assert match.cached_len == 4
    assert cache.size_info.evictable_size == 4


def test_partial_eviction_removes_old_leaf_then_trims_next() -> None:
    cache = make_radix_cache()
    insert(cache, list(range(10, 14)), list(range(100, 104)))
    newer_tokens = list(range(20, 28))
    insert(cache, newer_tokens, list(range(200, 208)))

    evicted = cache.evict(6, partial=True)
    newer_match = cache.match_prefix(torch.tensor(newer_tokens, dtype=torch.int32)).cuda_handle

    assert evicted.tolist() == list(range(100, 104)) + [206, 207]
    assert newer_match.cached_len == 6
    assert newer_match.get_matched_indices().tolist() == list(range(200, 206))
    assert cache.size_info.evictable_size == 6


def test_trimmed_prefix_can_be_extended_again() -> None:
    cache = make_radix_cache()
    tokens = list(range(10, 18))
    insert(cache, tokens, list(range(100, 108)))
    cache.evict(3, partial=True)

    result = cache.insert_prefix(
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(list(range(100, 105)) + list(range(205, 208)), dtype=torch.int32),
    )
    match = cache.match_prefix(torch.tensor(tokens, dtype=torch.int32)).cuda_handle

    assert result.cached_len == 5
    assert match.cached_len == 8
    assert match.get_matched_indices().tolist() == list(range(100, 105)) + list(range(205, 208))
    assert cache.size_info.evictable_size == 8


def test_partial_eviction_does_not_trim_protected_leaf() -> None:
    cache = make_radix_cache()
    protected_tokens = list(range(10, 18))
    protected_handle = insert(cache, protected_tokens, list(range(100, 108)))
    insert(cache, list(range(20, 28)), list(range(200, 208)))
    cache.lock_handle(protected_handle)

    evicted = cache.evict(3, partial=True)

    assert evicted.tolist() == [205, 206, 207]
    assert (
        cache.match_prefix(torch.tensor(protected_tokens, dtype=torch.int32)).cuda_handle.cached_len
        == 8
    )
    assert cache.size_info.protected_size == 8
    assert cache.size_info.evictable_size == 5


def test_recent_match_protection_precedes_lru_timestamp() -> None:
    cache = make_radix_cache()
    cold_handle = insert(cache, list(range(10, 14)), list(range(100, 104)))
    hot_handle = insert(cache, list(range(20, 24)), list(range(200, 204)))
    hot_handle.node.timestamp = cold_handle.node.timestamp - 1

    evicted = cache.evict(4, partial=True, protected_nodes={hot_handle.node})

    assert evicted.tolist() == list(range(100, 104))
    assert (
        cache.match_prefix(
            torch.tensor(list(range(20, 24)), dtype=torch.int32)
        ).cuda_handle.cached_len
        == 4
    )


def test_decayed_hotness_precedes_lru_timestamp() -> None:
    cache = make_radix_cache()
    cold_handle = insert(cache, list(range(30, 34)), list(range(300, 304)))
    hot_handle = insert(cache, list(range(40, 44)), list(range(400, 404)))
    hot_handle.node.timestamp = cold_handle.node.timestamp - 1

    evicted = cache.evict(
        4,
        partial=True,
        hotness_scores={hot_handle.node: 2.0},
    )

    assert evicted.tolist() == list(range(300, 304))


def test_cache_manager_partial_evict_keeps_page_accounting() -> None:
    page_size = 4
    num_pages = 4
    core.set_global_ctx(core.Context(page_size=page_size))
    manager = CacheManager(
        num_pages,
        page_size,
        torch.empty((1, num_pages * page_size), dtype=torch.int32),
        type="radix",
        radix_partial_eviction=True,
    )
    initial_pages = manager._allocate(num_pages)
    tokens = torch.arange(num_pages * page_size, dtype=torch.int32)
    manager.prefix_cache.insert_prefix(tokens, manager._page_to_token(initial_pages))

    reclaimed_page = manager._allocate(1)
    manager._free(manager._page_to_token(reclaimed_page))

    manager.check_integrity()
    assert manager.prefix_cache.match_prefix(tokens).cuda_handle.cached_len == 12
    assert manager.prefix_cache.size_info.evictable_size == 12


def test_partial_eviction_rejects_non_radix_cache() -> None:
    core.set_global_ctx(core.Context(page_size=1))

    with pytest.raises(ValueError, match="only supported by the radix cache"):
        CacheManager(
            4,
            1,
            torch.empty((1, 4), dtype=torch.int32),
            type="naive",
            radix_partial_eviction=True,
        )


def make_reserve_manager(
    *,
    num_pages: int,
    page_size: int = 1,
    reserve_pages: int,
    telemetry_enabled: bool = False,
) -> CacheManager:
    core.set_global_ctx(core.Context(page_size=page_size))
    return CacheManager(
        num_pages,
        page_size,
        torch.empty((1, num_pages * page_size), dtype=torch.int32),
        type="radix",
        telemetry_enabled=telemetry_enabled,
        radix_partial_eviction=True,
        radix_partial_eviction_reserve_pages=reserve_pages,
    )


def seed_full_cache(manager: CacheManager) -> torch.Tensor:
    pages = manager._allocate(manager.num_pages)
    tokens = torch.arange(manager.num_pages * manager.page_size, dtype=torch.int32)
    manager.prefix_cache.insert_prefix(tokens, manager._page_to_token(pages))
    return tokens


def test_partial_eviction_reserve_leaves_free_pages_and_records_target() -> None:
    manager = make_reserve_manager(
        num_pages=8,
        reserve_pages=2,
        telemetry_enabled=True,
    )
    tokens = seed_full_cache(manager)

    manager._allocate(1)

    snapshot = manager.telemetry.snapshot()
    assert len(manager.free_slots) == 2
    assert manager.prefix_cache.match_prefix(tokens).cuda_handle.cached_len == 5
    assert snapshot["eviction_requested_tokens"] == 1
    assert snapshot["eviction_target_tokens"] == 3
    assert snapshot["evicted_tokens"] == 3
    assert snapshot["operation_samples"][-1]["target_tokens"] == 3


def test_partial_eviction_reserve_is_capped_by_evictable_pages() -> None:
    manager = make_reserve_manager(num_pages=4, reserve_pages=4)
    allocated = manager._allocate(manager.num_pages)
    tokens = torch.arange(2, dtype=torch.int32)
    manager.prefix_cache.insert_prefix(tokens, allocated[:2])

    reclaimed = manager._allocate(1)

    assert len(reclaimed) == 1
    assert len(manager.free_slots) == 1
    assert manager.prefix_cache.size_info.evictable_size == 0


def test_partial_eviction_reserve_respects_page_alignment() -> None:
    manager = make_reserve_manager(num_pages=4, page_size=4, reserve_pages=2)
    tokens = seed_full_cache(manager)

    allocated = manager._allocate(1)

    assert len(manager.free_slots) == 2
    assert torch.all(manager.free_slots % manager.page_size == 0)
    assert allocated.item() % manager.page_size == 0
    assert manager.prefix_cache.match_prefix(tokens).cuda_handle.cached_len == 4
    manager._free(manager._page_to_token(allocated))
    manager.check_integrity()


@pytest.mark.parametrize("reserve_pages", [-1, 1])
def test_partial_eviction_reserve_requires_valid_partial_mode(reserve_pages: int) -> None:
    core.set_global_ctx(core.Context(page_size=1))
    match = "non-negative" if reserve_pages < 0 else "requires partial-leaf eviction"

    with pytest.raises(ValueError, match=match):
        CacheManager(
            4,
            1,
            torch.empty((1, 4), dtype=torch.int32),
            type="radix",
            radix_partial_eviction=False,
            radix_partial_eviction_reserve_pages=reserve_pages,
        )


def test_adaptive_reserve_uses_remaining_batch_pages_and_cap() -> None:
    manager = make_reserve_manager(num_pages=16, page_size=4, reserve_pages=0)
    manager.radix_partial_eviction_adaptive_reserve_max_pages = 6
    reqs = [SimpleNamespace(remain_len=5), SimpleNamespace(remain_len=13)]

    assert manager._get_adaptive_reserve_pages(reqs) == 6

    manager.radix_partial_eviction_adaptive_reserve_max_pages = 8
    assert manager._get_adaptive_reserve_pages(reqs) == 6


def test_age_aware_adaptive_reserve_discounts_nearly_finished_requests() -> None:
    manager = make_reserve_manager(num_pages=32, page_size=4, reserve_pages=0)
    manager.radix_partial_eviction_adaptive_reserve_max_pages = 8
    manager.radix_partial_eviction_adaptive_reserve_mode = "age-aware"
    reqs = [
        SimpleNamespace(remain_len=16, output_len=16),
        SimpleNamespace(remain_len=4, output_len=16),
    ]

    assert manager._get_adaptive_reserve_pages(reqs) == 5


def test_disabled_adaptive_reserve_defers_to_fixed_policy() -> None:
    manager = make_reserve_manager(num_pages=8, reserve_pages=2)

    assert manager._get_adaptive_reserve_pages([SimpleNamespace(remain_len=7)]) is None


@pytest.mark.parametrize(
    (
        "partial",
        "fixed",
        "adaptive",
        "match",
        "mode",
        "protected_matches",
        "hotness_decay",
    ),
    [
        (True, 0, -1, "must be non-negative", "raw", 0, 0.0),
        (False, 0, 16, "requires partial-leaf eviction", "raw", 0, 0.0),
        (True, 16, 16, "mutually exclusive", "raw", 0, 0.0),
        (True, 0, 16, "mode must be 'raw' or 'age-aware'", "invalid", 0, 0.0),
        (False, 0, 0, "requires partial-leaf eviction", "raw", 1, 0.0),
        (True, 0, 0, "Hotness decay must be in", "raw", 0, 1.0),
        (False, 0, 0, "Hotness ranking requires", "raw", 0, 0.9),
    ],
)
def test_adaptive_reserve_rejects_invalid_configuration(
    partial: bool,
    fixed: int,
    adaptive: int,
    match: str,
    mode: str,
    protected_matches: int,
    hotness_decay: float,
) -> None:
    core.set_global_ctx(core.Context(page_size=1))

    with pytest.raises(ValueError, match=match):
        CacheManager(
            8,
            1,
            torch.empty((1, 8), dtype=torch.int32),
            type="radix",
            radix_partial_eviction=partial,
            radix_partial_eviction_reserve_pages=fixed,
            radix_partial_eviction_adaptive_reserve_max_pages=adaptive,
            radix_partial_eviction_adaptive_reserve_mode=mode,
            radix_partial_eviction_protect_recent_matches=protected_matches,
            radix_partial_eviction_hotness_decay=hotness_decay,
        )
