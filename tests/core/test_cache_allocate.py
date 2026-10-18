"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
"""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

import minisgl.core as core
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.scheduler import Scheduler


@pytest.fixture(autouse=True)
def reset_global_ctx():
    """Reset global context before and after each test."""
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    """Helper to create a CacheManager with radix cache on CPU."""
    page_table = torch.empty((1,))
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _insert_evictable(cm: CacheManager, input_ids: torch.Tensor, indices: torch.Tensor):
    """Insert a prefix into the radix cache so it becomes evictable."""
    cm.prefix_cache.insert_prefix(input_ids, indices)


def _assert_all_page_aligned(tensor: torch.Tensor, page_size: int, label: str = ""):
    """Assert every element in tensor is a multiple of page_size."""
    if len(tensor) == 0:
        return
    misaligned = tensor[tensor % page_size != 0]
    assert (
        len(misaligned) == 0
    ), f"{label} contains non-page-aligned values: {misaligned.tolist()}, page_size={page_size}"


def _assert_no_overlap(pages: torch.Tensor, page_size: int):
    """Assert that page-aligned starts, when expanded, produce no overlapping ranges."""
    if len(pages) <= 1:
        return
    expanded = set()
    for p in pages.tolist():
        token_range = set(range(p, p + page_size))
        overlap = expanded & token_range
        assert len(overlap) == 0, f"Overlapping tokens: {overlap}"
        expanded.update(token_range)


class TestAllocateEvictPageAlignment:
    """Tests for _allocate handling eviction with page_size > 1."""

    def test_can_allocate_paged_uses_visible_capacity(self):
        cm = _make_cache_manager(num_pages=2, page_size=1)
        req = SimpleNamespace(cached_len=0, device_len=1)
        assert cm.can_allocate_paged([req, req]) is True
        cm._allocate(1)
        assert cm.can_allocate_paged([req, req]) is False
        assert cm.can_allocate_paged([req]) is True

    def test_scheduler_trims_decode_batch_to_visible_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(
            can_allocate_paged=lambda reqs: len(reqs) <= 2
        )
        reqs = [SimpleNamespace(uid=index) for index in range(3)]
        batch = core.Batch(reqs=reqs, phase="decode")

        trimmed = scheduler._fit_decode_batch(batch)

        assert trimmed is not None
        assert [req.uid for req in trimmed.reqs] == [0, 1]

    def test_scheduler_round_robin_avoids_decode_starvation(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(
            can_allocate_paged=lambda reqs: len(reqs) <= 1
        )
        reqs = [SimpleNamespace(uid=index) for index in range(3)]
        selected_uids = []
        for _ in range(9):
            batch = scheduler._fit_decode_batch(core.Batch(reqs=reqs, phase="decode"))
            assert batch is not None
            selected_uids.append(batch.reqs[0].uid)

        assert selected_uids == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_scheduler_round_robin_handles_mixed_page_demand(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(
            can_allocate_paged=lambda reqs: sum(req.demand for req in reqs) <= 3
        )
        reqs = [
            SimpleNamespace(uid=0, demand=2),
            SimpleNamespace(uid=1, demand=1),
            SimpleNamespace(uid=2, demand=1),
        ]
        selected = []
        for _ in range(3):
            batch = scheduler._fit_decode_batch(core.Batch(reqs=reqs, phase="decode"))
            assert batch is not None
            selected.append([req.uid for req in batch.reqs])

        assert selected == [[0, 1], [2, 0], [1, 2]]

    def test_scheduler_cursor_survives_request_removal(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(
            can_allocate_paged=lambda reqs: len(reqs) <= 2
        )
        initial = [SimpleNamespace(uid=index) for index in range(4)]
        first = scheduler._fit_decode_batch(core.Batch(reqs=initial, phase="decode"))
        assert first is not None
        assert [req.uid for req in first.reqs] == [0, 1]

        remaining = [initial[1], initial[2], initial[3]]
        second = scheduler._fit_decode_batch(core.Batch(reqs=remaining, phase="decode"))

        assert second is not None
        assert [req.uid for req in second.reqs] == [2, 3]

    def test_scheduler_tracks_capacity_wait_rounds(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(
            can_allocate_paged=lambda reqs: len(reqs) <= 1
        )
        reqs = [SimpleNamespace(uid=index) for index in range(3)]

        first = scheduler._fit_decode_batch(core.Batch(reqs=reqs, phase="decode"))
        assert first is not None
        assert scheduler._decode_capacity_wait_rounds == {1: 1, 2: 1}
        assert scheduler._last_decode_capacity_wait_rounds == {0: 1}

        second = scheduler._fit_decode_batch(core.Batch(reqs=reqs, phase="decode"))
        assert second is not None
        assert [req.uid for req in second.reqs] == [1]
        assert scheduler._decode_capacity_wait_rounds == {2: 2, 0: 1}
        assert scheduler._last_decode_capacity_wait_rounds == {1: 2}

    def test_scheduler_capacity_trace_records_dynamic_fairness(self):
        """Exercise trim/deferred telemetry across completion, cancel, and new UID events."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.prefill_telemetry_enabled = True
        scheduler._is_primary_rank = True
        scheduler._pipeline_events = []
        scheduler._pipeline_batch_ids = {}
        scheduler.engine = SimpleNamespace(
            graph_runner=SimpleNamespace(can_use_cuda_graph=lambda batch: False)
        )

        requests = {
            uid: SimpleNamespace(uid=uid, demand=demand)
            for uid, demand in {
                0: 2,
                1: 1,
                2: 1,
                3: 2,
                4: 1,
                5: 1,
            }.items()
        }

        def run_round(active_uids: list[int], capacity: int) -> list[int]:
            active = [requests[uid] for uid in active_uids]
            scheduler.cache_manager = SimpleNamespace(
                can_allocate_paged=lambda reqs: sum(req.demand for req in reqs) <= capacity
            )
            requested_uids = [req.uid for req in active]
            batch = scheduler._fit_decode_batch(core.Batch(reqs=active, phase="decode"))
            if batch is None:
                scheduler._record_pipeline_event(
                    "decode_capacity_deferred",
                    requested_batch_size=len(active),
                    deferred_uids=requested_uids,
                    wait_rounds=scheduler._decode_capacity_wait_rounds,
                )
                return []

            selected_uids = [req.uid for req in batch.reqs]
            if len(selected_uids) < len(requested_uids):
                selected = set(selected_uids)
                scheduler._record_pipeline_event(
                    "decode_capacity_trimmed",
                    batch,
                    requested_batch_size=len(active),
                    selected_batch_size=len(selected_uids),
                    deferred_uids=[uid for uid in requested_uids if uid not in selected],
                    selected_wait_rounds=scheduler._last_decode_capacity_wait_rounds,
                )
            return selected_uids

        # Nine fixed rounds establish the 2/1/1 mixed-demand fairness cycle.
        fixed_selections = [
            run_round([0, 1, 2], capacity=3)
            for _ in range(9)
        ]
        assert fixed_selections == [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 0],
            [1, 2],
        ]

        # UID 0 completes and UID 3 joins; then UID 1 is cancelled and UID 4 joins.
        assert run_round([1, 2, 3], capacity=3) == [1, 2]
        assert run_round([2, 3, 4], capacity=3) == [3, 4]

        # No visible capacity must defer the candidate without corrupting the cursor.
        assert run_round([3], capacity=0) == []
        assert run_round([3, 4, 5], capacity=3) == [3, 4]

        events = scheduler._pipeline_events
        assert len(events) == 13
        trim_events = [event for event in events if event["event"] == "decode_capacity_trimmed"]
        deferred_events = [event for event in events if event["event"] == "decode_capacity_deferred"]
        assert len(trim_events) == 12
        assert len(deferred_events) == 1
        assert deferred_events[0]["deferred_uids"] == [3]
        assert deferred_events[0]["wait_rounds"] == {3: 1}
        assert trim_events[-1]["deferred_uids"] == [5]
        assert trim_events[-1]["selected_wait_rounds"] == {3: 2, 4: 1}

    def test_scheduler_trims_against_real_cache_capacity(self):
        cm = _make_cache_manager(num_pages=4, page_size=1)
        cm._allocate(3)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = cm
        reqs = [SimpleNamespace(uid=index, cached_len=0, device_len=1) for index in range(3)]

        trimmed = scheduler._fit_decode_batch(core.Batch(reqs=reqs, phase="decode"))

        assert trimmed is not None
        assert [req.uid for req in trimmed.reqs] == [0]
        assert cm.available_size == 1

    def test_pipeline_event_preserves_requested_decode_size(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.prefill_telemetry_enabled = True
        scheduler._is_primary_rank = True
        scheduler._pipeline_events = []
        scheduler._pipeline_batch_ids = {}
        scheduler.engine = SimpleNamespace(
            graph_runner=SimpleNamespace(can_use_cuda_graph=lambda batch: False)
        )
        batch = core.Batch(
            reqs=[SimpleNamespace(uid=0), SimpleNamespace(uid=1)], phase="decode"
        )

        scheduler._record_pipeline_event(
            "decode_capacity_trimmed",
            batch,
            requested_batch_size=4,
            selected_batch_size=2,
        )

        event = scheduler._pipeline_events[0]
        assert event["requested_batch_size"] == 4
        assert event["selected_batch_size"] == 2
        assert event["batch_size"] == 2

    def test_allocate_after_evict_returns_page_aligned(self):
        """Allocated pages after eviction must be page-aligned."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)
        assert len(cm.free_slots) == 0

        # Insert 2 pages worth of data into the cache (evictable)
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        # Simulate page table entries: page 0 = [0,1,2,3], page 1 = [4,5,6,7]
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — triggers eviction
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")

    def test_consecutive_allocations_after_evict_no_overlap(self):
        """Multiple allocations after eviction must not produce overlapping pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 2 pages into cache
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages one by one
        page_a = cm._allocate(1)
        page_b = cm._allocate(1)
        all_pages = torch.cat([page_a, page_b])

        _assert_all_page_aligned(all_pages, page_size, "all_pages")
        _assert_no_overlap(all_pages, page_size)

    def test_free_slots_stay_page_aligned_after_evict(self):
        """_free_slots must remain page-aligned after eviction refills them."""
        page_size = 8
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 4 pages worth of data (4 * 8 = 32 tokens)
        n_tokens = page_size * 4
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        # Indices: page starts at 0, 8, 16, 24
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — evicts and refills _free_slots
        cm._allocate(1)

        # All remaining free slots must be page-aligned
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots")

    def test_allocate_exact_pages_needed_from_evict(self):
        """When exactly N pages are needed, eviction must provide at least N pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all
        cm._allocate(num_pages)

        # Insert 3 pages into cache
        n_tokens = page_size * 3
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages at once — needs eviction of at least 2 pages
        allocated = cm._allocate(2)
        assert len(allocated) == 2
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_no_overlap(allocated, page_size)

    def test_page_to_token_expansion_correct_after_evict(self):
        """_page_to_token on eviction-allocated pages must produce correct consecutive ranges."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        cm._allocate(num_pages)

        # Insert 2 pages: tokens [0..7] with indices [0..7]
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages via eviction
        pages = cm._allocate(2)
        tokens = cm._page_to_token(pages)

        # Each page should expand to page_size consecutive tokens
        assert len(tokens) == 2 * page_size
        for i, page_start in enumerate(pages.tolist()):
            chunk = tokens[i * page_size : (i + 1) * page_size].tolist()
            expected = list(range(page_start, page_start + page_size))
            assert chunk == expected, f"Page {page_start}: got {chunk}, expected {expected}"

    def test_check_integrity_passes_after_evict_cycle(self):
        """CacheManager integrity check should pass after allocate-free-evict cycles."""
        page_size = 4
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Allocate 2 pages (simulating a request using them)
        pages_for_cache = cm._allocate(2)
        # free_slots: 6 pages remaining

        # Insert those 2 pages into radix cache (simulating a finished request)
        # The token indices come from _page_to_token expansion of the allocated pages
        token_indices = cm._page_to_token(pages_for_cache)
        n_tokens = len(token_indices)
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, token_indices)
        # Now: free=6, cache=2, total=8

        # This should not raise
        cm.check_integrity()

        # Now exhaust free slots and trigger eviction
        cm._allocate(6)
        assert len(cm.free_slots) == 0

        # Allocate 1 more page — must evict from cache
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated after evict")
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")


if __name__ == "__main__":
    pytest.main([__file__])
