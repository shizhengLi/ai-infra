from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.kvcache import SizeInfo
from minisgl.scheduler.cache import CacheManager, CacheTelemetry
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def make_cache_manager(*, telemetry_enabled: bool) -> CacheManager:
    core.set_global_ctx(core.Context(page_size=1))
    return CacheManager(
        num_pages=32,
        page_size=1,
        page_table=torch.empty((1, 32), dtype=torch.int32),
        type="radix",
        telemetry_enabled=telemetry_enabled,
    )


def pending_req(uid: int, token_ids: list[int]) -> PendingReq:
    return PendingReq(
        uid=uid,
        input_ids=torch.tensor(token_ids, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
    )


def test_disabled_cache_telemetry_does_not_update() -> None:
    telemetry = CacheTelemetry()
    telemetry.record_match(
        uid=1,
        matchable_tokens=8,
        matched_tokens=4,
        size_info=SizeInfo(4, 0),
    )
    telemetry.record_insert(
        uid=1,
        input_tokens=8,
        existing_tokens=4,
        inserted_tokens=4,
        finished=True,
        size_info=SizeInfo(8, 0),
    )
    telemetry.record_eviction(
        requested_tokens=2,
        evicted_tokens=4,
        size_info=SizeInfo(4, 0),
    )

    snapshot = telemetry.snapshot()
    assert snapshot["operations"] == 0
    assert snapshot["match_requests"] == 0
    assert snapshot["operation_samples"] == []


def test_disabled_cache_manager_does_not_update_telemetry() -> None:
    manager = make_cache_manager(telemetry_enabled=False)

    result = manager.match_req(pending_req(1, [10, 11, 12]))

    assert result.cuda_handle.cached_len == 0
    assert manager.telemetry.snapshot()["operations"] == 0


def test_cache_telemetry_classifies_hits_and_tracks_tokens() -> None:
    telemetry = CacheTelemetry(enabled=True)
    telemetry.record_match(
        uid=1,
        matchable_tokens=10,
        matched_tokens=0,
        size_info=SizeInfo(0, 0),
    )
    telemetry.record_match(
        uid=2,
        matchable_tokens=10,
        matched_tokens=4,
        size_info=SizeInfo(4, 2),
    )
    telemetry.record_match(
        uid=3,
        matchable_tokens=10,
        matched_tokens=10,
        size_info=SizeInfo(8, 2),
    )
    telemetry.record_insert(
        uid=3,
        input_tokens=12,
        existing_tokens=10,
        inserted_tokens=2,
        finished=False,
        size_info=SizeInfo(8, 4),
    )
    telemetry.record_eviction(
        requested_tokens=3,
        evicted_tokens=5,
        size_info=SizeInfo(5, 4),
    )

    snapshot = telemetry.snapshot()
    assert snapshot["operations"] == 5
    assert snapshot["match_requests"] == 3
    assert snapshot["matchable_tokens"] == 30
    assert snapshot["matched_tokens"] == 14
    assert snapshot["token_hit_rate"] == pytest.approx(14 / 30)
    assert snapshot["miss_requests"] == 1
    assert snapshot["partial_hit_requests"] == 1
    assert snapshot["full_hit_requests"] == 1
    assert snapshot["inserted_tokens"] == 2
    assert snapshot["eviction_requested_tokens"] == 3
    assert snapshot["evicted_tokens"] == 5
    assert snapshot["max_resident_tokens"] == 12
    assert len(snapshot["operation_samples"]) == 5

    telemetry.drain_samples()
    assert telemetry.snapshot()["operation_samples"] == []
    assert telemetry.snapshot()["operations"] == 5


def test_cache_manager_records_miss_partial_and_full_match() -> None:
    manager = make_cache_manager(telemetry_enabled=True)
    manager.prefix_cache.insert_prefix(
        torch.tensor([10, 11, 12], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    full = manager.match_req(pending_req(1, [10, 11, 12, 13]))
    partial = manager.match_req(pending_req(2, [10, 11, 99, 13]))
    miss = manager.match_req(pending_req(3, [99, 11, 12, 13]))

    assert full.cuda_handle.cached_len == 3
    assert partial.cuda_handle.cached_len == 2
    assert miss.cuda_handle.cached_len == 0
    snapshot = manager.telemetry.snapshot()
    assert snapshot["matched_tokens"] == 5
    assert snapshot["matchable_tokens"] == 9
    assert snapshot["full_hit_requests"] == 1
    assert snapshot["partial_hit_requests"] == 1
    assert snapshot["miss_requests"] == 1


def test_cache_manager_records_actual_eviction_size() -> None:
    manager = make_cache_manager(telemetry_enabled=True)
    allocated = manager._allocate(manager.num_pages)
    manager.prefix_cache.insert_prefix(
        torch.tensor([20, 21], dtype=torch.int32),
        allocated[:2],
    )

    reclaimed = manager._allocate(1)

    assert len(reclaimed) == 1
    snapshot = manager.telemetry.snapshot()
    assert snapshot["eviction_calls"] == 1
    assert snapshot["eviction_requested_tokens"] == 1
    assert snapshot["evicted_tokens"] == 2
