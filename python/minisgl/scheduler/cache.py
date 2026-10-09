from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, SizeInfo, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


@dataclass
class CacheTelemetry:
    enabled: bool = False
    operations: int = 0
    match_requests: int = 0
    matchable_tokens: int = 0
    matched_tokens: int = 0
    miss_requests: int = 0
    partial_hit_requests: int = 0
    full_hit_requests: int = 0
    insert_calls: int = 0
    insert_input_tokens: int = 0
    insert_existing_tokens: int = 0
    inserted_tokens: int = 0
    eviction_calls: int = 0
    eviction_requested_tokens: int = 0
    eviction_target_tokens: int = 0
    evicted_tokens: int = 0
    current_evictable_tokens: int = 0
    current_protected_tokens: int = 0
    max_resident_tokens: int = 0
    operation_samples: list[dict[str, int | bool | str]] = field(default_factory=list)

    def _update_size(self, size_info: SizeInfo) -> None:
        self.current_evictable_tokens = size_info.evictable_size
        self.current_protected_tokens = size_info.protected_size
        self.max_resident_tokens = max(self.max_resident_tokens, size_info.total_size)

    def record_match(
        self,
        *,
        uid: int,
        matchable_tokens: int,
        matched_tokens: int,
        size_info: SizeInfo,
    ) -> None:
        if not self.enabled:
            return
        assert 0 <= matched_tokens <= matchable_tokens
        self.operations += 1
        self.match_requests += 1
        self.matchable_tokens += matchable_tokens
        self.matched_tokens += matched_tokens
        if matched_tokens == 0:
            self.miss_requests += 1
            classification = "miss"
        elif matched_tokens == matchable_tokens:
            self.full_hit_requests += 1
            classification = "full"
        else:
            self.partial_hit_requests += 1
            classification = "partial"
        self._update_size(size_info)
        self.operation_samples.append(
            {
                "event": "match",
                "uid": uid,
                "matchable_tokens": matchable_tokens,
                "matched_tokens": matched_tokens,
                "classification": classification,
                "resident_tokens": size_info.total_size,
            }
        )

    def record_insert(
        self,
        *,
        uid: int,
        input_tokens: int,
        existing_tokens: int,
        inserted_tokens: int,
        finished: bool,
        size_info: SizeInfo,
    ) -> None:
        if not self.enabled:
            return
        assert 0 <= existing_tokens <= input_tokens
        assert inserted_tokens >= 0
        self.operations += 1
        self.insert_calls += 1
        self.insert_input_tokens += input_tokens
        self.insert_existing_tokens += existing_tokens
        self.inserted_tokens += inserted_tokens
        self._update_size(size_info)
        self.operation_samples.append(
            {
                "event": "insert",
                "uid": uid,
                "input_tokens": input_tokens,
                "existing_tokens": existing_tokens,
                "inserted_tokens": inserted_tokens,
                "finished": finished,
                "resident_tokens": size_info.total_size,
            }
        )

    def record_eviction(
        self,
        *,
        requested_tokens: int,
        target_tokens: int | None = None,
        evicted_tokens: int,
        size_info: SizeInfo,
    ) -> None:
        if not self.enabled:
            return
        target_tokens = target_tokens if target_tokens is not None else requested_tokens
        assert requested_tokens > 0
        assert evicted_tokens >= target_tokens >= requested_tokens
        self.operations += 1
        self.eviction_calls += 1
        self.eviction_requested_tokens += requested_tokens
        self.eviction_target_tokens += target_tokens
        self.evicted_tokens += evicted_tokens
        self._update_size(size_info)
        self.operation_samples.append(
            {
                "event": "evict",
                "requested_tokens": requested_tokens,
                "target_tokens": target_tokens,
                "evicted_tokens": evicted_tokens,
                "resident_tokens": size_info.total_size,
            }
        )

    def snapshot(self) -> dict[str, object]:
        token_hit_rate = (
            self.matched_tokens / self.matchable_tokens if self.matchable_tokens > 0 else 0.0
        )
        return {
            "operations": self.operations,
            "match_requests": self.match_requests,
            "matchable_tokens": self.matchable_tokens,
            "matched_tokens": self.matched_tokens,
            "token_hit_rate": token_hit_rate,
            "miss_requests": self.miss_requests,
            "partial_hit_requests": self.partial_hit_requests,
            "full_hit_requests": self.full_hit_requests,
            "insert_calls": self.insert_calls,
            "insert_input_tokens": self.insert_input_tokens,
            "insert_existing_tokens": self.insert_existing_tokens,
            "inserted_tokens": self.inserted_tokens,
            "eviction_calls": self.eviction_calls,
            "eviction_requested_tokens": self.eviction_requested_tokens,
            "eviction_target_tokens": self.eviction_target_tokens,
            "evicted_tokens": self.evicted_tokens,
            "current_evictable_tokens": self.current_evictable_tokens,
            "current_protected_tokens": self.current_protected_tokens,
            "max_resident_tokens": self.max_resident_tokens,
            "operation_samples": list(self.operation_samples),
        }

    def drain_samples(self) -> None:
        self.operation_samples.clear()


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        *,
        telemetry_enabled: bool = False,
        radix_partial_eviction: bool = False,
        radix_partial_eviction_reserve_pages: int = 0,
        radix_partial_eviction_adaptive_reserve_max_pages: int = 0,
        radix_partial_eviction_adaptive_reserve_mode: str = "raw",
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.telemetry = CacheTelemetry(enabled=telemetry_enabled)
        if radix_partial_eviction and type != "radix":
            raise ValueError("Partial-leaf eviction is only supported by the radix cache")
        if radix_partial_eviction_reserve_pages < 0:
            raise ValueError("Partial-eviction reserve pages must be non-negative")
        if radix_partial_eviction_reserve_pages > 0 and not radix_partial_eviction:
            raise ValueError("Partial-eviction reserve requires partial-leaf eviction")
        if radix_partial_eviction_adaptive_reserve_max_pages < 0:
            raise ValueError("Adaptive partial-eviction reserve maximum must be non-negative")
        if radix_partial_eviction_adaptive_reserve_max_pages > 0 and not radix_partial_eviction:
            raise ValueError("Adaptive reserve requires partial-leaf eviction")
        if radix_partial_eviction_adaptive_reserve_mode not in {"raw", "age-aware"}:
            raise ValueError("Adaptive reserve mode must be 'raw' or 'age-aware'")
        if (
            radix_partial_eviction_reserve_pages > 0
            and radix_partial_eviction_adaptive_reserve_max_pages > 0
        ):
            raise ValueError("Fixed and adaptive partial-eviction reserves are mutually exclusive")
        self.radix_partial_eviction = radix_partial_eviction
        self.radix_partial_eviction_reserve_pages = radix_partial_eviction_reserve_pages
        self.radix_partial_eviction_adaptive_reserve_max_pages = (
            radix_partial_eviction_adaptive_reserve_max_pages
        )
        self.radix_partial_eviction_adaptive_reserve_mode = (
            radix_partial_eviction_adaptive_reserve_mode
        )

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        matchable_ids = req.input_ids[: input_len - 1]
        result = self.prefix_cache.match_prefix(matchable_ids)
        if self.telemetry.enabled:
            self.telemetry.record_match(
                uid=req.uid,
                matchable_tokens=len(matchable_ids),
                matched_tokens=result.cuda_handle.cached_len,
                size_info=self.prefix_cache.size_info,
            )
        return result

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            reserve_pages = self._get_adaptive_reserve_pages(reqs)
            allocated = self._page_to_token(
                self._allocate(needed_pages, reserve_pages=reserve_pages)
            )
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def _get_adaptive_reserve_pages(self, reqs: List[Req]) -> int | None:
        max_pages = self.radix_partial_eviction_adaptive_reserve_max_pages
        if max_pages == 0:
            return None
        if self.radix_partial_eviction_adaptive_reserve_mode == "raw":
            remaining_pages = sum(div_ceil(req.remain_len, self.page_size) for req in reqs)
        else:
            remaining_pages = 0
            for req in reqs:
                if req.remain_len <= 0:
                    continue
                # Generated output is the age proxy: a nearly finished request should
                # contribute less future demand than a newly admitted request.
                remaining_fraction = min(1.0, req.remain_len / max(req.output_len, 1))
                remaining_pages += max(
                    1,
                    int(div_ceil(req.remain_len, self.page_size) * remaining_fraction),
                )
        return min(remaining_pages, max_pages)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)
        if self.telemetry.enabled:
            self.telemetry.record_insert(
                uid=req.uid,
                input_tokens=len(insert_ids),
                existing_tokens=cached_len,
                inserted_tokens=new_handle.cached_len - cached_len,
                finished=finished,
                size_info=self.prefix_cache.size_info,
            )

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int, *, reserve_pages: int | None = None) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            requested_tokens = (needed_pages - free_pages) * self.page_size
            reserve_pages = (
                self.radix_partial_eviction_reserve_pages
                if reserve_pages is None
                else reserve_pages
            )
            reserve_tokens = reserve_pages * self.page_size
            evictable_tokens = self.prefix_cache.size_info.evictable_size
            assert requested_tokens <= evictable_tokens, (
                f"Cannot satisfy {requested_tokens} cache tokens with only "
                f"{evictable_tokens} evictable"
            )
            target_tokens = min(
                requested_tokens + reserve_tokens,
                evictable_tokens,
            )
            evicted = self.prefix_cache.evict(
                target_tokens,
                partial=self.radix_partial_eviction,
            )
            if self.telemetry.enabled:
                self.telemetry.record_eviction(
                    requested_tokens=requested_tokens,
                    target_tokens=target_tokens,
                    evicted_tokens=len(evicted),
                    size_info=self.prefix_cache.size_info,
                )
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
