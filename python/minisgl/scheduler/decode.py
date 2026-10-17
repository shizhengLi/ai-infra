from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.utils import div_ceil

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    paused_reqs: list[Req] = field(default_factory=list)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        try:
            self.paused_reqs.remove(req)
        except ValueError:
            pass

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        for index, req in enumerate(self.paused_reqs):
            if req.uid == uid:
                return self.paused_reqs.pop(index)
        return None

    def retract_candidate(self, protected_reqs: Set[Req] | None = None) -> Req | None:
        """Remove the largest future KV consumer that is not in an in-flight batch.

        The scheduler passes requests belonging to the currently submitted GPU
        batch as ``protected_reqs``.  This makes retraction a CPU-side operation
        with no possibility of invalidating a pending CUDA result.
        """
        protected_reqs = protected_reqs or set()
        candidates = [
            req
            for req in self.running_reqs
            if req not in protected_reqs and req.can_decode
        ]
        if not candidates:
            return None
        req = max(
            candidates,
            key=lambda item: (div_ceil(item.remain_len, self.page_size), item.remain_len, item.uid),
        )
        self.running_reqs.remove(req)
        return req

    def pause_candidate(self, protected_reqs: Set[Req] | None = None) -> Req | None:
        req = self.retract_candidate(protected_reqs)
        if req is not None:
            req.mark_retracted()
            self.paused_reqs.append(req)
        return req

    def resume_paused(self) -> None:
        for req in self.paused_reqs:
            if req.can_decode:
                req.mark_resumed()
                self.running_reqs.add(req)
        self.paused_reqs.clear()

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        # Keep one page per paused request available for the next decode append;
        # the remaining future demand is intentionally released to admission.
        paused_tokens = self.page_size * len(self.paused_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved + paused_tokens

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
