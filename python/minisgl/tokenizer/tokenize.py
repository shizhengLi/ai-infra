from __future__ import annotations

from typing import List, Sequence

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        if not msgs:
            return []

        # Chat templates and plain prompts have different tokenizer entry points. Batch
        # each class independently, then write results back to the original message order.
        grouped: dict[str, list[tuple[int, str | list[dict[str, str]]]]] = {
            "plain": [],
            "chat": [],
        }
        for index, msg in enumerate(msgs):
            kind = "chat" if isinstance(msg.text, list) else "plain"
            grouped[kind].append((index, msg.text))

        results: list[torch.Tensor | None] = [None] * len(msgs)
        if grouped["plain"]:
            indices, prompts = zip(*grouped["plain"], strict=True)
            token_ids = self._batch_encode_texts(prompts)
            self._store_results(results, indices, token_ids)
        if grouped["chat"]:
            indices, conversations = zip(*grouped["chat"], strict=True)
            token_ids = self._batch_encode_chats(conversations)
            self._store_results(results, indices, token_ids)

        assert all(result is not None for result in results)
        return [result for result in results if result is not None]

    def _batch_encode_texts(self, prompts: Sequence[str]) -> Sequence[Sequence[int]]:
        encoded = self.tokenizer(
            list(prompts),
            add_special_tokens=True,
            padding=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        return encoded["input_ids"]

    def _batch_encode_chats(
        self, conversations: Sequence[list[dict[str, str]]]
    ) -> Sequence[Sequence[int]]:
        encoded = self.tokenizer.apply_chat_template(
            list(conversations),
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            return_tensors=None,
        )
        return encoded

    @staticmethod
    def _store_results(
        results: list[torch.Tensor | None],
        indices: Sequence[int],
        token_ids: Sequence[Sequence[int]],
    ) -> None:
        assert len(indices) == len(token_ids)
        for index, ids in zip(indices, token_ids, strict=True):
            results[index] = torch.tensor(ids, dtype=torch.int32)
