from __future__ import annotations

import torch

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.server.args import parse_args
from minisgl.tokenizer.tokenize import TokenizeManager


class RecordingTokenizer:
    def __init__(self) -> None:
        self.plain_calls: list[list[str]] = []
        self.chat_calls: list[list[list[dict[str, str]]]] = []

    def __call__(self, prompts, **kwargs):
        self.plain_calls.append(prompts)
        assert kwargs == {
            "add_special_tokens": True,
            "padding": False,
            "return_attention_mask": False,
            "return_tensors": None,
        }
        return {"input_ids": [[len(prompt), index] for index, prompt in enumerate(prompts)]}

    def apply_chat_template(self, conversations, **kwargs):
        self.chat_calls.append(conversations)
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "padding": False,
            "return_tensors": None,
        }
        return [[len(conversation), 100 + index] for index, conversation in enumerate(conversations)]


def _msg(uid: int, text) -> TokenizeMsg:
    return TokenizeMsg(uid=uid, text=text, sampling_params=SamplingParams())


def test_tokenize_batches_plain_and_chat_while_restoring_order() -> None:
    tokenizer = RecordingTokenizer()
    manager = TokenizeManager(tokenizer)  # type: ignore[arg-type]
    messages = [
        _msg(0, "first"),
        _msg(1, [{"role": "user", "content": "chat"}]),
        _msg(2, "second"),
        _msg(3, [{"role": "user", "content": "chat 2"}]),
    ]

    result = manager.tokenize(messages)

    assert [tensor.tolist() for tensor in result] == [
        [5, 0],
        [1, 100],
        [6, 1],
        [1, 101],
    ]
    assert tokenizer.plain_calls == [["first", "second"]]
    assert len(tokenizer.chat_calls) == 1
    assert [conversation[0]["content"] for conversation in tokenizer.chat_calls[0]] == [
        "chat",
        "chat 2",
    ]
    assert all(tensor.dtype == torch.int32 for tensor in result)


def test_tokenize_empty_batch() -> None:
    tokenizer = RecordingTokenizer()
    assert TokenizeManager(tokenizer).tokenize([]) == []  # type: ignore[arg-type]


def test_tokenizer_batch_size_is_configurable_and_positive() -> None:
    base_args = ["--model", "unused", "--dtype", "bfloat16"]
    default_args, _ = parse_args(base_args)
    assert default_args.tokenizer_batch_size == 32

    args, _ = parse_args([*base_args, "--tokenizer-batch-size", "16"])
    assert args.tokenizer_batch_size == 16

    try:
        parse_args([*base_args, "--tokenizer-batch-size", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("zero tokenizer batch size must be rejected")
