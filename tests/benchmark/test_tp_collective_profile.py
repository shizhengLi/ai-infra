import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "benchmark/online/bench_tp_collective_profile.py"
SPEC = importlib.util.spec_from_file_location("bench_tp_collective_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dtype_parser() -> None:
    import torch

    assert MODULE._dtype("float16") is torch.float16
    assert MODULE._dtype("bfloat16") is torch.bfloat16


def test_collective_nvtx_cpu_path_is_a_noop() -> None:
    import torch

    from minisgl.distributed.impl import _collective_nvtx

    tensor = torch.zeros(4)
    with _collective_nvtx("all_reduce", tensor):
        tensor.add_(1)
    assert tensor.tolist() == [1, 1, 1, 1]
