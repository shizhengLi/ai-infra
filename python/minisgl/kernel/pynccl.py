from __future__ import annotations

import functools
import os
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@functools.cache
def _find_nccl_library() -> Path | None:
    search_dirs: list[Path] = []

    for env_name in ("NCCL_HOME", "NCCL_ROOT"):
        if root := os.environ.get(env_name):
            search_dirs.extend([Path(root) / "lib", Path(root) / "lib64"])

    if spec := find_spec("nvidia.nccl"):
        for root in spec.submodule_search_locations or []:
            search_dirs.append(Path(root) / "lib")

    for root in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if root:
            search_dirs.append(Path(root))

    for directory in search_dirs:
        for name in ("libnccl.so", "libnccl.so.2"):
            library = directory / name
            if library.is_file():
                return library.resolve()
    return None


def _nccl_linker_flags() -> list[str]:
    if library := _find_nccl_library():
        return [str(library), f"-Wl,-rpath,{library.parent}"]
    return ["-lnccl"]


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_linker_flags())


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
