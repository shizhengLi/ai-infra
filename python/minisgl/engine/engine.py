from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


@dataclass
class ForwardTiming:
    model_start: torch.cuda.Event
    model_end: torch.cuda.Event
    sample_end: torch.cuda.Event
    copy_end: torch.cuda.Event
    execution_kind: str


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    timing: ForwardTiming | None = None


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        self.config = config
        self.kv_cache_calibration_summary = config.kv_cache_calibration_summary
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        if config.dtype == torch.bfloat16:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                config.allow_bf16_reduced_precision_reduction
            )

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.kv_cache_dtype = _resolve_kv_cache_dtype(config)
        self.ctx = Context(config.page_size, dtype=config.dtype)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.kv_cache_dtype,
            scale=config.kv_cache_scale,
            calibration=config.kv_cache_calibration,
            calibration_summary=config.kv_cache_calibration_summary,
            fused_store=config.kv_cache_fused_store,
            native_pair=config.kv_cache_native_pair,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            cuda_graph_shape_policy=config.cuda_graph_shape_policy,
            cuda_graph_disable_bs=config.cuda_graph_disable_bs,
            max_running_req=config.max_running_req,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            result = {}
            for key, value in self.model.state_dict().items():
                if key.endswith(".weight_scale"):
                    continue
                dtype = self.dtype if value.dtype == torch.float8_e4m3fn else value.dtype
                result[key] = torch.randn(value.shape, dtype=dtype, device=self.device)
            return result
        else:
            result = {}
            for key, value in load_weight(config.model_path, self.device):
                quantization = config.quantization_config
                keep_checkpoint_dtype = quantization is not None and (
                    quantization.method == "fp8"
                    and (value.dtype == torch.float8_e4m3fn or key.endswith(".weight_scale"))
                    or quantization.method == "awq"
                    and key.endswith((".qweight", ".qzeros"))
                )
                result[key] = value if keep_checkpoint_dtype else value.to(self.dtype)
            return result

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.kv_cache_dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(
            f"Allocating {num_tokens} tokens for KV cache ({self.kv_cache_dtype}), "
            f"K + V = {mem_GB(real_kv_size)}"
        )
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs, *, profile_timing: bool = False
    ) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        timing: ForwardTiming | None = None
        model_start = model_end = sample_end = None
        if profile_timing:
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            sample_end = torch.cuda.Event(enable_timing=True)
            model_start.record(self.stream)
        execution_kind = "graph" if self.graph_runner.can_use_cuda_graph(batch) else "eager"
        with self.ctx.forward_batch(batch):
            if execution_kind == "graph":
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()
        if model_end is not None:
            model_end.record(self.stream)

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        if sample_end is not None:
            sample_end.record(self.stream)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event(enable_timing=profile_timing)
        copy_done_event.record(self.stream)
        if model_start is not None and model_end is not None and sample_end is not None:
            timing = ForwardTiming(
                model_start=model_start,
                model_end=model_end,
                sample_end=sample_end,
                copy_end=copy_done_event,
                execution_kind=execution_kind,
            )
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event, timing)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        if self.kv_cache_calibration_summary is not None:
            self.kv_cache.write_calibration_summary(
                self.kv_cache_calibration_summary, self.config.tp_info.rank
            )
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if config.kv_cache_scale <= 0.0:
        raise ValueError(f"--kv-cache-scale must be positive, got {config.kv_cache_scale}")
    if config.kv_cache_native_pair and not config.kv_cache_fused_store:
        raise ValueError("--kv-cache-native-pair requires --kv-cache-fused-store")
    if config.page_size <= 0:
        raise ValueError(f"--page-size must be positive, got {config.page_size}")
    if config.cuda_graph_shape_policy not in {"coarse", "l20", "explicit"}:
        raise ValueError(
            f"Unsupported --cuda-graph-shape-policy: {config.cuda_graph_shape_policy}"
        )
    if config.cuda_graph_shape_policy == "explicit" and config.cuda_graph_bs is None:
        raise ValueError("--cuda-graph-shape-policy explicit requires --cuda-graph-bs")
    if config.kv_cache_calibration not in {"fixed", "first_batch"}:
        raise ValueError(f"Unsupported KV cache calibration: {config.kv_cache_calibration}")
    if config.kv_cache_dtype == "float8_e4m3fn":
        if config.attention_backend != "fi":
            raise ValueError(
                "FP8 KV cache currently requires the FlashInfer attention backend (--attn fi)"
            )
    elif config.kv_cache_calibration != "fixed":
        raise ValueError("KV cache calibration is only supported for FP8 KV cache")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.quantization_config is not None:
        if config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Quantized linear kernels require float16 or bfloat16 model dtype")
        if config.model_config.is_moe:
            raise ValueError("Linear quantization currently supports dense models only")
        if config.quantization_config.method == "awq" and config.use_dummy_weight:
            raise ValueError("AWQ requires a calibrated checkpoint and cannot use dummy weights")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")


def _resolve_kv_cache_dtype(config: EngineConfig) -> torch.dtype:
    if config.kv_cache_dtype == "auto":
        return config.dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    try:
        return dtype_map[config.kv_cache_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported KV cache dtype: {config.kv_cache_dtype}") from exc
