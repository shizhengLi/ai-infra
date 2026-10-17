from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    tokenizer_batch_size: int = 32
    tokenizer_batch_wait_ms: float = 0.0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--quantization",
        choices=["fp8", "awq"],
        default=ServerArgs.quantization,
        help=(
            "Experimental FP8 or calibrated AWQ-Marlin linear-layer quantization. "
            "Embeddings, LM head, KV cache, and collectives retain the model dtype."
        ),
    )

    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "float16", "bfloat16", "float8_e4m3fn"],
        default=ServerArgs.kv_cache_dtype,
        help=(
            "KV cache storage dtype. 'auto' follows --dtype; float8_e4m3fn is an "
            "experimental FlashInfer-only L20 path."
        ),
    )

    parser.add_argument(
        "--kv-cache-scale",
        type=float,
        default=ServerArgs.kv_cache_scale,
        help=(
            "FP8 KV dequantization scale. Keys/values are stored as value / scale; "
            "use a calibrated positive scale when enabling FP8 KV."
        ),
    )

    parser.add_argument(
        "--kv-cache-calibration",
        choices=["fixed", "first_batch"],
        default=ServerArgs.kv_cache_calibration,
        help=(
            "FP8 KV scale policy. 'fixed' uses --kv-cache-scale; 'first_batch' estimates "
            "one max-abs scale per layer from the first eager prefill batch."
        ),
    )

    parser.add_argument(
        "--kv-cache-calibration-summary",
        default=ServerArgs.kv_cache_calibration_summary,
        help=(
            "Optional JSON path prefix for per-rank FP8 KV calibration scale and saturation "
            "summary, written when the scheduler shuts down."
        ),
    )

    parser.add_argument(
        "--kv-cache-fused-store",
        action="store_true",
        default=ServerArgs.kv_cache_fused_store,
        help="Use the experimental fused BF16/FP16-to-FP8 KV cast and page-store kernel.",
    )

    parser.add_argument(
        "--kv-cache-native-pair",
        action="store_true",
        default=ServerArgs.kv_cache_native_pair,
        help=(
            "Use the experimental native BF16/FP16 pairwise conversion in the fused "
            "FP8 KV store; requires --kv-cache-fused-store."
        ),
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--disable-bf16-reduced-precision-reduction",
        action="store_false",
        dest="allow_bf16_reduced_precision_reduction",
        help=(
            "Disable reduced-precision BF16 GEMM reductions. This is an opt-in "
            "performance/accuracy experiment; the default preserves PyTorch behavior."
        ),
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--cuda-graph-bs",
        type=int,
        nargs="+",
        default=ServerArgs.cuda_graph_bs,
        help="Explicit CUDA graph batch sizes. Overrides --cuda-graph-max-bs shape generation.",
    )

    parser.add_argument(
        "--cuda-graph-shape-policy",
        choices=["coarse", "l20", "explicit"],
        default=ServerArgs.cuda_graph_shape_policy,
        help=(
            "CUDA graph shape policy: coarse uses powers/eights, l20 captures exact small "
            "batches and sparse larger shapes, explicit requires --cuda-graph-bs."
        ),
    )

    parser.add_argument(
        "--cuda-graph-disable-bs",
        type=int,
        nargs="+",
        default=ServerArgs.cuda_graph_disable_bs,
        help="CUDA graph batch sizes to execute eagerly. Intended for targeted profiling only.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=ServerArgs.tokenizer_batch_size,
        help="Maximum number of queued tokenizer messages handled in one batch.",
    )

    parser.add_argument(
        "--tokenizer-batch-wait-ms",
        type=float,
        default=ServerArgs.tokenizer_batch_wait_ms,
        help="Optional micro-batch wait after a tokenize message arrives; default is 0 ms.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--prefill-reserve-policy",
        choices=["token", "page-aware"],
        default=ServerArgs.prefill_reserve_policy,
        help=(
            "KV admission reservation policy. 'token' preserves the legacy estimate; "
            "'page-aware' rounds new input/output demand to page boundaries."
        ),
    )

    parser.add_argument(
        "--prefill-new-token-ratio",
        type=float,
        default=ServerArgs.prefill_new_token_ratio,
        help="Output fraction reserved by the page-aware prefill policy (0, 1].",
    )

    parser.add_argument(
        "--max-prefill-streak",
        type=int,
        default=ServerArgs.max_prefill_streak,
        help=(
            "Maximum consecutive prefill batches while decode is runnable. "
            "Zero preserves unlimited prefill priority."
        ),
    )

    parser.add_argument(
        "--decode-active-prefill-length",
        type=int,
        dest="decode_active_prefill_tokens",
        default=ServerArgs.decode_active_prefill_tokens,
        help=(
            "Prefill token budget while decode is runnable. Zero reuses the normal prefill "
            "length; a positive value must not exceed it."
        ),
    )

    parser.add_argument(
        "--decode-overload-prefill-length",
        type=int,
        dest="decode_overload_prefill_tokens",
        default=ServerArgs.decode_overload_prefill_tokens,
        help="Prefill token budget used when decode is runnable and queue pressure is high.",
    )

    parser.add_argument(
        "--decode-overload-prefill-threshold",
        type=int,
        default=ServerArgs.decode_overload_prefill_threshold,
        help="Pending prefill tokens required to activate the decode-overload budget.",
    )

    parser.add_argument(
        "--decode-overload-active-threshold",
        type=int,
        default=ServerArgs.decode_overload_active_threshold,
        help=(
            "Active decode request count required in addition to pending-token pressure. "
            "Zero preserves the queue-only overload policy."
        ),
    )

    parser.add_argument(
        "--decode-result-before-prefill",
        action="store_true",
        default=ServerArgs.decode_result_before_prefill,
        help=(
            "Process and send the previous decode result before submitting pending prefill work."
        ),
    )

    parser.add_argument(
        "--decode-graph-tail-prefill-batch-size",
        type=int,
        default=ServerArgs.decode_graph_tail_prefill_batch_size,
        help=(
            "When positive, finish a previous CUDA-Graph decode result before submitting "
            "prefill if its padded batch size matches this value. Zero disables the guard."
        ),
    )

    parser.add_argument(
        "--decode-graph-tail-prefill-priority-batch-size",
        type=int,
        default=ServerArgs.decode_graph_tail_prefill_priority_batch_size,
        help=(
            "When positive, prefer one decode batch before pending prefill after a matching "
            "CUDA-Graph decode batch. Zero disables the non-blocking priority policy."
        ),
    )

    parser.add_argument(
        "--prefill-telemetry-path",
        default=ServerArgs.prefill_telemetry_path,
        help="Optional JSONL path for cumulative prefill scheduling telemetry.",
    )

    parser.add_argument(
        "--cache-telemetry-path",
        default=ServerArgs.cache_telemetry_path,
        help="Optional rank-0 JSONL path for cumulative prefix-cache telemetry.",
    )

    parser.add_argument(
        "--radix-partial-eviction",
        action="store_true",
        default=ServerArgs.radix_partial_eviction,
        help="Allow page-aligned tail eviction from unprotected radix leaf nodes.",
    )

    parser.add_argument(
        "--radix-partial-eviction-reserve-pages",
        type=int,
        default=ServerArgs.radix_partial_eviction_reserve_pages,
        help="Extra pages to reclaim per partial eviction to amortize later allocations.",
    )

    parser.add_argument(
        "--radix-partial-eviction-adaptive-reserve-max-pages",
        type=int,
        default=ServerArgs.radix_partial_eviction_adaptive_reserve_max_pages,
        help="Bound output-aware partial-eviction reserve to this many pages.",
    )

    parser.add_argument(
        "--radix-partial-eviction-adaptive-reserve-mode",
        choices=["raw", "age-aware"],
        default=ServerArgs.radix_partial_eviction_adaptive_reserve_mode,
        help=(
            "Adaptive reserve accounting: raw sums remaining demand; age-aware discounts "
            "requests according to their remaining output fraction."
        ),
    )

    parser.add_argument(
        "--radix-partial-eviction-protect-recent-matches",
        type=int,
        default=ServerArgs.radix_partial_eviction_protect_recent_matches,
        help=(
            "Prefer cold radix leaves during eviction while protecting this many recently "
            "matched prefix nodes. Zero disables the protection."
        ),
    )

    parser.add_argument(
        "--radix-partial-eviction-hotness-decay",
        type=float,
        default=ServerArgs.radix_partial_eviction_hotness_decay,
        help=(
            "Per-match hotness decay in [0, 1). Positive values rank radix leaves by "
            "decayed reuse evidence during eviction."
        ),
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    if kwargs["tokenizer_batch_size"] < 1:
        parser.error("--tokenizer-batch-size must be at least 1")
    if kwargs["tokenizer_batch_wait_ms"] < 0:
        parser.error("--tokenizer-batch-wait-ms must be non-negative")

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
