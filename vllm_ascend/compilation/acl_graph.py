# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.attention.utils import using_paged_attention

from ..utils import weak_ref_tensors
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id

@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None
    input_tensor_infos: Optional[list[dict[str, Any]]] = None


_ACL_GRAPH_DIAG_ENABLE = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_DIAG", "0") in ("1", "true", "True")
    or os.environ.get("VLLM_ASCEND_SPLIT_DIAG", "0") in ("1", "true", "True")
)
_ACL_GRAPH_DIAG_MAX_LOGS = int(
    os.environ.get("VLLM_ASCEND_ACLGRAPH_DIAG_MAX_LOGS", "600"))
_acl_graph_diag_count = 0
_ACLGRAPH_REPLAY_GLOBAL_SYNC = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_REPLAY_GLOBAL_SYNC", "0")
    in ("1", "true", "True")
)
_ACL_GRAPH_DEBUG_FILE = os.environ.get(
    "VLLM_ASCEND_ACL_GRAPH_DEBUG_FILE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "acl_graph_debug.log")),
)


def _append_acl_graph_debug(tag: str, payload: Any) -> None:
    try:
        with open(_ACL_GRAPH_DEBUG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{}] {}: {}\n".format(ts, tag, payload))
    except Exception as e:
        logger.warning(
            "Failed to write acl graph debug file %s: %s",
            _ACL_GRAPH_DEBUG_FILE,
            e,
        )


def _safe_tensor_shape(tensor: Any):
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return list(tensor.shape)
    except Exception:
        return None


def _safe_tensor_ptr(tensor: Any):
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return int(tensor.data_ptr())
    except Exception:
        return None


def _safe_tensor_head(tensor: Any, max_items: int = 4):
    # Keep compatibility for older payload keys; intentionally disabled.
    return None


def _resolve_callable_arg_names(runnable: Callable) -> Optional[list[str]]:
    try:
        return list(inspect.signature(runnable).parameters.keys())
    except Exception:
        return None


def _resolve_callable_name(runnable: Callable) -> str:
    if hasattr(runnable, "__qualname__"):
        return str(getattr(runnable, "__qualname__"))
    if hasattr(runnable, "__name__"):
        return str(getattr(runnable, "__name__"))
    return type(runnable).__name__


def _collect_tensor_arg_infos(
    args: tuple[Any, ...],
    arg_names: Optional[list[str]] = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    for arg_index, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue
        ptr = _safe_tensor_ptr(arg)
        if ptr is None:
            # Keep list lengths stable for index-based mismatch diagnostics.
            ptr = -1
        addresses.append(ptr)
        tensor_infos.append({
            "tensor_index": len(addresses) - 1,
            "arg_index": arg_index,
            "arg_name": (
                arg_names[arg_index]
                if arg_names is not None and arg_index < len(arg_names)
                else None
            ),
            "shape": _safe_tensor_shape(arg),
            "dtype": str(arg.dtype),
            "device": str(arg.device),
            "stride": list(arg.stride()),
            "is_contiguous": bool(arg.is_contiguous()),
        })
    return addresses, tensor_infos


def _build_input_address_mismatch(
    expected: list[int],
    got: list[int],
    expected_infos: Optional[list[dict[str, Any]]],
    got_infos: Optional[list[dict[str, Any]]],
    max_items: int = 8,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for idx, (exp, new) in enumerate(zip(expected, got)):
        if exp == new:
            continue
        capture_info = expected_infos[idx] if expected_infos and idx < len(expected_infos) else None
        replay_info = got_infos[idx] if got_infos and idx < len(got_infos) else None
        mismatches.append({
            "tensor_index": idx,
            "arg_index": (
                capture_info.get("arg_index")
                if isinstance(capture_info, dict) and "arg_index" in capture_info
                else (
                    replay_info.get("arg_index")
                    if isinstance(replay_info, dict) and "arg_index" in replay_info
                    else None
                )
            ),
            "expected_ptr": exp,
            "got_ptr": new,
            "capture": capture_info,
            "replay": replay_info,
        })
        if len(mismatches) >= max_items:
            break

    return {
        "expected_len": len(expected),
        "got_len": len(got),
        "length_mismatch": len(expected) != len(got),
        "mismatch_count": sum(1 for exp, new in zip(expected, got) if exp != new),
        "mismatches": mismatches,
    }


def _extract_block_table_from_metadata(metadata: Any):
    metadata_block_table = None
    metadata_block_source = None
    for attr in ("block_table", "block_tables", "block_table_tensor"):
        candidate = getattr(metadata, attr, None)
        if isinstance(candidate, torch.Tensor):
            metadata_block_table = candidate
            metadata_block_source = attr
            break

    if metadata_block_table is None:
        decode_metadata = getattr(metadata, "decode", None)
        for attr in ("block_table", "block_tables", "block_table_tensor"):
            candidate = getattr(decode_metadata, attr, None)
            if isinstance(candidate, torch.Tensor):
                metadata_block_table = candidate
                metadata_block_source = f"decode.{attr}"
                break

    return metadata_block_table, metadata_block_source


def _extract_graph_param_block_table(runtime_shape: Any):
    if runtime_shape is None:
        return None, None

    graph_params = get_graph_params()
    if graph_params is None:
        return None, None
    shape_params = graph_params.attn_params.get(runtime_shape)
    if not shape_params:
        return None, None

    first_param = shape_params[0]
    # Prefer FIA slot first, then PA/MLA-like layouts as fallback.
    for idx, source in ((3, "fia.param[3]"), (6, "pa.param[6]"), (10, "mla.param[10]")):
        if len(first_param) > idx and isinstance(first_param[idx], torch.Tensor):
            return first_param[idx], source

    return None, None


def _refresh_block_table_in_place(graph_block_table: Any,
                                  metadata_block_table: Any) -> bool:
    """Refresh captured graph block_table from runtime metadata in-place.

    This is intended for split replay path only. Unsplit path should keep
    original behavior to minimize runtime risk.
    """
    if (not isinstance(graph_block_table, torch.Tensor)
            or not isinstance(metadata_block_table, torch.Tensor)):
        return False

    # No work needed when both already alias the same storage.
    if graph_block_table.data_ptr() == metadata_block_table.data_ptr():
        return False

    try:
        if graph_block_table.ndim == 2 and metadata_block_table.ndim == 2:
            rows = min(graph_block_table.shape[0], metadata_block_table.shape[0])
            cols = min(graph_block_table.shape[1], metadata_block_table.shape[1])
            if rows <= 0 or cols <= 0:
                return False
            graph_block_table[:rows, :cols].copy_(
                metadata_block_table[:rows, :cols], non_blocking=False)
            return True

        # Fallback for non-2D layouts.
        count = min(graph_block_table.numel(), metadata_block_table.numel())
        if count <= 0:
            return False
        graph_block_table.view(-1)[:count].copy_(
            metadata_block_table.view(-1)[:count], non_blocking=False)
        return True
    except Exception:
        return False


def _build_replay_block_table_diag(forward_context: Any, runtime_shape: Any) -> dict[str, Any]:
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not attn_metadata:
        return {}

    first_key = next(iter(attn_metadata), None)
    if first_key is None:
        return {}

    metadata = attn_metadata[first_key]
    metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(metadata)
    graph_block_table, graph_block_table_source = _extract_graph_param_block_table(runtime_shape)
    return {
        "first_key": first_key,
        "runtime_shape": runtime_shape,
        "meta_block_table_source": metadata_block_source,
        "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
        "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
        "graph_block_table_source": graph_block_table_source,
        "graph_block_table_shape": _safe_tensor_shape(graph_block_table),
        "graph_block_table_ptr": _safe_tensor_ptr(graph_block_table),
    }


def _maybe_log_acl_graph_diag(tag: str, payload: Any) -> None:
    global _acl_graph_diag_count
    if ((not _ACL_GRAPH_DIAG_ENABLE and envs.VLLM_LOGGING_LEVEL != "DEBUG")
            or _acl_graph_diag_count >= _ACL_GRAPH_DIAG_MAX_LOGS):
        return
    logger.info("%s: %s", tag, payload)
    _acl_graph_diag_count += 1


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 cudagraph_options: Optional[CUDAGraphOptions] = None,
                 device: torch.device = None,):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.device = device

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self.runnable_name = _resolve_callable_name(runnable)
        self.runnable_arg_names = _resolve_callable_arg_names(runnable)

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()
        # Parallel stream MUST use an independent pool so that two graphs
        # replaying concurrently never share intermediate-tensor addresses.
        self.graph_pool_parallel_streams = torch.npu.graph_pool_handle()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}
        self.concrete_aclgraph_entries2: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or \
                            aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)
        current_concrete_aclgraph_entries = self.concrete_aclgraph_entries2 if in_parallel_streams else self.concrete_aclgraph_entries
        is_new_entry = batch_descriptor not in current_concrete_aclgraph_entries
        if is_new_entry:
            # create a new entry for this batch descriptor
            current_concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = current_concrete_aclgraph_entries[batch_descriptor]
        entry_select_payload = {
            "batch_descriptor": str(batch_descriptor),
            "ubatch_num": getattr(forward_context, "ubatch_num", None),
            "in_parallel_streams": in_parallel_streams,
            "entry_created": is_new_entry,
            "entry_has_graph": entry.aclgraph is not None,
            "entry_id": id(entry),
            "runtime_mode": (
                aclgraph_runtime_mode.name
                if isinstance(aclgraph_runtime_mode, CUDAGraphMode)
                else str(aclgraph_runtime_mode)
            ),
        }
        _append_acl_graph_debug("acl_graph_entry_select", entry_select_payload)
        selected_pool=(self.graph_pool_parallel_streams
                     if in_parallel_streams else self.graph_pool)
        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()
            input_addresses, input_tensor_infos = _collect_tensor_arg_infos(
                args,
                self.runnable_arg_names,
            )
            entry.input_addresses = input_addresses
            entry.input_tensor_infos = input_tensor_infos
            aclgraph = torch.npu.NPUGraph()
            
            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                set_graph_pool_id(selected_pool)
                with torch.npu.graph(aclgraph, pool=selected_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses, new_input_tensor_infos = _collect_tensor_arg_infos(
                args,
                self.runnable_arg_names,
            )
            if new_input_addresses != entry.input_addresses:
                mismatch_detail = _build_input_address_mismatch(
                    entry.input_addresses,
                    new_input_addresses,
                    entry.input_tensor_infos,
                    new_input_tensor_infos,
                )
                _append_acl_graph_debug(
                    "acl_graph_input_addr_mismatch",
                    {
                        "entry_id": id(entry),
                        "runnable": self.runnable_name,
                        "runnable_arg_names": self.runnable_arg_names,
                        "batch_descriptor": str(batch_descriptor),
                        "ubatch_num": getattr(forward_context, "ubatch_num", None),
                        "in_parallel_streams": in_parallel_streams,
                        **mismatch_detail,
                    },
                )
                raise AssertionError(
                    "Input addresses for aclgraphs are different during replay. "
                    f"mismatch_detail={mismatch_detail}"
                )

        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        runtime_shape = getattr(batch_descriptor, "num_tokens", None)
        replay_payload = {
            "entry_id": id(entry),
            "batch_descriptor": str(batch_descriptor),
            "ubatch_num": getattr(forward_context, "ubatch_num", None),
            "in_parallel_streams": in_parallel_streams,
            "replay_global_sync": bool(_ACLGRAPH_REPLAY_GLOBAL_SYNC
                                         and not in_parallel_streams),
            "phase": "pre",
            "runtime_shape": runtime_shape,
        }
        _append_acl_graph_debug("acl_graph_replay", replay_payload)
        # Keep legacy ordering for the main stream path, but avoid global
        # barriers for split parallel replay so two streams can overlap.
        if _ACLGRAPH_REPLAY_GLOBAL_SYNC and not in_parallel_streams:
            torch.npu.synchronize()
        set_graph_pool_id(selected_pool)
        entry.aclgraph.replay()
        replay_post_diag = _build_replay_block_table_diag(
            forward_context, runtime_shape)
        # _maybe_log_acl_graph_diag(
        #     "acl_graph_replay_post",
        #     {
        #         "entry_id": id(entry),
        #         "batch_descriptor": str(batch_descriptor),
        #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
        #         "in_parallel_streams": in_parallel_streams,
        #         "phase": "post",
        #         **replay_post_diag,
        #     },
        # )
        return entry.output


def _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                           refresh_block_table: bool = False):
    graph_params = get_graph_params()
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                seq_lens,
                output,
            ) = param
            seq_lens = forward_context.attn_metadata[key].seq_lens

            metadata = forward_context.attn_metadata[key]
            metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(
                metadata)
            block_table_refreshed = False
            if refresh_block_table:
                block_table_refreshed = _refresh_block_table_in_place(
                    block_table, metadata_block_table)
            # _maybe_log_acl_graph_diag(
            #     "acl_graph_attn_update_diag",
            #     {
            #         "attn_impl": "pa",
            #         "key": key,
            #         "runtime_shape": runtime_shape,
            #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
            #         "seq_lens_shape": _safe_tensor_shape(seq_lens),
            #         "seq_lens_len": len(seq_lens)
            #         if hasattr(seq_lens, "__len__") else None,
            #         "graph_block_table_shape": _safe_tensor_shape(block_table),
            #         "graph_block_table_ptr": _safe_tensor_ptr(block_table),
            #         "meta_block_table_source": metadata_block_source,
            #         "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
            #         "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
            #         "block_table_refreshed": block_table_refreshed,
            #                 },
            # )

            # When using FULL_DECODE_ONLY, there are some rare bugs for FULL_DECODE_ONLY
            # mode with GQA. This is triggered by getting workspace for _npu_paged_attention
            # in torch_npu. On some rare cases, _npu_paged_attention with smaller seq_lens
            # might encounter a bigger workspace, while currently we use max_model_len to
            # calculate max workspace in capturing. So additional get_workspace is added
            # here to avoid such bugs.
            # TODO(Angazenn): we will remove this once _npu_paged_attention is fully
            # replaced by npu_fused_infer_attention_score which does not contain such bugs.
            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output)
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(query=query,
                                           key_cache=key_cache,
                                           value_cache=value_cache,
                                           num_kv_heads=num_kv_heads,
                                           num_heads=num_heads,
                                           scale_value=scale,
                                           block_table=block_table,
                                           context_lens=seq_lens,
                                           out=output,
                                           workspace=workspace)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                            refresh_block_table: bool = False):
    graph_params = get_graph_params()
    # For Qwen3-next, since the kv_cache_config has already categorized
    # linear_attn and self_attn, the attn_metadata is first arranged with
    # self_attn followed by linear_attn. Therefore, using zip directly
    # filters out the update operations for linear_attn.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (query, key_cache, value, block_tables, attn_mask, block_size,
             seq_lens, query_start_loc, num_kv_heads, num_heads, scale,
             attn_output, softmax_lse) = param

            metadata = forward_context.attn_metadata[key]
            seq_lens = metadata.seq_lens_list
            actual_seq_lengths_q = metadata.actual_seq_lengths_q
            metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(
                metadata)
            block_table_refreshed = False
            if refresh_block_table:
                block_table_refreshed = _refresh_block_table_in_place(
                    block_tables, metadata_block_table)
            # _maybe_log_acl_graph_diag(
            #     "acl_graph_attn_update_diag",
            #     {
            #         "attn_impl": "fia",
            #         "key": key,
            #         "runtime_shape": runtime_shape,
            #         "ubatch_num": getattr(forward_context, "ubatch_num", None),
            #         "seq_lens_shape": _safe_tensor_shape(seq_lens),
            #         "seq_lens_len": len(seq_lens)
            #         if hasattr(seq_lens, "__len__") else None,
            #         "graph_block_table_shape": _safe_tensor_shape(block_tables),
            #         "graph_block_table_ptr": _safe_tensor_ptr(block_tables),
            #         "meta_block_table_source": metadata_block_source,
            #         "meta_block_table_shape": _safe_tensor_shape(metadata_block_table),
            #         "meta_block_table_ptr": _safe_tensor_ptr(metadata_block_table),
            #         "block_table_refreshed": block_table_refreshed,
            #                 },
            # )
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value,
                block_table=block_tables,
                atten_mask=attn_mask,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=seq_lens,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=3,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_attn_params(update_stream, forward_context, runtime_shape,
                       vllm_config):
    if using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape)


def update_attn_params_split(update_stream, forward_context,
                             runtime_shape, vllm_config):
    """Split-only attn update with block_table in-place refresh enabled."""
    if using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
        )
    else:
        _update_attn_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
        )


def update_mla_attn_params(update_stream, forward_context, runtime_shape,
                           speculative_config):
    if forward_context.is_mtp_model:
        graph_params = get_mtp_graph_params()
    else:
        graph_params = get_graph_params()
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (q_nope, k_nope, q_pe, k_pe, num_heads, num_kv_heads, input_layout,
             spec_attn_mask, sparse_mode, scale, block_table, block_size,
             seq_lens_list, actual_seq_lengths, attn_output,
             softmax_lse) = param
            seq_lens_list = forward_context.attn_metadata[
                key].decode.seq_lens_list
            if speculative_config and speculative_config.method == "mtp" \
                    and not forward_context.is_mtp_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                spec_multiple = speculative_config.num_speculative_tokens + 1
                seq_lens_list = seq_lens_list + [0] * (
                    runtime_shape // spec_multiple - len(seq_lens_list))
                actual_seq_lengths = [
                    spec_multiple * (i + 1)
                    for i in range(runtime_shape // spec_multiple)
                ]
            elif forward_context.is_mtp_model:
                actual_seq_lengths = forward_context.attn_metadata[
                    key].decode.actual_seq_lengths_q
                block_table = forward_context.attn_metadata[
                    key].decode.block_table
                # TODO: This is a hack and should be fixed in the future.
                if speculative_config.disable_padded_drafter_batch:
                    block_table = block_table[:len(actual_seq_lengths)]
                seq_lens_list = seq_lens_list + [0] * (
                    len(actual_seq_lengths) - len(seq_lens_list))
            else:
                seq_lens_list = seq_lens_list + [0] * (runtime_shape -
                                                       len(seq_lens_list))
            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout=input_layout,
                atten_mask=spec_attn_mask,
                sparse_mode=sparse_mode,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=seq_lens_list,
                actual_seq_lengths=actual_seq_lengths,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_attn_dcp_pcp_params(update_stream, forward_context, runtime_shape):
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    graph_params = get_graph_params()
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (q_nope, k_nope, value, num_heads, num_kv_heads, scale,
             block_table, block_size, actual_seq_lengths_kv,
             actual_seq_lengths_q, attn_output, softmax_lse, dcp_size,
             pcp_rank, dcp_rank) = param
            attn_metadata = forward_context.attn_metadata[key]
            actual_seq_lengths_kv = attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:,
                                                                                             pcp_rank,
                                                                                             dcp_rank]
            pad_length = runtime_shape - len(actual_seq_lengths_kv)
            if pad_length > 0:
                pad_tensor = np.zeros(pad_length,
                                      dtype=actual_seq_lengths_kv.dtype)
                actual_seq_lengths_kv = np.concatenate(
                    [actual_seq_lengths_kv, pad_tensor])

            actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[:
                                                                      attn_metadata
                                                                      .
                                                                      num_decode_tokens]
            if (runtime_shape - len(actual_seq_lengths_q)):
                actual_seq_lengths_q = actual_seq_lengths_q + [
                    actual_seq_lengths_q[-1]
                ] * (runtime_shape - len(actual_seq_lengths_q))
            if dcp_size > 1:
                num_heads = num_heads * dcp_size

            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                value,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=scale,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths_q,
                workspace=graph_params.workspaces.get(runtime_shape),
                out=[attn_output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_mla_attn_dcp_pcp_params(update_stream, forward_context,
                                   runtime_shape):
    graph_params = get_graph_params()
    # FIXME: Behold! We are using a temporary hack here to update the args
    # for each layer's attention op in the graph.
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[runtime_shape],
                graph_params.handles[runtime_shape],
                graph_params.events[runtime_shape],
        ):
            (q_nope, q_pe, k_nope, k_pe, block_table, seq_len, num_heads,
             scale, num_kv_heads, attn_output, softmax_lse) = param

            decode_meta = forward_context.attn_metadata[key].decode
            seq_len = decode_meta.cp_seq_len

            # For pcp + spec decode, we flatten seq_lens
            # to avoid irregular spec_attn_mask shape,
            # so there's no need to divide runtime_shape by spec_multiple
            pad_length = runtime_shape - len(seq_len)
            pad_tensor = torch.zeros(pad_length,
                                     dtype=seq_len.dtype,
                                     device=seq_len.device)
            seq_len = torch.cat([seq_len, pad_tensor], dim=0)

            torch.npu.graph_task_update_begin(update_stream, handle)

            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                block_table,
                seq_len,
                num_heads,
                scale,
                num_kv_heads,
                return_lse=True,
                calc_type="calc_type_ring",
                workspace=graph_params.workspaces.get(runtime_shape),
                output=attn_output,
                lse=softmax_lse)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = weak_ref_tensors(workspace)


def get_graph_params():
    return _graph_params


_mtp_graph_params: Optional[GraphParams] = None


def set_mtp_graph_params(aclgraph_capture_sizes: list[int]):
    global _mtp_graph_params
    if _mtp_graph_params is not None:
        raise ValueError("MTPGraph parameters have already been set!")
    _mtp_graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )


def update_mtp_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _mtp_graph_params
    if _mtp_graph_params is not None:
        _mtp_graph_params.workspaces[num_tokens] = workspace


def get_mtp_graph_params():
    return _mtp_graph_params
