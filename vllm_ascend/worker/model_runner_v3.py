#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#
import os
import json
import math
import time
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import regex as re
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm  # type: ignore
from typing_extensions import TypeAlias
from vllm.attention.backends.abstract import AttentionBackend, AttentionType, AttentionMetadata
from vllm.attention.layer import Attention, MLAAttention
from vllm.attention.selector import get_attn_backend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_dcp_group, get_dp_group,
                                             get_pcp_group, get_pp_group,
                                             get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import (BatchDescriptor, DPMetadata,
                                  get_forward_context,
                                  override_forward_context)
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             SamplerOutput,
                             make_empty_encoder_model_runner_output)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.gpu_model_runner import (AsyncGPUModelRunnerOutput,
                                             GPUModelRunner)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds
from vllm.v1.worker.utils import (AttentionGroup, gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         AscendPrefillContextParallelMetadata)
# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               set_graph_params,
                                               set_graph_params_parallel,
                                               set_mtp_graph_params,
                                               update_attn_dcp_pcp_params,
                                               update_attn_params,
                                               update_attn_params_split,
                                               update_mla_attn_dcp_pcp_params,
                                               update_mla_attn_params)
# yapf: enable
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.sample.logits_processor import build_logitsprocs
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (AscendDeviceType, ProfileExecuteDuration,
                               enable_sp, get_ascend_device_type, is_moe_model,
                               lmhead_tp_enable, maybe_trans_nz)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.npu_ubatch_wrapper import (AscendUBatchWrapper,
                                                   AscendUbatchMetadata)
from vllm_ascend.worker.ubatch_utils import (SplitBatchSlices,
                                             split_batch_split, ubatch_split)
from vllm_ascend.attention.utils import split_attn_metadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType, create_ascend_forward_context, get_mc2_tokens_capacity,
    select_moe_comm_method, set_ascend_forward_context, set_mc2_mask,
    set_mc2_tokens_capacity)

from msprobe.pytorch import AclGraphDumper

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if get_ascend_device_type() == AscendDeviceType._310P:
    torch_npu.npu.set_compile_mode(jit_compile=False)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[list[AttnMetadataDict],
                                        AttnMetadataDict]



@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


_SPLIT_METADATA_DEBUG_FILE = os.environ.get(
    "VLLM_ASCEND_SPLIT_METADATA_DEBUG_FILE",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "split_metadata_debug.log")
    ),
)

# Perf stats log file: each decode step appends one JSON line with timing info.
# Set VLLM_ASCEND_PERF_STATS_FILE to a path to enable; empty string disables.
_PERF_STATS_FILE = os.environ.get(
    "VLLM_ASCEND_PERF_STATS_FILE",
    "",
)

def _write_perf_stats(stats: dict) -> None:
    """Append one JSON line to the perf stats file (if enabled)."""
    if not _PERF_STATS_FILE:
        return
    try:
        import json as _json
        with open(_PERF_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(stats, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to write perf stats to %s: %s",
                       _PERF_STATS_FILE, e)
# Rollback switch for split context rebuild coordinate validation.
# - default "1": use local coordinate slices for rebuilt split context
# - set VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD=0 to restore legacy behavior
_SPLIT_LOCAL_CONTEXT_REBUILD = os.environ.get(
    "VLLM_ASCEND_SPLIT_LOCAL_CONTEXT_REBUILD", "1") not in ("0", "false", "False")


def _append_split_metadata_debug(tag: str, payload: Any) -> None:
    try:
        with open(_SPLIT_METADATA_DEBUG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{}] {}: {}\n".format(ts, tag, payload))
    except Exception as e:
        logger.warning(
            "Failed to write split metadata debug file %s: %s",
            _SPLIT_METADATA_DEBUG_FILE,
            e,
        )


def _clone_attn_metadata_block_tables(attn_metadata: Any) -> Any:
    """Return a copy of attn_metadata with cloned block_tables tensors.

    When capturing ACL graphs for the parallel stream, both _graph_params and
    _graph_params_parallel would otherwise bind the *same* device block_table
    storage (from self.input_batch.block_table).  At runtime the two concurrent
    _refresh_block_table_in_place calls would both write to block_table[:N, :]
    starting from row 0, causing a data race that corrupts KV-cache lookups.

    Cloning gives _graph_params_parallel its own device buffer so the two
    in-place refreshes target distinct memory regions.
    """
    import dataclasses

    def _clone_single(meta: Any) -> Any:
        if meta is None or not dataclasses.is_dataclass(meta):
            return meta
        kwargs: dict = {}
        if getattr(meta, "block_tables", None) is not None:
            kwargs["block_tables"] = meta.block_tables.clone()
        # Also clone sub-metadata that carry their own block_tables.
        for sub_field in ("prefill", "decode_meta"):
            sub = getattr(meta, sub_field, None)
            if sub is not None and dataclasses.is_dataclass(sub):
                sub_kwargs: dict = {}
                if getattr(sub, "block_tables", None) is not None:
                    sub_kwargs["block_tables"] = sub.block_tables.clone()
                if sub_kwargs:
                    kwargs[sub_field] = dataclasses.replace(sub, **sub_kwargs)
        return dataclasses.replace(meta, **kwargs) if kwargs else meta

    if isinstance(attn_metadata, dict):
        return {k: _clone_single(v) for k, v in attn_metadata.items()}
    if isinstance(attn_metadata, list):
        return [
            {k: _clone_single(v) for k, v in d.items()}
            if isinstance(d, dict) else _clone_single(d)
            for d in attn_metadata
        ]
    return _clone_single(attn_metadata)


def _safe_tensor_ptr(tensor: Any) -> Optional[int]:
    if isinstance(tensor, torch.Tensor):
        return int(tensor.data_ptr())
    return None


def _safe_tensor_head(tensor: Any, max_items: int = 3) -> Any:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        if tensor.ndim == 2:
            return tensor[:, :max_items].detach().cpu().tolist()
        return tensor[:max_items].detach().cpu().tolist()
    except Exception:
        return None


def _safe_tensor_shape(tensor: Any) -> Optional[list[int]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return list(tensor.shape)
    except Exception:
        return None


def _safe_context_id(context: Any) -> Optional[int]:
    if context is None:
        return None
    try:
        return id(context)
    except Exception:
        return None


def _build_split_tensor_debug(name: str, tensor: Any) -> dict[str, Any]:
    return {
        f"{name}_ptr": _safe_tensor_ptr(tensor),
        f"{name}_shape": _safe_tensor_shape(tensor),
        f"{name}_head": _safe_tensor_head(tensor),
    }


def _extract_attn_positions(attn_metadata: Any) -> tuple[Optional[int], Any]:
    candidate = attn_metadata
    if isinstance(candidate, dict) and candidate:
        candidate = next(iter(candidate.values()))
    if isinstance(candidate, list) and candidate:
        candidate = candidate[0]

    common_attn_metadata = getattr(candidate, "common_attn_metadata", None)
    if common_attn_metadata is not None:
        candidate = common_attn_metadata

    positions = getattr(candidate, "positions", None)
    return _safe_tensor_ptr(positions), _safe_tensor_head(positions)




def _iter_attn_metadata_objects(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield from _iter_attn_metadata_objects(value)
        return
    if isinstance(attn_metadata, list):
        for value in attn_metadata:
            yield from _iter_attn_metadata_objects(value)
        return
    if attn_metadata is not None:
        yield attn_metadata


def _get_slot_mapping_from_attn_metadata(
        attn_metadata: Any) -> Optional[torch.Tensor]:
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        slot_mapping = getattr(metadata_obj, "slot_mapping", None)
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping
        common_attn_metadata = getattr(metadata_obj, "common_attn_metadata",
                                       None)
        slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping
    return None


def _set_slot_mapping_for_attn_metadata(attn_metadata: Any,
                                        slot_mapping: torch.Tensor) -> int:
    updated = 0
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        if hasattr(metadata_obj, "slot_mapping"):
            setattr(metadata_obj, "slot_mapping", slot_mapping)
            updated += 1
        common_attn_metadata = getattr(metadata_obj, "common_attn_metadata",
                                       None)
        if (common_attn_metadata is not None
                and hasattr(common_attn_metadata, "slot_mapping")):
            setattr(common_attn_metadata, "slot_mapping", slot_mapping)
            updated += 1
    return updated


def _validate_split_attn_metadata_count(
    tag: str,
    common_attn_metadata_list: Any,
    expected_splits: int,
) -> None:
    actual_len = len(common_attn_metadata_list) if isinstance(common_attn_metadata_list, list) else None
    payload = {
        "tag": tag,
        "expected_splits": expected_splits,
        "actual_type": type(common_attn_metadata_list).__name__,
        "actual_len": actual_len,
    }
    _append_split_metadata_debug("split_attn_metadata_count", payload)

    if (
        not isinstance(common_attn_metadata_list, list)
        or actual_len is None
        or actual_len != expected_splits
    ):
        raise RuntimeError(
            "split_attn_metadata returned unexpected split count: "
            f"expected={expected_splits}, actual_type={type(common_attn_metadata_list).__name__}, "
            f"actual_len={actual_len}, tag={tag}"
        )


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group(
            ).rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        if envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP:
            self.prefetch_stream = torch.npu.Stream(device=device)
        else:
            self.prefetch_stream = None
        self.sampler = AscendSampler()
        self.attn_mask = None
        self.attn_state = None

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        self.weight_prefetch_method = WeightPrefetchMethod(
            self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config
        self.dump_enable = dump_cfg.enable_dump
        self.debugger = None
        if self.dump_enable:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger
                self.debugger = PrecisionDebugger(dump_cfg.config_path)
            else:
                raise RuntimeError(
                    "Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False


        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_config,
                                  "index_topk")
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs,
                        self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs,
                                self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.pcp_allgather_restore_idx = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device)
        self.cp_kv_recover_idx_for_chunk: List[List[int]] = [
            [] for _ in range(self.pcp_size)
        ]

        self.num_pcp_pads = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        self.pcp_padded_slot_mapping = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device)
        self.num_actual_tokens_pcp_padded = 0
        if self.speculative_config and self.pcp_size > 1:
            self.input_ids_pcp_full = self._make_buffer(self.max_num_tokens,
                                                        dtype=torch.int32)
            self.query_start_loc_pcp_full = self._make_buffer(
                self.max_num_reqs + 1, dtype=torch.int32)
            self.positions_pcp_full = torch.zeros(self.max_num_tokens,
                                                  dtype=torch.int64,
                                                  device="cpu",
                                                  pin_memory=True)
            self.decode_token_per_req += self.speculative_config.num_speculative_tokens
            self.positions_pcp_full_np = self.positions_pcp_full.numpy()
        self.decode_threshold = 1 + (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        self.dynamic_eplb = self.ascend_config.dynamic_eplb or self.ascend_config.expert_map_record_path
        if self.dynamic_eplb:
            EPLBParamUtils.check_dynamic_eplb(self.ascend_config.dynamic_eplb)
            EPLBParamUtils.check_expert_map_record_path(
                self.ascend_config.expert_map_record_path)
            self.is_eplb_warmuped = False
            self.policy_type = self.ascend_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({
                "expert_map": None,
                "moe_load": None,
                "expert_maps": None
            })
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                            policy_type=self.policy_type,
                                            enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            ascend_config = get_ascend_config()
            self.eplb_updator = EplbUpdator(ascend_config, self.eplb_loader,
                                            self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config else 0),
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        #并行新增输入地址
        self.input_ids_parallel_streams=self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.inputs_embeds_parallel_streams=self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.positions_parallel_streams=self._make_buffer(self.max_num_tokens, dtype=torch.int64)

        self.stream_main = torch.npu.current_stream()
        self.stream_parallel = torch.npu.Stream(device=self.device)

        # Performance measurement accumulators.
        # _perf_accum holds running totals across all decode steps so that
        # callers can compute TPOT = total_replay_ms / total_output_tokens.
        # _last_step_perf holds the most recent step's breakdown.
        self._perf_accum: dict = {
            "total_replay_ms": 0.0,
            "total_header_ms": 0.0,
            "total_output_tokens": 0,
            "num_decode_steps": 0,
        }
        self._last_step_perf: dict = {}
        self._t_replay_start: float = 0.0
        self._t_replay_end: float = 0.0
        self._t_header_start: float = 0.0


    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.spec_attn_mask = None
        self.drafter: Optional[Union[NgramProposer, EagleProposer, MtpProposer,
                                     SuffixDecodingProposer]] = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            self.spec_attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask(
            )
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                self.rejection_sampler = RejectionSampler(self.sampler)
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1,
                      self.decode_token_per_req))
        self.discard_request_indices = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method,
                                      self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.compilation_config.mode == CompilationMode.VLLM_COMPILE and not self.model_config.enforce_eager

    def _skip_all_reduce_acorss_dp_group(self) -> bool:
        # NOTE: We can skip the all_reduce operation and avoid paading tokens
        # to max_tokens_acrodd_dp in D nodes. In MoE models, we must ensure that
        # num_tokens DOES NOT exceed mc2_tokens_capacity which means that moe_comm_method
        # of each rank is MC2. For dense models, skipping all_reduce is not necessary
        # since collective-communication is not time-consuming since dp_size in dense
        # model deployments is always small and can be overlapped by async scheduling.
        if not is_moe_model(self.vllm_config):
            return False
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_num_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_num_tokens = self.max_num_reqs * self.uniform_decode_query_len
        # To ensure skipping all_reduce across dp group is valid, we need to ensure that
        # moe_comm_method of each rank is MC2 and recomputation would never happen in D
        # nodes. So here we check whether recompute_scheduler_enable is True.
        return self.is_kv_consumer and self.ascend_config.recompute_scheduler_enable and select_moe_comm_method(
            potential_max_num_tokens,
            self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

    def _sync_metadata_across_dp(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill, enable_dbo

        if self._skip_all_reduce_acorss_dp_group():
            num_tokens_after_padding = torch.tensor([num_tokens] *
                                                    self.dp_size,
                                                    device="cpu",
                                                    dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill, enable_dbo

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor([
            num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
        ],
                                         dtype=torch.int32,
                                         device="cpu")

        flags_tensor = torch.tensor(
            [int(with_prefill), int(enable_dbo)],
            dtype=torch.int32,
            device="cpu")

        packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
        # use cpu_group to avoid cpu synchronization issue.
        # it can be overlapped with main moell execution on npu.
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:-2]
        synced_flags = packed_tensor[-2:]

        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])
        # all the ranks should execute dummy run at the same time
        global_enable_dbo = (synced_flags[1] == get_dp_group().world_size
                             or synced_flags[1] == 0)

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, global_enable_dbo

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model,
                      (ACLGraphWrapper, AscendUBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def _make_attention_mask(self, attn_state) -> torch.Tensor:
        # pcp situation.
        if self.attn_mask_builder is None:
            raise ValueError("Attn mask builder is None")
        # Pooling situation.
        if self.model_config.runner_type == "pooling":
            return self.attn_mask_builder.get_attn_mask(2048, torch.bool)

        if self.vllm_config.model_config.use_mla:
            if self.pcp_size > 1:
                return self.attn_mask_builder.get_pcp_mla_mask(self.dtype)
            # mla prefill
            if attn_state != AscendAttentionState.DecodeOnly:
                return self.attn_mask_builder.get_mla_mask(self.dtype)
        return self.attn_mask_builder.get_splitfuse_attn_mask()

    def generate_kv_idx(self, scheduler_output):
        if not self.pcp_size > 1:
            return
        self.cp_kv_recover_idx_for_chunk = [[] for _ in range(self.pcp_size)]

        for i, req_id in enumerate(self.input_batch.req_ids):
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            is_prefill = self.input_batch.num_computed_tokens_cpu[
                i] < self.input_batch.num_prompt_tokens[i]
            if is_prefill:
                num_cp_padded_scheduled_tokens = cdiv(
                    num_scheduled_tokens,
                    2 * self.pcp_size) * (2 * self.pcp_size)
                full_indices = list(
                    range(self.max_num_tokens * self.pcp_size * self.dcp_size +
                          self.pcp_size * self.dcp_size * self.max_num_reqs))
                chunk_size = num_cp_padded_scheduled_tokens // (2 *
                                                                self.pcp_size)
                num_added_recover_tokens = len(
                    self.cp_kv_recover_idx_for_chunk[0]) * self.pcp_size
                for rank in range(self.pcp_size):
                    self.cp_kv_recover_idx_for_chunk[rank].extend(
                        full_indices[rank * chunk_size +
                                     num_added_recover_tokens:(rank + 1) *
                                     chunk_size + num_added_recover_tokens])
                    self.cp_kv_recover_idx_for_chunk[rank].extend(
                        full_indices[num_cp_padded_scheduled_tokens -
                                     (rank + 1) * chunk_size +
                                     num_added_recover_tokens:
                                     num_cp_padded_scheduled_tokens -
                                     rank * chunk_size +
                                     num_added_recover_tokens])

        cp_kv_recover_idx_for_chunk = torch.from_numpy(
            np.concatenate(
                self.cp_kv_recover_idx_for_chunk)).to(device=self.device)
        cp_kv_recover_idx_for_chunk.copy_(torch.tensor(
            np.array(self.cp_kv_recover_idx_for_chunk).flatten().tolist()),
                                          non_blocking=True)
        self.cp_kv_recover_idx_for_chunk = cp_kv_recover_idx_for_chunk.to(
            torch.float32).argsort().to(torch.int32)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[PerLayerAttnMetadata, torch.Tensor, np.ndarray, int,
               torch.Tensor, int, torch.Tensor, SpecDecodeMetadata,
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], int, Optional[UBatchSlices],
               Optional[SplitBatchSlices], Optional[torch.Tensor]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)

        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        _, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        total_num_pcp_pads = 0
        if self.pcp_size > 1:
            if not self.vllm_config.model_config.use_mla:
                self.generate_kv_idx(scheduler_output)
            tokens, position_pcp, pcp_unpad_mask = self._update_tokens_for_pcp(
                tokens)
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            total_num_pcp_pads = torch.sum(self.num_pcp_pads).item()
        else:
            position_pcp, pcp_unpad_mask = None, None
            self.num_pcp_pads = self.num_pcp_pads[:num_reqs]

        max_num_scheduled_tokens = max(tokens)
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(tokens, dtype=np.int32)
        else:
            num_valid_tokens = np.array([
                num_tokens -
                len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
                                        dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.cudagraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = num_tokens_unpadded
        uniform_decode = \
            (max_num_scheduled_tokens == self.uniform_decode_query_len) and \
            (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
        moe_comm_type = select_moe_comm_method(num_input_tokens,
                                               self.vllm_config)
        ubatch_slices, num_tokens_after_padding = \
            ubatch_split(num_scheduled_tokens,
                         num_tokens_unpadded,
                         num_tokens_padded,
                         uniform_decode=uniform_decode,
                         vllm_config=self.vllm_config,
                         moe_comm_type=moe_comm_type,
                         use_mla = self.model_config.use_mla)
        if ubatch_slices is not None:
            enable_dbo = True
        else:
            enable_dbo = False

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        if not enable_dbo:
            ubatch_slices = None


        split_batch_slices: Optional[SplitBatchSlices] = None
        split_ubatch_slices: Optional[UBatchSlices] = None
        # Split batch - compute split slices for large decode batches
        # Split batch and DBO never conflict:
        # - DBO is for overlapping compute/communication
        # - Split batch is for splitting large uniform decode batches
        if uniform_decode and ubatch_slices is None:  # Only split if DBO is not active
            cudagraph_capture_sizes = set(
                self.compilation_config.cudagraph_capture_sizes or []
            ) if self.use_aclgraph else None
            custom_split_sizes = None
            _should_split = True  # whether to call split_batch_split at all
            if cudagraph_capture_sizes and self.use_aclgraph:
                # Find the largest main-stream graph size that fits within
                # num_reqs without padding.  The remainder goes to the parallel
                # stream (which may be padded to its own nearest graph size).
                sorted_main_sizes = sorted(cudagraph_capture_sizes)
                max_main_size = sorted_main_sizes[-1]

                # Helper: ceil num to the nearest graph size in a sorted list.
                def _ceil_to_graph(n: int, sizes: list[int]) -> int:
                    for s in sizes:
                        if s >= n:
                            return s
                    return sizes[-1]  # clamp to max if n exceeds all sizes

                main_reqs = max(
                    (s for s in sorted_main_sizes if s <= num_reqs),
                    default=0,
                )
                parallel_reqs = num_reqs - main_reqs

                # force_split: skip threshold check and always split as
                # (main_reqs + parallel_reqs) whenever there is a non-trivial
                # remainder and main_reqs hits a captured graph.
                _split_cfg = getattr(
                    self.ascend_config, "split_batch_config", None)
                force_split = bool(
                    _split_cfg is not None
                    and getattr(_split_cfg, "force_split", False)
                )

                if main_reqs == num_reqs:
                    if force_split and len(sorted_main_sizes) >= 2:
                        # Force split even when num_reqs exactly hits a graph.
                        # Use the second-largest graph size <= num_reqs as the
                        # main slice so there is a non-zero parallel remainder.
                        candidates = [s for s in sorted_main_sizes
                                      if s < num_reqs]
                        if candidates:
                            main_reqs = max(candidates)
                            parallel_reqs = num_reqs - main_reqs
                            custom_split_sizes = [main_reqs, parallel_reqs]
                        else:
                            # num_reqs == smallest graph size; cannot split.
                            _should_split = False
                    else:
                        # num_reqs exactly hits a main-stream graph: no split.
                        # Let split_ubatch_slices stay None so the forward pass
                        # goes through _generate_process_reqs_hidden_states.
                        _should_split = False
                elif main_reqs > 0 and parallel_reqs > 0:
                    if force_split:
                        # Force split: always use (main_reqs + parallel_reqs)
                        # regardless of padding savings or whether num_reqs
                        # exceeds the largest captured graph size.
                        custom_split_sizes = [main_reqs, parallel_reqs]
                    elif num_reqs > max_main_size:
                        # num_reqs exceeds all captured sizes; no graph to pad
                        # to, so no split benefit under normal threshold logic.
                        _should_split = False
                    else:
                        # Evaluate whether splitting saves enough padding.
                        #
                        # padding_saved = padding wasted without split
                        #               - padding the parallel slice still needs
                        #
                        # Without split: num_reqs pads up to the next graph.
                        # With split:    main_reqs hits a graph (0 padding);
                        #                parallel_reqs pads to its nearest graph.
                        threshold = getattr(
                            self.compilation_config,
                            "cudagraph_split_pad_threshold",
                            0,
                        )
                        # Parallel-stream capture sizes (may differ from main).
                        parallel_sizes = sorted(
                            getattr(
                                self, "cudagraph_batch_sizes_parallel", None)
                            or sorted_main_sizes
                        )
                        original_padded = _ceil_to_graph(
                            num_reqs, sorted_main_sizes)
                        original_padding = original_padded - num_reqs
                        remainder_padded = _ceil_to_graph(
                            parallel_reqs, parallel_sizes)
                        remainder_padding = remainder_padded - parallel_reqs
                        padding_saved = original_padding - remainder_padding
                        if padding_saved > threshold:
                            custom_split_sizes = [main_reqs, parallel_reqs]
                        else:
                            # Not worth splitting; let the batch pad normally.
                            _should_split = False
                else:
                    # main_reqs == 0: num_reqs is smaller than all captured
                    # graph sizes; fall through to split_batch_split's own logic.
                    pass
            if _should_split:
                split_batch_slices, _ = split_batch_split(
                    num_scheduled_tokens,
                    num_tokens_unpadded,
                    num_tokens_padded,
                    vllm_config=self.vllm_config,
                    cudagraph_capture_sizes=cudagraph_capture_sizes,
                    custom_split_sizes=custom_split_sizes,
                )
                if split_batch_slices:
                    split_ubatch_slices = [
                        UBatchSlice(s.request_slice, s.token_slice)
                        for s in split_batch_slices
                    ]


        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        if self.pcp_size > 1:
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                   position_pcp[:total_num_scheduled_tokens],
                   out=positions_np)
        else:
            self.positions.np[:total_num_scheduled_tokens] = positions_np

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or
                                                   self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()

        # OPTIMIZATION: If split batch is enabled, directly write the second split's
        # data to parallel_streams buffers during preparation, avoiding the copy
        # overhead in _make_split_batch_metadata_parallel_streams.
        # NOTE: inputs_embeds is handled separately in _make_split_batch_metadata_parallel_streams
        # because it's populated after _prepare_inputs returns.
        if split_batch_slices is not None and len(split_batch_slices) > 1:
            second_split = split_batch_slices[1]
            second_token_start = second_split.token_slice.start
            second_token_end = second_split.token_slice.stop
            second_num_tokens = second_token_end - second_token_start
            second_padded_tokens = second_split.padded_num_tokens

            # Copy positions for second split to parallel_streams buffer,
            # then zero-pad the tail up to padded_num_tokens.
            if self.positions.gpu.ndim == 2:
                # M-RoPE case
                self.positions_parallel_streams.gpu[:, :second_num_tokens].copy_(
                    self.positions.gpu[:, second_token_start:second_token_end])
                if second_padded_tokens > second_num_tokens:
                    self.positions_parallel_streams.gpu[
                        :, second_num_tokens:second_padded_tokens].fill_(0)
            else:
                self.positions_parallel_streams.gpu[:second_num_tokens].copy_(
                    self.positions.gpu[second_token_start:second_token_end])
                if second_padded_tokens > second_num_tokens:
                    self.positions_parallel_streams.gpu[
                        second_num_tokens:second_padded_tokens].fill_(0)

            # Copy input_ids for second split to parallel_streams buffer,
            # then zero-pad the tail up to padded_num_tokens.
            self.input_ids_parallel_streams.gpu[:second_num_tokens].copy_(
                self.input_ids.gpu[second_token_start:second_token_end])
            if second_padded_tokens > second_num_tokens:
                self.input_ids_parallel_streams.gpu[
                    second_num_tokens:second_padded_tokens].fill_(0)

            # Zero-pad main stream (split[0]) input_ids and positions beyond
            # actual token count up to its padded capture size.
            first_split = split_batch_slices[0]
            first_num_tokens = first_split.num_tokens
            first_padded_tokens = first_split.padded_num_tokens
            if first_padded_tokens > first_num_tokens:
                self.input_ids.gpu[first_num_tokens:first_padded_tokens].fill_(0)
                if self.positions.gpu.ndim == 2:
                    self.positions.gpu[:, first_num_tokens:first_padded_tokens].fill_(0)
                else:
                    self.positions.gpu[first_num_tokens:first_padded_tokens].fill_(0)

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp

        attn_metadata: PerLayerAttnMetadata = {}
        split_ubatch_slices_for_metadata: Optional[UBatchSlices] = None
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]
            split_ubatch_slices_for_metadata = ubatch_slices
        elif split_ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(split_ubatch_slices))]
            split_ubatch_slices_for_metadata = split_ubatch_slices

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                np.array(tokens_original, dtype=np.int32))
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
            ):
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)

                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                    scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the acl graph all the time. The v0
            # engine avoids this by "double compiling" the acl graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the acl graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions.gpu[:num_input_tokens]
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]

        # type: ignore
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # If both flashcomm1 and pp are used simultaneously,
            # the shape of the received data and the shape of the space to be copied to will not match,
            # requiring a recalculation of the incoming data's shape.
            tp_size = get_tensor_model_parallel_world_size()
            num_input_tokens_with_flashcomm1 = num_input_tokens
            if enable_sp():
                num_input_tokens_with_flashcomm1 = (num_input_tokens +
                                                    tp_size - 1) // tp_size
                if split_ubatch_slices_for_metadata is not None:
                    # for dbo, we calculate the size of intermediate tensors
                    # later in ubatch_wrapper
                    num_input_tokens_with_dbo = (
                        (ubatch_slices[0].num_tokens + tp_size - 1) // tp_size
                    ) + (
                        (ubatch_slices[1].num_tokens + tp_size - 1) // tp_size)
                    intermediate_tensor_size = next(
                        iter(self.intermediate_tensors.tensors.values())).size(
                            0)
                    if intermediate_tensor_size < num_input_tokens_with_dbo:
                        self.intermediate_tensors = (
                            self.model.make_empty_intermediate_tensors(
                                batch_size=num_input_tokens_with_dbo,
                                dtype=self.dtype,
                                device=self.device))
                    num_input_tokens_with_flashcomm1 = max(
                        num_input_tokens_with_flashcomm1,
                        num_input_tokens_with_dbo)
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[
                    k][:num_input_tokens_with_flashcomm1].copy_(
                        v[:num_input_tokens_with_flashcomm1],
                        non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k:
                v[:num_input_tokens_with_flashcomm1]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = torch.from_numpy(
                    cu_num_tokens
                ) * self.pcp_size - self.num_pcp_pads[:num_reqs] - 1
                logits_indices = logits_indices.pin_memory().to(
                    self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1:num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, self.num_pcp_pads[:num_reqs])
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        if self.speculative_config and self.pcp_size > 1:
            self._generate_pcp_mtp_input(
                num_reqs, scheduler_output.total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens)

        long_seq_metadata = self._generate_pcp_metadata(
            total_num_scheduled_tokens)
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            # NOTE: This is strange, why did we use total_num_scheduled_tokens before?
            slot_mapping_size = (total_num_scheduled_tokens
                                 if self.pcp_size == 1 else
                                 total_num_scheduled_tokens * self.pcp_size -
                                 total_num_pcp_pads)
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                blk_table.slot_mapping.gpu[slot_mapping_size:].fill_(0)
                if self.pcp_size > 1:
                    slot_mapping_for_pcp = blk_table.slot_mapping.gpu[:
                                                                      long_seq_metadata
                                                                      .
                                                                      num_actual_tokens_pcp_padded]
                    slot_mapping_for_pcp[slot_mapping_size:].fill_(-1)
                    assert pcp_unpad_mask is not None
                    pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[:
                                                                           pcp_unpad_mask
                                                                           .
                                                                           shape[
                                                                               0]]
                    pcp_padded_slot_mapping.fill_(-1)
                    pcp_padded_slot_mapping[
                        pcp_unpad_mask] = slot_mapping_for_pcp[:
                                                               slot_mapping_size]
                    slot_mapping_for_pcp[:long_seq_metadata.
                                         num_actual_tokens_pcp_padded] = pcp_padded_slot_mapping
                    blk_table.slot_mapping.gpu[:long_seq_metadata.num_actual_tokens_pcp_padded] = \
                        slot_mapping_for_pcp
                slot_mapping = blk_table.slot_mapping.gpu

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
                and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            max_decode_tokens = self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                uniform_decode and self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens:
                num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                pad_size = num_reqs_padded - num_reqs
                if pad_size > 0:
                    last_query_loc = self.query_start_loc.np[num_reqs]

                    self.query_start_loc.np[
                        num_reqs + 1:num_reqs_padded + 1] = self.arange_np[
                            1:pad_size +
                            1] * self.uniform_decode_query_len + last_query_loc
                    self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)

                # So we are trying to simulate the behavior of GPUModelRunner's
                # prepare_inputs for uniform decode mode by padding query_start_loc
                num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                seq_lens=self.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=slot_mapping_size,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions.gpu,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                max_query_len=max_num_scheduled_tokens,
                decode_token_per_req=self.decode_token_per_req,
                prefill_context_parallel_metadata=long_seq_metadata,
            )

            if self.speculative_config and self.pcp_size > 1:
                # For pcp + spec decode, we flatten block_table
                # to avoid irregular spec_attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens = self.query_start_loc_pcp_full.cpu[1:num_reqs + 1] - \
                    self.query_start_loc_pcp_full.cpu[:num_reqs]
                num_prefill_reqs = (ori_query_lens
                                    > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = num_decode_reqs * self.decode_threshold
                blk_table_tensor[
                    num_decode_reqs_flatten:num_decode_reqs_flatten +
                    num_prefill_reqs].copy_(
                        blk_table_tensor[num_decode_reqs:num_decode_reqs +
                                         num_prefill_reqs].clone())
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(
                        self.decode_threshold, dim=0))
                common_attn_metadata.block_table_tensor = \
                    blk_table_tensor[:num_decode_reqs_flatten + num_prefill_reqs]

            if self.speculative_config and \
                self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata
                if self.speculative_config.method in ("eagle", "eagle3") and \
                        self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                    self.spec_decode_common_attn_metadata = \
                        self.spec_decode_common_attn_metadata.unpadded(
                            total_num_scheduled_tokens, base_num_reqs)

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.
                            num_decode_draft_tokens.cpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)

                    if split_ubatch_slices_for_metadata is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            split_ubatch_slices_for_metadata, common_attn_metadata,
                            self.max_num_tokens)
                        _validate_split_attn_metadata_count(
                            "decode_gdn",
                            common_attn_metadata_list,
                            len(split_ubatch_slices_for_metadata),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            attn_metadata_i = (attn_group.get_metadata_builder(
                                ubatch_id=ubid).build(
                                    common_prefix_len=common_prefix_len,
                                    common_attn_metadata=common_attn_metadata,
                                ))
                            for layer_name in kv_cache_group_spec.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=common_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            **extra_attn_metadata_args)

                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i
                elif self.model_config.runner_type == "pooling":
                    # TODO: support ubatch here
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    if split_ubatch_slices_for_metadata is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            split_ubatch_slices_for_metadata, common_attn_metadata,
                            self.max_num_tokens)
                        _validate_split_attn_metadata_count(
                            "decode_full",
                            common_attn_metadata_list,
                            len(split_ubatch_slices_for_metadata),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            attn_metadata_i = (attn_group.get_metadata_builder(
                                ubatch_id=ubid).build(
                                    common_prefix_len=common_prefix_len,
                                    common_attn_metadata=common_attn_metadata,
                                    model=self.get_model()))
                            for layer_name in kv_cache_group_spec.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=common_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            model=self.get_model(),
                            **extra_attn_metadata_args)
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        # update global cos, sin
        update_cos_sin(positions)

        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens, ubatch_slices,
                split_batch_slices, num_tokens_after_padding)

    def _generate_process_reqs_hidden_states(self, maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds):
        assert self.model is not None
        forward_context = get_forward_context()
        torch.npu.set_stream_limit(self.stream_main, cube_num=20, vector_num=20)
        self._t_replay_start = time.perf_counter()
        with torch.npu.stream(self.stream_main):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **self._init_model_kwargs(maybe_padded_num_tokens))
        # Synchronize so that the wall-clock time captured by the caller
        # reflects actual NPU completion rather than just kernel dispatch.
        self.stream_main.synchronize()

        if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not self.use_sparse
                ):
            self._update_attn_params_for_wrapper(forward_context, maybe_padded_num_tokens)


        if get_forward_context().sp_enabled and not get_forward_context(
        ).dbo_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        
        return hidden_states
    
    def _update_attn_params_for_wrapper(self, forward_context, num_tokens):
        """Update attention parameters based on the wrapper type.
        
        Handles different wrapper scenarios:
        - Split-batch: handled in execute_model
        - ACLGraphWrapper: attn_metadata is a dict, update directly
        - AscendUBatchWrapper: attn_metadata is a list, update each ubatch separately
        """
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL \
            and not self.use_sparse:
            # TODO: maybe_padded_num_tokens will be removed, use num_input_tokens instead
            if self.vllm_config.model_config.use_mla:
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_params(self.update_stream, forward_context,
                                           num_tokens,
                                           self.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               num_tokens)
                else:
                    update_attn_params(self.update_stream, forward_context,
                                       num_tokens,
                                       self.vllm_config)
    
 
    def _update_attn_params_for_split_ubatch(self, forward_context,
                                             num_tokens: int,
                                             parallel_streams: bool = False) -> None:
        if (forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or forward_context.capturing or self.use_sparse):
            return
        update_stream = self.update_stream_parallel if parallel_streams else self.update_stream_main
        if self.vllm_config.model_config.use_mla:
            if self.pcp_size * self.dcp_size > 1:
                update_mla_attn_dcp_pcp_params(update_stream,
                                               forward_context,
                                               num_tokens,
                                               in_parallel_streams=parallel_streams)
            else:
                update_mla_attn_params(update_stream, forward_context,
                                       num_tokens,
                                       self.speculative_config,
                                       in_parallel_streams=parallel_streams)
        else:
            if self.pcp_size * self.dcp_size > 1:
                update_attn_dcp_pcp_params(update_stream,
                                           forward_context,
                                           num_tokens,
                                           in_parallel_streams=parallel_streams)
            else:
                update_attn_params_split(update_stream,
                                         forward_context,
                                         num_tokens,
                                         self.vllm_config,
                                         in_parallel_streams=parallel_streams)
    def _slice_split_batch_inputs(self, tokens_slice: slice, input_ids,
                                  positions, inputs_embeds,
                                  intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[
            tokens_slice] if inputs_embeds is not None else None

        if intermediate_tensors is not None:
            if enable_sp():
                tp_size = get_tensor_model_parallel_world_size()
                start = (tokens_slice.start + tp_size - 1) // tp_size
                if start != 0:
                    stop = start + (tokens_slice.stop - tokens_slice.start +
                                    tp_size - 1) // tp_size
                else:
                    stop = (tokens_slice.stop + tp_size - 1) // tp_size
                tokens_slice = slice(start, stop)

            sliced_intermediate_tensors = intermediate_tensors[
                tokens_slice] if intermediate_tensors else None
        else:
            sliced_intermediate_tensors = None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds,
                sliced_intermediate_tensors)
    
    def _make_split_batch_metadata_parallel_streams(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode) -> list[AscendUbatchMetadata]:

        forward_contexts = []
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata

        for i, split_slice in enumerate(split_batch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_num_tokens = split_slice.num_tokens
            ubatch_num_reqs = split_slice.num_requests
            # Use dispatcher to get the correct BatchDescriptor that matches
            # what was captured at graph capture time. This ensures the key
            # used at runtime matches the key stored in concrete_aclgraph_entries
            # (or concrete_aclgraph_entries2 for parallel streams).
            # uniform_decode=True because split batch only runs on uniform decode.
            _, ubatch_batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=split_slice.padded_num_tokens,
                uniform_decode=batch_descriptor.uniform,
                has_lora=batch_descriptor.has_lora,
            )
            # For non-first splits, execute create_ascend_forward_context on
            # stream_parallel so that the GPU ops inside it (update_cos_sin,
            # clone) overlap with the stream_main work for split-0, eliminating
            # the bubble that was visible just before set_stream_limit.
            ctx_stream = self.stream_parallel if i > 0 else self.stream_main
            with torch.npu.stream(ctx_stream):
                forward_contexts.append(
                    create_ascend_forward_context(
                        cur_forward_context,
                        attn_metadata=ubatch_attn_metadata,
                        vllm_config=self.vllm_config,
                        dp_metadata=dp_metadata,
                        ubatch_slices=split_ubatch_slices,
                        batch_descriptor=ubatch_batch_descriptor,
                        cudagraph_runtime_mode=aclgraph_runtime_mode,
                        ubatch_num=i,
                        positions=positions,
                        in_parallel_streams=(i > 0),
                        cos_sin_slot_id=i,
                    ))

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                split_slice.token_slice, input_ids, positions, inputs_embeds,
                intermediate_tensors)
            
            # For non-first split, rebind to dedicated parallel buffers.
            # NOTE: input_ids and positions are already copied in _prepare_inputs
            # to avoid copy bubbles between graph replays.
            # inputs_embeds still needs copy here because it's populated after
            # _prepare_inputs returns (in the embedding layer).
            if i > 0:
                num_tokens = split_slice.num_tokens
                padded_tokens = split_slice.padded_num_tokens

                # Rebind input_ids and positions to padded parallel buffers.
                # The tail [num_tokens:padded_tokens] was already zeroed in
                # _prepare_inputs, so the graph sees a full padded-size tensor.
                if sliced_input_ids is not None:
                    sliced_input_ids = self.input_ids_parallel_streams.gpu[:padded_tokens]

                if sliced_positions is not None:
                    if sliced_positions.ndim == 2:
                        sliced_positions = self.positions_parallel_streams.gpu[:, :padded_tokens]
                    else:
                        sliced_positions = self.positions_parallel_streams.gpu[:padded_tokens]

                # Copy inputs_embeds to parallel buffer (not done in _prepare_inputs).
                # Run on stream_parallel so it overlaps with stream_main work.
                if sliced_inputs_embeds is not None:
                    with torch.npu.stream(self.stream_parallel):
                        self.inputs_embeds_parallel_streams.gpu[:num_tokens].copy_(
                            sliced_inputs_embeds, non_blocking=True)
                        if padded_tokens > num_tokens:
                            self.inputs_embeds_parallel_streams.gpu[
                                num_tokens:padded_tokens].fill_(0)
                    sliced_inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:padded_tokens]
            
            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=forward_contexts[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.padded_num_tokens))

        return ubatch_metadata

    def _make_split_batch_metadata(
            self, split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor], positions: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode) -> list[AscendUbatchMetadata]:

        forward_contexts = []
        cur_forward_context = get_forward_context()
        dp_metadata = cur_forward_context.dp_metadata

        ubatch_cudagraph_mode = CUDAGraphMode.FULL if aclgraph_runtime_mode != CUDAGraphMode.NONE else CUDAGraphMode.NONE

        for i, split_slice in enumerate(split_batch_slices):
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_num_tokens = split_slice.num_tokens
            ubatch_num_reqs = split_slice.num_requests
            # Use dispatcher to get the correct BatchDescriptor matching capture time.
            _, ubatch_batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=split_slice.padded_num_tokens,
                uniform_decode=batch_descriptor.uniform,
                has_lora=batch_descriptor.has_lora,
            )
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=split_ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=ubatch_cudagraph_mode,
                    ubatch_num=i,
                    positions=positions,
                ))

        ubatch_metadata: list[AscendUbatchMetadata] = []
        for i, split_slice in enumerate(split_batch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = self._slice_split_batch_inputs(
                split_slice.token_slice, input_ids, positions, inputs_embeds,
                intermediate_tensors)
            
            
            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=forward_contexts[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.padded_num_tokens))

        return ubatch_metadata

    def _merge_intermediate_tensors(self, intermediate_tensor_list):
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat(
                [it.tensors[key] for it in intermediate_tensor_list], dim=0)
        return IntermediateTensors(result)

    def _trim_split_output(self, output: Any, num_tokens: int) -> Any:
        """Trim a padded split output to its actual token count.

        When padding is applied to align each split to a cudagraph capture size,
        the model output has shape [padded_N, ...].  This helper slices off the
        padding rows so that _merge_split_outputs concatenates only real tokens.
        """
        if isinstance(output, torch.Tensor):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(
                t[:num_tokens] if isinstance(t, torch.Tensor) else t
                for t in output
            )
        if isinstance(output, IntermediateTensors):
            return IntermediateTensors(
                {k: v[:num_tokens] for k, v in output.tensors.items()}
            )
        # Fallback: return as-is (e.g. None or unknown type)
        return output

    def _merge_split_outputs(self, outputs: list[Any]) -> Any:
        if not outputs:
            return None
        first = outputs[0]
        if isinstance(first, IntermediateTensors):
            return self._merge_intermediate_tensors(outputs)
        if isinstance(first, tuple):
            merged = []
            for idx in range(len(first)):
                parts = [o[idx] for o in outputs]
                if isinstance(parts[0], torch.Tensor):
                    merged.append(torch.cat(parts, dim=0))
                else:
                    merged.append(parts)
            return tuple(merged)
        return torch.cat(outputs, dim=0)

    def _run_split_batch_gr0(
            self,
            split_ubatch_slices: UBatchSlices,
            split_batch_slices: SplitBatchSlices,
            attn_metadata: PerLayerAttnMetadata,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor],
            model_kwargs: dict[str, Any],
            batch_descriptor: BatchDescriptor,
            aclgraph_runtime_mode: CUDAGraphMode,
        ) -> Any:
            """
            执行split-batch，确保每个batch重放时地址一致。

            核心逻辑：
            - 图捕获时，输入地址在 self.input_ids.gpu、self.positions.gpu 等的起始位置
            - 第一个ubatch执行时，数据已经在正确位置
            - 第二个ubatch执行前，将其数据复制到起始位置，然后用起始位置执行
            - 函数结束前恢复被覆盖的起始位置数据
            """
            # Step 1: 为所有split准备元数据
            ubatch_metadata = self._make_split_batch_metadata(
                split_ubatch_slices,
                split_batch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                batch_descriptor,
                aclgraph_runtime_mode,
            )

            results: list[Any] = []
            original_forward_context = get_forward_context()

            # 获取第一个 split 的 token 数量，用于确定固定缓冲区大小
            first_split_num_tokens = split_batch_slices[0].num_tokens

            # 备份可能被覆盖的前缀区域，函数结束时恢复
            backup_input_ids = None
            if input_ids is not None:
                backup_input_ids = self.input_ids.gpu[:first_split_num_tokens].clone()

            if positions.ndim == 2:
                backup_positions = self.positions.gpu[:, :first_split_num_tokens].clone()
            else:
                backup_positions = self.positions.gpu[:first_split_num_tokens].clone()

            backup_inputs_embeds = None
            if inputs_embeds is not None:
                backup_inputs_embeds = self.inputs_embeds[:first_split_num_tokens].clone()

            base_attn_metadata = attn_metadata[0] if isinstance(
                attn_metadata, list) else attn_metadata
            base_slot_mapping = _get_slot_mapping_from_attn_metadata(
                base_attn_metadata)
            slot_mapping_backup_len = 0
            backup_slot_mapping = None
            if base_slot_mapping is not None:
                slot_mapping_backup_len = min(first_split_num_tokens,
                                              int(base_slot_mapping.shape[0]))
                if slot_mapping_backup_len > 0:
                    backup_slot_mapping = base_slot_mapping[
                        :slot_mapping_backup_len].clone()

            try:
                # Step 2: 依次执行每个split
                for slice_idx, split_slice in enumerate(split_batch_slices):
                    metadata = ubatch_metadata[slice_idx]
                    current_num_tokens = split_slice.num_tokens
                    _append_split_metadata_debug(
                        "gr0_loop_enter",
                        {
                            "slice_idx": slice_idx,
                            "token_slice": [split_slice.token_slice.start,
                                            split_slice.token_slice.stop,
                                            split_slice.token_slice.step],
                            "request_slice": [split_slice.request_slice.start,
                                              split_slice.request_slice.stop,
                                              split_slice.request_slice.step],
                            "num_tokens": current_num_tokens,
                            "context_id": _safe_context_id(metadata.context),
                            **_build_split_tensor_debug("input_ids", metadata.input_ids),
                            **_build_split_tensor_debug("positions", metadata.positions),
                        },
                    )

                    if slice_idx == 0:
                        # 第一个ubatch：数据已在正确位置（self.input_ids.gpu起始处）
                        context_attn_positions_ptr, context_attn_positions_head = _extract_attn_positions(
                            getattr(metadata.context, "attn_metadata", None)
                        )
                        with override_forward_context(metadata.context):
                            _append_split_metadata_debug(
                                "gr0_pre_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )
                            # Refresh attn params before replay to avoid carrying
                            # stale settings from the previous split.

                            result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            # [关键] 在模型执行之后更新 attention 参数
                            self._update_attn_params_for_split_ubatch(
                                metadata.context, current_num_tokens)
                            _append_split_metadata_debug(
                                "gr0_post_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )
                    else:
                        # 后续ubatch：需要将数据复制到第一个ubatch的位置
                        # [关键修复] 同步等待前一个图完成执行
                        torch.npu.synchronize()

                        # [关键修复] 将当前ubatch的数据复制到起始位置
                        # input_ids
                        if metadata.input_ids is not None:
                            self.input_ids.gpu[:current_num_tokens].copy_(
                                metadata.input_ids, non_blocking=False)

                        # positions
                        if metadata.positions is not None:
                            if metadata.positions.ndim == 2:
                                self.positions.gpu[:, :current_num_tokens].copy_(
                                    metadata.positions, non_blocking=False)
                            else:
                                self.positions.gpu[:current_num_tokens].copy_(
                                    metadata.positions, non_blocking=False)

                        # inputs_embeds
                        if metadata.inputs_embeds is not None:
                            self.inputs_embeds[:current_num_tokens].copy_(
                                metadata.inputs_embeds, non_blocking=False)

                        # [关键修复] 使用起始位置的张量执行（与图捕获时地址一致）
                        metadata.input_ids = (
                            self.input_ids.gpu[:current_num_tokens]
                            if metadata.input_ids is not None else None
                        )
                        if metadata.positions is not None:
                            if metadata.positions.ndim == 2:
                                metadata.positions = self.positions.gpu[:, :current_num_tokens]
                            else:
                                metadata.positions = self.positions.gpu[:current_num_tokens]
                        metadata.inputs_embeds = (
                            self.inputs_embeds[:current_num_tokens]
                            if metadata.inputs_embeds is not None else None
                        )
                        # Rebuild current split forward context to avoid stale context reuse.
                        ubatch_attn_metadata = None
                        if attn_metadata is not None:
                            if isinstance(attn_metadata, list):
                                _append_split_metadata_debug(
                                    "gr0_attn_metadata_select",
                                    {
                                        "slice_idx": slice_idx,
                                        "attn_metadata_type": type(attn_metadata).__name__,
                                        "attn_metadata_len": len(attn_metadata),
                                    },
                                )
                                if slice_idx >= len(attn_metadata):
                                    raise RuntimeError(
                                        "gr0 attn_metadata list too short: "
                                        f"slice_idx={slice_idx}, len={len(attn_metadata)}"
                                    )
                                ubatch_attn_metadata = attn_metadata[slice_idx]
                            else:
                                ubatch_attn_metadata = attn_metadata

                        split_slot_mapping = _get_slot_mapping_from_attn_metadata(
                            ubatch_attn_metadata)
                        if base_slot_mapping is not None and split_slot_mapping is not None:
                            copy_len = min(current_num_tokens,
                                           int(base_slot_mapping.shape[0]),
                                           int(split_slot_mapping.shape[0]))
                            if copy_len != current_num_tokens:
                                raise RuntimeError(
                                    "gr0 slot_mapping length mismatch: "
                                    f"required={current_num_tokens}, "
                                    f"base={int(base_slot_mapping.shape[0])}, "
                                    f"split={int(split_slot_mapping.shape[0])}")
                            base_slot_mapping[:copy_len].copy_(
                                split_slot_mapping[:copy_len],
                                non_blocking=False)
                            relocated_slot_mapping = base_slot_mapping[:copy_len]
                            updated_count = _set_slot_mapping_for_attn_metadata(
                                ubatch_attn_metadata, relocated_slot_mapping)
                            _append_split_metadata_debug(
                                "gr0_slot_mapping_relocated",
                                {
                                    "slice_idx": slice_idx,
                                    "copy_len": copy_len,
                                    "updated_count": updated_count,
                                    **_build_split_tensor_debug(
                                        "relocated_slot_mapping",
                                        relocated_slot_mapping),
                                },
                            )

                        ubatch_batch_descriptor = BatchDescriptor(
                            num_tokens=current_num_tokens,
                            num_reqs=split_slice.num_requests,
                            uniform=batch_descriptor.uniform,
                            has_lora=batch_descriptor.has_lora,
                        )
                        ubatch_cudagraph_mode = (
                            CUDAGraphMode.FULL
                            if aclgraph_runtime_mode != CUDAGraphMode.NONE
                            else CUDAGraphMode.NONE
                        )
                        if _SPLIT_LOCAL_CONTEXT_REBUILD:
                            rebuild_ubatch_slices = [
                                UBatchSlice(
                                    slice(0, split_slice.num_requests),
                                    slice(0, current_num_tokens),
                                )
                            ]
                            rebuild_positions = metadata.positions
                            rebuild_ubatch_num = 0
                        else:
                            rebuild_ubatch_slices = split_ubatch_slices
                            rebuild_positions = positions
                            rebuild_ubatch_num = slice_idx

                        metadata.context = create_ascend_forward_context(
                            original_forward_context,
                            attn_metadata=ubatch_attn_metadata,
                            vllm_config=self.vllm_config,
                            dp_metadata=original_forward_context.dp_metadata,
                            ubatch_slices=rebuild_ubatch_slices,
                            batch_descriptor=ubatch_batch_descriptor,
                            cudagraph_runtime_mode=ubatch_cudagraph_mode,
                            ubatch_num=rebuild_ubatch_num,
                            positions=rebuild_positions,
                        )
                        torch.npu.synchronize()


                        with override_forward_context(metadata.context):
                            # Refresh attn params before replay to avoid carrying
                            # stale settings from the previous split.
                            result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            self._update_attn_params_for_split_ubatch(
                                metadata.context, current_num_tokens)
                            _append_split_metadata_debug(
                                "gr0_post_model",
                                {
                                    "slice_idx": slice_idx,
                                    "context_id": _safe_context_id(metadata.context),
                                    "num_tokens": current_num_tokens,
                                },
                            )

                    results.append(self._trim_split_output(
                        result, split_batch_slices[slice_idx].num_tokens))

                with override_forward_context(original_forward_context):
                    result = self._merge_split_outputs(results)

                if not getattr(self, "_split_batch_dumped", False):
                    dump_path = os.path.join(
                        os.getcwd(), "split_batch_merged_first_result_gg.json")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(self._to_jsonable(result), f, ensure_ascii=False)
                    self._split_batch_dumped = True

                return result

            finally:
                if backup_input_ids is not None:
                    self.input_ids.gpu[:first_split_num_tokens].copy_(
                        backup_input_ids, non_blocking=False)

                if positions.ndim == 2:
                    self.positions.gpu[:, :first_split_num_tokens].copy_(
                        backup_positions, non_blocking=False)
                else:
                    self.positions.gpu[:first_split_num_tokens].copy_(
                        backup_positions, non_blocking=False)

                if backup_inputs_embeds is not None:
                    self.inputs_embeds[:first_split_num_tokens].copy_(
                        backup_inputs_embeds, non_blocking=False)
                if backup_slot_mapping is not None and base_slot_mapping is not None:
                    base_slot_mapping[:slot_mapping_backup_len].copy_(
                        backup_slot_mapping, non_blocking=False)
                torch.npu.synchronize()


    def _run_split_batch_parallel(
        self,
        split_ubatch_slices: UBatchSlices,
        split_batch_slices: SplitBatchSlices,
        attn_metadata: PerLayerAttnMetadata,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        model_kwargs: dict[str, Any],
        batch_descriptor: BatchDescriptor,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> Any:
        """
        执行split-batch，确保每个batch重放时地址不一致。
        
        核心逻辑：
        - 图捕获时，输入地址在 self.input_ids.gpu、self.positions.gpu 等的起始位置
        - 第一个ubatch执行时，数据已经在正确位置
        - 第二个ubatch执行前，将其数据复制到起始位置，然后用起始位置执行
        """
        return self._run_split_batch_parallel_impl(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            model_kwargs,
            batch_descriptor,
            aclgraph_runtime_mode,
        )
    
    def _run_split_batch_parallel_impl(
        self,
        split_ubatch_slices: UBatchSlices,
        split_batch_slices: SplitBatchSlices,
        attn_metadata: PerLayerAttnMetadata,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        model_kwargs: dict[str, Any],
        batch_descriptor: BatchDescriptor,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> Any:
        # Step 1: 为所有split准备元数据
        ubatch_metadata = self._make_split_batch_metadata_parallel_streams(
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            batch_descriptor,
            aclgraph_runtime_mode,
        )
        
        results: list[Any] = []
        original_forward_context = get_forward_context()
        torch.npu.set_stream_limit(self.stream_main, cube_num=15, vector_num=20)
        torch.npu.set_stream_limit(self.stream_parallel, cube_num=15, vector_num=20)
        num_splits = len(split_batch_slices)
        results: list[Optional[Any]] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()

        self._t_replay_start = time.perf_counter()
        def _run_split_replay_worker(slice_idx: int) -> None:
            try:
                split_slice = split_batch_slices[slice_idx]
                metadata = ubatch_metadata[slice_idx]
                current_num_tokens = split_slice.num_tokens
                # attn_params are keyed by the padded graph size, not the
                # actual token count.  Use padded_num_tokens as runtime_shape.
                current_padded_num_tokens = split_slice.padded_num_tokens
                parallel_streams = slice_idx > 0
                target_stream = self.stream_parallel if parallel_streams else self.stream_main
                with torch.inference_mode():
                    # if parallel_streams:
                    #     split_attn_metadata = attn_metadata
                    #     if isinstance(attn_metadata, list):
                    #         split_attn_metadata = attn_metadata[slice_idx]

                    #     split_slot_mapping = _get_slot_mapping_from_attn_metadata(
                    #         split_attn_metadata)
                    #     if split_slot_mapping is not None:
                    #         copy_len = min(current_num_tokens,
                    #                         int(split_slot_mapping.shape[0]))
                    #         relocated_slot_mapping = split_slot_mapping[:copy_len]
                    #         updated_count = _set_slot_mapping_for_attn_metadata(
                    #             split_attn_metadata, relocated_slost_mapping)

                    #     ubatch_batch_descriptor = BatchDescriptor(
                    #         num_tokens=current_num_tokens,
                    #         num_reqs=split_slice.num_requests,
                    #         uniform=batch_descriptor.uniform,
                    #         has_lora=batch_descriptor.has_lora,
                    #     )
                    #     if aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                    #         ubatch_batch_descriptor = (
                    #             ubatch_batch_descriptor
                    #             .relax_for_mixed_batch_cudagraphs())
                    #     if _SPLIT_LOCAL_CONTEXT_REBUILD:
                    #         rebuild_ubatch_slices = [
                    #             UBatchSlice(
                    #                 slice(0, split_slice.num_requests),
                    #                 slice(0, current_num_tokens),
                    #             )
                    #         ]
                    #         rebuild_positions = metadata.positions
                    #         rebuild_ubatch_num = 0
                    #     else:
                    #         rebuild_ubatch_slices = split_ubatch_slices
                    #         rebuild_positions = positions
                    #         rebuild_ubatch_num = slice_idx

                    #     metadata.context = create_ascend_forward_context(
                    #         metadata.context,
                    #         attn_metadata=split_attn_metadata,
                    #         vllm_config=self.vllm_config,
                    #         dp_metadata=original_forward_context.dp_metadata,
                    #         ubatch_slices=rebuild_ubatch_slices,
                    #         batch_descriptor=ubatch_batch_descriptor,
                    #         cudagraph_runtime_mode=aclgraph_runtime_mode,
                    #         ubatch_num=rebuild_ubatch_num,
                    #         positions=rebuild_positions,
                    #         in_parallel_streams=True,
                    #         cos_sin_slot_id=slice_idx,
                    #     )

                    with torch.npu.stream(target_stream):
                        with override_forward_context(metadata.context):
                            split_result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )

                            if aclgraph_runtime_mode == CUDAGraphMode.FULL:
                                self._update_attn_params_for_split_ubatch(
                                    metadata.context,
                                    current_padded_num_tokens,
                                    parallel_streams=parallel_streams)
                            results[slice_idx] = self._trim_split_output(
                                split_result,
                                split_batch_slices[slice_idx].num_tokens)
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        split_workers: list[threading.Thread] = []
        for slice_idx in range(num_splits):
            worker = threading.Thread(target=_run_split_replay_worker,
                                        args=(slice_idx,),
                                        name=f"split-replay-{slice_idx}")
            split_workers.append(worker)
            worker.start()

        for worker in split_workers:
            worker.join()

        if split_errors:
            split_errors.sort(key=lambda item: item[0])
            failed_slice_idx, first_error = split_errors[0]
            raise RuntimeError(
                f"split replay worker failed at slice_idx={failed_slice_idx}"
            ) from first_error

        # Wait per stream instead of using a device-wide barrier to preserve
        # overlap between split-0(main) and split-1(parallel) replay.
        logger.debug("[split_batch] synchronizing stream_main")
        self.stream_main.synchronize()
        logger.debug("[split_batch] stream_main synchronized")
        if len(split_batch_slices) > 1:
            logger.debug("[split_batch] synchronizing stream_parallel")
            self.stream_parallel.synchronize()
            logger.debug("[split_batch] stream_parallel synchronized")
        merged_results: list[Any] = [result for result in results
                                        if result is not None]

        logger.debug("[split_batch] merging %d split outputs", len(merged_results))
        with override_forward_context(original_forward_context):
            result = self._merge_split_outputs(merged_results)
        logger.debug("[split_batch] merge done, returning result")

        if not getattr(self, "_split_batch_dumped", False):
            dump_path = os.path.join(
                os.getcwd(), "split_batch_merged_first_result_gg.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(self._to_jsonable(result), f, ensure_ascii=False)
            self._split_batch_dumped = True

        return result

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        if np.array_equal(self.seq_lens.np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == 'mtp':
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(
                cu_num_scheduled_tokens - num_sampled_tokens,
                num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(
                logits_indices_pcp).pin_memory().to(self.device,
                                                    non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = (
            torch.from_numpy(cu_num_draft_tokens).pin_memory().to(
                self.device, non_blocking=True))
        cu_num_sampled_tokens = (
            torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(
                self.device, non_blocking=True))
        logits_indices = (torch.from_numpy(logits_indices).pin_memory().to(
            self.device, non_blocking=True))
        target_logits_indices = (
            torch.from_numpy(target_logits_indices).pin_memory().to(
                self.device, non_blocking=True))
        bonus_logits_indices = torch.from_numpy(
            bonus_logits_indices).pin_memory().to(self.device,
                                                  non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: PerLayerAttnMetadata,
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            # TODO: attn_metadata is only used in torchair generate_token_ids, check it
            draft_token_ids = self.drafter.generate_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, aux_hidden_states)
        return draft_token_ids

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors] | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        # Record the header-overhead start time: everything from execute_model
        # entry up to this point (prepare_inputs, metadata, etc.) is "header".
        self._t_header_start = time.perf_counter()
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors, max_query_len, ubatch_slices,
             split_batch_slices, num_tokens_after_padding) = (
                 self._prepare_inputs(scheduler_output, intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        need_dump = self.dump_enable and self.debugger is not None
        if need_dump:
            assert self.debugger is not None
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(
                getattr(dbg_cfg, "level",
                        "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        if self.ascend_config.enable_async_exponential != 0:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators)

        split_ubatch_slices = None
        if split_batch_slices is not None:
            split_ubatch_slices = [
                UBatchSlice(s.request_slice, s.token_slice)
                for s in split_batch_slices
            ]

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens
            num_tokens_across_dp = num_tokens_after_padding

        model_kwargs = self._init_model_kwargs(maybe_padded_num_tokens)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        split_enable_parallel_streams = bool(split_cfg is not None
                                 and getattr(split_cfg, "enable_parallel_streams", False))
        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=self.with_prefill,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method,
                    ubatch_slices=(ubatch_slices or split_ubatch_slices),
            ):
                self.maybe_setup_kv_connector(scheduler_output)

                if split_ubatch_slices is not None:
                    if split_enable_parallel_streams:
                        #logger.info("Running split batch with parallel streams, split_cfg=%s", split_cfg)
                        hidden_states = self._run_split_batch_parallel(
                            split_ubatch_slices,
                            split_batch_slices,
                            attn_metadata,
                            input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            model_kwargs,
                            batch_descriptor,
                            aclgraph_runtime_mode)
                        self._t_replay_end = time.perf_counter()
                    else:
                        #logger.info("Running split batch without parallel streams, split_cfg=%s", split_cfg)
                        self._t_replay_start = time.perf_counter()
                        hidden_states = self._run_split_batch_gr0(
                        split_ubatch_slices,
                        split_batch_slices,
                        attn_metadata,
                        input_ids,
                        positions,
                        intermediate_tensors,
                        inputs_embeds,
                        model_kwargs,
                        batch_descriptor,
                        aclgraph_runtime_mode,
                    )
                else:
                    #logger.info("Running without split batch")
                    hidden_states = self._generate_process_reqs_hidden_states(
                        maybe_padded_num_tokens, input_ids, positions,
                        intermediate_tensors, inputs_embeds)
                    self._t_replay_end = time.perf_counter()
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            # Accumulate perf stats for TPOT / header-overhead tracking.
            # Only count uniform-decode steps (1 output token per request).
            if uniform_decode:
                _header_ms = (self._t_replay_start - self._t_header_start) * 1000.0
                _replay_ms = (self._t_replay_end - self._t_replay_start) * 1000.0
                _n_tokens = self.input_batch.num_reqs
                self._last_step_perf = {
                    "header_ms": _header_ms,
                    "replay_ms": _replay_ms,
                    "batch_size": _n_tokens,
                    "is_split": split_ubatch_slices is not None,
                }
                self._perf_accum["total_header_ms"] += _header_ms
                self._perf_accum["total_replay_ms"] += _replay_ms
                self._perf_accum["total_output_tokens"] += _n_tokens
                self._perf_accum["num_decode_steps"] += 1
                _write_perf_stats(self._last_step_perf)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)   
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    pool_output = self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np)
                    if need_dump:
                        assert self.debugger is not None
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                # Sometimes, after the model is compiled through the AOT backend,
                # the model output may become a list containing only one Tensor object.
                if isinstance(hidden_states, list) and \
                        len(hidden_states) == 1 and \
                        isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        need_dump = self.dump_enable and self.debugger is not None
        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with ProfileExecuteDuration().capture_async("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.use_eagle() and \
                    not self.speculative_config.disable_padded_drafter_batch
                if use_padded_batch_for_eagle:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch_for_eagle:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({"kv_connector_output": kv_connector_output})

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            if need_dump:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if need_dump:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[:len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        if self.need_accepted_tokens:  # TODO remove this if
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids)
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
            LogprobsLists | None,
            list[list[int]],
            dict[str, LogprobsTensors | None],
            list[str],
            dict[str, int],
            list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    return_cu_num_tokens=logprobs_tensors is not None,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [
                    -1
                ] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (logprobs_tensors.tolists(cu_num_tokens)
                          if not self.use_async_scheduling
                          and logprobs_tensors is not None else None)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    def _build_dummy_attn_metadata(
        self,
        with_prefill: bool,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        num_scheduled_tokens: np.ndarray,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        ubatch_slices=None,
    ) -> Optional[PerLayerAttnMetadata]:

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        if force_attention or aclgraph_runtime_mode == CUDAGraphMode.FULL:
            assert with_prefill is False, \
                "Full decode graph only supports uniform batch now."

            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cu_num_tokens, arange = self._get_cumsum_and_arange(
                num_scheduled_tokens)

            self.query_start_loc.cpu[1:num_reqs +
                                     1] = torch.Tensor(cu_num_tokens)
            self.query_lens = torch.from_numpy(num_scheduled_tokens)
            self.attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask()

            num_computed_tokens_cpu = (
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                block_table_tensor = self.input_batch.block_table[
                    kv_cache_group_id].get_device_tensor()
                slot_mapping = self.input_batch.block_table[
                    kv_cache_group_id].slot_mapping
                self.cp_kv_recover_idx = torch.zeros(self.max_num_tokens,
                                                     dtype=torch.int32,
                                                     device=self.device)
                long_seq_metadata = self._generate_pcp_metadata(num_tokens)
                if long_seq_metadata is not None:
                    pcp_world_size = get_pcp_group().world_size
                    dcp_world_size = get_dcp_group().world_size
                    num_computed_tokens_of_pcp_dcp = [[
                        [0] * dcp_world_size for _ in range(pcp_world_size)
                    ] for _ in range(num_tokens)]
                    long_seq_metadata.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp
                # QUESTION: Why do we separately set query_start_loc for spec in the first place?
                # While in _prepare_inputs we don't?
                if self.speculative_config:
                    self.query_start_loc.gpu[:num_reqs + 1] = torch.tensor(
                        [0] + self.actual_seq_lengths_q[:num_reqs],
                        device=self.device,
                        dtype=torch.int32)
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens_cpu=self.seq_lens.cpu,
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    actual_seq_lengths_q=self.actual_seq_lengths_q,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    positions=self.positions.gpu,
                    attn_mask=self.attn_mask,
                    spec_attn_mask=self.spec_attn_mask,
                    attn_state=self.attn_state,
                    max_query_len=max_query_len,
                    decode_token_per_req=self.decode_token_per_req,
                    prefill_context_parallel_metadata=long_seq_metadata,
                )
                if self.pcp_size > 1:
                    common_attn_metadata.block_table_tensor = \
                        block_table_tensor[:num_reqs * self.decode_threshold]
                attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and \
                        self.speculative_config.method == "mtp":
                    attn_state = AscendAttentionState.SpecDecoding

                common_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    seq_lens=self.seq_lens.cpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    _num_computed_tokens_cpu=num_computed_tokens_cpu,
                    max_query_len=max_query_len,
                    max_seq_len=seq_lens)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    builder = attn_group.get_metadata_builder()
                    if ubatch_slices is not None:
                        # TODO: check dummy attn construct logic
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata,
                            self.max_num_tokens)
                        _validate_split_attn_metadata_count(
                            "dummy_capture",
                            common_attn_metadata_list,
                            len(ubatch_slices),
                        )
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = (attn_group\
                                               .get_metadata_builder(ubatch_id=ubid)\
                                               .build_for_cudagraph_capture(common_attn_metadata, attn_state, self.get_model()))
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        if isinstance(builder, GDNAttentionMetadataBuilder):
                            attn_metadata_gdn_attention = builder.build_for_cudagraph_capture(
                                common_metadata)
                        else:
                            attn_metadata_full_attention = builder.build_for_graph_capture(
                                common_attn_metadata, attn_state,
                                self.get_model())
                        for layer_name in kv_cache_group_spec.layer_names:
                            if "linear_attn" in layer_name:
                                attn_metadata[
                                    layer_name] = attn_metadata_gdn_attention
                            else:
                                attn_metadata[
                                    layer_name] = attn_metadata_full_attention

        return attn_metadata

    def _generate_dummy_run_hidden_states(self, input_ids, positions,
                                          num_tokens, intermediate_tensors,
                                          inputs_embeds):
        hidden_states = self.model(input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            )
        forward_context = get_forward_context()
        assert forward_context is not None
        model_updates_attn_params_internally = bool(
            getattr(self.model, "updates_attn_params_internally", False))
        if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not forward_context.capturing and not self.use_sparse
                and not model_updates_attn_params_internally):
            if self.vllm_config.model_config.use_mla:
                # FIXME: Try using `auto_dispatch_capture=True`
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    # When using npu_split_wrapper, attn_metadata is a list (one per ubatch)
                    if isinstance(forward_context.attn_metadata, list):
                        for ubatch_attn_metadata in forward_context.attn_metadata:
                            temp_context = type('obj', (object,), {
                                'attn_metadata': ubatch_attn_metadata,
                                'is_mtp_model': forward_context.is_mtp_model,
                                'capturing': forward_context.capturing,
                                'is_mla_model': forward_context.is_mla_model
                            })
                            update_mla_attn_params(self.update_stream, temp_context,
                                                   num_tokens,
                                                   self.vllm_config.speculative_config)
                    else:
                        update_mla_attn_params(self.update_stream, forward_context,
                                               num_tokens,
                                               self.vllm_config.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               num_tokens)
                else:
                    # When using npu_split_wrapper, attn_metadata is a list (one per ubatch)
                    if isinstance(forward_context.attn_metadata, list):
                        for ubatch_attn_metadata in forward_context.attn_metadata:
                            temp_context = type('obj', (object,), {
                                'attn_metadata': ubatch_attn_metadata,
                                'is_mtp_model': forward_context.is_mtp_model,
                                'capturing': forward_context.capturing,
                                'is_mla_model': forward_context.is_mla_model
                            })
                            update_attn_params(self.update_stream, temp_context,
                                               num_tokens,
                                               self.vllm_config)
                    else:
                        update_attn_params(self.update_stream, forward_context,
                                           num_tokens,
                                           self.vllm_config)


        if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        allow_microbatching: bool = True,
        in_parallel_streams: bool=False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.seperate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        # TODO: create_mixed_batch should be Fasle in Ascend now
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_reqs = cdiv(num_tokens, 1)
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        # TODO: create_mixed_batch should be Fasle in Ascend now
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req -
                            1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        # dbo
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())
        ubatch_slices = None

        moe_comm_type = select_moe_comm_method(num_tokens, self.vllm_config)
        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, _ = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
            )
         # Split batch - compute split slices for large decode batches (similar to _prepare_inputs)
        # Split batch and DBO never conflict by design
        if uniform_decode and ubatch_slices is None :  # Only split if DBO is not active
            cudagraph_capture_sizes = set(
                self.compilation_config.cudagraph_capture_sizes or []
            ) if self.use_aclgraph else None
            ubatch_slices, _ = split_batch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                vllm_config=self.vllm_config,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            )

        # Padding for DP
        # currently, we check the dp scenario that some ranks have tokens
        # but others execute dummy run
        if ubatch_slices is not None:
            enable_dbo = True
        else:
            enable_dbo = False

        (num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_tokens, with_prefill,
                                                     enable_dbo)
        moe_comm_type = select_moe_comm_method(num_tokens, self.vllm_config)
        if not enable_dbo:
            ubatch_slices = None
            

       

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = (batch_descriptor.num_reqs if
                           batch_descriptor.num_reqs is not None else num_reqs)
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # filter out the valid batch descriptor
        if aclgraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if aclgraph_runtime_mode != CUDAGraphMode.NONE and aclgraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}.")
        else:
            aclgraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            force_attention=force_attention,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens,
                                            num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model:
                input_ids = None
                if in_parallel_streams:
                    inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:num_tokens_padded]
                else:
                    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                if in_parallel_streams:
                    inputs_embeds = self.inputs_embeds_parallel_streams.gpu[:num_tokens_padded]
                else:
                    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                if in_parallel_streams:
                    input_ids = self.input_ids_parallel_streams.gpu[:num_tokens_padded]
                else:
                    input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            else:
                if in_parallel_streams:
                    positions = self.positions_parallel_streams.gpu[:num_tokens_padded]
                else:
                    positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=actual_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k:
                    v[:num_tokens_padded]
                    for k, v in self.intermediate_tensors.items()
                })

            need_dummy_logits = (not is_profile and lmhead_tp_enable())
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                        dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(
                        self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(
                        hidden_states[dummy_indices])

            with set_ascend_forward_context(
                    # When capturing for the parallel stream, clone block_tables
                    # in every per-layer AscendMetadata so that
                    # _graph_params_parallel binds a *different* device address
                    # than _graph_params.  Without this, both graph params point
                    # to the same block_table storage, and the two concurrent
                    # _refresh_block_table_in_place calls at runtime both write
                    # to block_table[:8, :], causing a data race that corrupts
                    # the KV-cache lookup for split-0.
                    _clone_attn_metadata_block_tables(attn_metadata)
                    if in_parallel_streams else attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=is_profile,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method,
                    in_parallel_streams=in_parallel_streams,
                    ubatch_slices=ubatch_slices,):
                with torch.npu.stream(self.stream_parallel if in_parallel_streams else self.stream_main):
                    hidden_states = self._generate_dummy_run_hidden_states(
                        input_ids, positions, num_tokens_padded,
                        intermediate_tensors, inputs_embeds)
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile)
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[
            -1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        # Sometimes, after the model is compiled through the AOT backend,
        # the model output may become a list containing only one Tensor object.
        if isinstance(hidden_states, list) and \
            len(hidden_states) == 1 and \
            isinstance(hidden_states[0], torch.Tensor):
            hidden_states = hidden_states[0]
            hidden_states = hidden_states[logit_indices]
            output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and \
            select_moe_comm_method(mc2_tokens_capacity, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity,
                            with_prefill=True,
                            is_profile=True)
        super().profile_run()

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.dynamic_eplb:
                model_register(self.model, self.model_config)
            if self.drafter:
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
                if self.drafter.name == SpecDcodeType.EAGLE3:
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # Wrap model with the correct runtime wrapper.
        # Priority (mutually exclusive at runtime):
        # - DBO: AscendUBatchWrapper
        # - Split-batch: handled in execute_model
        # - Full graph only: ACLGraphWrapper
        split_enabled = bool(
            getattr(self.ascend_config, "split_batch_config",
                    None) is not None
            and self.ascend_config.split_batch_config.enabled)

        if self.parallel_config.enable_dbo:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.model = AscendUBatchWrapper(self.model, self.vllm_config,
                                                 CUDAGraphMode.FULL,
                                                 self.device)
            else:
                self.model = AscendUBatchWrapper(self.model, self.vllm_config,
                                                 CUDAGraphMode.NONE,
                                                 self.device)
        elif self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.update_stream_main: torch.npu.Stream = torch.npu.Stream()
            self.update_stream_parallel: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL,
                                         device=self.device)


    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = (len(self.attn_groups) > 1)
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any([
            isinstance(attn_group[0].kv_cache_spec, MambaSpec)
            for attn_group in self.attn_groups
        ])

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def _align_memory(self, tensor: torch.Tensor,
                      alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        from vllm.v1.worker.utils import bind_kv_cache
        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cahce and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
         """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                              Optional[torch.Tensor]]] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # for mamba linear attention
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                             dtype=torch.int8,
                                             device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned,
                                             dtype=torch.int8,
                                             device=self.device)
                        tensor = self._align_memory(
                            tensor, alignment)[:kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    if self.model_config.use_mla:
                        head_size = self.model_config.hf_text_config.qk_rope_head_dim + \
                            self.model_config.hf_text_config.kv_lora_rank

                    dsa_k_cache_factor = None
                    dsa_k_cache_size = None
                    if not self.model_config.use_mla:
                        # for non-mla model, use FullAttentionSpec
                        k_tensor_split_factor = 2
                        v_tensor_split_factor = 2
                    elif self.use_sparse:
                        # for deepseek v3.2, DSA use FullAttentionSpec
                        # FullAttentionSpec allocate 2 * mla page size bytes,
                        # and we use half of that for k cache in DSA
                        dsa_k_cache_factor = 2
                        k_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.qk_rope_head_dim
                        dsa_k_cache_size = int(kv_cache_tensor.size //
                                               dsa_k_cache_factor)
                    else:
                        # for other deepseek models, use MLAAttentionSpec
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim

                    k_tensor_size = int(kv_cache_tensor.size //
                                        k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size //
                                        v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size,
                                dtype=torch.int8,
                                device=self.device)
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        k_tensor = self._align_memory(
                            k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(
                            v_tensor, alignment)[:v_tensor_size]
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None and dsa_k_cache_size is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size + alignment,
                                dtype=torch.int8,
                                device=self.device)
                            dsa_k_cache_tensor = self._align_memory(
                                dsa_k_cache_tensor,
                                alignment)[:dsa_k_cache_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if ("attn" in layer_name_inner
                                and "linear_attn" not in layer_name_inner):
                            kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor) if \
                                not self.use_sparse else (k_tensor, v_tensor, dsa_k_cache_tensor)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    raw_dsa_k_tensor = None
                    if self.use_sparse:
                        raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        assert raw_dsa_k_tensor is not None
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_block_size"
                               ) and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_block_size()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk, block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if not self.model_config.use_mla:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank
                        ]
                        v_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim
                        ]
                    k_cache = raw_k_tensor.view(dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(dtype).view(v_shape)
                    if get_ascend_device_type() == AscendDeviceType._310P:
                        k_cache = maybe_trans_nz(k_cache)
                        v_cache = maybe_trans_nz(v_cache)
                    if self.use_sparse and raw_dsa_k_tensor is not None:
                        dsa_k_cache_shape = (num_blocks,
                                             kv_cache_spec.block_size, 1, 128)
                        dsa_k_cache_size = (
                            num_blocks
                        ) * kv_cache_spec.block_size * 128 * dtype.itemsize
                        dsa_k_cache = raw_dsa_k_tensor[:dsa_k_cache_size].view(
                            dtype).view(dsa_k_cache_shape)
                        kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel(
                    ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel(
                    ) // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(
                kv_cache_config.kv_cache_groups):

            if isinstance(kv_cache_group.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                if attn_groups and self.use_hybrid_blocks:
                    # Use the backend's supported block size list
                    backend = attn_groups[0].backend
                    supported_sizes = backend.get_supported_block_size()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = (supported_sizes
                                              if supported_sizes else
                                              [self.cache_config.block_size])
                else:
                    # Fallback to cache config block_size if no backend found
                    kernel_block_size_list = [self.cache_config.block_size]
                kernel_block_sizes.append(kernel_block_size_list)
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                kernel_block_sizes.append([0])
        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.model_config.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
                kernel_block_sizes=kernel_block_sizes,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> dict[AttentionGroupKey, list[str]]:
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
                kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return {
                attn_backends[k]: v
                for k, v in attn_backend_layers.items()
            }
        def _get_num_attn_metadata_builders() -> int:
            """How many metadata builders we need per AttentionGroup.

            - DBO(ubatch): 2 builders (ubatch_id=0/1).
            - Split-batch: N builders, N = split_config.num_splits.
            - Default: 1 builder.
            """
            if self.parallel_config.enable_dbo:
                return 2

            split_cfg = getattr(self.ascend_config, "split_batch_config", None)
            split_enabled = bool(split_cfg is not None
                                 and getattr(split_cfg, "enabled", False))
            if split_enabled:
                n = int(getattr(split_cfg, "num_splits", 1))
                return max(1, n)

            return 1

        def create_attn_groups(attn_backends_map: dict[AttentionBackend,
                                                       list[str]],
                               kv_cache_group_id: int) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = [
                    attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        layer_names,
                        self.vllm_config,
                        self.device,
                    ) for _ in range(_get_num_attn_metadata_builders())
                ]
                attn_group = AttentionGroup(attn_backend, layer_names,
                                            kv_cache_spec, kv_cache_group_id,
                                            attn_metadata_builders)
                attn_groups.append(attn_group)
            return attn_groups

        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec)
            self.attn_groups.append(create_attn_groups(attn_backends, i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i,
                       "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = (
                    attn_metadata_builder_i.reorder_batch_threshold)
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != \
                            self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}")
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                  AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                # TODO: Support other attention modules, e.g., cross-attention
                # TODO(lucas): move the attention specs into the model layers like
                # the attention backends
                if attn_module.attn_type == AttentionType.DECODER:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                elif attn_module.attn_type in (AttentionType.ENCODER,
                                               AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")

            elif isinstance(attn_module, MLAAttention):
                if use_mla and not self.use_sparse:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.cache_config.cache_dtype)
                else:
                    # TODO(cmq): This is a hack way to fix deepseek kvcache when
                    # using DSA. Fix the spec in vLLM is a finnal way.
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if (self.vllm_config.speculative_config is not None
                    and self.vllm_config.model_config.hf_config.model_type
                    not in ["qwen3_next"]):
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type,
                    num_speculative_blocks=(
                        self.speculative_config.num_speculative_tokens
                        if self.speculative_config else 0),
                )

        return kv_cache_spec

    def initialize_aclgraph_capture(self) -> None:
        min_ag_support = AttentionCGSupport.ALWAYS
        min_ag_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.get_metadata_builder()
            graph_support = None
            if hasattr(builder, 'aclgraph_support'):
                graph_support = builder.aclgraph_support.value
                builder_aclgraph = builder.aclgraph_support
            else:
                graph_support = builder._cudagraph_support.value
                builder_aclgraph = builder._cudagraph_support
            if graph_support < min_ag_support.value:
                min_ag_support = builder_aclgraph
                min_ag_builder_name = builder.__class__.__name__

        # This is an imitation of compilation_config.splitting_ops_contain_attention()
        splitting_ops_contain_attention = (
            self.compilation_config.splitting_ops is not None
            and all(op in self.compilation_config.splitting_ops for op in [
                "vllm.mla_forward",
            ]))

        # Flexible resolve the aclgraph mode
        aclgraph_mode = self.compilation_config.cudagraph_mode
        # check graph for mixed batch is supported
        if aclgraph_mode.mixed_mode() == CUDAGraphMode.FULL \
            and min_ag_support != AttentionCGSupport.ALWAYS:
            msg = (f"ACLGraphMode.{aclgraph_mode.name} is not supported "
                   f"with {min_ag_builder_name} backend (support: "
                   f"{min_ag_support})")
            if min_ag_support == AttentionCGSupport.NEVER:
                # if not supported any full graphs, just raise it.
                msg += "; please try cudagraph_mode=PIECEWISE, and "\
                    "make sure compilation level is piecewise"
                raise ValueError(msg)

            # attempt to resolve the full graph related mode
            if splitting_ops_contain_attention:
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                aclgraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE)
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                aclgraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY)
            logger.warning(msg)

        # double check that we can support full graph if they are requested
        # even after automatic downgrades
        if aclgraph_mode.has_full_cudagraphs() \
            and min_ag_support == AttentionCGSupport.NEVER:
            raise ValueError(f"CUDAGraphMode.{aclgraph_mode.name} is not "
                             f"supported with {min_ag_builder_name} backend ("
                             f"support:{min_ag_support}) "
                             "; please try cudagraph_mode=PIECEWISE, "
                             "and make sure compilation level is piecewise")

        if (aclgraph_mode.decode_mode() == CUDAGraphMode.FULL
                and aclgraph_mode.separate_routine()
                and self.uniform_decode_query_len > 1):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len,
                self.parallel_config.tensor_parallel_size)
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            self.cudagraph_batch_sizes = (capture_sizes
                                          if capture_sizes is not None else [])

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        set_graph_params(self.cudagraph_batch_sizes)
        # Parallel stream needs its own GraphParams to avoid races when both
        # streams call graph_task_update_begin/end concurrently.
        # If split_batch_config.parallel_capture_sizes is set, use those sizes
        # for the parallel-stream graph pool; otherwise fall back to the main
        # capture sizes (legacy behaviour).
        _split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        _parallel_sizes = (
            _split_cfg.parallel_capture_sizes
            if _split_cfg is not None
            and _split_cfg.parallel_capture_sizes is not None
            else self.cudagraph_batch_sizes
        )
        self.cudagraph_batch_sizes_parallel = _parallel_sizes
        set_graph_params_parallel(self.cudagraph_batch_sizes_parallel)
        if self.speculative_config:
            set_mtp_graph_params(self.cudagraph_batch_sizes)

        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

    def _capture_aclgraphs(self, compilation_cases: list[int],
                           aclgraph_runtime_mode: CUDAGraphMode,
                           uniform_decode: bool,
                           in_parallel_streams: bool = False
                           ):
        assert aclgraph_runtime_mode != CUDAGraphMode.NONE and \
            aclgraph_runtime_mode in [CUDAGraphMode.FULL,
                                      CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            logger.info(
                "Starting to capture ACL graphs for cases: %s, "
                "mode: %s, uniform_decode: %s", compilation_cases,
                aclgraph_runtime_mode.name, uniform_decode)
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing ACL graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    aclgraph_runtime_mode.name))

        force_attention = (aclgraph_runtime_mode == CUDAGraphMode.FULL)
        # When the kv cache spec is empty, PiecewiseBackend is not initialized, and
        # compilation_case=1 will cause the dynamic shape position to be incorrectly derived.
        if not self.get_kv_cache_spec():
            self._dummy_run(2,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode,
                            in_parallel_streams=in_parallel_streams,)
        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:

            allow_microbatching = self.parallel_config.enable_dbo \
                and aclgraph_runtime_mode == CUDAGraphMode.FULL \
                and uniform_decode \
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                self._dummy_run(num_tokens,
                                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode,
                                allow_microbatching=allow_microbatching,
                                in_parallel_streams=in_parallel_streams,
                                # allow_parallel_streams=False
                                )
            #真正捕获的运行
            self._dummy_run(num_tokens,
                            aclgraph_runtime_mode=aclgraph_runtime_mode,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode,
                            allow_microbatching=allow_microbatching,
                            in_parallel_streams=in_parallel_streams,
                            )

    def _capture_model(self):
        if not self.use_aclgraph:
            logger.warning(
                "Skipping ACL graph capture. To turn on ACL graph capture, "
                "ensure `aclraph_mode` was not manually set to `NONE`")
            return
        else:
            self.initialize_aclgraph_capture()

        set_cudagraph_capturing_enabled(True)
        # Trigger ACL graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        aclgraph_mode = self.compilation_config.cudagraph_mode

        # First capture (original cases)
        with graph_capture(device=self.device):
            if aclgraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                aclgraph_runtime_mode = aclgraph_mode.mixed_mode()
                # make sure we capture the largest batch size first
                compilation_cases = list(reversed(self.cudagraph_batch_sizes))

                try:
                    self._capture_aclgraphs(
                        compilation_cases,
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        uniform_decode=False)
                except Exception as e:
                    error_msg = str(e)
                    error_code = '0x7020023'
                    pattern = r'retCode=([^,\s\.]+)'
                    match = re.search(pattern, error_msg)
                    if match:
                        retCode = match.group(1)
                    # Determine whether the error message is caused by stream capture failure.
                    if match and retCode == error_code:
                        logger.error(
                            f"ACLgraph sizes capture fail: {type(e).__name__}:\n"
                            "ACLgraph has insufficient available streams to capture the configured number of sizes. "
                            "Please verify both the availability of adequate streams and the appropriateness of the configured size count.\n\n"
                            "Recommended solutions:\n"
                            "1. Manually configure the compilation_config parameter "
                            "with a reduced set of sizes: '{\"cudagraph_capture_sizes\":[size1, size2, size3, ...]}'.\n"
                            "2. Utilize ACLgraph's full graph mode as an alternative to the piece-wise approach.\n\n"
                            f"{str(e)}")
                    raise

            if aclgraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                aclgraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    x <= max_num_tokens and x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_aclgraphs(
                    compilation_cases=compilation_cases_decode,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                    in_parallel_streams=False)

        # Second capture (parallel-stream graph pool).
        # Uses self.cudagraph_batch_sizes_parallel which is set in
        # initialize_aclgraph_capture() and respects
        # split_batch_config.parallel_capture_sizes when provided.
        if self.ascend_config.split_batch_config.enable_parallel_streams:
            with graph_capture(device=self.device):
                if aclgraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                    aclgraph_runtime_mode = aclgraph_mode.mixed_mode()
                    # make sure we capture the largest batch size first
                    compilation_cases = list(
                        reversed(self.cudagraph_batch_sizes_parallel))

                    try:
                        self._capture_aclgraphs(
                            compilation_cases,
                            aclgraph_runtime_mode=aclgraph_runtime_mode,
                            uniform_decode=False,
                            in_parallel_streams=True)
                    except Exception as e:
                        error_msg = str(e)
                        error_code = '0x7020023'
                        pattern = r'retCode=([^,\s\.]+)'
                        match = re.search(pattern, error_msg)
                        if match:
                            retCode = match.group(1)
                        # Determine whether the error message is caused by stream capture failure.
                        if match and retCode == error_code:
                            logger.error(
                                f"ACLgraph sizes capture fail: {type(e).__name__}:\n"
                                "ACLgraph has insufficient available streams to capture the configured number of sizes. "
                                "Please verify both the availability of adequate streams and the appropriateness of the configured size count.\n\n"
                                "Recommended solutions:\n"
                                "1. Manually configure the compilation_config parameter "
                                "with a reduced set of sizes: '{\"cudagraph_capture_sizes\":[size1, size2, size3, ...]}'.\n"
                                "2. Utilize ACLgraph's full graph mode as an alternative to the piece-wise approach.\n\n"
                                f"{str(e)}")
                        raise
                if aclgraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                aclgraph_mode.separate_routine():
                    max_num_tokens = self.scheduler_config.max_num_seqs * \
                            self.uniform_decode_query_len
                    decode_cudagraph_batch_sizes = [
                        x for x in self.cudagraph_batch_sizes_parallel if
                        x <= max_num_tokens and x >= self.uniform_decode_query_len
                    ]
                    compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                    self._capture_aclgraphs(
                        compilation_cases=compilation_cases_decode,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        uniform_decode=True,
                        in_parallel_streams=True)

        # Disable aclgraph capturing globally, so any unexpected aclgraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may doing lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

    def capture_model(self) -> None:

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        self._capture_model()

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _update_tokens_for_pcp(self, tokens):
        num_reqs = self.input_batch.num_reqs
        self.num_pcp_pads = self.num_pcp_pads[:num_reqs]
        tokens = np.array(tokens, dtype=np.int32)
        num_decode_reqs = sum(
            self.input_batch.num_computed_tokens_cpu[:num_reqs] >=
            self.input_batch.num_prompt_tokens[:num_reqs])
        num_decode_tokens = sum(tokens[:num_decode_reqs])
        num_padded_scheduled_tokens = np.ceil(
            tokens /
            (2 * self.pcp_size)).astype(np.int32) * (2 * self.pcp_size)
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            tokens[:num_decode_reqs] * self.pcp_size)
        self.num_pcp_pads = torch.tensor(num_padded_scheduled_tokens - tokens)
        cu_padded_tokens, pcp_padded_arange = \
            self._get_cumsum_and_arange(num_padded_scheduled_tokens)
        unpad_mask = torch.from_numpy(
            pcp_padded_arange < np.repeat(tokens, num_padded_scheduled_tokens))
        unpad_mask_decode = unpad_mask[:num_decode_tokens * self.pcp_size]
        unpad_mask_decode = unpad_mask_decode.reshape([-1, self.pcp_size])
        unpad_mask_decode[:, 0] = True
        unpad_mask_decode[:, 1:] = False

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_size
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]
        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes)
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes,
                                                     pcp_tokens)

        def get_current_rank_positions(cu_tokens, rank):
            positions_start_loc = np.zeros_like(cu_tokens)
            positions_start_loc[1:] = cu_tokens[:-1]
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = positions_start_loc + \
                (2 * self.pcp_size - rank - 1) * pcp_chunk_sizes
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + \
                np.repeat(head_start_loc, pcp_chunk_sizes)
            # Decode reqs do not have tail chunks.
            positions[~pcp_head_chunk_mask] = \
                pcp_chunk_arange[num_decode_tokens:] + \
                np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            return positions

        positions = get_current_rank_positions(
            np.zeros(num_reqs, dtype=np.int32), self.pcp_rank)
        # Decode tokens are duplicate and their positions always be 0.
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                tokens[:num_decode_reqs])[1]

        all_positions = [
            get_current_rank_positions(cu_padded_tokens, rank_i)
            for rank_i in range(self.pcp_size)
        ]
        all_positions_tensor = torch.from_numpy(np.concatenate(all_positions))
        self.pcp_allgather_restore_idx[:all_positions_tensor.shape[0]].copy_(
            all_positions_tensor.float().argsort().long(), non_blocking=True)
        return pcp_tokens, positions, unpad_mask

    def _get_cp_local_seq_lens(
        self,
        seq_lens: torch.Tensor,
        pcp_world_size: int = 1,
        dcp_world_size: int = 1,
        cp_kv_cache_interleave_size: int = 1,
    ) -> torch.Tensor:
        """While using pcp or dcp, kv_cache size stored on each rank may be different,
        use this function to calculate split decode seq_lens of each (p/d)cp rank.
        """
        num_requests = seq_lens.size(0)
        total_world_size = pcp_world_size * dcp_world_size
        seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
        rank_offsets = (torch.arange(total_world_size,
                                     dtype=torch.int32).unsqueeze(0).repeat(
                                         num_requests, 1))
        base = (seq_lens_tiled // cp_kv_cache_interleave_size //
                total_world_size * cp_kv_cache_interleave_size)
        remainder = seq_lens_tiled - base * total_world_size
        remainder = torch.clip(
            remainder - rank_offsets * cp_kv_cache_interleave_size,
            0,
            cp_kv_cache_interleave_size,
        )
        dcp_local_seq_lens = (base + remainder).reshape(
            [-1, pcp_world_size, dcp_world_size])
        return dcp_local_seq_lens

    def _generate_pcp_metadata(self, total_num_scheduled_tokens):
        # In dummy run num_reqs == 0, update it from seq_lens
        num_reqs = self.input_batch.num_reqs or self.query_lens.size(0)
        num_decodes = sum(self.input_batch.num_computed_tokens_cpu[:num_reqs]
                          >= self.input_batch.num_prompt_tokens[:num_reqs])
        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        if self.pcp_size * self.dcp_size > 1:
            decode_context_lens = self.input_batch.num_tokens[:num_decodes]
            prefill_context_lens = self.input_batch.num_computed_tokens_cpu[
                num_decodes:num_reqs]
            context_lens = np.concatenate(
                [decode_context_lens, prefill_context_lens])
            num_computed_tokens_of_pcp_dcp = torch.zeros(
                [
                    num_reqs * self.decode_threshold, self.pcp_size,
                    self.dcp_size
                ],
                dtype=torch.int32,
            )
            # For pcp + spec decode, we flatten seq_lens
            # to avoid irregular spec_attn_mask shape
            for decode_idx in range(self.decode_threshold):
                num_computed_tokens_of_pcp_dcp[
                    self.decode_threshold - 1 - decode_idx::self.decode_threshold] = \
                    self._get_cp_local_seq_lens(
                        torch.tensor(context_lens),
                        self.pcp_size,
                        self.dcp_size,
                        self.parallel_config.cp_kv_cache_interleave_size,
                    )
            long_seq_metadata = AscendPrefillContextParallelMetadata(
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.
                numpy())
            if self.pcp_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                q_req_offset = 0
                kv_req_offset = 0
                q_head_chunk_id = self.pcp_rank
                q_tail_chunk_id = self.pcp_size * 2 - 1 - self.pcp_rank
                for i, seq_len in enumerate(self.query_lens):
                    if i < num_decodes:
                        continue
                    chunk_len = seq_len // 2
                    chunk_seqlens.append(chunk_len)
                    q_head_idx.extend(
                        list(range(q_req_offset, q_req_offset + chunk_len)))
                    kv_with_q_head_nomask_idx.extend(
                        list(
                            range(kv_req_offset, kv_req_offset +
                                  chunk_len * q_head_chunk_id)))
                    kv_with_q_head_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_head_chunk_id,
                                kv_req_offset + chunk_len *
                                (q_head_chunk_id + 1))))
                    kv_with_q_head_nomask_seqlens.append(chunk_len *
                                                         q_head_chunk_id)

                    q_tail_idx.extend(
                        list(
                            range(q_req_offset + chunk_len,
                                  q_req_offset + chunk_len * 2)))
                    kv_with_q_tail_nomask_idx.extend(
                        list(
                            range(kv_req_offset, kv_req_offset +
                                  chunk_len * q_tail_chunk_id)))
                    kv_with_q_tail_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                                kv_req_offset + chunk_len *
                                (q_tail_chunk_id + 1))))
                    kv_with_q_tail_nomask_seqlens.append(chunk_len *
                                                         q_tail_chunk_id)

                    q_req_offset += seq_len
                    kv_req_offset += seq_len * self.pcp_size

                # Convert lists to tensors and move to device
                def _list_to_tensor(lst, device, dtype=torch.int32):
                    tensor_npu = torch.zeros(len(lst),
                                             dtype=dtype,
                                             device=device)
                    tensor_npu.copy_(torch.tensor(lst, dtype=dtype),
                                     non_blocking=True)
                    return tensor_npu

                q_head_idx_tensor = _list_to_tensor(q_head_idx, self.device)
                q_tail_idx_tensor = _list_to_tensor(q_tail_idx, self.device)
                self.q_head_idx_tensor = q_head_idx_tensor
                self.q_tail_idx_tensor = q_tail_idx_tensor

                q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
                q_full_idx = q_full_idx.to(torch.float32).argsort().to(
                    torch.int32)
                self.q_full_idx = q_full_idx

                self.kv_idx_names = {
                    'kv_with_q_head_nomask_idx_tensor':
                    kv_with_q_head_nomask_idx,
                    'kv_with_q_head_mask_idx_tensor': kv_with_q_head_mask_idx,
                    'kv_with_q_tail_nomask_idx_tensor':
                    kv_with_q_tail_nomask_idx,
                    'kv_with_q_tail_mask_idx_tensor': kv_with_q_tail_mask_idx
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = _list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_mask_seqlens = torch.tensor(
                    [chunk_seqlens, chunk_seqlens], dtype=torch.int32)
                head_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_head_nomask_seqlens],
                    dtype=torch.int32)
                tail_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_tail_nomask_seqlens],
                    dtype=torch.int32)
                pcp_prefill_mask = self.attn_mask

                self.extra_long_seq_kwargs = {
                    'attn_mask_seqlens': attn_mask_seqlens,
                    'head_attn_nomask_seqlens': head_attn_nomask_seqlens,
                    'tail_attn_nomask_seqlens': tail_attn_nomask_seqlens,
                    'pcp_prefill_mask': pcp_prefill_mask
                }
                long_seq_metadata.pcp_allgather_restore_idx = self.pcp_allgather_restore_idx[:
                                                                                             num_actual_tokens_pcp_padded]
                long_seq_metadata.cp_kv_recover_idx_for_chunk = self.cp_kv_recover_idx_for_chunk
                long_seq_metadata.q_head_idx_tensor = self.q_head_idx_tensor
                long_seq_metadata.q_tail_idx_tensor = self.q_tail_idx_tensor
                long_seq_metadata.q_full_idx = self.q_full_idx
                long_seq_metadata.kv_with_q_head_nomask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_head_nomask_idx_tensor']
                long_seq_metadata.kv_with_q_head_mask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_head_mask_idx_tensor']
                long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_tail_nomask_idx_tensor']
                long_seq_metadata.kv_with_q_tail_mask_idx_tensor = self.kv_idx_names[
                    'kv_with_q_tail_mask_idx_tensor']
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs[
                    'attn_mask_seqlens']
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    'head_attn_nomask_seqlens']
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    'tail_attn_nomask_seqlens']
                long_seq_metadata.pcp_prefill_mask = self.extra_long_seq_kwargs[
                    'pcp_prefill_mask']
            self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata

    def _generate_pcp_mtp_input(
        self,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: dict[str, int],
    ):
        """
        While pcp > 1, model inputs (input_ids, position, etc.) are split across pcp group,
        but mtp need to shift original input_ids before pcp splitting,
        so we record original input_ids here.
        """
        total_num_scheduled_tokens_pcp_full = total_num_scheduled_tokens
        num_scheduled_tokens_pcp_full = np.empty(num_reqs, dtype=np.int32)
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_scheduled_tokens_pcp_full[i] = num_scheduled_tokens[req_id]
        req_indices_pcp_full = np.repeat(self.arange_np[:num_reqs],
                                         num_scheduled_tokens_pcp_full)
        cu_num_tokens_pcp_full = np.cumsum(num_scheduled_tokens_pcp_full)
        self.query_start_loc_pcp_full.np[0] = 0
        self.query_start_loc_pcp_full.np[1:num_reqs +
                                         1] = cu_num_tokens_pcp_full
        self.query_start_loc_pcp_full.np[num_reqs + 1:].fill(-1)
        cumsums_offsets_pcp_full = np.repeat(
            cu_num_tokens_pcp_full - num_scheduled_tokens_pcp_full,
            num_scheduled_tokens_pcp_full)
        arange_pcp_full = self.arange_np[:
                                         total_num_scheduled_tokens_pcp_full] - cumsums_offsets_pcp_full
        positions_pcp_full_np = self.positions_pcp_full_np[:
                                                           total_num_scheduled_tokens_pcp_full]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices_pcp_full],
               arange_pcp_full,
               out=positions_pcp_full_np)
        token_indices_pcp_full = (
            positions_pcp_full_np +
            req_indices_pcp_full * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices_pcp_full),
                           out=self.input_ids_pcp_full.
                           cpu[:total_num_scheduled_tokens_pcp_full])
        self.query_start_loc_pcp_full.copy_to_gpu()
        self.input_ids_pcp_full.gpu[:total_num_scheduled_tokens_pcp_full].copy_(
            self.input_ids_pcp_full.cpu[:total_num_scheduled_tokens_pcp_full],
            non_blocking=True,
        )
    def _to_jsonable(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        # 尝试将对象的属性字典序列化
        if hasattr(obj, "__dict__"):
            return {k: self._to_jsonable(v) for k, v in vars(obj).items()}
        return str(obj)


@contextmanager
def _torch_cuda_wrapper():

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        yield
    except Exception:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
    


