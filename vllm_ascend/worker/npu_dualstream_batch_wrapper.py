# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id
from vllm.forward_context import (DPMetadata, get_forward_context,
                                  override_forward_context)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper

from vllm_ascend.ascend_forward_context import create_ascend_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.dbo.utils import select_dbo_templates
from vllm_ascend.utils import dbo_current_stream, enable_sp
from vllm_ascend.worker.ubatching import dbo_yield, make_ubatch_contexts

logger = init_logger(__name__)


@dataclass
class DualStreamUbatchMetadata(UbatchMetadata):
    pass


@dataclass
class DualStreamNPUGraphMetaData:
    aclgraph_0: torch.npu.NPUGraph
    aclgraph_1: torch.npu.NPUGraph
    graph_pool_0: Any
    graph_pool_1: Any
    compute_stream_0: torch.npu.Stream
    compute_stream_1: torch.npu.Stream
    ubatch_metadata: list[DualStreamUbatchMetadata]
    outputs_0: Any | None = None
    outputs_1: Any | None = None


class DualStreamUBatchWrapper(UBatchWrapper):

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.npu.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        # Two ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(3)

        self.cudagraphs: dict[int, DualStreamNPUGraphMetaData] = {}

        self.cudagraph_wrapper = None
        # Create separate graph pools for two streams
        self.graph_pool_0 = None
        self.graph_pool_1 = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(runnable,
                                                     vllm_config,
                                                     runtime_mode=runtime_mode)
            # Create separate graph pools for each stream
            self.graph_pool_0 = torch.npu.graph_pool_handle()
            self.graph_pool_1 = torch.npu.graph_pool_handle()

        self.device = device
        self.overlap_template = None

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"cudagraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_dual_streams(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture ACLgraphs for dual stream run.
        
        The flow is as follows:
        1. Create separate graph pools and compute streams for two streams
        2. Sequentially capture the same ACLgraph on both streams using the same input data
        3. Store the captured graphs along with their metadata
        """
        
        @torch.inference_mode()
        def _capture_stream_thread(results, ubatch_metadata, stream_id, 
                                  compute_stream, graph_pool):
            torch.npu.set_device(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.npu.stream(compute_stream):
                _ = torch.npu.current_blas_handle()
            with torch.npu.stream(ubatch_context.comm_stream):
                _ = torch.npu.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((stream_id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        num_tokens = ubatch_metadata[0].num_tokens + \
            ubatch_metadata[1].num_tokens

        # Create two separate compute streams
        compute_stream_0 = torch.npu.Stream(device=self.device)
        compute_stream_1 = torch.npu.Stream(device=self.device)

        # Use the same input data for both streams during capture
        # We'll use the first ubatch metadata for both captures to ensure same graph
        capture_metadata = ubatch_metadata[0]

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            # Capture graph on stream 0
            aclgraph_0 = torch.npu.NPUGraph()
            set_graph_pool_id(self.graph_pool_0)
            with torch.npu.graph(aclgraph_0,
                                 stream=compute_stream_0,
                                 pool=self.graph_pool_0):
                # Run model on stream 0 with same metadata for both streams
                ubatch_thread_0 = threading.Thread(
                    target=_capture_stream_thread,
                    args=(results, capture_metadata, 0, 
                          compute_stream_0, self.graph_pool_0))
                ubatch_thread_0.start()
                self.ready_barrier.wait()
                capture_metadata.context.cpu_wait_event.set()
                ubatch_thread_0.join()
            
            # Capture graph on stream 1 (same graph, different stream and pool)
            # Use the same input data to capture the same graph
            aclgraph_1 = torch.npu.NPUGraph()
            set_graph_pool_id(self.graph_pool_1)
            with torch.npu.graph(aclgraph_1,
                                 stream=compute_stream_1,
                                 pool=self.graph_pool_1):
                # Run model on stream 1 with same metadata to capture same graph
                ubatch_thread_1 = threading.Thread(
                    target=_capture_stream_thread,
                    args=(results, capture_metadata, 1,
                          compute_stream_1, self.graph_pool_1))
                ubatch_thread_1.start()
                self.ready_barrier.wait()
                capture_metadata.context.cpu_wait_event.set()
                ubatch_thread_1.join()
            
            sorted_results = [value for position, value in sorted(results)]
            result_0 = sorted_results[0] if len(sorted_results) > 0 else None
            result_1 = sorted_results[1] if len(sorted_results) > 1 else None
            
            # Store the captured graphs
            cudagraph_metadata = DualStreamNPUGraphMetaData(
                aclgraph_0=aclgraph_0,
                aclgraph_1=aclgraph_1,
                graph_pool_0=self.graph_pool_0,
                graph_pool_1=self.graph_pool_1,
                compute_stream_0=compute_stream_0,
                compute_stream_1=compute_stream_1,
                ubatch_metadata=ubatch_metadata,
                outputs_0=result_0,
                outputs_1=result_1,
            )
            self.cudagraphs[num_tokens] = cudagraph_metadata
            
            # Return combined result
            if result_0 is not None and result_1 is not None:
                return torch.cat([result_0, result_1], dim=0)
            elif result_0 is not None:
                return result_0
            elif result_1 is not None:
                return result_1
            else:
                raise RuntimeError("No results from capture")

    def _split_metadata(self, ubatch_metadata: DualStreamUbatchMetadata, 
                       split_point: int, first_half: bool) -> DualStreamUbatchMetadata:
        """
        Split metadata in half for dual stream execution.
        
        Args:
            ubatch_metadata: Original metadata to split
            split_point: Point at which to split the data
            first_half: If True, return first half; if False, return second half
        """
        if first_half:
            token_slice = slice(0, split_point)
        else:
            token_slice = slice(split_point, ubatch_metadata.num_tokens)
        
        sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
        sliced_intermediate_tensors = \
            self._slice_model_inputs(
                token_slice, 
                ubatch_metadata.input_ids,
                ubatch_metadata.positions,
                ubatch_metadata.inputs_embeds,
                ubatch_metadata.intermediate_tensors)
        
        return DualStreamUbatchMetadata(
            context=ubatch_metadata.context,
            input_ids=sliced_input_ids,
            positions=sliced_positions,
            inputs_embeds=sliced_inputs_embeds,
            intermediate_tensors=sliced_intermediate_tensors,
            num_tokens=token_slice.stop - token_slice.start)

    def _run_dual_streams(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Run model on dual streams with split data.
        
        The flow is as follows:
        1. Split input data in half
        2. Run on two streams in parallel using captured graphs
        3. Merge results from both streams
        """
        
        @torch.inference_mode()
        def _run_stream_thread(results, metadata, aclgraph, graph_pool, 
                              compute_stream, stream_id):
            torch.npu.set_device(self.device)
            with torch.npu.stream(compute_stream):
                set_graph_pool_id(graph_pool)
                with metadata.context:
                    # Run model with split metadata
                    # Note: We run the model directly with split data instead of replaying
                    # the graph, since the graph was captured with different input data
                    model_output = model(
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        intermediate_tensors=metadata.intermediate_tensors,
                        inputs_embeds=metadata.inputs_embeds,
                    )
                    dbo_current_stream().synchronize()
                    dbo_yield()
                results.append((stream_id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        num_tokens = ubatch_metadata[0].num_tokens + \
            ubatch_metadata[1].num_tokens
        
        # Get stored graph metadata
        graph_metadata = self.cudagraphs[num_tokens]
        
        # Combine all input data first, then split in half
        # Concatenate input_ids from both ubatches
        all_input_ids = torch.cat([ubatch_metadata[0].input_ids, 
                                   ubatch_metadata[1].input_ids], dim=0)
        # Concatenate positions
        if ubatch_metadata[0].positions.ndim == 2:
            all_positions = torch.cat([ubatch_metadata[0].positions,
                                      ubatch_metadata[1].positions], dim=1)
        else:
            all_positions = torch.cat([ubatch_metadata[0].positions,
                                      ubatch_metadata[1].positions], dim=0)
        # Concatenate inputs_embeds if present
        all_inputs_embeds = None
        if ubatch_metadata[0].inputs_embeds is not None:
            all_inputs_embeds = torch.cat([ubatch_metadata[0].inputs_embeds,
                                          ubatch_metadata[1].inputs_embeds], dim=0)
        # Concatenate intermediate_tensors if present
        all_intermediate_tensors = None
        if ubatch_metadata[0].intermediate_tensors is not None:
            all_intermediate_tensors = IntermediateTensors({
                key: torch.cat([
                    ubatch_metadata[0].intermediate_tensors.tensors[key],
                    ubatch_metadata[1].intermediate_tensors.tensors[key]
                ], dim=0)
                for key in ubatch_metadata[0].intermediate_tensors.tensors
            })
        
        # Split in half
        total_tokens = num_tokens
        split_point = total_tokens // 2
        
        # Create split metadata for stream 0 (first half)
        split_slice_0 = slice(0, split_point)
        sliced_input_ids_0, sliced_positions_0, sliced_inputs_embeds_0, \
        sliced_intermediate_tensors_0 = \
            self._slice_model_inputs(
                split_slice_0, all_input_ids, all_positions,
                all_inputs_embeds, all_intermediate_tensors)
        
        metadata_0 = DualStreamUbatchMetadata(
            context=ubatch_metadata[0].context,
            input_ids=sliced_input_ids_0,
            positions=sliced_positions_0,
            inputs_embeds=sliced_inputs_embeds_0,
            intermediate_tensors=sliced_intermediate_tensors_0,
            num_tokens=split_point)
        
        # Create split metadata for stream 1 (second half)
        split_slice_1 = slice(split_point, total_tokens)
        sliced_input_ids_1, sliced_positions_1, sliced_inputs_embeds_1, \
        sliced_intermediate_tensors_1 = \
            self._slice_model_inputs(
                split_slice_1, all_input_ids, all_positions,
                all_inputs_embeds, all_intermediate_tensors)
        
        metadata_1 = DualStreamUbatchMetadata(
            context=ubatch_metadata[1].context,
            input_ids=sliced_input_ids_1,
            positions=sliced_positions_1,
            inputs_embeds=sliced_inputs_embeds_1,
            intermediate_tensors=sliced_intermediate_tensors_1,
            num_tokens=total_tokens - split_point)

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            # Run on two streams in parallel
            thread_0 = threading.Thread(
                target=_run_stream_thread,
                args=(results, metadata_0, graph_metadata.aclgraph_0,
                      graph_metadata.graph_pool_0, 
                      graph_metadata.compute_stream_0, 0))
            thread_1 = threading.Thread(
                target=_run_stream_thread,
                args=(results, metadata_1, graph_metadata.aclgraph_1,
                      graph_metadata.graph_pool_1,
                      graph_metadata.compute_stream_1, 1))
            
            thread_0.start()
            thread_1.start()
            self.ready_barrier.wait()
            metadata_0.context.cpu_wait_event.set()
            thread_0.join()
            thread_1.join()
        
        sorted_results = [value for position, value in sorted(results)]
        
        # Handle sequence parallelism and padding if needed
        if get_forward_context().sp_enabled and get_pp_group().is_last_rank:
            for i in range(len(sorted_results)):
                sorted_results[i] = tensor_model_parallel_all_gather(
                    sorted_results[i], 0)
                if i < len(ubatch_metadata):
                    pad_size = ubatch_metadata[i].context.forward_context.pad_size
                    if pad_size > 0:
                        sorted_results[i] = sorted_results[i][:-pad_size, :]
        
        if not get_pp_group().is_last_rank:
            # Merge the IntermediateTensors in pp scenarios
            result = self._merge_intermediate_tensors(sorted_results)
        else:
            result = torch.cat(sorted_results, dim=0)
        
        # update outside forward context
        get_forward_context().dbo_enabled = True
        return result

    def _make_ubatch_metadata(
            self, ubatch_slices, attn_metadata, input_ids, positions,
            inputs_embeds, intermediate_tensors, compute_stream, dp_metadata,
            batch_descriptor,
            cudagraph_runtime_mode) -> list[DualStreamUbatchMetadata]:

        # Create one forward context per ubatch
        forward_contexts = []
        cur_forward_context = get_forward_context()
        dbo_template = select_dbo_templates(self.vllm_config)
        # Construct forward context list based on the current forward context
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=attn_metadata[i]
                    if attn_metadata is not None else None,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    positions=positions,
                    dbo_template=dbo_template,
                ))

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier)

        ubatch_metadata: list[DualStreamUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                DualStreamUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop -
                    ubatch_slice.token_slice.start))

        return ubatch_metadata

    def _slice_model_inputs(self, tokens_slice: slice, input_ids, positions,
                            inputs_embeds, intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[
            tokens_slice] if inputs_embeds else None
        # consider pp scenario
        if intermediate_tensors is not None:
            # if enable sp, dbo should not split intermediate tensors using token_slice
            # instead, it should calculate the tensor lens after reduce scatter
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

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:

            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE,
                                          CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (ubatch_slices[0].token_slice.stop -
                      ubatch_slices[0].token_slice.start) * 2
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.npu.current_stream()

        ubatch_dp_metadata = []
        dp_size = self.vllm_config.parallel_config.data_parallel_size

        for ubatch_slice in ubatch_slices:
            if dp_size > 1:
                ubatch_num_tokens_across_dp = torch.tensor(
                    [ubatch_slice.num_tokens] * dp_size,
                    device="cpu",
                    dtype=torch.int32)
                ubatch_dp_metadata.append(
                    DPMetadata.make(
                        self.vllm_config.parallel_config,
                        ubatch_slice.num_tokens,
                        ubatch_num_tokens_across_dp,
                    ))
            else:
                ubatch_dp_metadata.append(None)


        if num_tokens not in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                # npu graph should be captured in non-default stream
                compute_stream=torch.npu.Stream(
                    device=torch.npu.current_device()),
                # after padding for cudagraph
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._capture_dual_streams(ubatch_metadata, self.model)
        elif num_tokens in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            return self._run_dual_streams(
                self._make_ubatch_metadata(
                    ubatch_slices=ubatch_slices,
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    compute_stream=compute_stream,
                    dp_metadata=ubatch_dp_metadata,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE),
                self.model)
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._run_dual_streams(ubatch_metadata, self.model)

    def _merge_intermediate_tensors(self, intermediate_tensor_list):

        assert len(intermediate_tensor_list) == 2
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat([
                intermediate_tensor_list[0].tensors[key],
                intermediate_tensor_list[1].tensors[key]
            ],
                                    dim=0)

        res = IntermediateTensors(result)
        return res

