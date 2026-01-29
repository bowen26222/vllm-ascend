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
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

logger = init_logger(__name__)


@dataclass
class DualStreamGraphEntry:
    """Entry for dual stream graph, similar to ACLGraphEntry"""
    batch_descriptor: BatchDescriptor
    aclgraph_0: torch.npu.NPUGraph | None = None
    aclgraph_1: torch.npu.NPUGraph | None = None
    graph_pool_0: Any = None
    graph_pool_1: Any = None
    compute_stream_0: torch.npu.Stream | None = None
    compute_stream_1: torch.npu.Stream | None = None
    output_0: Any | None = None
    output_1: Any | None = None
    # Store input addresses and tensors from capture for replay
    # Separate buffers for each stream
    input_addresses_0: list[int] | None = None
    input_addresses_1: list[int] | None = None
    capture_input_ids_0: torch.Tensor | None = None
    capture_positions_0: torch.Tensor | None = None
    capture_inputs_embeds_0: torch.Tensor | None = None
    capture_intermediate_tensors_0: Any | None = None
    capture_input_ids_1: torch.Tensor | None = None
    capture_positions_1: torch.Tensor | None = None
    capture_inputs_embeds_1: torch.Tensor | None = None
    capture_intermediate_tensors_1: Any | None = None


class DualStreamUBatchWrapper(UBatchWrapper):
    """
    Wrapper that captures and replays ACLgraphs on dual streams.
    
    Capture phase: Uses full batch data (like ACLGraphWrapper) to capture
    the same graph on two separate streams with separate graph pools.
    
    Runtime phase: Splits input data in half and replays both graphs in parallel
    on two streams, then merges the results.
    """

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.npu.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.runtime_mode = runtime_mode
        self.device = device

        # Create separate graph pools for two streams
        self.graph_pool_0 = torch.npu.graph_pool_handle()
        self.graph_pool_1 = torch.npu.graph_pool_handle()

        # Store captured graphs, keyed by batch_descriptor
        self.graph_entries: dict[BatchDescriptor, DualStreamGraphEntry] = {}

        # Fallback wrapper for non-FULL modes
        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(runnable,
                                                     vllm_config,
                                                     runtime_mode=runtime_mode)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"dual stream wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _split_inputs(self, input_ids, positions, inputs_embeds, 
                     intermediate_tensors, num_tokens):
        """Split inputs in half for dual stream execution."""
        split_point = num_tokens // 2
        
        # Split input_ids
        input_ids_0 = input_ids[:split_point]
        input_ids_1 = input_ids[split_point:]
        
        # Split positions
        if positions.ndim == 2:
            positions_0 = positions[:, :split_point]
            positions_1 = positions[:, split_point:]
        else:
            positions_0 = positions[:split_point]
            positions_1 = positions[split_point:]
        
        # Split inputs_embeds
        inputs_embeds_0 = inputs_embeds[:split_point] if inputs_embeds is not None else None
        inputs_embeds_1 = inputs_embeds[split_point:] if inputs_embeds is not None else None
        
        # Split intermediate_tensors
        intermediate_tensors_0 = None
        intermediate_tensors_1 = None
        if intermediate_tensors is not None:
            from vllm.sequence import IntermediateTensors
            tensors_0 = {}
            tensors_1 = {}
            for key, tensor in intermediate_tensors.tensors.items():
                tensors_0[key] = tensor[:split_point]
                tensors_1[key] = tensor[split_point:]
            intermediate_tensors_0 = IntermediateTensors(tensors_0)
            intermediate_tensors_1 = IntermediateTensors(tensors_1)
        
        return (input_ids_0, positions_0, inputs_embeds_0, intermediate_tensors_0,
                input_ids_1, positions_1, inputs_embeds_1, intermediate_tensors_1)

    def _capture_dual_streams(self, *args, **kwargs) -> torch.Tensor:
        """
        Capture ACLgraphs on two streams using full batch data.
        Similar to ACLGraphWrapper, but captures on two separate streams.
        """
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        
        # Create entry if not exists
        if batch_descriptor not in self.graph_entries:
            self.graph_entries[batch_descriptor] = DualStreamGraphEntry(
                batch_descriptor=batch_descriptor)
        
        entry = self.graph_entries[batch_descriptor]
        
        # Store input addresses and tensors for replay
        input_ids = kwargs.get('input_ids')
        positions = kwargs.get('positions')
        inputs_embeds = kwargs.get('inputs_embeds')
        intermediate_tensors = kwargs.get('intermediate_tensors')
        
        # Create separate input buffers for each stream
        # These will be used during replay to copy split data
        entry.capture_input_ids_0 = input_ids.clone() if input_ids is not None else None
        entry.capture_positions_0 = positions.clone() if positions is not None else None
        entry.capture_inputs_embeds_0 = inputs_embeds.clone() if inputs_embeds is not None else None
        entry.capture_input_ids_1 = input_ids.clone() if input_ids is not None else None
        entry.capture_positions_1 = positions.clone() if positions is not None else None
        entry.capture_inputs_embeds_1 = inputs_embeds.clone() if inputs_embeds is not None else None
        
        if intermediate_tensors is not None:
            from vllm.sequence import IntermediateTensors
            entry.capture_intermediate_tensors_0 = IntermediateTensors({
                key: tensor.clone() for key, tensor in intermediate_tensors.tensors.items()
            })
            entry.capture_intermediate_tensors_1 = IntermediateTensors({
                key: tensor.clone() for key, tensor in intermediate_tensors.tensors.items()
            })
        
        # Get input addresses from args (tensors) - these will be the same for both graphs
        # during capture, but we'll use separate buffers during replay
        input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        entry.input_addresses_0 = input_addresses
        entry.input_addresses_1 = input_addresses
        
        # Create two separate compute streams
        compute_stream_0 = torch.npu.Stream(device=self.device)
        compute_stream_1 = torch.npu.Stream(device=self.device)
        entry.compute_stream_0 = compute_stream_0
        entry.compute_stream_1 = compute_stream_1
        
        # Save original values and temporarily set to avoid nested graph capture and ubatch issues
        original_runtime_mode = forward_context.cudagraph_runtime_mode
        original_ubatch_slices = forward_context.ubatch_slices
        
        # Temporarily set ubatch_slices to None to avoid creating multiple metadata builders
        forward_context.ubatch_slices = None
        
        # Capture graph on stream 0 (using full batch data)
        aclgraph_0 = torch.npu.NPUGraph()
        set_graph_pool_id(self.graph_pool_0)
        forward_context.capturing = True
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE  # Avoid nested capture
        with torch.npu.graph(aclgraph_0, stream=compute_stream_0, pool=self.graph_pool_0):
            output_0 = self.runnable(*args, **kwargs)
        
        # Capture graph on stream 1 (using same full batch data)
        aclgraph_1 = torch.npu.NPUGraph()
        set_graph_pool_id(self.graph_pool_1)
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE  # Avoid nested capture
        with torch.npu.graph(aclgraph_1, stream=compute_stream_1, pool=self.graph_pool_1):
            output_1 = self.runnable(*args, **kwargs)
        
        # Restore original values
        forward_context.cudagraph_runtime_mode = original_runtime_mode
        forward_context.ubatch_slices = original_ubatch_slices
        
        # Store captured graphs and outputs
        entry.aclgraph_0 = aclgraph_0
        entry.aclgraph_1 = aclgraph_1
        entry.graph_pool_0 = self.graph_pool_0
        entry.graph_pool_1 = self.graph_pool_1
        entry.output_0 = output_0
        entry.output_1 = output_1
        
        # Return combined output (for capture, we return the first output)
        return output_0

    def _run_dual_streams(self, *args, **kwargs) -> torch.Tensor:
        """
        Run on dual streams using the same full data (not split).
        Copies full input data to capture buffers and replays both graphs in parallel.
        """
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        
        entry = self.graph_entries.get(batch_descriptor)
        if entry is None or entry.aclgraph_0 is None or entry.aclgraph_1 is None:
            # Fallback to single stream if graph not captured
            return self.runnable(*args, **kwargs)
        
        # Save original ubatch_slices and temporarily set to None
        original_ubatch_slices = forward_context.ubatch_slices
        forward_context.ubatch_slices = None
        
        try:
            # Extract inputs
            input_ids = kwargs.get('input_ids')
            positions = kwargs.get('positions')
            inputs_embeds = kwargs.get('inputs_embeds')
            intermediate_tensors = kwargs.get('intermediate_tensors')
            
            num_tokens = input_ids.shape[0] if input_ids is not None else positions.shape[0]
            
            # Copy full data to capture buffers (not split)
            # For stream 0: copy full data to buffer 0
            if entry.capture_input_ids_0 is not None and input_ids is not None:
                entry.capture_input_ids_0[:num_tokens].copy_(input_ids)
            if entry.capture_positions_0 is not None and positions is not None:
                if positions.ndim == 2:
                    entry.capture_positions_0[:, :num_tokens].copy_(positions)
                else:
                    entry.capture_positions_0[:num_tokens].copy_(positions)
            if entry.capture_inputs_embeds_0 is not None and inputs_embeds is not None:
                entry.capture_inputs_embeds_0[:num_tokens].copy_(inputs_embeds)
            if entry.capture_intermediate_tensors_0 is not None and intermediate_tensors is not None:
                for key in intermediate_tensors.tensors:
                    if key in entry.capture_intermediate_tensors_0.tensors:
                        entry.capture_intermediate_tensors_0.tensors[key][:num_tokens].copy_(
                            intermediate_tensors.tensors[key])
            
            # For stream 1: copy full data to buffer 1
            if entry.capture_input_ids_1 is not None and input_ids is not None:
                entry.capture_input_ids_1[:num_tokens].copy_(input_ids)
            if entry.capture_positions_1 is not None and positions is not None:
                if positions.ndim == 2:
                    entry.capture_positions_1[:, :num_tokens].copy_(positions)
                else:
                    entry.capture_positions_1[:num_tokens].copy_(positions)
            if entry.capture_inputs_embeds_1 is not None and inputs_embeds is not None:
                entry.capture_inputs_embeds_1[:num_tokens].copy_(inputs_embeds)
            if entry.capture_intermediate_tensors_1 is not None and intermediate_tensors is not None:
                for key in intermediate_tensors.tensors:
                    if key in entry.capture_intermediate_tensors_1.tensors:
                        entry.capture_intermediate_tensors_1.tensors[key][:num_tokens].copy_(
                            intermediate_tensors.tensors[key])
            
            # Run on two streams in parallel
            results = [None, None]
            errors = [None, None]
            
            def run_stream_0():
                try:
                    torch.npu.set_device(self.device)
                    set_graph_pool_id(entry.graph_pool_0)
                    with torch.npu.stream(entry.compute_stream_0):
                        torch.npu.synchronize()
                        entry.aclgraph_0.replay()
                    results[0] = entry.output_0[:num_tokens] if entry.output_0 is not None else None
                except Exception as e:
                    errors[0] = e
            
            def run_stream_1():
                try:
                    torch.npu.set_device(self.device)
                    set_graph_pool_id(entry.graph_pool_1)
                    with torch.npu.stream(entry.compute_stream_1):
                        torch.npu.synchronize()
                        entry.aclgraph_1.replay()
                    results[1] = entry.output_1[:num_tokens] if entry.output_1 is not None else None
                except Exception as e:
                    errors[1] = e
            
            # Launch threads
            thread_0 = threading.Thread(target=run_stream_0)
            thread_1 = threading.Thread(target=run_stream_1)
            
            thread_0.start()
            thread_1.start()
            thread_0.join()
            thread_1.join()
            
            # Check for errors
            if errors[0] is not None:
                raise errors[0]
            if errors[1] is not None:
                raise errors[1]
            
            # Both streams used the same full data, so results should be the same
            # Use result from stream 0 (or stream 1 if stream 0 failed)
            if results[0] is not None:
                return results[0]
            elif results[1] is not None:
                return results[1]
            else:
                raise RuntimeError("No results from dual stream execution")
        finally:
            # Restore original ubatch_slices
            forward_context.ubatch_slices = original_ubatch_slices

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # Only use dual stream if ACLgraph is enabled
        # Don't depend on ubatch_slices or FULL mode - we handle data splitting ourselves
        if aclgraph_runtime_mode != CUDAGraphMode.NONE:
            # Using ACLgraph: capture or replay on dual streams
            assert batch_descriptor is not None
            
            # Check if graph needs to be captured
            if batch_descriptor not in self.graph_entries or \
               self.graph_entries[batch_descriptor].aclgraph_0 is None:
                # Capture phase: use full batch data
                return self._capture_dual_streams(*args, **kwargs)
            else:
                # Runtime phase: split data and run on dual streams
                return self._run_dual_streams(*args, **kwargs)
        else:
            # No ACLgraph: fallback to runnable or cudagraph_wrapper
            if self.cudagraph_wrapper is not None:
                return self.cudagraph_wrapper(*args, **kwargs)
            else:
                return self.runnable(*args, **kwargs)
