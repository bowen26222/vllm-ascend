# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatch_utils import (UBatchSlice, UBatchSlices,
                                         is_second_ubatch_empty,
                                         check_ubatch_thresholds)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import dbo_current_stream
from vllm_ascend.worker.npu_ubatch_wrapper import NPUCoreControlContextManager

logger = init_logger(__name__)


def should_ubatch_across_dp(
        should_ubatch: bool, orig_num_tokens_per_ubatch: int,
        padded_num_tokens_per_ubatch: int, dp_size: int,
        dp_rank: int) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]
    """

    device = current_platform.device_type
    tensor = torch.zeros(3, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0

    from vllm.distributed.parallel_state import get_dp_group
    dist.all_reduce(tensor, group=get_dp_group().device_group)

    result: bool = bool(torch.all(tensor[2] == 1).item())
    if not result:
        return result, None

    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_second_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
        logger.debug("Aborting ubatching %s %s", orig_min_num_tokens,
                     padded_max_num_tokens)
        return False, None
    return result, padded_num_tokens_tensor.cpu()


def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return should_ubatch_across_dp(should_ubatch, orig_num_tokens_per_ubatch,
                                   padded_num_tokens_per_ubatch, dp_size,
                                   dp_rank)


def get_dp_padding_ubatch(
        num_tokens_unpadded: int, num_tokens_padded: int,
        should_attempt_ubatching: bool,
        vllm_config: VllmConfig) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        # Early exit.
        tokens_per_ubatch = torch.tensor([num_tokens_padded // 2])
        #return False, None
        return True, tokens_per_ubatch

    # If this DP rank doesn't want to attempt microbatching
    if not should_attempt_ubatching:
        (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
            False, 0, 0, vllm_config)
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(
            "Empty second µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
        should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch,
        vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    return should_ubatch, num_tokens_after_padding

def create_ubatch_slices(num_scheduled_tokens: np.ndarray, split_point: int, use_mla: bool) \
    -> UBatchSlices:
    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    if use_mla:
        first_ubatch_token_slice = slice(0, split_point)
        second_ubatch_token_slice = slice(split_point, cu_num_tokens[-1])

        first_ubatch_req_stop = int(
            np.searchsorted(cu_num_tokens, split_point, side="left"))
        second_ubatch_req_start = int(
            np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
        # Determine request slices using exclusive stop semantics
        # First ubatch includes requests whose tokens overlap [0, split_point)
        first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

        # Second ubatch starts at the request that contains the split_point
        # or the request starting exactly at split_point (if on boundary)
        second_ubatch_req_slice = slice(second_ubatch_req_start,
                                        len(cu_num_tokens) - 1)
    else:
        # currently split by requests
        second_ubatch_req_start = int(
            np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
        first_ubatch_req_slice = slice(0, second_ubatch_req_start)
        second_ubatch_req_slice = slice(second_ubatch_req_start,
                                        len(cu_num_tokens) - 1)
        first_ubatch_token_slice = slice(
            0, cu_num_tokens[second_ubatch_req_start])
        second_ubatch_token_slice = slice(
            cu_num_tokens[second_ubatch_req_start], cu_num_tokens[-1])

    return [
        UBatchSlice(first_ubatch_req_slice, first_ubatch_token_slice),
        UBatchSlice(second_ubatch_req_slice, second_ubatch_token_slice)
    ]


def ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    uniform_decode: bool,
    vllm_config: VllmConfig,
    moe_comm_type: Optional[MoECommType],
    use_mla: bool = True,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to 
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if ubatch_slices is None
    ]

    """
    parallel_config = vllm_config.parallel_config
    # Don't bother with the should_ubatch handshaking unless microbatching
    # is enabled
    if not parallel_config.enable_dbo:
        return (None, None)

    # Check preconditions for microbatching
    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )

    # Don't microbatch unless every other DP worker is also microbatching
    should_ubatch, num_tokens_after_padding = get_dp_padding_ubatch(
        num_tokens_unpadded,
        num_tokens_padded,
        should_attempt_ubatching,
        vllm_config,
    )

    if not should_ubatch or moe_comm_type == MoECommType.MC2:
        return (None, None)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_tokens_after_padding is not None
    token_split_point = int(num_tokens_after_padding[0].item())

    if not use_mla:
        cu_num_tokens = np.zeros(len(num_scheduled_tokens_per_request) + 1,
                                 dtype=np.int32)
        np.cumsum(num_scheduled_tokens_per_request,
                  dtype=np.int32,
                  out=cu_num_tokens[1:])

        split_point = int(
            np.searchsorted(cu_num_tokens, token_split_point, side="right") -
            1)
        imbalance_ratio = (token_split_point -
                           cu_num_tokens[split_point]) / cu_num_tokens[-1]
        if len(num_scheduled_tokens_per_request) == 1 or imbalance_ratio > 0.5:
            return (None, None)

    ubatch_slices = create_ubatch_slices(num_scheduled_tokens_per_request,
                                         token_split_point, use_mla)

    return (ubatch_slices, num_tokens_after_padding)


def create_core_control_context(aic_core: int, aiv_core: int):
    comm_aic_core = aic_core
    comm_aiv_core = aiv_core
    current_stream = dbo_current_stream()

    return NPUCoreControlContextManager(comm_aiv_core=comm_aiv_core,
                                        comm_aic_core=comm_aic_core,
                                        curren_stream=current_stream)


# ==================== Split Batch Support ====================
# Split batch is similar to DBO (ubatch) but with different conditions:
# - DBO: splits batch for overlapping compute/communication
# - Split batch: splits large decode batches for memory/performance optimization
# They never conflict because they are enabled under different conditions.

@dataclass
class SplitBatchSlice:
    """Represents a slice of the batch for split batch execution."""
    request_slice: slice  # Slice of requests
    token_slice: slice    # Slice of tokens
    # Padded token count aligned to the nearest cudagraph capture size.
    # Set by split_batch_split when cudagraph_capture_sizes is provided;
    # equals num_tokens otherwise.
    padded_num_tokens: int = 0

    def __post_init__(self):
        if self.padded_num_tokens == 0:
            self.padded_num_tokens = self.num_tokens

    @property
    def num_requests(self) -> int:
        return self.request_slice.stop - self.request_slice.start
    
    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start
    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )


SplitBatchSlices = list[SplitBatchSlice]


def create_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    num_splits: int,
    custom_split_sizes: Optional[list[int]] = None,
) -> SplitBatchSlices:
    """Create split batch slices by dividing requests evenly.
    
    Unlike DBO which splits by token count, split batch divides by request count
    for uniform decode batches (where each request has 1 token).
    
    Args:
        num_scheduled_tokens_per_request: Array of token counts per request
        num_splits: Number of splits to create
        custom_split_sizes: Optional list of exact split sizes (number of requests per split).
                          If provided, must sum to total number of requests.
        
    Returns:
        List of SplitBatchSlice objects
    """
    num_reqs = len(num_scheduled_tokens_per_request)
    
    # Compute cumulative token counts
    cu_num_tokens = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens_per_request, dtype=np.int32, out=cu_num_tokens[1:])
    
    # Validate custom_split_sizes if provided
    if custom_split_sizes is not None:
        if sum(custom_split_sizes) != num_reqs:
            raise ValueError(
                f"Sum of custom_split_sizes ({sum(custom_split_sizes)}) "
                f"must equal total number of requests ({num_reqs})"
            )
        if len(custom_split_sizes) != num_splits:
            raise ValueError(
                f"Length of custom_split_sizes ({len(custom_split_sizes)}) "
                f"must equal num_splits ({num_splits})"
            )
    
    
    slices = []
    if custom_split_sizes is not None:
        # Use custom split sizes
        start_req = 0
        for split_size in custom_split_sizes:
            end_req = start_req + split_size
            
            if start_req >= num_reqs:
                break
                
            token_start = int(cu_num_tokens[start_req])
            token_end = int(cu_num_tokens[end_req])
            
            slices.append(SplitBatchSlice(
                request_slice=slice(start_req, end_req),
                token_slice=slice(token_start, token_end),
            ))
            
            start_req = end_req
    else:
        # Calculate requests per split (ceil division for even distribution)
        reqs_per_split = (num_reqs + num_splits - 1) // num_splits
        
        for i in range(num_splits):
            start_req = i * reqs_per_split
            end_req = min((i + 1) * reqs_per_split, num_reqs)
            
            if start_req >= num_reqs:
                break
                
            token_start = int(cu_num_tokens[start_req])
            token_end = int(cu_num_tokens[end_req])
            
            slices.append(SplitBatchSlice(
                request_slice=slice(start_req, end_req),
                token_slice=slice(token_start, token_end),
            ))
    
    return slices


def split_batch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    vllm_config: VllmConfig,
    cudagraph_capture_sizes: Optional[set] = None,
    custom_split_sizes: Optional[list[int]] = None,
) -> tuple[Optional[SplitBatchSlices], Optional[int]]:
    """
    Determine if and how to split the batch for split batch execution.
    
    Split batch is designed for:
    - Large batch sizes that exceed min_batch_size_for_split
    - FULL graph mode where each split can use a captured graph
    
    Args:
        num_scheduled_tokens_per_request: Token counts per request
        num_tokens_unpadded: Total tokens without padding
        num_tokens_padded: Total tokens with padding
        vllm_config: vLLM configuration
        cudagraph_capture_sizes: Set of captured graph sizes (for validation)
        custom_split_sizes: Optional list of exact split sizes (number of requests per split).
                          If provided, must sum to total number of requests.
        
    Returns:
        tuple[Optional[SplitBatchSlices], Optional[int]]:
            - split_slices: List of SplitBatchSlice if splitting, None otherwise
            - padded_total_tokens: Total tokens after padding each split
    """
    from vllm_ascend.ascend_config import get_ascend_config
    
    ascend_config = get_ascend_config()
    split_config = ascend_config.split_batch_config
    
    # Check if split batch is enabled
    if not split_config.enabled:
        return (None, None)
    
    num_reqs = len(num_scheduled_tokens_per_request)
    
    # Check minimum batch size threshold
    if num_reqs < split_config.min_batch_size_for_split:
        return (None, None)
    
    num_splits = split_config.num_splits
    
    # Create split slices
    split_slices = create_split_batch_slices(
        num_scheduled_tokens_per_request,
        num_splits,
        custom_split_sizes,
    )
    
    # If we couldn't create valid splits, return None
    if not split_slices or len(split_slices) < 2:
        return (None, None)
    
    # Validate that each split size has a corresponding captured graph
    # (only relevant when cudagraph is enabled)
    if cudagraph_capture_sizes:
        padded_total = 0
        sorted_capture_sizes = sorted(cudagraph_capture_sizes)
        max_capture_size = max(cudagraph_capture_sizes)
        for split_slice in split_slices:
            split_size = split_slice.num_tokens
            # Find the smallest capture size >= split_size
            padded_size = next(
                (cs for cs in sorted_capture_sizes if cs >= split_size),
                max_capture_size,
            )
            split_slice.padded_num_tokens = padded_size
            padded_total += padded_size
        return (split_slices, padded_total)
    
    return (split_slices, num_tokens_padded)


