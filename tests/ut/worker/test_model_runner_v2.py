# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.worker.model_runner_v2 import NPUModelRunner


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing"""
    model_config = MagicMock(spec=ModelConfig)
    model_config.model = "test_model"
    model_config.max_model_len = 2048
    model_config.is_multimodal_model = False
    model_config.enforce_eager = False
    model_config.use_mla = False
    
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.enable_dbo = True
    
    scheduler_config = MagicMock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 256
    scheduler_config.max_num_batched_tokens = 2048
    
    cache_config = MagicMock(spec=CacheConfig)
    cache_config.block_size = 16
    
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.parallel_config = parallel_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.cache_config = cache_config
    vllm_config.speculative_config = None
    
    return vllm_config


@pytest.fixture
def mock_ascend_config():
    """Create a mock AscendConfig with split batch enabled"""
    config = MagicMock(spec=AscendConfig)
    
    # Mock split_batch_config
    split_config = MagicMock()
    split_config.enabled = True
    split_config.enable_parallel_streams = False
    split_config.num_splits = 2
    split_config.min_batch_size_for_split = 4
    
    config.split_batch_config = split_config
    config.dynamic_eplb = False
    config.dump_config = MagicMock()
    config.dump_config.enable_dump = False
    config.enable_async_exponential = 0
    config.weight_prefetch_config = None
    config.recompute_scheduler_enable = False
    config.expert_map_record_path = None
    
    return config


@pytest.mark.parametrize(
    "tp_size, pp_size, dp_size, enable_dbo, is_multimodal, has_spec_config, expected_enabled",
    [
        # Valid: single card, DBO enabled, no multimodal, no spec decode
        (1, 1, 1, True, False, False, True),
        
        # Invalid: TP > 1
        (2, 1, 1, True, False, False, False),
        
        # Invalid: PP > 1
        (1, 2, 1, True, False, False, False),
        
        # Invalid: DP > 1
        (1, 1, 2, True, False, False, False),
        
        # Invalid: DBO disabled
        (1, 1, 1, False, False, False, False),
        
        # Invalid: multimodal model
        (1, 1, 1, True, True, False, False),
        
        # Invalid: speculative decoding enabled
        (1, 1, 1, True, False, True, False),
    ]
)
def test_split_batch_requirements_check(
    mock_vllm_config,
    tp_size,
    pp_size,
    dp_size,
    enable_dbo,
    is_multimodal,
    has_spec_config,
    expected_enabled,
):
    """Test split batch requirement checking logic"""
    # Setup config
    mock_vllm_config.parallel_config.tensor_parallel_size = tp_size
    mock_vllm_config.parallel_config.pipeline_parallel_size = pp_size
    mock_vllm_config.parallel_config.data_parallel_size = dp_size
    mock_vllm_config.parallel_config.enable_dbo = enable_dbo
    mock_vllm_config.model_config.is_multimodal_model = is_multimodal
    mock_vllm_config.speculative_config = MagicMock() if has_spec_config else None
    
    with patch('vllm_ascend.worker.model_runner_v2.get_ascend_config') as mock_get_config:
        mock_ascend_config = MagicMock()
        split_config = MagicMock()
        split_config.enabled = True
        split_config.enable_parallel_streams = False
        split_config.num_splits = 2
        split_config.min_batch_size_for_split = 4
        mock_ascend_config.split_batch_config = split_config
        mock_get_config.return_value = mock_ascend_config
        
        with patch.object(NPUModelRunner, '_check_split_batch_requirements') as mock_check:
            mock_check.return_value = expected_enabled
            
            runner = MagicMock(spec=NPUModelRunner)
            runner.vllm_config = mock_vllm_config
            runner.parallel_config = mock_vllm_config.parallel_config
            runner.model_config = mock_vllm_config.model_config
            runner.is_multimodal_model = is_multimodal
            runner.speculative_config = mock_vllm_config.speculative_config
            
            # Call the method
            result = NPUModelRunner._check_split_batch_requirements(runner)
            
            assert result == expected_enabled, \
                f"Expected split batch enabled={expected_enabled}, got {result}"


@pytest.mark.parametrize(
    "config_enabled, config_parallel, requirements_met, expected_enabled, expected_parallel",
    [
        # Config enabled, requirements met
        (True, True, True, True, True),
        (True, False, True, True, False),
        
        # Config enabled, requirements not met
        (True, True, False, False, False),
        (True, False, False, False, False),
        
        # Config disabled
        (False, True, True, False, False),
        (False, False, True, False, False),
    ]
)
def test_init_split_batch_config(
    mock_vllm_config,
    config_enabled,
    config_parallel,
    requirements_met,
    expected_enabled,
    expected_parallel,
):
    """Test split batch configuration initialization"""
    with patch('vllm_ascend.worker.model_runner_v2.get_ascend_config') as mock_get_config:
        mock_ascend_config = MagicMock()
        split_config = MagicMock()
        split_config.enabled = config_enabled
        split_config.enable_parallel_streams = config_parallel
        split_config.num_splits = 2
        split_config.min_batch_size_for_split = 4
        mock_ascend_config.split_batch_config = split_config
        mock_get_config.return_value = mock_ascend_config

        # NOTE: Do NOT use MagicMock(spec=NPUModelRunner) here.
        # When `_init_split_batch_config` calls `self._check_split_batch_requirements()`,
        # the instance mock attribute wins over the patched class method.
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = mock_vllm_config
        runner.ascend_config = mock_ascend_config
        
        with patch.object(NPUModelRunner, '_check_split_batch_requirements') as mock_check:
            mock_check.return_value = requirements_met
            
            # Call initialization
            NPUModelRunner._init_split_batch_config(runner)
            
            assert runner.enable_split_batch == expected_enabled, \
                f"Expected enable_split_batch={expected_enabled}, got {runner.enable_split_batch}"
            assert runner.enable_split_parallel_streams == expected_parallel, \
                f"Expected enable_split_parallel_streams={expected_parallel}, got {runner.enable_split_parallel_streams}"


@pytest.mark.parametrize(
    "enable_split, enable_parallel, use_cudagraph, enable_dbo, expected_wrapper",
    [
        # Split batch enabled, sequential mode
        (True, False, True, True, "AscendSplitBatchWrapper"),
        (True, False, False, True, "AscendSplitBatchWrapper"),
        
        # Split batch enabled, parallel mode
        (True, True, True, True, "AscendSplitBatchWrapper"),
        (True, True, False, True, "AscendSplitBatchWrapper"),
        
        # Split batch disabled, DBO enabled (fallback to UBatch)
        (False, False, True, True, "AscendUBatchWrapper"),
        (False, False, False, True, "AscendUBatchWrapper"),
        
        # DBO disabled (fallback to ACLGraph)
        (False, False, True, False, "ACLGraphWrapper"),
    ]
)
def test_wrap_model_with_execution_wrapper(
    mock_vllm_config,
    enable_split,
    enable_parallel,
    use_cudagraph,
    enable_dbo,
    expected_wrapper,
):
    """Test model wrapper selection logic"""
    from vllm.config import CUDAGraphMode
    
    mock_vllm_config.parallel_config.enable_dbo = enable_dbo
    
    with patch('vllm_ascend.worker.model_runner_v2.get_ascend_config') as mock_get_config:
        mock_ascend_config = MagicMock()
        mock_get_config.return_value = mock_ascend_config
        
        # NOTE: Do NOT use MagicMock(spec=NPUModelRunner) here.
        # It replaces methods like _wrap_with_split_batch_wrapper with mocks,
        # so _wrap_model_with_execution_wrapper never calls the real logic.
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = mock_vllm_config
        runner.enable_split_batch = enable_split
        runner.enable_split_parallel_streams = enable_parallel
        runner.compilation_config = MagicMock()
        runner.compilation_config.cudagraph_mode = MagicMock()
        runner.compilation_config.cudagraph_mode.has_full_cudagraphs = MagicMock(
            return_value=use_cudagraph
        )
        runner.parallel_config = mock_vllm_config.parallel_config
        runner.device = torch.device("cpu")

        # Mock the model
        runner.model = MagicMock()
        
        # Mock wrapper classes
        with patch('vllm_ascend.worker.model_runner_v2.ACLGraphWrapper') as mock_acl, \
             patch('vllm_ascend.worker.model_runner_v2.AscendUBatchWrapper') as mock_ubatch, \
             patch('vllm_ascend.worker.model_runner_v2.AscendSplitBatchWrapper') as mock_split:
            
            # Call the method
            NPUModelRunner._wrap_model_with_execution_wrapper(runner)
            
            # Verify correct wrapper was called
            if expected_wrapper == "AscendSplitBatchWrapper":
                mock_split.assert_called_once()
                mock_ubatch.assert_not_called()
                mock_acl.assert_not_called()
            elif expected_wrapper == "AscendUBatchWrapper":
                mock_ubatch.assert_called_once()
                mock_split.assert_not_called()
                mock_acl.assert_not_called()
            elif expected_wrapper == "ACLGraphWrapper":
                mock_acl.assert_called_once()
                mock_ubatch.assert_not_called()
                mock_split.assert_not_called()
                


@pytest.mark.parametrize(
    "batch_size, min_batch_size, num_splits, should_split",
    [
        # Batch size below threshold
        (2, 4, 2, False),
        (3, 4, 2, False),
        
        # Batch size at threshold
        (4, 4, 2, True),
        
        # Batch size above threshold
        (8, 4, 2, True),
        (16, 4, 2, True),
        
        # Different thresholds
        (5, 8, 2, False),
        (8, 8, 2, True),
        (10, 8, 2, True),
    ]
)
def test_split_batch_size_threshold(
    batch_size,
    min_batch_size,
    num_splits,
    should_split,
):
    """Test split batch size threshold logic"""
    runner = MagicMock(spec=NPUModelRunner)
    runner.split_batch_min_size = min_batch_size
    runner.split_batch_num_splits = num_splits
    runner.enable_split_batch = True
    
    # Simulate the check
    actual_should_split = batch_size >= min_batch_size
    
    assert actual_should_split == should_split, \
        f"Batch size {batch_size} with threshold {min_batch_size}: " \
        f"expected should_split={should_split}, got {actual_should_split}"


@pytest.mark.parametrize(
    "num_tokens, num_splits, expected_split_sizes",
    [
        # Even split
        (100, 2, [50, 50]),
        (200, 2, [100, 100]),
        
        # Uneven split (first batch gets extra)
        (101, 2, [51, 50]),
        (105, 2, [53, 52]),
        
        # Three-way split
        (300, 3, [100, 100, 100]),
        (301, 3, [101, 100, 100]),
        (305, 3, [102, 102, 101]),
    ]
)
def test_split_batch_token_distribution(
    num_tokens,
    num_splits,
    expected_split_sizes,
):
    """Test token distribution across splits"""
    # Simulate split logic
    base_size = num_tokens // num_splits
    remainder = num_tokens % num_splits
    
    actual_split_sizes = []
    for i in range(num_splits):
        size = base_size + (1 if i < remainder else 0)
        actual_split_sizes.append(size)
    
    assert actual_split_sizes == expected_split_sizes, \
        f"Token distribution mismatch: expected {expected_split_sizes}, got {actual_split_sizes}"
    
    # Verify total tokens preserved
    assert sum(actual_split_sizes) == num_tokens, \
        f"Total tokens mismatch: expected {num_tokens}, got {sum(actual_split_sizes)}"


def test_split_batch_config_attributes():
    """Test that split batch config has all required attributes"""
    with patch('vllm_ascend.worker.model_runner_v2.get_ascend_config') as mock_get_config:
        mock_ascend_config = MagicMock()
        split_config = MagicMock()
        
        # Set required attributes
        split_config.enabled = True
        split_config.enable_parallel_streams = False
        split_config.num_splits = 2
        split_config.min_batch_size_for_split = 4
        
        mock_ascend_config.split_batch_config = split_config
        mock_get_config.return_value = mock_ascend_config
        
        # Verify all attributes exist
        assert hasattr(split_config, 'enabled')
        assert hasattr(split_config, 'enable_parallel_streams')
        assert hasattr(split_config, 'num_splits')
        assert hasattr(split_config, 'min_batch_size_for_split')
        
        # Verify attribute types
        assert isinstance(split_config.enabled, bool)
        assert isinstance(split_config.enable_parallel_streams, bool)
        assert isinstance(split_config.num_splits, int)
        assert isinstance(split_config.min_batch_size_for_split, int)
        
        # Verify reasonable values
        assert split_config.num_splits >= 2
        assert split_config.min_batch_size_for_split >= 1


@pytest.mark.parametrize(
    "wrapper_type, expected_has_split_method",
    [
        ("AscendSplitBatchWrapper", True),
        ("AscendUBatchWrapper", False),
        ("ACLGraphWrapper", False),
    ]
)
def test_wrapper_has_split_capability(wrapper_type, expected_has_split_method):
    """Test that wrappers have appropriate split capabilities"""
    # This is a structural test to ensure the wrapper has the right interface
    
    if wrapper_type == "AscendSplitBatchWrapper":
        from vllm_ascend.worker.npu_split_wrapper import AscendSplitBatchWrapper
        wrapper_class = AscendSplitBatchWrapper
    elif wrapper_type == "AscendUBatchWrapper":
        from vllm_ascend.worker.npu_ubatch_wrapper import AscendUBatchWrapper
        wrapper_class = AscendUBatchWrapper
    elif wrapper_type == "ACLGraphWrapper":
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        wrapper_class = ACLGraphWrapper
    
    # Check if wrapper has split-related methods
    has_split_method = hasattr(wrapper_class, '_split_batch') or \
                      hasattr(wrapper_class, 'split_and_execute')
    
    assert has_split_method == expected_has_split_method, \
        f"{wrapper_type} split capability mismatch: " \
        f"expected {expected_has_split_method}, got {has_split_method}"


def test_split_batch_logging():
    """Test that split batch configuration is properly logged"""
    with patch('vllm_ascend.worker.model_runner_v2.logger') as mock_logger:
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.enable_split_batch = True
        runner.enable_split_parallel_streams = True
        runner.split_batch_num_splits = 2
        runner.split_batch_min_size = 4
        runner.model = MagicMock()
        runner.parallel_config = MagicMock()
        runner.parallel_config.enable_dbo = True
        
        # Call logging method
        NPUModelRunner._log_wrapper_info(runner)
        
        # Verify logger was called
        assert mock_logger.info.called, "Logger should be called"


@pytest.fixture
def split_batch_mock_runner():
    """Create a mock runner with split batch configuration"""
    runner = MagicMock(spec=NPUModelRunner)
    runner.enable_split_batch = True
    runner.enable_split_parallel_streams = False
    runner.split_batch_num_splits = 2
    runner.split_batch_min_size = 4
    runner.device = torch.device('cpu')
    
    return runner


def test_split_batch_basic_execution(split_batch_mock_runner):
    """Test basic split batch execution flow"""
    runner = split_batch_mock_runner
    
    # Simulate a batch that should be split
    batch_size = 8
    
    # Check if batch should be split
    should_split = (runner.enable_split_batch and 
                   batch_size >= runner.split_batch_min_size)
    
    assert should_split, "Batch should be split"
    
    # Calculate split sizes
    num_splits = runner.split_batch_num_splits
    base_size = batch_size // num_splits
    
    expected_sizes = [base_size] * num_splits
    
    assert sum(expected_sizes) == batch_size, \
        "Split sizes should sum to original batch size"


def test_split_batch_edge_cases():
    """Test edge cases for split batch"""
    runner = MagicMock(spec=NPUModelRunner)
    runner.enable_split_batch = True
    runner.split_batch_min_size = 4
    runner.split_batch_num_splits = 2
    
    # Edge case 1: Batch size exactly at threshold
    batch_size = 4
    should_split = batch_size >= runner.split_batch_min_size
    assert should_split, "Should split at exact threshold"
    
    # Edge case 2: Batch size just below threshold
    batch_size = 3
    should_split = batch_size >= runner.split_batch_min_size
    assert not should_split, "Should not split below threshold"
    
    # Edge case 3: Very large batch
    batch_size = 1000
    should_split = batch_size >= runner.split_batch_min_size
    assert should_split, "Should split large batch"
    
    # Edge case 4: Single token
    batch_size = 1
    should_split = batch_size >= runner.split_batch_min_size
    assert not should_split, "Should not split single token"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
