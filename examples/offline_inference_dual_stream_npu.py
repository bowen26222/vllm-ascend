#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-ascend/examples/offline_inference_npu.py
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
#

# isort: skip_file
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


def main():
    """
    Test script for DualStreamUBatchWrapper.
    
    This script demonstrates how to use the dual stream wrapper which:
    1. Allocates separate memory pools for two streams during capture phase
    2. Captures the same ACLgraph on both streams sequentially
    3. Splits input data in half and runs on two streams separately during execution
    
    Requirements:
    - enable_dual_stream_wrapper must be True (use DualStreamUBatchWrapper)
    - enable_dbo is optional (dual stream wrapper works independently)
    """
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Artificial intelligence will",
        "Machine learning is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    
    # Create an LLM with dual stream wrapper enabled.
    # Note: enable_dual_stream_wrapper is independent of enable_dbo
    llm = LLM(
        model="Qwen/Qwen3-8B",
        enable_dual_stream_wrapper=True,  # Use DualStreamUBatchWrapper
        # Optional: enable_dbo can be used together with dual stream wrapper
        # enable_dbo=True,  # Enable dual batch overlap (optional)
        # Optional: set data parallel size if needed
        # data_parallel_size=2,
    )

    print("Starting inference with DualStreamUBatchWrapper...")
    print(f"Number of prompts: {len(prompts)}")
    
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    
    print("\n" + "=" * 80)
    print("Generation Results:")
    print("=" * 80)
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[{i}] Prompt: {prompt!r}")
        print(f"    Generated: {generated_text!r}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()

