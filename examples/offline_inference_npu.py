#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/basic.py
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
import argparse
from pathlib import Path
import time

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
import torch


def parse_args():
  parser = argparse.ArgumentParser(description="Offline Inference with Profiling")
  parser.add_argument(
      "--enable-profiling",
      default=False,
      action="store_true",
      help="Enable PyTorch profiling"
  )
  parser.add_argument(
      "--profile-dir",
      type=str,
      default="./prof_result/offline_inference_npu",
      help="Directory to save profiling results"
  )
  parser.add_argument(
      "--model",
      type=str,
      default="Qwen/Qwen3-4B",
      help="Model name or path"
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=100,
      help="Maximum number of tokens to generate"
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.0,
      help="Sampling temperature"
  )
  return parser.parse_args()


def main():
  args = parse_args()
  
  prompts = [
      "Hello, my name is",
      "The president of the United States is",
      "The capital of France is",
      "The future of AI is",
  ]
  
  compilation_config = {
      "cudagraph_mode": "FULL_DECODE_ONLY",
  }
  
  # Create a sampling params object.
  sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
  
  # Create an LLM.
  llm = LLM(model="Qwen/Qwen3-0.6B", compilation_config=compilation_config)

  # Setup profiling if enabled
 
  print("Starting generation...")
  outputs = llm.generate(prompts, sampling_params)
  
  # Print results
  for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
  main()