#!/usr/bin/env bash
set -euo pipefail

# Run vLLM OpenAI server under msprof so you can inspect multi-stream overlap.
#
# Usage examples:
#   bash benchmarks/scripts/msprof_profile_vllm_serve.sh \
#     --out ./msprof_out_qwen3_split \
#     vllm serve Qwen/Qwen3-8B --max-model-len 16384 \
#       --additional-config '{"split_batch_config": {"enabled": true, "num_splits": 2, "enable_parallel_streams": true, "min_batch_size_for_split": 4}}'
#
# Notes:
# - msprof arguments vary by CANN version; adjust flags as needed.
# - Consider binding to NVMe output dir for large traces.

OUT_DIR=""
if [[ "${1:-}" == "--out" ]]; then
  OUT_DIR="$2"
  shift 2
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="./msprof_out_$(date +%Y%m%d_%H%M%S)"
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: missing server command" >&2
  exit 2
fi

CMD=("$@")

mkdir -p "$OUT_DIR"

echo "[msprof] output: $OUT_DIR" >&2
echo "[msprof] cmd: ${CMD[*]}" >&2

# Minimal invocation; you can append options like --sys-hardware=on depending on your msprof.
msprof --output="$OUT_DIR" --application="${CMD[*]}"
