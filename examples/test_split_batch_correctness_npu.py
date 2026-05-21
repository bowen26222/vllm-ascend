#!/usr/bin/env python3
# isort: skip_file
"""Offline correctness check for Ascend split-batch.

目标:
- 验证开启 split-batch 不会改变生成结果（token_ids 与 text 严格一致）。

运行方式（推荐：主控模式，子进程分别跑 disabled/enabled，再做严格对比）:
  python vllm-ascend/examples/test_split_batch_correctness_npu.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 64 --batch-size 8

输出:
- 会在 --output-dir 下创建时间戳子目录，保存:
  - prompts.json
  - outputs_split_disabled.json
  - outputs_split_enabled.json
  - diff.json (仅失败时)
  - metadata.json
  - summary.json
  - console.log （抓取 stdout/stderr）

备注:
- 为确保 split-batch 实际生效，batch_size 应 >= min_batch_size_for_split（默认 4）。
- 默认强制确定性解码（temperature=0），并使用 seed 生成固定 prompts。
"""

import gc
import logging
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import torch
import torch_npu

# Match existing offline example behavior.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Ensure a supported All2All backend is selected before vLLM config validation.
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "3")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
import vllm.logger as vllm_logger_module  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


CONFIGS: dict[str, dict[str, Any]] = {
    "split_disabled": {
        "enabled": False,
        "description": "Split-batch disabled",
    },
    "split_enabled": {
        "enabled": True,
        "description": "Split-batch enabled",
    },
}


class _TeeTextIO:
    """Write to multiple text streams (used to tee stdout/stderr to a file)."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
            except Exception:
                # Best-effort: keep other streams working.
                pass
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for st in self._streams:
            try:
                if hasattr(st, "isatty") and st.isatty():
                    return True
            except Exception:
                continue
        return False


def create_parser() -> FlexibleArgumentParser:
    """Create a CLI parser in the same style as vLLM offline examples."""
    parser = FlexibleArgumentParser()

    # Add all standard engine/vllm args (model, tokenizer, compilation_config, etc).
    EngineArgs.add_cli_args(parser)

    # Match previous behavior in this script.
    parser.set_defaults(trust_remote_code=True)

    # Default compilation config: pass explicitly so Ascend platform sees it.
    # IMPORTANT: Only include keys that exist on vLLM's `CompilationConfig`.
    cudagraph_sizes = [1,2,3,4,5,6,7,8]
    compilation_config = {
        "level": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
    }
    parser.set_defaults(compilation_config=compilation_config)

    test_group = parser.add_argument_group("Split-batch correctness test")
    test_group.add_argument("--max-tokens", type=int, default=64)
    test_group.add_argument("--batch-size", type=int, default=8)
    test_group.add_argument("--num-splits", type=int, default=2)
    test_group.add_argument("--min-batch-size-for-split", type=int, default=4)
    test_group.add_argument(
        "--enable-parallel-streams",
        action="store_true",
        help="Enable split parallel streams (if supported).",
    )
    test_group.add_argument(
        "--parallel-capture-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated capture sizes for the parallel-stream graph pool "
            "(split_batch_config.parallel_capture_sizes). "
            "When omitted the parallel pool reuses cudagraph_capture_sizes. "
            "Example: --parallel-capture-sizes 1,2,4,8,16,32,64,128"
        ),
    )
    test_group.add_argument(
        "--force-split",
        action="store_true",
        help=(
            "Force split-batch for every decode step regardless of padding "
            "savings (split_batch_config.force_split). Useful for benchmarking "
            "the split path on all batch sizes including exact graph hits."
        ),
    )
    test_group.add_argument(
        "--run",
        type=str,
        default="both",
        choices=["both", "disabled", "enabled"],
        help=(
            "Which run(s) to execute: both (compare), disabled only, enabled only. "
            "Coordinator mode runs children when run=both (default)."
        ),
    )
    test_group.add_argument(
        "--compare-mode",
        type=str,
        default="subprocess",
        choices=["subprocess", "inproc"],
        help=(
            "When --run=both: 'subprocess' (recommended) runs two child processes and "
            "compares saved outputs; 'inproc' runs two engines in-process."
        ),
    )
    test_group.add_argument(
        "--output-dir",
        type=str,
        default="./split_batch_correctness_results",
        help=(
            "Base output directory. This script will always create a timestamp subdir "
            "and write all results there (including console.log)."
        ),
    )
    test_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "JSON file containing list[str] prompts. In coordinator mode, this is "
            "auto-generated and passed to child runs."
        ),
    )
    test_group.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Single-run output JSON path (used by coordinator children). If set, "
            "the output directory is derived from this path."
        ),
    )

    prof_group = parser.add_argument_group("PyTorch Profiler (torch_npu)")
    prof_group.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Enable torch_npu profiler and write TensorBoard traces. "
            "NOTE: For meaningful NPU kernel traces with vLLM V1, run with "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 so model executes in-process."
        ),
    )
    prof_group.add_argument(
        "--profile-dir",
        type=str,
        default="./torch_profiler",
        help="Output directory for TensorBoard traces.",
    )
    prof_group.add_argument(
        "--profile-target",
        type=str,
        default="enabled",
        choices=["disabled", "enabled", "both"],
        help="Which run to profile: split disabled, enabled, or both.",
    )
    prof_group.add_argument(
        "--profile-record-shapes",
        action="store_true",
        help="Record operator input shapes (more overhead).",
    )
    prof_group.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Record Python call stacks (more overhead).",
    )
    return parser


def _build_llm_from_args(
    llm_args: dict[str, Any],
    *,
    additional_config: dict[str, Any],
) -> LLM:
    args = dict(llm_args)
    args["additional_config"] = additional_config
    return LLM(**args)


def _extract_first_output(output) -> tuple[list[int], str]:
    # vLLM RequestOutput: output.outputs is a list (n=best_of)
    if not output.outputs:
        return [], ""
    o0 = output.outputs[0]
    token_ids = list(getattr(o0, "token_ids", []) or [])
    text = str(getattr(o0, "text", "") or "")
    return token_ids, text


def _default_prompts() -> list[str]:
    return [
        "Hello, my name is",
        "The capital of France is",
        "Explain in one sentence: what is split-batch?",
        "Write a short list of 3 items about: apples",
        "The president of the United States is",
        "Once upon a time in a land far away,",
        "In computer science, a binary tree is",
        "The quick brown fox jumps over the lazy dog.",
    ]


def _generate_prompts(*, batch_size: int, seed: int | None) -> list[str]:
    base = _default_prompts()
    prompts = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]

    # Make prompt order deterministic but seed-dependent.
    rng = random.Random(int(seed or 0))
    rng.shuffle(prompts)
    return prompts


def _load_prompts(prompts_file: str) -> list[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError("--prompts-file must be a JSON list[str]")
    return list(prompts)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _serialize_outputs(prompts: list[str], outputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, out in enumerate(outputs):
        token_ids, text = _extract_first_output(out)
        rows.append(
            {
                "index": i,
                "prompt": prompts[i] if i < len(prompts) else None,
                "token_ids": token_ids,
                "text": text,
            }
        )
    return rows


def _compare_serialized(
    disabled_rows: list[dict[str, Any]],
    enabled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    if len(disabled_rows) != len(enabled_rows):
        mismatches.append(
            {
                "index": None,
                "reason": "output length differs",
                "disabled_len": len(disabled_rows),
                "enabled_len": len(enabled_rows),
            }
        )

    n = min(len(disabled_rows), len(enabled_rows))
    for i in range(n):
        a = disabled_rows[i]
        b = enabled_rows[i]
        if a.get("prompt") != b.get("prompt"):
            mismatches.append(
                {
                    "index": i,
                    "reason": "prompt differs",
                    "disabled_prompt": a.get("prompt"),
                    "enabled_prompt": b.get("prompt"),
                }
            )
            continue

        if a.get("token_ids") != b.get("token_ids") or a.get("text") != b.get("text"):
            mismatches.append(
                {
                    "index": i,
                    "prompt": a.get("prompt"),
                    "disabled": {
                        "token_ids": a.get("token_ids"),
                        "text": a.get("text"),
                    },
                    "enabled": {
                        "token_ids": b.get("token_ids"),
                        "text": b.get("text"),
                    },
                }
            )

    return mismatches


def _cleanup_llm(llm: LLM) -> None:
    """Release engine resources before starting the next run."""
    try:
        # vLLM's `LLM` does not expose a stable `shutdown()` method.
        # For the V1 engine, we must explicitly shutdown the underlying
        # EngineCore client to terminate the EngineCore subprocess(es),
        # otherwise the first run can keep NPU memory reserved and make the
        # second run see `Available memory: 0`.
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    finally:
        # Best-effort memory release
        if hasattr(torch, "npu"):
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
        gc.collect()


def _run_with_torch_profiler(
    *,
    enabled: bool,
    profile_dir: str,
    run_name: str,
    record_shapes: bool,
    with_stack: bool,
    fn,
):
    if not enabled:
        return fn()

    try:
        from torch_npu import profiler as npu_profiler
    except Exception as e:
        raise RuntimeError(
            "--profile was set but torch_npu.profiler is not available"
        ) from e

    os.makedirs(profile_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(profile_dir, f"{ts}_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    activities = [npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU]
    trace_handler = npu_profiler.tensorboard_trace_handler(out_dir)

    # Use an explicit schedule to avoid "stop while RECORD" edge cases.
    schedule = npu_profiler.schedule(wait=0, warmup=0, active=1, repeat=1)

    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text
            ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        mstx=False,    # 原参数名msprof_tx改为mstx，新版本依旧兼容原参数名msprof_tx
        mstx_domain_include=[],
        mstx_domain_exclude=[],
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False
    )

    with npu_profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_modules=True,
        on_trace_ready=trace_handler,
        profile_memory=False,
        experimental_config=experimental_config
    ) as prof:
        result = fn()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        prof.step()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        return result


def _build_split_additional_config(
    *,
    enabled: bool,
    num_splits: int,
    enable_parallel_streams: bool,
    min_batch_size_for_split: int,
    parallel_capture_sizes: list[int] | None = None,
    force_split: bool = False,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": enabled,
        "num_splits": num_splits,
        "enable_parallel_streams": enable_parallel_streams,
        "min_batch_size_for_split": min_batch_size_for_split,
    }
    if parallel_capture_sizes is not None:
        cfg["parallel_capture_sizes"] = parallel_capture_sizes
    if force_split:
        cfg["force_split"] = True
    return {"split_batch_config": cfg}


def _run_single(
    *,
    llm_args: dict[str, Any],
    prompts: list[str],
    sampling: SamplingParams,
    additional_config: dict[str, Any],
    profile_enabled: bool,
    profile_dir: str,
    profile_target: str,
    profile_record_shapes: bool,
    profile_with_stack: bool,
    run_name: str,
):
    llm = _build_llm_from_args(llm_args, additional_config=additional_config)
    outputs = _run_with_torch_profiler(
        enabled=profile_enabled and profile_target in (run_name, "both"),
        profile_dir=profile_dir,
        run_name=f"split_{run_name}",
        record_shapes=profile_record_shapes,
        with_stack=profile_with_stack,
        fn=lambda: llm.generate(prompts, sampling),
    )
    return outputs, llm


def _coordinator_subprocess(*, base_args: dict[str, Any], prompts: list[str], **ctx) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(str(ctx["output_dir_base"]), timestamp)
    os.makedirs(out_dir, exist_ok=True)

    prompts_path = os.path.join(out_dir, "prompts.json")
    _save_json(prompts_path, prompts)

    metadata = {
        "timestamp": timestamp,
        "model": base_args.get("model"),
        "tokenizer": base_args.get("tokenizer") or base_args.get("model"),
        "seed": base_args.get("seed"),
        "batch_size": ctx["batch_size"],
        "max_tokens": ctx["max_tokens"],
        "max_model_len": base_args.get("max_model_len"),
        "gpu_memory_utilization": base_args.get("gpu_memory_utilization"),
        "num_splits": ctx["num_splits"],
        "enable_parallel_streams": ctx["enable_parallel_streams"],
        "min_batch_size_for_split": ctx["min_batch_size_for_split"],
        "compilation_config": base_args.get("compilation_config"),
        "compare_mode": "subprocess",
    }
    _save_json(os.path.join(out_dir, "metadata.json"), metadata)

    outputs_disabled_path = os.path.join(out_dir, "outputs_split_disabled.json")
    outputs_enabled_path = os.path.join(out_dir, "outputs_split_enabled.json")

    script_path = str(Path(__file__).resolve())

    # Forward argv, but remove coordinator-only flags and --run.
    forward_argv: list[str] = []
    skip_next = False
    coordinator_flags_with_value = {
        "--run",
        "--compare-mode",
        "--output-dir",
        "--prompts-file",
        "--output-file",
    }
    profiler_flags_with_value = {
        "--profile-dir",
        "--profile-target",
    }
    profiler_flags_bool = {
        "--profile",
        "--profile-record-shapes",
        "--profile-with-stack",
    }

    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in coordinator_flags_with_value:
            skip_next = True
            continue
        # Don't forward profiler flags by default; we attach per-child below.
        if tok in profiler_flags_with_value:
            skip_next = True
            continue
        if tok in profiler_flags_bool:
            continue
        forward_argv.append(tok)

    def _child_cmd(*, child_run: str, child_out: str) -> list[str]:
        cmd = [
            sys.executable,
            script_path,
            *forward_argv,
            "--run",
            child_run,
            "--compare-mode",
            "inproc",
            "--prompts-file",
            prompts_path,
            "--output-file",
            child_out,
            "--output-dir",
            out_dir,  # ensure children write console.log into the same run dir
        ]

        if ctx["profile_enabled"] and ctx["profile_target"] in (child_run, "both"):
            cmd.append("--profile")
            cmd.extend(["--profile-dir", ctx["profile_dir"]])
            cmd.extend(["--profile-target", child_run])
            if ctx["profile_record_shapes"]:
                cmd.append("--profile-record-shapes")
            if ctx["profile_with_stack"]:
                cmd.append("--profile-with-stack")
        return cmd

    print("=== Coordinator (subprocess) ===")
    print("Output dir:", out_dir)

    print("\n=== Run 1/2: split disabled (child process) ===")
    r0 = subprocess.run(_child_cmd(child_run="disabled", child_out=outputs_disabled_path))
    if r0.returncode != 0:
        print(f"FAIL: child(disabled) exit code={r0.returncode}", file=sys.stderr)
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "reason": "child(disabled) non-zero exit",
                "returncode": r0.returncode,
            },
        )
        return r0.returncode or 1

    print("\n=== Run 2/2: split enabled (child process) ===")
    r1 = subprocess.run(_child_cmd(child_run="enabled", child_out=outputs_enabled_path))
    if r1.returncode != 0:
        print(f"FAIL: child(enabled) exit code={r1.returncode}", file=sys.stderr)
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "reason": "child(enabled) non-zero exit",
                "returncode": r1.returncode,
            },
        )
        return r1.returncode or 1

    with open(outputs_disabled_path, "r", encoding="utf-8") as f:
        disabled_obj = json.load(f)
    with open(outputs_enabled_path, "r", encoding="utf-8") as f:
        enabled_obj = json.load(f)

    disabled_rows = disabled_obj.get("outputs", [])
    enabled_rows = enabled_obj.get("outputs", [])

    mismatches = _compare_serialized(disabled_rows, enabled_rows)
    if mismatches:
        diff_path = os.path.join(out_dir, "diff.json")
        _save_json(diff_path, {"mismatches": mismatches[:50], "total": len(mismatches)})
        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "FAIL",
                "mismatch_total": len(mismatches),
                "diff_path": diff_path,
            },
        )
        print(f"\nFAIL: {len(mismatches)} mismatches. See {diff_path}", file=sys.stderr)
        return 1

    sample = enabled_rows[0] if enabled_rows else {}
    _save_json(
        os.path.join(out_dir, "summary.json"),
        {
            "status": "PASS",
            "count": len(enabled_rows),
            "sample": {
                "token_ids_len": len(sample.get("token_ids", []) or []),
                "text_preview": (sample.get("text") or "")[:200],
            },
        },
    )

    print(f"\nPASS: {len(prompts)}/{len(prompts)} outputs match exactly")
    print("Sample output[0] token_ids_len=", len(sample.get("token_ids", []) or []))
    print("Sample output[0] text=", repr((sample.get("text") or "")[:200]))
    return 0


def _make_run_dir(*, output_dir_base: str, output_file: str | None) -> tuple[str, str]:
    """Return (out_dir, timestamp). If output_file is set, derive out_dir from it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        out_dir = os.path.dirname(str(output_file)) or "."
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, timestamp
    out_dir = os.path.join(str(output_dir_base), timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, timestamp


def main() -> int:
    parser = create_parser()
    args: dict[str, Any] = vars(parser.parse_args())

    max_tokens = int(args.pop("max_tokens"))
    batch_size = int(args.pop("batch_size"))
    num_splits = int(args.pop("num_splits"))
    min_batch_size_for_split = int(args.pop("min_batch_size_for_split"))
    enable_parallel_streams = bool(args.pop("enable_parallel_streams"))
    _parallel_capture_sizes_raw = args.pop("parallel_capture_sizes")
    parallel_capture_sizes: list[int] | None = (
        [int(s.strip()) for s in _parallel_capture_sizes_raw.split(",") if s.strip()]
        if _parallel_capture_sizes_raw
        else None
    )
    force_split = bool(args.pop("force_split"))
    run_mode = str(args.pop("run"))
    compare_mode = str(args.pop("compare_mode"))
    output_dir_base = str(args.pop("output_dir"))
    prompts_file = args.pop("prompts_file")
    output_file = args.pop("output_file")

    profile_enabled = bool(args.pop("profile"))
    profile_dir = str(args.pop("profile_dir"))
    profile_target = str(args.pop("profile_target"))
    profile_record_shapes = bool(args.pop("profile_record_shapes"))
    profile_with_stack = bool(args.pop("profile_with_stack"))

    # In vLLM V1, setting VLLM_ENABLE_V1_MULTIPROCESSING=0 forces in-proc execution.
    # In this mode, releasing NPU memory fully between two separate engine instantiations
    # is unreliable (weights/KV cache/graphs may stay resident).
    v1_multiproc_env = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    inproc_v1 = v1_multiproc_env is not None and str(v1_multiproc_env) in (
        "0",
        "false",
        "False",
    )

    if batch_size < 1:
        print("ERROR: --batch-size must be >= 1", file=sys.stderr)
        return 2

    seed = args.get("seed")
    if prompts_file:
        prompts = _load_prompts(str(prompts_file))
    else:
        prompts = _generate_prompts(batch_size=batch_size, seed=seed)

    if len(prompts) != batch_size:
        print(
            f"ERROR: prompts length ({len(prompts)}) != batch_size ({batch_size})",
            file=sys.stderr,
        )
        return 2

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    split_disabled_cfg = _build_split_additional_config(
        enabled=False,
        num_splits=num_splits,
        enable_parallel_streams=enable_parallel_streams,
        min_batch_size_for_split=min_batch_size_for_split,
        parallel_capture_sizes=parallel_capture_sizes,
        force_split=force_split,
    )
    split_enabled_cfg = _build_split_additional_config(
        enabled=True,
        num_splits=num_splits,
        enable_parallel_streams=enable_parallel_streams,
        min_batch_size_for_split=min_batch_size_for_split,
        parallel_capture_sizes=parallel_capture_sizes,
        force_split=force_split,
    )

    # Preferred: coordinator mode spawns 2 child processes then compares.
    if run_mode == "both" and compare_mode == "subprocess":
        return _coordinator_subprocess(
            base_args=args,
            prompts=prompts,
            output_dir_base=output_dir_base,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_splits=num_splits,
            enable_parallel_streams=enable_parallel_streams,
            min_batch_size_for_split=min_batch_size_for_split,
            profile_enabled=profile_enabled,
            profile_dir=profile_dir,
            profile_target=profile_target,
            profile_record_shapes=profile_record_shapes,
            profile_with_stack=profile_with_stack,
        )

    # For all non-coordinator paths: always create an output dir and write all artifacts.
    out_dir, timestamp = _make_run_dir(output_dir_base=output_dir_base, output_file=output_file)
    prompts_path = os.path.join(out_dir, "prompts.json")
    _save_json(prompts_path, prompts)

    metadata = {
        "timestamp": timestamp,
        "model": args.get("model"),
        "tokenizer": args.get("tokenizer") or args.get("model"),
        "seed": seed,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "max_model_len": args.get("max_model_len"),
        "gpu_memory_utilization": args.get("gpu_memory_utilization"),
        "num_splits": num_splits,
        "enable_parallel_streams": enable_parallel_streams,
        "min_batch_size_for_split": min_batch_size_for_split,
        "compilation_config": args.get("compilation_config"),
        "run": run_mode,
        "compare_mode": compare_mode,
        "profile_enabled": profile_enabled,
        "profile_dir": profile_dir,
        "profile_target": profile_target,
        "profile_record_shapes": profile_record_shapes,
        "profile_with_stack": profile_with_stack,
        "output_file": output_file,
        "output_dir": out_dir,
        "notes": {
            "inproc_v1": inproc_v1,
        },
    }
    _save_json(os.path.join(out_dir, "metadata.json"), metadata)

    console_path = os.path.join(out_dir, "console.log")
    os.makedirs(out_dir, exist_ok=True)
    console_f = open(console_path, "a", encoding="utf-8")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _TeeTextIO(old_stdout, console_f)
    sys.stderr = _TeeTextIO(old_stderr, console_f)

    # Re-bind vLLM logger streams after tee so diagnostics land in console.log.
    try:
        vllm_logger_module._configure_vllm_root_logger()
        for _name in ("vllm", "vllm_ascend"):
            _lg = logging.getLogger(_name)
            for _h in _lg.handlers:
                if isinstance(_h, logging.StreamHandler):
                    _h.setStream(sys.stderr)
    except Exception as _e:
        print(f"Warning: failed to rebind vLLM loggers to tee stream: {_e}")

    try:
        print("=== Output ===")
        print("Output dir:", out_dir)
        print("Console log:", console_path)

        print("\n=== Config ===")
        print(
            json.dumps(
                {
                    "model": args.get("model"),
                    "tokenizer": args.get("tokenizer") or args.get("model"),
                    "seed": seed,
                    "max_model_len": args.get("max_model_len"),
                    "gpu_memory_utilization": args.get("gpu_memory_utilization"),
                    "batch_size": batch_size,
                    "max_tokens": max_tokens,
                    "split_batch_config": split_enabled_cfg["split_batch_config"],
                    "compilation_config": args.get("compilation_config"),
                    "run": run_mode,
                    "compare_mode": compare_mode,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        # In-process comparison path or single-run path.
        out0 = None
        out1 = None

        outputs_disabled_path = os.path.join(out_dir, "outputs_split_disabled.json")
        outputs_enabled_path = os.path.join(out_dir, "outputs_split_enabled.json")

        if run_mode in ("both", "disabled"):
            print("\n=== Run: split disabled ===")
            out0, llm0 = _run_single(
                llm_args=args,
                prompts=prompts,
                sampling=sampling,
                additional_config=split_disabled_cfg,
                profile_enabled=profile_enabled,
                profile_dir=profile_dir,
                profile_target=profile_target,
                profile_record_shapes=profile_record_shapes,
                profile_with_stack=profile_with_stack,
                run_name="disabled",
            )

            disabled_payload = {
                "config": "split_disabled",
                "description": CONFIGS["split_disabled"]["description"],
                "split_batch_config": split_disabled_cfg["split_batch_config"],
                "prompts_file": prompts_file,
                "prompts_path": prompts_path,
                "outputs": _serialize_outputs(prompts, out0),
            }
            # Always write results; also honor --output-file if provided.
            _save_json(outputs_disabled_path, disabled_payload)
            if output_file and run_mode == "disabled":
                _save_json(str(output_file), disabled_payload)

            if run_mode == "both":
                if inproc_v1:
                    print(
                        "WARNING: VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-proc) may not support "
                        "running two engines back-to-back reliably due to NPU memory fragmentation.",
                        file=sys.stderr,
                    )
                _cleanup_llm(llm0)

        if run_mode in ("both", "enabled"):
            print("\n=== Run: split enabled ===")
            out1, llm1 = _run_single(
                llm_args=args,
                prompts=prompts,
                sampling=sampling,
                additional_config=split_enabled_cfg,
                profile_enabled=profile_enabled,
                profile_dir=profile_dir,
                profile_target=profile_target,
                profile_record_shapes=profile_record_shapes,
                profile_with_stack=profile_with_stack,
                run_name="enabled",
            )

            enabled_payload = {
                "config": "split_enabled",
                "description": CONFIGS["split_enabled"]["description"],
                "split_batch_config": split_enabled_cfg["split_batch_config"],
                "prompts_file": prompts_file,
                "prompts_path": prompts_path,
                "outputs": _serialize_outputs(prompts, out1),
            }
            # Always write results; also honor --output-file if provided.
            _save_json(outputs_enabled_path, enabled_payload)
            if output_file and run_mode == "enabled":
                _save_json(str(output_file), enabled_payload)

            if run_mode == "both":
                _cleanup_llm(llm1)

            if out0 is None:
                sample_ids, sample_text = _extract_first_output(out1[0])
                print("\nDONE: ran split enabled only")
                print("Sample output[0] token_ids_len=", len(sample_ids))
                print("Sample output[0] text=", repr(sample_text[:200]))
                _save_json(
                    os.path.join(out_dir, "summary.json"),
                    {
                        "status": "DONE",
                        "run": "enabled",
                        "count": len(out1),
                        "sample": {
                            "token_ids_len": len(sample_ids),
                            "text_preview": sample_text[:200],
                        },
                        "outputs_path": outputs_enabled_path,
                        "console_log": console_path,
                    },
                )
                _cleanup_llm(llm1)  # ensure clean shutdown in enabled-only mode
                return 0

        if out1 is None:
            sample_ids, sample_text = _extract_first_output(out0[0])
            print("\nDONE: ran split disabled only")
            print("Sample output[0] token_ids_len=", len(sample_ids))
            print("Sample output[0] text=", repr(sample_text[:200]))
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "DONE",
                    "run": "disabled",
                    "count": len(out0),
                    "sample": {
                        "token_ids_len": len(sample_ids),
                        "text_preview": sample_text[:200],
                    },
                    "outputs_path": outputs_disabled_path,
                    "console_log": console_path,
                },
            )
            _cleanup_llm(llm0)  # ensure clean shutdown in disabled-only mode
            return 0

        if len(out0) != len(out1):
            msg = f"FAIL: output length differs: {len(out0)} vs {len(out1)}"
            print(msg, file=sys.stderr)
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "FAIL",
                    "reason": "output length differs",
                    "disabled_len": len(out0),
                    "enabled_len": len(out1),
                    "outputs_disabled_path": outputs_disabled_path,
                    "outputs_enabled_path": outputs_enabled_path,
                    "console_log": console_path,
                },
            )
            return 1

        # Compare and write diff if needed.
        disabled_rows = _serialize_outputs(prompts, out0)
        enabled_rows = _serialize_outputs(prompts, out1)
        mismatches = _compare_serialized(disabled_rows, enabled_rows)

        if mismatches:
            diff_path = os.path.join(out_dir, "diff.json")
            _save_json(diff_path, {"mismatches": mismatches[:50], "total": len(mismatches)})
            print(f"\nFAIL: {len(mismatches)}/{len(prompts)} mismatches", file=sys.stderr)
            print(json.dumps(mismatches[:3], ensure_ascii=False, indent=2), file=sys.stderr)
            if len(mismatches) > 3:
                print(f"... (and {len(mismatches) - 3} more)", file=sys.stderr)

            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "FAIL",
                    "mismatch_total": len(mismatches),
                    "diff_path": diff_path,
                    "outputs_disabled_path": outputs_disabled_path,
                    "outputs_enabled_path": outputs_enabled_path,
                    "console_log": console_path,
                },
            )
            return 1

        print(f"\nPASS: {len(prompts)}/{len(prompts)} outputs match exactly")
        sample_ids, sample_text = _extract_first_output(out1[0])
        print("Sample output[0] token_ids_len=", len(sample_ids))
        print("Sample output[0] text=", repr(sample_text[:200]))

        _save_json(
            os.path.join(out_dir, "summary.json"),
            {
                "status": "PASS",
                "count": len(prompts),
                "sample": {
                    "token_ids_len": len(sample_ids),
                    "text_preview": sample_text[:200],
                },
                "outputs_disabled_path": outputs_disabled_path,
                "outputs_enabled_path": outputs_enabled_path,
                "console_log": console_path,
            },
        )
        return 0

    except Exception as e:
        try:
            _save_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "status": "ERROR",
                    "error": repr(e),
                    "console_log": console_path,
                },
            )
        except Exception:
            pass
        raise
    finally:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        finally:
            try:
                console_f.flush()
                console_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())