#!/usr/bin/env python3
# isort: skip_file
"""Benchmark: DECODE TPOT and ACLgraph replay header overhead.

Two modes per run (select via --mode):
  eager    : cudagraph_mode=NONE, no split-batch
  aclgraph : cudagraph_mode=FULL_DECODE_ONLY + split-batch parallel streams
             (force_split=True so every batch size exercises the split path)

Metrics written per decode step to VLLM_ASCEND_PERF_STATS_FILE (JSONL):
  replay_ms  : time from replay-worker launch to stream sync done
               = TPOT for a single request (1 token/step in uniform decode)
  header_ms  : time from execute_model entry to replay start
               = ACLgraph replay header overhead

Dataset : LongBench-v2 (data.json), context field used as prompt text.
Seq-len : random in [--min-seq-len, --max-seq-len] (default 1000-2000).
Max tokens: fixed 200, ignore_eos=True.
"""

import gc
import json
import os
import random
import time
from typing import Any, Optional

# ── Environment (before vLLM imports) ────────────────────────────────────────
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ASCEND_ACLGRAPH_ADDR_CHECK", "1")
os.environ.setdefault("VLLM_ASCEND_SPLIT_REPLAY_TRACE", "1")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
MAIN_CAPTURE_SIZES: list[int] = (
    [1, 2, 4, 8, 16, 32, 64] + [i * 128 for i in range(1, 5)]
)  # [1,2,4,8,16,32,64,128,256,384,512]

PARALLEL_CAPTURE_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

BATCH_MATRIX: list[tuple[int, str, Optional[int]]] = [
    (32,  "hit",  32),  (32,  "near", 40),  (32,  "mid",  48),  (32,  "next", 60),
    (64,  "hit",  64),  (64,  "near", 72),  (64,  "mid",  96),  (64,  "next", 120),
    (128, "hit",  128), (128, "near", 144), (128, "mid",  192), (128, "next", 240),
    (256, "hit",  256), (256, "near", 272), (256, "mid",  320), (256, "next", 368),
    (384, "hit",  384), (384, "near", 400), (384, "mid",  448), (384, "next", 496),
    (512, "hit",  512),
]

EXPERIMENT_MODELS: dict[str, str] = {
    "1": "Qwen/Qwen3-4B",
    "2": "Qwen/Qwen3-0.6B",
}

# ── Parser ────────────────────────────────────────────────────────────────────

def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description=__doc__)
    # Add all standard EngineArgs (model, tokenizer, compilation_config, etc.)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(trust_remote_code=True)
    parser.set_defaults(max_model_len=16384)
    parser.set_defaults(enable_chunked_prefill=False)
    parser.set_defaults(enable_prefix_caching=False)
    # Allow up to 512 concurrent sequences; batched-token budget covers
    # 512 reqs × 2000 tokens (max seq-len) for the prefill phase.
    parser.set_defaults(max_num_seqs=512)
    parser.set_defaults(max_num_batched_tokens=270000)
    parser.set_defaults(gpu_memory_utilization=0.9)

    # Default compilation_config — mode will be overridden in main() based on --mode
    parser.set_defaults(compilation_config={
        "level": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": MAIN_CAPTURE_SIZES,
    })

    bench = parser.add_argument_group("Benchmark parameters")
    bench.add_argument(
        "--mode",
        choices=["eager", "aclgraph"],
        default="aclgraph",
        help="eager=NONE (no graph), aclgraph=FULL_DECODE_ONLY+split parallel.",
    )
    bench.add_argument(
        "--experiment",
        choices=["1", "2"],
        default="1",
        help="1=Qwen3-4B, 2=Qwen3-0.6B. Sets default model.",
    )
    bench.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes. Overrides built-in BATCH_MATRIX.",
    )
    bench.add_argument(
        "--dataset-path",
        type=str,
        default="/vllm-workspace/LongBench-v2/data.json",
        help="Path to LongBench-v2 data.json.",
    )
    bench.add_argument("--max-tokens", type=int, default=200,
                       help="Fixed generation length per request.")
    bench.add_argument("--min-seq-len", type=int, default=500)
    bench.add_argument("--max-seq-len", type=int, default=500)
    bench.add_argument(
        "--perf-stats-file",
        type=str,
        default="/tmp/benchmark_split_perf_stats.jsonl",
        help="Path for per-step perf stats (VLLM_ASCEND_PERF_STATS_FILE).",
    )
    bench.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_split_results",
        help="Directory to save results JSON.",
    )
    bench.add_argument("--bench-seed", dest="bench_seed", type=int, default=42,
                       help="Random seed for seq-len and prompt sampling.")
    return parser


# ── LLM helpers (same pattern as test_split_batch_correctness_npu.py) ─────────

def _build_llm_from_args(
    llm_args: dict[str, Any],
    *,
    additional_config: dict[str, Any],
) -> LLM:
    args = dict(llm_args)
    args["additional_config"] = additional_config
    return LLM(**args)


def _build_aclgraph_additional_config() -> dict[str, Any]:
    """additional_config for FULL_DECODE_ONLY + parallel-stream split-batch."""
    return {
        "split_batch_config": {
            "enabled": True,
            "enable_parallel_streams": True,
            "force_split": True,
            "parallel_capture_sizes": PARALLEL_CAPTURE_SIZES,
            "num_splits": 2,
            "min_batch_size_for_split": 1,
        }
    }


def _build_eager_additional_config() -> dict[str, Any]:
    return {}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[dataset] loaded {len(data)} samples from {path}")
    return data


def _get_context(sample: dict) -> str:
    for key in ("context", "input", "question"):
        if key in sample and sample[key]:
            return str(sample[key])
    return str(list(sample.values())[0])


def _truncate_or_repeat(text: str, target_tokens: int,
                         avg_chars: float = 4.0) -> str:
    target_chars = max(1, int(target_tokens * avg_chars))
    if len(text) >= target_chars:
        return text[:target_chars]
    repeated = text * (target_chars // max(1, len(text)) + 1)
    return repeated[:target_chars]


def _truncate_by_tokens(text: str, target_tokens: int, tokenizer) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= target_tokens:
        return text
    return tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)


def prepare_prompts(dataset: list[dict], batch_size: int,
                    seq_len: int, rng: random.Random,
                    tokenizer=None) -> list[str]:
    prompts = []
    for _ in range(batch_size):
        sample = dataset[rng.randint(0, len(dataset) - 1)]
        if tokenizer is not None:
            text = _truncate_by_tokens(_get_context(sample), seq_len, tokenizer)
        else:
            text = _truncate_or_repeat(_get_context(sample), seq_len)
        prompts.append(f"{text}\n\nPlease summarize the above text briefly:")
    return prompts


# ── Perf stats reader ─────────────────────────────────────────────────────────

def clear_perf_stats(path: str) -> None:
    try:
        open(path, "w").close()
    except Exception:
        pass


def read_perf_stats(path: str) -> dict[str, Any]:
    """Aggregate all decode-step records from the JSONL file.

    TPOT = avg replay_ms per step.
    Each uniform-decode step produces exactly 1 token per request, so
    replay_ms is already the per-token latency for a single request.
    """
    lines: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass

    if not lines:
        return {
            "num_steps": 0,
            "tpot_ms": float("nan"),
            "avg_header_ms": float("nan"),
            "split_steps": 0,
        }

    n = len(lines)
    return {
        "num_steps": n,
        "tpot_ms": sum(l.get("replay_ms", 0.0) for l in lines) / n,
        "avg_header_ms": sum(l.get("header_ms", 0.0) for l in lines) / n,
        "split_steps": sum(1 for l in lines if l.get("is_split", False)),
    }


# ── Single test case ──────────────────────────────────────────────────────────

def run_one_case(
    llm: LLM,
    dataset: list[dict],
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    max_tokens: int,
    perf_stats_file: str,
    rng: random.Random,
    graph_size: int,
    case_label: str,
    tokenizer=None,
) -> dict[str, Any]:
    seq_len = rng.randint(min_seq_len, max_seq_len)
    prompts = prepare_prompts(dataset, batch_size, seq_len, rng, tokenizer)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                              ignore_eos=True)

    clear_perf_stats(perf_stats_file)
    t0 = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        success = True
        error = None
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        total_out = 0
        success = False
        error = str(exc)
        import traceback
        traceback.print_exc()
    finally:
        # Always abort any pending requests so they don't bleed into the next
        # case. This handles both exception paths and cases where generate()
        # returns early (e.g. steps=0 due to KV cache exhaustion).
        try:
            engine = llm.llm_engine
            op = getattr(engine, "output_processor", None)
            if op is not None and hasattr(op, "request_states"):
                pending_ids = list(op.request_states.keys())
                if pending_ids:
                    engine.abort_request(pending_ids)
                    print(f"  [CLEANUP] aborted {len(pending_ids)} pending requests")
        except Exception as cleanup_exc:
            print(f"  [CLEANUP] abort failed: {cleanup_exc}")

    stats = read_perf_stats(perf_stats_file)
    result = {
        "graph_size": graph_size,
        "case": case_label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_output_tokens": total_out,
        "elapsed_s": round(elapsed, 3),
        "tpot_ms": round(stats["tpot_ms"], 3),
        "avg_header_ms": round(stats["avg_header_ms"], 3),
        "num_decode_steps": stats["num_steps"],
        "split_steps": stats["split_steps"],
        "success": success,
        "error": error,
    }
    status = "OK" if success else "FAIL"
    print(
        f"  [{status}] S={graph_size:<4} {case_label:<5} bs={batch_size:<4} "
        f"seq={seq_len}  TPOT={stats['tpot_ms']:.3f}ms  "
        f"header={stats['avg_header_ms']:.3f}ms  "
        f"steps={stats['num_steps']}(split={stats['split_steps']})"
    )
    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], mode: str, model: str) -> None:
    print(f"\n{'=' * 95}")
    print(f"Summary  mode={mode}  model={model}")
    print(f"{'=' * 95}")
    print(f"{'S':>6} {'case':<5} {'bs':>5} {'TPOT(ms)':>10} "
          f"{'header(ms)':>12} {'steps':>7} {'split':>7} {'status'}")
    print("-" * 95)
    for r in results:
        tpot = f"{r['tpot_ms']:.3f}" if r["success"] else "—"
        hdr  = f"{r['avg_header_ms']:.3f}" if r["success"] else "—"
        print(f"{r['graph_size']:>6} {r['case']:<5} {r['batch_size']:>5} "
              f"{tpot:>10} {hdr:>12} "
              f"{r['num_decode_steps']:>7} {r['split_steps']:>7} "
              f"{'OK' if r['success'] else 'FAIL'}")


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_llm(llm: LLM) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    finally:
        try:
            import torch
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
                torch.npu.empty_cache()
        except Exception:
            pass
        gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = create_parser()
    args: dict[str, Any] = vars(parser.parse_args())

    # Pop benchmark-specific args (not EngineArgs)
    mode         = args.pop("mode")
    experiment   = args.pop("experiment")
    batch_sizes_arg = args.pop("batch_sizes")
    dataset_path = args.pop("dataset_path")
    max_tokens   = args.pop("max_tokens")
    min_seq_len  = args.pop("min_seq_len")
    max_seq_len  = args.pop("max_seq_len")
    perf_stats_file = args.pop("perf_stats_file")
    output_dir   = args.pop("output_dir")
    bench_seed   = args.pop("bench_seed")

    # Set default model from experiment (can be overridden by --model)
    if args.get("model") is None:
        args["model"] = EXPERIMENT_MODELS[experiment]

    # Override compilation_config cudagraph_mode based on --mode
    cc = dict(args.get("compilation_config") or {})
    if mode == "eager":
        cc["cudagraph_mode"] = "NONE"
    else:
        cc["cudagraph_mode"] = "FULL_DECODE_ONLY"
        cc["cudagraph_capture_sizes"] = MAIN_CAPTURE_SIZES
    args["compilation_config"] = cc

    # Set VLLM_ASCEND_PERF_STATS_FILE so model_runner writes timing data
    os.environ["VLLM_ASCEND_PERF_STATS_FILE"] = perf_stats_file

    rng = random.Random(bench_seed)

    # Determine test cases
    if batch_sizes_arg:
        test_cases = [(0, "custom", int(s.strip()))
                      for s in batch_sizes_arg.split(",") if s.strip()]
    else:
        test_cases = [(gs, lbl, bs) for gs, lbl, bs in BATCH_MATRIX
                      if bs is not None]

    # max_num_seqs defaults to 512 (set in parser); user can override via CLI.
    # Warn if the requested max batch size exceeds max_num_seqs.
    max_bs = max(bs for _, _, bs in test_cases)
    if (args.get("max_num_seqs") or 0) < max_bs:
        print(f"[WARNING] max_num_seqs={args.get('max_num_seqs')} < "
              f"max batch_size={max_bs}. Consider passing "
              f"--max-num-seqs {max_bs}")

    # Build additional_config
    if mode == "eager":
        additional_config = _build_eager_additional_config()
    else:
        additional_config = _build_aclgraph_additional_config()

    print(f"\n{'=' * 70}")
    print(f"Benchmark: mode={mode}  experiment={experiment}  model={args['model']}")
    print(f"  capture_sizes (main): {MAIN_CAPTURE_SIZES}")
    if mode == "aclgraph":
        print(f"  capture_sizes (parallel): {PARALLEL_CAPTURE_SIZES}")
        print(f"  additional_config: {additional_config}")
    print(f"  max_tokens={max_tokens}  seq_len=[{min_seq_len},{max_seq_len}]")
    print(f"  test cases: {len(test_cases)}  max_num_seqs={args.get('max_num_seqs')}  "
          f"max_num_batched_tokens={args.get('max_num_batched_tokens')}")
    print(f"  perf_stats_file: {perf_stats_file}")
    print(f"{'=' * 70}")

    dataset = load_dataset(dataset_path)

    from transformers import AutoTokenizer
    print("Loading tokenizer for prompt truncation...")
    _tokenizer = AutoTokenizer.from_pretrained(args["model"], trust_remote_code=True)

    print("\nInitializing LLM...")
    llm = _build_llm_from_args(args, additional_config=additional_config)

    results: list[dict] = []
    for graph_size, case_label, batch_size in test_cases:
        result = run_one_case(
            llm=llm,
            dataset=dataset,
            batch_size=batch_size,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            max_tokens=max_tokens,
            perf_stats_file=perf_stats_file,
            rng=rng,
            graph_size=graph_size,
            case_label=case_label,
            tokenizer=_tokenizer,
        )
        result["mode"] = mode
        result["model"] = args["model"]
        result["experiment"] = experiment
        results.append(result)

    print_summary(results, mode=mode, model=args["model"])

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir,
                            f"results_{mode}_exp{experiment}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    cleanup_llm(llm)


if __name__ == "__main__":
    main()

