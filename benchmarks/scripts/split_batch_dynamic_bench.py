#!/usr/bin/env python3
"""Dynamic batch/sequence-length benchmark for split-batch on vLLM serve.

This script targets the acceptance scenarios:
- Batch size dynamically changes in [1, 256]
- Batch size in [1, 256] AND input sequence length in [1k, 16k] tokens

It drives concurrency against an OpenAI-compatible endpoint (vLLM serve),
measuring:
- TTFT: time to first token (streaming)
- Decode TPOT: (t_end - t_first_token) / output_tokens

Notes:
- "Batch size" here is approximated via concurrent requests arriving together.
- Decode TPOT is estimated from streaming timestamps and tokenization.
- For best comparability, keep server-side settings fixed across runs.

Example:
  # Start server (in another terminal):
  vllm serve Qwen/Qwen3-8B --max-model-len 16384 \
    --additional-config '{"split_batch_config": {"enabled": true, "num_splits": 2, "enable_parallel_streams": true, "min_batch_size_for_split": 4}}'

  # Run batch-only scenario
  python benchmarks/scripts/split_batch_dynamic_bench.py \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
    --scenario bs_only --iters 50 --max-new-tokens 128 \
    --tokenizer Qwen/Qwen3-8B --out results_split_bs_only.json

  # Run batch+seqlen scenario
  python benchmarks/scripts/split_batch_dynamic_bench.py \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
    --scenario bs_and_seqlen --iters 50 --max-new-tokens 128 \
    --tokenizer Qwen/Qwen3-8B --out results_split_bs_seqlen.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for token-accurate prompt/output lengths. "
        "Please install it in your environment."
    ) from e


@dataclass
class RequestMetrics:
    ok: bool
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    t_start: float
    t_first_token: Optional[float]
    t_end: Optional[float]

    @property
    def ttft_s(self) -> Optional[float]:
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.t_start

    @property
    def decode_tpot_s(self) -> Optional[float]:
        if self.t_first_token is None or self.t_end is None:
            return None
        if self.output_tokens <= 0:
            return None
        return (self.t_end - self.t_first_token) / self.output_tokens


def _health_check(base_url: str, timeout_s: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _load_jsonl_texts(path: Path, max_lines: Optional[int] = None) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    texts.append(obj)
                    continue
                if isinstance(obj, dict):
                    for key in ("prompt", "input", "question", "text", "instruction"):
                        if key in obj and isinstance(obj[key], str) and obj[key].strip():
                            texts.append(obj[key])
                            break
                    else:
                        texts.append(line)
                else:
                    texts.append(line)
            except Exception:
                texts.append(line)
    if not texts:
        raise ValueError(f"No usable lines found in dataset: {path}")
    return texts


def _make_text_with_exact_tokens(tokenizer, base_text: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        base_ids = tokenizer.encode("hello", add_special_tokens=False)

    ids: List[int] = []
    while len(ids) < target_tokens:
        ids.extend(base_ids)

    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _openai_chat_payload(model: str, prompt: str, max_new_tokens: int, stream: bool) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "stream": bool(stream),
    }


def _stream_chat_completion(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    tokenizer,
    barrier,
    timeout_s: float,
) -> RequestMetrics:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    try:
        barrier.wait()
    except Exception:
        # If barrier is broken, still try to proceed.
        pass

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    out_text_parts: List[str] = []

    try:
        with session.post(
            url,
            headers=headers,
            data=json.dumps(_openai_chat_payload(model, prompt, max_new_tokens, stream=True)),
            stream=True,
            timeout=timeout_s,
        ) as resp:
            if resp.status_code != 200:
                return RequestMetrics(
                    ok=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:2000]}",
                    input_tokens=input_tokens,
                    output_tokens=0,
                    t_start=t0,
                    t_first_token=None,
                    t_end=None,
                )

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    t_end = time.perf_counter()
                    break
                try:
                    payload = json.loads(data)
                except Exception:
                    continue

                # vLLM uses OpenAI streaming schema.
                delta = (
                    payload.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if isinstance(delta, str) and delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    out_text_parts.append(delta)

        if t_end is None:
            t_end = time.perf_counter()

        out_text = "".join(out_text_parts)
        output_tokens = len(tokenizer.encode(out_text, add_special_tokens=False))
        return RequestMetrics(
            ok=True,
            error=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            t_start=t0,
            t_first_token=t_first,
            t_end=t_end,
        )

    except Exception as e:
        return RequestMetrics(
            ok=False,
            error=str(e),
            input_tokens=input_tokens,
            output_tokens=0,
            t_start=t0,
            t_first_token=None,
            t_end=None,
        )


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        k = int(round((p / 100.0) * (n - 1)))
        return values_sorted[max(0, min(n - 1, k))]

    return {
        "count": float(n),
        "mean": float(sum(values_sorted) / n),
        "p50": float(pct(50)),
        "p90": float(pct(90)),
        "p99": float(pct(99)),
        "min": float(values_sorted[0]),
        "max": float(values_sorted[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="token-abc123", help="vLLM OpenAI server usually ignores this")
    ap.add_argument("--model", required=True, help="Model name as seen by server")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer name/path (default: --model)")

    ap.add_argument("--scenario", choices=["bs_only", "bs_and_seqlen"], required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup-iters", type=int, default=5)

    ap.add_argument("--bs-min", type=int, default=1)
    ap.add_argument("--bs-max", type=int, default=256)
    ap.add_argument("--seqlen-min", type=int, default=1024)
    ap.add_argument("--seqlen-max", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=128)

    ap.add_argument("--dataset-jsonl", default=None, help="Optional JSONL file (e.g. CLongEval) for base texts")
    ap.add_argument("--dataset-max-lines", type=int, default=2000)

    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out", required=True, help="Output JSON path")

    args = ap.parse_args()

    random.seed(args.seed)

    if not _health_check(args.base_url):
        print(f"ERROR: server health check failed: {args.base_url}/health", file=sys.stderr)
        return 2

    tok_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    base_texts: List[str]
    if args.dataset_jsonl:
        base_texts = _load_jsonl_texts(Path(args.dataset_jsonl), max_lines=args.dataset_max_lines)
    else:
        base_texts = [
            "You are a helpful assistant. "
            "Please answer concisely. "
            "Context: " + ("A " * 2048)
        ]

    def sample_base_text() -> str:
        return random.choice(base_texts)

    # Validate worker capacity for worst-case batch size.
    # We use threads to approximate simultaneous arrivals.
    max_workers = args.bs_max

    results: Dict[str, Any] = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "tokenizer": tok_name,
            "scenario": args.scenario,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
            "bs_range": [args.bs_min, args.bs_max],
            "seqlen_range": [args.seqlen_min, args.seqlen_max],
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "requests": [],
        "summary": {},
    }

    def run_one_iteration(iter_idx: int, batch_size: int, seqlen_tokens: int) -> List[RequestMetrics]:
        prompts = [
            _make_text_with_exact_tokens(tokenizer, sample_base_text(), seqlen_tokens)
            for _ in range(batch_size)
        ]
        barrier = __import__("threading").Barrier(batch_size)

        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            session_factory = lambda: requests.Session()

            futures = []
            for prompt in prompts:
                sess = session_factory()
                futures.append(
                    ex.submit(
                        _stream_chat_completion,
                        sess,
                        args.base_url,
                        args.api_key,
                        args.model,
                        prompt,
                        args.max_new_tokens,
                        tokenizer,
                        barrier,
                        args.timeout_s,
                    )
                )

            out: List[RequestMetrics] = []
            for fut in as_completed(futures):
                out.append(fut.result())
            return out

    # Warmup
    for i in range(args.warmup_iters):
        bs = max(1, min(args.bs_max, args.bs_min))
        seqlen = args.seqlen_min if args.scenario == "bs_and_seqlen" else args.seqlen_min
        _ = run_one_iteration(i, bs, seqlen)

    # Main
    for iter_idx in range(args.iters):
        bs = random.randint(args.bs_min, args.bs_max)
        if args.scenario == "bs_only":
            seqlen = args.seqlen_min
        else:
            seqlen = random.randint(args.seqlen_min, args.seqlen_max)

        metrics_list = run_one_iteration(iter_idx, bs, seqlen)

        for m in metrics_list:
            results["requests"].append(
                {
                    "iter": iter_idx,
                    "batch_size": bs,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "ok": m.ok,
                    "error": m.error,
                    "ttft_s": m.ttft_s,
                    "decode_tpot_s": m.decode_tpot_s,
                }
            )

    ttft_vals = [r["ttft_s"] for r in results["requests"] if r.get("ttft_s") is not None and r.get("ok")]
    tpot_vals = [r["decode_tpot_s"] for r in results["requests"] if r.get("decode_tpot_s") is not None and r.get("ok")]
    ok_count = sum(1 for r in results["requests"] if r.get("ok"))
    fail_count = len(results["requests"]) - ok_count

    results["summary"] = {
        "ok": ok_count,
        "failed": fail_count,
        "ttft_s": _summarize(ttft_vals),
        "decode_tpot_s": _summarize(tpot_vals),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
