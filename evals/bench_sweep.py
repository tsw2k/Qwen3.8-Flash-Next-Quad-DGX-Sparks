#!/usr/bin/env python3
"""Concurrency sweep for the rung-3 numbers. Reports, per concurrency level:
aggregate tok/s, per-stream tok/s, and TTFT (time to first streamed token).

Rules this script enforces about honest numbers:
  - real prompts and real answers (no ignore_eos — it produces meaningless
    numbers with this model);
  - the prompt set is printed with the results, because speculative-decode
    tok/s is a property of the prompt as much as the config;
  - check --max-num-seqs on the server first: a low cap makes saturation and
    queuing indistinguishable in tok/s (jschmied's finding).

Usage:
  evals/bench_sweep.py --base http://HEAD:8000 --model qwen3.8-flash-next \
      [--levels 1,2,4,8,16,32] [--max-tokens 300] [--json out.json]

Stdlib only.
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

PROMPTS = [
    # structured / code — high draft acceptance
    "Write a Python function that merges two sorted lists, then explain it briefly.",
    "Write a bash script that checks whether four hosts are reachable over ssh and prints a table.",
    # freeform prose — low draft acceptance
    "Describe an autumn morning in a mountain village, about 250 words.",
    "Explain to a curious teenager why the sky is blue, conversationally.",
    # agentic / mixed
    "You are debugging a server that answers HTTP 200 but returns garbage after 24k tokens. List hypotheses and an ordered test plan.",
    "Summarize the trade-offs between tensor and pipeline parallelism for a 2-node deployment, with a recommendation.",
]


def stream_one(base, model, prompt, max_tokens):
    body = json.dumps({
        "model": model, "temperature": 0.7, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
            for ch in chunk.get("choices", []):
                if ttft is None and (ch.get("delta", {}).get("content") or
                                     ch.get("delta", {}).get("reasoning_content")):
                    ttft = time.time() - t0
    total = time.time() - t0
    return {"ttft": ttft or total, "total": total, "tokens": completion_tokens}


def run_level(base, model, level, max_tokens):
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (run tag {i})" for i in range(level)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(level) as ex:
        results = list(ex.map(lambda p: stream_one(base, model, p, max_tokens), prompts))
    wall = time.time() - t0
    toks = sum(r["tokens"] for r in results)
    return {
        "concurrency": level,
        "wall_s": round(wall, 2),
        "total_tokens": toks,
        "aggregate_tok_s": round(toks / wall, 1),
        "per_stream_tok_s": round(sum(r["tokens"] / r["total"] for r in results) / level, 1),
        "ttft_avg_s": round(sum(r["ttft"] for r in results) / level, 2),
        "ttft_max_s": round(max(r["ttft"] for r in results), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--json", default=None, help="also write results as JSON")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"# bench sweep — {args.model} @ {args.base}, max_tokens={args.max_tokens}")
    print("# prompts: mixed structured/prose/agentic (see PROMPTS in this file); "
          "quote them with any number you publish.")
    rows = []
    for level in levels:
        # warm the level with one throwaway request at c=1 before the first level only
        if not rows:
            stream_one(args.base, args.model, "Say 'warm'.", 10)
        r = run_level(args.base, args.model, level, args.max_tokens)
        rows.append(r)
        print(f"x{r['concurrency']:<3} aggregate {r['aggregate_tok_s']:>7} tok/s   "
              f"per-stream {r['per_stream_tok_s']:>6} tok/s   "
              f"TTFT avg {r['ttft_avg_s']}s max {r['ttft_max_s']}s   "
              f"({r['total_tokens']} toks / {r['wall_s']}s)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config_note": "record launcher config alongside this",
                       "prompts": PROMPTS, "results": rows}, f, indent=2)
        print(f"# wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
