#!/usr/bin/env python3
"""Gate suite for the TP4 boot ladder. A config that boots and answers a short
prompt is not a config that works — these gates are what "works" means here:

  G1 deep-decode     long prompt AND a long forced answer (>=150 completion
                     tokens), prompt varied per run so the prefix cache cannot
                     turn the gate into a no-op
  G2 concurrent      3 overlapping ~30k-token prefills (the gate that killed
                     the 32 GiB GLM config after single-prefill passed)
  G3 determinism     identical greedy request twice, byte-identical answers
                     (EXACT_TOPK=1 contract)
  G4 niah            passkey retrieval at 0/50/100% depth for each --niah size
  G5 health          /health stays 200 through all of the above

Usage:
  evals/gate_suite.py --base http://HEAD:8000 --model qwen3.8-flash-next \
      --niah 4096,32768,131072 [--seed 1] [--skip niah]

Exit 0 = all gates passed. Stdlib only; no external deps.
"""

import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = ("fabric rank tensor spark rail switch kernel cache token expert "
         "gate head path pool shard stream prefix decode prefill draft").split()


def chat(base, model, messages, max_tokens, temperature=0.0, timeout=1800):
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        # thinking off: gates measure the deterministic no-reasoning path,
        # and with thinking on the content field can be None entirely
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.time() - t0
    msg = out["choices"][0]["message"]["content"] or ""
    usage = out.get("usage", {})
    return msg, usage, dt


def health(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def haystack(rng, n_tokens, passkey, depth):
    """Synthetic filler with a passkey sentence planted at `depth` (0..1).
    Filler is ~1 token/word; sizes are approximate, which is fine for a gate."""
    words = [rng.choice(WORDS) for _ in range(n_tokens)]
    sentence = f" The secret passkey is {passkey}. Remember it. "
    pos = min(len(words) - 1, int(len(words) * depth))
    words.insert(pos, sentence)
    return " ".join(words)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def g1_deep_decode(base, model, rng):
    print(">> G1 deep-decode (long prompt, forced long answer)")
    nonce = rng.randrange(10**9)
    prompt = (haystack(rng, 8000, nonce, 0.5)
              + "\n\nSummarize what kind of text this was in about 200 words, "
                f"then state the passkey. Session nonce: {nonce}.")
    msg, usage, dt = chat(base, model, [{"role": "user", "content": prompt}], 600)
    toks = usage.get("completion_tokens", 0)
    ok = toks >= 150 and str(nonce) in msg
    return gate("deep-decode", ok,
                f"{toks} completion tokens in {dt:.0f}s, passkey {'found' if str(nonce) in msg else 'MISSING'}")


def g2_concurrent_prefill(base, model, rng, n=3, size=30000):
    print(f">> G2 concurrent prefills ({n} x ~{size} tokens)")
    def one(i):
        key = rng.randrange(10**9)
        prompt = haystack(random.Random(i * 7919 + key), size, key, 0.5) \
            + "\nWhat is the secret passkey? Answer with the number only."
        msg, _, dt = chat(base, model, [{"role": "user", "content": prompt}], 50)
        return str(key) in msg, dt
    with cf.ThreadPoolExecutor(n) as ex:
        results = list(ex.map(one, range(n)))
    ok = all(r for r, _ in results)
    return gate("concurrent-prefill", ok,
                " / ".join(f"{'ok' if r else 'WRONG'} {dt:.0f}s" for r, dt in results))


def g3_determinism(base, model, rng):
    print(">> G3 determinism (same greedy request twice)")
    prompt = f"Write exactly three sentences about RoCEv2. Nonce {rng.randrange(10**9)}."
    a, _, _ = chat(base, model, [{"role": "user", "content": prompt}], 200)
    b, _, _ = chat(base, model, [{"role": "user", "content": prompt}], 200)
    return gate("determinism", a == b, "identical" if a == b else "outputs differ")


def g4_niah(base, model, rng, size):
    print(f">> G4 NIAH {size}")
    results = []
    for depth in (0.0, 0.5, 1.0):
        key = rng.randrange(10**9)
        prompt = haystack(rng, size, key, depth) \
            + "\nWhat is the secret passkey mentioned above? Answer with the number only."
        try:
            msg, _, dt = chat(base, model, [{"role": "user", "content": prompt}], 50)
            results.append((depth, str(key) in msg, f"{dt:.0f}s"))
        except Exception as e:  # noqa: BLE001 — a gate must report, not crash
            results.append((depth, False, type(e).__name__))
    ok = all(r for _, r, _ in results)
    return gate(f"niah-{size}", ok,
                " ".join(f"{int(d*100)}%:{'ok' if r else 'FAIL'}({x})" for d, r, x in results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--niah", default="4096,32768",
                    help="comma-separated approximate prompt sizes in tokens")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: time-based, so every run varies (defeats prefix cache)")
    ap.add_argument("--skip", default="", help="comma-separated gate names: deep,concurrent,determinism,niah")
    args = ap.parse_args()
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    skip = set(args.skip.split(",")) if args.skip else set()

    if not health(args.base):
        print("FAIL: /health not 200 before gating"); return 1
    passed = True
    if "deep" not in skip:
        passed &= g1_deep_decode(args.base, args.model, rng)
    if "concurrent" not in skip:
        passed &= g2_concurrent_prefill(args.base, args.model, rng)
    if "determinism" not in skip:
        passed &= g3_determinism(args.base, args.model, rng)
    if "niah" not in skip:
        for size in (int(s) for s in args.niah.split(",")):
            passed &= g4_niah(args.base, args.model, rng, size)
    ok_health = health(args.base)
    passed &= gate("health-after", ok_health, "200" if ok_health else "server unhealthy after gates")

    print(">> GATES " + ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
