#!/usr/bin/env bash
set -euo pipefail
# Quick serve check against the head. NOT a gate: a config that answers a short
# prompt is not a config that works — ladder KV only through the gate suite (README).
#
#   scripts/smoke-test.sh [head-host]     # default: localhost
HOST="${1:-127.0.0.1}"
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env" >&2; exit 3; }
set -a; . ./.env; set +a
BASE="http://$HOST:$PORT"

echo ">> /health"
curl -sf "$BASE/health" >/dev/null && echo "   200 OK"

echo ">> coherence (greedy, short)"
R1="$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED_NAME\", \"temperature\": 0, \"max_tokens\": 60,
  \"messages\": [{\"role\":\"user\",\"content\":\"Name the four largest moons of Jupiter, comma-separated, nothing else.\"}]}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')"
echo "   $R1"
case "$R1" in *Ganymede*|*ganymede*) echo "   coherent" ;; *) echo "   SUSPECT ANSWER" >&2; exit 1 ;; esac

echo ">> determinism (same greedy request twice)"
Q='{"model":"'"$SERVED_NAME"'","temperature":0,"max_tokens":80,"messages":[{"role":"user","content":"Write a two-line haiku-style summary of tensor parallelism."}]}'
A="$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$Q" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"
B="$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$Q" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"
[ "$A" = "$B" ] && echo "   identical" || echo "   DIFFERENT (expected identical with EXACT_TOPK=1)" >&2

echo ">> single-stream decode speed (real answer, no ignore_eos)"
python3 - "$BASE" "$SERVED_NAME" <<'EOF'
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "temperature": 0, "max_tokens": 400,
  "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists, then explain it briefly."}]}).encode()
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    base + "/v1/chat/completions", body, {"Content-Type": "application/json"})))
dt = time.time() - t0
toks = r["usage"]["completion_tokens"]
print(f"   {toks} tokens in {dt:.1f}s = {toks/dt:.1f} tok/s (single stream; the prompt matters — quote it with the number)")
EOF

echo ">> smoke OK"
