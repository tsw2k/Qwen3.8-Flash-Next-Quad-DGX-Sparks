#!/usr/bin/env bash
set -euo pipefail
# Download the checkpoint on THIS node, then rsync to the rest. Resumable both ways.
#
# The checkpoint (~122 GiB) must end up on LOCAL NVMe on EVERY node: the PLE n-gram
# table is mmapped and read at runtime, and mmap over NFS is how you turn a 2.5 KB
# lookup into a network stall. One HF download + three rsyncs also beats four nodes
# hitting HF concurrently.
#
# Run from the repo checkout on the head node: scripts/sync-weights.sh
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env — cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

HF_REPO="${HF_REPO:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
SELF="$(hostname -s)"

echo ">> downloading $HF_REPO -> $MODEL_HOST_PATH (resumable)"
mkdir -p "$MODEL_HOST_PATH"
hf download "$HF_REPO" --local-dir "$MODEL_HOST_PATH"

test -f "$MODEL_HOST_PATH/config.json" || { echo "download incomplete: no config.json" >&2; exit 1; }

for host in $SSH_HOSTS; do
  case "$host" in "$SELF") continue ;; esac
  echo ">> rsync -> $host:$MODEL_HOST_PATH"
  ssh "$host" "mkdir -p '$MODEL_HOST_PATH'"
  rsync -a --info=progress2 --partial "$MODEL_HOST_PATH/" "$host:$MODEL_HOST_PATH/"
done

echo ">> verifying config.json on every node:"
for host in $SSH_HOSTS; do
  case "$host" in
    "$SELF") test -f "$MODEL_HOST_PATH/config.json" && echo "   $SELF: OK" ;;
    *) ssh "$host" "test -f '$MODEL_HOST_PATH/config.json'" && echo "   $host: OK" || { echo "   $host: MISSING" >&2; exit 1; } ;;
  esac
done
echo ">> weights in place on every node"
