#!/usr/bin/env bash
set -euo pipefail
# Tear down ALL ranks. Not optional before any (re)launch: a fresh rank that
# rendezvouses with a dying one hangs, and a retry after a failed boot is the
# common case. Captures logs BEFORE removing containers.
#
# Run from the repo checkout on the head node: scripts/teardown.sh
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env — cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

NAME="vllm_qwen38"
SELF="$(hostname -s)"
LOGDIR="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

for host in $SSH_HOSTS; do
  case "$host" in
    "$SELF")
      docker logs "$NAME" > "$LOGDIR/$host.log" 2>&1 || true
      docker rm -f "$NAME" 2>/dev/null || true ;;
    *)
      ssh "$host" "docker logs $NAME 2>&1" > "$LOGDIR/$host.log" || true
      ssh "$host" "docker rm -f $NAME 2>/dev/null" || true ;;
  esac
  echo ">> $host: torn down (log -> $LOGDIR/$host.log)"
done
echo ">> all ranks down. Logs in $LOGDIR/"
