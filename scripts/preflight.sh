#!/usr/bin/env bash
set -uo pipefail
# Fleet preflight — run from the repo checkout on the head node before any launch.
# Checks, per node: weights, image ID equality, free memory, swappiness, IB devices,
# rail connectivity, leftover containers. Exit 0 = safe to run the memory ritual
# and launch.
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env — cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

NAME="vllm_qwen38"
SELF="$(hostname -s)"
FAIL=0
say() { printf '   %-10s %s\n' "$1" "$2"; }
run() { local h=$1; shift; case "$h" in "$SELF") bash -c "$*" ;; *) ssh "$h" "$*" ;; esac }

REF_ID=""
for host in $SSH_HOSTS; do
  echo ">> $host"

  if run "$host" "test -f '$MODEL_HOST_PATH/config.json'" 2>/dev/null; then
    say weights OK
  else say weights "MISSING $MODEL_HOST_PATH (scripts/sync-weights.sh)"; FAIL=1; fi

  ID="$(run "$host" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null" || true)"
  if [ -z "$ID" ]; then say image "MISSING $IMAGE (scripts/sync-image.sh)"; FAIL=1
  elif [ -z "$REF_ID" ]; then REF_ID="$ID"; say image "$ID"
  elif [ "$ID" = "$REF_ID" ]; then say image "OK (matches)"
  else say image "MISMATCH: $ID"; FAIL=1; fi

  SWP="$(run "$host" "sysctl -n vm.swappiness" 2>/dev/null || echo '?')"
  if [ "$SWP" = 0 ]; then say swappiness OK
  else say swappiness "$SWP — set 0 (swappiness>0 can livelock the UVM driver on GB10)"; FAIL=1; fi

  AVAIL="$(run "$host" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo '?')"
  say mem-avail "${AVAIL} GiB"
  case "$AVAIL" in (\?|[0-9]|[0-9][0-9]) [ "$AVAIL" != "?" ] && [ "$AVAIL" -lt 100 ] && { say mem-avail "under 100 GiB free — reboot or free the node before a boot"; FAIL=1; } ;; esac

  IBDEV="$(run "$host" "ls /dev/infiniband 2>/dev/null | head -c 200" || true)"
  if [ -n "$IBDEV" ]; then say infiniband OK
  else say infiniband "no /dev/infiniband — RoCE stack down?"; FAIL=1; fi

  LEFT="$(run "$host" "docker ps -a --format '{{.Names}}' | grep -x $NAME" 2>/dev/null || true)"
  if [ -n "$LEFT" ]; then say container "leftover '$NAME' — scripts/teardown.sh first"; FAIL=1
  else say container clean; fi

  # bracket trick: the pattern must not match the ssh/bash wrapper carrying it
  if run "$host" "pgrep -f '[f]lusher-unconditional.sh' >/dev/null" 2>/dev/null; then say flusher running
  else say flusher "not running (start it on every node before launching)"; fi
done

echo ">> rail connectivity from $SELF (ping over the compute fabric):"
for ip in "$RANK0_IP" "$RANK1_IP" "$RANK2_IP" "$RANK3_IP"; do
  if ping -c1 -W1 "$ip" >/dev/null 2>&1; then say "$ip" reachable
  else say "$ip" UNREACHABLE; FAIL=1; fi
done

[ "$FAIL" = 0 ] && echo ">> PREFLIGHT OK" || { echo ">> PREFLIGHT FAILED — fix the lines above before launching" >&2; exit 1; }
