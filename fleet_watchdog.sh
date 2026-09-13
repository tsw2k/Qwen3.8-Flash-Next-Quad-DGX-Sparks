#!/usr/bin/env bash
# fleet_watchdog.sh, auto-recovery for the TP4 fleet. Runs on the head (rank 0).
# Probes every rank's container, /health, and a one-token canary request; on N
# consecutive failures: tears down ALL ranks, runs the GB10
# memory ritual, starts the unconditional flusher everywhere, relaunches
# workers-first (3 to 2 to 1), then the head, waits for ready, stops the flushers.
#
# Pattern from tonyd2wild's GLM fleet_watchdog (see NOTICE), because the failure
# modes are identical: vLLM v1 cannot recover a dead engine core, and Docker
# restart policies are unsafe here, headless workers exit 0 on head death
# (on-failure never fires) and a dead head often never exits at all. Full
# orchestrated relaunch is the only cure.
#
# One deliberate difference from the GLM original: after recovery the flushers
# are STOPPED. The PLE mmap path wants a warm page cache at serve time; an
# unconditional flusher left running would keep evicting the n-gram table.
#
# Why more than /health: on DeepSeek-V4.1-Flash, same vLLM multi-node mp executor, a
# killed worker left the head at /health 200 with nothing logged, and requests hung
# until an NCCL timeout ~6 min later (tsw2k/Deepseek-4.1-Flash-Quad-DGX-Sparks,
# failover test 2026-09-13). Not reproduced on this stack; the mechanism is the same.
#
# Recovery is ~15+ min on this stack, tune FAIL_THRESHOLD before pointing this
# at a busy endpoint. Not started automatically; run it once serving is gated:
#   setsid nohup ./fleet_watchdog.sh > watchdog.out 2>&1 &
set -u
cd "$(dirname "$0")"
test -f .env || { echo "MISSING: .env, cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

### ---- config -------------------------------------------------------------
HEALTH_URL="http://127.0.0.1:${PORT}/health"  # NOT /v1/models: that returns 200
                                              # even with a dead engine core.
CHECK_INTERVAL=60
FAIL_THRESHOLD=3
CURL_TIMEOUT=15
CANARY_TIMEOUT=90           # chunked prefill interleaves the canary with long prompts
MAX_RECOVERIES=3            # failed relaunches in a row before giving up
READY_TIMEOUT=3600          # matches VLLM_ENGINE_READY_TIMEOUT_S in the launcher
CONTAINER="vllm_qwen38"
REPO_DIR="${REPO_DIR:-$PWD}"  # same checkout path expected on every node
LOCKFILE="$HOME/.qwen38_watchdog.lock"
PAUSE_FLAG="$HOME/.qwen38_watchdog.pause"    # touch to pause for maintenance; rm to resume
GIVEUP_FLAG="$HOME/.qwen38_watchdog.gaveup"  # written after MAX_RECOVERIES; rm to re-arm
LOGFILE="$PWD/fleet_watchdog.log"
POST_TEARDOWN_SLEEP=10      # let master-port TIME_WAIT / NVRM settle
INTER_WORKER_SLEEP=5
### -------------------------------------------------------------------------

# rank to host, from .env's SSH_HOSTS (rank order). Launch order: 3,2,1 then 0.
read -r -a HOSTS <<< "$SSH_HOSTS"
[ "${#HOSTS[@]}" = 4 ] || { echo "SSH_HOSTS must list 4 hosts in rank order" >&2; exit 3; }
SELF="$(hostname -s)"
RANK_ORDER=(3 2 1 0)

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

run_on() {  # run_on <rank> <command string>
  local rank="$1"; shift
  local host="${HOSTS[$rank]}"
  if [ "$host" = "$SELF" ]; then
    bash -lc "$*" >> "$LOGFILE" 2>&1
  else
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" "$*" >> "$LOGFILE" 2>&1
  fi
}

healthy() { curl -sf -m "$CURL_TIMEOUT" -o /dev/null "$HEALTH_URL"; }

# probe: prints the reason and returns 1 on the first failed check.
probe() {
  local r host state
  for r in "${RANK_ORDER[@]}"; do
    host="${HOSTS[$r]}"
    if [ "$host" = "$SELF" ]; then
      state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
    else
      state=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" "docker inspect -f '{{.State.Status}}' $CONTAINER 2>/dev/null || echo missing" 2>/dev/null || echo unreachable)
    fi
    [ "$state" = running ] || { echo "rank $r container: $state"; return 1; }
  done
  healthy || { echo "health"; return 1; }
  curl -sf -m "$CANARY_TIMEOUT" -o /dev/null "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"prompt\":\"1\",\"max_tokens\":1,\"temperature\":0}" \
    || { echo "canary"; return 1; }
  return 0
}

start_flusher() {
  run_on "$1" "pkill -f '[f]lusher-unconditional.sh' 2>/dev/null; cd '$REPO_DIR' && setsid nohup ./flusher-unconditional.sh >/tmp/flusher-unconditional.log 2>&1 < /dev/null & sleep 1; pgrep -f '[f]lusher-unconditional.sh' >/dev/null && echo flusher:RUNNING || echo flusher:FAILED"
}

stop_flusher() { run_on "$1" "pkill -f '[f]lusher-unconditional.sh' 2>/dev/null || true"; }

mem_ritual() {
  run_on "$1" 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "WARN: drop_caches failed (sudo -n?)"; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null || echo "WARN: compact_memory failed"'
}

recover() {
  log "=== RECOVERY START: $FAIL_THRESHOLD consecutive health failures ==="

  # 1. Tear down EVERYTHING first. A worker must never start while the old head
  #    is dying: it joins the stale rendezvous and wedges the new head's boot.
  for r in "${RANK_ORDER[@]}"; do
    log "teardown rank $r (${HOSTS[$r]})"
    run_on "$r" "docker logs $CONTAINER --tail 200 2>&1 | sed 's/^/[pre-teardown rank $r] /'; docker rm -f $CONTAINER 2>/dev/null || true"
  done
  sleep "$POST_TEARDOWN_SLEEP"

  # 2. Memory ritual + unconditional flusher on all nodes, for the whole boot.
  for r in "${RANK_ORDER[@]}"; do
    log "mem ritual + flusher on rank $r"
    mem_ritual "$r"
    start_flusher "$r"
  done

  # 3. Relaunch workers-first, head last.
  for r in "${RANK_ORDER[@]}"; do
    log "launch rank $r on ${HOSTS[$r]}"
    if ! run_on "$r" "cd '$REPO_DIR' && ./launch-qwen38-tp4.sh $r"; then
      log "ERROR: launch of rank $r reported failure; continuing (head may still rendezvous)"
    fi
    [ "$r" != 0 ] && sleep "$INTER_WORKER_SLEEP"
  done

  # 4. Wait for the engine (TP4 load takes many minutes), then stop the flushers
  #    so the PLE page cache can warm back up.
  log "waiting up to ${READY_TIMEOUT}s for $HEALTH_URL"
  local waited=0
  until healthy; do
    sleep 30; waited=$((waited + 30))
    if (( waited >= READY_TIMEOUT )); then
      log "ERROR: fleet not healthy within ${READY_TIMEOUT}s, will retry via main loop"
      return 1
    fi
  done
  for r in "${RANK_ORDER[@]}"; do stop_flusher "$r"; done
  log "=== RECOVERY COMPLETE: healthy after ${waited}s, flushers stopped ==="
  return 0
}

### ---- main ---------------------------------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "fleet_watchdog already running (lock: $LOCKFILE)" >&2
  exit 1
fi
log "watchdog started (pid $$, interval ${CHECK_INTERVAL}s, threshold $FAIL_THRESHOLD)"

fails=0
failed_recoveries=0
while true; do
  if [ -f "$GIVEUP_FLAG" ] || [ -f "$PAUSE_FLAG" ]; then
    fails=0; sleep "$CHECK_INTERVAL"; continue
  fi
  if why=$(probe); then
    (( fails > 0 )) && log "probe OK again after $fails failure(s)"
    fails=0
    failed_recoveries=0
  else
    fails=$((fails + 1))
    log "probe FAIL ($fails/$FAIL_THRESHOLD): $why"
    if (( fails >= FAIL_THRESHOLD )); then
      if recover; then
        failed_recoveries=0
      else
        failed_recoveries=$((failed_recoveries + 1))
        log "recovery attempt failed ($failed_recoveries/$MAX_RECOVERIES)"
        # A boot that keeps failing needs a person; relaunching every few minutes hides the cause.
        if (( failed_recoveries >= MAX_RECOVERIES )); then
          echo "gave up $(date '+%F %T') after $failed_recoveries failed recoveries" > "$GIVEUP_FLAG"
          log "=== GIVING UP; rm $GIVEUP_FLAG to re-arm ==="
        fi
      fi
      fails=0
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
