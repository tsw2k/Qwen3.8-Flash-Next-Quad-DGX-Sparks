#!/usr/bin/env bash
# Unconditional drop_caches for the whole boot window. Pattern and hard-won rationale
# from tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark (see NOTICE):
#
# It MUST be unconditional. A threshold-triggered flusher (flush only when Cached > N)
# can sit below its threshold and still leave the NVRM allocator short, which shows up
# as the SAME command booting or OOMing depending on the moment.
#
# Run on EVERY node, started BEFORE the launcher, and leave it running for the full
# boot. Stop it once the engine is serving: pkill -f flusher-unconditional.sh
#
# NOTE for THIS model: the PLE mmap path *wants* warm page cache at serve time — the
# flusher is for the boot window (weight load + KV allocation) only. Stopping it after
# "Application startup complete" matters more here than on GLM.
set -u
DURATION="${1:-5400}"   # seconds; default 90 min

if ! sudo -n true 2>/dev/null; then
  echo "FATAL: passwordless sudo is required (this loop runs 'sudo tee /proc/sys/vm/drop_caches')." >&2
  echo "       Without it the flusher fails silently and you reproduce the OOM it exists to prevent." >&2
  exit 1
fi

echo "flusher: starting, unconditional, every 60s for ${DURATION}s (pid $$)"
end=$((SECONDS+DURATION))
while [ $SECONDS -lt $end ]; do
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "flusher: WARN drop_caches failed"
  sleep 60
done
echo "flusher: window elapsed after ${DURATION}s -- exiting. If the engine is still booting, restart it."
