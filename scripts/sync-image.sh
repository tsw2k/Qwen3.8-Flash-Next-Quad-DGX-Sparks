#!/usr/bin/env bash
set -euo pipefail
# Build the image ONCE (on the node you run this from), fan it out, verify IDs.
#
# Rule (paid for in GLM boots): verify the image ID, not the tag name. Four nodes that
# each built the tag locally get four different images, and a rank silently on a
# divergent image is the hardest multi-node failure to diagnose.
#
# Run from the repo checkout on the head node: scripts/sync-image.sh
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env, cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

SELF="$(hostname -s)"

echo ">> building $IMAGE locally (official day-0 image + patch stack, ~1 min after the base pull)"
docker build -t "$IMAGE" .
LOCAL_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
echo ">> built $IMAGE = $LOCAL_ID"

for host in $SSH_HOSTS; do
  # shellcheck disable=SC2029
  case "$host" in "$SELF") continue ;; esac
  REMOTE_ID="$(ssh "$host" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null" || true)"
  if [ "$REMOTE_ID" = "$LOCAL_ID" ]; then
    echo ">> $host: already has $LOCAL_ID, skipping"
    continue
  fi
  if [ -n "${IMAGE_RSYNC_URL:-}" ]; then
    # Daemon pull over the compute rails, ssh streams are AES-capped ~1 Gbps on GB10.
    # Tar is named by image ID, so a stale tar from a previous build can never ship.
    TAR="${IMAGE_TAR_DIR:-/var/tmp}/qwen38-image-${LOCAL_ID#sha256:}.tar.zst"
    if [ ! -f "$TAR" ]; then
      echo ">> writing $TAR once for daemon pulls"
      docker save "$IMAGE" | zstd -3 -T0 > "$TAR.tmp" && mv "$TAR.tmp" "$TAR"
    fi
    echo ">> $host: pulling image tar via $IMAGE_RSYNC_URL"
    ssh "$host" "rsync -a --partial '$IMAGE_RSYNC_URL/$(basename "$TAR")' /var/tmp/ && zstd -d < /var/tmp/$(basename "$TAR") | docker load"
  else
    echo ">> $host: shipping image (docker save | zstd | ssh | docker load)"
    docker save "$IMAGE" | zstd -3 -T0 | ssh "$host" "zstd -d | docker load"
  fi
done

echo ">> verifying image IDs across the fleet:"
FAIL=0
for host in $SSH_HOSTS; do
  case "$host" in
    "$SELF") ID="$LOCAL_ID" ;;
    *) ID="$(ssh "$host" "docker image inspect $IMAGE --format '{{.Id}}'" || echo MISSING)" ;;
  esac
  echo "   $host: $ID"
  [ "$ID" = "$LOCAL_ID" ] || FAIL=1
done
[ "$FAIL" = 0 ] && echo ">> OK: identical image ID on every node" || { echo ">> MISMATCH, do not launch" >&2; exit 1; }
