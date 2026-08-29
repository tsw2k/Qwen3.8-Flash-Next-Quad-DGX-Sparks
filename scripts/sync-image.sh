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
test -f .env || { echo "MISSING: .env — cp .env.example .env first" >&2; exit 3; }
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
  echo ">> $host: shipping image (docker save | zstd | ssh | docker load)"
  docker save "$IMAGE" | zstd -3 -T0 | ssh "$host" "zstd -d | docker load"
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
[ "$FAIL" = 0 ] && echo ">> OK: identical image ID on every node" || { echo ">> MISMATCH — do not launch" >&2; exit 1; }
