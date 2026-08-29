#!/usr/bin/env bash
set -euo pipefail
# One-time preparation of the "hybrid" checkpoint on EVERY node: NVFP4 experts as
# published, dense side layers (GDN in/out projections, QSA q/k/v/o, shared experts —
# ~15 GiB of bf16) rewritten as blockwise fp8-e4m3 (DeepSeek layout, 128x128 blocks).
# Those layers are read in full on every decoded token; halving them bought +20%
# decode on one box (blazux) — whether it holds at TP4 is a roadmap question.
#
# Strategy: the conversion is deterministic math, and every node already holds the
# full checkpoint on local NVMe — so each node converts ITS OWN copy (~10 min, +13 GB
# disk, zero fabric/WAN traffic), and we verify the converted indexes and shard sizes
# match across the fleet afterwards. Conversion tool: tools/fp8_convert.py by
# @Saren-Arterius (Apache-2.0, see NOTICE).
#
# Run from the repo checkout on the head node: scripts/prepare-hybrid.sh
# Then serve with HYBRID=1 in .env.
cd "$(dirname "$0")/.."
test -f .env || { echo "MISSING: .env — cp .env.example .env first" >&2; exit 3; }
set -a; . ./.env; set +a

SELF="$(hostname -s)"
DST="${MODEL_HOST_PATH%/}-fp8hybrid"
REPO_DIR="${REPO_DIR:-$PWD}"

# The in-container conversion script, executed identically on every node.
# cp -al hardlinks the untouched shards (instant, no extra disk for them); the
# converter renames rewritten shards to .bf16.bak and writes fp8 shards as new
# files, so the source dir's entries are never modified through the hardlinks.
# index.json IS rewritten in place -> replace it with a real copy first.
CONVERT_CMD="
set -euo pipefail
if [ -f '$DST/.prepared' ]; then echo '>> already prepared'; exit 0; fi
rm -rf '$DST'
cp -al '$MODEL_HOST_PATH' '$DST'
rm '$DST/model.safetensors.index.json'
cp '$MODEL_HOST_PATH/model.safetensors.index.json' '$DST/model.safetensors.index.json'
docker run --rm -v '$DST:/ckpt' -v '$REPO_DIR/tools:/tools:ro' \
  --entrypoint python3 '$IMAGE' /tools/fp8_convert.py /ckpt | tee '$DST/fp8_convert.log'
n=\$(python3 -c \"import json;print(sum(1 for k in json.load(open('$DST/model.safetensors.index.json'))['weight_map'] if k.endswith('weight_scale_inv')))\")
[ \"\$n\" -gt 0 ] || { echo '!! conversion produced no fp8 tensors'; exit 1; }
rm -f '$DST'/*.bf16.bak
touch '$DST/.prepared'
echo \">> converted: \$n fp8 side-layer tensors\"
"

for host in $SSH_HOSTS; do
  echo ">> $host: converting (this takes ~10 min per node; nodes run sequentially so"
  echo "   you can watch the first one — Ctrl-C before the next if it looks wrong)"
  case "$host" in
    "$SELF") bash -c "$CONVERT_CMD" ;;
    *) ssh "$host" "$CONVERT_CMD" ;;
  esac
done

echo ">> verifying converted checkpoints match across the fleet (index hash + shard sizes):"
REF=""
for host in $SSH_HOSTS; do
  SUM_CMD="cd '$DST' && sha256sum model.safetensors.index.json | cut -d' ' -f1 && ls -l *.safetensors | awk '{print \$5, \$NF}' | sort | sha256sum | cut -d' ' -f1"
  case "$host" in
    "$SELF") SUM="$(bash -c "$SUM_CMD" | paste -sd/ -)" ;;
    *) SUM="$(ssh "$host" "$SUM_CMD" | paste -sd/ -)" ;;
  esac
  echo "   $host: $SUM"
  [ -z "$REF" ] && REF="$SUM"
  [ "$SUM" = "$REF" ] || { echo ">> MISMATCH on $host — do not serve HYBRID=1" >&2; exit 1; }
done
echo ">> hybrid checkpoint ready on every node. Set HYBRID=1 in .env and relaunch."
