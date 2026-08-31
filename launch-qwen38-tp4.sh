#!/usr/bin/env bash
set -euo pipefail
# Qwen3.8-Flash-Next-NVFP4, TP4 across four DGX Sparks / GB10, via vLLM.
#
# Run ON each node, WORKER-FIRST:  ./launch-qwen38-tp4.sh 3   (then 2, 1, head 0 last).
# A fresh rank that rendezvouses with a dying one hangs — tear down ALL ranks first
# (scripts/teardown.sh), every time, including after a failed boot.
#
# Config comes from .env (copy .env.example). Prereqs on every node, checked below:
#   - $MODEL_HOST_PATH on local NVMe (~122 GiB; the PLE table is mmapped from it)
#   - the image with the SAME ID everywhere (scripts/sync-image.sh)
#   - flusher-unconditional.sh running for the whole boot (scripts/preflight.sh checks)
NODE_RANK="${1:?usage: launch-qwen38-tp4.sh <0|1|2|3>}"

cd "$(dirname "$0")"
test -f .env || { echo "MISSING: .env — cp .env.example .env and edit for your fabric" >&2; exit 3; }
set -a; . ./.env; set +a

NAME="vllm_qwen38"
MODEL_PATH="/models/qwen38-flash-next-nvfp4"

case "$NODE_RANK" in
  0) HOST_IP="$RANK0_IP"; HEADLESS="" ;;
  1) HOST_IP="$RANK1_IP"; HEADLESS="--headless" ;;
  2) HOST_IP="$RANK2_IP"; HEADLESS="--headless" ;;
  3) HOST_IP="$RANK3_IP"; HEADLESS="--headless" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

# HYBRID=1 serves the fp8-side-layers copy prepared by scripts/prepare-hybrid.sh.
if [ "${HYBRID:-0}" = 1 ]; then
  MODEL_HOST_PATH="${MODEL_HOST_PATH%/}-fp8hybrid"
  test -f "$MODEL_HOST_PATH/.prepared" || { echo "MISSING: $MODEL_HOST_PATH/.prepared — run scripts/prepare-hybrid.sh" >&2; exit 3; }
fi

# Fail loudly on missing prereqs rather than letting Docker create empty dirs over them.
test -f "$MODEL_HOST_PATH/config.json" || { echo "MISSING: $MODEL_HOST_PATH/config.json — run scripts/sync-weights.sh" >&2; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "MISSING: image $IMAGE — run scripts/sync-image.sh" >&2; exit 3; }
pgrep -f '[f]lusher-unconditional.sh' >/dev/null || echo "WARNING: flusher-unconditional.sh is not running — on GB10 the boot can OOM without it" >&2

# --enable-expert-parallel is REQUIRED at TP4 with this checkpoint: plain TP shards
# the MoE intermediate 640 -> 160/rank, which the NVFP4 FLASHINFER_CUTLASS kernels
# cannot pad (NotImplementedError at load). EP keeps experts whole, 128/rank.
# --ulimit nofile=1M: the PLE table is 128 memmapped shards; together with 4-node
# NCCL/EP sockets that blows through Docker's default fd limit (Errno 24 at load).

# The PLE gather is a CPU op + a host->device copy: it MUST run outside CUDA graphs.
# Declared as a splitting op with PIECEWISE capture (never FULL*). List from blazux.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'

# YaRN (Qwen's published recipe) past the native 262144.
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
OVR_ARGS=(); ALLOW_LONG=0
[ "${YARN:-0}" != 0 ] && { OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; }

# MTP + YaRN: dict hf_overrides don't reach the draft model — force its max_model_len
# through the speculative config (blazux finding). MTP at TP>=2 is the first thing to
# verify on this stack (cf. vllm#52480 on a sibling arch); MTP=0 is the fallback lane.
SPEC=()
if [ "${MTP:-2}" != 0 ]; then
  if [ "${YARN:-0}" != 0 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
  fi
fi

# Optional explicit KV pin (bytes) once you have laddered to your ceiling.
KV_ARGS=()
[ -n "${KV_GIB:-}" ] && KV_ARGS=(--kv-cache-memory $(( KV_GIB * 1024 * 1024 * 1024 )))

PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-1}" = 1 ] && PC_ARG=--enable-prefix-caching

HYBRID_ENV=()
[ "${HYBRID:-0}" = 1 ] && HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)

# PLE lane: mmap (default; table on local NVMe, ~0 GPU cost) or offload
# (native VLLM_PLE_CPU_OFFLOAD; ~13 GiB/rank pinned host at TP4 — the table is
# vocab-sharded. Needs SYS_PTRACE for the pidfd handover vs yama).
PLE_ENV=(); PLE_CAPS=()
case "${PLE_MODE:-mmap}" in
  mmap)    PLE_ENV=(-e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}") ;;
  offload) PLE_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1); PLE_CAPS=(--cap-add SYS_PTRACE) ;;
  *) echo "PLE_MODE must be mmap or offload" >&2; exit 2 ;;
esac

docker rm -f "$NAME" 2>/dev/null || true

# --memory 112g protects the node: GB10 wedges (no panic, power-cycle only) when the
# unified pool overcommits. EXPANDABLE_SEGMENTS=1 opts into
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the GLM TP4 deployments run it
# on this hardware against allocator fragmentation from repeated deep-prefill
# transients; the dual-Spark SGLang lore advises against it. Measure, don't assume.
# shellcheck disable=SC2086
docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --ulimit nofile=1048576:1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  ${EXPANDABLE_SEGMENTS:+-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True} \
  "${PLE_ENV[@]}" "${PLE_CAPS[@]}" \
  -e VLLM_QSA_EXACT_TOPK="${EXACT_TOPK:-1}" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"} \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_IB_ADDR_RANGE" \
  ${NCCL_IB_TC:+-e NCCL_IB_TC="$NCCL_IB_TC"} \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$CTRL_IFNAME" \
  -e TP_SOCKET_IFNAME="$CTRL_IFNAME" -e MN_IF_NAME="$CTRL_IFNAME" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "${HYBRID_ENV[@]}" \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --load-format safetensors \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" \
    --gpu-memory-utilization "$GPU_MEM" "${KV_ARGS[@]}" \
    $PC_ARG --enable-chunked-prefill --max-num-batched-tokens "${MNBT:-8192}" \
    -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="$SPLIT" \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype auto \
    "${OVR_ARGS[@]}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    "${SPEC[@]}" \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$RANK0_IP" --master-port "$MASTER_PORT" \
    $EXTRA $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP tp4 ctx=$CTX yarn=${YARN:-0} mtp=${MTP:-2} gpu_mem=$GPU_MEM kv_gib=${KV_GIB:-auto}"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME (capture logs BEFORE docker rm -f)" >&2; exit 1; }
