# Qwen3.8-Flash-Next on 4x DGX Spark (GB10) at TP4, via vLLM.
#
# Official day-0 vLLM image + the single-Spark GB10 patch stack from
# blazux/qwen3.8-Flash-DGX (see NOTICE). Every patch is per-node and TP-agnostic;
# the multi-node part of this repo lives in the launcher, not the image.
#
#   1. PLE table served from disk via mmap            (VLLM_PLE_MMAP=1)
#   2. GB10 FLA fixes                                  (always on, harmless elsewhere)
#   3. Mamba state-copy race fix + bounds guard        (always on)
#   4. Prefix-caching block_size fix                   (always on)
#   5. Exact, deterministic QSA top-k                  (VLLM_QSA_EXACT_TOPK=1)
#   6. NVFP4 experts + fp8 side-layers "hybrid" mode   (VLLM_FP8_HYBRID=1)
#
# BUILD ON ONE NODE ONLY, then fan out with scripts/sync-image.sh. Four nodes that
# each build the tag locally get four different image IDs, and a rank silently on a
# divergent image is the hardest multi-node failure to diagnose.
#
#   docker build -t qwen38-flash-quad:local .
#
# Pinned by digest for reproducibility (same digest blazux validated on GB10).
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

# --- 1. PLE n-gram table from disk (VLLM_PLE_MMAP=1) ------------------------------
COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-quad: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# --- 2. GB10 FLA fixes (by @Saren-Arterius) ---------------------------------------
# sm_121 has 99 KiB shared memory per block; the FLA gate asked for 100 KiB, so all
# 36 GDN layers silently ran on small tiles. Plus fla#953: tl.dot race on Blackwell.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# --- 3. Mamba state copy: vllm#50729 + bounds guard --------------------------------
ARG MU=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MU} ${MU}.orig
COPY src/mamba_utils_guarded.py ${MU}
RUN python3 -c "import ast; ast.parse(open('${MU}').read()); print('mamba_utils.py guarded OK')"

# --- 4. Prefix caching: block_size fix ---------------------------------------------
# vLLM's EngineCore overwrites cache_config.block_size with the smallest KV-group
# block size while the Mamba block is 1600; a prefix hit then restored an all-zero
# Mamba state. With this, --enable-prefix-caching is correct.
COPY src/patch_mamba_block_size.py /tmp/patch_mamba_block_size.py
RUN python3 /tmp/patch_mamba_block_size.py ${SP} && rm /tmp/patch_mamba_block_size.py

# --- 5. Exact QSA top-k (VLLM_QSA_EXACT_TOPK=1) ------------------------------------
# The stock persistent_topk kernel is non-deterministic on GB10 and can drop real
# top-k candidates (vllm#51782). Exact torch.topk path; opt-in via env.
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py && rm /tmp/patch_qsa_exact_topk.py

# --- 7. PLE lane C: native CPU offload with ModelOpt checkpoints ------------------
# Upstream's offload gate only accepts Fp8Config; this lets VLLM_PLE_CPU_OFFLOAD=1
# work with the NVFP4 checkpoint (fp8 PLE shards + weight_scale pass through).
# Runtime needs --cap-add SYS_PTRACE (pidfd handover vs yama) — launcher handles it.
COPY src/patch_ple_offload_modelopt.py /tmp/patch_ple_offload_modelopt.py
RUN python3 /tmp/patch_ple_offload_modelopt.py ${PLE} && rm /tmp/patch_ple_offload_modelopt.py

# --- 6. Hybrid mode: NVFP4 experts + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) --
ARG MO=${SP}/vllm/model_executor/layers/quantization/modelopt.py
ARG QSA=${SP}/vllm/models/qwen3_8_flash_next/nvidia/qsa.py
COPY src/vllm_fp8_hybrid_modelopt.py ${SP}/vllm_fp8_hybrid_modelopt.py
RUN cp ${MO} ${MO}.orig \
 && printf '\n\n# --- qwen38-quad: NVFP4 + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${MO} \
 && python3 -c "import ast; ast.parse(open('${MO}').read()); print('modelopt.py hooked OK')" \
 && cp ${QSA} ${QSA}.orig \
 && sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' ${QSA} \
 && sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded/' ${QSA} \
 && grep -q "_fp8_hybrid_excluded(quant_config)" ${QSA} && grep -q "^from vllm_fp8_hybrid_modelopt import" ${QSA} \
 && python3 -c "import ast; ast.parse(open('${QSA}').read()); print('qsa.py hooked OK')"
