# TP4 design notes (pre-boot, from source reading)

Findings from reading the vLLM day-0 branch (`peakcrosser7/vllm`,
`release/qwen38next`, PR vllm-project/vllm#53896, still open and not in main)
and the vendored patch stack, before first hardware boot. Everything here is
"verified in source, not yet on metal" unless marked otherwise.

## PLE n-gram table at TP4: three lanes

Upstream implements the PLE table as a `VocabParallelEmbedding`, at TP4 the
20M-row table is **sharded by vocab rows, ~1/4 per rank** (load path honours
`shard_indices.org_vocab_start/end_index`). That opens a lane that a single
box cannot have:

| Lane | Mechanism | GPU cost/rank | Notes |
|---|---|---|---|
| **A (default)** | blazux mmap from local NVMe | ~0 | full table on every node's disk; lookups hit page cache |
| **B (TP4-only)** | stock resident, vocab-sharded | ~11-13 GiB | impossible on one box (44 GiB fp8); affordable at TP4 since weights shrink to ~20 GiB/rank. Needs jschmied's fix: `_get_ple_embedding_quant_method()` accepts only `Fp8Config`, rejects `modelopt_fp4` checkpoints |
| C | `VLLM_PLE_CPU_OFFLOAD=1` pinned-host worker | host-side (same pool on GB10) | needs `--cap-add=SYS_PTRACE` (pidfd over yama); least attractive on GB10 |

How the mmap patch behaves at TP>1 (from `src/vllm_ple_mmap.py`): the
patched `__init__` swaps `VocabParallelEmbedding` for a placeholder that
ignores the TP sharding kwargs, and the patched `forward_impl` gathers the
complete embedding row from disk on every rank, bypassing both the sharded
lookup and its all-reduce. Since the stock path's contract is
"return the complete (post-all-reduce) tensor", this is semantically
equivalent; each rank just does the same small disk gather redundantly
(~2.5 KB/token, local NVMe, no fabric traffic).

Rung-0 verification item: confirm no double-counting anyway. Compare
first-token logprobs of a short greedy prompt at TP4 vs the published TP1
numbers. If lane A misbehaves at TP4, lane B is the fallback (and possibly
the better default: no disk on the hot path, at a memory price TP4 can pay).

## MTP at TP4: risk downgraded

The concern was vllm#52480 (MTP drafter shape mismatch at TP>=2 on a sibling
Qwen arch, via the generic `llm_base_proposer` path). The qwen4_exp branch
ships a purpose-built `Qwen4ExpMTP` instead: it reuses the backbone with
PLE forced off, carries its own checkpoint-path mapping in `load_weights`,
and resolves its own draft quant config. Different code path from the broken
one; still verify on first boot, but this is no longer the top risk.

## KV math (per rank, bf16, from config geometry)

- 12 full-attention (QSA) layers of 48 (36 are GDN/linear, no growing KV).
- 2 KV heads × 256 head_dim: full-model KV is about 12 × 2 × 256 × 2 (K+V) × 2 B,
  or ~24 KiB/token; per rank at TP4 (one replicated head) that is ~12 KiB/token.
- A 1M-token request needs ~15 GiB of KV per rank in bf16. This is why the repo
  exists: one box can't hold that next to ~76 GiB of weights; a TP4 rank
  with ~20 GiB of weights can.
- Not in this ledger: the GDN/mamba state cache, QSA indexer pools, raw-key
  ring, and CUDA graphs. Budget for tens of GiB before trusting any pool
  prediction. Ladder and measure; do not paste these numbers anywhere.

Head divisibility for TP4 is clean: 24 attention heads (6/rank), GDN 16 key /
48 value heads (4/12 per rank), 512 experts. KV heads = 2 < 4 ranks → vLLM
replicates (legal: 4 % 2 == 0).

## Patch-stack portability watch

The pinned day-0 image lays the model out under
`vllm/models/qwen3_8_flash_next/`; the PR branch has since renamed it to
`vllm/models/qwen4_exp/` (with `nvidia/` and `amd/` backends). Our Dockerfile
paths are correct for the pinned digest and will break loudly (guarded
`grep`/`ast.parse` steps) on any rebase. When the PR merges and an official
image supersedes the day-0 tag, expect to re-path the patches and re-check
that upstream didn't absorb some of them (the FLA gate and vllm#50729 are
natural merge candidates).
