# Qwen3.8-Flash-Next-NVFP4 · 4x DGX Spark · vLLM TP4

Serving **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**
(~176B params: 125B main + 51B n-gram table, 6B active; ~122 GiB checkpoint) across
**four NVIDIA DGX Spark (GB10 / SM121) class nodes** at tensor-parallel 4, with
**vLLM**, over a switched dual-rail RoCEv2 fabric.

> ⚠️ **Status: single-node baseline validated; TP4 pending.** The full patched
> stack has booted and passed the gate suite at TP1 on one node of the target
> cluster (see *Measured so far*). The TP4-across-four-Sparks part — the work
> this repo exists to do — is waiting on a hardware window. Numbers below
> marked TP1 are measured; everything about TP4 is still a plan.

## Measured so far (TP1 baseline, 2026-08-30)

One node, this repo's image (`GPU_MEM=0.80`, native 262k, MTP=2, mmap PLE,
`EXACT_TOPK=1`, thinking off):

| | |
|---|---|
| Boot to `Application startup complete` | ~14.5 min |
| PLE mmap | 47.7 GiB table served from NVMe, decode-sized gather ~12 ms |
| KV pool (bf16) | **453,320 tokens** — 1.73x a full 262k request |
| Single-stream decode | **30.8 tok/s** ("write a function then explain it", temp 0) |
| Gate suite | **all PASS** — deep-decode, 3x ~30k concurrent prefills (44–58 s), determinism (byte-identical), NIAH 4k/32k at 0/50/100% depth |

This matches the upper end of the published single-Spark vLLM numbers
(26–31 tok/s) and validates every per-node piece — image, patches, PLE lane A,
MTP on this architecture — before any fabric involvement.

## Why four Sparks, and why vLLM

Every existing multi-Spark recipe for this model
([MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
[tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark),
[Weschera](https://github.com/Weschera/qwen38-flashnext-dgx-spark)) runs **SGLang**
at TP2. On the vLLM side there is a solid **single**-Spark recipe
([blazux](https://github.com/blazux/qwen3.8-Flash-DGX)) — but no multi-node
deployment of this model on vLLM has been published for this hardware. That is
the gap.

The reason it is worth filling is memory, not vanity:

- **vLLM's QSA layers require a bf16 KV cache** (fp8 is refused). In bf16, a single
  full 1M-token request needs ~30 GiB of KV — which is why the single-box vLLM
  recipe tops out at 500k context and calls 1M "out of reach".
- **TP4 frees the memory that KV needs.** With only 2 KV heads in this
  architecture, KV shards across two ranks and is replicated beyond that — a 1M
  request still costs ~15 GiB of KV per rank at TP4, same as TP2. What TP4 adds
  is weight sharding (~19 GiB of non-PLE weights per rank instead of ~76 GiB),
  freeing tens of GiB per node for the KV pool and concurrency.

So the headline this repo is chasing: **the model's full 1M context (YaRN), on
vLLM, with room for concurrency** — something neither one box nor two can do on
this stack. Aggregate throughput under concurrency is the secondary target;
single-stream decode over a switched fabric is explicitly *not* the game (6B
active params means the per-layer all-reduce is a real tax — we will measure and
publish the delta honestly).

## What's in the image

`Dockerfile` = the official day-0 image `vllm/vllm-openai:qwen38-flash-next`
(pinned by digest) + the GB10 patch stack vendored from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (Apache-2.0,
see [NOTICE](NOTICE)). All of it is per-node and TP-agnostic:

| # | Patch | Why |
|---|---|---|
| 1 | PLE n-gram table mmapped from NVMe (`VLLM_PLE_MMAP=1`) | 44 GiB of lookup table never enters the unified pool; weights drop ~122 → ~76 GiB per *node* before TP sharding |
| 2 | FLA shared-memory gate + `num_warps` pin | GB10 reports 99 KiB shmem; stock gate wants 100 KiB → all 36 GDN layers silently on small tiles; fla#953 race |
| 3 | Mamba state-copy race fix (vllm#50729) + bounds guard | dead CUDA context otherwise |
| 4 | Prefix-caching block_size fix | stock vLLM silently restored an all-zero Mamba state on every cache hit |
| 5 | Exact QSA top-k (`VLLM_QSA_EXACT_TOPK=1`) | stock `persistent_topk` is non-deterministic on GB10 and drops candidates (vllm#51782) |
| 6 | Optional NVFP4 + fp8 side-layers hybrid (`VLLM_FP8_HYBRID=1`) | +20% decode on one box, same tournament quality |

What this repo adds on top: the 4-node launcher (`launch-qwen38-tp4.sh`), the
fleet scripts (image/weight fan-out with ID verification, preflight, teardown,
per-node hybrid conversion), the gate suite and bench harness (`evals/`), a
self-healing watchdog (`fleet_watchdog.sh`), and the GB10 multi-node
operational discipline learned the hard way in
[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
unconditional page-cache flusher during boot, worker-first launch, tear down all
ranks before relaunching any, capture logs before `docker rm -f`.

## Quickstart (4 nodes)

Prereqs: 4x GB10 nodes (128 GB each), Docker + NVIDIA runtime, passwordless ssh
between them, a RoCEv2 fabric (switched or direct), ~130 GB free NVMe per node.

```bash
git clone https://github.com/tsw2k/Qwen3.8-Flash-Next-Quad-DGX-Sparks.git
cd Qwen3.8-Flash-Next-Quad-DGX-Sparks
cp .env.example .env        # edit: rank->IP map, NCCL devices/interfaces, model path

scripts/sync-weights.sh     # HF download on this node, rsync to the rest (~122 GiB, resumable)
scripts/sync-image.sh       # build ONCE here, ship to peers, verify identical image IDs
scripts/preflight.sh        # weights / image / memory / swappiness / rails, per node
```

Memory ritual, **on every node** (GB10 wedges — no panic, power-cycle only — when
the unified pool overcommits; swappiness>0 can livelock the UVM driver):

```bash
sudo sysctl -w vm.swappiness=0            # put it in /etc/sysctl.d/ or lose a boot to it later
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
setsid nohup ./flusher-unconditional.sh > flusher.log 2>&1 &
grep -q "flusher: starting" flusher.log || { echo "FLUSHER DID NOT START"; tail flusher.log; }
```

Launch **worker-first**, head last (run on each node):

```bash
./launch-qwen38-tp4.sh 3    # worker
./launch-qwen38-tp4.sh 2    # worker
./launch-qwen38-tp4.sh 1    # worker
./launch-qwen38-tp4.sh 0    # head — serves http://<head>:8000/v1
```

Once serving: stop the flusher (`pkill -f flusher-unconditional.sh` — the PLE mmap
path *wants* warm page cache at serve time), then `scripts/smoke-test.sh`.

Any relaunch, including after a failed boot: `scripts/teardown.sh` first, always.

## The boot ladder (plan)

Nothing gets promoted to a default here without passing
[`evals/gate_suite.py`](evals/gate_suite.py): long prompt AND long forced
answer (varied per run so the prefix cache can't fake it), concurrent
prefills, determinism, NIAH passkey retrieval, `/health` 200 throughout.
A config that boots and answers a short prompt is not a config that works.
Throughput numbers come from [`evals/bench_sweep.py`](evals/bench_sweep.py)
(concurrency sweep with TTFT, real prompts, no `ignore_eos`).

1. **Rung 0 — rendezvous:** TP4, native 262k, `GPU_MEM=0.75`, MTP=2. First
   verification of vLLM multi-node (`--nnodes 4`, mp backend) for this
   architecture; if the MTP drafter fails to load at TP4 (cf. vllm#52480 on a
   sibling arch), fall back to `MTP=0` and file upstream.
2. **Rung 1 — native-context gates:** NIAH at 32k/128k/256k, concurrent prefills,
   determinism (`EXACT_TOPK=1`), vision.
3. **Rung 2 — 1M YaRN:** `YARN=1 CTX=1048576`, then ladder the KV pin
   (`KV_GIB=`) upward, gating every step, until the first failure — then back
   off one rung and pin that as the default here.
4. **Rung 3 — throughput:** concurrency sweep (1/2/4/8/16/32 streams) at the
   pinned config; publish aggregate + per-stream + TTFT, with prompts quoted.

## Roadmap

In order; nothing skips ahead of the gates.

1. **bf16-KV lane through the gate suite** — the boot ladder above, on stock
   NVFP4 weights. This is the config the repo ships as default once it passes.
2. **Hybrid weights lane** — tooling shipped
   ([`scripts/prepare-hybrid.sh`](scripts/prepare-hybrid.sh): each node
   converts its local copy, deterministic, verified to match across the
   fleet; serve with `HYBRID=1`). Still to do: measure whether the +20%
   decode from one box holds at TP4.
3. **KV-dtype port for the QSA path (stretch goal)** — vLLM's QSA layers
   currently refuse anything but bf16 KV. SGLang proved fp8 and even NVFP4 KV
   work for this model (MiaAI-Lab: 0.93M → 1.75M → 2.85M token pools at the
   same memory budget, NIAH-validated to 128k). Porting that to vLLM's kernels
   would cut per-rank KV for a 1M request from ~15 GiB to ~7.5 (fp8) or
   ~4.7 GiB (NVFP4) and is a candidate upstream contribution. Starts only
   after the bf16 lane has passed its gates.

Not on the roadmap: reshaping the checkpoint to 4 KV heads. With 2 KV heads,
vLLM already replicates one head per rank at TP4; duplicating heads in the
weights changes nothing (the per-rank floor of one full head stays), and
*real* extra heads would mean retraining. The KV lever is dtype, not head
count.

## Open questions this repo will answer

- Does vLLM's `--nnodes 4` / mp-backend path work for `qwen4exp` on GB10 at all?
  (Validated upstream on GB300 trays; never published on Sparks.)
- Does MTP load and win at TP4? At what acceptance on real traffic?
- Which PLE lane wins at TP4 — mmap from local NVMe, or the stock
  vocab-sharded resident table (~11–13 GiB/rank), which no single box can
  afford but TP4 can? Source analysis in
  [docs/TP4-DESIGN-NOTES.md](docs/TP4-DESIGN-NOTES.md).
- What does the per-layer all-reduce cost on a switched 2x100G dual-rail fabric
  vs the direct-cabled 200G the TP2 recipes use?
- Where is the KV ceiling per rank with bf16 KV + the flusher discipline, and
  what does the resulting pool mean in concurrent-1M-request terms?

## Fabric notes

The `.env.example` ships the fabric this is developed on: dual-rail RoCEv2
(one PF per QSFP cage, one per PCIe domain — GB10's ConnectX-7 is socket-direct,
and using two PFs of the *same* cage just splits one wire), DSCP 26 → TC3
matched on the switch, MTU 9000, GID index 3. Get `NCCL_IB_HCA` /
`NCCL_SOCKET_IFNAME` wrong and you hang at rendezvous with very little in the
log. Check names with `ibdev2netdev`; verify the fabric with `ib_write_bw`
before blaming the recipe.

## Credits

- **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** — the
  entire single-node GB10 vLLM patch stack this builds on (vendored, Apache-2.0),
  itself carrying work by **@Saren-Arterius** (FLA fixes, PLE gather fast path,
  state-copy guard, fp8 conversion), **@AndreasKaratzas** (vllm#50729),
  **@k3dani** (persistent_topk diagnosis), **@jschmied** (reproduction,
  concurrency measurements).
- **[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)**
  — the 4-Spark operational playbook: the unconditional flusher, the gate
  discipline, worker-first launch, image-ID rule.
- **[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)**
  — the first multi-Spark serve of this model (SGLang TP2) and the GB10
  unified-memory crash post-mortems.
- **Qwen team, Alibaba** — the model; **[RadixArk](https://huggingface.co/RadixArk)**
  — the NVFP4 checkpoint; **vLLM** — the day-0 image and engine.
