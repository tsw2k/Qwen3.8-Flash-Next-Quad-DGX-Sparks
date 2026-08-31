# Qwen3.8-Flash-Next-NVFP4 · 4x DGX Spark · vLLM TP4

Serving [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(~176B params: 125B main plus a 51B n-gram table, 6B active; ~122 GiB checkpoint)
across four NVIDIA DGX Spark (GB10 / SM121) class nodes at tensor-parallel 4,
with vLLM, over a switched dual-rail RoCEv2 fabric.

> 💬 Discussion: [NVIDIA developer forums thread](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-4x-dgx-spark-vllm-tp4-serving-4-7m-token-kv-pool-and-the-three-fixes-you-will-need/381897)

> ✅ Status: TP4 serving and gated at native 262k with `GPU_MEM=0.80`, which
> is the shipped default (see *Measured*). Known limit: prompts past a wall at
> roughly 76.8k tokens can wedge the engine within a few requests. We
> binary-searched the wall and filed it upstream as
> [vllm#54629](https://github.com/vllm-project/vllm/issues/54629); prompts up
> to ~73k have been stable in every test. The 1M YaRN lane boots with a 4.27x
> pool but is unusable for deep prompts until that closes. Run
> `fleet_watchdog.sh` on any serving deployment. Details in
> [docs/OPEN-PROBLEMS.md](docs/OPEN-PROBLEMS.md).

## Measured: TP4, four nodes (2026-08-31)

This repo's image and launcher as shipped (`GPU_MEM=0.80`, native 262k, MTP=2,
EP, mmap PLE, `EXACT_TOPK=1`, thinking off, dual-rail switched RoCEv2):

| | |
|---|---|
| Boot to `Application startup complete` | ~8 min (sharded weight load; TP1 takes 14.5) |
| KV pool (bf16) | 5,211,726 tokens, 19.88x a full 262k request (4,696,556 at the earlier `GPU_MEM=0.75`; 0.85 overcommits the head node and is documented as fatal) |
| Single-stream decode | 31.0 tok/s ("write a function then explain it", temp 0). The TP1 baseline on the same image is 30.8, so the fabric costs nothing here |
| Aggregate decode | 46.0 (x2) · 53.3 (x4) · 97.0 (x8) · 157.0 tok/s (x16), per-stream 12.3 at x16 |
| TTFT | 0.24 s (x1) to 1.25 s (x16), max_tokens=300, mixed structured/prose/agentic prompts |
| Gate suite | all pass: deep decode, 3x ~30k concurrent prefills (32 to 44 s), byte-identical greedy decode, NIAH 4k/32k at 0/50/100% depth |

The memory story this repo was built on, in one line: TP1 fits 1.73 full
contexts of KV, TP4 fits 17.92.

Also verified on this lane: vision (image input), tool calling (clean
`message.tool_calls` with the `qwen3_coder` parser), prefix caching (0.7 s
TTFT on a cached 20k prefix), thinking, and MTP acceptance of 0.856 on mixed
traffic. One client note: this vLLM build returns the chain of thought in
`message.reasoning`, not `message.reasoning_content`. Point your agent client
at the right field, or thinking looks silently absent.

### TP1 baseline (same image, one node, 2026-08-30)

`GPU_MEM=0.80`: boot ~14.5 min · KV pool 453,320 tokens (1.73x) · 30.8 tok/s
single-stream · PLE mmap serves the 47.7 GiB table from NVMe with ~12 ms
decode gathers · same gate suite, all pass. This matches the upper end of the
published single-Spark vLLM numbers (26 to 31 tok/s), so the patch stack costs
nothing on the way through. Every per-node piece was validated before the
fabric got involved.

## The three TP4 fixes you will need (each cost us a boot)

None of these appear in any single- or dual-node recipe. All are defaults here.

1. Plain TP4 cannot load this checkpoint. TP sharding slices the MoE
   intermediate 640 to 160 per rank and the NVFP4 FLASHINFER_CUTLASS backend
   dies with `NotImplementedError: Intermediate size padding for w1 and w3`.
   `--enable-expert-parallel` deals the 512 experts out whole, 128 per rank.
2. Raise the fd limit. The PLE table is 128 memmapped shards; add 4-node
   NCCL/EP sockets and Docker's default `nofile` overflows. The boot dies with
   `OSError: [Errno 24] Too many open files` deep in PLE setup.
   `--ulimit nofile=1048576` fixes it for good.
3. Do not hardcode `NCCL_IB_GID_INDEX`. After a link bounce the RoCE v2 GID
   can land on a different index on one node (we caught 4 vs 3 fleet-wide).
   A pinned index then reads a zero GID and NCCL dies at init with
   `unhandled system error`. With `NCCL_IB_ROCE_VERSION_NUM=2` and
   `NCCL_IB_ADDR_RANGE` set, NCCL picks the right GID per device on its own.

## Why four Sparks, and why vLLM

Every existing multi-Spark recipe for this model
([MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
[tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark),
[Weschera](https://github.com/Weschera/qwen38-flashnext-dgx-spark)) runs SGLang
at TP2. On the vLLM side there is a solid single-Spark recipe
([blazux](https://github.com/blazux/qwen3.8-Flash-DGX)), but nobody had
published a multi-node vLLM deployment of this model on this hardware. That is
the gap.

The reason it is worth filling is memory:

- vLLM's QSA layers require a bf16 KV cache (fp8 is refused). In bf16, one
  full 1M-token request needs ~30 GiB of KV, which is why the single-box vLLM
  recipe tops out at 500k context and calls 1M "out of reach".
- TP4 frees the memory that KV needs. This architecture has only 2 KV heads,
  so KV shards across two ranks and is replicated beyond that: a 1M request
  still costs ~15 GiB of KV per rank at TP4, same as TP2. What TP4 adds is
  weight sharding (~19 GiB of non-PLE weights per rank instead of ~76 GiB),
  which frees tens of GiB per node for the KV pool and concurrency.

So the target: the model's full 1M context (YaRN), on vLLM, with room for
concurrency. Neither one box nor two can do that on this stack. Aggregate
throughput under concurrency is the second target. Single-stream decode over
a switched fabric was never the game: 6B active params means the per-layer
all-reduce is a real tax, and we measure and publish the delta as it is.

## What's in the image

The `Dockerfile` is the official day-0 image `vllm/vllm-openai:qwen38-flash-next`
(pinned by digest) plus the GB10 patch stack vendored from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(Apache-2.0, see [NOTICE](NOTICE)). All of it is per-node and TP-agnostic:

| # | Patch | Why |
|---|---|---|
| 1 | PLE n-gram table mmapped from NVMe (`VLLM_PLE_MMAP=1`) | 44 GiB of lookup table never enters the unified pool; weights drop from ~122 to ~76 GiB per node before TP sharding |
| 2 | FLA shared-memory gate + `num_warps` pin | GB10 reports 99 KiB shmem, the stock gate wants 100 KiB, so all 36 GDN layers silently ran on small tiles; also the fla#953 race |
| 3 | Mamba state-copy race fix (vllm#50729) + bounds guard | dead CUDA context otherwise |
| 4 | Prefix-caching block_size fix | stock vLLM silently restored an all-zero Mamba state on every cache hit |
| 5 | Exact QSA top-k (`VLLM_QSA_EXACT_TOPK=1`) | the stock `persistent_topk` kernel is non-deterministic on GB10 and drops candidates (vllm#51782) |
| 6 | Optional NVFP4 + fp8 side-layers hybrid (`VLLM_FP8_HYBRID=1`) | +20% decode on one box, same tournament quality |
| 7+8 | PLE lane switches (`PLE_MODE`) | opens the offload/resident paths to ModelOpt checkpoints; details in the launcher comments |

On top of that, this repo adds the 4-node launcher (`launch-qwen38-tp4.sh`),
fleet scripts (weight and image fan-out with image-ID verification, preflight,
teardown, per-node hybrid conversion), the gate suite and bench harness in
`evals/`, a self-healing watchdog (`fleet_watchdog.sh`), and the GB10
multi-node discipline learned in
[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
an unconditional page-cache flusher during boot, worker-first launch, tearing
down all ranks before relaunching any, and capturing logs before
`docker rm -f`.

## Quickstart (4 nodes)

Prereqs: 4x GB10 nodes (128 GB each), Docker with the NVIDIA runtime,
passwordless ssh between them, a switched RoCEv2 fabric (a GB10 has two QSFP
ports, so four nodes cannot be direct-cabled; TP4 needs a switch), and
~130 GB of free NVMe per node.

```bash
git clone https://github.com/tsw2k/Qwen3.8-Flash-Next-Quad-DGX-Sparks.git
cd Qwen3.8-Flash-Next-Quad-DGX-Sparks
cp .env.example .env        # edit: rank->IP map, NCCL devices/interfaces, model path

scripts/sync-weights.sh     # HF download on this node, rsync to the rest (~122 GiB, resumable)
scripts/sync-image.sh       # build ONCE here, ship to peers, verify identical image IDs
scripts/preflight.sh        # weights / image / memory / swappiness / rails, per node
```

The memory ritual, on every node. GB10 wedges with no kernel panic and needs a
power cycle when the unified pool overcommits, and swappiness above 0 can
livelock the UVM driver:

```bash
sudo sysctl -w vm.swappiness=0            # put it in /etc/sysctl.d/ or lose a boot to it later
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
setsid nohup ./flusher-unconditional.sh > flusher.log 2>&1 &
grep -q "flusher: starting" flusher.log || { echo "FLUSHER DID NOT START"; tail flusher.log; }
```

Launch worker-first, head last (run on each node):

```bash
./launch-qwen38-tp4.sh 3    # worker
./launch-qwen38-tp4.sh 2    # worker
./launch-qwen38-tp4.sh 1    # worker
./launch-qwen38-tp4.sh 0    # head, serves http://<head>:8000/v1
```

Once serving, stop the flusher (`pkill -f flusher-unconditional.sh`; the PLE
mmap path wants a warm page cache at serve time), then run
`scripts/smoke-test.sh`.

Any relaunch, including after a failed boot, starts with
`scripts/teardown.sh`. Always.

## The boot ladder

Nothing gets promoted to a default here without passing
[`evals/gate_suite.py`](evals/gate_suite.py): a long prompt AND a long forced
answer (varied per run so the prefix cache can't fake it), concurrent
prefills, determinism, NIAH passkey retrieval, and `/health` staying 200
throughout. A config that boots and answers a short prompt is not a config
that works. Throughput numbers come from
[`evals/bench_sweep.py`](evals/bench_sweep.py): a concurrency sweep with TTFT,
real prompts, and no `ignore_eos`.

1. Rung 0, rendezvous: done. vLLM multi-node (`--nnodes 4`, mp backend) works
   for this architecture once the three fixes above are in.
2. Rung 1, native-context gates: done, see *Measured*.
3. Rung 2, 1M YaRN: boots with a 4.78x pool; blocked on the deep-prefill
   wedge in [docs/OPEN-PROBLEMS.md](docs/OPEN-PROBLEMS.md). After that, the
   KV pin (`KV_GIB=`) ladders upward with a gate at every step.
4. Rung 3, throughput at the pinned config: aggregate, per-stream and TTFT,
   with the prompts quoted.

## Roadmap

In order, and nothing skips ahead of the gates.

1. Close the 1M-lane wedge. Filed upstream as
   [vllm#54629](https://github.com/vllm-project/vllm/issues/54629) with the
   repro, matrix and stack.
2. Rebase onto merged main: #53896 merged on 2026-08-31, so official images
   with the renamed `qwen4_exp` layout are coming. Re-path the patch stack
   (the guarded build steps will flag what upstream absorbed) and retest the
   wedge on a post-merge build.
3. The hybrid weights lane: measured, and not worth it at TP4. Two findings.
   First, shared experts cannot be converted at all (their 640-wide
   projections shard to 160 per rank and blockwise fp8 needs multiples of
   128), so the conversion covers GDN and QSA projections only. Second, with
   that subset the result is 29.5 tok/s against 31.0 stock: the single-box
   +20% came from reading whole dense layers per token, and a TP4 rank
   already reads a quarter of them. Tooling stays in the repo
   ([`scripts/prepare-hybrid.sh`](scripts/prepare-hybrid.sh), `HYBRID=1`)
   for anyone who wants to reproduce the measurement.
4. A KV-dtype port for the QSA path (stretch goal). vLLM's QSA layers
   currently refuse anything but bf16 KV. SGLang proved fp8 and even NVFP4 KV
   work for this model (MiaAI-Lab measured 0.93M / 1.75M / 2.85M token pools
   at the same memory budget, NIAH-validated to 128k). Porting that to vLLM's
   kernels would cut per-rank KV for a 1M request from ~15 GiB to ~7.5 (fp8)
   or ~4.7 GiB (NVFP4), and it is a candidate upstream contribution.

Not on the roadmap: reshaping the checkpoint to 4 KV heads. With 2 KV heads,
vLLM already replicates one head per rank at TP4. Duplicating heads in the
weights changes nothing, since the per-rank floor of one full head stays, and
real extra heads would mean retraining. The KV lever is dtype, not head count.

## Fabric notes

`.env.example` ships the fabric this was developed on: dual-rail RoCEv2, one
PF per QSFP cage and one per PCIe domain (GB10's ConnectX-7 is socket-direct,
and using two PFs of the same cage just splits one wire), DSCP 26 mapped to
TC3 on the switch, MTU 9000. Get `NCCL_IB_HCA` or `NCCL_SOCKET_IFNAME` wrong
and you hang at rendezvous with very little in the log. Check names with
`ibdev2netdev` and verify the fabric with `ib_write_bw` before blaming the
recipe.

## Credits

- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX):
  the single-node GB10 vLLM patch stack this builds on (vendored, Apache-2.0),
  itself carrying work by @Saren-Arterius (FLA fixes, PLE gather fast path,
  state-copy guard, fp8 conversion), @AndreasKaratzas (vllm#50729), @k3dani
  (the persistent_topk diagnosis), and @jschmied (reproduction, native-offload
  fixes, concurrency measurements).
- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
  the 4-Spark operational playbook. The unconditional flusher, the gate
  discipline, worker-first launch, and the image-ID rule all come from there.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks):
  the first multi-Spark serve of this model (SGLang TP2) and the GB10
  unified-memory crash post-mortems.
- The Qwen team at Alibaba for the model,
  [RadixArk](https://huggingface.co/RadixArk) for the NVFP4 checkpoint, and
  vLLM for the day-0 image and engine.
