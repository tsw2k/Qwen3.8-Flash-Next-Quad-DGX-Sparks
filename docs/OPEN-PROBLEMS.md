# Open problems

Live list. An entry leaves this file only with a gate-passing fix or an
upstream resolution.

## 1M-lane wedge: the Nth deep prefill hangs the GPU stream (OPEN)

**Symptom.** On the 1M YaRN lane (`YARN=1 CTX=1048576 MNBT=1024
GPU_MEM=0.70`), the engine serves flawlessly — until the ~3rd request whose
prompt exceeds ~100k tokens. That request never completes: a worker rank's
GPU stream stops advancing, the head's `sample_tokens` RPC times out, and
vLLM v1 declares the engine dead. `/health` is 200 right up to the fatal
request. Requests under ~32k never trigger it, in any quantity. On the 262k
native lane the same pattern needed ~5 deep prefills to fire, so the lane
boundary is about rate, not immunity.

**What it is NOT** (each eliminated by a dedicated boot, one variable at a
time, on a freshly rebooted fleet):

| Eliminated | Evidence |
|---|---|
| Prefix caching / mamba `align` interplay | dies with `PREFIX_CACHE=0` too |
| MTP speculative decoding | dies with `MTP=0` (a boot later, but dies) |
| CUDA allocator fragmentation | dies with `expandable_segments:True` |
| Page-cache starvation (the GLM lesson) | dies with an unconditional flusher running through serving |
| Prefill chunk size | `MNBT` 8192 → 1024 moved the wall deeper but did not remove it |
| A competing supervisor (see below) | dies with every other workload and systemd unit stopped |
| Fabric | all NCCL links healthy; no async events in the window |

**The stack.** py-spy captured at hang time, identical on every surviving
worker rank:

```
Thread (active): "MainThread"
    forward (vllm_ple_mmap.py:263)        # ids.to("cpu") — stream sync point
    forward_impl (ple_layer.py:438)
    _lookup_impl (vllm_ple_mmap.py:407)
    ...
    forward (qwen3_8_flash_next/nvidia/model.py:464)
```

Line 263 is a device-to-host copy — a CUDA stream synchronization. The CPU
thread is parked at the first sync point after the wedge; **the actual hang
is a device-side kernel enqueued earlier in the forward** (GDN / QSA /
EP all-to-all are the candidates), spinning and cascading to all ranks
through the collectives. The PLE mmap gather is where the stack points, not
where the bug lives.

| PIECEWISE CUDA-graph replay | dies identically with `--enforce-eager` |

**Remaining untested variable:** the PLE mmap patch itself (lane A). The
stack parks at its stream-sync point; swapping to lane C
(`VLLM_PLE_CPU_OFFLOAD` — at TP4 the vocab-sharded table is only ~13 GB of
pinned host per node) removes the whole patch from the equation and is the
next planned experiment. If lane C dies too, this is a GDN/QSA/EP kernel
bug on sm121 at long context, and it goes upstream with the repro and the
stack.

**Meanwhile:** the 262k native lane is the shipped default — it passed the
full gate suite, the bench sweep, and multiple deep prefills. Treat the 1M
lane as: boots, 4.78x pool of a full 1M request, serves NIAH-128k at all
depths once per boot — not yet a config that works (a config that answers
one deep prompt is not a config that works, either).

## Lessons already promoted to defaults (closed here)

- Expert parallelism is mandatory at TP4 (NVFP4 MoE padding) — launcher default.
- `--ulimit nofile=1048576` (128 PLE mmap shards + 4-node sockets) — launcher default.
- Never hardcode `NCCL_IB_GID_INDEX` — auto-select, see `.env.example`.
- `MNBT` ≤ 2048 for long-context work — `.env.example`.
- **Audit systemd units before a multi-hour ladder.** A fleet supervisor left
  over from a previous deployment on the same nodes re-armed itself after a
  node reboot and fought this deployment for the master port and memory for
  several boots. `systemctl list-units | grep -i <anything model-shaped>`
  before you trust your failure data.
