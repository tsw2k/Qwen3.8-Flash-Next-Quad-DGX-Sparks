# Open problems

Live list. An entry leaves this file only with a gate-passing fix or an
upstream resolution.

## 1M-lane wedge: the Nth deep prefill hangs the GPU stream (OPEN)

Symptom. On the 1M YaRN lane (`YARN=1 CTX=1048576 MNBT=1024 GPU_MEM=0.70`)
the engine serves flawlessly until roughly the third request whose prompt
exceeds ~100k tokens. That request never completes: a worker rank's GPU
stream stops advancing, the head's `sample_tokens` RPC times out, and vLLM v1
declares the engine dead. `/health` returns 200 right up to the fatal
request. Binary search later put the wall in (73,728 .. 77,824] tokens:
nine requests at 73,728 run clean on one boot, one request at 77,824 kills
the engine, and 76,800 = 48 layers x 1600 (the mamba block) sits inside
that window. Below the wall the engine is indifferent to volume. Above it,
death comes within the first few requests, faster the closer the prompt is
to the wall. While wedged, nvidia-smi shows the stuck worker at ~96% GPU
utilization: the kernel spins, it does not wait. Details in the issue
thread.

What it is NOT. Each row below took a dedicated boot, one variable at a time,
on a freshly rebooted fleet:

| Eliminated | Evidence |
|---|---|
| Prefix caching / mamba `align` interplay | dies with `PREFIX_CACHE=0` too |
| MTP speculative decoding | dies with `MTP=0` (a boot later, but dies) |
| CUDA allocator fragmentation | dies with `expandable_segments:True` |
| Page-cache starvation (the GLM lesson) | dies with an unconditional flusher running through serving |
| Prefill chunk size | `MNBT` 8192 to 1024 moved the wall deeper but did not remove it |
| A competing supervisor (see below) | dies with every other workload and systemd unit stopped |
| Fabric | all NCCL links healthy, no async events in the window |
| PIECEWISE CUDA-graph replay | dies identically with `--enforce-eager` |

The stack, captured with py-spy at hang time, identical on every surviving
worker rank:

```
Thread (active): "MainThread"
    forward (vllm_ple_mmap.py:263)        # ids.to("cpu"), a stream sync point
    forward_impl (ple_layer.py:438)
    _lookup_impl (vllm_ple_mmap.py:407)
    ...
    forward (qwen3_8_flash_next/nvidia/model.py:464)
```

Line 263 is a device-to-host copy, which synchronizes with the CUDA stream.
The CPU thread is parked at the first sync point after the wedge. The actual
hang is a device-side kernel enqueued earlier in the forward (GDN, QSA and
the EP all-to-all are the candidates), spinning and cascading to all ranks
through the collectives. The PLE mmap gather is where the stack points, not
necessarily where the bug lives.

Remaining untested variable: the PLE mmap patch itself (lane A). Swapping the
PLE path removes that patch from the equation. Both alternatives have been
tried:

- Lane C (`PLE_MODE=offload`, native `VLLM_PLE_CPU_OFFLOAD`) does not survive
  contact with a 128 GB node: the offload worker is TP-unaware, loads the
  full table in one process, and its load peak (~85 GB anon RSS observed,
  fp8 table plus a temporary upcast) draws a global OOM kill at any
  `GPU_MEM`. Not viable here without an upstream loader fix.
- Lane B (`PLE_MODE=resident`, the stock vocab-sharded table, ~12 GiB per
  rank on GPU) boots at `GPU_MEM=0.78` with a 1,692,092-token pool and dies
  on the third deep prefill, exactly like every other configuration.

Lane B settled it: the mmap patch is not the trigger. The wedge reproduces
with three different PLE paths, so the bug sits in the shared forward
(GDN, QSA or the EP all-to-all) on sm121 at long context. Reported upstream: first in PR comments
(https://github.com/vllm-project/vllm/pull/53896#issuecomment-5477335926),
then as a proper issue after the PR merged:
https://github.com/vllm-project/vllm/issues/54629

Related report, different engine: a withdrawn 2x GB10 SGLang profile for the
same checkpoint hit "invalid sampling probabilities" with CUDA asserts and
Xid 43 under multi-turn agentic traffic, with the same trigger shape (long
prefill, decode, tool turn, longer prefill). The author fixed a
request-lifecycle branch and still withdrew the profile over a remaining
state-correctness defect:
https://forums.developer.nvidia.com/t/381836
Two engines, one hardware platform, one traffic shape. Our matrix adds one
fact theirs lacks: the failure survives MTP=0, so speculative decoding is
not required for it.

Meanwhile the 262k native lane is the shipped default: it passes the full
gate suite and the bench sweep, and traffic under ~32k has never triggered
the wedge. Sustained deep-prefill traffic can still hit it there, a few
requests in, so a production endpoint should run `fleet_watchdog.sh`:
recovery from a wedge death is an orchestrated relaunch, about 15 minutes.
The 1M lane boots and holds 4.78x of a full 1M request in KV, but treat it
as a demo until the wedge closes.

## Lessons already promoted to defaults (closed here)

- Expert parallelism is mandatory at TP4 (NVFP4 MoE padding). Launcher default.
- `--ulimit nofile=1048576` (128 PLE mmap shards plus 4-node sockets). Launcher default.
- Never hardcode `NCCL_IB_GID_INDEX`. Auto-select; see `.env.example`.
- `MNBT` of 2048 or less for long-context work. See `.env.example`.
- Audit systemd units before a multi-hour ladder. A fleet supervisor left
  over from a previous deployment on the same nodes re-armed itself after a
  node reboot and fought this deployment for the master port and memory for
  several boots. Run `systemctl list-units | grep -i <anything model-shaped>`
  before you trust your failure data.
