# Owned input staging

## Conclusion

The scheduler can hand host inputs to a backend that owns their bytes until upload completion. CUDA uses pinned slots and bidirectional event ordering instead of a host wait before the destination write and a host wait after upload. Meta forwards mirrored inputs to every rank. Unsupported paths retain ordered synchronous uploads. Sampling code, RNG calls, proposal probabilities and rejection arithmetic are unchanged.

GPU speed and GPU proposal/q identity are unmeasured. The inherited forecast is G0.15-0.30 ms/round at K=3 and G0.25-0.45 at K=5, against the trace reader's D0.338-0.340 ms exposed inside input scopes. Additional event APIs, conservative fallbacks and slot backpressure can erase the gain. Queue the frozen trace-off A/B before adopting this change.

M = observed/measured; D = arithmetic or an invariant derived from source; G = forecast. Revisions, seeds, K, dimensions, issue IDs and command options are identifiers/configuration.

## Contract and source

Base is `7fc0244d7`, the supplied graph-reuse-safe stack. Production commits are `5f73e984d` and `ec17de5fb` on `topic/input-staging` in `codex/local.git`. The worktree's original `.git` points outside it and is read-only. Use `git --git-dir=codex/local.git`; no external branch was changed. `codex/input-staging.bundle` carries this topic with the base as prerequisite.

- `ggml/src/ggml-backend-impl.h:168`: optional `set_tensor_batch_owned` owns source bytes before returning and orders destination writes with compute. Refusal must not submit the requested transfers. An empty list queries capability. The backend ABI version increases; other backends explicitly initialize the appended hook to null.
- `ggml/src/ggml-backend.cpp:1767`: batch host user inputs and transient compute outputs from synchronous producers, then try owned upload before the destination-slot fence. CPU token-embedding `GET_ROWS` output uses this path. Refusal retains the old fence and synchronous batch API. Host weights, asynchronous host producers, device-source copies and cross-split ordering retain their paths.
- `ggml/src/ggml-cuda/common.cuh:1451`: D2 slots per CUDA backend, each with pinned storage, capacity, upload event and pending state. Storage is backend-owned rather than graph- or caller-owned.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2557`: validate transfers before any upload; query/wait the selected slot's old event before overwriting or freeing it; allocate with `cudaHostAlloc`, with synchronous fallback on failure. Completed slots need no host event wait.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2607`: record prior compute, make the upload stream wait for that event, queue copies, record the slot upload event, then make compute wait for upload. Pinned storage has no consumer after its upload event. Device destinations remain protected by compute-to-upload ordering. CUDA peer copies run on their source stream; concurrent compute and AR copy streams join compute, so the handoff follows those reads too (`ggml-cuda.cu:2721`, `ggml-cuda.cu:4526`, `allreduce.cu:691`). Ring reuse never relies on MTP sampling to have synchronized first.
- Non-overlapping ranges keep `cudaMemcpyBatchAsync`. Overlapping ranges, including view aliases, use ordered copies on the upload stream. CUDA without the batch API uses the same ordered path. HIP/MUSA retain synchronous fallback.
- `ggml/src/ggml-cuda/ggml-cuda.cu:703`: teardown waits pending uploads before destroying their events or freeing their host storage. Destination buffers retain the existing asynchronous compute lifetime contract.
- `ggml/src/ggml-backend-meta.cpp:2042`: preflight every rank's capability and mirrored split state before writing. A later refusal drains all ranks, performs ordered synchronous fallback, then reports completion. Non-mirrored inputs use the scheduler's existing fallback.
- `ggml/src/ggml-backend.cpp:2220` and `src/llama-context.cpp:1534`: bypass alternation's host fence only if both graphs have valid allocation UID/epoch, one scheduler copy, no evaluation callback, host-backed inputs, and synchronous executing backends or capable asynchronous backends that cannot read mutable host buffers directly. Direct device inputs, integrated-device host access, pipeline paths and unsupported backends retain the fence. Miss/reset/allocation fences remain.

D: operations retain the existing serialized backend/scheduler-family contract. There is no sampling policy, server-specific upload path, device data buffer or allocator lifetime exception. Trace counters `input_owned` and `input_slot_wait` identify owned batches and incomplete-slot waits; `input_fence_staged` / `input_fence_sync` identify the alternation decision.

## CPU evidence

M: eight synthetic seeds and four real-model seeds, K=3/K=5, CPU and Meta(CPU,CPU), up/patch/patch/up, twelve rounds per run. Every recorded sampled token, ordered candidate ID, probability/q and logit byte matches `7fc0244d7`. Per configuration/arm: synthetic M16 runs and M144 retained rounds; real M8 runs and M72 retained rounds, after discarding the first three rounds from timing. Both arms retain C/D1 graph hits and D0 build/meta-build time in those retained phases. Raw bytes include warm-up rounds.

The teacher-forced fixture follows V -> C -> D1..Dk. The original `Qwen3.5-4B-UD-Q4_K_XL.gguf` has a tied target head unsupported by baseline TP backend sampling, so its TP target uses the existing worktree-only untied-head adaptation with byte-identical copied embedding blocks. Drafting uses the original compact MTP head. This is a fixture adaptation; unmodified real-model TP target sampling is not claimed.

Decode wall includes CPU compute and excludes sampling readback/trace flush. Values below are D medians of per-run mean microseconds per round. CPU differences are noisy and do not establish a staging speedup; normal CPU backends take the synchronous fallback.

| Backend / K | Synthetic up D | Synthetic patch D | Real up D | Real patch D |
| --- | ---: | ---: | ---: | ---: |
| CPU / 3 | 882.5 | 879.6 | 157725.9 | 157888.4 |
| CPU / 5 | 1241.0 | 1224.6 | 279914.3 | 282164.6 |
| Meta(CPU,CPU) / 3 | 1805.2 | 1847.0 | 315672.8 | 324194.4 |
| Meta(CPU,CPU) / 5 | 2494.8 | 2467.3 | 420871.6 | 408506.1 |

`codex/input-cpu.cpp` installs test-only synchronous owned-transfer hooks through the CPU device factory. This exercises production Meta dispatch and the capability path against the same base byte records; it does not model CUDA asynchrony. M: four seeds per model and K, Meta(CPU,CPU), twelve rounds per run. Every owned-path byte matches the base. Each run has M18 retained C/D1 alternations taking the staged path and M0 alternation host fences. This includes the real model's synchronous CPU token-embedding prefix. Results and counters: `codex/owned-input-results.json`.

`codex/test_input_staging.py` separately extracts the actual production upload function, slot fields and teardown into CPU delayed queues, without CUDA headers or runtime libraries. Copies execute after source mutation and repeated ring reuse; unordered batches execute in reverse order to expose illegal overlap batching.

M: the intact delayed-queue implementation passes owned-byte, ring, overlap, refusal, allocation-fallback and teardown checks. Removing slot-completion synchronization returns M1 with stale staging. Removing upload-to-compute waiting returns M1 with lost overlap order. Removing compute-to-upload waiting returns M1 with stale staging/destination order. These mutate production statements, not reference output. Results: `codex/input-contract-results.json`.

M: `codex/input-scheduler-contract.cpp` verifies staging of a synchronous producer's transient output without a host fence, unsupported fallback with its fence, direct-host/direct-device/unallocated refusal, Meta(CPU,CPU) rank preflight and both-rank drain on late refusal. Existing graph binding/epoch/container contracts, `test-llama-archs --tp-sampling` and `test-sampling` pass.

M: the real in-process CPU server rejection fixture matches completion token and final target-logit bytes for K=3/K=5 and seeds 1729/2718, with M48 completion tokens per run, both backend samplers, probabilistic drafting and subset vocabulary. It does not record every rejection-stage q. Server hashes: `codex/server-input-results.json`.

M: the full CPU runner passed at the final production source, including its owned-path checks and real rejection cases. No GPU query, GPU executable, process named `llama-server` or network call ran in this task.

Reproduce inside this worktree:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_input_cpu.sh
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_input_staging.py
```

The runner uses `cmake -B build-cpu -DGGML_CUDA=OFF`, `-j8`, nice priority 19 and at most eight compute threads. It never launches `llama-server`, `llama-cli` or a GPU executable. The rejection fixture links the CPU server context and runs its task loop in process. Summaries: `codex/synthetic-input-results.json` and `codex/real-input-results.json`; raw records remain in ignored `codex/input-results/`.

## Frozen parent A/B

M: both CUDA builds complete and are frozen in `codex/frozen/input-up` (source `7fc0244d7`) and `codex/frozen/input-patch` (source `30bea8abd`). Both top-level and binary SHA256SUMS checks pass. SOURCE and manifests are versioned; binary files remain in this worktree. M: dry K=3/K=5 plans make no launch/network call; the inherited mocked harness passes its M12 controls, including frozen snapshot, execute gate, budget, completion/counter identity and Xid rejection.

`codex/freeze_input.sh` builds base and patch sequentially with CUDA 13.2, NCCL off, `CMAKE_CUDA_ARCHITECTURES=120` (mapped by the project to `120a`), `CMAKE_CUDA_COMPILER_FORCED=ON` from the first configure, and `-j8`. Device probing and optional dependency downloads are disabled. The script compiles, links and copies; it never runs a CUDA executable. M: neither build contains a CUDA compiler ABI/device-probe binary, and both generated native-architecture fields are empty. The compiler-ID binary is compiled but not executed. Frozen server binaries are named `round-trace`.

`codex/gpu_ab.sh` is the graph-reuse wrapper with arms changed to `input-up` and `input-patch`. The inherited harness keeps up/patch/patch/up boots, source/binary SHA checks before/after execution, fixed requests/prompts/seeds, trace-off timing, completion and authoritative counter identity as a hard gate, pair/port/process checks, exact PID/start-time shutdown, Xid checks, and maximum configured G2400-second budget. The metric is D `predicted_ms / drafts`. This gate does not prove GPU proposal/q byte identity.

Parent-only, not run here:

```sh
RUN_GPU=1 CUDA_VISIBLE_DEVICES=0,1 XID_SINCE='<parent timestamp>' nice -n 19 bash codex/gpu_ab.sh --execute --k 3
RUN_GPU=1 CUDA_VISIBLE_DEVICES=0,1 XID_SINCE='<parent timestamp>' nice -n 19 bash codex/gpu_ab.sh --execute --k 5
```

## Upstream fit and risks

This follows the documented #15749/#15750 feedback: a backend ownership capability rather than server hacks. It preserves the existing batch API for synchronous callers and extends the scheduler/meta boundary. The production delta is D22 files, +276/-8 lines, including mechanical null initialization of the new hook in other backends. The API/ABI addition and CUDA stream/events still require maintainer review. Issue status was not refreshed under the no-network rule; this is a private prototype and no submission was made.

Open: GPU throughput/exactness, CUDA graph capture/replay, multi-context stress, slot-wait frequency, conservative-path frequency and pinned-memory high water. D: each backend retains up to twice its maximum observed input payload in new slots, plus descriptors and any existing fallback staging. Prompt-sized allocations remain live until backend teardown. Meta's existing metadata mutex still serializes planning/upload metadata. Direct-host/device, non-mirrored and unsupported paths can keep waits. D2 slots bound ownership and can impose backpressure when uploads lag. CPU identity and delayed queues do not replace GPU runtime validation.
