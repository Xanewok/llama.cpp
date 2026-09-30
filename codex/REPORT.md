# MTP round attribution and capturable internal AllReduce

## Conclusion

The opt-in attribution patch is complete and marked **DO NOT MERGE**. `LLAMA_ROUND_TRACE=<path>` records server decode ranges, graph reuse, meta rebuilds, CUDA graph submissions, internal AR calls/waits, host transfers and host sampling. CUDA builds also emit matching NVTX ranges. No sampling arithmetic, RNG consumption, backend selection or CUDA synchronization was changed.

Observed CPU result: base, tracing disabled and tracing enabled produce identical token/logit bytes in single-CPU and Meta(CPU,CPU) runs. CUDA compilation passed; CUDA execution, GPU identity, Nsight correlation and current-stack critical-path attribution remain unverified. This task supplies instrumentation and queue scripts, not a GPU speed result.

Inference, high confidence: capturing the internal AR/meta segment is feasible in principle but requires a communication-plan and replay-state contract. Removing `cudaEventSynchronize` or capturing the existing callback unchanged is unsafe. The callback currently chooses slots/arrival tokens on the host, and the wait protects peer reads before scratch reuse.

M = measurement or observed source fact; D = arithmetic/inference; G = forecast. Source lines, hashes, phase names, seeds, configuration flags and requested sample thresholds are identifiers/configuration rather than performance results.

## Provenance and isolation

Worktree: `/home/xanewok/repos/round-trace/llama.cpp`, branch `topic/round-trace`, base `0dba27b9002b8347c6f14a9f4a1651b3478065ef`. The instrumentation commit is `dfeef79179c2aa86994c9e369b5fa925b6f58a6a`. The frozen trace binary was built from that commit. Earlier `68d99c67d` timing budgets are background, not measurements of this descendant.

`/home/xanewok/repos/round-trace/local.git` is a local shared-object bare clone. Its worktree metadata, refs and index are inside the authorized directory. A detached base checkout at `/home/xanewok/repos/round-trace/up` supplies the untouched CPU control. Directly registering a worktree in the campaign repository would have modified its Git metadata, so that approach was avoided. The initial local pack-based clone failed on a missing promisor object; shared local cloning succeeded without fetching. Other repositories were read-only.

Read evidence is pinned in `codex/evidence.json`: both anatomy reports, the sampling/attribution reports, Patch A report/review, CPU model recipe and queue examples. `codex/inputs/PROVENANCE.json` pins the request source and the actual weekend S3 template. `weekend_common.py:62` and the campaign's S3 `launch.json` establish the subset/backend-sampling flags. S3/S4/S5 differ by draft count; this harness takes `K`, default 3. No completed phase-1 winner is assumed while the campaign is running.

No GPU binary, GPU query, server executable or process named `llama-server` was executed. The CPU fixture links `server_context` and uses its in-process task queue; it opens no listener and sends no network request. Build/test/hash work used nice priority 19, builds used `-j8`, and inference used at most 8 compute threads (4 per simple backend for meta).

## Instrumentation contract and touch points

| Data | Source location | Meaning |
| --- | --- | --- |
| V decode begin/end | `tools/server/server-context.cpp:3817` | The `llama_process` call; the existing following synchronize is outside the decode range |
| C decode begin/end | `common/speculative.cpp:1611` | MTP catch-up call; preparation before C remains in the preceding host boundary |
| D1..Dk decode begin/end | `common/speculative.cpp:1713` | MTP forward calls, with their existing per-step sampler/feedback work afterward |
| Graph hit/miss/build | `src/llama-context.cpp:1529`, `:1541` | Actual reuse branch; build interval includes graph reset/build/allocation |
| Meta rebuild | `ggml/src/ggml-backend-meta.cpp:2062`, `:2077` | UID miss; timer covers reconstruction and reservation of the simple-backend plan |
| CUDA instantiate/launch | `ggml/src/ggml-cuda/ggml-cuda.cu:2747`, `:4500`, `:4508` | Successful API submissions, including re-instantiation after an exec-update failure; grouped by CUDA device |
| CUDA capture/direct submission | `ggml/src/ggml-cuda/ggml-cuda.cu:4328` | Capture/evaluation intervals; direct submissions have their own `cuda_submit` kind |
| Internal AR call/wait | `ggml/src/ggml-cuda/allreduce.cu:374`, `:755` | One logical AR call; one wait event per existing `cudaEventSynchronize` invocation in slot acquisition |
| Input upload/readback | `ggml/src/ggml-cuda/ggml-cuda.cu:791`, `:801`, `:832`, `:879`, `:891`, `:2531`, `:2542`, `:2553`, `:2566` | Actual CUDA tensor-transfer paths, including batched and 2D transfers |
| Host sampling | `common/sampling.cpp:795`, `:939` | Row sampling and rejection wrapper; nested intervals use their union, not their sum |
| Other output synchronization | `ggml/src/ggml-cuda/ggml-cuda.cu:2632` | Existing backend stream synchronization; not automatically classified as pure sampling or readback |
| JSONL/NVTX lifetime | `ggml/src/ggml-round-trace.cpp:102`, `:118`, `:128`; `tools/server/server-queue.cpp:280` | Submission-thread state, phase identity, and final shutdown flush |

Schema 1 emits one JSON object per V..next-V round, with `pid`, `tid`, `round`, `next_begin_ns`, `phases` and `events`. Each phase carries decode timestamps, token count, result and grouped `counters`. Each counter has `kind`, `device`, `count`, `bytes`, `host_us`. Times use CLOCK_MONOTONIC nanoseconds; `host_us` is the union duration for a kind/device/phase. The first prefill V and final partial round are retained as evidence; analysis selects complete `V,C,D1..Dk` shapes with `V.n_tokens == k+1`.

The environment gate is read once. Disabled tracing performs no trace allocation, file I/O, clock read or NVTX call. Hooks still execute an enabled check; zero behavior change is a semantic claim, not a claim of zero machine instructions.

Input `count` counts logical transfer regions, and `input_api` counts CUDA upload APIs. A CUDA batch call can carry several regions; 2D byte counts exclude padding. Counts/bytes are actual device-side tensor-transfer submissions and therefore include meta replication. Peer copies and AR staging transfers are not host model inputs/readbacks. The current dense Swift route is covered; the separate MoE operator's private host-ID staging is outside this diagnostic's scope.

AR wait count is API invocation count, including calls whose event is already complete. Its duration measures host time inside that existing call, including active GPU work. It is not exposed idle or a removable saving. Host sampling starts after the usual initial context synchronization; lazy full-logit fallback inside a sampler can still include transfer/wait time. Those child intervals are retained, and exclusive attribution separates them.

File writes occur before the next V range, with a final flush on queue exit. `next_begin_ns` is the pre-write boundary timestamp, while the next V's `begin_ns` is after the write. The analyzer uses actual successive phase starts, so serialization cost remains in the host residual. Enabled tracing can perturb scheduling and allocation; use the overhead A/B described below.

## CPU evidence

`codex/run_cpu.sh` passed against `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. It builds the untouched base and the patch, compiles the same offline server fixture against each, and runs `up`, `off`, `on` for single and meta. Each output contains M48 generated token IDs, one final full target-logit row and M6 teacher-forced full MTP-logit rows. The MTP raw-logit fixture uses a single CPU backend in both modes; the server generation itself exercises meta when selected.

| Mode | Identical bytes M per arm | SHA256, all three arms |
| --- | ---: | --- |
| Single CPU | 6,953,160 | `29cfcb6376b6d9a4b213e6d62b811fafba3e07f2fff6e65806cbb16ecee2e916` |
| Meta(CPU,CPU) | 6,953,160 | `ea6c93c51c7b7ca218fe8a0660d04239ee36793bfa014e0d876ac5caeeebf38f` |

Token-byte and logit-byte perturbations are rejected by the identity control. This is same-seed evidence for the tested CPU workload; it is not a universal GPU sampling proof. Target and draft sampling code/RNG logic is unchanged.

| Mode | up wall seconds M | off wall seconds M | on wall seconds M |
| --- | ---: | ---: | ---: |
| Single CPU | 23.499650 | 10.382491 | 10.306313 |
| Meta(CPU,CPU) | 13.292355 | 13.260421 | 13.305656 |

These times include model loading, server generation and the separate MTP fixture. The first single-CPU run had different cache/loading conditions; this sequential identity test is not a latency A/B. No speed inference follows from its timing differences.

The on-path JSONL validates for both modes. With broad CPU diagnostic filtering (`--skip 0 --trim 1`), the analyzer retains M14 single and M15 meta rounds. Their V-to-V medians are M248.453 ms and M409.569 ms. In those retained sets, C and D1 each miss M14/M15 times; D2 and D3 each hit M14/M15 times. Meta records M15 C rebuilds and M15 D1 rebuilds. This establishes counter plumbing and the alternation mechanism on CPU, not its GPU cost.

`codex/test_attrib.py` passes M9 rejection controls (reversed decode, duplicate round, wrong count, negative bytes, cross-phase event, summed nested sampling, wrong owner, insufficient steady rounds, mismatched NVTX label), plus nested-scope and process-correlation collision checks. Its synthetic SQLite uses Nsight's tagged global PID/TID representation and checks the D3->next-V input interval. `codex/test_harness.py` passes M5 execution/frozen-file controls and proves that plan mode performs no subprocess or socket calls.

Evidence artifacts: `codex/cpu-results/{0,1}.jsonl`, `{0,1}-attrib.json`, `summary.json`; local build/test logs remain alongside them or in `codex/*.log`. Binary CPU outputs are local, ignored artifacts with hashes in `summary.json`.

Reproduce from this worktree:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_cpu.sh
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_attrib.py
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_harness.py
CUDA_VISIBLE_DEVICES='' nice -n 19 bash -n codex/trace.sh codex/gpu_ab.sh codex/run_cpu.sh
```

The CPU script creates a fresh result directory by default; `CPU_OUT` may select another directory inside `codex`. No external scratchpad script is executed.

## CUDA build and frozen artifacts

Both initial and final CUDA builds passed. The final build command was:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 cmake -B build-cuda \
  -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_NCCL=OFF -DGGML_CCACHE=OFF \
  '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
CUDA_VISIBLE_DEVICES='' nice -n 19 cmake --build build-cuda --target llama-server -j8
```

The compiler was forced before CUDA language enablement so CMake skipped its CUDA ABI probe. The existing CMake logic maps 120 to 120a. CUDA 13.2 NVTX headers are linked into `ggml-base` through their host-only interface. No compiled CUDA executable was run.

`codex/frozen/trace/{bin,SOURCE,SHA256SUMS}` contains the renamed `round-trace` executable and M22 shared-library names, including internal symlinks. `bin/SHA256SUMS` verifies the entire M23-file set. `codex/frozen/up` is a read-only copy of the campaign's frozen base, also renamed. Manifests/SOURCE are committed; the binaries/libraries are deliberately ignored local artifacts, so preserve this worktree for the queued run. No file named `llama-server` exists inside the frozen snapshots.

## Parent-only trace and attribution

`codex/trace.sh` defaults to a read-only plan. Execute requires both `--execute` and `RUN_GPU=1` with nonempty selected `CUDA_VISIBLE_DEVICES`. The harness verifies frozen bytes and complete file sets, copies them to a fresh in-worktree OUT, verifies the model SHA, uses the pinned weekend template and subset vocabulary, checks selected-GPU memory, occupied port and boot Xid, bounds wall time, and stops only retained exact PIDs whose start times still match. Its Nsight session is owned by that boot.

The server flags match the weekend subset arm: context 102400, target KV q8_0/q8_0, draft KV f16/f16, tensor 50,50, batch 2048, ubatch 128, probabilistic draft, draft backend sampling, target backend sampling and 8 threads. Request settings are temperature 1, top-k 20, top-p 0.95, min-p 0.05. `K` defaults to 3.

One boot under `nsys launch` runs warmup, then repeated unprofiled think/tool requests until each case has at least the requested 40 retained rounds. Capture starts afterward with `nsys start`; both cases again collect at least 40 retained rounds before `nsys stop` and SQLite export. The env gate is set at boot because it is cached. Request responses, authoritative round metrics, JSONL windows, commands, hashes and both host/GPU attribution outputs are retained. Insufficient rounds or paired-GPU gaps fail the run. Default G1200 seconds is a queue allocation, not a runtime estimate.

Only the parent may execute, after the campaign permits GPU work:

```sh
RUN_GPU=1 CUDA_VISIBLE_DEVICES=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f \
  K=3 bash codex/trace.sh --execute
```

`attrib.py` prints separate decode-call and host-boundary tables without SQLite. It rejects broken JSONL/counter ownership and partitions nested intervals exclusively into inputs, readback, AR wait, other sync, graph build, sampling, other CUDA API and host residual. Residual wall time is an upper bound on host CPU work: CPU descheduling and uninstrumented work are not measured separately.

With SQLite, matching NVTX labels align clocks; process-scoped runtime correlations assign executed kernels to phases. A GPU boundary runs from the last previous-phase kernel on either GPU to the first next-phase kernel on either GPU. It is a kernel boundary, not an assertion that copy engines were idle. `device_idle_us` additionally subtracts the union of kernel, memcpy and memset activity in that interval. Asynchronous readback API time and subsequent synchronization remain separate; JSONL alone cannot isolate device-copy duration from a pending compute wait. Missing correlations/clock alignment are reported or rejected; synthetic coverage does not establish compatibility with the parent's eventual export.

Example offline analysis:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/attrib.py PATH/rounds.jsonl --k 3 --min-rounds 40
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/attrib.py PATH/rounds.jsonl --sqlite PATH/capture.sqlite --k 3 --json
```

`codex/gpu_ab.sh` uses frozen `up,patch,patch,up` boots with identical warmup/tool/think requests, seed, flags and authoritative counters. It checks completion/finish/counter identity, excluding only generated tool-call IDs. Both arms default to tracing off, isolating the compiled instrumentation's off-path effect. `--trace-patch` enables the patch's trace to measure total diagnostic perturbation against the unchanged base. It records measured generation times and D milliseconds per authoritative round. Chat identity is not token/q-byte GPU identity. Execute mode, real Nsight collection and GPU A/B were not run here.

## Capturable internal AllReduce: report only

### Exact blockers and smallest safe design

| Touch point | Observed contract | Required capture change |
| --- | --- | --- |
| `allreduce.cu:235`, `:319`, `:365` | Host `call_count` chooses a two-slot ring; acquisition waits for both ranks' `ev_pool[slot].ker` before reuse | Preassign graph-boundary scratch, or implement device completion/consumption ownership before a slot can be written again |
| `allreduce.cu:162`, `:165`, `:378`, `:934` | The kernel publishes and waits for an integer token chosen by the host | Generate a fresh replay epoch on the device, paired across ranks and combined with boundary/chunk identity; no captured constant token may match a stale replay |
| `allreduce.cu:343`, `:487`, `:507`, `:931` | Arrival and small-path payload storage are indexed by slot | Persistent addresses and bounded generations; reset/wrap must occur only after both ranks have consumed old data |
| `allreduce.cu:616`, `:634`, `:669`, `:686`, `:707` | Copy-engine path uses host slot acquisition plus cross-stream peer-read and scratch-kernel fences | Retain those producer/consumer dependencies as capturable event/node ordering; initialize and normalize warmup validity state before capture |
| `ggml-backend-meta.cpp:2528`, `:2551` | The host dispatches per-device subgraphs then invokes communication | A backend-owned immutable communication plan with paired-rank replay; retain generic fallback for unsupported shapes/backends and exact gathers |
| `ggml-cuda.cu:4578`, `:4585`, `:4588` | Capture is inside one simple CUDA backend, below meta | Enclose a stable per-device draft segment, including its communication nodes, rather than nesting the present simple-backend capture/launch driver |
| `common/speculative.cpp:1714`, `:1734`; `src/llama-context.cpp:1542` | Draft continuation still consumes host proposals/hidden feedback and graph inputs | Stable graph/input/sampler ownership and device successor state are separate prerequisites for capturing multiple draft steps |

Inference: a first experiment should capture a fixed, warmed single-draft meta plan with internal AR, not the complete MTP loop. Allocate all scratch/events/arrival state before capture; pair both rank submissions, preserve stream dependencies and the existing F32/F16/BF16 reduction/cast order. Prove repeated replay, then address draft continuation separately. The two-slot payload offsets already separate overlapping small-path calls; the arrival ring does not itself acknowledge that the peer finished reading. Completion must cover all kernel blocks and both ranks before overwrite.

A device epoch must advance exactly once per paired replay, with a documented cancellation/reset/wrap contract. Independent host increments baked into graph kernel parameters will not advance on replay. Reusing the same constant token can let a fast rank read stale peer payload. Resetting arrival flags independently can erase a live peer signal. Device-side polling must not deadlock when one rank is late, cancelled or resource-starved; paired launch and bounded fallback/error behavior need GPU validation.

A larger ring is neither capturability nor an exact device proposal contract. It can allow the host to queue farther ahead while all the existing waits/operations remain semantically present. Copy-engine host buffers and device scratch still have their own reuse fences.

### Smallest causal diagnostic

No pool-size knob or AR engine change was implemented. For a subsequent isolated diagnostic, add a validated active ring size (default 2; diagnostic 8) used consistently by acquisition modulo/lap checks, event creation/free, arrival allocation and small-path payload allocation. Keep every existing synchronization and reduction operation. A compile-time pool-size-only pair is smaller still, but requires distinct frozen binaries. Do not change only the modulo or suppress waits.

Going from pool 2 to pool 8 adds D12 MiB of mapped payload storage across the ranks, D6 KiB of arrival storage and D1572 CUDA event objects with the current eager event allocation. These are source-derived resource counts, not measured resident memory. Exclude allocation/warmup from steady comparisons.

Queue otherwise identical frozen pool-2/pool-8 arms at the same k, subset, seed, flags and request order. Use trace-off runs for net period and matched trace-on captures for mechanisms. Check token/candidate/q identity, authoritative rounds/acceptance, Xid, peak memory and the AR wait histograms. Include a deliberately stale epoch/early-overwrite control in a later capture test, which must fail identity or timeout cleanly.

Falsification: if wait duration/frequency falls substantially while V-to-V time and kernel-boundary idle do not improve beyond boot variation, host slot waits are primarily backpressure on already active GPU work in this configuration. They are not the dominant removable gap. If a larger ring merely moves waits to `llama_synchronize` or leaves the same submission spacing, whole-loop capture has no demonstrated saving from this experiment. If waits do not fall, the diagnostic did not change effective look-ahead and is inconclusive. Unchanged host-gap time alone cannot falsify all capture benefits: reducing repeated launch/submission overhead is a separate mechanism.

### Effort and upstream shape

G0.5-1 engineer-day for the bounded pool diagnostic and queue validation. G3-5 engineer-days for a small-path paired-rank capture prototype after device epoch/slot ownership is specified. G2-4 engineer-weeks for an upstream-quality communication plan covering copy-engine chunking, lifetime/reset, cancellation/fallback and replay conformance. These estimates assume access to the two GPUs and existing model fixtures; CPU-only compilation cannot validate the cross-device protocol.

Prefer a narrow backend communication-plan/capture capability with backend-owned persistent resources and explicit replay validity. Keep generic meta fallback and exact gather behavior. Do not export an AR pool performance knob as the final optimization or import Patch A's unvalidated plan/container ownership. The full draft segment also needs correct proposal q, sampler binding/RNG state, KV metadata and hidden/token alignment; communication capture alone does not remove host validation or uploads.

## Risks and open evidence

The collector is Linux-only and assumes inference/backend submission stays on the server queue thread, which `server-queue.cpp:235` explicitly preserves. Concurrent contexts have separate thread-local records and serialized writes; the analyzer requires complete shapes and matching thread/process identity. Multi-slot aggregation, non-MTP drafters, multiple MTP heads and container PID translation have not been validated. Failed decode ranges cause analysis failure rather than a performance table.

Open: current-stack paired-GPU trace; real exported NVTX/CUPTI compatibility; trace-on overhead; GPU token/q identity; AR wait overlap versus exposed gaps; pool diagnostic; replay epochs/peer consumption; cancellation and event-capture legality; safe persistent graph/sampler/input ownership; peak memory. The old M3.3 ms host-gap budget and old AR wait totals must not be promoted to current-stack savings.
