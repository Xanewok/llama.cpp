Current stack10 candidate results: [REPORT-stack10-cand.md](REPORT-stack10-cand.md).

# stack3

CPU-verified consolidation on `topic/stack3`. Graph retention and prefill pooling default ON; owned input staging defaults OFF. GPU integration identity, speed, and the first-large-prompt latency are unmeasured here. No GPU execution/query, CUDA application, network request, server process named `llama-server`, push, or submission ran.

Labels: M observed/measured, D calculated from measurements, G estimate or hypothesis. Commit IDs, paths, seeds, K values, flags, architecture names, and configured limits are identifiers, not measurements.

## Provenance

Base: `0dba27b9002b8347c6f14a9f4a1651b3478065ef`, the frozen stack2 descendant of `68d99c67d`. `codex/provenance.json` records full IDs and successful `git merge-base --is-ancestor <commit> 0dba27b90` checks. The source topics below come from the read-only `~/repos/qwen38-eval/serving/llama-patches.yaml` and PLAN Lever ledger. These are the actual stack2 copies, rather than the original topic IDs.

| Existing lever | Actual stack2 commits | Source branch |
| --- | --- | --- |
| fmq | `b0118d2ff`, `5926c67a7`, `68d99c67d` | `topic/fattn-mma-quant-kv` |
| input-upload | `e7071647c` | `topic/host-input-upload` |
| verify-sampling | `70036f47d` | `topic/backend-verify-sampling` |
| draft-vocab | `faee46be9`, `b8668a92e` | `topic/mtp-draft-vocab` |
| verify-prefill | `46c388284`, `0dba27b90` | `topic/verify-prefill` |
| resync | `df5f4fa3e` | `topic/fix-server-noncanonical-resync` |
| checkpoint-skip | `740ca4aa0`, `e6f9d8ba2` | `topic/ckpt-skip-draft-kv` |
| checkpoint-pool | `a4adcd97c` | `topic/ckpt-buffer-pool` |
| prefilter | `63cf1afe6`, `6f562b8b9`, `adcbc2f0e` | `topic/sampler-topk-prefilter` |
| probabilistic-draft | `eaefd4746` | `topic/pr27694-probabilistic-draft` |
| masked-mtp-hidden | `2f649b4bc` | `topic/mtp-h-nextn-masked` |

## Integration commits

| Input | Topic commit | Purpose |
| --- | --- | --- |
| `c23324d5f` | `772e1cde3` | Iterative split-state resolution and existing test-alloc regression |
| `dfeef7917` | `b9de10f47` | Required round trace dependency; LLAMA_ROUND_TRACE unset is inert; DO NOT MERGE upstream |
| `7839e00ba` | `67eccbd71` | Shared allocation epochs and retained meta split plans |
| `58ecab9bb` | `aab941526` | Output/catch-up scheduler retention and sampler binding snapshots |
| `eac50e4a4` | `7b177636d` | Fetch deferred logits before graph invalidation |
| `0654a0c0f` | `a4c2bb3d8` | Meta storage helper declarations |
| `79575de0c` | `e67c557cd` | Prefill/allocation trace scopes and required server fixture |
| `19814e139` | `5ac8b1e13` | One spare host arena per meta backend |
| `cd5638edd` | `2f635cf2d` | Hash visited tensors in the ownership traversal |
| `b4dc5515a` | `b22fe8e8b` | Retain graph-array high-water capacity |
| `19759fb85` | `710b61582` | Owned CUDA upload slots and event ordering |
| `6918e7773` | `6fb301b31` | Stage synchronous host producers; capability-checked fence bypass |
| `new` | `735c2342c` | Runtime switches and requested defaults |
| `new` | `f5d476035` | Separate geometric staging growth change |
| `selected fixtures` | `fbb208fdc` | CPU identity, allocation, ownership, and negative controls |
| `selected harness` | `b8a3c2786` | Parent-only GPU identity gate and frozen baseline manifests |
| `new` | `06af71301` | Real memory-fixture parser and exact-PID signal cleanup |

Tests/docs-only source commits were not cherry-picked as a series. Only fixtures used here were copied from `import/graph-reuse-tandem`; the old harness paths and baseline trace assumptions were adapted. No upstream package or external repository was written.

### Conflict resolutions

- `7839e00ba`, `ggml/src/ggml-backend-meta.cpp`: retained the meta fix's `ggml_backend_meta_calculate_split_state` and explicit pending-state vector. Moved the incoming recursive mutex guard to the iterative `ggml_backend_meta_get_split_state(stc, ...)` entry, protecting the complete lookup/clear/DFS/store sequence. The public overload retains its existing guard. The recursive mutex permits nesting. The incoming plan/container retention code merged independently and remains intact.
- `79575de0c`, `codex/reuse-server-cpu.cpp`: modify/delete conflict because its earlier test-only introduction was skipped. Kept the required fixture, then copied its final tandem version and added `CPU_TEMP` to test greedy and seeded requests. No additional production conflict occurred.

## Flags and contracts

| Environment | Default | Effect |
| --- | --- | --- |
| `LLAMA_MTP_GRAPH_REUSE` | `1` | `0` avoids the output child scheduler and requires the active result for a reuse hit. Older same-result reuse remains available. |
| `GGML_META_PLAN_REUSE` | `1` | `0` frees plans rather than retaining the spare host arena. Dependency hashing remains part of the safe retention implementation. |
| `GGML_SCHED_INPUT_STAGING` | `0` | `1` enables owned backend uploads, batching from synchronous host producers, and supported alternating-graph fence bypass. |
| `LLAMA_GRAPH_REUSE_DISABLE` | unset | Existing broad control: `1` forces all llama graph misses. |
| `LLAMA_ROUND_TRACE` | unset | Env-gated private instrumentation; excluded from production upstream packaging. |

The first two switches are read when their context/backend is created. Staging is read at the first scheduler use and is fixed for that process. Sweeps restart the same frozen executable with different environments.

Design evidence: `src/llama-context.cpp:1521` selects a shared-storage output scheduler; `:1533` requires a valid allocation and exact graph-parameter compatibility; `src/llama-graph.cpp:1281` restores sampler bindings before setting inputs; `ggml/src/ggml-backend.cpp:2212` rejects wrong UID/stale epoch in public compute. `ggml/src/ggml-backend-meta.cpp:1221` uses a heap-backed DFS with bounded call stack; `:2231` retains reachable tensor owners under the mutex; `:2766` forces complete topology rebuilding when a spare plan is selected; `:2780` releases container ownership before pooling. Device arithmetic and sampling draws were not changed by the consolidation.

### First-large-prompt staging change

`ggml/src/ggml-cuda/ggml-cuda.cu:2589` now doubles each slot's capacity until the next input batch fits, with overflow-safe growth. The upload-completion wait still precedes reuse/free. If spare-capacity allocation fails, it retries at the exact requested size and leaves the old allocation intact until replacement succeeds.

M: the CPU delayed-queue build of the actual production upload function requires M 16 allocations for M 256 increasing transfers and preserves every copied byte. The exact-size-growth mutant fails `linear staging allocation growth`; the removed retry fails `exact-size fallback refused`; all three event-wait mutants fail. `codex/input-contract-results.json` records M 1 positive implementation and M 5 rejected mutations. This proves allocation and ordering contracts, not GPU latency. G: avoiding repeated pinned allocation as the prefill mask grows may reduce the reported first-prompt penalty; the full reported penalty is not attributed or claimed fixed.

## CPU verification

All builds/tests ran with `CUDA_VISIBLE_DEVICES=""`, nice priority 19, Release CPU libraries, at most `-j8` build workers, and at most eight compute threads (four per CPU rank in Meta). The exact stack2 source was archived into ignored `codex/base-src`; only the baseline test executables link a standalone trace helper. Stack2 production libraries carry no trace overlay; only their test executables link the standalone helper.

Reproduce: `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_cpu.sh`. CMake uses `-DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF`. The command builds but never executes the target named `llama-server`; optional HTTP-capable CPU copies are named `cpu-serving`. Server checks use the in-process `server_context` fixture.

- M PASS: `test-alloc`, M 16 cases, including same/other-branch rebuilds under the M 1 MiB child stack. Loading a separately linked recursive meta-source mutant makes the other-branch child die with signal 11 and the parent exit -6. See `codex/test-alloc.log`, `codex/negative-meta.log`, and `build-negative-meta-recursive/result.log`.
- M PASS: `reuse-contract`, M 3 contracts: sampler binding/uniform bytes and committed draws, public UID/epoch rejection, borrowed meta-input ownership through M 40 container rotations. M 2 binding/epoch mutant libraries exit 1 with the intended failures (`codex/negative-results.json`).
- M PASS: `input-scheduler-contract`, CPU/Meta dispatch, synchronous-producer batching, capability refusal, allocation/direct-input fallback, late-rank drain, and fence decisions. M PASS: `test-llama-archs --tp-sampling` and `test-sampling`.
- M PASS: synthetic M 128 runs and real-model M 64 runs, interleaved stack2/stack3/stack3/stack2, K=3/K=5, CPU/Meta(CPU,CPU), raw token/ID/q/probability/logit bytes identical. M 8 synthetic seeds and M 4 real seeds; M 12 rounds per run, with M 9 retained after trimming. `codex/{synthetic,real}-reuse-results.json` retains counts and derived timings. Baseline build/allocator counters are unavailable because its production libraries have no instrumentation; zero baseline counters mean uninstrumented, not zero work.
- M PASS: `LLAMA_MTP_GRAPH_REUSE=0` removes the M 18 retained C/D1 hits in the synthetic control, yielding M 0, with identical bytes (`codex/reuse-flag-results.json`).
- M PASS: server identity, M 30 runs across M 6 backend/temperature/seed cases, M 2 requests per boot. The five arms are exact stack2, stack3 defaults, staging ON, retained graph reuse OFF, and spare pooling OFF. Every token ID and final-target-logit byte matches. M 6 perturbed-token controls are rejected (`codex/cpu-identity-results.json`).
- M PASS: repeated cold prefill, M 3 arms and M 2 requests per arm. Default target arena creations M 0 per request; `GGML_META_PLAN_REUSE=0` creates M 5, requesting M 87,516,000 arena bytes per request, and fails the allocation contract. Disabling all graph reuse removes hits and preserves bytes (`codex/prefill-summary.json`).
- M PASS: owned Meta input fixture, M 8 runs across real/synthetic, K=3/K=5 and two seeds. Every token/ID/q/logit byte matches the saved stack2 reference. M 18 staged C/D1 decisions per run and M 0 host fences (`codex/owned-input-results.json`).

No arithmetic operator changed. `test-backend-ops` was not run, honoring the explicit prohibition; transport/scheduler behavior is covered by the targeted contracts above.

Fixture: `~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. Direct Meta candidate/q checks use the worktree-only untied target-head adaptation, which duplicates quantized embedding bytes; draft weights remain the original compact head. Server identity uses the original tied-head model: Meta target backend sampling correctly falls back to CPU (`set_sampler` log); the draft still exercises the Meta path. These are CPU contracts, not Swift GPU validation.

### CPU decode timing

D medians of per-run mean decode wall, ms/round, from the M trace intervals. These exclude host sampling/readback outside the decode scope and establish no GPU speed. Host load was uncontrolled.

| Backend / K | Synthetic stack2 D | Synthetic stack3 D | Real stack2 D | Real stack3 D |
| --- | ---: | ---: | ---: | ---: |
| CPU / 3 | 1.111 | 0.926 | 178.460 | 173.811 |
| CPU / 5 | 1.405 | 1.281 | 209.423 | 205.685 |
| Meta(CPU,CPU) / 3 | 2.831 | 1.899 | 251.639 | 248.092 |
| Meta(CPU,CPU) / 5 | 3.706 | 2.691 | 389.689 | 356.203 |

## CUDA freeze and parent GPU item

M PASS: CUDA Release compile/link with toolkit `/usr/local/cuda-13.2`, `CMAKE_CUDA_ARCHITECTURES=120a`, `GGML_CUDA_NCCL=OFF`, and `CMAKE_CUDA_COMPILER_FORCED=ON` to suppress the device/ABI probe. `GGML_OPENMP_FETCH=OFF` and `LLAMA_BUILD_UI=OFF` avoid downloads. Command: `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/freeze_stack3.sh`. No CUDA application was run.

Frozen candidate: `codex/frozen/stack3`, source `b8a3c2786f240c95487f217acda45abf7f9d8bcd`, executable `bin/round-trace`, pinned libraries and two-level SHA manifests. M 23 payload files per arm verify. The renamed stack2 copy and all its libraries match the original read-only weekend manifest exactly. One inherited compiler warning remains at `src/llama-model.cpp:1526` (`-Wsign-compare`); that source file is unchanged from stack2. See `codex/{cmake,build}-cuda.log` and `codex/frozen-check-results.json`.

The parent-only draft is `codex/stack3-identity.toml`, invoked by `codex/gpu_ab.sh` and `codex/stack3_gpu.py`; it reuses the frozen/hash/PID/SSE helpers from graph-reuse. Fixed S5 configuration: Swift 1.5 IQ4_XS, tensor split 50/50, internal AllReduce, target q8_0 KV, draft f16 KV, probabilistic MTP K=5, backend sampling, draft vocabulary subset, `-np 2`. New optimization flags are sanitized per boot. Trace is OFF.

Canary: one staging-ON boot with one roughly 32k seeded request, one short temp-0 request and its exact repeat, plus simultaneous short two-stream requests. Full order: stack2/default/staged/staged/default/stack2. Every boot records all requests before gates; a failed boot gets no performance summary. Full output comparison preserves content/reasoning/function arguments/finish reason as canonical UTF-8 completion bytes, omitting random tool-call IDs and timing fields. It does not capture GPU proposal/q/logit bytes.

Gates: frozen/model/input hashes; per-request cold prompt size/cache and generated-length bounds; at least 40 interior fixed-K SSE intervals (configured minimum) for each sequential request; authoritative counters; two-stream generation overlap greater than the configured 0.5 s and a configured per-stream rate band; exact output comparison; word-bounded kernel fault detection; exact PID plus start time for stop. Concurrent request-local Prometheus deltas overlap, so their pair-wide delta alone is reconciled with both final SSE counters. The summaries retain steady ms/round, cold long-prompt time, request wall, and paired deltas per arm.

M PASS parser replay on real artifacts: `~/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/`, `fixtures/xid-lines.txt`, and the task-named `~/repos/graph-reuse/codex/runs/graph-reuse-tandem/full-a2/k5-1-up/long-tool-s1-1.json`. The short recorded fixture fails the configured 40-round gate. Mutations reject altered outputs, missing/inconsistent counters, missing final chunks, cold-size/cache/length errors, missing overlap, impossible rates, tampered frozen bytes/metadata, empty/faulted journal, missing metrics, forbidden execution, and budgets. Mocked one-boot canary and full interleaving pass without processes or sockets. Transport and actual Swift size/overlap behavior remain real-canary checks. Test-supplied prompt sizes/arrival timestamps are not GPU measurements.

Budget estimates: G 620 s full, G 160 s canary, D sum 780 s. Configured full/canary limits are 960/240 s, total 1200 s (20 min); D 70% margins are 672/168/840 s. Estimates include hashing, snapshots, idle/load, warm-up, requests, stop, and journal checks. The recorded M 8,412-token up prompt took M 10,339.168 ms; extrapolating to 32k is only a G sizing assumption, not a measured prefill rate at that depth. Expiry records `budget`, not a valid speed result.

M PASS: full and one-boot canary dry plans; exact M 65-file pin inventory; `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 /home/xanewok/repos/gpuq/bin/gpuq submit --check /home/xanewok/repos/stack3/codex/stack3-identity.toml` returns `valid`. Only `--check` was executed. The draft has `ab=true`, `retry=fresh`, unique attempt outputs, protected identity/size/hash thresholds, and no editable runtime files.

Additional parent-harness checks read the real `fixtures/nvidia-smi-query.csv` directly, reject malformed/impossible memory rows, and exercise the wait-for-release loop with a busy-then-idle replay. GPUq supplies ownership/utilization debounce, RAM, and boot-ID gates. The termination handler stops the exact owned PID before unwinding HTTP workers, restores the prior handler, and records `interrupted`; its mocked signal control passes. Reproduce parser/control tests with `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_stack3_gpu.py`.

The requested SP/overnight/items destination is outside the worktree and conflicts with hard rule 4. No file there was modified. The parent can copy the validated worktree TOML to `SP/overnight/items/stack3-identity.toml`; all paths and pins already reference this worktree. No GPU submission occurred.

## Risks and open items

- GPU byte identity, CUDA capture/replay, np2 scheduling, and actual steady/request latency remain untested. Run the real canary and full identity gate before serving stack3. CPU identity does not prove GPU numerical identity.
- Geometric growth reduces allocation frequency in the CPU contract. It does not establish the cause or removal of the reported first-large-prompt penalty. Staging stays OFF.
- Two pinned slots per CUDA backend retain high-water capacity; D less than four times the largest successfully staged payload in total per backend, plus one spare meta host arena. Host RSS/pinned high water and CUDA backpressure were not measured.
- The graph reuse/shared-scheduler primitives are preserved from the adopted source. Pipeline parallelism and evaluation callbacks still avoid the child scheduler; GPU cross-context stress is not covered.
- Preserve ignored frozen binary payloads and model adaptations for the parent queue. `codex/PROMPT.md` is the user's pre-existing untracked file and was not staged.
