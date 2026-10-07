# Round-trace writer

Implemented; build and runtime proof remain pending. No compilation, linking, tests, models, CUDA executables, or network work ran in this turn. Static checks and both gpuq manifest contract checks pass. No item was submitted and no freeze exists yet.

## Defect and design

Observed in stack6: `ggml/src/ggml-round-trace.cpp:50` performs append/open, grouping, sorting, formatting and close on the decode thread; `:105` stamps the next begin before that work. The brief's original line claims are correct. `tools/server/server-queue.cpp:280` supplies the shutdown flush guard. The supplied audit reports 0.0043 ms/KB + 0.145 ms (M, owner audit), correlation 0.983 (M), with approximately 0.95 ms for a 195 KB round (M). These measurements were read, not repeated.

The new `trace_flush` at `ggml/src/ggml-round-trace.cpp:185` captures producer PID/TID and moves phase/event vectors into an owning record. A FIFO and a lazily started worker replace decode-thread formatting. The worker opens the file once per writer session, uses a 1 MiB buffer (D, source constant), writes with the original format strings and map/sort logic (`:74`), and flushes each complete record for live readers (`:144`). Merely increasing stdio buffering would leave grouping and formatting on the critical path.

The queue admits at most 64 pending rounds (D, `:151`), plus one worker-owned round (D); a full queue blocks producers without dropping data. Submission order is mutex acquisition order, as with the old writer; concurrently racing producers never had a deterministic cross-thread ordering guarantee. `next_begin_ns` remains stamped before flushing (`:201`); the following V phase begins after enqueueing, so their difference measures producer-side writer latency.

`GGML_ROUND_TRACE_SYNC=1` selects the old synchronous formatting/open/close path (`:121`, `:153`). The mode is cached on first writer creation. Public flush submits the calling thread's final round, drains the FIFO, closes the file and joins the worker (`:229`). Concurrent flushes serialize; later submissions restart the worker. TLS destruction queues each producer's tail (`:193`), and writer destruction joins (`:123`). Normal shutdown requires producers to finish before process-wide static destruction; the existing server guard remains in place. With `LLAMA_ROUND_TRACE` unset, no worker starts.

Caller search: `rg -n ggml_round_trace ggml src tools common tests`. Phases in `common/speculative.cpp:1713` can come from temporary strings and are already owned. Current event kinds are literals, including the conditional literal at `src/llama-context.cpp:1553`; the new event `std::string` also owns non-static kinds (`ggml/src/ggml-round-trace.cpp:30`). Scalars and containers have no pointers into producer storage; formatter pointers refer only to its own record. Sampling, RNG state, backend submission and trace schema are unchanged.

## Deferred validation

`codex/test-round-trace.cpp` includes the actual writer source, replacing only trace clock/PID/TID calls; worker scheduling and file I/O stay real. `codex/trace-test/CMakeLists.txt` builds the patched source, an independent stack6 source extracted with `git show`, and a mutant that omits the public flush's final enqueue. No model/backend library is required.

`codex/check_trace_writer.py` will check:

- Raw byte equality of old, new sync and async output, including grouping, overlapping/tied event intervals, phase results, explicit flush/restart and ordered producer threads. Expected records: 17 (D, synthetic sequence).
- Final-round visibility immediately after public flush; the mutant must fail with rc 15 and `flush visibility: expected 5 records, got 4` (D, test contract). Exit-time recovery cannot hide this failure because the assertion runs before TLS destruction.
- Owned temporary event kinds, concurrent producer/flush stress with 768 records per arm (D), joined producer and main-thread exit tails, and the disabled path's empty containers/single process thread.
- Interleaved sync/async timing boots, measuring only `begin("V")`; queue draining happens outside the timed span. Each measured round has 1800 events (D) and the actual file must exceed 200000 bytes (D, minimum). Report medians and all boot samples; the sole timing threshold is async/sync < 0.2 (D, requested gate).

CPU times: unknown; tests were expressly prohibited now. Future measured results go to `codex/bw-results-aN/cpu/cpu-evidence.json`. `codex/trace_gap.py --fixtures` will exercise standard depth parsers on real SSE/memory/journal fixtures and gap/counter parsers on `qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/round-trace.jsonl`. It also checks missing-next-begin and incomplete-line controls. Parser tests are deferred with the rest of the CPU work, not claimed as passed.

Completed commands (all with `CUDA_VISIBLE_DEVICES=""`, Git with `GIT_NO_LAZY_FETCH=1`, CPU shell tools under `nice -n 19`): individual `bash -n` checks, Python `compile()` syntax checks without executing test bodies, TOML parsing, `git diff --check`, `bash codex/bw-trace-writer.sh --preflight`, and `gpuq submit --check` on the two manifests. Code-review scope/security/general/backend/design/test checklists were applied statically. No upstream search was done under the no-network rule; this is local fork work.

## Build window and GPU item

Queueable build-window manifest: `codex/bw-trace-writer.toml`, id `bw-trace-writer`, filler, priority 1 (D, manifest). Payload: `codex/bw-trace-writer.sh [OUT | --preflight]`, invoked through the existing `work/build-window-1007/bw-run.sh`. Estimated duration 2400 s (G); budget 3600 s (G = estimate x 1.5). Dominant uncertainty is CUDA cache warmth. Pins include the wrapper, payload, gate/freeze helpers, all executed Python/shell helpers, trace sources, test, harness code and recorded fixtures.

Stage 1 configures the isolated CPU test with `-DGGML_CUDA=OFF -DGGML_CCACHE=ON`, builds through `buildgate` with `-j4`, then uses `cpugate` for the CPU suite and recorded parser checks. Stage 2 uses the exact stack6 CUDA recipe: CUDA 13.2, architecture 120a, CCCL v3.4.3 with its existing disconnected source, NCCL off, shared ccache. It verifies 25 DeviceTopK strings (D, stack6 recipe) and the ggml HEAD stamp, copies the server as `round-trace`, then calls `wt freeze` as `trace-writer`. The recipe record stays in `codex/RECIPE-trace-writer.txt`; only the explicitly requested freeze operation writes outside this worktree. Stage 3 regenerates the GPU item against actual frozen files and passing CPU evidence and runs `gpuq submit --check`; it never submits anything.

GPU experiment template: `codex/trace-writer-depth-j5q.toml.in`. Generator: `codex/bw_make_trace_item.py`. Prepared files: `codex/items/trace-writer-depth-j5q.toml` and `codex/items/trace-writer-depth-j5q.gpuq.toml`. They are ignored generated artifacts. The prepared item is contract-valid but explicitly marked PREPARED ONLY; do not submit it before the window regenerates it with actual CPU evidence and a freeze. It already names future `frozen/trace-writer/{SOURCE,build/bin/SHA256SUMS,build/bin/round-trace}` pins; regeneration adds every actual frozen file.

The generator uses `harness.mkitem` for contract/budget/pins and replaces its entry with `codex/gpu_ab.sh`, retaining standard depth gates/cleanup and adding the writer-gap gate. Both arms use the same frozen binary; sync env is 1 vs 0 (D), scrubbed and recorded with the other knobs. J5q, np1, k5, tensor split, counters enabled in both arms, cross-arm token/draft-counter identity; ABBA-style interleaved boots. The one-boot canary uses the smallest configured depth and the same parsers, completeness and async latency gate, capped at 600 s (D). Full-run boot allowance is 420 s each (G); the generated budget is 5950 s (G/D, harness arithmetic), including margin/canary/cleanup.

`writer-gap.json` records paired-round counts, median and mean of following V begin minus preceding next-begin stamp, restricted to complete full-k rounds inside each request. It rejects fewer than 40 pairs (D) or async median above 0.1 ms (G, target). Standard `summary.json` reports `steady_ms_per_round_D`; expected delta approximately -0.85 ms/round (G, extrapolated from the audit), not an observed GPU improvement. All request artifacts are saved before gates; standard harness supplies frozen hashes, exact-PID stop, budget and boot-scoped kernel-journal checks.

## Exit codes

The build-window wrapper maps all nonzero payload returns to gpuq rc 8; payload codes never use 3 or 4. The GPU item retains the standard fixed gpuq map, where 3/4 have their required hazard/transient meanings.

| Payload rc | Failing check |
| --- | --- |
| 0 | All stages pass |
| 10 | Preflight, scope, cleanliness, build-window guard or prerequisites |
| 20 | CPU configure/build |
| 21 | CPU correctness, drain/ownership/stress, negative control or timeout |
| 22 | Async/sync timing ratio fails |
| 23 | Recorded parser checks |
| 24 | Independent old-writer or mutant source preparation |
| 30 | CUDA configure/build or renamed server copy |
| 31 | Freeze |
| 32 | ggml stamp |
| 33 | CCCL DeviceTopK strings |
| 34 | HEAD moved during build |
| 40 | GPU item generation/contract |

Synthetic executable diagnostic returns are 15 (visibility), 16 (kind ownership), 17 (disabled path) and 18 (usage); the controller expects 15 only for the mutant and maps unexpected failures to 21 (D, source contracts).

## Risks and open items

- Compile and runtime behavior are unverified until the build-window item passes, including the requested CPU timing ratio and real-artifact parser tests.
- Normal exit drains joined producers and the main TLS tail; abrupt exit, crash or SIGKILL can lose the queued tail and the stdio buffer. Flush cannot collect an active round owned by another still-running producer.
- Queue length is bounded by rounds, not bytes. Individual rounds/event names remain unbounded as before; a stalled filesystem causes decode backpressure. Moving vectors also gives up producer capacity reuse, and owning event strings adds capture-side work that needs the GPU comparison.
- The async file remains open during a session: environment path changes and file rotation during tracing are outside the startup-only contract. Explicit flush closes it and a later session reopens it.
- Concurrent producer interleavings can differ between executions; ordered submissions preserve bytes/order. The deterministic comparison deliberately controls cross-thread submission order.
- The GPU result remains unknown. First queue/review the build window, then review CPU evidence and the regenerated item before submitting the real-server canary/full A/B.
