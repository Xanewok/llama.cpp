# Round-trace writer review fixes

Review findings addressed in source. Compilation, runtime correctness, TSan, CPU timing and GPU performance remain unverified: this task permits static checks only. No model, CUDA executable, build, test suite, parser fixture test or network operation ran. No queue item was submitted and no freeze was created.

## Design and findings addressed

The writer still moves producer phase/event vectors into a FIFO record (`ggml/src/ggml-round-trace.cpp:198`), records the producer PID/TID, and performs grouping, sorting and formatting on its worker (`:74`). The synchronous arm uses the same binary and original formatter. The queue admits at most 64 pending rounds plus one worker-owned record (D, source contract); a full queue blocks producers. Public flush and normal TLS/process exit retain their drain behavior. Sampling and RNG code are untouched.

- S1: `trace_event::kind` is `const char *` again (`ggml/src/ggml-round-trace.cpp:30`), eliminating per-event string construction on the producer in both arms. `ggml/include/ggml-round-trace.h:16` requires immutable static storage for event/count/scope kinds. Phase names remain owned because `common/speculative.cpp` supplies temporary phase strings. The ownership test (`codex/test-round-trace.cpp:118`) pauses the async writer, queues a record, mutates the source phase string and the next round's phase/event fields, then checks that the original phase, result and event fields survive. Its long kind has static storage.
- S2: `codex/trace-test/CMakeLists.txt:5` adds `GGML_ROUND_TRACE_TSAN=ON`, selecting `trace-test-tsan` with compile/link `-fsanitize=thread`. The payload uses a separate `build-cpu-tsan` directory and its own build/stress stage (`codex/bw-trace-writer.sh:83`). `--stress` checks every producer's sequence and final count (`codex/test-round-trace.cpp:194`). A mutex-protected worker pause forces an async submission to encounter the full queue before release (`:181`); `test_full_queue_waits` must be positive. Both hooks are compiled only under `GGML_ROUND_TRACE_TEST_HOOKS` (`ggml/src/ggml-round-trace.cpp:121`), which the CUDA build does not enable. Stress emits the observed full-queue counter into CPU evidence.
- Recipe placement: the payload writes `frozen/trace-writer/RECIPE.txt` after `wt freeze` (`codex/bw-trace-writer.sh:121`), matching the graph-capture-fix payload. Only the parent's future payload performs those explicitly requested frozen-build writes outside the worktree.
- GPU control gate: `codex/trace_gap.py:25` requires every sync boot/depth median >= 0.3 ms (D, requested control threshold) and every async median <= 0.1 ms (G, target). An async-only failure returns rc 8 with `writer async <= 0.1 ms gate (G) failed`, without raising. `writer-gap.json` now contains `boots`, `by_depth`, and labeled thresholds. For each depth, `by_depth` reports each arm's median of boot medians (M) and async/sync ratio (D); the gate still checks individual boots. Missing boot/depth artifacts or insufficient complete pairs return rc 5 (`:141`).
- Timing: `codex/test-round-trace.cpp:214` submits a warm-up record before the timed loop and joins only after it. The measured records fit below the queue bound (D), so thread creation and joins are excluded and queue saturation cannot contaminate the enqueue measurement.
- Canary: `codex/trace-writer-depth-j5q.toml.in:68` uses both `up` and `patch` at depth 8192 (D, configuration), with the same parsers and gates as the full run. Separate boots are necessary because the sync switch is cached at writer creation. The combined canary cap remains 600 s (D, configuration).
- Estimate: `est_s` is now 2300 s (G), with the arithmetic below.

## Caller audit and static evidence

`grep -rnE "ggml_round_trace_(event|count|scope)" ggml src common tools tests codex/test-round-trace.cpp` and the broader source search across `examples`, `pocs`, `tests`, `tools`, `common`, `src`, and `ggml` found no non-static caller kinds. Production callers use literals or conditional literals (`ggml/src/ggml-cuda/ggml-cuda.cu:4584`, `src/llama-context.cpp:1553`, `:1562`). The header's scope/count wrappers forward those pointers; the revised ownership test uses `static const char kind[]`. No exceptional caller needs a copy. The serializer's string keys remain on the worker in the async arm.

Completed with `CUDA_VISIBLE_DEVICES=""`, `GIT_NO_LAZY_FETCH=1` and `nice -n 19`:

- `bash -n codex/bw-trace-writer.sh` and `bash -n codex/gpu_ab.sh`.
- `python3 -m py_compile codex/check_trace_writer.py codex/trace_gap.py codex/bw_make_trace_item.py`.
- `git diff --check`.
- `gpuq submit --check codex/bw-trace-writer.toml`: valid.

The code-review scope/security/general/backend/test checklists were applied statically. No upstream lookup was attempted under the no-network rule. Existing untracked `codex/PROMPT.md` and `codex/PURPOSE` were left untouched.

## Deferred build window and parser proof

`codex/bw-trace-writer.toml` retains its 2400 s estimate (G) and 3600 s budget (G/D); CUDA cache warmth dominates this separate build-window estimate. The payload builds the independent old writer, patched writer and omitted-final-enqueue mutant through `buildgate` with `-j4`, then runs CPU checks through `cpugate`. Raw old/sync/async bytes must match. The mutant must return rc 15 before exit-time drain can conceal its missing tail. The suite also checks ownership, ordered concurrent flush/stress, joined TLS tails, disabled behavior, and interleaved warm-worker timing with async/sync < 0.2 (D, requested CPU gate).

The separate TSan stage runs async `--stress` with `TSAN_OPTIONS=halt_on_error=1:exitcode=66` and a 120 s timeout (D, payload settings). Failures stop before the CUDA freeze; its log and raw child rc are retained. CPU times and TSan result are unknown. Future outputs are `codex/bw-results-aN/{cpu/cpu-evidence.json,build-cpu-tsan.log,tsan-stress.log,tsan-stress.jsonl}`.

`trace_gap.py --fixtures` remains deferred. It uses the real standard SSE/memory/journal fixtures under `qwen38-eval/orchestration/gpuq/fixtures` and the recorded stack6 trace at `qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/round-trace.jsonl`. It checks incomplete lines and missing stamps, then derives passing latency records from real trace structure and verifies separate failing sync/async controls. These tests have been written and syntax-checked, not executed.

After those stages pass, the existing stack6 CUDA recipe builds and freezes the renamed server `round-trace`. `bw_make_trace_item.py` regenerates the GPU experiment/item using the actual freeze and current CPU evidence, then runs `gpuq submit --check`; it never submits. Existing ignored PREPARED ONLY items must be regenerated before use.

## GPU estimate and budget

Reference: `/home/xanewok/repos/qwen38-eval-worktrees/wt/graph-capture-fix/codex/gpu-results/depth-notrace/full-a1/console.log`. It reports the full eight-boot run at the same depths as 1529 s (M); individual boots took 189-196 s (M). This is about 25.5 min (D), below the brief's approximate 27 min wall estimate. Async/sync tracing can change that cost, so 2300 s remains G: allow 1600 s for the full run, 200 s for both shallow canary boots, 150 s cleanup and 350 s contingency (all G allowances).

The GPU budget remains 5950 s (D, `harness.mkitem.budget`): run = ceil-to-50(8 x 420 x 1.5 + 150) = 5200 s; total = 5200 + 600 canary + 150 cleanup. Planned full stages are 3360 s, below 0.7 x 5200 = 3640 s (D). Per-boot allowance 420 s is G. The expected round-time delta in the template remains -0.85 ms/round (G), based on the earlier owner trace-writer audit, not a GPU result. `codex/gpu_ab.sh` delegates frozen hashes, identity, exact-PID stop, wall budget and boot-scoped kernel-journal checks to the standard depth harness.

## Exit codes

The build-window wrapper maps nonzero payload returns to gpuq rc 8. Payload codes below are diagnostic contracts (D), not GPU hazard/transient codes.

| Payload rc | Failure |
| --- | --- |
| 10 | Preflight, scope, cleanliness, build-window guard or prerequisites |
| 20 | CPU configure/build |
| 21 | CPU correctness, ownership, drain, order/backpressure, mutant or timeout |
| 22 | CPU async/sync timing ratio |
| 23 | Recorded parser checks |
| 24 | Old-writer/mutant source preparation |
| 25 | TSan configure/build (new) |
| 26 | TSan stress, sanitizer report, timeout or existing output (new) |
| 30 | CUDA configure/build or renamed server copy |
| 31 | Freeze or frozen recipe write |
| 32 / 33 / 34 | ggml stamp / CCCL strings / moving HEAD |
| 40 | GPU item generation/contract |

Executable diagnostics are 15 visibility, 16 record ownership, 17 disabled path, 18 usage, 19 producer order (new), and 20 unreached backpressure (new). The controller maps unexpected executable failures to payload rc 21; TSan reports use child rc 66 and map to payload rc 26. GPU latency/control failures use rc 8; incomplete data uses rc 5; inherited rc 3/4 keep hazard/transient semantics. Success is rc 0.

## Risks and open items

Runtime behavior remains unverified until the parent runs the build window, including TSan runtime support, ownership/drain/order tests, timing and recorded parser controls. No GPU speed claim is supported yet. Normal shutdown requires producers to finish before process-wide writer destruction; crashes/SIGKILL can lose queued tails. Queue capacity bounds rounds rather than bytes, and slow storage causes backpressure. Event kinds now require immutable static lifetime; future dynamic-kind callers must be handled explicitly. Phase/event vector moves still surrender producer capacity reuse. File rotation or environment changes during an open writer session remain outside the startup contract.
