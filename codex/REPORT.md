# graph-reuse-prefill

Confirmed CPU host overhead: fresh meta-plan arenas and dependency retention, with unchanged prefill rebuild counts. The bounded arena fix preserves tested token/logit bytes. Attribution of the recorded GPU penalty remains unresolved. See [REPORT-prefill.md](REPORT-prefill.md) for evidence, controls, frozen builds and canary-first parent commands.

# graph-reuse-safe

Retained C/D graph results and meta plans pass seeded CPU token/id/q/logit byte identity against the supplied base. Binding and epoch mutation controls fail as required. GPU validation remains parent-only.

[REPORT-graph-reuse.md](REPORT-graph-reuse.md) contains the design contract, CPU numbers, source/test evidence, model-fixture qualification, CUDA-probe compliance exception, upstream fit and risks. [gpu_ab.sh](gpu_ab.sh) queues the frozen trace-off S5 A/B; [run_reuse_cpu.sh](run_reuse_cpu.sh) reproduces CPU evidence.

Topic commits are in `codex/local.git`; `codex/graph-reuse.bundle` is the portable handoff. The prior round-trace report remains in the base commit and the read-only parent worktree.

## graph-reuse-ab2

See `codex/REPORT-graph-reuse.md`, section `graph-reuse-ab2`, for the first GPU A/B diagnosis, CPU cached-request evidence and replacement harness. The initial K=5 tool difference is D+17.0135 ms/request; its GPU attribution remains unknown. Production/sampling sources and frozen binaries are unchanged. The harness now requires at least 40 interior full-K timing intervals per long request and separately reports estimated request cost and cached follow-ups.

## graph-reuse-ab3

The replacement harness accepts the captured Swift SSE stream and preserves a complete boot's request data before rejecting a short workload. No GPU execution, network calls, inference, or builds were performed for this Python/input-only change.

Observed in the supplied captures: the old long request first repaired the parser with `apply_patch` and stopped at M 122 tokens / M 30 rounds. `cold-tool-1.json` inspected the tests via `exec_command` at M 31 tokens; `warmup-0.json` inspected both source and tests, with brief reasoning, at M 56 tokens. The new long prompts remove this competing repair/inspection context.

### Design and evidence

- `codex/gpu_harness.py:353`: skip chunks without a non-null `draft_n`; retain monotonic draft/time checks and final-counter bounds for chunks that have it. Repeated counters remain valid and contribute one timing point per counter.
- `codex/fixtures/swift-long-tool-s1.json:1`: reduced fixture from `codex/runs/20261003-160526/1-up/long-tool-s1-1.json`, with original response/counters, ordered SSE timing fields, empty chunks, and source SHA. It contains M 95 chunks, M 93 timing rows, and M 1 timing row without `draft_n`. CPU parsing recovers M 24 interior intervals with D 37.19825 ms/round from the parent's captured timings; this is a parser regression value, not a new GPU performance result. The normal gate rejects it with `insufficient steady rounds: 24 < 40`.
- `codex/inputs/long-tool.json:1` and `codex/inputs/long-think.json:1`: replace the parser-repair conversation with a standalone `write_file` tool whose only argument is `content`. The prompt explicitly spells out D 256 numbered lines, D 34,047 file bytes and D 5,120 whitespace-separated words. It requires one literal file, forbids scripts/ranges/placeholders/samples, and requests visible arguments. This removes the competing repair/inspection tasks seen in the long, cold and warm-up captures. G: copying the full file should exceed the configured generation cap of 2048 tokens. Both workloads retain `tool_choice=auto` and disabled prompt caching; the think variant enables thinking with `reasoning_effort=low` and requests brief planning before the visible file.
- `codex/gpu_harness.py:380`, `:409`, `:574`: collect the cold, long and cached follow-up requests first. Then validate each long request's interior-round minimum, at least 2048 generated tokens, and visible `write_file` arguments. Cache failures also accumulate at boot end. Write each `*-steady.json`, `boot-validation.json` and `ab-results.json`, stop the owned server, check Xid and frozen boot bytes, then raise on validation failure. Existing interleaving, identity comparison, input/binary hashes, exact-PID cleanup and budget guards remain in place.
- `codex/test_harness.py:60`, `:78`, `:142`: real-stream regression, missing/null counter insertions, backwards/out-of-bounds counter/time controls, literal prompt checks, short/hidden-output controls, and a mocked execution-path regression. The failing boot saves M 13 suite request records and M 4 long-request diagnostic files, checks Xid, stops, and never starts the next boot. Warm-up remains separately recorded by `Harness.boot`.

### CPU validation

- `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_harness.py`: M PASS, rc 0. Plans and the mocked failed boot forbid subprocess launches and network sockets; the execution environment marker is mocked inside the test only.
- `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_attrib.py`: M PASS, rc 0; existing attribution/accounting and rejection controls pass.
- `CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/gpu_ab.sh --k 5`: M PASS, rc 0; frozen trace-off `up/patch/patch/up` dry plan, no launch.
- `CUDA_VISIBLE_DEVICES='' nice -n 19 bash -n codex/gpu_ab.sh`: M PASS, rc 0.
- CPU mutation check: restore `timing.get("draft_n", 0)` in an in-memory harness copy and parse the same fixture. M PASS: the broken parser raises `reversed or inconsistent cumulative SSE timing`; the real-stream regression fails with that mutation. The source file is not modified by this control.

### Risks and next action

Prompt compliance is G, not measured on Swift. The gate checks length and a streamed tool argument, not complete file correctness. Generation is expected to hit the token cap inside the file argument; a truncated tool call is valid benchmark traffic. The longer input increases prompt work, which remains reported separately. HTTP, incomplete-SSE and server errors still abort immediately; deferred failure applies to completed request validation.

The parent should queue the existing `codex/gpu_ab.sh --execute --k 5` with its usual `RUN_GPU=1`, selected GPUs and `XID_SINCE`. A failed boot retains request data and per-case errors; it produces no paired summary. Swift workload length, grammar-active generation and GPU A/B speed remain open until that run.
