# graph-reuse-prefill

Labels: M measured, D derived, G hypothesis/estimate. No GPU executable was run. Work and commits stay in this worktree and `codex/local.git`, on `topic/graph-reuse-safe`.

## Conclusion

The patch adds host work to existing graph misses, rather than causing extra target prefill rebuilds in the tested workload. Fresh meta plans allocate graph arenas and traverse every reachable tensor to retain its container. These costs are confirmed on CPU. Their contribution to the recorded GPU prefill/request regressions is **unknown**; this investigation does not establish the dominant GPU cause or claim GPU speed.

The candidate retains one spare host plan per meta backend, releases its tensor owners immediately, and rebuilds topology before reuse. It also hashes the dependency visited set. Allocation controls pass, as do tested token, proposal probability/logit, and final target-logit bytes. CPU timings do not establish a reliable total prefill improvement. Queue diagnostics before considering adoption.

## Recorded GPU evidence

Read both decisive runs: `codex/runs/20261003-173851` and `codex/runs/20261003-174710`, all `server.log`, `ab-results.json`, and `ab-summary.json` files. The target's cumulative `graphs reused` sequence is identical across every arm within each run. For example, the first run starts M `9, 13, 375, 741, 1103, 1467`; the second starts M `26, 36, 568, 1103, 1635, 2169`. They do not expose draft reuse, per-ubatch time, meta-plan creation, or CUDA capture. They therefore cannot attribute prompt or request excess.

The source comparison is `0dba27b9` (the older `codex/frozen/up/SOURCE`) through `0654a0c0f` (measured patch), with the actual A/B base `2b645a655` in `codex/frozen/graph-up/SOURCE` / `codex/BASE`. Changes between the earlier up source and that base are attribution instrumentation; no sampling algorithm changes. Background reports and the final findings of `SP/codex-reviews/patchA.txt` were read before designing the candidate.

## CPU experiment

Use the existing `codex/reuse-server-cpu.cpp` in-process server, with no HTTP listener. Both builds have `GGML_CUDA=OFF`. Model: the existing Qwen3.5-4B MTP fixture, with `output.weight` copied byte-for-byte from embedding blocks by the existing `prepare_cpu_target.cpp`. This permits target TP backend sampling; the original tied-head model is not an adequate substitute for that path. The compact draft vocabulary remains unchanged. Meta(CPU,CPU) uses configured threads `4 + 4`; all work runs at nice `19` with `CUDA_VISIBLE_DEVICES=''`.

Configured `CPU_PREFILL_WORDS=512`, `n_batch=128`, `CPU_UBATCH=32`, MTP K=3, seed `1729`, active numeric grammar and `n_predict=16`. The appended words produce M 534 target prompt tokens initially. Two cached tasks were measured with up/patch/patch/up plus frozen-unpooled and reuse-disabled controls. A second experiment disabled prompt caching and repeated cold prefill with up/patch/unpooled.

| Target prefill, repeated cold requests | up | instrumented unpooled patch | candidate |
|---|---:|---:|---:|
| ubatches, requests A/B | M 17 / 18 | M 17 / 18 | M 17 / 18 |
| graph misses, each request | M 5 | M 5 | M 5 |
| meta builds, each request | M 5 | M 5 | M 5 |
| meta executes, requests A/B | D 17 / 18 | D 17 / 18 | D 17 / 18 |
| fresh graph arenas, each request | M 0 | M 5 | M 0 |
| cumulative arena bytes requested, each request | M 0 | M 87,516,000 | M 0 |
| llama/meta build union, requests A/B | D 57.334 / 48.354 ms | D 64.893 / 55.072 ms | D 61.101 / 54.676 ms |
| build union per ubatch, requests A/B | D 3.372 / 2.686 ms | D 3.817 / 3.060 ms | D 3.594 / 3.038 ms |
| container-retention time, requests A/B | absent | M 4.287 / 3.545 ms | M 2.923 / 2.719 ms |

The cold run is one boot per arm; timing differences are observations, not a statistically resolved speed claim. The cached interleaving also leaves candidate build work above up: D mean 62.350 vs 57.713 ms over the initial prefill. The same run's unpooled control records D 62.587 ms. Container-retention work remains in the candidate. Arena bytes are cumulative requested capacity, not RSS or peak live memory.

`llama_build` includes allocation and eager meta preparation in the patch; base `meta_build` happens during compute submission. The table unions those intervals to avoid double-counting the patch or comparing unlike ranges. `meta_compute` includes preparation, so actual execution count is derived by subtracting `meta_plan_create`. Per-ubatch events, token/output counts and host duration are retained in raw JSONL and `results.json`; summary files retain aggregate counters.

Evidence: `codex/prefill-summary.json` (cold), `codex/prefill-cached-summary.json` (cached), `codex/prefill-ab2-results.json` (existing steady/cached A/B). Raw artifacts are under `codex/reuse-results/prefill/` and `codex/reuse-results/ab2-candidate/`.

## Hypotheses and source evidence

- **Extra prefill rebuild/re-plan:** not observed. GPU cumulative target reuse is identical. CPU target ubatch/miss/compute counts agree across arms, including repeated cold requests. Draft prefill also agrees at M 17 / 18 ubatches and M 4 misses per request. Retaining C/D plans does not force a rebuild on every prefill ubatch in this fixture.
- **Additional allocation/retention cost:** confirmed. `ggml/src/ggml-backend-meta.cpp:2645` creates a plan on each allocation; previously every plan started with empty capacity. `:2408` computes its graph-arena size. `:2120` measures the dependency walk/container ownership work. The spare plan at `:1933`, `:2649`, `:2666` releases all container owners before reuse, forces `uid=0`, and preserves host arrays/arena capacity. `:2421` sizes recycled graph arrays to the retained high-water capacity. The visited set at `:2122` changes storage only; `used_buffers` remains ordered, so container-reset order stays unchanged.
- **Sampler attachment invalidation:** present and required for correctness. `src/llama-context.cpp:1385`, `:1406`, `:2576` reset retained graphs on attach/detach; the base only invalidates the active result. Across the cold boot, recorded reset scopes total M 0.397 ms unpooled versus M 0.371 ms candidate, not evidence for the GPU's much larger excess. Recorded plan-destruction scopes total M 0.340 versus M 0.053 ms over M 36 frees in each boot. Events can be recorded under the preceding phase, so use their absolute timestamps for request ownership. Initial attachment before the first traced phase is not recorded.
- **Extra synchronization:** graph misses and alternating hits synchronize in `src/llama-context.cpp:1537`, `:1546`; CPU cannot measure the GPU wait. Existing trace sync events will distinguish waiting from host allocation.
- **CUDA graph key/warmup churn:** G. `ggml/src/ggml-cuda/ggml-cuda.cu:2678` keys by `cgraph->nodes[0]`. New meta arenas also change auxiliary-node addresses; retained compute containers can change main-node addresses. `:4564` requires stable properties across calls before capture; new keys or changed properties can leave several graphs submitting kernels directly. CPU allocation identity does not prove CUDA warmup behavior. New opt-in events at `:4544` record the raw key in event `bytes`, property-check duration, changed properties, and warmup resets. Existing `cuda_submit`, `cuda_capture`, `cuda_instantiate` and `cuda_launch` scopes complete the diagnostic. The `bytes` field for `cuda_key` is a pointer, not a transfer size.

## Exactness and controls

M PASS: all cold/cached prefill arms preserve every returned token and the final target-logit byte array. The frozen unpooled snapshot fails `allocation_contract()` because it creates arenas; its output bytes remain identical. Reuse-disabled control removes all hits while preserving bytes. The existing K=5 up/patch/patch/up cached-request test and disable control retain byte identity and at least M 40 interior SSE intervals per request.

M PASS: `reuse-contract` checks sampler input ownership/RNG draws, wrong UID/stale epoch rejection, borrowed meta input ownership through M 40 container rotations. The synthetic Meta(CPU,CPU) fixture also compares every target/draft token, candidate ID, proposal probability and sampled logit byte over M 12 rounds at K=3, seed `1729`; M 20,832 output bytes match up. The stale-binding mutation fails with `stale bindings or RNG draw count`. The candidate changes neither sampling draws nor arithmetic nodes; pooling releases ownership before allocation and forces complete topology rebuilding. General GPU numerics, capture and concurrency remain untested.

Commands, all CPU-only:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF
CUDA_VISIBLE_DEVICES='' nice -n 19 cmake --build build-cpu --target llama-server -j8
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/prefill_cpu.py
CUDA_VISIBLE_DEVICES='' CPU_CACHE_PROMPT=0 CPU_ARMS=up,patch,unpooled nice -n 19 python3 codex/prefill_cpu.py
CUDA_VISIBLE_DEVICES='' CPU_TARGET_MODEL="$PWD/build-tmp/reuse-untied-target.gguf" nice -n 19 python3 codex/run_ab2_cpu.py
CUDA_VISIBLE_DEVICES='' nice -n 19 build-cpu/bin/reuse-contract
CUDA_VISIBLE_DEVICES='' nice -n 19 build-base-cpu/bin/reuse-cpu - 1 3 1729 12 build-tmp/prefill-sampling-up.bin 0
CUDA_VISIBLE_DEVICES='' nice -n 19 build-cpu/bin/reuse-cpu - 1 3 1729 12 build-tmp/prefill-sampling-patch.bin 0
CUDA_VISIBLE_DEVICES='' nice -n 19 cmp build-tmp/prefill-sampling-up.bin build-tmp/prefill-sampling-patch.bin
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_harness.py
CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_attrib.py
```

The server fixture must be compiled against each CPU build as in `codex/run_reuse_cpu.sh`. `unpooled` additionally needs the saved libraries in `codex/reuse-results/prefill-unpooled/bin`; that snapshot carries the instrumented pre-fix implementation from `79575de0c`. This is an explicit fixture dependency, not an environment-only toggle. Running `prefill_cpu.py` replaces its summary with the latest run; preserve prior summaries when comparing experiments.

## Frozen CUDA artifacts and parent-only run

Candidate: `codex/frozen/graph-prefill`, source `7dd59ccc7`, with immutable SHA manifests. Built with CUDA toolkit `13.2`, architecture `120`, `GGML_CUDA_NCCL=OFF`, and `CMAKE_CUDA_COMPILER_FORCED=ON` to suppress the CMake device probe. No compiled CUDA executable was run.

Diagnostic unpooled control: `codex/frozen/graph-prefill-control`. Its SOURCE explicitly declares the meta-source overlay from `79575de0c` on `7dd59ccc7`, including the meta source SHA. It has identical tracing but lacks the spare-plan/hash/capacity changes. Both binaries are renamed `round-trace`. The original frozen up/patch builds are preserved.

The canary uses one patch boot, one cold request, one long tool and one long think request, and two cached follow-ups. The same length/round/visible-tool/cache gates apply as in the full run. It also compares the initial requests' content, authoritative draft/accept counters and token counts against the decisive recorded up artifacts. G stage estimate 185 seconds fits the configured canary budget; full-run estimate G 1500 seconds fits its configured budget with the contract's margin. Gates run after recording the boot's data. Exact-PID/start-time stop, Xid checks, frozen hashes and input pins are retained.

M PASS: parsers/gates tested on recorded request artifacts from **both** decisive GPU runs, with counter/content/cache failure controls. Xid parsing also uses `gpuq/fixtures/xid-lines.txt`. A mocked full canary replays real recorded output without launching subprocesses or opening sockets. No new GPU trace-file parser is introduced; `rounds.jsonl` retains the existing schema. The new key/warmup counters still require a real canary to validate their observed CUDA behavior.

Parent only, with the normal gpuq environment (`RUN_GPU=1`, selected GPUs, `XID_SINCE`), unique output directories and all reference/input/frozen files in `pins.files`:

```sh
bash codex/gpu_ab.sh --execute --k 3 --patch codex/frozen/graph-prefill-control --canary --budget 600 --diagnostic-trace --out codex/runs/prefill-control-canary &&
bash codex/gpu_ab.sh --execute --k 3 --patch codex/frozen/graph-prefill-control --diagnostic-trace --out codex/runs/prefill-control-diagnostic
```

This is the attribution run for the original regression: compare prefill ubatch event ranges and first-round work, not only `request-builds.json`, whose existing summary selects full decode rounds. For each request, separate large V/C prefill phases from decode, then compare graph misses, meta build/retention/arena work, per-GPU sync and `cuda_submit`/capture/launch/changed/key events. Many changed keys or direct submissions with unchanged graph counts support the CUDA hypothesis. A host build/retention increase matching the prompt delta supports host allocation instead. If neither accounts for it, retain the uncertainty and use nsys rather than infer speed from CPU.

After diagnostics, candidate validation/performance is a separate trace-off item:

```sh
bash codex/gpu_ab.sh --execute --k 3 --patch codex/frozen/graph-prefill --canary --budget 600 --out codex/runs/prefill-fixed-canary &&
bash codex/gpu_ab.sh --execute --k 3 --patch codex/frozen/graph-prefill --out codex/runs/prefill-fixed-ab
```

Repeat at K=5 only if the parent needs that serving configuration. These commands were not executed here.

## Risks and open items

The spare retains one maximum-sized **host** arena per meta backend until backend destruction; RSS was not measured. Container pins are released on plan free; model/KV/compute ownership is not extended by the spare. No additional device buffers are introduced by this change. Hash visitation preserves the reachable tensor set; compute and buffer-reset order stay unchanged. High-water graph capacity prevents reusing smaller graph arrays after topology changes.

The CPU prompt is shorter and uses a different model than Swift, so its cost is not a quantitative GPU prediction. The cold run has one boot per arm; the bounded allocation reduction is proven, while total speed is not. The measured GPU penalty and capture/key attribution remain open. Parent canary and A/B must pass exact-output gates before the candidate is adopted.

Incremental handoff: `codex/graph-reuse-prefill.bundle` exports the new commits and requires parent commit `79211b413`. Full-history regeneration of the older `codex/graph-reuse.bundle` failed because object `1c21e55aaa011d7d8462ed151cfa29bd03924d26` is missing locally; that older export remains unchanged. The requested topic commits are available in `codex/local.git`. No fetch or network recovery was attempted.
