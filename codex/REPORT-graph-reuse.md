# graph-reuse-safe

## Conclusion

Keep the existing output/no-output graph arenas and give the output arena one shared-storage scheduler. Each scheduler split owns a meta graph plan through the existing graph-plan API. Graph-local sampler bindings and storage-epoch checks make alternation safe without Patch A's configurable LRU, buffer registry, or rematerialization.

M: seeded CPU A/B preserves every recorded sampled token, candidate ID, probability and logit byte. C and D1 hit every retained round, including Meta(CPU,CPU). Mutation builds that omit binding restoration or the compute epoch check fail their contract checks. GPU speed and GPU exactness are unmeasured.

G: plausible saving remains 0.6-0.85 ms/round at K=3 and 0.6-0.9 ms/round at K=5, from the supplied trace's D0.843-0.850 ms exposed build budget. Added synchronization can reduce that gain. Queue the frozen trace-off GPU A/B before accepting the optimization.

M = observed/measured; D = arithmetic on timestamps/counts or a source-derived invariant; G = forecast. Commit IDs, issue IDs, seeds, K, dimensions and command flags are configuration/identifiers.

## Source and design contract

Base: `2b645a655`, the supplied worktree's stack2 plus round-trace instrumentation, not the older Patch A base `68d99c67d`. Production commits in `codex/local.git`, branch `topic/graph-reuse-safe`:

- `7839e00ba`: scheduler families, storage validity and owned meta plans.
- `58ecab9bb`: retain the existing two graph results and restore sampler bindings.
- `eac50e4a4`: fetch deferred logits before invalidation, including sampler detach.
- `0654a0c0f`: declare internal storage helpers; removes new compiler warnings.

The original `.git` points outside this worktree and is read-only. The worktree-local Git directory holds these commits; use `git --git-dir=codex/local.git ...`. The bundle carries the topic commits with `2b645a655` as its prerequisite. No external branch was rewritten.

`src/llama-context.cpp:1518` creates one child scheduler for output graphs; catch-up uses the parent. Both share one gallocr. A hit requires the allocated graph UID/storage epoch and the full existing `can_reuse(gparams)` check at `:1529`. Alternation synchronizes before input writes at `:1534`. A miss synchronizes, releases that split's plans, resets the result, builds, allocates, then writes inputs. No shape-only hit or padded draft batch is introduced. Pipeline parallelism and evaluation callbacks retain the single-scheduler path.

The epoch tracks storage validity, not allocator planning: `ggml/src/ggml-alloc.c:938` increments it before buffer replacement, and `:1078` before a destructive backend reset. Replanning offsets in unchanged buffers does not move an already allocated graph's tensors. An unconditional replan epoch prevented C/D reuse on this stack because no-output TP graphs omit the sampler subgraph. Shared backend-ID history describes the allocator's latest plan. Family operations must be serialized; allocated topology and tensor bindings must remain unchanged until reset.

`ggml/src/ggml-backend.cpp:2191` checks UID and epoch again in public `compute_async`. A stale allocated graph returns `GGML_STATUS_ALLOC_FAILED` without replay or reallocation, because inputs may already have been uploaded to invalid storage. The caller must rebuild and upload again. Llama checks before input writes; serialized family use prevents an intervening allocation.

`src/llama-sampler.cpp:4416` captures binding restorers for chains, dist uniforms, penalty inputs/layout flags, and logit-bias inputs. Stateless samplers, including the empty wrapper used for disabled temperature, need no state restoration. Unsupported input hooks prohibit reuse. `src/llama-graph.cpp:1270` captures after all sampler branches have been built; `:1282` restores before the unchanged active-sequence input loop. No RNG object, draw counter, acceptance state, sampling arithmetic or distribution filter is copied or changed.

`ggml/src/ggml-backend.cpp:2166` creates the meta plan after tensor initialization. Plans retain every referenced compute container, walking nodes, leaves, view sources and source dependencies at `ggml/src/ggml-backend-meta.cpp:2116`. Only containers with no plan references can be reset. Initialization selects the writable container; lookups can find retained owners. Allocation through plan creation and shared container/cache mutations use the same recursive mutex. Plans are released before resplitting, reset or free. Static weight/KV buffers retain their existing caller-owned lifetime contract. Reduction scratch remains backend-owned and shared across plans. D: no extra device data buffer is introduced; host plan/context memory increases.

`src/llama-context.cpp:2573` fetches deferred logits before clearing graph results and maps. A real server test initially found a post-detach `backend != nullptr` assertion; the final fix preserves the later raw-logit read. Sampler replacement/detach, reserve, memory updates and optimizer work invalidate both results. Backend member ordering also keeps schedulers valid during construction failure.

## CPU evidence

The teacher-forced fixture follows V -> C -> D1..Dk, with backend samplers attached to target and draft, a compact non-contiguous draft vocabulary, and top-k/top-p/min-p/temp/dist. It records sampled tokens, ordered candidate IDs, q/probabilities and logits. It does not implement rejection verification; the separate in-process server fixture exercises the actual probabilistic MTP rejection path.

M: synthetic Qwen3.5 hybrid/MTP fixture, 16 seeds, K=3/K=5, CPU and Meta(CPU,CPU), order up/patch/patch/up, 12 rounds per run. Every byte matches. Each configuration/arm has M32 runs and M288 retained rounds after discarding the first three. Patch C and D1 each hit M288/288; base each misses M288/288. Meta rebuilds in those phases fall from M576 to M0 per configuration.

M: real Qwen3.5-4B MTP weights, four seeds and the same K/backend/order/round count. Each configuration/arm has M8 runs and M72 retained rounds. Every byte matches. C and D1 each hit M72/72 in the patch and miss M72/72 in base; TP meta rebuilds fall from M144 to M0 per configuration.

The real GGUF has a tied target head that baseline TP backend sampling refuses. The TP target fixture therefore uses a worktree-only GGUF with byte-identical embedding blocks also named `output.weight`, loaded without MTP tensors. Drafting uses the original GGUF and its original compact MTP head. Single-CPU target and draft use original weights. The synthetic GGUF has an explicit output head and matches the Swift head contract. This is a test adaptation, not a production model/sampling change; unmodified real-4B TP target backend sampling is not claimed.

Build/meta-plan host intervals below are the union of `llama_build` and `meta_build` events in C and D1, avoiding nested double counting. D0 means no build interval, not zero total host work. Decode wall includes all V/C/D CPU compute and excludes sampling readback and trace flush; real-model wall differences are noisy and do not establish a speedup.

| Fixture / backend / K | Build+meta us/round, up D | Patch D | Decode wall us/round, up D | Patch D |
| --- | ---: | ---: | ---: | ---: |
| synthetic / CPU / 3 | 117.3 | 0 | 1002.2 | 872.8 |
| synthetic / CPU / 5 | 122.5 | 0 | 1355.4 | 1218.0 |
| synthetic / Meta(CPU,CPU) / 3 | 989.4 | 0 | 2787.8 | 1714.1 |
| synthetic / Meta(CPU,CPU) / 5 | 987.6 | 0 | 3495.1 | 2418.2 |
| real / CPU / 3 | 217.7 | 0 | 163733.7 | 165938.6 |
| real / CPU / 5 | 210.1 | 0 | 192327.1 | 194816.6 |
| real / Meta(CPU,CPU) / 3 | 1076.3 | 0 | 240285.1 | 238208.0 |
| real / Meta(CPU,CPU) / 5 | 1092.1 | 0 | 332030.0 | 333963.7 |

Tables are medians across per-run means. Summaries: `codex/synthetic-reuse-results.json`, `codex/real-reuse-results.json`. Raw records and traces remain in ignored `codex/reuse-results/`; `codex/summarize_reuse.py` reproduces the timing columns. The full matrix preceded the final detach fix. A final real-model seed rerun covers both K values and both backends after that fix; every recorded byte still matches. Steady build logic is unchanged.

`codex/reuse-contract.cpp` verifies two-versus-one-output uniform bytes and committed RNG draws, wrong UID and stale epoch rejection, and pipeline scheduler sharing refusal. Its meta fixture keeps a plan borrowing an input from another container, frees the input's original plan, rotates containers M40 times and checks exact outputs; it also reserves a CPU split after an allocated meta split. `codex/negative_reuse.py` links CPU mutation libraries from the build objects. Removing binding restore returns M1 with `stale bindings or RNG draw count`; removing only the compute epoch comparison returns M1 with `stale epoch escaped public compute guard`. These are implementation mutations, not perturbed reference files.

M: existing `test-llama-archs --tp-sampling` passes, including its 64-seed multi-output checks and 4000-draw distribution check; `test-sampling` passes. M: final real in-process server, original GGUF, both backend samplers, subset drafting, K=3/K=5, seeds 1729/2718, M48 completion tokens per run, preserves token and final target-logit bytes against base. Hashes are in `codex/server-reuse-results.json`. No HTTP server process was started.

Reproduce with `CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_reuse_cpu.sh`. It builds CPU-only, generates the target fixture inside `build-tmp`, runs contract/mutation checks, the seeded matrices, and real rejection tests. All compute uses at most eight threads.

## Frozen GPU A/B and limitations

M: base and patch CUDA builds completed and were frozen; SOURCE and both SHA256SUMS manifests are versioned for each arm. The patch binary comes from `0654a0c0f` (production source is identical to the final evidence commit). `codex/freeze_reuse.sh` compiles with CUDA 13.2, architecture 120 (the project maps it to 120a), NCCL off, and `-j8`. It copies the server as `round-trace`, copies libraries, and hashes binary files plus SOURCE/manifests. It does not launch a model/server executable. The original base archive is inside this worktree.

Parent command: `RUN_GPU=1 CUDA_VISIBLE_DEVICES=0,1 XID_SINCE='<parent-selected timestamp>' nice -n 19 bash codex/gpu_ab.sh --execute`. Default K=5 is the weekend IQ4_XS S5 configuration; use `--k 3` for a separate K=3 series. Order is up/patch/patch/up, tracing off, frozen byte verification before/after each boot, fixed think/tool prompts and seeds, normalized completion plus authoritative counter identity as a hard gate, `predicted_ms / drafts` for D ms/round, exact executable PID/start-time stop, idle pair/port/process checks, Xid check since `XID_SINCE`, and a configured 2400-second maximum budget. M: the dry plan launches no subprocess and makes no network call; the mocked harness passes M12 execution/frozen/budget/identity/Xid controls, including a complete frozen boot snapshot before its blocked launch. GPU completion identity does not replace GPU proposal/q identity.

Compliance exception: initial CMake CUDA configuration executed its native-architecture ABI helper despite the explicit architecture flag. `CUDA_VISIBLE_DEVICES=''` was set and the helper reported no devices. This violated the prohibition on executing any CUDA binary. No model/server/benchmark CUDA executable or GPU monitoring command ran. Subsequent configurations use `CMAKE_CUDA_COMPILER_FORCED=ON` to prevent that probe. All deliberate writes are inside this worktree; no network, external posting or external Git mutation occurred.

## Upstream fit and open items

This extends #28549's existing two arenas with retained split state and gives #22041's UID-based meta reuse an explicit plan lifetime. It uses the existing graph-plan API rather than adding a second cache policy. The input ordering concern in #28058 is handled conservatively for alternation and reset; the existing same-result non-pipeline path is preserved. Issue status and maintainer acceptance were not refreshed under the no-network rule. This is a private prototype, not a submission.

Open: trace-off GPU gain, GPU token/proposal/q exactness, CUDA graph replay behavior, K=5 phase attribution, host RSS/virtual-arena high-water usage, and multi-context stress. The global metadata mutex can serialize planning/uploads across shared-model contexts. Container pools retain their high-water allocation; the existing large compute arenas multiply with retained owners. Public scheduler family APIs require review, and other architectures/backends lack this task's exactness matrix. Pipeline sharing is disabled rather than generalized. Deferred-logit fetching can add request-boundary readback cost. The forecast is neither an additive saving nor a measured GPU result.

## graph-reuse-ab2: first GPU A/B and decisive replacement (2026-10-03)

Conclusion: the first A/B is inconclusive for steady performance. The harness defect is fixed; no production or sampling code changed. The K=5 tool regression is D17.0135 ms per request, but neither the logs nor CPU execution proves whether its GPU cost is fixed or recurring. CUDA warm-up/capture remains an unverified hypothesis. Do not present the whole-request ratio as a steady GPU result.

### What the first logs establish

Sources: `codex/runs/20261003-141403/{ab-results,ab-summary}.json` and `codex/runs/20261003-141705/{ab-results,ab-summary}.json`, plus each boot's `server.log`. Each configuration has M2 boots per arm and identical normalized completion/counters. All times below are arithmetic means/differences of recorded server `predicted_ms`; rounds are the authoritative verification counter, not draft-token count divided by K.

| K / case | Rounds/request M | up generation ms D | patch generation ms D | patch-up ms/request D | patch-up ms/round D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 / tool | 5 | 202.0750 | 219.0885 | +17.0135 | +3.4027 |
| 5 / think | 24 | 872.1630 | 850.6740 | -21.4890 | -0.8954 |
| 3 / tool | 11 | 362.9680 | 364.8985 | +1.9305 | +0.1755 |
| 3 / think | 24 | 787.3480 | 785.4605 | -1.8875 | -0.0786 |

Observed: K=5 tool finishes after M31 tokens despite a configured 384-token cap. Raising that cap alone cannot extend this prompt. The original AB branch never called its steady-round collector or enforced the configured minimum. The target's reported `graphs reused` is M13 for tool and M36 for think in every arm. Those target-only counts do not report draft C/D1 reuse, meta planning, or CUDA capture. The logs lack the phase intervals needed to assign the D17.0135 ms to any of those operations.

Source-derived behavior:

- `common/sampling.cpp:563` and `:768`: stack2 supports a candidate-only backend chain under grammar/budget; this differs from the older 68d99c67d report. Grammar/budget and rejection RNG stay on the host, while target top-k runs on the backend. `tools/server/server-context.cpp:1914` attaches that request's target sampler, and `:390` detaches it at release. Patch `src/llama-context.cpp:1385`, `:1406`, and `:2573` invalidate target graphs and fetch pending logits before deleting bindings. This is request-boundary work required by the existing post-detach logit API, not a per-round attachment. A target reserve is conditional on node capacity (`:1387`), not unconditional on each request. Attachment precedes prefill, and final generation time is set at `tools/server/server-context.cpp:4127` before release. Thus the post-release fetch cannot directly explain the reported generation-time excess; it can affect request/next-request latency (D).
- `common/speculative.cpp:1470`, `:1512`, `:1522`, and `:1676`: the draft backend top-k chain is attached at speculative construction, detached at destruction, and retained between requests. Request begin resets the host draft RNG; `spec_retune` changes its host chain only for a changed temperature/seed. None of these installs a backend sampler each round.
- `src/llama-context.cpp:1518`, `:1530`, and `:1543`: a miss builds/allocates one graph for that arena; a hit executes the retained plan. Prefill overwrites the no-output catch-up arena, and KV-mask/accepted-batch shape changes can miss the exact compatibility check. `ggml/src/ggml-backend.cpp:2166` creates an owned meta plan on allocation. CPU traces show one meta build per corresponding miss, not a duplicate plan build during the ensuing compute.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2680` and `:4552`: CUDA keys graphs by the first node address and requires stable properties before capture. Retained meta containers change graph ownership/addresses, so different capture/warm-up history is plausible (G). There are no capture counters in the first GPU logs. CPU cannot test this explanation.

Agent traffic reaches the target attach/detach and prefill paths on each call (D). If the GPU excess is a request startup cost, it can recur on short calls and outweigh a round saving (G). CPU evidence below rules out a recurring C/D1 rebuild in the tested shapes; it does not rule out CUDA or request-boundary readback cost.

### Replacement A/B

`codex/gpu_harness.py:248` records each input body, response, SSE chunks and authoritative metrics. `:344` measures server-time differences between adjacent full-K draft counters after the first three rounds and before the last K+1 output tokens. The tail exclusion matters: `tools/server/server-context.cpp:489` reduces drafts near the token limit, so `draft_n / K` is not a valid total-round count. Final draft/accepted counters must match metrics, and the possible terminal draft deficit is bounded. Missing, reversed, inconsistent, or insufficient intervals fail the run. Every long request must independently retain at least the configured 40 intervals; repeating short requests cannot satisfy that gate.

Each boot (`:377`) runs the original tool request immediately after the original warm-up, then long tool/think requests at seeds 1 and 2, then eight sequential short agent follow-ups. Long prompts ask for a 1000-function patch with a 2048-token cap so generation is expected to reach the cap; premature completion still fails if it has too few intervals. The tool case keeps thinking off; think keeps it on. Follow-ups retain assistant/tool history with deterministic IDs and `cache_prompt=true`; all except the first must show positive `cache_n`. Generated commands are fixture text and are never executed.

`codex/gpu_harness.py:410` reports per-request `prompt_ms_M`, `outside_prompt_generation_ms_D`, `generation_excess_ms_D`, and `request_fixed_ms_estimate_D` separately from `steady_ms_per_round_D`. The fixed estimate is `wall_ms - prompt_ms - rounds * steady_ms_per_round`; generation excess omits outside time. Short requests borrow the same boot's long-tool mean. These are affine residuals, not a pure setup measurement: context, content, acceptance, HTTP/dispatch/release and terminal work can affect them. Negative estimates remain visible. Mean/median/spread and early/late half means expose nonstationarity. Pair differences and their mean use up/patch/patch/up; no outlier trimming hides warm-up or shape transitions.

Guards retained: model/input hashes, full frozen manifests and boot snapshots, fixed seeds, normalized completion plus metric/prompt/cache counter identity, idle selected GPU pair, occupied-port/other-server rejection, Xid checks, PID plus start-time ownership, exact-PID stop, signal cleanup and the configured <=2400-second budget per invocation/K. `up` additionally must have SOURCE `2b645a655e59d584c66fb491253d5b9f2c2867e2`. Default tracing is off. `--diagnostic-trace` enables existing instrumentation in both arms and writes `request-builds.json` with first-three/later llama/meta/CUDA build counters; never compare those timings directly with trace-off runs.

Parent-only queue commands (not executed here): `RUN_GPU=1 CUDA_VISIBLE_DEVICES=0,1 XID_SINCE='<parent timestamp>' nice -n 19 bash codex/gpu_ab.sh --execute --k 5` and a separate `--k 3` invocation. If the cold/short excess persists, run the same series with `--diagnostic-trace` to test the capture hypothesis; that execution also has the budget guard. Dry plans are `codex/graph-ab-plan.json`, `codex/graph-ab-k3-plan.json`, and `codex/graph-ab-diagnostic-plan.json`.

No binary-producing production source changed, so both existing frozen builds and SHA256SUMS are retained. Refreezing identical production code would add build-version differences without testing a fix. The patch snapshot remains the production code of `0654a0c0f`; up remains stack2. No GPU executable, GPU monitor, HTTP request, or network call was executed by this worker.

### CPU verification and limits

`codex/reuse-server-cpu.cpp:54` now serves sequential tasks inside one in-process server with no listener, records partial server timings and actual verification counts, retains prompt prefixes, and saves every token plus the final target logit bytes. `codex/run_ab2_cpu.py:67` runs up/patch/patch/up at K=3/K=5 on Meta(CPU,CPU), two 64-token requests per boot, active numeric grammar, original Qwen3.5-4B MTP weights and the existing compact draft subset. Reproduce after compiling that fixture against each CPU build: `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/run_ab2_cpu.py`. Both build caches have `GGML_CUDA=OFF`; fixture compilation uses the existing CPU server/static libraries and no CUDA toolchain.

M: all original-model boot/request pairs preserve token and final-target-logit bytes, metric counts, and cached prefixes. Each arm retains M204 SSE intervals at K=3 and M198 at K=5. After the first three complete rounds, patch C and D1 each hit M212/212 (K=3) and M200/200 (K=5); up misses every one and rebuilds one meta plan per phase/miss. Patch's retained C/D1 build union is D0; up spends D1.001 ms/round (K=3) and D1.034 ms/round (K=5). Patch's first-three build unions are D0.413-2.193 ms/request at K=3 and D0.495-0.564 ms/request at K=5. Follow-ups rebuild C once and retain D1; no per-request scheduler reserve is logged after initial setup. Raw evidence and CPU timing residuals are in `codex/ab2-cpu-results.json`.

CPU wall is dominated by model compute and varies with host load. The affine generation residuals range from D-659 to +355 ms across these runs, so they do not establish a fixed-cost sign or GPU speed. The original tied target head refuses TP backend sampling: this first matrix exercises host target sampling, not Swift's candidate target path. The explicit-head matrix below exercises that additional path.

M: explicit-head K=5 Meta(CPU,CPU), up/patch/patch/up, also preserves every request token and final logit byte. Both arms retain M198 SSE intervals. Patch C and D1 each hit M200/200 retained rounds; up misses every one. Later build union changes from D1.122 to D0 ms/round; patch's first-three C/D1 build unions are D0.443-0.771 ms/request. Every follow-up reuses M89 prompt tokens. Target attach/detach is logged for each request with no TP fallback; a target scheduler reserve occurs only on the initial attachment. The fixture adds an explicit `output.weight` made from byte-identical embedding blocks using the existing `prepare_cpu_target.cpp`; all original MTP tensors remain in the file. This is a test model adaptation, not a serving model change. Reproduce with `CUDA_VISIBLE_DEVICES='' CPU_TARGET_MODEL="$PWD/build-tmp/reuse-untied-target.gguf" nice -n 19 python3 codex/run_ab2_cpu.py`. Results: `codex/ab2-candidate-cpu-results.json`. Its reuse-disable negative control also preserves bytes and removes all retained C/D1 hits.

| CPU fixture / K | up interior ms/round D | patch interior ms/round D | Interpretation |
| --- | ---: | ---: | --- |
| original tied head / 3 | 351.879 | 364.622 | host target sampling; model-compute/host-load noise |
| original tied head / 5 | 454.968 | 465.093 | host target sampling; model-compute/host-load noise |
| explicit head / 5 | 481.682 | 484.900 | target backend candidates enabled; no GPU inference |

CPU columns are medians of the four per-request interior means in each arm. They are not estimates of GPU behavior. M: the earlier single-CPU pilot validated the same counters with M52/M47/M52 retained intervals and M89/M159 cached tokens in sequential requests. An initial streaming fixture attempt expected final-chunk tokens, which the server intentionally leaves empty; it failed before comparison and was corrected to collect partial-chunk tokens.

M: `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_harness.py` passes the plan/no-launch, frozen, budget, identity, Xid and new interval controls. Synthetic timing data recovers exactly D30 ms/round and D40 ms generation excess; a simulated patch recovers D-1 ms/round and D+10 ms/request independently. Negative controls reject short windows, missing adjacent counters, reversed time, mismatched draft/accepted metrics, absent cached prefixes, and unfinished/error SSE responses. `run_ab2_cpu.py:78` disables graph reuse: bytes remain identical while every retained C/D1 hit disappears, failing the reuse condition as intended. No new test file was added under `tests/`.

Open: the precise cause of the first GPU tool excess, native GPU proposal/q identity, capture costs and the new long prompts' behavior on Swift. The queue run must pass the new size/cache/identity gates before any speed decision. If it fails a long-size gate, change the workload and rerun all arms; do not silently accept or pool short requests.
