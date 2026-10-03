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
