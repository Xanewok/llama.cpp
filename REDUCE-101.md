# Package 101 reduction

Scope: changes above package 94 (`1ffc1dcf3`) on master, or above fork `726f57ffe`. Original 101 tips: `4f752eb85` and `5733c813f`. Decisions below are source-based; CUDA compilation and runtime behavior remain unverified. No submission or network search is part of this private WIP.

## Interface decision

Keep one optional procedure, discovered through `ggml_backend_reg_get_proc_address`, like the existing `ggml_backend_comm_init`, `free`, and `allreduce_tensor` procedures:

    typedef bool (*ggml_backend_comm_graph_compute_t)(void * comm_ctx, struct ggml_cgraph ** graphs, size_t n_graphs);

`graphs[g*n_ranks + rank]` contains the existing meta split graphs in execution order. `n_graphs` counts steps per rank; ranks follow backend order from comm initialization. Execute each step on every rank, and sum its last tensors in place between steps; tensors without `GGML_TENSOR_FLAG_COMPUTE` contribute zero. True means the whole sequence was queued asynchronously; false means no device work was submitted and the caller executes its existing split path. Keep graphs, tensors, buffers and comm resources alive until every rank backend synchronizes. No graph ownership transfers.

Can `allreduce_tensor` itself be captured? Yes by source inspection for the device-epoch chunked P2P path: it only enqueues kernels/memsets and does not acquire a host slot, synchronize, or advance a host token. The reduced implementation calls that exact proc with every rank stream capturing. Copy-engine and host-staged paths are rejected before capture. This has not been established with nvcc or a runtime test.

Can a capability bit suffice? No for the current abstraction: meta cannot start CUDA captures, and ordinary backend compute owns capture of each separate graph. Some backend must own the sequence containing both compute and reductions. An optional collective graph-compute proc expresses that work without exposing streams, node construction, or CUDA types.

Is collective readiness needed? A separate public readiness protocol is not needed. Identical split planning fixes site order, but does not imply identical device compatibility, graph cache state, or module initialization. The one collective compute call preflights all ranks and uses existing CUDA graph property comparison and warmup state. Any property or boundary change returns false for one normal split execution; unchanged input contents need no warmup. All captures/instantiations finish before either rank launches.

Is a dedicated op more natural? A public reduction op would need rank/context ownership and cross-backend semantics that ordinary tensor ops do not currently express. Neither a new op nor a private CUSTOM marker is needed when the backend records the existing comm call between split graphs.

## Concept inventory

| Concept in original 101 or its replacement | Problem | Existing mechanism on master (or 94 dependency) | Decision |
| --- | --- | --- | --- |
| `ggml_backend_comm_allreduce_graph_t`, query/init modes | Qualify sites and construct rank nodes | Registry comm procedures, meta split graphs | Fold into `comm_graph_compute`; remove factory/query dual mode. |
| `ggml_backend_comm_graph_ready_t` | Agree before rank launches | Central meta execution, CUDA property checks and warmup | Drop public readiness; preflight inside collective compute. |
| `ggml_backend_comm_graph_compute_t` | Record compute and comm together | Existing comm init/free/boolean allreduce dispatch | Keep one optional, context-first registry proc; false uses existing fallback. |
| Private `GGML_OP_CUSTOM`, callback identity, `ggml_custom_op_params`, userdata, padding initialization | Carry comm context through graph nodes | Existing comm context and ordered subgraphs | Drop all node encoding and CUSTOM dispatch/support changes. |
| `ggml_cuda_comm_ar_unsupported`, node submission aborts, `comm_allreduce` tensor name | Reject accidental eager/non-rank execution | Collective proc owns every participating backend | Drop callbacks, node name and abort paths. |
| `cgraphs_ar`, `nodes_ar`, AR sites, merged offsets, `ar_layout` | Build a second graph layout | Meta `cgraphs`, offsets, auxiliary graphs and plan arena | Drop second layout and its selection state. |
| `n_subgraphs_ar`, `max_subgraphs_ar`, `max_ar_nodes`, `ar_raised` | Size and reconstruct second arena | Existing plan high-water marks and arena reconstruction | Drop added allocations, high-water marks and lifetime rules. |
| Plan execution with a gather (fork only) | Preserve bit-exact gather semantics | Fork's existing exact/gather boundary dispatch | Keep existing dispatch; reject collective sum optimization for plans with gathers. |
| `ggml_cuda_graph_get_key` comm-node scan | Avoid split/merged executable aliasing | CUDA context graph cache | Fold: normal keys unchanged; collective keys use first split graph object's address. |
| `has_ar_node`, `capture_required`, retained readiness observations | Preserve recapture decisions consumed by readiness | `ggml_cuda_graph::warmup_complete`, `node_props`, `instance` | Drop new flags; compare all ranks inside compute, reset existing warmup together. |
| `subgraph_sizes`, temporary flattened property view | Detect changed boundaries and properties across steps | Existing tensor/source snapshot comparison | Keep boundary lengths beside existing node properties; do not mutate or allocate ggml graphs. |
| Aggregate UID handling | Prevent one subgraph's UID masking another's change | Existing UID-zero full comparison | Keep `flat.uid = 0`; compare all nodes each call. Extra host cost is unmeasured. |
| Cold/changed layout stabilization | Load libraries/modules without a spinning launch | Existing CUDA two-observation warmup | Fold into existing state; false executes the unchanged split plan. |
| Capture/update/instantiate/launch sequence | Build reusable CUDA executables | `ggml_cuda_graph_evaluate_and_capture`, `ggml_cuda_graph_update_executable`, capture allocation lock | Keep helpers; capture eager split evaluation and `allreduce_tensor` under all rank captures, then launch. |
| Redundant initial ExecUpdate suppression | Avoid updating a freshly instantiated graph | Existing instantiate/update branch | Keep the `else if` correction already in 101. |
| `EPOCH` kernel template mode, parity ring, per-block counters | Make replay tokens advance on device | 94 host tokens/events cannot advance on graph replay | Keep device epochs; split and captured chunked calls share the same counters. |
| `p2p_kernel`, epoch allocation/free/error log | Enable epochs only for eligible P2P builds | 94 pipeline ownership, existing allocation failure cleanup | Keep private state and allocation diagnostic; no public flag. |
| Epoch pool-size assertion and no early dependent launch contract | Prevent overwriting a peer reader's slot | 94 two-slot staging ring and stream ordering | Keep invariant: epoch e reuses e-2 only after both ranks passed e-1. |
| `ggml_cuda_ar_use_bf16`, `ggml_cuda_ar_use_copy_engine` | Make eligibility match real dispatch | 94 thresholds and wire policies | Keep shared predicates; preserve thresholds and arithmetic. |
| Separate in-graph rank launcher and extracted chunk launcher | Launch comm CUSTOM nodes | Existing `ggml_cuda_ar_allreduce` launches every rank | Drop rank-only entry point; fold chunk launches back into existing function. |
| `ggml_backend_cuda_comm_internal_accepts` | Qualify reductions before any recording | Existing `allreduce_internal` validation | Keep extracted validation shared with ordinary dispatch. |
| `ggml_cuda_ar_in_graph_supported` | Reject host waits and copy-engine capture | 94 pipeline path selection | Keep backend-private predicate, including MUSA false stub. |
| Old timeout env knob, globaltimer, spin counters and `__trap` | Detect a missing replay peer | Kernel clock, mapped host allocation and backend synchronization | Keep a fixed 60-billion-cycle wait bound with sticky per-block failure flags; return without trapping and abort on the host after synchronization. Add no env knob or recovery protocol. |
| `allreduce: P2P kernel sync` and `CUDA AllReduce graph capture` logs | Trace chosen path and captures | Existing CUDA graph debug facility | Drop both production log lines and log-count test coupling. |
| Production env knobs | Select implementation and thresholds | Existing `GGML_CUDA_P2P`, `GGML_CUDA_ALLREDUCE`, graph disable/debug and copy/BF16 settings | Keep existing meanings; add none. Spike-only knobs remain absent. |
| `test-ar-p2p-graph.cpp` and separate binary | Bitwise split/merged comparison | 94 `test-ar-p2p.cpp` and binary | Fold into `graph_cases` in the existing binary. |
| Test registry spies, UID generator, split weights/graphs | Select reference and ensure optimized work actually executes | Existing registry interface and meta split API | Keep one optional-proc spy; drop init/node/readiness spies and context tracking. |
| Replay, inactive shard, arena shrink/grow, property-change phases | Catch stale results and stale plan pointers | Existing backend/AllReduce test patterns | Keep one bitwise comparison of residual and rank-split projection across these phases. |
| Copy-engine interleaving and forced unsupported middle segment | Exercise mixed paths and retained merged slots | Existing AllReduce dispatcher and graph procedure | Test captured/copy-engine/captured transitions on one comm context. Whole-plan unsupported fallback replaces partial merging. |
| Timeout child, `--timeout`, `test-ar-p2p-timeout.cmake`, timeout CTest | Test fatal peer timeout | Existing host test assertions | Drop trap test/script; assert null/empty/incomplete-rank calls return false. |
| Graph-disabled CTest registration | Verify support query skips | Existing graph disable env | Drop separate registration; the graph configuration returns skip code 77 when graphs are disabled. |
| `test-ar-p2p-graph` CTest registration | Opt in to peer staging during graph test | Existing `llama_test` and existing binary | Fold registration onto `test-ar-p2p`; reuse existing env variables, skip code and timeout. |
| `DESIGN-101.md`, `VALIDATION-101.md`, `NOTES-101-port.md` | State contract, evidence and port constraints | This private package record | Fold into this file; exclude package bookkeeping from the upstream patch. |
| `NOTES-94-port.md` | None for 101 | Existing 94 history | Leave master unchanged; remove the fork-only addition. |
| `REDUCE-101.md` | Record requested concept audit and decision | No existing reduction inventory | Keep this private review artifact. |
| Fixed per-request cost prediction and new mitigation | Explain previously observed latency | Brief reports trace disproved capture/instantiate attribution | Drop prediction and add no mitigation without evidence. |

## Limits

Whole-plan preflight gives up merging eligible segments around an unsupported boundary. Deterministic central submission removes a missing-rank caller mistake, not GPU scheduling failures. The old successful GPU evidence covers a different implementation and cannot establish this reduction. The optional graph-compute interface, conservative whole-plan fallback, full property-scan overhead, and CUDA-only comm support still need maintainer scrutiny. No evidence supports attributing the fixed per-request cost to capture/instantiate; this reduction adds no workaround for it.

## Failure bound and lifetime

Each block stops polling after 60,000,000,000 device clock cycles, approximately 30 seconds at 2 GHz. This is a cycle bound, not a wall-clock guarantee across clock changes or device scheduling stalls. The timing check is shared by split and captured chunked calls, including host staging and HIP. Thread 0 writes its block's sticky flag in mapped pinned host memory, fences it, then releases the block to return without reading peer data. Later kernels on that stream return immediately for failed blocks instead of spending another wait budget. A missing peer on the other rank reaches the same bound independently. No device trap, Xid request, environment setting or recovery is introduced.

The CUDA backend context owns one mapped flag per block, allocated before any capture and kept until backend destruction. Captures retain the device alias; the CPU reads the host allocation after `ggml_backend_cuda_synchronize` completes its existing stream synchronization, or after the existing slot-reuse event synchronization. A nonzero flag calls `GGML_ABORT("CUDA AllReduce peer wait timed out (device %d, block %d)", ...)`. Backend destruction drains its streams before checking and freeing the flags. Results following failure are invalid and must not be consumed; this is bounded fatal failure reporting, not cancellation of subsequent graph nodes or recovery.

Both split and captured chunked P2P calls advance the same per-block device epochs. Epoch e reuses e-2's parity slot only after both ranks passed e-1 in stream order. Copy-engine calls use separate staging resources. Do not enable early dependent kernel completion. Collective executables use graph-object keys; full property scans with UID zero and boundary lengths invalidate changed plans, followed by one split warmup before capture/update.

## Source validation and test contract

Environment for checks: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, nice 19, at most two independent compiler processes. No built binaries, tests, GPU tools, nvcc or builds are executed. Host syntax commands on both branches:

    env CUDA_VISIBLE_DEVICES= GIT_NO_LAZY_FETCH=1 nice -n 19 g++ -std=c++17 -Wall -Wextra -fsyntax-only -Iggml/include -Iggml/src ggml/src/ggml-backend-meta.cpp tests/test-ar-p2p.cpp
    env CUDA_VISIBLE_DEVICES= GIT_NO_LAZY_FETCH=1 nice -n 19 g++ -std=c++17 -Wall -Wextra -fsyntax-only -DGGML_TEST_CUDA_GRAPH -I/usr/local/cuda-13.2/targets/x86_64-linux/include -Iggml/include -Iggml/src tests/test-ar-p2p.cpp
    env CUDA_VISIBLE_DEVICES= GIT_NO_LAZY_FETCH=1 nice -n 19 git diff --check

The graph CTest entry uses the same `test-ar-p2p` binary with existing settings `GGML_CUDA_ALLREDUCE=internal`, `GGML_CUDA_P2P=1`, `GGML_CUDA_AR_COPY_THRESHOLD=2097152` and `GGML_CUDA_AR_BF16_THRESHOLD=1`. Native CUDA graph builds link the CUDA runtime only for independent hardware qualification. Two distinct physical devices, Volta or newer, and bidirectional P2P must qualify before collective execution is required. Missing devices, disabled graphs, graph-disabled builds and non-P2P pairs return CTest skip code 77. A qualified configuration fails if the procedure is absent or always declines; successful execution never defines support. The ordinary CTest entry retains the direct reductions without requiring graph support.

`graph_cases` compares fractional F32 residuals and rank-split projection bytes across split and collective execution. It retains inactive-rank zero contribution, five/one/five reduction arena rebuilds and a changed scale parameter/UID. Stable calls must execute collectively. Null context and incomplete rank lists must decline without submitting work.

`replay_cases` captures F32 with BF16 wire, native F16 and native BF16 at 256 and 524296 elements. The latter uses two 1 MiB staging chunks with a vector tail. It warms both arms, then queues six changed-input executions per arm with pinned asynchronous uploads/downloads and one synchronization after the batch. Every rank and replay must match the split bytes and an independent exact integer-sum oracle. It then alternates three copy-engine fallbacks at 1048576 elements with captured calls on the same communication context, requiring exact outputs and resumed collective execution. The success line is:

    PASS: captured F32/F16/BF16, multi-chunk epochs, queued replays and captured/copy-engine transitions

Host syntax checks do not establish CUDA device compilation, capture legality, runtime identity, progress, failure reporting, cache reuse or performance. No deliberate device fault or timeout child is run or added.

## Fork port constraints

Upstream package 94 base: `1ffc1dcf3`, above master `ff5a23376`. Fork base: `726f57ffe`, already containing 94. Pre-fix reduced tips: `cf89c208e` and `52b65d14b`.

Preserve owner-local communication state, plan-local split graphs, the `prepare_only` return before collective compute, and exact split fallback for any gather boundary. Preserve all existing fork input-staging procedures and environment aliases. CUDA, header and test changes are shared; fork differences remain in the meta integration. Remove fork-only `NOTES-94-port.md`. All package design, validation and port bookkeeping lives here and is excluded from an upstream patch.
