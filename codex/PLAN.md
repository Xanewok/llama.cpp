# graph-capture-spike plan (implementer)

Spec: codex/DESIGN-graph-capture.md. Env switch GGML_META_AR_NODE (unset/0 = stack6 behaviour).

## Files
- ggml/src/ggml-cuda/allreduce.cu/.cuh: node-mode plan/launch, shared phase-1/phase-3 device helpers, node kernel
  (device-resident gen[site], per-block done guard, bounded spin + __trap), per-site slab staging/ctl memory.
- ggml/src/ggml-cuda/ggml-cuda.cu: proc `ggml_backend_comm_allreduce_node`, `case GGML_OP_CUSTOM` in compute_forward,
  graph bookkeeping (has AR node -> always capture, skip warm-up; per-forward `ar_node` count).
- ggml/include/ggml-backend.h: typedef ggml_backend_comm_allreduce_node_t. ggml/src/ggml-impl.h: node magic.
- ggml/src/ggml-meta-ar-plan.h (pure planner, header-only) + ggml/src/ggml-backend-meta.cpp: resolve proc, qualify
  boundaries in the boundary loop, insert CUSTOM nodes, size arena, log `meta_ar_node:` lines, CPU dry-run.
- codex/: PLAN.md, test-ar-node-plan.cpp, ar-node-server-cpu.cpp (np1/np2 fixture), cpu scripts, cpu-results/,
  freeze script, memcheck/, depth-experiment.toml, depth-item.toml.

## Deviations from the design (decided while reading code)
1. AR node must always carry GGML_TENSOR_FLAG_COMPUTE (CUDA backend skips non-compute nodes; a skipped AR deadlocks
   the peer); the shard's compute flag lives in the descriptor (descriptor key includes it).
2. Node tensors live in a separate per-plan pool (`nodes_ar`), not nodes_aux: nodes_aux entries are rewritten on every
   compute call by get_node_aux.
3. Descriptors are deduplicated in the pipeline by (rank, site, data, count, type, compute): stable pointer across
   plan re-creation, data pointer resolved at plan time (plans are created after sched alloc).
4. `ar_node` counter is emitted by the CUDA backend per forward (count = AR nodes of the graph, rank-0 only, so it is
   comparable to ar_call), not in launch: launch is not executed on CUDA-graph replay.
5. Graphs with AR nodes skip CUDA-graph warm-up (eager submission of a whole forward on device 0 before device 1 is
   launched can exceed the host launch queue and block); non-capturable graphs abort loudly.
6. Qualification also requires wire count % 8 == 0 (no tail elements): makes block->index ownership independent of
   the count, so the per-block done guard is exact when V/C/D graphs share a site.

## Tests (CPU, cpugate)
a. base vs spike identity, env unset, CPU/Meta(CPU,CPU), np1/np2, temp 0/1, MTP draft.
b. env=1 on CPU: identical bytes, `unavailable` line, would-be plan lines (M->K per graph kind).
c. codex/test-ar-node-plan.cpp unit test + wrong-expectation negative control.
d. test-only hook GGML_META_AR_NODE_FORCE_TEST=1: node inserted on CPU, stub aborts.
e. test-alloc, test-sampling.
