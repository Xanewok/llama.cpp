# Package 101: record existing AllReduce calls

Private WIP above package 94. One optional registry procedure:

    typedef bool (*ggml_backend_comm_graph_compute_t)(void * comm_ctx, struct ggml_cgraph ** graphs, size_t n_graphs);

Meta passes its existing split graphs as `graphs[step*n_ranks + rank]`. The comm backend queues each step and an in-place sum of its last tensors between steps. True queues the entire plan; false queues no work and preserves normal split execution. No new ggml op, CUSTOM node, graph arena, production environment variable, or public readiness protocol.

CUDA qualifies every rank and reduction before capture. Only the existing internal chunked P2P path with device epochs is eligible. The existing `allreduce_tensor` call is recorded with all rank streams capturing. NCCL, HIP, MUSA, disabled/incompatible graphs, host staging and copy-engine boundaries use the existing split plan. A fork plan containing a gather also stays split.

Collective executables use graph-object cache keys, separate from ordinary tensor keys. Existing tensor-property snapshots and warmup state cover all ranks; boundary lengths also participate in invalidation. Full property comparison uses UID zero because one subgraph UID cannot certify the rest. A cold or changed sequence gets one split warmup; the next unchanged sequence captures or updates. Finish all captures/instantiations before launching either rank. There is no claim about fixing per-request latency.

Both split and captured chunked P2P calls advance the same per-block device epochs. Epoch e uses parity to reuse e-2's staging slot only after both ranks passed e-1 by stream order. Copy-engine calls use separate resources. Do not enable early dependent kernel completion. Missing peers after submission can hang; there is no device trap or recovery protocol.

`test-ar-p2p` owns the bitwise split/collective replay check, inactive-rank case, arena rebuild and property-change phases, plus host rejection of incomplete collective inputs. Its P2P CTest entry uses the same binary. Tests are source only here. Host syntax checks do not establish CUDA compilation, capture legality, replay progress or performance. See `REDUCE-101.md` for decisions and lost coverage.
