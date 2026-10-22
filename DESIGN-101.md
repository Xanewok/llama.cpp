# Package 101: graph-recordable AllReduce

Base: ff5a23376 plus package 94's root series. Private WIP; no submission or runtime validation is authorized here.

## Decisions

(a) Two optional comm procedures, discovered by the existing registry mechanism: `ggml_backend_comm_allreduce_graph` queries support and initializes caller-owned per-rank nodes; `ggml_backend_comm_graph_ready` collectively checks the per-rank graphs before execution. Meta owns graph placement and lifetime; the comm backend owns the node representation, capture constraints and stability checks. Meta knows no CUDA types, thresholds, epochs or node marker. CUDA uses a private CUSTOM node identified by its callback, not a new public operator or magic in ggml-impl.h. A capability query alone cannot express the executable work or ensure both ranks choose the same layout.

(b) No new production environment variables. Existing `GGML_CUDA_P2P`, the selected internal AllReduce implementation, compiled CUDA graphs, architecture and graph compatibility gate the path. NCCL, HIP, MUSA, graphs disabled and unsupported sizes retain split execution. Existing copy/BF16 threshold settings keep their meanings. Tests suppress optional procedures at registry lookup to obtain a split reference without a production toggle.

(c) One 101 change above the separate 94 dependency. Device epochs and graph placement must ship and revert together: epochs alone measured no gain, and host tokens cannot be replayed. Host-driven chunked P2P calls share device epochs with graph calls so their staging slots cannot disagree. Only captured calls have the peer timeout.

(d) Observed by reading: CUDA compares complete tensor properties after a UID change; the spike always captures a changed merged graph immediately. A change to one node thus visits the whole forward instead of one subgraph. `cudaGraphExecUpdate` failure destroys and reinstantiates the complete executable. A CUSTOM ggml node captures ordinary CUDA kernel/memset nodes, so CUSTOM alone does not imply update failure. Meta arena replacement is another source of pointer changes, not proof of driver instantiation cost. Avoid split/merged cache-key aliasing by keying a merged CUDA graph on its first comm node. Readiness records properties without launching work; after any change, execute the split layout until all ranks have a second unchanged observation. Never eagerly execute a merged graph. Reuse the executable when stable, recapture after a real change, and do not call ExecUpdate immediately after a fresh instantiate.

## Interface contract

`comm_allreduce_graph(ctx, tensors, NULL)` is a side-effect-free support query over all ranks' allocated tensors. With non-NULL nodes, it fills one caller-owned tensor per rank and returns true under the same conditions. Nodes alias the corresponding shard in place, run even when that shard is inactive, and stay valid until the next synchronized plan rebuild or comm teardown. Unsupported calls return false without modifying nodes.

`comm_graph_ready(ctx, graphs)` observes one graph per rank, without launching device work or waiting for a peer. It returns true only if every graph is compatible and its properties are unchanged since its previous observation. Meta checks every merged segment on every compute, including a cold plan, and chooses one common layout for all ranks. The CUDA capture path consumes a retained dirty bit so readiness cannot hide a required recapture. UID reuse follows ggml's existing contract: callers must issue a new UID after a property change.

## Ordering and failure

Both ranks use the same ordered AllReduce sites. Per-block unsigned epochs choose a two-slot ring by parity. Rank r at epoch e has passed e-1, which requires its peer to have arrived at e-1; that peer's e-2 reader finished first by stream order. Do not enable dependent early kernel completion. Copy-engine calls use separate resources and do not advance epochs. In-graph calls trap after 10 seconds of executing peer polling; split-only calls retain an unbounded wait. This is a fatal device error, not recovery, and cannot bound an unscheduled kernel or a stuck driver call.

The first split execution loads kernel modules and library state before a spinning graph can launch. All merged segments must pass collective readiness; an incompatible segment keeps the entire forward split. Arena replacement reconstructs every retained graph and auxiliary-node slot through the high-water marks, including inactive slots after shrinking. No fork plan cache is added to master.

## Falsifiable prediction

Inference, not measurement: most of the observed +21..25 ms (np1), +43..49 ms (np2) per request will appear in capture plus instantiate/update of changed merged target graphs, with about twice as many affected graph variants for np2. It should not appear as steady epoch-kernel time. The package should eliminate merged captures of shapes/properties seen only once, prevent split/merged cache collisions and remove the redundant initial ExecUpdate. Stable decode still captures once per variant and replays thereafter. If the trace attributes less than half of the excess to capture/instantiate/update, or stable forwards repeatedly rebuild arenas, this prediction is false and the fixed-cost cause remains open. A residual cold capture cost is expected; no claim that stability gating eliminates every new-request capture.

## Verification contract

Add source tests for bitwise split/merged identity with repeated replay and changed input, copy-engine interleaving, collective property-change stabilization/recapture, and arena growth after a rebuild with fewer subgraphs. Test graph-disabled builds/runtime as skips. An isolated one-rank graph must fail with the epoch-timeout diagnostic rather than time out externally. No tests, executables, GPU tools, nvcc or builds are run in this task. Only host `g++ -fsyntax-only` checks are allowed. CUDA acceptance and runtime progress require the owner's queued GPU validation.
