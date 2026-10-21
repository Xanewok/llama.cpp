# In-kernel P2P AllReduce for decode (spike, step 1)

Base: fork-ff5a tip 726f57ffe (served). Flag: `GGML_CUDA_AR_P2P_KERNEL=1`, default off, needs the package 94 P2P path
(`GGML_CUDA_AR_P2P=1` or `GGML_CUDA_P2P`, peer access enabled); otherwise a warning and the existing path.
Labels: M measured, D derived, G guess. "By reading" = read from this tree.

## 1. What the decode AllReduce is on this tip (by reading)

- Issue point: `ggml-backend-meta.cpp` `ggml_backend_meta_graph_compute_impl`. The plan ends a subgraph at every PARTIAL node
  (`new_subgraph = ... || split_state.axis == PARTIAL`, ~l.2605). The compute loop (~l.2853-2890) then, per subgraph i:
  `ggml_backend_graph_compute_async(dev j, subgraph i)` for j = 0, 1 (one CUDA graph per subgraph, keyed by cgraph,
  `ggml-cuda.cu` `ggml_cuda_graph_evaluate_and_capture`), then `comm_allreduce` -> `ggml_cuda_ar_allreduce` (non-gather nodes).
- Per forward on Qwen3.8: 128 ARs (64 layers x post-mixer + post-FFN) + 129 subgraphs per device (insights doc, M by reading).
- Decode sizes: rows 1-12 (np1 verify k=5: 6 rows; np2: 12) x n_embd, F32 rounded to the BF16 wire (threshold 1): tens to
  ~120 KiB, below the 1 MiB copy-engine threshold -> the chunked-kernel path, one chunk, 8 blocks x 256 threads.
- Package 94 P2P chunked path is already an in-kernel exchange: `ggml_cuda_ar_kernel<..., P2P=true>` stores the rank's
  BF16 partial into the peer's device buffer (`host_buf[peer]`, cudaMalloc'd on the peer), publishes a per-block flag in
  the peer's memory with `st.release.sys`, spins on its own flag with `ld.acquire.sys` + `__nanosleep(100)`, then reads the
  peer partial with `__ldcg` and sums. No cross-device event, no host copy.
- What still involves the host per AR (chunked path):
  1. `ggml_cuda_ar_acquire_slot`: from the 3rd AR on, `cudaEventSynchronize(ev_pool[i][slot].ker)` on both devices, i.e. the
     host blocks until AR N-2 has finished on both GPUs before it may launch AR N (host run-ahead <= 2 ARs).
  2. The token is a host counter passed as a kernel argument (changes every call: not replayable from a CUDA graph).
  3. `cudaEventRecord(ev.ker)` per AR per device.
  4. The graph split itself: every AR ends a CUDA graph, so each AR costs 2 graph launches + 2 kernel launches + the wait.
- Prior evidence on these costs: host STEP 0.51/0.55 ms/round from the acquire wait (host staging, attrib-stack6, M); pool
  depth 2 -> 4/8 gave no decode gain (Dead ends, D); AR-as-graph-node with host-staged AR: no steady gain, host waits 28.6 of
  32.7 ms/verify in ar_wait (D). So removing (1)-(3) alone attacks the host side only.

## 2. Step 1 design (implemented)

`allreduce.cu` `ggml_cuda_ar_kernel<T_dst, T_wire, P2P=true, EPOCH=true>`, launched from `ggml_cuda_ar_allreduce` chunked path
when `p->p2p_kernel`:

- Epoch counters instead of flag resets or host tokens: one int per block in the launching device's memory
  (`p->epoch[i]`, zeroed at init). Thread 0 of each block increments it at kernel start (unsigned wrap) and broadcasts it via
  shared memory; the epoch is the arrival token and its parity selects the slot. Flags are never reset: slot s holds the token
  of epoch e-2 until epoch e overwrites it, so `!= token` spinning needs no reset. Launch arguments (buffer bases, flag bases,
  epoch pointer, data pointer) are identical on every call: graph-replay safe for step 2.
- Double buffering: the existing 2-slot `host_buf` (1 MiB per slot) and arrival ring, slot = epoch & 1. Slot reuse safety
  without a host wait: a rank at epoch e has passed epoch e-1, which required the peer's arrival at e-1; the peer launches e-1
  only after its e-2 kernel (the last reader of slot s) has exited, by plain stream order. Holds per block: every block of
  this rank's e-1 kernel saw the peer's arrival before the e kernel can start.
- Host side per AR: optional inactive-shard memset + one kernel launch per device. No acquire_slot (call_count untouched),
  no event record. The copy-engine path (>= 1 MiB, prefill) is unchanged and keeps its own events; it does not use
  `host_buf`/arrival, so the two paths interleave safely (the selftest interleaves them).
- Smallest diff: same kernel body (phase 1/2/3 unchanged), one template flag, three extra args; no new knob beyond the flag;
  no model- or size-specific constants.

## 3. Summation order and identity expectation

Phase 3 computes, per element, `cast<T_dst>(float(cast<T_wire>(own)) + float(peer_wire))`, where the peer value was
rounded to T_wire by the peer the same way. Rank 0 computes fl(a' + b'), rank 1 fl(b' + a'): IEEE addition is commutative,
so both ranks are bitwise equal (also asserted by the existing test, which compares to the exact sum). The EPOCH kernel runs
the identical phase-3 code on the identical slot data, so it is bitwise identical to the package 94 P2P path, which the
bw-stack12b gate already proves byte-identical to host staging (`ar-host.bin == ar-peer.bin`). Expectation: np1 tokens
match the off arm exactly (D). np2: the arithmetic is the same, but np2 token trajectories depend on ubatch arrival timing
(harness depth.py note), and the on arm changes host timing; a np2 mismatch with np1 identity points at timing, not values.

## 4. Deadlock and ordering analysis

- Co-residency: 8 blocks per device; each block's thread 0 waits only for the same block index on the peer; no inter-block
  barrier, so progress needs only that each peer block is eventually scheduled. Only AR kernels spin, and they sit on one
  compute stream per device in the same host-issued order, so no device waits on itself through the peer.
- Cross-device event waits exist only in the copy-engine path; every such wait targets peer work enqueued at an earlier or
  equal AR index, and that work completes by induction on the AR index (same argument as package 94; unchanged by step 1).
- Launch skew: device 0 is launched first; its AR kernel spins (8 of 36 SMs) until device 1's starts. The next graph on
  device 0 is stream-ordered behind the AR anyway, so spinning costs nothing beyond the skew itself.
- New in step 1, host run-ahead: the host can now enqueue many ARs ahead (before: <= 2). Safety rests on the device-side
  argument in section 2. Not ruled out: (a) any future launch of this kernel with programmatic dependent launch, or an early
  `cudaTriggerProgrammaticLaunchCompletion` in it, breaks the slot proof; (b) host code elsewhere that implicitly relied on
  the AR's host wait as a throttle (single-GPU builds never had it, so G: none).
- Epoch desync: counters agree only while both ranks launch the EPOCH kernel equally often. The host launches both in one
  loop and aborts on any launch error. A desync would hang both GPUs forever (no timeout, same as today's spin).

## 5. Does the meta backend still split the graph at each AR?

Yes (section 1). Step 1 does not change it: each AR still ends a CUDA graph on both devices. Step 2 (scope only):

- Meta: stop ending subgraphs at PARTIAL nodes whose AR the backend can run in-graph; emit the reduction as a node in each
  device's cgraph (a backend op carrying the comm handle), only for chunked-path sizes. The copy-engine path keeps the split:
  its cross-device event waits cannot be captured into one device's graph.
- CUDA: evaluate that node inside `ggml_cuda_graph_evaluate_and_capture`; EPOCH launch args are already static. The inactive
  shard memset must be static too (capturable memset node, or a compute flag folded into the kernel).
- Graph reuse: one graph per forward per device; capture cost of one-off shapes was +40..+112 ms/request in the pkg 86 spike
  (D), so capture only on reuse-stable decode shapes.
- Safety additions needed before step 2: a bounded spin with `__trap()` and a log (graph replays on one device only, or
  mismatched AR node sets, would desync epochs and hang); an identity test that replays captured graphs N times.
- Prior: the 10-06 AR-node spike crashed in the meta arena (fixed as package 86) and used host-staged AR; with P2P + EPOCH
  the in-graph AR has no host interaction at all, which that spike could not test.

## 6. Expected effect (G)

Step 1 removes the per-AR host slot wait and event records only. Bound from the 10-21 budget (np1 IQ4 round 31.28 ms): COMM
1.70 ms above a ~1.1 ms latency floor, plus host-blocked 0.66-0.70 ms of IDLE. The pool-depth dead end says the host wait is
mostly off the critical path, so step 1 likely lands at the low end (0..-0.3 ms/round np1, G); the larger share needs step 2.
