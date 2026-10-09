# AllReduce duplex review

**Verdict: do not treat the current kernel as PTX-proven exact.** The first round has a producer-side release-ordering gap. The reader's acquire-plus-barrier pattern is valid once publication covers every producer thread. Slot overwrite and per-element arithmetic are sound under the existing serialized launch contract; the GPU negative controls have not been demonstrated.

Reviewed `f16365941..f9d0748b6`, local commit `10fc9f6fd`, and the driver/window at HEAD `f3dc3f2db`. Locations below refer to HEAD unless a revision is named. This is source review: no builds, tests, CUDA executables, GPU work, commits, or performance measurements. Existing CPU/SASS results in `REPORT-ar-duplex.md` are author-reported evidence, not independently reproduced evidence. Only this review file was written; the pre-existing modification to `codex/PROMPT.md` was preserved.

| Question | Verdict | Confidence / limit |
|---|---|---|
| 1. Every peer load sees round-r data | **defect** | High confidence in the missing first-round PTX ordering edge; an actual stale read on this hardware is unmeasured. |
| 2. Remove the old read-side system fence | **OK, conditional on fixing Q1** | Acquire plus the reader CTA barrier suffices for properly released generic-proxy data. The fence is not semantically limited to ordering own stores. |
| 3. Fast rank overwrites an unread slot | **OK** | Serialized host calls, identical rank schedules, and ordered streams are required. Token wrap is a separate defect described below. |
| 4. Per-element arithmetic unchanged | **OK** | Source-level expression and ownership are unchanged. Universal compiled byte identity remains unverified. |
| 5. Gates really catch a missing wait | **unsure** | The gate is structurally fail-closed, but there is no recorded GPU mutant result here. It does not test Q1. |

## 1. System-scope ordering

Relevant PTX rules: a volatile access is system-scope relaxed; fence plus strong flag store forms a release pattern covering operations before its first instruction. Matching system-scope flags/fences are morally strong. CTA barriers synchronize participating threads, and base causality composes program order and synchronization. The causality axiom excludes older writes for a causally ordered data read; the weak data accesses need not themselves be morally strong. See PTX ISA 9.2 [operation types](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#volatile-operation), [release/acquire patterns](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#release-and-acquire-patterns), [morally strong operations](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#morally-strong-operations), [synchronization](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#memory-synchronization), and [causality](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#axiom-causality).

**Producer release gap:** `ggml/src/ggml-cuda/allreduce.cu:254-258`. Thread 0 fences before the publication barrier, then performs a volatile flag store. For another producer thread, the barrier establishes its stores before thread 0's flag store, but not before thread 0's earlier release fence. The other thread's fence cannot form a release pattern with a flag store executed by thread 0. Choosing thread 0's fence earlier in Fence-SC order than that other thread's fence leaves no release path from that thread's data to the peer acquire. This inference uses the release-pattern restriction above, not an assumption about PCIe write-drain latency.

Concrete permitted first-round execution, with `n_rounds == 1` and an active vector owned by thread 32 in block 0:

1. Rank A thread 0 issues its data store and system fence, then waits at the publication barrier.
2. Rank B completes its publication fences and its final signal fence. Its thread 0 waits for A's flag.
3. A thread 32 issues its data store and fence, then joins A's publication barrier. Fence-SC order places this fence after B's completed publication/signal fences.
4. A thread 0 stores the flag. B acquires it before A thread 0's trailing system fence establishes any further ordering, crosses its reader barrier, and loads the old value of A thread 32's data.

No release head downstream of A thread 32's barrier participation precedes the flag. A trailing fence after the flag is not that release head. This is a model-permitted counterexample, not an observed GPU failure. The same publication layout exists in the base at `f16365941:allreduce.cu:152-163`; a baseline match does not independently prove publication correctness.

Later rounds have an important difference: their stores precede the reader barrier at `:225`, which precedes thread 0's publication fence at `:254`. Thus the identified missing edge is specifically the initial round, including a single-round launch. The one-round local control retains it.

Recommended minimal repair: publish with a system-scope **release flag store after the publication barrier**, or put thread 0's release fence after that barrier and before its volatile flag store. This supplies the missing producer edge without reinstating a fence between next-round stores and previous-round loads. With that repair, the reader path `:213 -> :225 -> :235/:248` distributes the acquired data to every participating reader thread through CTA synchronization. No reader thread needs its own system acquire. This application follows [PTX's transitive synchronization rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-synchronization).

Cases requested in the brief:

- **L1:** ordinary generic-proxy loads are covered by a valid causality path; cache hints do not weaken that contract. Counting loads or assuming that `CCTL.IVALL` invalidates every relevant cache is not a substitute for the producer edge.
- **`ld.global.nc`:** it is outside the PTX consistency model. Removing outer `host_other` restrict at `:145` addresses the reported lowering, but inspect all wire/type and tail instantiations in the actual binary. The helper still has restrict parameters (`common.cuh:800`); source inspection alone is not a binary opcode gate. These distinctions follow [PTX applicability](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope-and-applicability-of-the-model) and [cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators).
- **Tail line:** `:198-199` and `:244-248` use block 0's final-round flag. A shared cache sector with another block's final vector does not itself introduce overlapping byte accesses. A single-round tail written by a nonzero thread shares the producer gap above; a later-round tail does not introduce another ordering gap.
- **Mapped-host flag atomicity:** naturally aligned flag words have one writer, and need no atomic RMW. CUDA preserves naturally aligned host-memory accesses as single accesses to other devices on supported topologies. `hostNativeAtomicSupported == 0` does not prohibit this plain load/store protocol. See [mapped memory](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-programming-guide/index.html#mapped-memory).
- **Virtual aliases:** no distinct peer alias is introduced here: both launches receive the stored device mapping. A future implementation using distinct virtual aliases needs to revisit proxy ordering.
- **In-place input/output:** safe for exact `sendbuf == recvbuf`, as explained in Q4; partial overlap is not claimed.

**Token wrap defect:** `:454` truncates the counter to the flag's width, while `:604` initializes every unused round flag to zero. D: the token period is `2^32` acquisitions. If a slot has previously used only round 0, its higher flags remain zero; at the acquisition with token zero, a larger launch can satisfy its higher-round waits before the peer publishes them. For example, after both ranks publish round 0, the faster rank reads an unproduced round 1 because the untouched round-1 flag already equals zero. This remains possible after repairing publication; use a coordinated reset before repetition or another protocol that excludes stale equal flags. The initial-round flag is continuously overwritten, so this is not the ordinary slot-overwrite case.

## 2. The former pre-load fence

At the base's `allreduce.cu:180`, the fence supplied system acquire/SC ordering as well as ordering earlier local operations. Describing it as only ordering own stores is incorrect: `__threadfence_system()` is a system-scope sequentially consistent fence, not merely a drain instruction. See [CUDA memory fences](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-programming-guide/index.html#memory-fence-functions).

For a properly released publication, replacing that reader fence with `ld.acquire.sys` in thread 0 followed by `__syncthreads()` is sufficient. The CTA scope of the latter does not erase the earlier system-scope acquisition. Q1 prevents unconditional approval of the current end-to-end protocol; restoring the old reader fence is not the recommended producer repair. Also, a fence cannot make `ld.global.nc` acquire semantics portable.

## 3. Slot and round reuse

Within a launch, rounds have disjoint vector addresses (`:185-199`, `:229-248`) and separate flags. Advancing a round never overwrites the previous round's storage. A fast block can therefore lead its peer without corrupting unread round data.

Across ordinary calls, `ggml_cuda_ar_acquire_slot()` synchronizes the reused slot's completion events on both devices (`:441-454`). For multiple chunks, only the final chunk records events (`:1109-1110`), so the report's assertion that events explicitly protect every chunk reuse is too strong. The ordered-kernel protocol nevertheless protects those intermediate chunks: before fast rank starts chunk N+2 and reuses N's slot, it has completed N+1, which requires observing the slow rank's N+1 flag. The slow rank cannot publish that flag until its N kernel, including all final reads, has completed on the same stream (`:1047-1066`).

This proof assumes serialized accesses to the shared pipeline and matching launch geometry. Concurrent host callers, unordered replacement streams, or a broken wait invalidate it; none is introduced by the reviewed change. Token wrap can cause premature higher-round reads as described in Q1, but I found no ordinary fast-rank slot overwrite.

## 4. Arithmetic and byte identity

The vector expression at `:238-240` and scalar tail expression at `:245-248` are unchanged from `f16365941:allreduce.cu:191-200`: `cast_dst(float(cast_wire(own)) + float(peer_wire))`. Wire selection and inactive-shard zeroing are unchanged. Local thread/round knobs only change which thread owns an element and when it executes.

Each thread stores every element it will subsequently update before writing that element's output. Future-round vector addresses and the tail are disjoint from completed outputs. Exact in-place operation therefore introduces no new input dependency or arithmetic reorder.

This establishes source-level arithmetic equivalence conditional on correct peer data. It does not establish universal byte identity across compilations, particularly NaN payload selection, or rescue Q1/token wrap. The gate compares raw output hashes against the base, including denormal and NaN bytes; its CPU reference separately skips denormal pairs and accepts any NaN payload (`ar-duplex-driver.cpp:115-125`). Hash equality is a finite-case identity check, not a proof of all output bytes for all inputs.

## 5. Identity and mutants

`bw-ar-duplex.sh:146-162` requires successful base/reference checks, engagement, case counts, and identical output files for every normal arm. `:166-181` adds a rank-1 stall with the wait intact, then requires every stalled no-wait arm to exit with a result mismatch, emit `ar-duplex: FAIL`, and differ from the base. A mutant that happens to pass or crashes is classified as a void negative control, not success. The unstalled mutant at `:183-184` is explicitly informational.

A missing-wait execution can be detected: rank 0 reads the peer staging bytes while rank 1 is stalled before its stores; the buffers contain an earlier invocation's contribution or uninitialized bytes. Inputs vary between repetitions (`ar-duplex-driver.cpp:206-208`), so stale finite values normally change checked outputs. The configured stall makes this plausible but does not force that scheduling outcome, and a passing gate would demonstrate only the recorded executions. `codex/window` is absent, and the author report explicitly says no GPU execution occurred; actual detection remains unverified.

**Coverage defect for memory ordering:** the CPU model replaces the actual volatile flag store with `__ATOMIC_RELEASE` (`ar-duplex-cpumodel.cpp:53`), which supplies exactly the missing post-barrier release edge. It also joins all threads between calls (`:96`, `:123`), so it does not exercise concurrent slot reuse across launches. Its reported passing runs cannot validate Q1 or the cross-launch proof. The no-wait mutants remove readiness and acquire together; they cannot distinguish a missing acquire from a missing producer release. A PTX ordering proof and inspection of the actual generic-proxy load/release lowering are required; successful GPU races alone cannot settle the formal guarantee.

The window additionally lacks selectable stages/canary support (`bw-ar-duplex.sh:237-246`) and real recorded driver-parser evidence. These are unresolved harness requirements, not evidence of a kernel failure. No parser or fixture was changed or executed during this read-only review. HIP correctness/lowering and actual duplex performance remain unverified.
