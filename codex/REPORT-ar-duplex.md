# ar-duplex: chunked AllReduce kernel with the store of round r overlapped with the load of round r-1

Tree `wt/ar-duplex`, branch `topic/ar-duplex-v2` (the first stack, reported earlier, stays unchanged on `topic/ar-duplex` at `f3dc3f2db`), base `f16365941` (stack10 HEAD; `wt/stack10` untouched). Status: code, CPU model, driver,
window and item template are done and CPU-checked; **nothing ran on a GPU** (brief: no CUDA run on this host). Every GPU claim below is D or G.

## Commits

| commit | what |
|---|---|
| `f9d0748b6` general | `ggml-cuda: allreduce chunked kernel: overlap the store of round r with the load of round r-1` (only `allreduce.cu`) |
| `af3762817` general fix | `release the round flag from thread 0 after a block barrier, 64-bit tokens` (only `allreduce.cu`; review fixes 1 and 2) |
| `9ea07867c` local | `local: allreduce kernel: block size and round size knobs ...` (cherry-pick of `10fc9f6fd` onto the fix, only `allreduce.cu`) |
| `73d556e7f` codex | driver, window, summary, item template/renderer, report (cherry-pick of `f3dc3f2db`) |
| codex commits after | CPU model rework (review fix 3), window revision list, this report |

`f9d0748b6` + `af3762817` are one change for upstream (squash); they are separate here only because the first form was already reported.

## Source changes (`ggml/src/ggml-cuda/allreduce.cu`)

General commit:
1. Kernel rewritten as rounds of one 16 B vector per thread (`n_rounds = max(1, ceil(count_vec / (blocks*threads)))`). Round r: store
   vectors of round r, then (r > 0) wait for the peer's flag for round r-1 and load/sum/write back round r-1, then publish flag r (below).
   A pass `r == n_rounds` waits for the last flag and loads the last round. The posted D2H stores of round r drain while the H2D loads of
   round r-1 are in flight.
2. Flags: one 64-bit word per (slot, rank, block, round); `GGML_CUDA_AR_MAX_ROUNDS = 128` words (1 KB) per block, arrival ring 32 KB (was 16 KB).
   Tokens are the 64-bit call number (starts at 1; flags are zeroed once). See "Review fixes" for why 64 bit.
3. The wait is `ld.acquire.sys` by thread 0 + `__syncthreads()`; the former `__threadfence_system()` after the barrier is gone for the
   loads. Reason (SASS: `MEMBAR.SC.SYS`): that fence waits for the thread's own posted stores of the current round, which re-serializes
   the two PCIe directions. Publish: see "Review fixes" (barrier, then thread 0: fence + flag store); the trailing fence after the flag store is
   kept only for the last round (an intermediate one would stall thread 0 for a PCIe round trip while its block waits at the next barrier).
4. `host_other` lost `__restrict__`: with `const __restrict__` nvcc emitted `LDG.E.128.CONSTANT` (ld.global.nc), which assumes read-only
   data for the launch; the peer writes it during the launch. (The base kernel has the same latent issue; its fence+`CCTL.IVALL` hid it.)
5. Tail (count % ELEMS_PER_VEC) is written and read in the last round by block 0. `GGML_CUDA_AR_KERNEL_THREADS = 256` (launch constant,
   was a literal). Copy path unchanged. HIP: `__hip_atomic_load(..., ACQUIRE, SYSTEM)`; not compiled here.

Local commit (A/B and tuning, env read at pipeline init): `GGML_CUDA_AR_THREADS` (64..1024; unset = auto: 128 for wire <= 256 KB else 256),
`GGML_CUDA_AR_ROUND_VECS` (default 1; 0 = one round, pipelining off), `GGML_CUDA_AR_MUTANT=1` (load without waiting), `GGML_CUDA_AR_STALL_US`
(rank 1 busy-waits before each round). The block size is only a launch parameter; rounds are merged on the host if a launch would exceed
128 rounds. `THREADS=256 ROUND_VECS=1` is the general commit's schedule (same source, so "gen" below is the local binary at those
settings, not a separate build).

## Identity argument

Each output element is `cast_dst(float(cast_wire(own)) + float(wire_peer))`, a function of that element only; the new kernel runs the same
code per element with the same vector striping (`i = round*gnt + gtid`, same on both GPUs) and the same in-place order per thread (a
thread stores and later writes back the same vector, so sendbuf is never overwritten before it was sent). Only when a vector is stored
and loaded changes. So the output differs from the base only if the peer data is not yet visible when loaded, which the flags exclude:
stores, barrier, thread 0: `MEMBAR.SC.SYS` + flag store | thread 0: acquire load of the flag, barrier, loads. Argued, not proven for the device
memory model: barrier + one releasing/acquiring thread is the pattern of cooperative-groups grid sync and CUTLASS semaphores (gpu scope); at
system scope over PCIe-hosted memory it is an assumption the GPU run must confirm (risk R1). L1 staleness: every line of a round chunk (4 KB aligned per block) is loaded once per launch; the acquire
issues `CCTL.IVALL`; only the tail line can be shared with a later chunk and it is loaded in the last pass (as in the base).

## Review fixes (Codex read-only review, `REVIEW-codex.md`)

1. **Producer release.** Before: every thread `__threadfence_system()`, barrier, thread 0 flag store. A fence in the thread that stored the data is not a release
   for a flag store done by another thread (PTX needs the release fence/`st.release` in the thread that stores the flag; the barrier gives CTA scope only), and
   thread 0's fence preceded the barrier in round 0. Now (`af3762817`): all threads store, block barrier (the load phase's barrier for r > 0, an explicit one for
   r == 0), then thread 0 only: `__threadfence_system()` + volatile flag store. The fence is cumulative over what the barrier made visible to thread 0, which is the
   documented pattern. **The per-thread fences are gone**: none sits between round r+1's stores and round r's loads (the overlap), and 255 of 256 threads no longer wait for
   a drain. SASS (below) shows `MEMBAR.SC.SYS` only inside the thread-0 branch, after `BAR.SYNC`.
   **The base has the same formal gap** (`f16365941:allreduce.cu:152-163`: per-thread fence, barrier, thread 0 stores the flag with no fence of its own, and the
   reader's volatile flag load has no acquire either). It works there because the hardware drains each thread's posted stores at its own fence; upstream finding, not fixed in the base here.
   Cost/risk of the new form (G, not measurable here): thread 0's `MEMBAR.SC.SYS` must also cover the other warps' posted stores, as PTX cumulativity requires. If the
   window's identity/stress stage ever differs from the base, the fallback is the old per-thread fence before the barrier plus thread 0's fence (formally correct and
   practically drained, at the old cost): one-line change.
2. **Token wrap.** The 32-bit design wrote tokens into per-round flags and never reset them; a round that short launches never touch keeps an old token, which equals a
   later call's token after 2^32 calls (days at thousands of calls per second) and would let a reader pass that round's wait early. Now tokens and flags are 64 bit
   (`ld.acquire.sys.b64`, 8 B aligned: 1 KB per block); 2^64 calls are unreachable, so no reset or epoch is needed. Alternatives rejected: a per-call epoch/reset costs
   a host-side memset or a kernel pass per call. CPU model case `stale-wrap` (below) fails on the 32-bit revision and passes on the 64-bit one. The base has one flag per
   block that every call rewrites, so it is not affected.
3. **Slot reuse wording.** `cudaEventSynchronize(ev.ker)` at `acquire_slot` protects reuse of a slot by call k+2 only where `ev.ker` was recorded (the last chunk of an
   outer call). Intermediate chunks rely on the ordered-kernel chain: the launches of one device are stream-ordered; GPU g's kernel of chunk c+1 returns only after the peer's kernel of
   chunk c+1 published, and that kernel starts only after the peer's kernel of chunk c finished (its loads of chunk c done), so g's stores of chunk c+2 into chunk c's slot come after
   the peer's loads of chunk c (D). Not changed by this work; the old "events protect every reuse" wording was too strong.
4. **SASS, all four instantiations** (`<bf16,bf16>`, `<half,half>`, `<float,float>`, `<float,bf16>`; `cuobjdump -sass` of `nvcc` sm_120a output; general kernel at `af3762817` and
   local kernel at `9ea07867c`): peer-data loads are `LDG.E.128` (tail loads plain `LDG.E`/`LDG.E.U16`), **no `.CONSTANT` anywhere** (0 occurrences in every function; the
   `__restrict__` of `ggml_cuda_memcpy_1`, `common.cuh:800`, does not reach the kernel because `host_other` has none). Per function: 2 x `MEMBAR.SC.SYS` (publish and last-round
   trailing), both after the `tid == 0` branch; 2 x `BAR.SYNC` (poll barrier, explicit round-0 barrier); 2 x `LDG.E.64.STRONG.SYS` (first probe and the loop), each followed by `CCTL.IVALL`; 1 x `STG.E.64.STRONG.SYS`
   (flag) after the first `MEMBAR`. No `MEMBAR` between the round's vector stores and the previous round's loads. Registers: general 32, local 38-40, no spills. HIP unbuilt.

## Gates (what exists, what each proves)

* CPU protocol model (`codex/ar-duplex-cpumodel.{cpp,sh}`): the kernel text of any revision runs unchanged on std::threads (2 GPUs x 8 blocks x
  64/128/256 threads), in-place, F32/BF16 wire, tails, 1/2/all-in-one vector rounds. Shims are the kernel's own operations, no stronger: relaxed flag store, seq_cst
  fence, acquire flag load (relaxed for the base), plain data (the former `__ATOMIC_RELEASE` store supplied exactly the edge under review and is gone).
  New cases: `slot-reuse-nojoin` (each GPU runs its own launches in order; the host waits for both GPUs' call k-2 before enqueuing call k, as `acquire_slot` does; nothing
  joins the GPUs, so one GPU runs call k+1 while the other is in call k; sizes alternate 18/12/6 rows, 1 and 7 elements so a slot holds rounds of a larger earlier call) and
  `stale-wrap` (tokens past 2^32, untouched rounds seeded with the token 2^32 calls older; expected to FAIL on 32-bit per-round flags, PASS on 64-bit).
  **What it cannot show:** the GPU memory model. It runs on x86 (TSO) where stores are ordered anyway, so a missing release/acquire edge or fence in the kernel
  (including the base's) is NOT detected; it checks indexing, flag/token logic, slot reuse under skew and the in-place order. Weak-memory behaviour, L1/L2 staleness, posted-write
  draining and PCIe proxy behaviour are covered only by the SASS review (above) and the GPU window.
  **Run here after the rework, CPU only, quick (128 threads; the full 64/128/256 sweep runs in the window; not rerun here):** `f16365941`, `f9d0748b6`, `af3762817`, `9ea07867c`
  all PASS, incl. `slot-reuse-nojoin` (24 calls, both thread counts, 0 wrong). `stale-wrap`: `f9d0748b6` (32-bit) fails as designed (32703 wrong elements, 3/3 calls), `af3762817` and
  `9ea07867c` (64-bit) pass; not applicable to the base. Wait-removed mutant + rank 1 late: 3/3 calls wrong on all four revisions (negative control holds); local `stall+MUTANT` fails, `stall+wait` passes.
  No negative control exists for `slot-reuse-nojoin` itself (it shows no failure on any revision; it is coverage, not proof), and the model was first run here unthrottled by `cpugate`
  (see the note to the coordinator); the reruns above went through `cpugate`.
* `nvcc` compile-only of `allreduce.cu` (sm_120a, project flags, -Wall -Wextra): clean for base, general, local; 32 regs general, 38-40 local, no spills.
  SASS of the general kernel (cuobjdump): poll `LDG.E.STRONG.SYS`+`CCTL.IVALL`, `BAR.SYNC`, plain `LDG.E.128` (no `.CONSTANT`), no `MEMBAR.SC.SYS` between the stores and loads of a round.
* Driver `codex/ar-duplex-driver.cpp` (public comm API only, so it runs on any build): `identity` = per-case FNV hashes of both GPUs' outputs and a CPU
  reference (F32 with BF16 wire, F32 wire via `GGML_CUDA_AR_BF16_THRESHOLD=0`, F16, BF16; tails; 8 row kinds: normal, tiny, denormal (reference skips),
  huge, Inf, NaN, +-0, exact cancellation; inactive shard; with `GGML_CUDA_AR_COPY_THRESHOLD=0` multi-launch sizes up to 3 MB). `perf ARM calls` = per-call
  us at 1/6/12/18/64 rows and 0.5 MB, batch 1 and 8, 240 calls per cell, median/p25/p75. `selftest` (CPU, run here): the checker flags one flipped bit.
* Window `codex/bw-ar-duplex.{toml,sh}` (filler, priority 15, `host_sensitive = false`, est 6000 s / budget 12000 s; shlint clean; preflight untested under the queue):
  40 cpu-model (4 revisions: base, `f9d0748b6`, `af3762817`, HEAD; each quick + mutated; through `cpugate`, so it waits while a measure item runs) | 41 build `llama-server` | 42 build base (`git archive f16365941`, ggml libs) | 43 stamps | 44 driver build + selftest |
  45 identity+mutants | 46 synccheck/racecheck (base is the control; both arms must have the same rc; a common timeout is recorded as void because the sanitizer
  may serialize the two spinning kernels) | 47 perf ABBA | 48 `wt freeze` -> `frozen/ar-duplex` | 49 render + `gpuq submit --check`.
  Identity gate: every new arm (gen, one, loc, t64, t128, rv2) x (default, F32 wire, copy engine off) must `cmp`-equal the base hashes, the base must agree with the CPU
  reference (else rc 6, driver defect), the correct wait under a 100 us stall must equal the base, and the no-wait mutants (gen, one, loc, with stall) must exit 1 with `ar-duplex: FAIL`
  and differ from the base (else rc 5, negative control void). Not covered: `test-backend-ops` has no case that reaches the meta AllReduce (grep: no
  `comm`/`allreduce` in tests/test-backend-ops.cpp), so it is not run; the end-to-end check is the rendered item (identical output with greedy settings is not part of that item).
* Item: `codex/items/ardup-np2-real-j5q.toml.in` + `codex/render_ardup_items.py` (base arm = `frozen/stack10` if its SOURCE exists, else `frozen/stack9-p55`; arm `ard` =
  `frozen/ar-duplex`; plan base ard ard base ard base base ard; depths 8192/90112, np2, J5q). Dry run here with `AR_DUPLEX_ROOT=frozen/stack10-cand2` standing in for the
  not yet existing freeze and the stack9-p55 base: `gpuq submit --check`: `ardup-np2-real-j5q: valid` (not submitted, output removed). The real render happens in stage 49 after the freeze:
  `codex/items/rendered/ardup-np2-real-j5q.gpuq.toml`. With the stack9-p55 base the arms differ by more than the AllReduce: say so when reading it.

## Expected, with falsifiers (D from the brief's model: fixed 4.7 us, W = R = 1.53 us/row, ~140 AR calls per round inferred from the brief; G = per-round overhead ignored)

Pipelined time per call = fixed + 1.53 x rows x (1 + 1/K), K = rounds (gen 256 threads: 2 / 4 / 6 for 6 / 12 / 18 rows; loc 128 threads: 4 / 8 / 12).

| rows | base (M) | gen D | loc D | gen ms/round | loc ms/round |
|---|---|---|---|---|---|
| 6 (np1) | 23.1 | 18.5 | 16.2 | -0.6 | -1.0 |
| 12 (np2) | 41.7 | 27.6 | 25.3 | -2.0 | -2.3 |
| 18 (np3) | 60.5 | 36.9 | 34.6 | -3.3 | -3.6 |

These are upper bounds: every round adds a flag poll round trip and a barrier (G, unmeasured), so K above the optimum can lose. The `loc` rows beat `gen` only if that
overhead is below the saving from the extra rounds; the window measures `t64 / t128 / rv2` for exactly this. 1 row and <= 3 rows (<= 32 KB wire at 256 threads, <= 16 KB at 128) are one round:
unchanged apart from the dropped fence on the read side (D: equal or ~0.1-0.5 us faster).
Falsifiers (perf.tsv, `slope` lines): slope 6->12 rows > 2.6 us/row for gen or loc (brief: the serialization is elsewhere, e.g. host-memory ordering or flag polling);
`one` (pipelining off, same fences) as fast as `gen` (the gain is not the overlap); any cell with median > base +2 % (`slower_than_base` lines) blocks default-on of that form.

## Which setups gain (D)

* Gain: two-GPU boxes without P2P/NVLink that stage through pinned host memory, i.e. the path where `ggml_cuda_ar_pipeline_init` succeeds (n_devices == 2, cc >= 7.0), for reductions
  between one round (32 KB wire at 256 threads) and the 1 MB copy threshold, if the link carries D2H and H2D concurrently. Gain ~ min(W, R) x (K-1)/K minus K x per-round overhead.
  Slower or asymmetric links (Gen3 x8, this box) gain more in absolute us; on a fast link (Gen4 x16) data time shrinks to the same order as the per-round overhead and the gain shrinks with it.
* Unchanged: payloads <= 1 round; the copy path (>= 1 MB, package 59); NCCL boxes (NVLink or NCCL on); any GPU count other than 2 (init returns null, the meta butterfly runs: this path does not
  support it, so "every supported GPU count" is 2); MUSA (no pipeline).
* General commit has no switch and no env var; the tuned form (128 threads below 256 KB, G) and all switches live only in `10fc9f6fd`.

## Risks and open points

R1 acquire + barrier on the read side and barrier + thread-0 release on the write side instead of per-thread system fences (above; fallback named in Review fixes 1). R2 the overlap may not materialize if the root complex or the GPU serializes posted writes behind reads: the falsifier covers it.
R3 `compute-sanitizer` may serialize the two spinning kernels (stage 46 records a void then). R4 the base build in stage 42 is a cold full ggml-cuda build if ccache misses (path-dependent `__FILE__`): the budget
assumes it. R5 HIP path unbuilt. R6 the mutant relies on `STALL_US` (assumes ~2 GHz: 100 us = 200000 cycles, ~80 us at 2.5 GHz).
