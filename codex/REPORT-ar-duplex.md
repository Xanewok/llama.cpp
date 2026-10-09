# ar-duplex: chunked AllReduce kernel with the store of round r overlapped with the load of round r-1

Tree `wt/ar-duplex`, branch `topic/ar-duplex`, base `f16365941` (stack10 HEAD; `wt/stack10` untouched). Status: code, CPU model, driver,
window and item template are done and CPU-checked; **nothing ran on a GPU** (brief: no CUDA run on this host). Every GPU claim below is D or G.

## Commits

| commit | what |
|---|---|
| `f9d0748b6` general | `ggml-cuda: allreduce chunked kernel: overlap the store of round r with the load of round r-1` (only `allreduce.cu`) |
| `10fc9f6fd` local | `local: allreduce kernel: block size and round size knobs ...` (only `allreduce.cu`) |
| codex commit | driver, CPU model, window, summary, item template/renderer, this report |

## Source changes (`ggml/src/ggml-cuda/allreduce.cu`)

General commit:
1. Kernel rewritten as rounds of one 16 B vector per thread (`n_rounds = max(1, ceil(count_vec / (blocks*threads)))`). Round r: store
   vectors of round r, then (r > 0) wait for the peer's flag for round r-1 and load/sum/write back round r-1, then fence and publish flag r.
   A pass `r == n_rounds` waits for the last flag and loads the last round. The posted D2H stores of round r drain while the H2D loads of
   round r-1 are in flight.
2. Flags: one int per (slot, rank, block, round); `GGML_CUDA_AR_MAX_ROUNDS = 128` ints (512 B) per block, arrival ring 16 KB. Token
   equality as before (tokens are strictly increasing, so old flags never match).
3. The wait is `ld.acquire.sys` by thread 0 + `__syncthreads()`; the former `__threadfence_system()` after the barrier is gone for the
   loads. Reason (SASS: `MEMBAR.SC.SYS`): that fence waits for the thread's own posted stores of the current round, which re-serializes
   the two PCIe directions. The publish side keeps the per-thread fence before the barrier; the trailing fence after the flag store is
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
store, per-thread `MEMBAR.SC.SYS`, barrier, flag store | acquire load of the flag, barrier, load. Argued, not proven for the device
memory model: the barrier-after-acquire pattern is the one CUTLASS semaphores use at gpu scope; at system scope it is an assumption the
GPU run must confirm (risk R1). L1 staleness: every line of a round chunk (4 KB aligned per block) is loaded once per launch; the acquire
issues `CCTL.IVALL`; only the tail line can be shared with a later chunk and it is loaded in the last pass (as in the base).

## Gates (what exists, what each proves)

* CPU protocol model (`codex/ar-duplex-cpumodel.{cpp,sh}`): the kernel text of any revision runs unchanged on std::threads (2 GPUs x 8 blocks x
  64/128/256 threads), 4 calls per case with rotating slots/tokens, in-place, F32/BF16 wire, tails, 1/2/all-in-one vector rounds. **Run here,
  CPU only:** `f16365941`, `f9d0748b6`, `10fc9f6fd` all PASS (quick: 128 threads; local full run also PASS: 64/128/256 threads, every size).
  Negative controls, all really fail: the wait loop compiled out + rank 1 late -> 3/3 calls wrong on every revision (90k-184k wrong elements);
  local `MUTANT=1` + stall -> 3/3 calls wrong; same stall with the wait -> 0 wrong. It does not model the device memory model.
* `nvcc` compile-only of `allreduce.cu` (sm_120a, project flags, -Wall -Wextra): clean for base, general, local; 30 regs general, 40 local, no spills.
  SASS of the general kernel (cuobjdump): poll `LDG.E.STRONG.SYS`+`CCTL.IVALL`, `BAR.SYNC`, plain `LDG.E.128` (no `.CONSTANT`), no `MEMBAR.SC.SYS` between the stores and loads of a round.
* Driver `codex/ar-duplex-driver.cpp` (public comm API only, so it runs on any build): `identity` = per-case FNV hashes of both GPUs' outputs and a CPU
  reference (F32 with BF16 wire, F32 wire via `GGML_CUDA_AR_BF16_THRESHOLD=0`, F16, BF16; tails; 8 row kinds: normal, tiny, denormal (reference skips),
  huge, Inf, NaN, +-0, exact cancellation; inactive shard; with `GGML_CUDA_AR_COPY_THRESHOLD=0` multi-launch sizes up to 3 MB). `perf ARM calls` = per-call
  us at 1/6/12/18/64 rows and 0.5 MB, batch 1 and 8, 240 calls per cell, median/p25/p75. `selftest` (CPU, run here): the checker flags one flipped bit.
* Window `codex/bw-ar-duplex.{toml,sh}` (filler, priority 15, `host_sensitive = false`, est 6000 s / budget 12000 s; shlint clean; preflight untested under the queue):
  40 cpu-model (3 revisions x quick+mutated) | 41 build `llama-server` | 42 build base (`git archive f16365941`, ggml libs) | 43 stamps | 44 driver build + selftest |
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

R1 acquire + barrier instead of a system fence (above). R2 the overlap may not materialize if the root complex or the GPU serializes posted writes behind reads: the falsifier covers it.
R3 `compute-sanitizer` may serialize the two spinning kernels (stage 46 records a void then). R4 the base build in stage 42 is a cold full ggml-cuda build if ccache misses (path-dependent `__FILE__`): the budget
assumes it. R5 HIP path unbuilt. R6 the mutant relies on `STALL_US` (assumes ~2 GHz: 100 us = 200000 cycles, ~80 us at 2.5 GHz).
