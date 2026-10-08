# K9: PDL for MMQ

Tree `wt/s10-k9`, branch `topic/s10-k9`, base `0cfc0723e` (stack10-cand). Labels: M measured, D derived, G guess.
Nothing CUDA was compiled or run: no CUDA on this host. Mac CPU results are in the section "Mac results".

## Commits

| Commit | Subject | Class |
|---|---|---|
| `4f439a86f` | cuda : launch the MMQ kernel and its stream-k fixup with PDL | general |
| `62741de4e` | cuda : launch the MMQ q8_1 quantize kernel with PDL | general, separate engagement check |
| `ad462a292` | tests : add an MMQ producer/consumer chain with byte hashes and replay checks | general test |
| `e970f0f5b` | local: MMQ PDL off switch, wait-removal mutants and engagement log | A/B switch, mutants, log |
| `d8fadf519` | codex : K9 Mac gates and CUDA window | window, payloads |

(Hashes as of the first report; rebases for fixes are listed under Mac results.)

## Design (general commits, no new env var)

- Every block of `mul_mat_q`, `mul_mat_q_stream_k_fixup` and `quantize_mmq_q8_1` calls `ggml_cuda_pdl_lc()` then `ggml_cuda_pdl_sync()` first, before any early return. The fixup kernel has early returns before its first read; `quantize_mmq_q8_1` had its wait after `if (i0 >= ne0) return;` and was a plain launch with no trigger.
- Kernel parameters are plain `*_ptr`; `GGML_CUDA_RESTRICT` locals follow the wait (empty on Hopper+ with PDL), as upstream #24030 does for the other PDL kernels. The inlined device helpers keep `__restrict__` parameters, as upstream leaves `vecdotq.cuh`. Residual risk: the compiler hoisting a load above the wait; the window byte-identity gates and mutants are the check.
- MMQ tiling, stream-k and fixup launches and the six `quantize_mmq_q8_1` launches use `ggml_cuda_kernel_launch`, which falls back to `<<<>>>` when PDL is off, on HIP/MUSA, on old toolkits.
- Trigger at kernel entry affects overlap only; the dependent's wait still blocks until the producer completes and flushes.
- Exact (D): arithmetic, partition, fixup order and stores unchanged; only launch API, restrict placement and wait/trigger position change.
- Dependent audit (scripted over 87 `ggml_cuda_kernel_launch` sites, 40 kernels): every PDL kernel reaches `ggml_cuda_pdl_sync`; none would now race.

## Every source change

- `ggml/src/ggml-cuda/mmq.cuh`, `ggml/src/ggml-cuda/quantize.cu` (FP4 quantize kernels untouched), `tests/test-backend-ops.cpp` (`mmq-chain DEVICE hash|perf [ITERS] [quick] [corrupt]`, `#include <chrono>`).
- Local commit only: `ggml/src/ggml-cuda/common.cuh`, extra `pdl_mutant` kernel parameter, `__nanosleep` loops, `GGML_CUDA_MMQ_PDL`, `GGML_CUDA_MMQ_PDL_MUTANT`, `GGML_CUDA_MMQ_PDL_LOG`.

## Test and window gates

`mmq-chain`: `rms_norm*w -> quantized MUL_MAT -> rms_norm*w -> quantized MUL_MAT -> add(x)` for IQ4_XS/Q4_K/Q6_K/Q8_0, k=5120, m1 in {3072,6144,8704}, n in {12,17,24} (36 cases), 200 recomputes with alternating inputs through CUDA graph replay; `hash` exits 1 if any recompute differs from the first with the same input.

Window `codex/bw-k9.toml` / `codex/bw-k9.sh`: filler, priority 18, non-preemptible, host-insensitive, est 5400 s (G), budget 12000 s (D); baseline = `git archive 0cfc0723e` + this tree's test file, built in `/mnt/scratch/s10-k9/base`. Gates on CUDA0 and CUDA1:

- engagement: log lines `kernel=mmq`, `quantize`, `fixup` all `pdl=1`, no `pdl=0`;
- K9 stdout byte-identical to the stock-MMQ baseline binary, to `GGML_CUDA_MMQ_PDL=0`, `GGML_CUDA_PDL=0`, `GGML_CUDA_DISABLE_GRAPHS=1`;
- slow-producer stress `MUTANT=24` (waits intact): unchanged output;
- negative controls (rc 1 + `mmq-chain: FAIL`, else rc 5 void): `MUTANT=9` (MMQ skips wait, quantize sleeps), `MUTANT=18` (fixup skips wait, MMQ sleeps);
- full MUL_MAT / MUL_MAT_ID on CUDA0, np2 subsets on CUDA1;
- perf: base, k9, off (`GGML_CUDA_MMQ_PDL=0`), base; `codex/k9_summary.py` -> `perf.tsv` with A/A spread and deltas.

No mutant for the wait in `quantize_mmq_q8_1` (its producer, the fused rms_norm, cannot be slowed without editing another kernel); covered by byte identity only.

## Expected effect and falsifier

X1 (M): PDL saves 0.27 ms/round at np1 and 0.20 at np2, MMQ not included. Brief budget 0.05-0.1 ms/round (G). Mine (G): 0.05-0.15 ms/round at np2, flat 8k to 90k. Chain microbench (G): k9-base about -2 to -6 us per graph.
Falsifiers: (1) k9-base inside the A/A spread or positive on a 12/24-column shape (off-base separates lost overlap from codegen cost); (2) np2 J5q real-prompt A/B with |delta| below the ~0.05 ms boot spread; (3) any byte difference or recompute mismatch in the correct-wait arms.

## Uncertainty

- CUDA code and the CUDA paths of the test are uncompiled. Likely first failures: template/argument mismatch at `ggml_cuda_kernel_launch` for `mul_mat_q`, `-Werror` on unused statics in the local `common.cuh` block.
- Restrict removal could cost MMQ throughput on sm_120; base-off in the perf table shows launch-only effects, so a base vs k9-with-PDL-off gap indicates codegen cost.
- Hoisting across the wait is detected only if a hoisted load lands in the race window; an SASS check (no `LDG.*CONSTANT` before the grid-dependency instruction) is a useful extra I did not script.

## Mac results

Run `bw-mac ... codex/bw-k9-mac.sh k9` on `mini-builder` at HEAD `72ab2e59b` (out `work/build-window-1007/k9-mac-a2`, sealed in `codex/k9-evidence/summary.json`), CPU build `-DGGML_NATIVE=OFF`, ccache, -j4:

| Stage | Wall s (M) | Result |
|---|---:|---|
| 20 CPU build of `test-backend-ops` | 38 | PASS (the new `mmq-chain` code compiles) |
| 21 chain harness on the CPU backend | 8 | PASS: two runs byte-identical, 2 cases with distinct hashes for the two inputs, `0 of 8` recompute mismatches, `corrupt` run exits 1 with `mmq-chain: FAIL 2 recomputes differ`, perf mode prints 2 rows |
| 31 bash -n, py_compile, shlint | 0 | PASS |

An earlier attempt (`k9-mac-a1`, 554 s) failed at stage 21 because the Mac model gate prints `WAIT` lines on stdout while other workers run, which polluted the hash files; the payload now keeps only `chain`/`perf` lines (`72ab2e59b`). No source file needed a fix. These gates validate the harness only; PDL, MMQ and the CUDA edits remain uncompiled until the window.
