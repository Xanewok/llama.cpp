# FA q8_0 scope validation (private notes)

## Conclusion

Turing staging, two-query MMA batches, and GQA tiles with `ncols2=4/2/1` are implemented as separate commits. No sampling code changed. No GPU program was executed. Parent GPU validation is required before submission: CPU evaluation and compilation cannot prove the new staging synchronization, token identity, or performance.

M = measured; D = derived from source or measured values; G = guess. No GPU speed estimate is given. CUDA compiler: M: CUDA 13.2.86. References to architecture, shape, flags and commit identifiers identify configurations, not performance measurements.

## Commit stack

Base: `869034b4b`, followed by the supplied kernel `6935009fc` and tests `b66f25e8d`. New commits:

- `6a9cbe0d9 ggml-cuda : stage q8_0 FA tiles synchronously on Turing`. D: 3 files changed, 43 insertions(+), 16 deletions(-).
- `554e0cbd4 ggml-cuda : dequantize q8_0 FA tiles for two-query MMA batches`. D: 5 files changed, 19 insertions(+), 4 deletions(-).
- `a098169a5 ggml-cuda : support q8_0 FA MMA with smaller GQA tiles`. D: 10 files changed, 78 insertions(+), 22 deletions(-).
- `7a8525093 tests : cover all small q8_0 FA batches with KV streams`. D: 1 file changed, 10 insertions(+).
- `b5a3a6a7a ggml-cuda : exclude q8_0 MMA instances from HIP and MUSA`. D: 1 file changed, 7 insertions(+).
- `8f0e21aa4 ggml-cuda : align packed q8_0 FA dequantization stores`. D: 1 file changed, 1 insertion(+), 1 deletion(-).
- `c332c2733 tests : benchmark q8_0 FA with ungrouped queries`. D: 1 file changed, 1 insertion(+), 1 deletion(-).
- `4ab70fe9d tests : retain existing GQA 6 FA performance cases`. D: 1 file changed, 10 insertions(+).

The Turing, two-query and smaller-GQA commits each carry cases in the existing `tests/test-backend-ops.cpp`. The stream-coverage, ungrouped-perf and restored original GQA 6 cases are independently removable. No test file or production experiment knob was added.

The supplied `.git` file and its external Git directory are read-only. Commits reside on `topic/fattn-mma-quant-kv-scope` in `codex/git`, with a read-only object alternate. Original shared refs were not changed. Use `GIT_DIR="$PWD/codex/git" GIT_WORK_TREE="$PWD"` to inspect this stack. The local bundle is `codex/fa-scope.bundle`; importing it is a parent action.

## Design and correctness

- Turing: the raw row loader uses synchronous aligned copies and the existing single-stage MMA loop. A block barrier separates raw loads from dequantization; the existing tile barrier separates conversion from MMA consumption. Turing uses `ncols1=4,ncols2=8` for wider query batches because its MMA kernel excludes more than D: 32 total columns. Forcing the existing wide instance would execute `NO_DEVICE_CODE`.
- Two queries: add `(2,8)` and lower the raw gate to D: `nb=2..8`. Dispatch remains behind `ggml_cuda_get_best_fattn_kernel`; vector selection is unchanged. This preserves the f16 tile geometry for two-query MMA.
- Smaller GQA: add `(8,1),(4,2),(8,2),(2,4),(4,4),(8,4)`. The selector preserves the existing f16 tile sizes. `ncols2=1` disables the async pipeline even on Ampere+, so these raw copies explicitly use the synchronous loader. Whole raw rows require padded KV; the gate requires `K->ne[1] % FATTN_KQ_STRIDE == 0` and aligned K/V base pointers and strides. Sparse and softcap cases retain conversion.
- Packing alignment: `half2 v[4]` feeds `ggml_cuda_memcpy_1<16>`, which reads it as `int4`. Explicit `__align__(16)` upholds that source alignment. This is a separate fix; final raw instruction identity is checked against the pre-fix objects.
- HIP/MUSA: q8_0 explicit instantiation, declarations, and the instance predicate are now excluded by preprocessor guards. NVIDIA preprocessing tokens are unchanged by that guard-only commit. Hardware intrinsics in both raw helpers remain inside `TURING_MMA_AVAILABLE`, which requires neither HIP nor MUSA.

The scalar contract is the same `__hmul2` operation with the same half-scale bits and exactly represented signed-byte operands as `convert.cu`. M: the CPU model verifies 16,252,928 finite-scale/sign-byte products and 256 packed pairs. Changing the reconstruction bias from the correct D: 1152 to D: 1151 makes the negative control differ. Non-finite payloads and device memory ordering are not established by the NumPy test.

M: all 54 raw q8_0 object SASS bodies, including encodings, match before and after explicit packing alignment (`alignment-identity.json`).

Raw and convert paths can still choose different stream-K partitions because resource use changes occupancy. Exact dequantization does not establish bit-identical whole attention outputs or tokens. No equivalent sampling-law claim is substituted for the parent identity check.

## Kernel selection before and after (D)

Assumptions: head 256, both K/V q8_0, an available matching compiled architecture, dense padded KV and compatible aligned layout. `MMA + convert` is the upstream fallback; `MMA + raw` skips whole-view conversion.

| cc | nb=1, before / after | nb=2, supplied branch / after | nb=3..8, supplied branch / after |
|---|---|---|---|
| sm_75 | VEC / VEC | MMA + convert / MMA + raw | MMA + convert / MMA + raw |
| sm_80, sm_86 | VEC / VEC | MMA + convert / MMA + raw | raw for GQA>4, otherwise convert / MMA + raw |
| sm_89, sm_90, sm_120 | VEC / VEC | VEC / VEC | raw for GQA>4, otherwise convert / MMA + raw |

On master all MMA entries convert. If KV is not padded, `can_use_vector_kernel` is false: D: nb=1 and nb=2 select MMA on all listed architectures, but retain conversion because the raw whole-row loader has no tail-row handling. Thus no one-query raw extension is applicable here. F16 KV selection is unchanged.

D: tile sizes selected by the raw dispatcher:

| ncols2 | nb=2 | nb=3..4 | nb=5..8 |
|---|---|---|---|
| 8 | ncols1=2 | ncols1=4 | ncols1=8; Turing uses 4 |
| 4 | ncols1=2 | ncols1=4 | ncols1=8 |
| 2 | ncols1=4 | ncols1=4 | ncols1=8 |
| 1 | ncols1=8 | ncols1=8 | ncols1=8 |

The two-query-only `(2,8)` and `(2,4)` instances remain compiled on Ada+, although its padded quantized two-query selection is VEC. Removing them there would duplicate selector assumptions; that compiler-cost reduction was not implemented.

## Compiler resources (M)

Each cell is registers / stack bytes / spill-store bytes / spill-load bytes. These are the dense, no-softcap raw main kernels; fixup kernels and all emitted commands are retained in the JSON/log evidence. `sm_120` is normalized by CMake to `sm_120a`.

| D: instance (ncols1,ncols2) | M: sm_75 | M: sm_80 | M: sm_86 | M: sm_89 | M: sm_90 | M: sm_120a |
|---|---|---|---|---|---|---|
| (8,1) | 235/16/0/0 | 226/16/0/0 | 226/16/0/0 | 226/16/0/0 | 239/16/0/0 | 253/16/0/0 |
| (4,2) | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 |
| (8,2) | 255/24/4/4 | 218/16/0/0 | 220/16/0/0 | 220/16/0/0 | 220/16/0/0 | 242/16/0/0 |
| (2,4) | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/24/4/4 |
| (4,4) | 255/24/4/4 | 220/16/0/0 | 222/16/0/0 | 222/16/0/0 | 222/16/0/0 | 242/16/0/0 |
| (8,4) | 255/248/484/864 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/24/4/4 | 255/40/40/28 |
| (2,8) | 255/16/0/0 | 220/16/0/0 | 222/16/0/0 | 222/16/0/0 | 222/16/0/0 | 242/16/0/0 |
| (4,8) | 255/240/460/840 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 255/24/4/4 | 255/40/44/32 |
| (8,8) | 24/40/0/0 | 255/88/88/192 | 255/88/84/184 | 255/88/84/184 | 255/80/140/140 | 255/128/296/296 |

`(8,8)` on sm_75 is an inactive diagnostic kernel, not a runnable wide tile. Existing instances are `(4,8)` and `(8,8)`; the other tuples are new. Turing `(4,8)` and `(8,4)` have substantial spills. This is a performance risk, not a measured GPU slowdown.

F16 controls, identical ptxas stats on master and scope (M), same cell format:

| D: instance (ncols1,ncols2) | M: sm_75 | M: sm_80 | M: sm_86 | M: sm_89 | M: sm_90 | M: sm_120a |
|---|---|---|---|---|---|---|
| (8,1) | 244/32/0/0 | 244/32/0/0 | 246/32/0/0 | 246/32/0/0 | 254/32/0/0 | 210/32/0/0 |
| (4,2) | 232/16/0/0 | 188/16/0/0 | 188/16/0/0 | 188/16/0/0 | 231/16/0/0 | 226/16/0/0 |
| (8,2) | 255/16/0/0 | 157/16/0/0 | 159/16/0/0 | 159/16/0/0 | 206/16/0/0 | 205/16/0/0 |
| (2,4) | 223/16/0/0 | 186/16/0/0 | 190/16/0/0 | 190/16/0/0 | 231/16/0/0 | 226/16/0/0 |
| (4,4) | 255/16/0/0 | 160/16/0/0 | 162/16/0/0 | 162/16/0/0 | 208/16/0/0 | 205/16/0/0 |
| (8,4) | 255/144/224/240 | 227/16/0/0 | 231/16/0/0 | 231/16/0/0 | 254/16/0/0 | 254/16/0/0 |
| (2,8) | 228/16/0/0 | 160/16/0/0 | 162/16/0/0 | 162/16/0/0 | 208/16/0/0 | 205/16/0/0 |
| (4,8) | 255/128/240/284 | 227/16/0/0 | 231/16/0/0 | 231/16/0/0 | 254/16/0/0 | 252/16/0/0 |
| (8,8) | 24/40/0/0 | 255/16/0/0 | 255/16/0/0 | 255/16/0/0 | 254/16/0/0 | 254/56/40/40 |

## CPU and compile evidence

- M: final CPU FLASH_ATTN_EXT evaluation: 5,407/5,407 passed, 358.44 s elapsed, two backend/quantization workers. Earlier scope evaluation: M: 5,398/5,398 passed, 194.40 s, four workers. The difference includes thread counts and concurrent host load; it is not a speed comparison.
- M: final architecture sweep compiles dispatch and all raw instances for sm_75/80/86/89/90/120a with `-Werror=all-warnings`. The first sweep hit transient filesystem exhaustion; every failed job was retried, and the final sweep passed.
- M: paired f16 control compilation: 108 translation units (54 pairs). All main-kernel resource statistics match. Active instruction sequences and encodings match; the inactive Turing `(8,8)` diagnostic embeds a different `NO_DEVICE_CODE` source line.
- M: single-arch full-library comparison: 6,513 common kernel resource records match; 174 active f16 FA SASS bodies match including encodings. D: 82 inactive diagnostic kernels are excluded from instruction identity because their source-line constants change. Template name changes and NVCC build-path hashes are normalized; instructions are not normalized. M: the portable default list also has 26,052 matching common resource records and 696 matching active f16 SASS bodies (`default-identity-default.json`).
- M: generated q8_0 source files match the generator for all nine tuples. `hipcc` and `mcc` are absent; no HIP/MUSA syntax compilation is claimed. `cpp -P` tests remove the entire q8_0 instance/declaration/predicate block under each backend macro and preserve NVIDIA tokens.

A build-local interposer caps `std::thread::hardware_concurrency()`; shipped test code is unchanged by this cap. Every executed shell command used `CUDA_VISIBLE_DEVICES=""` and priority 19. Builds used at most eight compiler workers, with nvcc `--threads=1` in cost builds. CUDA runtime binaries were not run; cuobjdump only read object/library files.

## Build and per-change costs

Clean library builds use Release, NCCL off, ccache off, `GGML_NATIVE=OFF`, and six jobs. The portable default CUDA architecture list is D: `75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real`. Explicit `120` avoids native GPU probing. `GGML_CUDA_FA_ALL_QUANTS=ON` is deprecated upstream but was used as requested.

| Configuration | M: master seconds / bytes | M: scope seconds / bytes | D: seconds delta / byte delta |
|---|---|---|---|
| single | 580.71 / 70,931,768 | 618.32 / 72,225,768 | +37.61 / +1,294,000 |
| default | 1827.34 / 162,075,328 | 1731.88 / 165,659,136 | -95.47 / +3,583,808 |
| all-quants | 2156.34 / 219,645,280 | 2214.09 / 223,327,400 | +57.75 / +3,682,120 |

Clean-build wall time is M under concurrent host load. The observed default-list scope build was faster than master, which demonstrates that these one-shot wall deltas are not a causal compiler-cost estimate. Use the instance counts and touched-unit audit to assess the growth raised in #28634/#7527.

The per-stage audit compiles `fattn.cu` plus that stage's raw instances with identical flags and relinks against identical common scope objects. It measures the touched-unit work and actual relinked library-size growth; it does not claim a separate clean-build wall time for every intermediate commit. F16 common device instructions are unchanged. Stage source paths have equal-length names. The final stage size matches the corresponding full scope build.

D: per-change relinked library byte growth; measured stage sizes are in `commit-costs.json`.

| Change | single | default | all-quants |
|---|---:|---:|---:|
| supplied kernel | +330,080 | +715,272 | +809,480 |
| Turing | +0 | +65,536 | +65,536 |
| two queries | +121,912 | +379,936 | +375,848 |
| smaller GQA | +842,008 | +2,423,064 | +2,431,256 |

D: summed measured touched-unit compiler seconds per stage; serial one-worker audit, excluding link time. This is not clean-build wall time.

| Stage | single | default | all-quants |
|---|---:|---:|---:|
| initial | 33.51 | 152.61 | 136.78 |
| turing | 33.51 | 154.12 | 139.70 |
| rows | 39.58 | 178.57 | 171.86 |
| gqa | 77.57 | 338.74 | 312.03 |

D: Turing adds no translation unit; two-query support adds one; smaller GQA adds six. The test commits add no CUDA translation unit. Backend exclusion and explicit packing alignment add no NVIDIA instruction/instance cost; M: NVIDIA tokens and all raw SASS respectively match.

M: per-instance standalone compilation seconds / object bytes in the final stage:

| D: (ncols1,ncols2) | single | default | all-quants |
|---|---|---|---|
| (8,1) | 6.91 / 193,248 | 28.33 / 458,520 | 26.02 / 458,520 |
| (4,2) | 6.91 / 202,144 | 27.66 / 475,072 | 25.44 / 475,072 |
| (8,2) | 5.75 / 177,776 | 24.06 / 431,040 | 22.44 / 431,040 |
| (2,4) | 6.90 / 202,080 | 27.75 / 473,888 | 25.41 / 473,888 |
| (4,4) | 5.88 / 177,552 | 23.88 / 431,264 | 22.22 / 431,264 |
| (8,4) | 6.09 / 193,720 | 26.59 / 473,424 | 24.87 / 473,424 |
| (2,8) | 5.89 / 177,264 | 23.91 / 429,408 | 22.11 / 429,408 |
| (4,8) | 6.27 / 192,728 | 26.90 / 471,280 | 24.54 / 471,280 |
| (8,8) | 6.73 / 207,736 | 27.66 / 442,576 | 25.60 / 442,576 |

Object sizes include host glue and compressed device images; they are not additive library deltas. The backend/alignment/test follow-ups have the same NVIDIA code as the measured GQA stage. Final relink sizes equal all full scope library sizes.

## Parent GPU scripts (not executed)

`gpu_scope.sh` and `arch_bench.sh` default to a CPU-only selftest. `--run` is explicit. Both build frozen master/scope binaries with an identical final test matrix, freeze executables and DSOs by SHA-256, hash the model once and check its inode/size/mtime at every boot, and run CUDA0 eval, memcheck, and interleaved `up scope scope up` kernel processes. The local script additionally serves the public base model at depths D: 8192 and 32768, with seed D: 12345 and temperatures D: 0 and 0.8, in the same boot order. It compares token IDs, content, and finish status and records M: tok/s. It requires actual MTP draft tokens and retries a capacity failure using both GPUs. The executable is renamed `fa-scope-srv` before execution.

D: the perf matrix has 320 matched cases per boot: nb=1..8, KV=512/2048/8192/32768/98304, GQA=1/2/4/8, and both q8_0 and f16 controls. GQA=1 is added because the newly synchronous `(8,1)` raw path reserves D: 51,328 shared bytes versus D: 33,920 for f16 on Ampere+. Any latency loss above D: 2% is flagged and causes a failing exit after results are saved. Two boots per arm screen regressions; they do not establish statistical confidence.

Local guard: readable kernel log since `2026-10-01 13:29:00`, no Xid/fallen-off record, no compute PID on either selected GPU, free port, and idle memory. Portable guard checks the current boot's kernel log; when unavailable, it requires healthy nvidia-smi recovery state for every selected GPU. That fallback cannot establish historical Xid absence. The total budget is at most D: 3600 s, including builds, with cleanup time reserved. Server cleanup stops only the captured PID; bounded commands clean up their own captured process groups.

CPU selftests include wrong-output, empty-output, incomplete-perf, over-threshold, Xid, busy-GPU and partial-recovery negative controls; guard tests use mock inputs. GPU execution, sanitizer behavior, public-model MTP availability, and runtime identity remain unverified.

Parent commands:

```sh
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/gpu_scope.sh --run --gpus 0,1
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/arch_bench.sh --run --arch 75 --gpus 0
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/arch_bench.sh --run --arch 86 --gpus 0
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/arch_bench.sh --run --arch 89 --gpus 0
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/arch_bench.sh --run --arch 80 --gpus 0
```

The portable worktree must contain the topic commits and master prerequisite, or the packaged source archives. No download or interactive step is performed by the scripts. Every run saves `summary-sm<arch>.json`; incomplete work fails and remains marked incomplete.

## Remaining risks and scope decisions

- GPU eval/memcheck/perf and model identity are pending for every extension. In particular, Turing spills and the GQA=1 shared-memory increase need measurements. Occupancy can change reduction order, so token identity cannot be proved from scalar packing alone.
- Memcheck does not establish freedom from shared-memory races. Before submission, run racecheck/synccheck on the small new cases as an additional parent check. The static barriers were reviewed, but not exercised on a GPU here.
- F16 device instructions/resources match; its head-256 host dispatch still performs the eligibility checks. No host or default-path runtime no-regression claim is made.
- Global f16 scratch reservation remains: D: zero allocated-device-byte savings are claimed. This skips conversion work, not allocation.
- HIP/MUSA preprocessing is verified, but full toolchain integration needs external CI. Historical Xids are unobservable in the portable nvidia-smi fallback.
- Optional head 128 is not trivial: D: its q8_0 row is 136 bytes, not a multiple of 16. It needs different global transfer alignment/tail handling, a staging layout, and lane/store validation. Head 512 has aligned rows but the existing f16 kernel uses partial head tiles; the whole-row dequantizer would need column-range support and shared-memory analysis.
- q4_0 needs its own nibble reconstruction and half-rounding audit plus new instances. Mixed K/V types need independent K/V template types and stage paths. Neither is implemented.
- The parent supplied the documented need/reviewer references. Current issue/PR duplication was not checked because network was forbidden. Public text, submission, and maintainer discussions belong to the human.

## Evidence and reproduction

- Source: `ggml/src/ggml-cuda/fattn.cu:155` (gate), `:196` (tile selection), `:590` (MMA/VEC selection); `fattn-mma-f16.cuh:502` (raw loading), `:541` (rounding), `:580` (packing alignment), `:780` and `:1149` (synchronous stages), `:1404` (whole-row invariant), `:2205` (shared bytes), `:2295` and `:2353` (backend guards); `common.cuh:288` (TURING availability) and `:822` (16-byte copy). Tests: `tests/test-backend-ops.cpp:11172` and the `make_test_cases_perf` KV loops; generator: `generate_cu_files.py:113`.
- CPU: `cpu-fa-final.log`, `cpu-fa-final.time`, `cpu-build-matrix.log`, `dequant-scope.log`.
- Detailed command/resource manifests are bundled in `compile-evidence.tar.gz`; the expanded JSON files remain beside it.
- Compiler: `scope-arches.json`, `scope-arches-final.log`, all `compile-*.log`, `f16-controls.json`, `f16-controls-analysis.log`, `default-identity-single.json`, `backend-guards.log`, `compiler-version.txt`, `alignment-identity.json`, `default-identity-default.json`.
- Costs: `build-costs.json`, `commit-costs.json`, `cost-*.log`, `stage-*.log`; commands and working directories are in the JSON records.
- Parent harness: `gpu-scope-dry.log`, `gpu-ab-dry.log`, `arch-bench-dry.log`, and `scope_gpu.py` selftests.

CPU reproduction:

```sh
CUDA_VISIBLE_DEVICES="" nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
CUDA_VISIBLE_DEVICES="" nice -n 19 cmake --build build-cpu --target test-backend-ops -j8
CUDA_VISIBLE_DEVICES="" nice -n 19 c++ -shared -fPIC -x c++ - -o build-cpu/cpu-thread-cap2.so <<'CPP'
#include <thread>
unsigned int std::thread::hardware_concurrency() noexcept { return 4; }
CPP
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LD_PRELOAD="$PWD/build-cpu/cpu-thread-cap2.so" nice -n 19 build-cpu/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT -j 1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 codex/dequant_audit.py
CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp" ARCH_CHECK_JOBS=1 nice -n 19 python3 codex/compile_arches.py
CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp" nice -n 19 python3 codex/compile_controls.py
CUDA_VISIBLE_DEVICES="" nice -n 19 python3 codex/default_identity.py single
```

`build_costs.py` requires fresh `build-cost-*` directories and the archived master tree. It intentionally refuses to overwrite a cold-build directory. `commit_costs.py` can recreate the bounded touched-unit/relink audit from local Git objects; no network is enabled.
