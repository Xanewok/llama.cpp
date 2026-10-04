# mmvq6

The compile-tested candidate is `94a537a1a`: four warps and two rows per block for Q5_K, Q6_K, Q4_K and IQ4_XS at 5..8 columns on sm_120. It is a candidate for GPU measurement, not a demonstrated speed improvement. No GPU program, GPU query or network request ran here.

Labels: M observed/measured, D calculated, G hypothesis/estimate. Commit IDs, architecture names, tensor dimensions, flags, tolerances and configured limits are identifiers or configuration, not performance measurements.

## Commits and frozen inputs

| Commit | Concern |
| --- | --- |
| `d825f4b16` | Direct MMVQ shape bench and metadata census; optional kernel template parameters have unchanged production defaults. |
| `e5487f582` | Make the standalone CUDA test target build and use the device's runtime warp size on its host launch path. |
| `909e7422f` | CPU scalar reference, negative controls, frozen-build script and parent-only GPU harness. |
| `94a537a1a` | Production launch-table candidate only: one file, +22/-2 lines (M). |
| `9b7bec28d` | Scale budget estimates with repeats and retain provenance; invalidate excessive A/B acceptance drift. |
| `d1dc31bd8` | Rotate weight copies through a working set at least four times L2 capacity. |
| `e79f5b737` | Exclude the private-helper bench from Windows shared builds, following existing test guards. |

Base: `98c695111df5b5517245b184ef4e818333cf5776`, branch `topic/mmvq-small-batch`. The previous stack3 report remains in `98c695111:codex/REPORT.md`.

`codex/frozen/mmvq-up/SOURCE` is `909e7422f42d0a34eb99e90f410a2eab15a7bb02`: the serving base plus bench/test scaffolding, with the original production launch/reduction rule. `codex/frozen/mmvq6/SOURCE` is `d1dc31bd85d9b7bc2ecfbac3ec14bc6f3bba28a1`: the production candidate plus the corrected bench. The later CMake guard does not alter Linux code generation. Both contain `bin/round-trace`, `bin/bench-mmvq-shapes`, `bin/cuda-op-check`, shared libraries and SHA256SUMS. Every frozen ELF has RUNPATH `$ORIGIN`; CUDA runtime libraries are resolved through the workstation's loader configuration. Binaries stay on disk; only their manifests are committed.

The production candidate applies cleanly to master `050439614` (M): `git show --format= 94a537a1a > build-tmp/mmvq-launch.patch`, then `git apply --check --directory=build-tmp/master build-tmp/mmvq-launch.patch` against that master's MMVQ source. It does not need the bench commits.

## Design and expected winner

`ggml/src/ggml-cuda/mmvq.cu:120` and `:146` select the same architecture table in device and host code. `:458` changes only the four target types at 5..8 columns; `calc_rows_per_block` retains the generic row rule. Q8_0, 1..4 columns, GB10, Turing, Ampere, Ada and AMD rules retain their prior values. Multi-column MUL_MAT_ID uses its separate MoE kernel with fixed row grouping and is unchanged.

G: `w4r2` is the first choice, especially for Q6_K. It increases concurrent K work without increasing the live row/column accumulator count per thread. D, for K=5120: Q6_K's maximum loop trips fall from 10 to 5; Q5_K/Q4_K from 5 to 3; IQ4_XS from 3 to 2. The cost is more shared reduction work and more idle warps in the final iteration. IQ4_XS is the least certain case. The bench also offers `w2r1`, `w4r1`, `w2r4` and `w8r1`; `base` explicitly reproduces the generic pre-candidate launch, and `default` follows production's current table. Selection exists only in the bench; no production environment knob was added.

G: expect 0..1.5 ms saved per round, with about 1 ms as a working hypothesis, not an adoption criterion. D: the supplied IQ4_XS trace's midpoint of 21.15 ms at 339 GB/s would save 1.51 ms if bandwidth reached 365 GB/s (G assumption), or 2.81 ms at the supplied 391 GB/s empirical ceiling (M input). These calculations do not establish J5q speed: its Q5_K bandwidth is unmeasured.

The existing MMVQ/MMQ routing is important: on sm_120, Q4_K goes to MMQ above 5 columns, Q5_K above 6 and Q6_K above 7 (`ggml_cuda_should_use_mmvq`). The bench launches MMVQ directly, including those otherwise hidden cases. This candidate leaves routing unchanged, so a Q4_K six-column microbench win does not affect ordinary six-column MUL_MAT serving.

## Metadata and bench contract

`codex/census.json` comes from local `gguf-py` `GGUFReader(..., mode="r")` tensor descriptors, without reading tensor payloads. J5q's quantized tensor-byte shares are Q6_K 46.343%, Q5_K 43.232%, IQ4_XS 7.310%, Q8_0 1.394%, Q4_0 1.195% (D). Both GGUFs have the following full shapes; half shapes follow the task's tensor-split convention (D).

| Shape | Full GGUF [K,M] | Per-device bench [K,M] |
| --- | --- | --- |
| FFN gate/up | [5120,17408] | [5120,8704] |
| FFN down | [17408,5120] | [8704,5120] |
| GDN attn_qkv | [5120,10240] | [5120,5120] |
| Attention attn_q | [5120,12288] | [5120,6144] |
| GDN attn_gate | [5120,6144] | [5120,3072] |
| ssm_out / attn_output | [6144,5120] | [3072,5120] |
| Q6_K output head | [5120,248320] | [5120,124160] |

`tests/bench-mmvq-shapes.cpp:28` defines the shapes. All five types run the first six shapes; Q6_K also runs the head, at columns 1,2,4,6,8. The full suite has 1085 cases (D): 31 type/shape pairs x 5 column counts x 7 variants. JSONL reports median event time across trials, microseconds (M), and decimal weight GB/s (D). Weight bytes are counted once per launch; this is effective weight bandwidth, not a DRAM hardware-counter measurement. Copies, CPU quantization and activation quantization are outside the timed interval. The weight ring spans at least four times the reported device L2 capacity and advances on every launch; small shapes therefore do not repeatedly reuse one cached matrix. The parser checks this working-set invariant, and a negative control breaks it.

`tests/bench-mmvq-shapes.cpp:189` checks sampled CPU scalar dots using independently dequantized weights and the actual CUDA-quantized RHS. `:209` compares the whole result to the original launch. `--negative-control` perturbs one output and must fail that gate. The normal bench uses fixed warmup duration and repeat counts, not a privileged clock lock.

| Type | CPU cases passed (M) | Maximum CPU NMSE (D) | GPU us / GB/s |
| --- | --- | --- | --- |
| Q5_K | 5 | 6.385e-14 | Unmeasured |
| Q6_K | 5 | 2.311e-13 | Unmeasured |
| Q4_K | 5 | 1.144e-13 | Unmeasured |
| IQ4_XS | 5 | 3.923e-14 | Unmeasured |
| Q8_0 | 5 | 1.602e-14 | Unmeasured |

## Local validation

All commands used `CUDA_VISIBLE_DEVICES=""` and `nice -n 19`. CPU build: `cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_OPENMP_FETCH=OFF`; build parallelism was 2. `codex/mmvq_cpu.cpp` computes CPU MUL_MAT, then compares it with independent double-precision scalar dots over dequantized weights and the CPU backend's quantized RHS. It uses at most 8 compute threads and includes an odd output row count.

Commands and evidence:

- `c++ -O2 -std=c++17 -Iggml/include -Iggml/src codex/mmvq_cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-cpu -lggml-base -o build-cpu/mmvq-cpu`; `build-cpu/mmvq-cpu`: 25 cases passed (M), 841.552 ms total wall time (M, concurrent compilation; not a GPU proxy). `codex/mmvq-cpu-results.txt` contains each error.
- `build-cpu/mmvq-cpu --negative-control`: exit 1 (M), first comparison NMSE 1.00003686945 (D); `codex/mmvq-cpu-negative.txt`.
- CPU-only `build-cpu/bin/test-quantize-fns`: exit 0 (M); `codex/quantize-results.txt`.
- `python3 codex/test_mmvq_params.py`: 210528 rule comparisons passed (M), comparing extracted actual master/current host and device table code; `codex/mmvq-params-results.txt`. This checks parameter preservation, not performance.
- `python3 codex/test_mmvq_gpu.py`: real recorded SSE, GPU-memory and kernel-journal artifacts plus negative controls passed (M). The long recorded request supplies 337 steady intervals (M). See `codex/mmvq-parser-results.json`.
- `bash -n codex/bench.sh codex/gpu_ab.sh codex/freeze_mmvq.sh` and `git diff --check`: passed (M).

`test-backend-ops` was compiled but never executed, as required by the hard prohibition. Its GPU execution and KLD remain the parent's checks. No model server was needed for this backend-only CPU fixture, and no sampling code changed.

CUDA build/freeze command: `FREEZE_JOBS=1 bash codex/freeze_mmvq.sh codex/frozen/mmvq6`; see the script for exact flags. CUDA 13.2, architecture 120 (CMake maps to 120a), NCCL OFF, CCCL v3.4.3 from `/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3`, fully disconnected FetchContent, UI download OFF. The initial full build used at most 6 compiler jobs; the candidate rebuild used 2, then a no-op retry used 1. The RAM guard stopped freezing once when MemAvailable fell below its 8 GiB floor; the retry completed after recovery (M). This guard does not monitor memory continuously during compilation.

Additional compile-only checks passed for the actual candidate MMVQ translation unit on sm_89 and sm_75 (M), using `nvcc -std=c++17 -O3 -use_fast_math -arch=sm_89` (or `sm_75`) `-Xcompiler=-fPIC -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -c ggml/src/ggml-cuda/mmvq.cu`. Logs and objects are under `build-tmp/`. The full sm_120 server, bench and CUDA op checker builds passed (M). Plan-mode runs verified both frozen file sets and their hashes without executing their ELFs (M).

## Parent queue

`codex/bench.sh` and `codex/gpu_ab.sh` default to plan mode. Execution requires `--execute`, `RUN_GPU=1`, the queue's selected `CUDA_VISIBLE_DEVICES`, `XID_SINCE`, and a unique `--out` inside this worktree. `--pins` emits the complete input inventory. No item was queued here.

Parent sequence:

1. `python3 codex/mmvq_gpu.py correctness --execute --out codex/runs/mmvq-correctness/a1`: CUDA0 MUL_MAT with the five quant types. Then use the parent's existing KLD workflow.
2. `bash codex/bench.sh --canary --execute --out codex/runs/mmvq-bench/canary-a1`, then the full bench without `--canary`. Canary uses one bench process, all types/shapes/variants, the same correctness/parser gates, repeat count 5 and one trial; full defaults are 100 repeats, 7 trials and 1000 ms warmup per column/shape. Both execute only CUDA device 0 of the reserved pair. Each attempt retains raw JSONL and writes per-variant JSON under its output directory and `codex/bench-<variant>.json`.
3. `bash codex/gpu_ab.sh --canary --execute --out codex/runs/mmvq-ab/canary-a1`, then the full A/B. Canary uses one patch boot with one request of each kind; full order is up, patch, patch, up. J5q, k=5, probabilistic host draft sampling, tensor split, q8_0 KV, trace off. Report steady ms/round and acceptance drift; a mean drift above 0.05 invalidates the full comparison; token identity is not a gate because reduction order may change rounding.

G estimates / configured budgets: bench canary 300/600 s, full bench 900/1800 s at default repeats; A/B canary 180/600 s, full A/B 720/1200 s. Repeat-dependent bench estimates scale with repeats and trials. The harness rejects stage estimates above 70% of its budget, retains requests before checking gates, stops exact owned PIDs, waits for GPU memory release, checks the kernel journal since the queue cutoff, checks frozen hashes, and emits budget/hazard/INVALID verdicts on failure. The inherited journal parser uses word boundaries and was tested against both the recorded fault and a harmless `stop_xid_hash_s_G` line.

## Risks and open items

- GPU correctness, KLD and all GPU timings are unmeasured. The new bench JSON protocol has synthetic schema/negative tests; its first actual recorded artifact must pass the real-binary canary before measurement. Existing server/memory/journal parsers were replayed against recorded GPU artifacts.
- sm_120 includes RTX 5090. Its performance needs measurement before treating this table as an upstream improvement. Compile success and unchanged rules on other architectures are not runtime/performance tests. HIP/MUSA builds were not performed.
- Wider blocks can lose occupancy, spend more on reductions or waste the final K iteration. The original dequantization work and weight-load implementation are unchanged, so an instruction bottleneck may persist. Gate/up fusion for multiple columns is not implemented.
- Effective microbench bandwidth can include launch gaps; the weight ring prevents deliberate single-matrix L2 reuse. Variant order is fixed; inspect variance and confirm any small gain with reordered single-variant runs and the interleaved end-to-end A/B. A/B acceptance drift can confound speed comparisons and is reported separately.
- The A/B uses the same inherited stack/serving settings in both arms, but does not reproduce the parent's whole real-agent campaign. Keep normal serving at the existing build until the parent checks pass.
- No online issue/PR search, upstream package tree, push or PR submission was performed under the no-network constraint.
