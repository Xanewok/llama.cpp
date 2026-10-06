# Wide MMVQ at np2

## Conclusion

This is an opt-in probe, not a dispatch recommendation. Default routing is unchanged. The candidate extends the existing dense MMVQ kernel to 9-16 columns for IQ4_XS, Q5_K, Q6_K, Q8_0 and Q4_K. It uses two warps over two weight rows; a second switch compares two balanced narrow launches after one activation quantization. No GPU program was run.

CPU reference: 40/40 registered cases passed (M). CUDA compilation passed (M). GPU correctness, speed, paired KLD and the proposed np2 improvement remain unverified. Queue memcheck first, then the kernel probe, then the depth experiments. Do not adopt routing from this report.

## Dispatch facts

For regular dense MUL_MAT with F32 activations/output and valid padding, `ggml-cuda.cu:2024` tries MMVQ before MMQ (`:2028`). `mmvq.cu:347` chooses by weight type, device capability and activation columns. `common.cuh:60` defines BLACKWELL as 1200: the RTX 5090 tuning comment in `mmvq.cu` does not change this constant's meaning. `mmq.cu:318` supports every served quant type; `:371` selects MMQ when Turing MMA is available, subject to the shared-memory and build-option gates above it.

| Weight type | 7 columns | 8 columns | 9-16 columns | Default MMVQ maximum |
|---|---|---|---|---|
| IQ4_XS | MMVQ | MMVQ | MMQ | 8 |
| Q5_K | MMQ | MMQ | MMQ | 6 |
| Q6_K | MMVQ | MMQ | MMQ | 7 |
| Q8_0 | MMVQ | MMVQ | MMQ | 8 |
| Q4_K | MMQ | MMQ | MMQ | 5 |
| Q4_0 | MMVQ | MMVQ | MMQ | 8 |

F32 is outside MMVQ/MMQ. At these widths it uses MMF if the dimensions/strides pass `mmf.cu:133`; otherwise it falls back to cuBLAS. MMVF permits at most three columns for F32 on this architecture (`mmvf.cu:810`). The TP2 SSM alpha/beta weights have M=24 (D); they fail the 32-row MMF divisibility requirement (`mmf.cuh:9`) and therefore use cuBLAS at 7-16 columns. The table is source-derived (D), not a dispatch trace. Full-attention or per-sequence operations whose local activation width remains six still retain their six-column paths.

## Model shapes and bandwidth floor

`model_facts.py` reads GGUF metadata without a backend. `model-facts.json` records the complete type-byte histogram and representative tensors. J5q has 89.575% Q5_K/Q6_K bytes (D from the observed histogram), confirming the brief. The target GEMM sum includes two-dimensional block weights in blocks 0-63 and the output head; it excludes embeddings, the MTP block, convolution weights and one-dimensional state/norm tensors.

TP2 rules: `src/llama-model.cpp:524` splits Q/K/V output rows; `:548` splits the GDN gate output rows; `:580` splits FFN up/gate rows; `:583` splits FFN down input columns; `:600` splits output-head rows. Qwen GDN segmentation is specified at `:645`.

| Projection | GGUF K x M | TP2 K x M |
|---|---|---|
| FFN gate/up | 5120 x 17408 | 5120 x 8704 |
| FFN down | 17408 x 5120 | 8704 x 5120 |
| Attention Q plus gate | 5120 x 12288 | 5120 x 6144 |
| Attention K/V | 5120 x 1024 | 5120 x 512 |
| Attention output / GDN output | 6144 x 5120 | 3072 x 5120 |
| GDN QKV | 5120 x 10240 | 5120 x 5120 |
| GDN gate | 5120 x 6144 | 5120 x 3072 |

All dimensions above are metadata observations followed by the source-defined split (D for TP2). The test constructor takes M, N, K, not K, M.

| Quantity (D) | IQ4_XS | J5q |
|---|---:|---:|
| Target GEMM weight bytes | 14,539,960,320 | 19,060,060,160 |
| Weight bytes per GPU per verify | 7,269,980,160 | 9,530,030,080 |
| One weight read at 448 GB/s/GPU | 16.228 ms | 21.272 ms |
| Two DRAM weight reads | 32.455 ms | 42.545 ms |

Formula: `sum(weight_bytes) / 2 / 448e9 * 1000`. This is a streaming-weight lower bound, not a bound on a hot-cache microbenchmark; it omits activation/output traffic, quantization, decode arithmetic, launches and communication.

The brief's IQ4_XS target FWD is about 24.6 ms (M, supplied np2 attribution), 1.52 times this floor (D). The supplied MMQ families plus quantization/fixup sum to about 22.45 ms (D from rounded M components); that subset does not contain all weight bytes, so dividing it by the whole-model floor would misstate MMQ efficiency. The supplied extra FWD cost versus np1 is about 3.7 ms/round (M). Those are existing measurements, not measurements of this patch.

Two six-column passes cannot win if every weight really goes to DRAM twice: their IQ4_XS floor alone exceeds the existing whole target FWD. They may win when the second pass reuses cache. For example, a TP2 IQ4_XS FFN matrix is 23,674,880 bytes and a Q5_K matrix is 30,638,080 bytes (D); the output head is 521,472,000 bytes/GPU (D) and cannot be treated as a small cached matrix. Only eligible GEMMs change, so the real duplicate-read increment uses their byte subtotal, not the complete target sum. The comparator retains one Q8_1 activation quantization and adds one GEMM launch. It does not share decoded weights in software.

## Design and compiler evidence

`mmvq.cu:326` limits the probe to five types. `:351` enables it only at sm_120 and 9-16 columns. `ggml-cuda.cu:2024` requires M divisible by four, so both the wide kernel and narrow comparator read complete row blocks. Odd-row tests retain MMQ. MoE, fusion, other architectures and all default thresholds are unchanged.

`GGML_CUDA_MMVQ_WIDE=all` enables the five types; an exact `ggml_type_name()` such as `q5_K` selects one type. Unset or `off` preserves baseline routing. `GGML_CUDA_MMVQ_WIDE_SPLIT=1` launches floor(N/2) and ceil(N/2) columns; N=12 produces two six-column launches. Both pointers advance by their column strides while channel/sample strides retain the complete tensor layout. These variables are cached per process; compare fresh processes.

`mmvq.cu:485` and `:618` fix wide geometry independently of architecture tables. This prevents a host sm_120 table from launching an older-architecture/PTX specialization with different geometry. `:1233` instantiates only non-fused dense kernels. `GGML_CUDA_DEBUG_MMVQ_WIDE=1` prints the first width/type/mode on each host thread for canary engagement.

The existing kernel holds one accumulator per output column and weight row per thread and repeats vec_dot across columns. It provides locality and compiler reuse, not a new explicit shared decoded-weight tile. At N=12, w2r4 needs 48 accumulator floats/thread (D); w2r2 needs 24 (D). The initial w2r4 candidate compiled without spills but consumed too many registers; the selected w2r2 candidate reduces them.

| Type | Six-column w2r4 registers/thread (M) | Twelve-column w2r4 (M) | Twelve-column w2r2 (M) |
|---|---:|---:|---:|
| IQ4_XS | 162 | 244 | 203 |
| Q5_K | 134 | 201 | 147 |
| Q6_K | 123 | 178 | 128 |
| Q8_0 | 96 | 146 | 112 |
| Q4_K | 157 | 223 | 165 |

All selected 9/12/16-column entries have zero stack, spill stores and spill loads (M). `registers.json`, `registers-w2r4.json` and the local ptxas logs hold the evidence. Registers alone do not establish occupancy or bandwidth. The unchanged two-warp K partition is preferable to changing the reduction merely to save registers.

A provisional whole-round forecast is minus 1-2 ms for IQ4_XS (G): 41.7-42.7 / 44.8-45.8 / 53.3-54.3 ms at 8k/32k/90k (G, supplied M baselines minus the guessed saving). A regression is plausible, particularly for IQ4_XS's 203 registers/thread. No quantitative J5q forecast is justified before the type probe. Reaching the streaming floor is not a prediction.

## Verification and failures

- `tests/test-backend-ops.cpp:9395` adds 360 cases per eval/perf list (D): five types, nine widths including the six-column control, seven TP2 projections and one odd-row fallback shape. All compiled in CPU and sm_120a CUDA builds (M).
- The renamed CPU-only executable passed every registered K/V projection case at N=9-16: 40/40 (M), `cpu-registered.sql`. Its timestamp span is two seconds (D), which is not a performance metric. Other production shapes were compiled, not run on CPU; the memcheck item covers them on CUDA.
- `thread_limit.cpp` is a local test-only preload shim. It makes `hardware_concurrency()` report 16 (M, `cpu-thread-count.log`); the test runner uses half, giving eight workers (D). The queued kernel item uses the same frozen shim. No CPU model/server was run.
- A preliminary exported-generic CPU test produced NaNs in both outputs (`cpu-results.sql`). The registered cases passed, but the generic-export cause is unresolved. The GPU items use registered cases exclusively.
- That preliminary run escaped `cpugate` after its queue-status lock read failed in the sandbox. Its test records span 09:36:35-09:36:37 UTC on 2026-10-06 (M), and the runner initially used its uncapped hardware-derived worker count. Disregard overlapping GPU timing samples. Subsequent CPU work used working gated access and the worker cap.
- `kernel_item.py --stage selftest` tests SQL parsing on the recorded GPU artifact `results/2026-10-04-mmvq-baseline/a1/raw/gen-gateup-r1.sql`, sanitizer parsing on the recorded GPU `results/2026-10-06-gdn-indexed-memcheck/a2/mc-idx.tail.txt`, and hazard parsing on the gpuq fixture. Native-case normalization is additionally checked on recorded CPU support output. Missing/duplicate cases, invalid time, failed correctness rows, sanitizer errors and missing sanitizer evidence are rejected. No GPU artifact for the new engagement message exists yet; the real canary must establish that gate before a full run.
- Existing depth parsers passed `harness.depth.fixture_test` on `orchestration/gpuq/fixtures/`, including partial counters, size and identity negative controls (`depth-parser-tests.log`). The np2 harness validates two streams, their overlap window, sufficient steady intervals and VRAM sampling.

Builds used `CUDA_VISIBLE_DEVICES=''`, `GIT_NO_LAZY_FETCH=1`, nice 19, `buildgate`, four compile workers and `GGML_CCACHE=ON`. Shared ccache was read-only in the sandbox; misses compiled locally with a worktree temporary directory. Configure/build logs are in `codex/`. CUDA flags match stack6: CUDA 13.2, NCCL off, architecture 120a, CCCL 3.4.3 from the existing local source, network fetches disabled.

## Numerics and adoption

No sampling code changed. Opting into another GEMM route can change logits and therefore the model's output distribution. MMVQ and MMQ have different activation layouts, scale rounding and reductions. Ledger row 141 reports provisional KLD +2.4e-4/token for routing Q5_K/Q6_K from six-column MMVQ to MMQ (M from that prior study); the reverse direction at twelve columns is not established by it.

Keep the switch until paired BF16-reference KLD and per-type performance are measured. GPU NMSE correctness is necessary but insufficient for KLD neutrality. np2 identity is disabled because batch composition is timing-dependent, not because the new numerics are proven equivalent. No claim of bit-identical tokens or an equivalent output law is made for enabled routing. Default routing preserves the existing path.

## Frozen build and queue order

The freeze is `codex/frozen/mmvq-np2`, made with `QWEN38_WT_ROOT=$PWD/codex bin/wt freeze`. Its SOURCE commit is `422fe0128`; the build was configured at that HEAD and the generated GGML stamp matches. `round-trace` is the renamed server. The helper checked local RUNPATH resolution and wrote binary SHA256SUMS. No CUDA executable was invoked to verify the stamp.

Prepared items, never submitted or run:

1. `items/memcheck.gpuq.toml`: GPU0 memcheck, all five types, widths 9-16, all seven TP2 projections, wide and split modes. Canary: one native test process, the smallest registered projection at widths 9/12/16. Both stages set `GPUQ_ACTIVE="1"`.
2. `items/perf.gpuq.toml`: both GPUs separately; up/wide/split/split/wide/up processes, seven projections at widths 6/12. One frozen test driver uses baseline stack6 libraries for up and candidate libraries for wide/split, with GGML stamp checks. This microbenchmark repeats matrices and can be cache-hot; it cannot establish cold streaming bandwidth or ms/round. Canary exercises all three modes on GPU0.
3. `items/depth-iq4.gpuq.toml`, `items/depth-j5q.gpuq.toml`: frozen stack6 versus this freeze, ABBA, np2, both slots busy, k=5, 8k/32k/90k, identity off. Here k=5 is intentional: np2 verification is 2 x (k+1) = 12 columns (D); k=3 would stay at eight columns. One patch boot at 8k is the <=10-minute canary. Actual model workloads and canary gates remain untested.

Depth manifests are rendered by `python3 -m harness.mkitem` in qwen38-eval. Serverless manifests use the same contract as the supplied memcheck model: `harness.probe` always boots a server and cannot render a serverless test suite without a harness change. No harness files were edited. All four manifests must pass `gpuq submit --check`; validation results are in `item-checks.log`.

`gpu_ab.sh` is a GPUQ_ACTIVE-guarded wrapper for the existing depth runner. That runner provides frozen-byte checks, exact-PID server stop, scoped journal checks, interleaved boots and budgets. The kernel perf runner additionally requires a clean full memcheck PASS for the same frozen commit. The parent must queue depth only after full memcheck passes; the existing depth runner has no cross-item dependency gate. Real canaries are mandatory before full measurements. Numerical review is required before adoption.

## Upstream split and open work

The source diff is one CUDA GEMM concern plus its backend tests. An upstream kernel-support change should contain the non-fused 9-16-column instantiations, fixed geometry, bounds eligibility and projection tests. The environment routing, split comparator and census remain local measurement controls. Promote a default route only after a measured type/width table shows it is strictly better and paired KLD is acceptable; submit neither an unsupported threshold table nor a model-specific flag.

Open: CUDA memcheck/NMSE; cold versus hot-cache interpretation; per-type winner and widths; separate paired KLD; np2 real-model canaries and replicated A/B; generic-export CPU NaNs. No issue/PR duplication search was performed under the no-network rule, and no upstream submission was made.
