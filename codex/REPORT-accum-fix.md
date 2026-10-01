# F32 MMA accumulator repair

## Conclusion

Two defects in `20d8cfee8` are fixed: a doubled float shared-memory row stride and a missing barrier between combine chunks when `np == 1`. Confidence is high for both source defects. CPU index checks and CUDA compilation pass. GPU numerical correctness, sanitizer cleanliness and performance remain unverified; the parent must run `codex/gpu_accum_ab.sh` before adoption.

Labels: M = measured/observed, D = derived, G = guess. Shapes, flags, source lines and architecture names are references/settings. Every command executed here had `CUDA_VISIBLE_DEVICES=""`, nice 19, and no more than eight worker threads. No GPU binary, sanitizer, GPU discovery command, server, network call or model inference ran.

## Defects and repair

1. Wrong address units, `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1822`. `tile_stride` already counts 32-bit words; `half2` and `float` have equal size. The old float writer used `j*(2*tile_stride)+k`, while the float2 reader uses `(j*tile_stride)/2+k`. The fix uses `j*tile_stride+k`, with optional device assertions under `GGML_CUDA_FA_MMA_DEBUG_BOUNDS` (assertions require `NDEBUG` unset).

   For DKQ=DV=256, ncols1=4, ncols2=8, there are D 64 physical combine rows, a D 132-word stride, and D 128 scalar values per float chunk. The last old store ends at D 67,040 bytes. Allocations are D 33,792 bytes for f16 and D 41,792 bytes for direct q8_0: overruns of D 33,248 and D 25,248 bytes. Thin kernels can fit the wrong stores inside a larger allocation and still read unrelated rows, so absence of an immediate memory exception does not establish correctness.

2. Scratch reuse race, `fattn-mma-f16.cuh:1956`. Halving `nbatch_combine` creates multiple float chunks. Writers own consecutive groups of query rows, while readers distribute rows across warps. At `np == 1`, the old loop lacks a barrier after those reads. A faster warp can overwrite the next chunk before another warp reads the previous one. The fix retains the existing `np > 1` barrier and also synchronizes f32 kernels with multiple chunks.

The scalar/packed tile mapping, half metadata mapping, combine chunk size, raw q8 staging layout, MMA instructions, numerical operations and launch sizing remain as before this repair. Source review and the enumerator found no additional defect in these mappings. The default specialization erases both changed branches at compile time.

M archived `CUDA0-boot2-new-suite.log` contains bad f16 results at both nb=1 and nb=2 (NMSE M 1.77829054 and M 1.88314594), separated by passing q8 vector cases. It reports the subsequent CUDA error at `cudaFuncSetAttribute`, and its stack names the f16 `<256,256,4,8>` wrapper. That delayed error cannot identify the actual faulting launch. The parent reported Xid 13; I did not read the live kernel journal. The source overrun is a concrete explanation, not a GPU reproduction or proof of which launch produced that Xid.

## CPU address audit

`codex/fa_accum_index_check.py:46` compiles the actual production `mma.cuh` tile maps and Ampere/Turing configuration functions into a host-only executable. It discovers the parameter tuples from every generated f16/q8 instance, rather than maintaining a separate shape list. `:169` audits both accumulation modes; `:420` injects each defect as a negative control.

M audit results: 442 supported configuration/mode combinations; 2,281,472 float stores; 48,000 metadata words; 982,528 shared-memory chunks; 1,126,400 output pairs; 9,022,940 global input/output ranges. M wall time including host compilation: 25.420 s. Both defect controls fail: `float writer/reader stride mismatch` and `cross-warp read/next-chunk write race`.

The enumerator includes query widths 1/2/4/8 and all other generated widths, every generated ncols2/head-size combination, scalar float C versus packed half2 C coverage, accumulator indices, rescale column identity, metadata ownership, combine read/write strides, shared Q/K/V/mask/raw-q8 regions, swizzled 16-byte chunks, q8 pair reads, final output guards and both fixup metadata banks. It tests aligned/tail KV lengths, query/head tails, both Q layouts, padded row strides, nonzero view/head/sequence offsets, and valid/empty sparse gathers. Turing excludes unsupported width-8/GQA-8 and q8 pipeline stubs; its width-4 fallback is covered. Ampere's configuration/address rules also apply to sm_86/89/120; the audit does not emulate their hardware instructions.

D bounds argument: a float store has `0 <= j < nwarps*cols_per_warp`, `0 <= k < 2*nbatch_combine`, and stride `2*nbatch_combine+4`, so it fits the host combine allocation and cannot touch the metadata padding. Float2 reads address those same pairs; metadata lies in the next two words. K/V loaders and ldmatrix instructions access only the allocated tile rectangle, and swizzling permutes aligned chunks inside each row. Raw q8 reads stay inside their paired blocks. Query/head guards keep global Q/sinks/dst indices live; synchronous KV tails read local zeros; multi-stage GQA requires the existing padded KV dispatch. Fixup reserves `grid*(2*ncols+ncols*DV/2)` float2 elements and all writers stay inside the corresponding metadata/data regions. Softcap does not change addresses; sparse staging uses less shared memory than the dense multi-stage allocation.

This is an address/ownership proof for the source formulas under the existing caller contracts: valid tensor strides/storage, mask extent, valid sparse indices, positive launch grid and padded multi-stage KV. It is not an exhaustive execution of CUDA, cp.async, warp scheduling, stream-K reduction or floating-point MMA. The race control proves that cross-warp scratch reuse requires a barrier; GPU racecheck remains useful.

## Builds and default preservation

M CPU and full sm_120a CUDA builds passed; the requested architecture 120 is mapped to 120a by this fork's CMake. Logs: `build-cpu/accum-fix-build.log`, `build-cuda/accum-fix-build.log`.

M focused before/fixed PTX compilation passed for sm_75/80/120a, covering f16 query widths 1/2/4/8 at ncols2=8 and both direct-q8 instances. Supported default kernel bodies are byte-identical: M 7 entries on sm_75, M 12 on sm_80, M 12 on sm_120a (`build-accum-fix/default-ptx.txt`). Unsupported Turing width-8 and q8 stubs only change their abort source-line constants. This compares PTX entry bytes, not whole ELF/cubin bytes or a GPU token stream. The other default shapes are unchanged by the same compile-time branch erasure.

M the focused sm_120a compilation with `GGML_CUDA_FA_MMA_DEBUG_BOUNDS` enabled and `NDEBUG` unset passed (`build-accum-fix/debug-bounds.log`). CPU execution of the attention implementation would not exercise the repaired CUDA kernel; it was not used as a correctness proxy. Sampling/RNG code is untouched.

Executed commands (each with `CUDA_VISIBLE_DEVICES="" nice -n 19`):

```sh
python3 codex/fa_accum_index_check.py
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu --target llama test-backend-ops -j8
cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target test-backend-ops -j8
bash -n codex/gpu_accum_ab.sh
bash codex/gpu_accum_ab.sh
sha256sum -c --quiet codex/cuda-accum.sha256
git diff --check
```

PTX inputs are preserved as `build-accum-fix/{before,fixed}.cu`, with the pre-repair header `before.cuh`. Compilation used CUDA 13.2 `nvcc -ptx -O3 -DNDEBUG -std=c++17 -arch=sm_<arch> -use_fast_math -extended-lambda`, include directories `ggml/src/ggml-cuda`, `ggml/src`, `ggml/include`, and defines `GGML_CUDA_USE_GRAPHS`, `GGML_CUDA_FA_F16_F16=1`, `GGML_CUDA_FA_Q8_0_Q8_0=1`. The debug-assert compile used `-c`, added `-DGGML_CUDA_FA_MMA_DEBUG_BOUNDS`, and omitted `-DNDEBUG`.

## Parent gate and open work

`codex/gpu_accum_ab.sh:94` now runs the frozen new binary under `compute-sanitizer --tool memcheck` before any ordinary suite/performance run, on both GPUs. The subset is kv=4096, nb=1..8, both KV types, prec=f32, causal/large. Nonzero sanitizer/process exit, missing/nonzero error summary, failed/unsupported cases or incomplete coverage abort the script. `codex/accum_gpu_results.py:42` requires all D 32 subset cases; M parser controls accept the clean synthetic log and reject nonzero summary, absent summary, missing case and numerical failure. M shell syntax and PLAN=1 pass. Existing old,new,new,old order, exact owned PID cleanup, frozen hashes, Xid/idle guards, corruption control and wall budget are retained. The binary/source manifest has been refreshed for this repair.

M the archived old suite still parses as 332 pass / 30 expected failures. No new GPU result exists. The previous f32 spill/register cost remains a risk; the extra barrier may add cost, and no speed claim is justified.

Parent invocation after the Xid/idle guards can pass:

```sh
RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_accum_ab.sh
```

Open: clean memcheck on both GPUs, full numerical suite, optional racecheck (memcheck alone does not detect shared-memory races), then the retained interleaved A/B timings. Keep the faulty frozen binaries read-only for attribution; do not queue them as the new arm.
