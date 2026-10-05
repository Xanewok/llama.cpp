# Decode KV bound alignment fix

## Conclusion

Fixed a shared-memory alignment defect exposed by the N1 scan's new `__shared__ int kv_end`. The mask scan's packed global loads were already guarded correctly. The added scalar changes shared-memory placement, while VEC's existing `KQ` buffer had only element alignment despite being accessed as `float2` and by vector copies. VEC, TILE and MMA vector-accessed shared buffers now explicitly require 16-byte alignment.

Source/freeze commit: `3785578235d2dfdf7b7bd7cafcd9f789508c4560`. CUDA freeze: `codex/frozen/fa-kvmax-decode-v5`. No GPU program, GPU query tool, sanitizer, server, network request or queue submission was run by this worker. GPU correctness and speed remain unknown. No sampling code changed.

## Root cause and failing arithmetic

Observed: the recorded v4 canary SQL contains a passing `f16/f16` result for `kv=4096,nb=1,tail=4095`, then aborts in the `q8_0/q8_0` VEC launch. The latter is selected by `fattn.cu:695` and `:707`; the backtrace identifies `ggml_cuda_flash_attn_ext_vec_case<256,8,8>`. Raw provenance: `codex/probe-fixtures/SOURCE`; copied SQL/error files preserve the recorded artifacts.

Defect: `fattn-common.cuh:28` adds a shared scalar. The v4 `fattn-vec.cuh:129` declares `KQ` as a shared float array with only 4-byte alignment. `fattn-vec.cuh:159` casts its temporary scale area to `float2 *`; `fattn-common.cuh:435` writes a `float2` there during Q quantization, before the decode mask scan runs. PTX emitted by CUDA 13.2 declares `.shared .align 4 ...E2KQ` and uses `st.shared.v2.b32` for that store. That instruction requires 8-byte alignment.

M: a compiler-only reproduction (`nvcc -ptx`, then `ptxas -g`, then host `readelf`) places v4's D=256, ncols=1 `KQ` at shared offset `0x504`. The address is `KQ + j*D*sizeof(float) + D + 8*(lane/8)`. For the canary's first query and first writer: `0x504 + 0 + 256 + 0 = 0x604`; `0x604 % 8 = 4`. The remaining writer lanes have the same residue. The fixed reproduction places `KQ` at `0x410`, so its first scale store is `0x510`, aligned to 8 bytes. Both softcap variants give the same offsets.

Confidence: high that this explains the recorded fault. The reproduction proves the defect and a concrete failing address, but its debug-assembled shared layout is not a captured address from the optimized frozen canary. The raw canary has no faulting instruction/address. Parent memcheck remains the direct confirmation. The passing f16 case and failing quantized VEC case are consistent with the quantization store, rather than a mask row/view fault.

## Fix and alignment proof

- `fattn-vec.cuh:126`, `:129`: align `KQ` to 16 bytes, covering temporary quantized Q/scales and the vector reduction workspace.
- `fattn-tile.cuh:885`: align `Q_tmp`, `KV_tmp` and `KQ` in both arithmetic branches. Their existing vector loads/stores must not depend on unrelated shared variables' placement.
- `fattn-mma-f16.cuh:1375`: explicitly align the dynamic shared workspace to 16 bytes for vector transfers and `cp.async`. The compiler already inferred this in the inspected CUDA variant; the declaration now states the invariant.
- `fattn-bound.h:14`, `:19`: share the packed dispatch/index calculation with a CPU test. Use `size_t(q)*nbq` rather than signed `q*nbq`, avoiding overflow for large row strides. The normal scan index and skip decision are unchanged.

For physical warp width W, each lane reads B = `256/(2*W)*4` bytes: D 16 bytes on CUDA W=32, D 8 bytes for W=64. Dispatch requires `mask % B == 0`, `nbq % B == 0`, and `nk % 256 == 0`. Every scanned `start` is nonnegative and divisible by 256. The lane address is `mask + q*nbq + 2*start + lane*B`, hence divisible by B. Its last byte is within `[2*start, 2*(start+256))` of that query row. Only `q < nq` is read; query padding and mask row padding are excluded. The destination `values` is explicitly aligned to 16 bytes.

This proof covers arbitrary base/view offsets and every row stride: unsupported residues return `nk` before any wide scan load. It also covers any accepted `nk`, tails and Q count: irregular extents return `nk`; aligned extents can only remove complete tiles. Scan work remains packed and launch-free. No scalar fallback or extra dispatch variants are needed.

## CPU and compiler evidence

`codex/fa-alignment-cpu.cpp` compiles the same `fattn-bound.h` helpers as the kernel. M: 114,688 offset/stride/nk/nq/warp tuples, 17,293,824 issued packed-load addresses, and 16,797,694 nk/tail combinations pass alignment, tile/row bounds and live-column preservation checks. All real query counts from D 1 through D 8 are included. Offsets cover every residue from D 0 through D 31, including odd view offsets that must bypass the scan. Extents include unpadded and irregular lengths and the deep padded length D 94,208. Every tile is checked, including the fully masked stop. A large-stride control verifies that row-offset multiplication does not narrow.

M: `--shared-base 0x504` fails the VEC float2 store check. `--shared-base 0x410` passes. This is the v4 negative control for the actual defect, not an invented failure of the already-correct packed scan math. `codex/check-fa-ptx.py` separately rejects the v4 declaration/layout and verifies the fixed VEC/TILE/MMA compiler declarations. `codex/compile-fa-ptx.sh` reproduces these checks without GPU execution; debug assembly is used only to expose symbols, not for the freeze.

M: the existing packed-mask bit tests pass D 96 cases: unequal query ends, padding exclusion, finite values, NaNs, positive infinity and alignment fallback. The existing stream-k arithmetic test passes D 9,216 partition/fixup cases. `git apply --check codex/negative-control.patch` passes; no GPU mutant was built or run.

Commands, all with `CUDA_VISIBLE_DEVICES=""` and `GIT_NO_LAZY_FETCH=1`:

- `buildgate cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON`.
- `buildgate cmake --build build-cpu --target ggml-cpu -j4`; M pass, `codex/build-cpu-v5.log`.
- `buildgate g++ -std=c++17 -O2 -Wall -Wextra -Werror codex/fa-alignment-cpu.cpp -o build-cpu/fa-alignment-cpu`; M pass.
- `nice -n 19 build-cpu/fa-alignment-cpu --shared-base 0x410`; expected success. The corresponding `0x504` control must exit nonzero.
- `bash codex/compile-fa-ptx.sh`; M pass, `codex/ptx-alignment-check.log`. All CUDA compiler invocations use buildgate.
- `nice -n 19 python3 codex/packed_mask_contract.py` and `codex/streamk_contract.py`; M pass.
- `FROZEN_NAME=fa-kvmax-decode-v5 bash codex/freeze-fa-kvmax.sh`; CUDA 13.2, SM120a, NCCL OFF, ccache ON/read-only, local CCCL343 and disconnected CMake. The script checks `ggml_commit()` against HEAD, freezes with `QWEN38_WT_ROOT=$PWD/codex`, and verifies local library resolution. See `codex/freeze-v5-check.log`.

No CPU model process was needed. The tests are single-process arithmetic/compiler checks, not GPU simulations or performance measurements. Existing CPU/Meta model evidence remains read-only at the v4 worktree; it does not prove this CUDA fix. `cpugate` was attempted for the first arithmetic run, but its `gpuq status` lock is outside the writable workspace and failed; the final single-core, short arithmetic checks use nice directly. No heavy CPU/model run was started after that gate failure.

## N1 bounds and synchronization audit

`git diff 52fc3fd2c..400e24962 -- ggml/src/ggml-cuda/fattn*` adds one vector memory transfer: the packed scan at `fattn-common.cuh:39`. The MMA partition and fixup changes add arithmetic and scalar scan calls, not additional global vector accesses. The defect is the scan scalar's effect on existing vector-accessed shared storage. No added half2/uint2/uint4/float4 cast bypasses the packed alignment gate.

- VEC (`fattn-vec.cuh:250`): selection requires `nk % 256 == 0` (`fattn.cu:692`). Its D 128-column iterations divide a retained D 256-column scan unit. The bound cannot cause a partial unguarded VEC tile. Empty blocks read no K/V/mask data and still write initialized partials for the original combine path. A completely masked row retains the existing zero-denominator behavior unless sinks provide weight; this patch does not define new output semantics for it.
- TILE (`fattn-tile.cuh:954`): the `ncols2==1` branch keeps its final-tile bounds checks. The GQA branch is selected only for aligned lengths; its tile size divides the scan unit. Empty loops retain the original output/sink/combine path. Shared vector buffers now have explicit alignment.
- MMA (`fattn-mma-f16.cuh:2030`): `iter_k = max(1, ceil(kv_end/nbatch_fa))`; zero live columns retain one work tile. A zero-length stream-k span never calls `process_tile`, so it never enters the unconditional preload/last iteration (`:1464`, `:1486`). Shortening changes only partition endpoints; original K/V strides and `ne11` remain the load bounds. Irregular `nk` bypasses the scan and retains the existing final-tile path.
- Uniform fixup (`fattn-common.cuh:813`, `:861`): live `iter_k==1` returns because the sole active block writes a complete normalized tile. Otherwise the last block is active; prior empty spans are skipped before partial reads.
- General fixup (`fattn-common.cuh:905`, `:929`, `:978`): live work and division constants are updated together. Empty current spans return before metadata reads; empty predecessors are skipped. The existing host arithmetic test checks coverage and those gates, including work smaller than the block count.
- Scan participation: every eligibility predicate is block-uniform, the first physical warp scans every real query row, and all block threads reach the result barrier. Fixups wait for their producer before scanning. No early per-thread return precedes the scan.

Residual scope: HIP/MUSA were not compiled. Arbitrary unaligned global views in pre-existing non-scan FA loaders are not newly supported by this patch. The public op currently asserts contiguous F16 masks (`ggml.c:5522`); the scanner's proof deliberately covers broader strides than that. This is an audit of N1's added accesses and partition changes, not a proof of all baseline CUDA loaders on malformed tensors.

## Parent GPU items: run in this order

1. `codex/memcheck-item.toml`: patched freeze only, one visible GPU. Canary runs the exact failing selector `kv=4096,nb=1,.*tail=4095$`, both KV types, one process. Full run selects only the new `tail=` correctness cases, including short unpadded references and verify Q 4/6/8. D internal budgets: canary 240 seconds, full 600 seconds; queue budget 1,000 seconds. G full estimate 400 seconds. Uses `--tool memcheck --error-exitcode 8 --destroy-on-device-error context --force-blocking-launches --print-limit 32`. Requires a sanitizer banner, zero error summary, zero process exit status, complete expected SQL rows, correct library stamp, engagement and a clean current-run kernel journal. No intentional broken GPU kernel is included. Racecheck is deferred because its cost is unmeasured and it does not check misalignment.
2. `codex/probe-item.toml`: v4 design cloned to new bytes, stack5 versus fixed N1, same test executable in both arms, ABBA. Canary is one patch boot with the extreme-tail correctness cases and smallest performance pair. D queue budget 1,400 seconds; G estimate 450 seconds. Both arms select their own frozen libraries. Failures save raw data and check the journal before another process starts.
3. `codex/depth-item.toml`, rendered from `codex/depth-experiment.toml` using `python3 -m harness.mkitem` in the eval repository: unchanged v4 depth design, stack5 versus fixed N1, ABBA, J5q, TP2, k=5, both `--kv-pad 4096`, within-arm identity. D queue budget 3,450 seconds; G estimate 2,300 seconds. Canary is one patch boot at D 8,192 depth, capped at D 600 seconds. `codex/gpu_ab.sh` is the parent wrapper.

M: all three manifests pass `gpuq submit --check`: `codex/memcheck-submit-check.log`, `codex/probe-submit-check.log`, `codex/depth-submit-check.log`. Probe and memcheck CPU preflights pass frozen-byte/library-resolution checks. The depth item passes the rendered-item check and its CPU preflight, including full model SHA and recorded parser fixtures (`codex/depth-preflight-v5-check.log`). The local depth preflight uses `--port 8099` because the sandbox forbids the generator's automatic socket allocation; no listener or server starts. No item was submitted.

The scripts verify frozen bytes, save raw artifacts before gating, stop only their exact owned process groups, cap wall time and scope kernel journals to the current item start. Every parser is tested on recorded GPU artifacts: old FA performance SQL, the task's actual v4 canary SQL, clean/fault memcheck reports, and the shared SSE/memory/journal fixtures. Missing summaries, failed correctness, non-finite timing, missing/duplicate shapes, wrong stamps and faults have negative controls. `codex/memcheck-parser-check.log` records these tests.

Memcheck does not guarantee prevention of an Xid. The locally installed CUDA 13.2 manual (`/usr/local/cuda-13.2/compute-sanitizer/docs/ComputeSanitizer/index.html`, sections Memcheck/Error Actions) says it detects misaligned/out-of-bounds accesses, terminates precise-error kernels and destroys contexts on hardware exceptions. The explicit context-destruction option avoids continuing dependent kernels after an error. Neither that behavior nor launch serialization is a promise that the driver will emit no Xid. A recorded earlier sanitizer fault run already demonstrates that a sanitizer run can fault. Parent must stop on any new hazard and must not run the kernel/depth items until memcheck passes.

## Open items

Parent runs the three canaries/items in order, confirms the fault attribution and numerical gates, and then evaluates performance. The source alignment proof and CPU controls pass; GPU validation, racecheck, HIP/MUSA builds and cross-arm token identity remain unverified. Shared padding can affect occupancy-selected launch geometry, so even this alignment fix does not justify a new bitwise-token claim. The inherited N1 MMA live partition already changes reduction order; the depth item keeps the v4 within-arm identity policy. No upstream package, PR text or remote action was created.
