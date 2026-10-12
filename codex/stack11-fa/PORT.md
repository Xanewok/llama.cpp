# stack11-fa: q8_0 FA MMA commits ported onto master's swizzle API

Branch `topic/stack11-fa`, base `abee0c847` (master). Static port only: nothing was compiled or run.

Series (oldest first):

| commit | subject | source (stack10) |
|---|---|---|
| 69ea57e19 | FA K/V to f16, one contiguous conversion per KV stream (pkg 13) | `git cherry-pick -x 2b53ebdca`, clean |
| a2b9851cc | FA per-stream K/V conversion only for dense, aligned streams (pkg 13) | `git cherry-pick -x 7e1f24e3f`, clean |
| ad05f046f | FA MMA kernel dequantizes q8_0 K/V tiles for small batches (pkg 14/0001) | cf2580acc |
| 1000bf397 | tests : FA with q8_0 K/V at speculative decoding batch sizes | 64730fd0d |
| a965db6a0 | spread q8_0 FA dequant stores over rows (pkg 53) | a655c25fa |
| 725017ca9 | add grouped one-row q8_0 flash attention (pkg 54) | a1fff47fc |
| 1b70e0b8c | keep the multi-query q8_0 MMA FA path free of single-query decode code | 2efa6f8dc |

## Net equivalence check

`diff` of the +/- lines of `git diff 050439614 2efa6f8dc` (stack10) against `git diff abee0c847 HEAD` (this branch,
before the codex commit), per file:

- `fattn-common.cuh`, `generate_cu_files.py`, all six `fattn-mma-f16-instance-q8_0-*.cu`, `tests/test-backend-ops.cpp`
  (the test lines of commits 2 and 4): identical (instance and generator blobs have the same git object ids as in 2efa6f8dc).
- `fattn-mma-f16.cuh`: differs only in the swizzle API lines listed below, plus the `async_kv_preload` argument.
- `fattn.cu`: differs only by the stack10 local knobs, which are not part of this series: ac6db6790
  (`GGML_CUDA_FA_MMA_Q8_0`, "DO NOT MERGE"), bd251ce1b (`GGML_CUDA_Q8_DECODE` and a `GGML_LOG_DEBUG` line in the one-row
  dispatch). With both env vars unset stack10 selects the same kernels as this branch.

## Swizzle semantics (why the layout is unchanged)

Old (stack10, `mma.cuh` before #29612): `swizzle_bytes<swz, half2>(i, j, stride) = 4*(i*stride + j)`, XOR `(i & 7) << 4`
when `swz`. `swz = turing_mma && nbatch_K2 % 32 == 0 && nbatch_V2 % 32 == 0 (both >= 32)`, stride = `nbatch_2` if swz
else `nbatch_2 + 4`.

New (abee0c847): `swizzle<stride, half2>(off, i)` with `stride_bytes = 4*stride`: unchanged if `stride_bytes % 32 != 0`;
XOR `((i/4)%2) << 2` if `% 64 != 0`; XOR `((i/2)%4) << 2` if `% 128 != 0`; else XOR `(i%8) << 2` (half2 units). The
pointer form `swizzle<stride>(ptr, off, i)` returns `ptr + swizzle<stride, T>(off, i)`. `swz` and the strides are
computed by the same rule (`ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols[, cc])`, `nbatch % 32 == 0`).

- swz strides are multiples of 32 half2 = 128 bytes, so the new XOR `(i%8) << 2` half2 = `(i%8) << 4` bytes equals the old one.
- Non-swz strides `nbatch_2 + 4` with `nbatch_2 % 8 == 0` are 16 mod 32 bytes: identity in both.
- Non-swz strides with `nbatch_2 % 8 == 4` would be swizzled by master but not by the old code. No q8_0 instance has one.

q8_0 instances on cp.async GPUs (Ampere config table, which sm_120 uses; Turing has no cp.async so the q8_0 path is off):

| instance (DKQ,DV,ncols1,ncols2) | ncols | nbatch_K2 / V2 | swz | stride K / V (half2) | old vs new layout |
|---|---|---|---|---|---|
| 256,256,4,8 / 8,8 (multi-query) | 32 / 64 | 128 / 128 | yes | 128 / 128 | same XOR (i%8)<<4 B |
| 64,64,1,8 | 8 | 32 / 32 | yes | 32 / 32 | same |
| 96,96,1,8 | 8 | 48 / 48 | no | 52 / 52 (208 B) | identity in both |
| 128,128,1,8 | 8 | 64 / 64 | yes | 64 / 64 | same |
| 256,256,1,8 | 8 | 128 / 128 | yes | 128 / 128 | same |
| 512,512,1,8 | 8 | 128 / 128 | yes | 128 / 128 | same |
| 192,128,1,8 / 1,16 | 8 / 16 | 96 / 64 | yes | 96 / 64 | same |
| 320,256,1,32 | 32 | 128 / 128 | yes | 128 / 128 | same |
| 576,512,1,16 / 4,4 | 16 | 288 / 256 | yes | 288 / 256 | same (V_is_K_view is false for q8_0) |

So every q8_0 f16 staging tile has byte-identical contents to stack10, and the master readers
(`load_ldmatrix_swizzled`, `load_ldmatrix_trans_swizzled`) read the same addresses as the old `load_ldmatrix<swz>` /
`load_ldmatrix_trans<swz>` (their tile bases `i_KQ_0`, `2*k0` are multiples of 16 rows, so the local-row XOR equals the
absolute-row XOR). The raw q8_0 staging tile (`tile_q8`, unswizzled rows of `D/32*34` bytes after the mask tile) and the
shared memory size are unchanged.

## Changed call sites (b)

Paths: stack10 = `2efa6f8dc:ggml/src/ggml-cuda/fattn-mma-f16.cuh`, port = `HEAD:ggml/src/ggml-cuda/fattn-mma-f16.cuh`.

| # | stack10 line | port line | before | after | layout reasoning |
|---|---|---|---|---|---|
| 1 | 575 | 564 | `template<int D, int stride_tile, bool swz, int nwarps, int nbatch_fa, bool one_row>` dequantize_tile_q8_0 | `swz` dropped | swizzle now derived from `stride_tile` |
| 2 | 627 | 616 | `memcpy_1<16>((char *) tile_KV + swizzle_bytes<swz, half2>(i, k, stride_tile), v)` (pair path) | `memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k, i), v)` | same byte address for the strides above |
| 3 | 646 | 635 | `... swizzle_bytes<swz, half2>(i, b*QK8_0/2 + c*4, stride_tile)` (per-block path) | `swizzle<stride_tile>(tile_KV, i*stride_tile + b*QK8_0/2 + c*4, i)` | same |
| 4 | 652 | 641 | `template<int stride_tile, bool swz, int nwarps, int nbatch_fa>` load_tile_q8_0_partial | `swz` dropped | — |
| 5 | 670 | 659 | `... swizzle_bytes<swz, half2>(i, k*4, stride_tile)` (partial loader) | `swizzle<stride_tile>(tile_KV, i*stride_tile + k*4, i)` | same; column 0 = `k0_start` as in master's f16 `load_tile` |
| 6 | 826 | 815 | `dequantize_tile_q8_0<DKQ, stride_tile_K, swz, nwarps, nbatch_fa, one_row>` | `swz` dropped | — |
| 7 | 831 | 820 | `load_tile<stride_tile_V, swz, ...>` (f16 V, nstages>1) | master form `load_tile<stride_tile_V, nwarps, ...>` | master code |
| 8 | 853 | 842 | `load_tile_q8_0_partial<stride_tile_K, swz, nwarps, nbatch_fa>` | `swz` dropped | — |
| 9 | 856 | 845 | `load_tile<stride_tile_K, swz, ...>` (f16 K, nstages<=1) | master form | master code |
| 10 | 1190 | 1179 | `dequantize_tile_q8_0<DV, stride_tile_V, swz, nwarps, nbatch_fa, one_row>` | `swz` dropped | — |
| 11 | 1202 | 1191 | `load_tile<stride_tile_K, swz, ...>` (f16 K preload) | master form | master code |
| 12 | 1220 | 1209 | `load_tile_q8_0_partial<stride_tile_V, swz, nwarps, nbatch_fa>` | `swz` dropped | — |
| 13 | 1223 | 1212 | `load_tile<stride_tile_V, swz, ...>` (f16 V, nstages<=1) | master form | master code |
| 14 | 1572 | 1560 | `load_tile<stride_tile_K, swz, ...>` (process_tile K preload) | master form | master code |
| 15 | 2316 | 2303-2305 | q8_0 `launch_fattn<...>(..., false, false, true, false, warp_size_host)` | `+ async_kv_preload = nstages == 2` | not a layout change, see below |

Untouched master code the q8_0 path relies on: `swz`/stride computation (port 794-796, 1450-1453, 2261-2263),
readers at 822/848 (`load_ldmatrix_swizzled<stride_tile_K>`), 1192 (`load_ldmatrix_trans_swizzled<stride_tile_V>`).

## Behavioural differences vs stack10

- `async_kv_preload` (#29435 d89651a7b): master passes `nstages == 2 && !use_sparse` for the f16 path; the q8_0 path (never
  sparse) now passes `nstages == 2`. It only matters when `cc == GGML_CUDA_CC_DGX_SPARK` and the mask scan runs
  (whole-tile scheduling instead of stream-K). On sm_120 (RTX 5060 Ti) the grid is unchanged. Passing `false` would
  reproduce stack10 exactly on DGX Spark too.
- #30103 1bb2b9fcb adds a `__syncthreads()` in process_tile when `nbatch_combine != DKQ/2` (race fix). Ampere config:
  64/96/128/256 instances have nbatch_combine == DKQ/2 (no change); 192/128, 320/256, 512, 576/512 take the extra
  barrier. A barrier that removes a race does not change race-free results.
- Expected: bitwise identical FA outputs to stack10 for the same kernel selection on sm_120. Token identity of the whole
  server also depends on the rest of the stack (outside this series).

## Template instances (c)

Added (all generated, byte-identical to stack10 blobs):

- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_4-ncols2_8.cu` (256,256,4,8) — commit 1
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_8-ncols2_8.cu` (256,256,8,8) — commit 1
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_8.cu` (64,96,128,256,512 square; 192/128) — commit 4
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_16.cu` (192/128, 576/512) — commit 4
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_32.cu` (320/256) — commit 4
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-q8_0-ncols1_4-ncols2_4.cu` (576/512) — commit 4

`generate_cu_files.py` (pure file generator: deletes `*.cu` in cwd and rewrites them) was run in a scratch copy and in
place: `diff -rq` against the tree was empty (127 files) and `git status` stayed clean. The generator keeps stack10's
duplicated `SOURCE_FATTN_MMA_Q8_0_CASE` assignment (harmless). The CMake glob `template-instances/fattn-mma*.cu` picks
the files up.

## Old-API grep (a)

`grep -rnE 'swizzle_bytes|ggml_cuda_fattn_mma_get_stride_tile|load_ldmatrix<|load_ldmatrix_trans<|ggml_cuda_fattn_mma_bank_aligned|<[^<>]*, swz,' ggml/ tests/`
returns nothing. All device `ggml_cuda_fattn_mma_get_swizzled` calls take 3 arguments (DKQ, DV, ncols); the 4-argument
one at port line 2261 is master's host overload `(DKQ, DV, ncols, cc)`.

## Compile / numeric risks (not verified, no build here)

- `swizzle<stride_tile>(tile_KV, uint, uint)` deduces `T = half2` from `half2 * const __restrict__` exactly like master's
  own `load_tile` call (port 473); int offsets convert to `uint32_t` implicitly. Its `static_assert(stride_bytes % 16 == 0)`
  holds for every q8_0 stride (all `nbatch % 4 == 0`).
- `swz` is still used for the strides and the combine-phase `__syncthreads`, so no unused-variable warning.
- If master later changes `swizzle` to also XOR padded strides with `nbatch_2 % 8 == 0`, the layout stays consistent
  (writer and reader share the function) but would no longer be byte-identical to stack10; values read are unaffected.

## test-backend-ops cases for the CUDA build window

`-p` is a `std::regex_search` over `vars()`, e.g.
`hsk=256,hsv=256,nh=4,nr23=[6,1],kv=512,nb=3,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=[0,2,1,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0`
(+ `,kv_stream_view=1`). Kernel path given for sm_120 (Ada+ selection, Ampere config).

Correctness (`test-backend-ops -m test -b CUDA0 ...`):

| # | filter | covers | exists |
|---|---|---|---|
| T1 | `-o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4,nr23=\[6,[12]\],kv=[0-9]+,nb=[1-8],.*type_K=q8_0'` | Qwen3.8 shape (D=256, 4 KV heads, GQA 6), nb 1..8: nb 3-4 -> `<256,256,4,8,Q8_0>`, nb 5-8 (incl. spec size 6) -> `<256,256,8,8,Q8_0>`, nb 1 with q8_0 V -> one-row `<256,256,1,8,Q8_0>` (kv 512/1024/4096), nb 2 and f16 V -> vec / f16 copy (controls); two streams (`kv_stream_view=1`, nb 3/4/8), head-major layout (permute `[0,1,2,3]`), sinks (nb 4), softcap 10 (f16 copy control) | after commit 2 |
| T2 | `-o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=1024,nb=5,'` | GQA 8 multi-query `<256,256,8,8,Q8_0>` | after commit 2 |
| T3 | `-o FLASH_ATTN_EXT -p 'kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0'` | all commit-4 one-row cases: hs 64/96/128/256 x GQA 2/4/6/8 (only 256 with GQA 4/6/8 passes the cost gate -> `<256,256,1,8>`; GQA 6 = Qwen TP-split shape with sinks; the rest are fallback controls), `nr23=[6,2]` with/without `kv_stream_view`, 192/128 GQA 8/16 (cost gate rejects on Ampere: control), 320/256 GQA 32 -> `<320,256,1,32>`, 512 GQA 2/6 -> `<512,512,1,8>`, 576/512 V-view-of-K GQA 16 -> `<576,512,1,16>`, GQA 20 -> `<576,512,4,4>` | after commit 4 |
| T4 | `-o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=(1024|16384),nb=1,.*type_K=q8_0'` | one-row `<256,256,1,8>` with 2 groups per KV head | before the series (master) |
| T5 | `-o FLASH_ATTN_EXT -p 'hsk=576,hsv=512,nh=1,nr23=\[8,1\],kv=(113|1024),nb=1,.*type_K=q8_0'` | kv 1024 -> `<576,512,4,4,Q8_0>`; kv 113 (not a multiple of 256) -> f16 copy control | before the series |
| T6 | `-o FLASH_ATTN_EXT -p 'type_K=q8_0'` | superset of T1-T5 plus the generic loops (hs 64/72 q8_0, nb 1/3/32/75, streams, prefill shapes) that must keep their old paths | partly after commits 2/4 |
| T7 | `-o FLASH_ATTN_EXT` | whole op: every f16 instance is rebuilt from the changed template (`type_KV` / `one_row` parameters) | — |

Minimum for the window: T1, T2, T3, T4, T5 (all fast); T6 if time allows; T7 once per build.
`ncols1=1, ncols2=8/16/32` = T3 (+T1 nb 1, T4); `ncols1 4/8, ncols2 8` = T1, T2; `ncols1 4, ncols2 4` = T3 (GQA 20), T5.

Perf (`-m perf`, commit 2): `-o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[6,1\],kv=(8192|32768|98304),nb=[1-8],.*kv_view=0'`
(f16 vs q8_0 K/V, one GPU of a two-way split; exists after commit 2).

Stack10 identity: run T1-T5 with the same filters on the frozen stack10 build and compare the per-case NMSE lines, or
use a bitwise FA dump if the harness has one; test-backend-ops only checks against the CPU within 5e-4 NMSE.
