# REPORT fa-q8-inkernel (Scope A): in-kernel q8_0 K/V dequantization for prefill FA tiles

Branch `topic/fa-q8-inkernel` on base `def4a4e20` (topic/s10c-fafix: source `2efa6f8dc`, the `GGML_ASSERT` comma fix, plus its codex commits). Nothing here was compiled or run:
no CUDA on the authoring host. The window below is the first compile, the first execution and the first measurement.

## Commits

| hash | subject | kind |
|---|---|---|
| `0690c602c` | cuda : dequantize q8_0 K/V tiles in the MMA kernel for prefill batches | general |
| `09c381d57` | local: query-tile cap env, engagement log and K/V swap mutant for the q8_0 prefill FA | local |
| (next) | codex : window, gates, bench, reserve probe, depth item template, this report | codex |

## Source changes (general commit)

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
  - `ggml_cuda_flash_attn_ext_mma_q8_0_instance(DKQ,DV,ncols1,ncols2)`: DKQ==DV in {128,256}, ncols2 in {2,4,8}, ncols1*ncols2 in {32,64}.
    Was: (256,256,{4,8},8) only.
  - `flash_attn_ext_f16_load_tile_q8_0` (multi-query branch): 16-byte `cp.async.cg` when the row is a multiple of 16 bytes
    (D=256: 272 B, unchanged code path via `if constexpr`), otherwise 8-byte `cp.async.ca` (D=128: 136 B).
  - `extern DECL_FATTN_MMA_Q8_0_CASE` for the 10 new instances (256: 16x2, 32x2, 8x4, 16x4; 128: 16x2, 32x2, 8x4, 16x4, 4x8, 8x8).
  - Comments (the "one-row code stays out of multi-query instances" contract is unchanged).
- `ggml/src/ggml-cuda/cp-async.cuh`: `cp_async_ca_8` (inline PTX, `NO_DEVICE_CODE` fallback); comment on the old 16-byte-only note.
- `ggml/src/ggml-cuda/fattn.cu`
  - `..._use_gqa_opt(dst)` factored out of `switch_ncols2` (no behaviour change); `..._get_ncols2(dst)`: the ncols2 the dispatch picks
    on Ampere and newer (asserted equal in `switch_ncols1`).
  - `..._mma_q8_0_max_query_tiles()` = 16 (G, to be set from the bench below).
  - `..._mma_q8_0_supported(cc,dst)` rewritten: NVIDIA, tensor cores + cp.async; K,V q8_0 with equal head size; mask; no ALiBi;
    no softcap; `Q->ne[1] >= 2`; both tile widths (32/ncols2, 64/ncols2) are instances; `ceil(rows/ncols1) <= cap`; K/V
    `nb[0]==sizeof(block_q8_0)` and data, `nb[1..3]` aligned to 16 (or 8 if the row is no multiple of 16); no sparse variant.
  - `switch_ncols1`: narrow/wide tile from ncols2 (was hard-coded 4/8); `Q->ne[1] <= narrow ? narrow : wide`.
  - `ggml_cuda_flash_attn_ext_get_alloc_size`: no f16 scratch when `decode_supported` or (kernel MMA_F16 and `mma_q8_0_supported`).
    Dispatch and size share one predicate.
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` + `fattn-mma-f16-instance-q8_0-ncols1_{16-ncols2_2,32-ncols2_2,
  8-ncols2_4,16-ncols2_4}.cu` (new) and `_4-ncols2_8`, `_8-ncols2_8` (gain D=128). The generator reproduces the files.
- `tests/test-backend-ops.cpp`: q8_0 K/V prefill cases: hs {128,256} x gqa {1,2,4,6,8} x nb {3,9,64,128,512} x 1/2 sequences
  (2 sequences = non-contiguous stream views) at kv 8192; hs 256 gqa 6 at kv 32768 and 90112 x nb {64,128,512} x 1/2 sequences;
  fall-through controls kv 8192+128 (not a multiple of 256), ALiBi, no mask.

Incidental behaviour changes beyond Swift (all measured by the window, none claimed beforehand): q8_0 K/V with D=128 and GQA 2-8,
and D=256 GQA 2/4, now use the in-kernel path for 3+ rows (before: f16 copy); on Ampere, 2-row batches take it too (on Ada and
newer two rows go to the vector kernel, unchanged). GQA 1 stays on the f16 copy (the kernel needs the multi-stage pipeline,
ncols2 >= 2).

## Local commit (`09c381d57`, not for upstream)

`GGML_CUDA_FA_Q8_QTILES` (cap override, read per call, 0 = path off), `GGML_CUDA_FA_Q8_MUTANT=1` (K/V swapped for the in-kernel
call), debug log `fattn_q8_mq: device= dk= ncols1= ncols2= rows= seqs=`.

## Scratch reserve contract (why the reserve may shrink)

`get_alloc_size` is evaluated on the worst-case reserve graph. It omits the scratch only if every smaller batch of the context also
needs none: tile count is monotone in rows; 1 row goes to the single-query MMA instances or the vector kernel; 2 rows go to the
vector kernel on Ada+ (and to this path on older GPUs, hence the lower bound 2); all other conditions depend on the shape, not the
batch size. Falsifier of the reserve claim: a boot at the served shape that logs `cudaMalloc failed` / graph re-reserve after the
first requests. With the cap at 16 tiles: np2 `-ub 128` (64 rows per sequence = 8 tiles) loses the scratch in the main and the draft
context; `-ub 512` (64 tiles) keeps it unless the cap is raised, so the brief's "main 845 -> <= ~100 MiB" holds for `-ub 128` only (D).

## Exactness argument

The new instances are the verify instance's code (stack9 `<256,256,8,8,Q8_0>`) at other tile widths. The staged q8_0 values are
dequantized into the f16 tile with the arithmetic of the f16 conversion kernel (q exact in f16, q*d rounded once), so each K/V
element equals the one the f16 copy would hold; the mask, softmax and MMA code are the f16 kernel's. Byte identity with convert +
f16 additionally needs the same grid: stream-k geometry follows `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, which the extra
q8 staging shared memory can change. So byte identity is measured, not assumed (G: identical at D=256; if occupancy differs the
difference is a summation order, NMSE ~1e-9). Gate: NMSE <= 1e-6 hard; byte identity counted (stage 45) and enforced last (stage 50).

## Window `codex/bw-fa-q8-inkernel.{toml,sh}` (filler, priority 18, runs after bw-s10c-fafix)

`gpuq submit --check`: valid; `python3 -m gpuq.shlint` clean on every `.sh`. Preflight (branch, clean tree, BASE ancestry, file-set
whitelist, `frozen/fa-q8-inkernel` absent, `frozen/stack10-cand2` present with SOURCE an ancestor of HEAD and identical outside
`codex/` to BASE, parser harness unchanged) passes under `env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin` against a stand-in
reference (`frozen/stack10-cand`); against the real reference it can only pass after `bw-s10c-fafix` froze `stack10-cand2` and this
branch is rebased onto the fafix commit (BASE in the script then moves to that commit).

| stage | name | gate |
|---|---|---|
| 40 | cuda-build | full CUDA 13.2 / sm_120a build of llama-server + test-backend-ops (first compile of this tree) |
| 41 | sass-gate | `fa_q8_sass_gate.py` vs `frozen/stack10-cand2`: `<256,256,8|4,8,Q8_0>` instr <= +1 %, STL/LDL not above, HMMA/REG/STACK/LOCAL/SHARED equal; no common kernel changes resources; new kernels = exactly the 10 expected; resources, STL/LDL, instruction count of each printed; control `frozen/stack10-cand` (measured regression) must fail |
| 42 | cuda-stamps | DeviceTopK count equals the reference, build stamp |
| 43 | cuda-fa-ops | `-o FLASH_ATTN_EXT` full on CUDA0 and CUDA1 (only failures shared with the stack10-cand2 binary tolerated); all q8_0 K/V cases at cap 1000000 (engagement logged) and at cap 0 (no engagement); decode subset |
| 44 | build-tools | `fa-q8-identity.cpp`, `fa-q8-bench.cpp` with `-Wall -Wextra -Werror` against the new libs |
| 45 | fa-byte-identity | decode/verify/other digests == base build; `prefill` group (51+ cases, B/A/M/D arms): B == base bytes, engagement == dispatch expectation, NMSE(A,B) <= 1e-6, mutant M differs from A; digest-mutation control |
| 46 | fa-bench | ABBA ik (cap huge) vs cv (cap 0), us/call incl. conversion, hs 256 gqa 6 nb 16..512 x kv 8k/32k/90k x 1/2 sequences + hs 128 and gqa 2/4/8 shapes, both GPUs; `fa_q8_bench_summary.py` suggests the cap (largest tile count with ratio <= 1.02 on both GPUs) |
| 47 | reserve-canary | 7 boots of the served J5q np2 shape (`-c 204800 -np 2`, f16 draft KV, MTP): base/new at `-ub 128` and 512, new with cap huge at 128/512, new with draft KV q8_0 (informational); all `compute buffer size` lines in `reserve-*.log.sizes`; gate: new <= base at equal `-ub` |
| 48 | freeze | `wt freeze` -> `frozen/fa-q8-inkernel` |
| 49 | render-and-check-items | renders `codex/items/rendered/fa-q8-depth-j5q.gpuq.toml` and runs `gpuq submit --check` (not submitted) |
| 50 | prefill-bytes-strict | A == B byte identity for every engaged case (runs last: failure falsifies exactness without losing artefacts) |

The cap is not tuned by the window: stage 46 reports the measured crossover; setting `max_query_tiles()` is the follow-up commit.

## Rendered item

`codex/items/fa-q8-depth-j5q.toml.in` -> np1 J5q depth item `fa-q8-depth-j5q` (priority 26): arms `base` (frozen/stack10-cand2) and
`all` (frozen/fa-q8-inkernel with `GGML_CUDA_FA_Q8_QTILES=1000000`, i.e. the in-kernel path at every ubatch), `-ub 512`, depths
8k/32k/90k, plan base/all/all/base (2 boots per arm, interleaved), cross-arm token identity (exactness claim), prefill `prompt_ms`
and decode ms/round. Dry render with `frozen/stack10-cand` substituted for both arms: `gpuq submit --check` valid, harness
`--preflight` ok, `--fixtures` PASS. Real render happens in stage 49.

## Expected (D derived / G guess) and falsifiers

- Reserve at np2 `-ub 128`, main and draft: scratch 845 MiB -> 0 (D from `get_alloc_size`); `-ub 512` unchanged at cap 16 (D).
  Falsifier: new >= base at `-ub 128` in stage 47.
- Kernel: in-kernel no slower than convert for tile counts up to the suggested cap (G: crossover 16-64 tiles; reads of q8 are 1/4 of
  f16, but each tile re-dequantizes the whole view). Falsifier: ratio > 1.02 at <= 8 tiles on either GPU -> the path is only a
  reserve lever, not a speed lever; ratio > 1 at every shape -> report best-effort numbers and stop (brief falsifier).
- Prefill at 90k, uncapped, `-ub 512` (item): -2..-5 % prompt time (G, brief). Falsifier: slower at every depth.
- SASS: new instances at 255 registers with spills like the verify instance (G); verify instances unchanged (D, gate 41).
- Byte identity: see exactness argument (G identical at D=256).

## Uncertainty

Nothing compiled: build errors in stage 40 are possible (templates/macros reviewed by hand; no template commas in macro arguments).
Cap 16 is a guess. GQA 1 is not covered. The D=128 8-byte `cp.async.ca` path has no prior measurement.
