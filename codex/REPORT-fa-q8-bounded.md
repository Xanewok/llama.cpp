# fa-q8-bounded: bounded f16 scratch for q8_0 K/V flash attention (Scope B)

Tree `~/repos/qwen38-eval-worktrees/wt/fa-q8-bounded`, branch `topic/fa-q8-bounded`, base `2efa6f8dc` (the pkg54 fix with the
hoisted `constexpr bool one_row`; it replaces the uncompilable `05ef88569`). Nothing here was compiled or run: this host may
not build CUDA, so the window's stage 40 is the first nvcc pass. Numbers are labelled D (derived) and G (guess); no M.

## Commits

| hash | subject | class |
|---|---|---|
| d64c741bf | cuda : bound the f16 K/V scratch of MMA flash attention by converting long quantized views in windows | general |
| 4b058bcde | local: window size switch, window engagement log and negative-control mutants for the K/V windows | local |
| (codex commit on top) | codex : window, SASS gate, windowed-vs-one-pass NMSE driver and np1 depth item | codex |

## Source changes (general commit)

- `ggml/src/ggml-cuda/fattn-common.cuh`
  - `GGML_CUDA_FATTN_KV_WINDOW` 8192 cells and helpers `ggml_cuda_fattn_kv_n_windows`, `_kv_window`, `_kv_window_max_cells`,
    `ggml_cuda_flash_attn_ext_kv_windowable` (non-sparse, K or V needs the f16 conversion). Windows are multiples of
    FATTN_KQ_STRIDE (256) cells, even in units, at least 2 units; a view of at most 8192 cells (and windows below 1024) is one window.
  - `ggml_cuda_flash_attn_ext_get_f16_extra_data(..., windowed)`: scratch = largest window (K and V, all heads and streams).
  - `launch_fattn`: loop over windows. Per window: shallow `K_win`/`V_win` copies (offset data, `ne[1]` = window) feed the
    existing three conversion paths (contiguous, per-stream dense, nc), the mask pointer is offset by the window start, the
    KV_max scan and the kernel see the window length. The sinks go to window 0 only (they are one more key). The MMA kernel
    is not modified: no new MMA instance exists.
  - Grid of a windowed launch: every block gets fewer than `iter_k` work units (>= ceil(T*k/(k-1)) blocks, rounded to whole
    waves, `GGML_ASSERT(ntiles_KV >= 2)`), so no block normalizes a whole tile in-kernel and every tile is finished by a
    `needs_fixup` block that writes an unnormalized result into the pool buffer `window_part`.
  - `flash_attn_stream_k_fixup_general<D,ncols1,ncols2,windowed>` (new bool; args `acc_ptr, acc_meta_ptr, window_pos`): after
    merging the blocks of a tile as before, the windowed instance merges the window's `(max, sum, VKQ)` with the running result
    in KQV / `window_meta` (one float2 per row) by the same max/rowsum rule and FTZ threshold (-20); the last window divides
    by the row sum. The unwindowed instance is the old code path (`<...,false>` with null args).
- `ggml/src/ggml-cuda/fattn.cu`: `ggml_cuda_flash_attn_ext_get_alloc_size` passes `windowed` (kernel is MMA_F16 and the
  view is windowable) so the graph reserve holds one window; same predicate as the launch, so alloc >= need.
- `tests/test-backend-ops.cpp`: eval cases (q8_0 K/V, K/V-cache layout, CPU reference, NMSE 5e-4): hs256 GQA 6 kv 8192 / 8448 /
  32768 nb 512; kv 20113 (not a multiple of 256) with sinks nb 128; nh 4 kv 90112 nb 128; 2 sequences kv 32768 nb 64 with and
  without a stream view; 2 sequences kv 90112 sinks stream view; hs128 GQA 1/4/8 kv 32768; hs256 GQA 4/8; hs128 ALiBi 8 and
  softcap 10 at kv 16384. Perf cases: hs256 nh4 GQA 6 kv {8192, 32768, 90112} x nb {64, 128, 512}, 1 and 2 sequences (stream
  view), hs128 GQA {1, 4, 8} kv 32768 nb 512.
- Local commit (`fattn-common.cuh` only): env `GGML_CUDA_FA_KV_WINDOW=<cells>` (0 = old one-pass), `GGML_CUDA_FA_KV_WINDOW_LOG=1`
  (per-window `fattn_kv_window: window i/n ...` line), `GGML_CUDA_FA_KV_WINDOW_MUTANT=1..5`: 1 sinks in every window, 2 mask not
  offset, 3 previous windows not rescaled, 4 last window not normalized, 5 windows one unit short. Incidental fixes: none.

## Exactness

- One window (kv view <= 8192 cells, or the switch off): the old code path executes; bytes equal the base. Gated bitwise
  (`fa-window`, kv 8192 case).
- More than one window: NOT bitwise equal to the one-pass path. Reasons: the stream-k partition of a window (more, shorter
  blocks than the one-pass grid) merges in a different order; P is rounded to f16 relative to a window-local running maximum, not
  the global one; the f32 row sums combine as ((w0 + w1) + w2) ... instead of by block. Each window is deterministic, so results
  repeat for the same shapes. Decode after a windowed prefill reads slightly different K/V cache values, so token paths may
  diverge from the base: the depth item gates identity within an arm only (`within-arm`), not across arms.
- Gates: (1) CPU reference NMSE 5e-4 in `test-backend-ops` on the cases above, at window 8192, 1024, 16384; (2) GPU windowed
  vs GPU one-pass NMSE <= 1e-5 (G; expected below 1e-7) on both devices, windows 8192/16384/4096/1024 (`fa-window`);
  (3) must-fail controls: each of mutants 1-5 must FAIL `test-backend-ops` against the CPU and must exceed 1e-5 in `fa-window`.

## Window

`codex/bw-fa-q8-bounded.{toml,sh}` (priority 18, filler, not preemptible, est 6600 s G, budget 16200 s). It needs `frozen/stack10-cand2`
(the output of `bw-s10c-fafix`, priority 21 runs first) as base; preflight does not require it, stage start does. Stages (`stages.tsv`
in `codex/window/fa-q8-bounded-a<N>/`):

| stage | name | content |
|---|---|---|
| 40 | cuda-build | recipe of bw-s10c-fafix, `llama-server` + `test-backend-ops`, `round-trace` copy |
| 41 | sass-gate | `sass_gate.py`: verify instances `<256,256,8|4,8,Q8_0>` equal stack9 and the base; no common kernel changed resources; only the fixup-general instances differ (windowed ones listed with STL/LDL in `sass/new-instances.json`); control = stack10-cand must fail |
| 42 | cuda-stamps | DeviceTopK count equals the base, commit stamp |
| 43 | cuda-fa-ops | full `-o FLASH_ATTN_EXT` on CUDA0/CUDA1 (failures tolerated only if the base fails them too); window cases at 8192 (with engagement log), 1024, 16384, and 0 (no window log) |
| 44 | window-mutants-cpu-gate | mutants 1-5 each must fail the CPU NMSE gate |
| 45 | window-nmse-gpu-vs-gpu | `fa-window` driver, windowed vs one-pass, mutants must exceed 1e-5, single window bitwise |
| 46 | freeze-fa-q8-bounded | `frozen/fa-q8-bounded` |
| 47 | render-and-check-items | `codex/render_fa_q8_bounded_items.py` -> `codex/items/rendered/fa-q8-bounded-depth-j5q.gpuq.toml`, `gpuq submit --check` only |
| 48 | reserve-boots | `swift15-j5q-duo` flags (np 2, -c 204800, q8_0 KV, MTP draft f16 KV), base vs new at -ub 128 and 512: main and draft `compute buffer size` lines (`reserve.json`) |
| 49 | perf-abba | `test-backend-ops perf` per device, window order 0,8192,16384,4096,4096,16384,8192,0; `perf-summary.json`: us/run per case incl. conversion, ratio to one-pass, A/A spread |

Item: `codex/items/fa-q8-bounded-depth-j5q.toml.in` (np1 J5q, base `frozen/stack10-cand2` vs `frozen/fa-q8-bounded`, plan base,bnd,bnd,base, depths
8k/32k/90k, `-ub 512`, prefill `prompt_ms` and decode ms/round, priority 17, canary at 32k requires `fattn_kv_window: window 1/[2-9]`).
Checked in trial mode (`render_fa_q8_bounded_items.py --trial`, stack10-cand as both arms) with `gpuq submit --check` and the item's
`--preflight` (fixtures, negative controls: a recorded log without the window marker fails); the real render runs in stage 47. Not submitted.

## Expected numbers

- Reserve (D): the f16 scratch of K+V is 2 x 102400 cells x 4 heads x 256 x 2 B x 2 = ~839 MiB of the 845.27 MiB. Window 8192: ~67 MiB,
  main buffer ~73 MiB (window 16384: ~140 MiB). Draft with f16 KV: unchanged (42 MiB). Falsifier: stage 48 main > 25 % of base.
  Pool temporaries outside the reserve: partial result + meta, ~ one KQV (12.6 MB at nb 512, np1) plus the fixup buffer (~60 MiB at np2 nb 512).
- Merge cost (D): per window and call one fixup pass of ~3 x |KQV| f32 traffic (~38 MB at nb 512, 24 heads: ~50 us at 800 GB/s) plus ~5
  extra launches: 11 windows at 90k ~0.7 ms per attention layer call. Prefill (G): +0.5...1.5 % at 90k, within +-1 % at 8k/32k
  (one window up to 8192, two at 8448...). Falsifier (brief): merge overhead > 2 % of prefill at 90k, i.e. depth item `prompt_ms` Delta at 90k > +2 %;
  then try window 16384 (halves the merges) before dropping.
- Kernel time (G): FA op with conversion +5...15 % at kv 90112 nb 512 (more waves, fixup), ~0 at kv <= 8192; perf-abba reports per case.

## Risks / uncertainty

- Uncompiled and unrun CUDA code; first compile in stage 40. Reading-only review of the fixup (`__syncthreads()` is block-uniform: all early returns depend on blockIdx only).
- The sparse path, the vec/tile kernels and single-window views take the old path unchanged.
- Window sizes are a knob (`GGML_CUDA_FA_KV_WINDOW` in the local commit); the general commit fixes 8192.
