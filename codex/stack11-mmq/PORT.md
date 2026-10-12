# stack11-mmq: K8/L1 MMQ small-I port onto master

Branch `topic/stack11-mmq`, base 5514f5f5d (topic/stack11-base = upstream abee0c847 + our series).
Code-only port, nothing compiled or run on this host.

## Commit map (stack10 fork, frozen/stack10 a588f2a44 -> stack11-mmq)

| stack10 | stack11-mmq | subject |
|---|---|---|
| 015a0cbc7 | cae982c0d | cuda : shape-gated small-I MMQ tiles for 16 and 24 columns on Blackwell |
| 0ab5b19e7 | afd162030 | tests : cover MMQ small-column tile selection (clean pick) |
| 3f0814173 | 7cce77c77 | cuda : MMQ stream-k partitions in default-tile units for narrow tiles |
| c5baada6d | 8f709e567 | cuda : take SMALL_I MMQ tiles on stream-k launches too (clean pick) |
| 9ccfb56e0 | c85745f34 | tests : cover MMQ small-I tiles on stream-k shapes (clean pick) |
| 7491039b9 | ad36da5da | local: K8/L1 MMQ tile A/B switch, auto-drop and stream-k partition mutants, mmq-hash tool |

The MMQ files do not change between 7491039b9 and a588f2a44, so a588f2a44's MMQ is 7491039b9's.
No codex/ files came in with the picks.

## Adaptations

- `tiles` threading (015a0cbc7): rebuilt on master's text, not merged hunk by hunk. Master files plus a scripted
  insertion of `ggml_cuda_mmq_tiles tiles` after `fallback` in every template header, template argument list,
  getter call and getter definition. Order: device `(type, J, fallback, tiles, prec_src1)`, host
  `(type, J, fallback, tiles, cc, prec_src1)`. Master's explicit `prec_src1` arguments (#30168) are kept:
  `GGML_PREC_Q8` in the Q8 load/dot paths, `GGML_PREC_Q4` in the fp4 dot, and no defaults on any getter.
- J_best selection (#30168 moved it from `mul_mat_q_switch_J` to host `ggml_cuda_mul_mat_q`): the J loop
  queries `GGML_CUDA_MMQ_TILES_DEFAULT` at mmq.cu:208. The small-I decision stays in `mul_mat_q_switch_J`
  and uses `args.J_best` (mmq.cuh:1617).
- `ggml_cuda_mmq_get_J_max` is gone on master (#29953), so stack10's edit of it is dropped.
- K8 `sk_group` / `mmq_get_stream_k_plan` (3f0814173): the conflicts were context only. The getter takes
  `prec_src1` without a default. The kernel and fixup hunks merged cleanly onto master.
- local (7491039b9): one context conflict (`switch (J_best)` became `switch (args.J_best)`).
- Kept from master: the NVFP4 in-place accumulation (#29857, `ggml_cuda_mmq_vec_dot_fp4_fp4_mma`, takes
  `tiles` only), the src1 padding and the `ids_dst` padding from the OOB fix (#29953, mmq.cu:226-233 and
  `ne_get_rows + J_best-1`), the n_expert padding fix (#29941), and `ggml_cuda_mmq_needs_fallback`.

Mechanical check (`cmp.py`). Both change sets were normalized by deleting the `prec_src1` parameters and
arguments. Then (015a0cbc7^ -> 7491039b9) was compared with (5514f5f5d -> ad36da5da):
- mmq-config-blackwell.cuh: 15/15 changed lines, identical.
- mmq-load-tiles.cuh: 284/284, identical.
- mmq-vec-dot.cuh: 116/116. One line differs only by master's extra `GGML_PREC_Q4` argument.
- mmq.cuh: 761 vs 757. The only differences are the J loop and J_max moving out (above) and `J_best` ->
  `args.J_best` (2 lines).
- mmq.cu: +1/-1, the J loop `TILES_DEFAULT` argument.
- test-backend-ops.cpp: identical. The only extra stack10 delta in that range is K10's TOPK_CANONICAL_ANY,
  which is not part of this series.

## Selection equality on sm_120 (static)

The served weights are q8_0, q4_K, q5_K, q6_K and iq4_xs, at ne11 9..32, on stream-k and non-stream-k
launches. On both sides (a588f2a44 vs HEAD), every selection input is computed by identical code from the
same values:

| step | a588f2a44 | stack11-mmq |
|---|---|---|
| fallback (`ne01 % 128 != 0`) | mmq.cu:180 | mmq.cu:182 (`ggml_cuda_mmq_needs_fallback`, same expression) |
| prec_src1 | mmq.cu:182 <- `ggml_cuda_mmq_get_prec_src1` mmq.cu:119 | mmq.cu:184 <- mmq.cu:118 (unchanged function) |
| ncols_opt | dense `ne1` (mmq.cu:233 `ne1, ne1`), MoE `ne12` (mmq.cu:308; RDNA only: per-expert) | dense `ne11` == ne1 (mmq.cu:193), MoE `ne12` (mmq.cu:196; RDNA only: per-expert at :201) |
| J_best loop (DEFAULT table, smpbo, min tiles, smallest J) | mmq.cuh:1631-1647 | mmq.cu:205-223 |
| SMALL_I gate (`ggml_cuda_mmq_use_small_i`) | mmq.cuh:1589, called with `J_best` at :1649 | mmq.cuh:1582, called with `args.J_best` at :1617 |
| stream-k plan / partition unit (`mmq_get_stream_k_plan`) | mmq.cuh:1462, launcher :1530 | mmq.cuh:1455, launcher :1523 |
| tile tables (default and small-I) | mmq-config-*.cuh | byte-identical (no master change in mmq-config-*) |

Result: mmq_x (J), mmq_y (I), the SMALL_I path and the stream-k unit are equal for every shape.
- n=9..16 gives J=16 and n=17..24 gives J=24. For iq4_xs/q4_K/q5_K/q6_K with ne01 % 128 == 0, these take
  SMALL_I (I=64, 128 threads, occupancy 2, sk_group 2) in auto mode, on plain tiling and on stream-k.
- n=25..32 gives J=32 with the default tile (I=128).
- q8_0 always takes the default tile.
- Dispatch into MMQ at these widths is unchanged too: MMVQ_MAX_BATCH_SIZE is 8 on both sides, and
  `ggml_cuda_should_use_mmq` is untouched.

Where the src1-precision parameter meets the selection:
- The host SMALL_I branch of `ggml_cuda_mmq_get_config` (mmq.cuh:242-245, both sides) and the device
  branch (mmq.cuh:282-289) return the small-I table without reading `prec_src1`.
- `prec_src1` is `GGML_PREC_Q4` only for MXFP4/NVFP4 on Blackwell (mmq.cu:127-132). Neither type has a
  small-I entry (mmq-config-blackwell.cuh:39-51), so Q4 and SMALL_I cannot meet.
- The port passes `prec_src1` explicitly in four places: the J loop (mmq.cu:208), the default config in
  `use_small_i` (mmq.cuh:1598), the plan (mmq.cuh:1523), and `ggml_cuda_mmq_get_sk_group` (device).
  In each place it equals the value stack10 got through the old default (Q8 for every served type).
- New on master: the src1 padding (mmq.cu:226-233) is sized from the default J_best config's nthreads (256).
  A SMALL_I tile loads with 128 threads, so its src1 over-read (J*MMQ_TILE_Y_K rounded up to 128 ints) is a
  prefix of the default tile's (rounded up to 256 ints), which the padding already covers. The selection
  does not change, and no extra padding is needed.

## Other arches and shapes: the gate stays off

Checked by reading the code. A launch takes SMALL_I only if all of these hold:
1. Host: `ggml_cuda_mmq_get_config(..., SMALL_I, cc, ...)` returns `ggml_cuda_mmq_config_none()` unless
   `blackwell_mma_available(cc)` (NVIDIA and 1200 <= highest compiled arch < 1300). The table then matches
   only `cc == GGML_CUDA_CC_BLACKWELL` (1200, the highest compiled arch). AMD, MUSA, pre-Blackwell, sm_121
   (DGX Spark) builds and Rubin get `GGML_TYPE_COUNT`, so `use_small_i` returns false.
2. Device: the SMALL_I branch is compiled only under `!GGML_USE_HIP && BLACKWELL_MMA_AVAILABLE` with
   `__CUDA_ARCH__ == 1200`. Everything else gets `config_none`, and the kernel body is `NO_DEVICE_CODE`.
3. Shapes: table entries exist only for fallback=false (ne01 % 128 == 0), J in {16, 24}, and the types
   IQ4_XS/Q4_K/Q5_K/Q6_K. `if constexpr (ggml_cuda_mmq_has_small_i(type))` keeps the SMALL_I launch from
   being instantiated for other types. `use_small_i` also requires the smem fit and ne01 % default I == 0.

When the gate is off, the code path is master's: J_best from mmq.cu, the DEFAULT table, and
`mmq_get_stream_k_plan` with group = 1. That plan gives the same `nparts`/`ntiles`/fixup as master's
inline `tiles_efficiency_percent` code: same formula, now in `mmq_tiling_efficient`.

## GPU window filter (test-backend-ops)

Eval, small-I and stream-k shapes (MUL_MAT and MUL_MAT_ID cases from afd162030 and c85745f34):

    test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID \
      -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,(n_mats=4,n_used=2,b=0,)?m=(3072|5120|6144|8704|9216|124160),n=(9|12|16|17|24|25|48),k=(256|512|1024),'

Run three arms:
- default (unset, i.e. auto).
- `GGML_CUDA_MMQ_TILES=default`: the local switch's off arm, never SMALL_I. It must pass.
- `GGML_CUDA_MMQ_TILES=auto-drop`: negative control. SMALL_I launches are skipped, so the engaged shapes must fail.

Add `GGML_CUDA_MMQ_TILES_LOG=1` to one auto run to confirm engagement (`small_i=1`, `units`/`parts`/`blocks`).

Bitwise equality: `test-backend-ops mmq-hash CUDA0` under `GGML_CUDA_MMQ_TILES=default` and under auto. The
hashes must match line for line. With `GGML_CUDA_MMQ_SK_MUTANT=1`, the stream-k shapes (e.g. m=3072/5120/6144
at n=12/24, id_*) must differ: negative control.

Perf: `test-backend-ops perf -b CUDA0 -o MUL_MAT -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704|9216|17408|124160),n=(6|9|12|16|17|24),'`
(optional: `GGML_TEST_PERF_COLD_MB=512`).

## Not ported / open

- Nothing left out of the six commits.
- Not compiled. Every getter now takes `prec_src1` explicitly, so a missed `tiles` argument fails to
  compile instead of silently changing a tile. The build is the first check.
- `patches/upstream/llama.cpp/80-mmq-small-i-tiles` root form is based on master 7049ff0cb. This branch is the
  form for master abee0c847 (post #30168/#29953/#29857/#29941).
