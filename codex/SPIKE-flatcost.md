# SPIKE flatcost: why the cp.async ring loses bandwidth, and a tensor-core few-column kernel (2026-10-08)

Labels: M measured, D derived, G guess. Branch `topic/flatcost-v2` in `wt/flatcost-v2`, rebased off the ring onto stack6
`ac7465885`. Unless noted otherwise, perf numbers are from `/mnt/scratch/work/flatcost-q6k-perf/a1/perf.md` (M, cold L2).

## Conclusion

1. **The ring is bound inside the SM, not by DRAM.** Its time is almost the same with the weights in L2 and in DRAM, at a
   flat ~265 GB/s for J=16 and ~300–317 GB/s for J=8 on every shape (M). The leading cause (D) is the copy path: 4-byte
   `cp.async.ca` copies issued by only 4 warps per SM, serialized with the compute of the same warps, through a ~28 KB L1
   beside a 100 KB shared-memory carveout. The I=64 config itself (1 block, 4 warps per SM) costs another 15–17% on its
   own (M). The ncu probe is ready to confirm it, but it is blocked on a driver option (see Blockers).
2. **New kernel `mmvq-mma`, implemented** in general upstreamable form: q6_K and iq4_xs at up to 16 columns on int8 mma.
   Each lane decodes its own fragment bytes straight from VRAM. There is no shared-memory tile and no barrier in the k
   loop. It sits behind ggml's normal dispatch with no env var: it takes the widths where stock ggml uses MMQ (above the
   per-arch MMVQ cut, up to 16 columns). The env switches and a tuned 5-column cut are a separate local commit.
   Not compiled yet: CUDA can only be built in a build window, and that window is prepared.
3. **Ceiling (D):**
   - At 400 GB/s cold, q6_K+iq4_xs per GPU per round:
     - J5q np2 (n=12): q6_K 14.07 → 11.58 ms (−2.5 ms).
     - IQ4 np2: iq4_xs ~18.4 → 15.1 ms (−3.3 ms).
     - np1 (n=6): J5q q6_K −1.25 ms, but only with the tuned cut, because the general rule leaves n ≤ 7 to MMVQ.
   - **Main risk (D/G): L1 wavefronts.** A fragment load covers 8 rows (or 8 columns) per instruction. My model puts q6_K
     at 2 tiles (n = 9–16) at ~0.27 L1 wavefronts/B. At 1 wavefront/clk per SM, that caps it near ~340 GB/s, about stock
     MMQ's 333 GB/s, so there would be no win. iq4_xs at 2 tiles (~0.25 wavefronts/B → ~370 GB/s) and every 1-tile case
     (≤ 0.21 wavefronts/B) stay under the cap. The perf item's ncu stage measures exactly this. The fixes are listed
     under Next steps.

## Findings: why the ring lowers achieved bandwidth (ranked)

Code: `wt/flatcost-smallbatch/ggml/src/ggml-cuda/mmq.cuh:968-1020`. The loop issues stage st+1 (`cp_async_ca_4`, l.994),
then `wait_group<1>` (l.1008), then `__syncthreads` (l.1012), then `load_tiles` from the stage (l.1017), then `dot_iter`.
Config: `mmq-config-blackwell.cuh:4-7` (128 threads, I=64, J 8|16, 2 blocks/stage). Smem per block ≈ 76 KB (tile_x
19.5 KB + 2 × 26.9 KB stages). That gives 1 block, 4 warps per SM. The `pipe0` arm keeps the same allocation.

Measured facts that discriminate between the hypotheses:
- **F1 (M):** the ring's time barely changes between warm L2 and cold. On 3072×5120 n=6 it is 41.7 µs warm vs 48.9 µs
  cold, and the bandwidth is flat across shapes. pipe0 does gain from L2 (24.8 µs warm). So the ring is not latency- or
  DRAM-bound.
- **F2 (M):** ring − pipe0 is about +17 µs on 3072×5120 for both J=8 and J=16 (≈ +1.3 µs per 2-block stage). The cost
  scales per weight byte (the copy path), not with MMA work or column count.
- **F3 (M):** pipe0 (4 warps/SM) is 15–17% slower than stock MMQ (8 warps/SM) on the big shapes:
  5120×8704 n=12 is 123.3 vs 107.3 µs hot.
- **SASS (M):** `cuobjdump` of the frozen libggml-cuda sm_120a, Q6_K MMQ.
  - The ring runs ~2050 dynamic instructions per 2 blocks vs ~1860 for pipe0, so instruction count does not explain it.
  - Each warp issues 64 `LDGSTS.E` (4 B) per stage, each with an ISETP/BRA guard, then `LDGDEPBAR` and
    `DEPBAR.LE SB0,0x1`.
  - Registers: J=16 has 255 regs with an 8 B stack at fallback 0, 255 regs with no stack at fallback 1. J=8 fallback 0
    has 206 regs.

| # | Hypothesis | Evidence for / against | What the ncu probe should show |
|---|---|---|---|
| H1 | Copy path: 4-byte `cp.async.ca` allocates in a ~28 KB L1 (100 KB carveout) while a stage is 26.9 KB (~35 KB of lines at a 4200 B row pitch); copy issue back-pressures (LSU/L1 allocation) and, with 1 warp per SMSP, serializes with that warp's compute | F1, F2 (per-byte, J-independent cost), SASS issue loop; pipe0 avoids it | ring: `stall_lg_throttle` and long-scoreboard share at the LDGSTS/DEPBAR PCs well above pipe0; LDGSTS L1 lookups ~all miss; launch shared-mem config 100 KB |
| H2 | Occupancy: 4 warps/SM, nothing hides LDS→IMMA dependencies or barriers | F3 (pipe0 −15–17% vs stock); shared by ring and pipe0, so it is not the ring delta | warps active ~8% of peak (4/48), achieved occupancy ≈ theoretical; issue active low in both ring and pipe0 |
| H3 | Barriers per stage (1 block/SM idles at each `__syncthreads`) | Present in pipe0 as well; adds to H2 | barrier stall share similar in ring and pipe0 |
| H4 | Smem bank conflicts in `load_tiles_q6_K` (`mmq-load-tiles.cuh:946`) | D: ql/qh reads are consecutive within a row; the stage row pitch is 105 words (odd), so the `d` reads are conflict-free; scales show a minor 2-way conflict on 3 banks | shared bank conflicts per LDS wavefront < 0.1 |
| H5 | J=16 register spill | Only an 8 B stack at fallback 0, and the ring−pipe0 delta is the same at J=8 (206 regs, no spill) | local-memory traffic ≈ 0 |
| H6 | Pipeline depth (1 stage ahead) | 27 KB in flight per SM vs a Little's-law need of ~8 KB (11 GB/s per SM × ~0.7 µs, D); and the same bound holds when data comes from L2 (F1) | DRAM throughput well below peak with no long-scoreboard wait on the DEPBAR |
| H7 | Stream-k split / fixup, L2 | Shared with pipe0; the warm/cold invariance excludes L2 | fixup kernel 1–2 µs, L2 hit rate irrelevant |

Ruled out or minor: H4, H5, H6, H7 (D). H1 explains the ring's own loss and H2 the loss of its I=64 config.
Neither can be fixed inside MMQ's design without giving up the smem tile, and that motivates the redesign below.

## Design: mmvq-mma (Marlin-style, general form)

Files: `ggml/src/ggml-cuda/mmvq-mma.cu`, `mmvq-mma.cuh`; dispatch hook at `ggml-cuda.cu:2028`.
- **Work split:**
  - Block = 4 warps on the same 16 rows. Warp w takes src0 blocks kb = w, w+4, … (K split).
  - At the end, partial sums are reduced through 1.5–3 KB of smem (`mmvq-mma.cu:209`).
  - Grid (⌈rows/16⌉, channels, samples), with channel/sample broadcast as in MMVQ.
  - `__launch_bounds__(128, 4)` (l.170): ≥ 16 warps/SM.
- **Data path:**
  - Lane (g, t) loads exactly the src0 bytes of its own mma A fragment (rows g and g+8) from VRAM: `get_int_b2` for
    2-byte-aligned q6_K, `get_int_b4` for iq4_xs.
  - The next block is prefetched into registers while the current one is multiplied. The decode stays in registers.
    There is no smem tile and no barrier in the k loop.
  - Each src0 byte is read once, whatever the column count.
- **MMA:**
  - q6_K: m16n8k16 per 16-value sub-block (one scale per sub-block; l.75).
    - ql int 16h+4s4+t, nibble (sb/4)%2; qh int 8h+4(sb%2)+t, shift 2((sb%8)/2).
    - Value = (ql | qh<<4) − 32 via `__vsub4`.
  - iq4_xs: m16n8k32 per 32-value sub-block (l.121). qs int 4ib+t gives low nibbles = k-int t and high nibbles =
    k-int 4+t, which is exactly the PTX A layout; decoded with `get_int_from_table_16`.
  - Epilogue per sub-block: `acc += float(C) · d_row·scale · d_col` (src1 is MMVQ's q8_1, so activation rounding is
    unchanged).
- **Columns:**
  - 1 or 2 n8 tiles (≤ 8 / ≤ 16 columns). Columns ≥ ncols are loaded as zero and their writes skipped.
  - Rows are clamped for loads and their writes skipped.
- **Dispatch (commit A, no env var):**
  - Order: `should_use_mmvq` → `ggml_cuda_should_use_mmvq_mma` (l.271) → MMQ.
  - Conditions: Turing+ NVIDIA, q6_K or iq4_xs (iq4_xs also needs 4-byte-aligned data/strides), ≤ 16 columns, plain
    MUL_MAT.
  - It therefore replaces MMQ exactly where ggml already decided MMVQ is too slow. The per-arch MMVQ cuts are untouched.
    No threshold is tuned to the 5060 Ti.
- **Local commit B:**
  - `GGML_CUDA_MMVQ_MMA=0` keeps the old path.
  - `GGML_CUDA_MMVQ_MMA_MIN=N` hands q6_K/iq4_xs from N columns to mmvq-mma below the MMVQ cut (the tuned candidate is
    5; see `mmvq.cu:329`).
  - `GGML_CUDA_DEBUG_MMVQ_MMA` logs launches.
  - test-backend-ops `GGML_TEST_FLATCOST=1|q6_k|iq4_xs` and `GGML_TEST_FLATCOST_COLD` (from the ring study).
  - A tuned-cut commit (e.g. q6_K MMVQ cut ≤ 4 on Blackwell) should be added only after min5 wins in the perf item.

Ceiling detail (D, per GPU, V phase):
- q6_K bytes per round:
  - J5q: 4631 MB (floor 11.58 ms at 400 GB/s; 12.52 ms at 370).
  - IQ4 model: 988 MB.
- iq4_xs bytes per round:
  - IQ4 model: 6032 MB (15.08 ms at 400; 16.30 ms at 370).
  - J5q: 393 MB.
- Stock cold at n=12:
  - J5q q6_K: 14.07 ms.
  - IQ4 iq4_xs: ~18.4 ms (target.py baseline).
- Stock cold GB/s (5120×8704):
  - n=12 (MMQ): 333.
  - n=6 (MMVQ): 367.
  - n=1: 409.
- Break-even for reopening k=6/7: ≈ 2 ms/round (verify rows 7–8). That needs the q6_K 2-tile path at ≥ 370 GB/s.

## Commits (`topic/flatcost-v2`)

| commit | form | content |
|---|---|---|
| `cdb265be1` | general (upstream candidate) | cuda : few-column quantized matmul on int8 tensor cores (q6_K, iq4_xs). New `mmvq-mma.{cu,cuh}`, dispatch hook, test-backend-ops cases (row tail, 1 and 3 blocks per row, channel/sample broadcast, strided src0 view, permuted operands; n = 1–16). Applies cleanly on stack6 (`git merge-tree` checked) |
| `9327428d8` | local | knobs (`GGML_CUDA_MMVQ_MMA`, `_MIN`, `DEBUG_MMVQ_MMA`) + `GGML_TEST_FLATCOST` served shapes with iq4_xs, cold-L2 rotation |
| `3ad1e8b4e` | codex | build-window payload `codex/bw-flatcost-v2.sh`, item templates `codex/flatcost-v2-{memcheck,perf}.toml` |

The ring (`topic/flatcost-smallbatch` e21e8fa7b) is not in this branch. It is slower than stock at every width ≥ 5 (M) and
should be dropped.

## Mac CPU evidence

See the "Mac CPU result" line at the end. It covers the clang build of test-backend-ops, the CPU test mode on all
q6_K/iq4_xs MUL_MAT eval cases (graph build + execution of the new cases, CPU vs CPU), and the support-mode count of the
`GGML_TEST_FLATCOST` lists. CUDA cannot be built there.

## Window manifest and items (none submitted)

| path | kind / priority | expected (G/D) |
|---|---|---|
| `work/build-window-1007/bw-flatcost-v2.toml` → payload `wt/flatcost-v2/codex/bw-flatcost-v2.sh` | filler / 1 | Steps: CUDA build (stack6 RECIPE flags, tests on), CCCL check, mmvq-mma register report, freeze `frozen/flatcost-v2`, base arm `work/flatcost-v2/base-bin`, copy + `gpuq submit --check` of both items. `gpuq submit --check` valid; payload `--preflight` rc 0 |
| `orchestration/gpuq/items/drafts/flatcost-ncu-q6k.toml` (`work/flatcost-ncu/run.sh`, `analyze.py`) | measure / **22**, CCX0+1 | 24 ncu reports (stack6 MMVQ/MMQ, ring, pipe0) on ssm_out 5120×3072 and ffn_down 5120×8704 at n {1,5,6,12}, plus a q5_K unchanged-path control (±25%). Expected: H1 signature in ring vs pipe0, H2 in both. `submit --check` valid. **Preflight refuses (rc 6) until profiling is allowed** |
| `drafts/flatcost-v2-memcheck.toml` (rendered by the window from `codex/`) | measure / 21 | compute-sanitizer on q6_K/iq4_xs eval cases (default dispatch and min1 = every width 1–16) and served shapes; plain MUL_MAT/MUL_MAT_ID all types vs base; engagement controls; VERDICT gates perf. `submit --check` valid (template) |
| `drafts/flatcost-v2-perf.toml` (`work/flatcost-v2/run.sh`, `analyze.py`) | measure / 20, CCX0+1 | Arms: hot b,p,off,p,b; cold b,p,min5 at n 4–16; ms/round for J5q/IQ4; optional ncu of the new kernel. Negative control: off vs base, median ≤ 3% or rc 6. Engagement: default dispatch hits q6_K n≥8 and iq4_xs n≥9, never n=4/17; min5 moves the cut. Expected: iq4_xs n 9–16 −10…−20% cold; q6_K 2-tile ±5% if L1-bound; 1-tile cases near 380–400 GB/s. `submit --check` valid (template) |

Order:
1. bw-flatcost-v2 (filler).
2. flatcost-v2-memcheck.
3. flatcost-v2-perf.
4. flatcost-ncu-q6k, independent, once the profiling option is set.

## Blockers

- **ncu needs counter access:** `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`. Fix (owner, sudo): add
  `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to `/etc/modprobe.d/`, then reload the driver. Until then,
  `flatcost-ncu-q6k` fails its preflight with rc 6 and the perf item skips its ncu stage (recorded, not a failure).
- The kernel is uncompiled. It was written against `mma.cuh` (int8 m16n8k16/k32 wrappers, PTX register order) and
  `vecdotq.cuh`; the first build window is its first compile.

## Risks

- **L1 wavefronts (D/G, above):** q6_K's 2-byte loads (2 × `LDG.U16` per int, 8 rows per instruction) and the
  activation fragment loads (8 columns per instruction plus 2 ds loads per sub-block) may bind before DRAM at 2 tiles.
- **Wave quantization:** 16 rows per block and up to 4 blocks per SM. 5120 rows → 320 blocks = 2.2 waves on 36 SMs, and
  3072 rows → 1.3 waves. That is a possible tail loss on the small shapes (D).
- **Registers:** the q6_K block struct is ×2 (current + prefetch, ~68 regs). The launch bound caps at 128; the window
  reports the spill.
- **Upstream:** a reviewer will want numbers from more than one GPU, and a check for overlap with in-flight upstream
  work (dupcheck before any post). The general rule defers to the existing MMVQ cuts, so it changes nothing below them.

## Next steps

1. Owner: enable profiling counters (driver option above). Parent: run the window, then memcheck, then perf, plus the ncu
   probe.
2. If perf shows q6_K 2-tile L1-bound (ncu `l1tex` throughput near peak, lgds wavefronts/KB ≈ model), fix in this order:
   - (a) aligned `LDG.32` + quad `__shfl` + `__funnelshift_r` for misaligned q6_K blocks (weights 528 → ~320 wavefronts
     per 16-row block);
   - (b) 32 rows per warp without the register prefetch (halves activation wavefronts per weight byte).
   Re-measure after each.
3. If min5 beats MMVQ at n = 5–8 cold, add a local commit with the tuned cut (Blackwell q6_K/iq4_xs), then re-run the
   J5q/IQ4 depth items with the serving env from `frozen/stack7`.

Mac CPU result (M, `bin/bw-mac` on HEAD 3ad1e8b4e, output `work/build-window-1007/flatcost-v2-mac-a1/`): rc 0.
Apple clang 21 build of test-backend-ops: the only warning is pre-existing (ggml-round-trace.cpp syscall).
`test -o MUL_MAT -b CPU -p type_a=(q6_K|iq4_xs),`: 173 cases built, ran and passed (CPU vs CPU, so no kernel numerics).
Support mode: `GGML_TEST_FLATCOST=1` gives 260 cases; `=q6_k` and `=iq4_xs` give 137 each, as expected.
No model run, so there was no RAM wait.
