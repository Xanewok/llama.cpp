# Grouped q8_0 one-row eligibility fix

The existing pipelined loader makes short q8 rows expensive: synchronous narrow copies, expanded shared tiles, and (at D=128) register spills. The selector now compares staging work against head reuse. No kernels or sampling code changed. Stack source commit: e7f0447e9. Master source commit: 47cf5e79b; master test commit: 9b574c486. GPU confirmation remains pending; probe7 is prepared for the parent, not submitted.

## Cause and confidence

M probe6 evidence: `codex/runs/q8-probe6/full-a1/{results,correctness}.csv`, both GPUs, interleaved ABBA, M 250 ms performance windows. All q8 correctness rows pass in both arms; f16 failures agree. M median patch/up ratios supplied in the brief, at KV 8192 / 32768 / 90112:

| D | GQA | M patch/up |
|---|---|---|
| 256 | 4 | 0.70 / 0.67 / 0.64 |
| 128 | 2 | 5.1 / 6.8 / 12.1 |
| 128 | 4 | 2.9 / 3.7 / 7.3 |
| 128 | 6 | 2.2 / 2.5 / 5.1 |
| 128 | 8 | 1.8 / 1.9 / 3.9 |
| 64 | 2 | 2.6 / 3.6 / 3.8 |
| 64 | 4 | 1.8 / 2.0 / 1.9 |
| 64 | 6 | 1.35 / 1.45 / 1.35 |
| 64 | 8 | approximately 1.0-1.1 |
| 96 | all tested | approximately 1.04 / 0.86 / 1.8 |

M D=256/GQA4 at long KV: 398 -> 255 us. M D=128/GQA2: 142 -> 1728 us. M D=128 patch time remains approximately 87 / 308 / 1720 us across GQA. M f16 median ratio: 1.000. `codex/fix2-probe6-medians.csv` derives per-shape medians from the recorded elapsed measurements, retaining both arms.

Observed source contracts:

- `ggml/src/ggml-cuda/fattn.cu`, `ggml_cuda_flash_attn_ext_mma_f16`: all tested small widths dispatch the q8 instance with ncols1=1, ncols2=8. One output tile covers a KV head and up to eight query heads; padded heads still cost MMA work. Thus D work and launch geometry are constant throughout tested GQA 1-8 (except D GQA1 VEC where available).
- `fattn-mma-f16.cuh:43-76`, `ggml_cuda_fattn_mma_get_config_ampere`: D all four shapes use 128 threads (four warps), target two blocks/SM, two stages, Q in registers. D KV tile is 128 rows for D64/96/128 and 64 rows for D256.
- `fattn-mma-f16.cuh:501-537`, `flash_attn_ext_f16_load_tile_q8_0`: D raw row bytes are 68/102/136/272. Only D256 divides by 16. D64/96/128 therefore execute a synchronous uint16_t global read and shared write for each two bytes. The multi-stage path still calls cp_async_wait_all, but those raw copies cannot overlap with MMA via cp.async. D256 uses 16-byte asynchronous copies. D raw-copy operation counts per K or V tile are 4352/6528/8704/1088; counting bytes alone missed this instruction amplification.
- `fattn-mma-f16.cuh:546-609`: D96 uses the odd-block converter and unswizzled padded rows. D128 has two block pairs per row and lacks the four-pair lane permutation used for D256. D128's 16-byte dequant stores map two lanes of each eight-lane store wave to the same bank quartet; D256's permutation spreads the wave across all banks. This is a D address mapping, not a measured stall count.
- `fattn-mma-f16.cuh:783-791,1147-1162`: raw K and V share staging storage, with waits/barriers around dequantization. Narrow synchronous staging sits in the tile loop. This is repeated work, not a constant launch-only penalty.
- `fattn-mma-f16.cuh:2257-2266` passes need_f16_K=false, need_f16_V=false to launch_fattn for direct q8. CPU extraction checks confirmed output-only allocation. Whole-view f16 scratch conversion did not remain active on the old direct q8 path.

Reproduction: `CUDA_VISIBLE_DEVICES="" nice -n 19 python3 codex/inspect_q8_resources.py`; the exact nvcc command is saved in `fix2-resource-command.txt`. cuobjdump reads ELF/cubin files only.

M compiler evidence: `codex/logs/fix2-ptxas-original.log` and `fix2-original-resources.txt`, CUDA 13.2 sm_120a; no GPU execution. cuobjdump inspected the exact frozen patch library; a separate nvcc compilation with -Xptxas=-v matched its kernel resources.

| D shape | M registers/thread | M stack bytes | M spill store/load bytes | M static shared bytes | D dynamic shared bytes | D resident blocks/SM |
|---|---|---|---|---|---|---|
| 64 | 255 | 16 | 0 / 0 | 1024 | 41744 | 2 |
| 96 | 255 | 16 | 0 / 0 | 1024 | 66576 | 1 |
| 128 | 255 | 48 | 56 / 64 | 1024 | 83216 | 1 |
| 256 | 255 | 16 | 0 / 0 | 1024 | 83088 | 1 |

D occupancy uses standard sm_120 limits (100 KiB shared, 65536 registers, register allocation rounded to 256 per thread), not a runtime device query. CUDA 13.2 `cuda_occupancy.h:989-1020` includes the 100 KiB sm_120 carveout. D active-warps fractions, assuming 48 warps/SM: 8/48 for D64 and 4/48 for the others. The target in __launch_bounds__ is not actual occupancy; dynamic plus static shared allocation prevents two blocks for D96/128/256. No occupancy API ran.

M static SASS opcode counts in the isolated instances (`codex/logs/fix2-sass-original.txt`):

| D | M LDG instructions | M LDGSTS instructions | M STS instructions | M LDL instructions |
|---|---|---|---|---|
| 64 | 417 | 6 | 540 | 6 |
| 96 | 624 | 6 | 714 | 6 |
| 128 | 825 | 6 | 1068 | 22 |
| 256 | 18 | 114 | 318 | 6 |

These are static counts including multiple tile-loop variants, masks and setup, not dynamic instruction counts. They establish the different generated loading code; they do not assign a percentage of runtime to each mechanism.

D launch geometry (`codex/fix2-launch.csv`): `fattn-common.cuh:1181-1228` queries actual kernel occupancy, enables stream-K on Ada and later, and fills the available SM waves. For the tested two KV heads, one query row, and GQA <= 8, there are two destination tiles. D64 launches 72 blocks across 36 SMs and splits each KV head 36 ways; D96/128/256 launch 36 blocks and split each head 18 ways. D KV iteration ceilings per split at KV 8192/32768/90112: D64 2/8/20; D96/128 4/15/40; D256 8/29/79. All use uniform stream-K fixup, with two destination blocks, not a split count of one. The mask-to-KV-max launch is also present. Resource-based geometry excludes too few blocks as the source explanation, conditional on the stated sm_120 limits.

Inference, confidence high: byte-only predictions failed because this path is instruction/staging limited at short head widths, with D128 spills adding pressure. G: cache misses amplify its long-KV penalty, and shared bank conflicts also contribute. Exact runtime shares, local-memory traffic and cache hit rates need GPU counters; source and static binaries cannot prove them.

D bandwidth accounting correction: D128/GQA2 at KV90112 has 49,020,928 unique q8 K/V bytes for two KV heads. Dividing by M 1728 us gives D 28.4 GB/s, not 57 GB/s. The brief's value implies twice these unique bytes. Neither figure is a DRAM bandwidth measurement. No new GPU speed is claimed.

## Design

`ggml_cuda_flash_attn_ext_mma_q8_0_decode_worthwhile` estimates pipelined staging work in 16-byte copy units. Let R_K/R_V be q8 row bytes, F the expanded f16 K+V row bytes, C_K/C_V the actual raw-copy widths (16 if row aligned, otherwise 2), and H=ceil(GQA/ncols2):

D grouped work proxy = H * [16*(R_K/C_K + R_V/C_V) + F].
D VEC work proxy = GQA * (R_K + R_V).
D conversion fallback proxy = R_K + R_V + (1+H)*F, accounting for raw read, f16 global write and each grouped f16 read.

The gate requires grouped_cost < fallback_cost. Configurations with nstages_target <= 1 keep their prior eligibility; their partial direct converter does not use the raw staging path under investigation. Existing supported-pair, capability, mask, layout, ALiBi, softcap and padding gates remain.

This is a G scheduling heuristic, not a cycle or bandwidth model: it treats the expanded shared store as an equivalent copy unit, omits common mask/fixup work, and does not pretend to know cache hits or occupancy on arbitrary devices. The raw-copy factor comes directly from the implemented loader. The expanded store term makes low head reuse insufficient even with asynchronous staging. There is no model name, GPU name, shape blacklist or context threshold in the new gate. ncols2 follows the existing legal-instance dispatch; keep the helper consistent if those configurations change.

D symmetric pipelined ratios relative to unique q8 bytes: grouped aligned staging costs H*(1+32/17), approximately 2.882*H; narrow staging costs H*(8+32/17), approximately 9.882*H. Since GQA/H cannot exceed eight on these instances, narrow staging cannot beat VEC under this proxy, even at full head-tile fill. This is the explicit reason smaller existing tiles are excluded, rather than an unexplained D==256 whitelist. D96 has no VEC instance and its conversion proxy is cheaper than narrow staging for any number of groups.

| D width(s) | D one-row route after fix | Reason |
|---|---|---|
| 64/64, 128/128 | VEC at every tested GQA | Narrow staging exceeds reuse |
| 96/96 | Existing f16 conversion + MMA | No VEC instance; narrow staging exceeds conversion proxy |
| 256/256 | VEC for GQA 1-2; direct q8 grouped MMA for GQA >= 3 | Pipelined aligned staging amortizes above proxy 2.882 per group |
| 192/128 | Existing f16 conversion + MMA | Narrow pipelined rows; no VEC instance |
| 320/256, 512/512, 576/512 | Prior direct q8 route, subject to existing gates | Single-stage partial converters unchanged |

D256/GQA3 is covered by CPU dispatch, but is outside the requested GPU matrix. With multiple groups the exact inequality is applied, rather than a minimum-GQA rule. Multi-query routes are unchanged. Sampling code and RNG state are untouched. Restored fallback routes execute the existing base kernels; retained grouped routes execute the same kernels that passed probe6. No new token-equivalence claim is made for VEC versus half-accumulating MMA.

## Checks and artifacts

M CPU command: CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_q8_general_cpu.sh. M 210 actual-selector dispatch shapes pass, including allocation and fallback gates. Both stack and master source functions pass (`fix2-{stack,master}-dispatch.log`). M six negative controls fail as intended: disabled dispatch, unwanted f16 scratch, bypassed cost gate, lost partial-load offset, unbounded sinks, integer-cast half scales. M 16 loader/swizzle cases preserve every half value. M eight attention split/reference cases pass, max split NMSE 5.61224804e-11; padding is bit-identical and every dropped-split control fails. D sum of measured CPU case times: 6.188 s, one thread (`codex/fix2-attn-cpu.txt`). This selector-only fix does not warrant an MTP model sampling run; GPU/server execution was forbidden.

M probe7 unit checks: CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 nice -n 19 python3 -m unittest discover -s codex/q8-probe7 -p test_probe.py. M 13 tests pass, including full unmodified probe6 test/perf recordings, identity and bad-commit controls, metric gates, timeout exact-PID termination, and real recorded device/memory/hazard formats. Fixture provenance and hashes are in `codex/q8-probe7/fixtures/SOURCE`. The no-GPU test refuses before any GPU command. No canary or performance run was executed.

M CUDA stack build passed: `nice -n 19 cmake -B codex/build-fix2 -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3 -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON`, then `cmake --build codex/build-fix2 --target llama-server -j8` and `--target test-backend-ops -j8`. All commands had CUDA_VISIBLE_DEVICES="" and GIT_NO_LAZY_FETCH=1. CUDA architecture resolves to sm_120a in existing CMake logic. Compiler detection was forced to avoid CUDA device probes. No built CUDA executable ran. Logs: `fix2-{configure,build,test-driver-build}.log`.

M master CUDA selector compilation passed (`fix2-master-nvcc.log`), against the existing master snapshot headers. This compiles the changed translation unit only; a new complete master binary was not frozen. The full stack CUDA build covers the identical new helper. M exact frozen fix2 kernel resources match the original instances (`fix2-resource-summary.txt`); kernel code was not changed.

M freeze passed: `QWEN38_WT_ROOT=$PWD/codex nice -n 19 ~/repos/qwen38-eval-worktrees/bin/wt freeze "$PWD" q8-general-fix2 --build "$PWD/codex/build-fix2" --desc ...`. SOURCE is the full e7f0447e9 commit, matching enclosing HEAD at configure/freeze. M strings checks on libggml-base.so and its aliases find e7f0447e9 (`fix2-stamp.txt`). Frozen hashes and $ORIGIN library resolution pass. The later evidence commit changes codex files only and does not alter this source identity.

M `OUT=$PWD/codex/runs/q8-probe7/preflight nice -n 19 bash codex/q8-probe7/run.sh --preflight` and `nice -n 19 gpuq submit --check "$PWD/codex/q8-probe7/item.toml"` pass. Nothing was submitted. Item id: q8-general-fa-ab7. Base arm: q8-general-base2; patch arm: q8-general-fix2. D full matrix remains 150 valid cases, 30 unsupported q8 widths and 16 multi-query controls; both GPUs, ABBA. D perf window is 250 ms. D budget is 1800 s across canary and run; run cap 1500 s, canary cap 280 s within queue timeout 300 s. The kernel-suite canary needs one paired up/patch cycle, with ten cases plus controls; no server boots occur. The parent queues the manifest, not the convenience wrapper.

D maximum probe6 SQL timestamp spans are test 29 s, controls 1 s and perf 74 s per arm; summed across eight arm suites gives 832 s before process startup and guards. G full estimate remains 1000 s, below D 0.7*1500=1050 s. The stage timeout caps are safety ceilings, not added runtime estimates. Recordings lack process-start timestamps, so no exact full-runtime claim is made.

The root and local cumulative package patches are refreshed against their declared bases. Application and byte-identity checks are logged in `fix2-package-check.log`. No PR text, comments, network operations, queue submission or GPU work occurred.

## G predictions and residual risks

| Shape group | G patch/up at requested KV lengths | Basis / uncertainty |
|---|---|---|
| D64 and D128, all tested GQA | approximately 1.00, plausible 0.96-1.04 | Same VEC route as base; no staging kernel runs |
| D96, all tested GQA | approximately 1.00, plausible 0.96-1.04 | Same f16 conversion route as base; removes the old patch's long-KV regression |
| D256/GQA1-2 | approximately 1.00, plausible 0.96-1.04 | Restores base VEC for GQA2; GQA1 was already VEC |
| D256/GQA4 | approximately 0.70 / 0.67 / 0.64 | Same retained kernel as probe6; repeat A/B needed |
| D256/GQA6-8 | approximately 1.00, plausible 0.96-1.04 | Same retained kernel as probe6 |
| D256/GQA3 or untested multiple groups | Unknown | CPU dispatch only; proxy does not establish speed |
| Large/asymmetric heads | Unknown | Small-pipelined regression scope only; no new GPU correctness evidence |
| f16 controls | approximately 1.00 | Selector is q8-only |

The main risk is heuristic portability: shared-memory bandwidth, copy issue rate, spills and cached VEC rereads differ across architectures. The model is deliberately conservative for the demonstrated scalar staging regression. GQA3 crossover and wide-head GPU correctness remain open. A faster dense raw loader could eventually admit small heads; it requires a separate kernel change and GPU evidence. Existing f16 numerical failures remain reportable and must agree across arms. Upstream issue/duplicate checks were not performed because network access is forbidden.
