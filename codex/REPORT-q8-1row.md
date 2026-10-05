# One-row q8_0 flash attention

Source confirms repeated VEC scans per Q head. The candidate reuses the grouped q8_0 loader. CPU checks and CUDA compilation pass; GPU correctness/speed remain unmeasured. Labels: M measured/recorded, D derived, G guess; configured limits are identifiers.

## Diagnosis

TP2 shape: D=256, 12 Q heads, 2 KV heads, GQA=6. Base `98c695111` selects `flash_attn_ext_vec<256,1,Q8_0,Q8_0,false>`. `fattn-vec.cuh:107-111` gives each Q head its own grid-z slice and identical K/V pointers within its group. The K dot loop and V dequantization repeat for each head (`:273`, `:344`). There is no whole-cache conversion on this compiled q8_0-q8_0 path. VEC quantizes Q to q8_1 and uses integer dot products.

M compile resources: 255 registers/thread, 4,352 B static shared memory, no spills. M device limits: 36 SMs, 65,536 registers/SM, 102,400 B shared/SM (`decode-90k.sqlite`, `TARGET_INFO_GPU`). D resource-limited residency is two blocks/SM. D `launch_fattn` selects grid 1x6x12, block 32x4x1 at all three depths; VEC partitions in 128-row iterations (maximum D 11/43/118 iterations/block). D combine grid is 1x12x1, block 256x1x1. No q8_0 launch trace was found.

| KV | D unique q8 KV bytes | D VEC payload scanned | D f16 unique bytes |
|---|---:|---:|---:|
| 8192 | 8,912,896 | 53,477,376 | 16,777,216 |
| 32768 | 35,651,584 | 213,909,504 | 67,108,864 |
| 90112 | 98,041,856 | 588,251,136 | 184,549,376 |

Formula: KV x 2 KV heads x 2 K/V x 8 q8 blocks x 34 B. VEC scans this payload GQA times, excluding repeated scale loads. These are address/payload counts, not DRAM counters: concurrent heads can share L2. Redundant scans and scalar work are the inferred cause; their timing contributions remain unknown.

M historical kernel-suite f16/q8 times are 14.59/51.94 us at 8k and 164.77/218.05 us at 32k (`fixtures/real-fa-perf.log`, provenance in `fixtures/SOURCE`). Repeated 8k inputs fit L2. D recorded end-to-end q8-minus-f16 round deltas are +0.709/+0.471/+1.323 ms (`draft-kv-q8-ab/full-a1/summary.json`); M acceptance changes at 32k from 0.931 to 0.549, so token-rate comparisons there are confounded. `diagnose_draft_attn.py` saves paths and evidence in `draft-attn-diagnosis.json`.

## Fix

`fattn.cu:186`, `:220`, `:722` route eligible single queries to `flash_attn_ext_f16<256,256,1,8,false,false,false,Q8_0>`. One instance reuses cp.async loading/dequantization (`fattn-mma-f16.cuh:507`, `:537`). GQA 6/8 read each KV head once. Larger groups require ceil(GQA/8) scans.

D=128, GQA<=4, ALiBi, softcap, sparse and incompatible layouts retain their paths. No model, GPU-name or context-length gates. `fattn.cu:819` shares eligibility with allocation, avoiding D 369,098,752 B/device of unused f16 storage at 90k. Fallback conversion storage remains available.

M candidate: 255 registers, 4 B spill stores/loads; padded 4x8 had 44/32 B spills. D candidate shared memory is 83,088 B, grid 36x1x1, block 32x4x1; uniform fixup grid 2x1x8, block 256. D peak cold-read floors are 19.89/79.57/218.83 us. G five-pass attention cost is 0.17/0.49/1.23 ms/round, assuming 418 GB/s plus 12 us/pass overhead; six passes multiply by 1.2. G savings against f16 are approximately 0.08/0.35/0.98 ms/round. Cache, spills and dequantization can invalidate these guesses.

## Validation and delivery

M CPU: `CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_q8_cpu.sh`: CUDA OFF, Release, offline, `ggml -j8`. `test_q8_dispatch.py` compiles actual host dispatch/allocation with mocked capabilities: M 30 shapes and fallback gates pass; restoring one-row rejection or f16 scratch fails. The CPU attention helper checks f16/q8 at 8k/32k/90k and 513 rows: M maximum split NMSE 5.62e-11, padded first row bit-identical, dropped-split controls all fail (minimum M NMSE 0.001306). D summed case times 6.45 s on one thread (`cpu-q8-attn-results.txt`). CPU only. Sampling is unchanged; seeded identity is not claimed.

M CUDA compile/freeze: `CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/freeze_q8_1row.sh`, CUDA 13.2, local CCCL v3.4.3, sm_120a, NCCL OFF, Release, -j6, continuous 10 GiB RAM guard. Freeze sources: patch `797465a5c`, base `98c695111`; shared N_THREADS=8 driver. `frozen/{q8-1row,q8-1row-base}` contain round-trace, test-backend-ops, local libs, SOURCE and hashes. No CUDA executable or server was run.

M five harness tests pass (real parsers, serialization, negative gates, exact-PID cleanup). Suite: CUDA-vs-CPU gate, then ABBA on both GPUs; 48 one-row cases and 16 controls. Canary: one boot, same gates, 300 s. Includes hazard/memory/identity/budget gates and raw output. Stage caps: 1200 s; run budget: 1800 s. M `gpuq submit --check` and frozen preflight pass. Existing FA tolerances retained; CPU f16 V drift at 90k is M NMSE 0.001966.

Parent command (not submitted): `gpuq submit /home/xanewok/repos/q8-1row/codex/draft-attn-probe/item.toml`

Open: CUDA canary/correctness, occupancy/traffic, speed and regressions. Do not adopt yet.
