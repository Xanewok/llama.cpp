# One-row draft attention

Keep this as an A/B candidate; GPU correctness and speed are unmeasured. The recorded llama kernel is primarily bandwidth-bound, and the HQ comparison uses a different KV dtype and forward count.

Labels: M recorded/measured; D arithmetic or source-derived; G unmeasured prediction. Shape constants, commit IDs, flags and configured limits are identifiers.

## Diagnosis

Global geometry is D=256, Q heads=24, KV heads=4; TP2 gives Q heads=12, KV heads=2, GQA=6 per device (`~/repos/attrib-pack/models/llama-swift15-iq4xs.json`). The base dispatch selects MMA `flash_attn_ext_f16<256,256,1,8,false,false,false,F16>`, not VEC or tile (`ggml/src/ggml-cuda/fattn.cu:697`, `:704`, `:315`). GQA already loads each KV head once for its six Q heads.

Recorded resources at 90k: M grid 36x1x1, block 32x4x1, 226 registers/thread, 65,680 B shared memory. D one resident block/SM, four resident warps; all 36 SMs receive a block. `launch_fattn` queries occupancy and launches one resident wave (`fattn-common.cuh:1181`, `:1204`). It uses stream-K, not the non-stream `parallel_blocks` loop: D two output tiles, 18 blocks per KV head, at most 79 iterations/block for a 64-row tile. The uniform fixup grid is M 2x1x8 with 256 threads; its mean is M 1.73 us. A serial combine is not the bottleneck.

`codex/diagnose_draft_attn.py` reads immutable trace databases and writes `codex/draft-attn-diagnosis.json`. Source: `~/repos/attrib-pack/runs/attrib-host-s3m/full-a1/decode-90k.sqlite`; M 397 observations/device for the unambiguous 1x8 kernel label:

| metric | GPU0 | GPU1 |
|---|---:|---:|
| M mean kernel us | 441.40 | 440.89 |
| D unique f16 KV bytes at padded 90,112 positions | 184,549,376 | 184,549,376 |
| D effective GB/s | 418.10 | 418.59 |
| D fraction of stated 448 GB/s peak | 93.33% | 93.43% |

Bytes = KV length * 2 KV heads * 256 dimensions * K/V * 2 B. This is effective bandwidth, not a DRAM-counter measurement. D the ideal cold-read floor is 411.94 us/pass and 2.060 ms/five passes; only about 0.15 ms/five passes remains before that floor. Low warp occupancy does not establish a parallelism bottleneck.

M 2.71 ms is aggregate draft attention including catch-up: `runs/attrib-llama/full-a1/pack/budget.md:73`. The newer s3m pack reports M six draft forwards and 2.72 ms aggregate at 90k; five D kernels alone total about D 2.21 ms. HQ reports M five forwards and 1.31 ms. Its log has `kv_cache_dtype=fp8`, and its recorded FlashInfer symbol contains `__nv_fp8_e4m3` (`codex/hq-fa-dtype.txt`, original `attrib-hq/full-a2/serve.log:101` and `hq.3.sqlite`). The comparison is not equal-byte or equal-work.

## Candidate

`fattn-mma-f16.cuh:69` changes only the existing Ampere-family 256/256 eight-column configuration: 128 -> 64 threads, KV tile 64 -> 32, launch-bound occupancy 2 -> 3. D dense one-row shared memory becomes 32,848 B. M compile-only resource check: 226 registers and zero spills (`cuda-resource-results.txt`). G three resident blocks/SM, six warps, 108 blocks and 54 splits/KV head; runtime occupancy remains unmeasured. G extra stream-K scratch about 0.57 MiB/device. Two warps satisfy the existing MMA tile divisibility assertions; the first four-warp compile attempt rejected the smaller tile.

The tuning has no model, context-length or GPU-name gate; the occupancy query adapts each launch. More split-K waves with unchanged resources would not increase residency. The sparse path and other eight-column query/GQA layouts also use this configuration; the queued controls cover Q rows/GQA of 8/1, 4/2, 2/4 and 1/8.

G nominal five-pass FA time, including existing fixups: 8k 0.25 ms; 32k 0.85 ms; 90k 2.22 ms. These predict unchanged time, not a measured gain. The 32k estimate interpolates the recorded 8k/90k kernels; increased rescaling or occupancy loss could regress any depth.

## Validation and freeze

M CPU-only build: `CUDA_VISIBLE_DEVICES="" nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF` with Release/UI-off/disconnected options, then target `ggml -j8`.

M `build-cpu/bin/draft-attn-cpu` checks f16 and q8_0 at 8192/32768/90112 and an unpadded 513-row boundary. Both partitionings pass NMSE < 5e-4 against a decoded-f32 V oracle; maximum M NMSE 5.61e-11. Dropping the final split fails every case: M NMSE 0.001306 to 0.020506. Total measured case time is D 6.46 s on one CPU thread. `cpu-draft-attn-results.txt` records every case. This checks partition/reduction arithmetic, not CUDA MMA execution. Native CPU f16 V accumulation itself has M NMSE 0.001966 at 90k; the queued generic test keeps its existing length-scaled tolerance (D 0.002253 at 90,112).

M CUDA compile and freeze pass via `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/freeze_draft_attn.sh`: CUDA 13.2, local CCCL v3.4.3, sm_120a, NCCL OFF, Release, -j6, RAM guard at 10 GiB. Patch kernel source is b43e64807; baseline libraries are the verified serving freeze 98c695111. Both use a patch-tree driver with N_THREADS=8; `ggml_commit()` identifies the arm. Layout: `codex/frozen/{draft-attn,draft-attn-base}/{bin,SOURCE,SHA256SUMS}`, `bin/round-trace`, `bin/test-backend-ops`, local libraries and pre/post-patchelf hashes. No CUDA executable ran.

M five CPU harness tests pass: recorded-artifact parsers, negative gates, C++ case serialization, exact-PID cleanup and no-GPU refusal. Fixtures record paths/hashes. M `gpuq submit --check` passes. The suite gates correctness before ABBA perf on each GPU, saves raw output, waits for GPU memory release, checks frozen identity and kernel hazards, and supplies a budgeted one-arm canary. Effective GB/s may exceed DRAM peak for repeated 8k cases that fit L2.

Parent submit command (not run here):

`gpuq submit /home/xanewok/repos/draft-attn/codex/draft-attn-probe/item.toml`

Open: CUDA-vs-CPU correctness, the real canary, performance/regression results and actual candidate occupancy. Do not adopt or claim a speedup before those pass. No network, GPU query/execution, server, push or submission ran.
