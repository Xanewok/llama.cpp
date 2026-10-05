# Blackwell per-type MMVQ

Conclusion: specialize only the Blackwell table at ncols 5-8. Q8_0, other types, columns 1-4 and all other tables retain general-form behavior. Host-selected template geometry and the separate one-row tail launch are unchanged. M = measured; D = derived; G = prediction.

| type | n5 | n6 | n7 | n8 | n6 GB/s (G) |
|---|---|---|---|---|---|
| q4_K | w1r4 | w1r4 | w1r4 | w1r4 | 328.93 |
| q5_K | w2r4 | w1r4 | w1r4 | w1r4 | 358.50 |
| iq4_xs | w1r4 | w1r4 | w1r4 | w1r4 | 389.84 |
| q6_K | w2r4 | w2r4 | w1r4 | w2r6 | body 363.68; head 373.33 |
| q8_0 (unchanged) | w2r4 | w2r4 | w2r4 | w2r4 | 404.70 |

Predictions reuse sweep-2 r1 kernel bandwidths (D), not serving measurements: `~/repos/qwen38-eval/results/2026-10-05-mmvq-sweep2/a1/results.md`. For the changed types, each configuration wins its aggregate body table.

J5q GGUFReader (`~/repos/llama.cpp/gguf-py`) finds M: 1,042,944,000 bytes in Q6_K `output.weight` out of M: 9,261,772,800 Q6_K bytes, D: 11.26%, rather than the brief's approximate G: 5%. Full head shape is [5120,248320]; TP halves the rows. At n6, byte/time weighting gives D: 364.74 GB/s for w2r4 versus 356.65 for w1r4. At n8, w2r6 gives D: 314.30 versus 306.20 (w1r4) and 304.96 (w1r8). Body wins justify the type rule. A future n6 head-row threshold needs more shapes; implement the type rule first.

Evidence: `ggml/src/ggml-cuda/mmvq-params.h:22,164,198`; `mmvq.cu:848,864` launch geometry/tails remain unchanged. `tests/test-mmvq-params.cpp:24` independently checks expected Blackwell choices and rows 1-33. `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/check_mmvq_pertype_cpu.sh`: M: 1840 configs pass, CTest 0.01 s; warp/row/tail mutants abort. M: 11008 baseline entries compared, only the intended 52 change. Recorded GPU SQL/device/memory/kernel-log parser controls pass.

Compile: M: 600 -> 612 regular CUDA specializations (nm), D: +2.0%; all-table count unchanged. M: matched sequential nvcc general 103.76 s, patch 102.55 s; D: -1.17%. One pair under variable host load does not establish a compile speedup. `codex/check_mmvq_pertype_compile.py`; commands/counts in `codex/logs-pertype/`. Freeze build reused unchanged, matching-flag objects.

Freeze: `codex/frozen/mmvq-pertype`, SOURCE `26a2a2758`; CUDA 13.2, local CCCL v3.4.3, NCCL OFF, 120a, Release. M: hashes and library resolution pass. Existing MMVQ freezes pass their hashes.

Risks: w1r4 changes the per-row K split/reduction order; token identity and equivalent sampling law are not proved. k=3 single-sequence verify uses unchanged ncols=4. Other batches need identity/KLD checks. No GPU execution or test-backend-ops run occurred. Parent: queue `codex/pertype-probe/item.toml` (validated), entered via `codex/gpu_ab.sh`; canary precedes interleaved frozen A/B kernel tests. This probe does not establish serving identity or speed. HIP/MUSA compilation remains untested.
