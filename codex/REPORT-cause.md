# Draft attention failure: cause and disposition

Verdict: drop `256406370` and `b43e64807` from the serving candidate. No measured speed gain exists. A noisy half-precision CPU reference is proven; an additional patch error in the original failing input remains unknown. No CUDA fix or tolerance increase is justified yet.

M = measured/recorded; D = derived; G = prediction. IDs, shapes, seeds and limits are identifiers.

## Recorded evidence

Let DA = `/home/xanewok/repos/draft-attn/codex/runs`, Q8 = `/home/xanewok/repos/q8-1row-probe2/codex/runs/q8-probe3/full-a1`.

* `DA/draft-attn-ab/full-a1/00-gpu0-up-test.sql:24`: base `0bb874d20` passed `draft-fa:f16:90112`. Its ERR is **not recorded**; successful rows printed no ERR. SOURCE `98c695111` and build `0bb874d20` differ only under `codex/`.
* `DA/draft-attn-ab3/canary-a1/00-gpu0-patch-test.sql:24`: patch M ERR 0.002410793 exceeds D tolerance 0.002252800. Base did not run. `tests/test-backend-ops.cpp:68,161,8739` use fresh random inputs/masks; these runs cannot establish an error increase on identical inputs.
* `Q8/00-gpu0-up-test.sql:42,55`: **two** base failures: `f16:32768:q1:g8:d128`, M ERR 0.000868768311; `f16:90112:q1:g4:d256`, M ERR 0.002406990994. GQA=6/D=256 at `:57` passed: M ERR 0.002085716267. All eight recorded arms/devices repeat these f16 errors exactly. This deterministic probe has no draft-attn kernel patch.

## Cause and limits

`tests/test-backend-ops.cpp:304,8723` compares normalized squared error (norm of the first output) and uses `5e-4 * max(1, kv/20000)`. `ggml/src/ggml-cpu/ops.cpp:9001,9008,9037` rescales/adds V into `VKQ16` and converts to f32 only after the KV loop. `ggml/src/ggml-cpu/vec.h:570,580,845,854` stores every update in f16. `GGML_PREC_F32` does not override this. The test forces the unsplit CPU reference (`tests/test-backend-ops.cpp:12199`; `ops.cpp:9488`).

M pre-fix CPU reproduction on identical decoded Q/K/V/mask values: seed 0, GQA=4, 90112 rows gives NMSE 0.002407125022 against f32 V accumulation, already above tolerance without CUDA. Across seeds 0..7, GQA=6 gives M 0.001767478..0.002117845; this sweep did not cross tolerance for GQA=6. Input hashes/oracles repeat exactly. A deliberately half-accumulating oracle gives zero error against the native reference and makes the proof fail (M exit 134). Results: `cause-cpu-results.txt`, `cause-cpu-positive-final.txt`, `cause-cpu-negative-final.txt`.

CUDA header = `ggml/src/ggml-cuda/fattn-mma-f16.cuh`. `:69` changes KV tile/warps/launch bound together. `:1083,1214,1354` require tile divisibility: D 32 % (4*2*8) != 0, so `256406370` alone is invalid; `b43e64807` supplies the necessary two warps. Half VKQ accumulation/rescaling (`:1025,1215`) and warp partitions change rounding; occupancy changes stream-K splits (`fattn-common.cuh:1181,1204`), with f32 fixup partials (`:757`). Original inputs were not saved; CPU arithmetic cannot isolate a CUDA rounding change.

## Probe and checks

`draft-attn-cause-probe/item.toml`: immutable base/patch libraries, one shared seeded driver, seeds 0/1/2, both GPUs, interleaved ABBA; every case saves ERR, tolerance, pass/fail. F16 saves f32-V oracle error, CPU error and input hash. Reference failures remain visible; oracle error > 5e-4 fails. Canary is one paired sweep on GPU0; no server. Full/canary budgets 600/180 s, total item 900 s; G total estimate 510 s. Xid guards, exact-PID cleanup and pins retained. `gpu_ab.sh` wraps this probe.

Fix: `ops.cpp:8851` uses f32 V accumulation in `use_ref` mode; native math is unchanged. M fixed-reference NMSE is 0. Reverting the guard fails `cpu_cause.cpp` (`cause-reference-mutant.txt`).

M CPU ggml build: `cmake -B build-cpu -DGGML_CUDA=OFF`, target `ggml -j8`. Shared driver compile, eight CPU harness tests, preflight and `gpuq submit --check` pass. All commands hid GPUs and used nice 19; no CUDA binary ran.

Speed: prior M 441 us; D 418 GB/s, 93% of peak, cold-read floor 412 us and 0.15 ms headroom over five passes. G unchanged time. GPU A/B and CUDA oracle checks remain open.
