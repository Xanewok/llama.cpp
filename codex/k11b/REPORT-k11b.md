# K11b: single staging copy, slowest host link first

Base `0cfc0723e`, branch `topic/s10-k11b`. Nothing here ran on a GPU; the CUDA code has never been compiled by nvcc.

## Commits

| hash | subject |
|---|---|
| 7baad92f9 | meta : stage mirrored host inputs once and upload slowest link first |
| b5bd35ca7 | local: add input staging A/B controls and negative-control mutants |
| f093da1aa | codex : Mac CPU gate for the shared input staging |
| 6eb1bfaf4 | codex : CUDA window, nsys item, host stand-in for the staging code |
| c6cc71a48 | codex : drop timeout(1) from the Mac gate, the model gate bounds runtime |
| (this commit chain tip) | codex : record the Mac gate result for the shared input staging |

Source changes (7baad92f9, general form, no env var):
- `ggml/src/ggml-backend-impl.h`: optional group functions `ggml_backend_input_stage_{init,free,set_batch}` (reg proc addresses, like `ggml_backend_comm_init`).
- `ggml/src/ggml-backend-meta.cpp`: `set_tensor_batch_owned` offers the batch to the staging first (`n_backends > 1`), falls back to the existing per-rank loop; dtor frees the stage; `n_transfers == 0` returns true early.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_input_stream_ensure` / `ggml_cuda_input_copies_enqueue` extracted from the per-rank function (which now calls them, behaviour unchanged); stage struct (2-slot `cudaHostAllocPortable` ring, per-rank `uploaded` event), link ranking by a 4 MiB best-of-5 pinned H2D probe with 15% log-class hysteresis; incidental: `#include <cmath>`.
- `tests/test-meta-input-stage.cpp` + `tests/CMakeLists.txt`: fake-rank meta tests and a CUDA test (skips with fewer than 2 devices).
Local-only (b5bd35ca7): `GGML_META_INPUT_STAGE=0` (off), `GGML_META_INPUT_RANK_ORDER=rev` (S1 per-rank reverse), `GGML_CUDA_INPUT_STAGE_ORDER=fwd|rev`, `GGML_META_INPUT_STAGE_MUTANT`, `GGML_CUDA_INPUT_STAGE_MUTANT` (negative controls).

## Design and why it is exact

The meta backend hands the whole batch of mirrored host inputs to the CUDA group once. Each source is memcpy'd once into the pinned slot; every rank then uploads from the slot (same bytes, same destinations, same stream/event ordering as before; the per-rank path skips its own memcpy). Order is the measured slowest link first (no hard-coded rank). Slot reuse waits on every rank's `uploaded` event before the host overwrites it (ring of 2, so the wait is normally free). Init failure, `cudaMemcpyBatchAsync` overlap, unsupported shapes: the stage declines with no side effects and the original per-rank path runs, so the result cannot differ.

## Mac CPU results (M, mini-builder, load 12, no llama process; source c6cc71a48, out `build-bw-mac-results/s10-k11b-mac-mac-a2`, untracked)

| stage | rc | wall s |
|---|---|---|
| cpu-build | 0 | 68 |
| units | 0 | 2 |
| glue-mutant | 0 | 0 |
| cuda-sim | 0 | 4 |
| cuda-sim-mutants | 0 | 19 |

- units: `test-meta-input-stage` PASS; fake cases `no-staging-exports`, `init-refused`, `batch-declined`, `staged`, `mutant-drops-a-write` OK; the CUDA case SKIPs on the Mac; `test-alloc` meta host tests pass.
- cuda-sim (`codex/cuda-sim/stage_sim.cpp`, host stand-in with lazy per-stream queues, ASan+UBSan): per-rank, declines, ring-lazy, ring-random, order slow1/slow0/sym/unmeasured all OK. Lazy queues mean a premature slot reuse corrupts the data, so the event wait is actually tested.

## Negative controls (all caught, M)

- fake-stage mutant (drops a write): test fails as required.
- META glue mutant (`GGML_META_INPUT_STAGE_MUTANT`, rank 1 reads `data+1`): aborts.
- sim mutants: no slot wait (ASan heap-use-after-free), rank-0-only wait (same), ascending order (order scenario FAIL), shifted source (`rank 0 round 0 B differs`).
- Window gate: `GGML_CUDA_INPUT_STAGE_MUTANT` (half-size memcpy) must fail the real-CUDA test.

## Window

- Build + GPU exactness: `codex/bw-s10-k11b.toml` -> `gpuq submit --check`: `bw-s10-k11b: valid`. Preflight on the committed tree: PASS (requires clean tracked tree and `codex/k11b-evidence/summary.json` tree hashes = HEAD ggml/tests trees).
- nsys: `codex/nsys/k11b_nsys.toml` -> `k11b-nsys: valid` (measure, canary, budget_s 2500). Arms: base (stage off), shared, sharedfwd, s1rev, base2 (same-arm repeat = spread). Same shape as `x2-hostprof-s8` (np2 8k, ub 512, cuda pass). Analysis: `codex/nsys/k11b_analyze.py` prints `K11B-VERDICT` lines. shlint clean on all three scripts. Not submitted.
- The nsys item needs the build window's `build-cuda/bin/round-trace` and `BUILD_STAMP`; submit the build window first.

## Expected (D from X2, G from overlap)

- D: GPU1 H2D chain ends 75.7/73.1 us after GPU0's end; slowest-first + one memcpy should cut the GPU1 start lag at V->C0 and D5->V to < 40 us and the block span by >= 40 us (model -96/-108 us at the two boundaries). Round: D -0.20 ms/round np2 8k; G -0.10..-0.20; 90k about 0.75x of that; not additive with K3/K12/K13.
- Falsifier: lag stays > 40 us at boundary 0 or 6, or block span does not shrink >= 40 us vs `base`, or the shared-vs-base gain is within the base-vs-base2 spread; `sharedfwd` vs `shared` separates order from the single copy; `s1rev` is the per-rank reverse A/B.

## Blockers and uncertainty

- nvcc compile of the stage code, the `cudaMemcpyBatchAsync` overlap fallback and real event semantics are unverified until the build window (the stand-in only proves the logic).
- 1 boot per arm: the nsys item is a mechanism capture, not a timing A/B; a timing claim needs a separate measure item.
- Link ranking is a pinned-copy probe at init; if CUDA1's probe at init disagrees with the loaded steady state the log line (`slowest host link first: CUDA1 ...`) shows it and the nsys item aborts on a wrong order.
