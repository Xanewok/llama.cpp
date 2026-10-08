# K10 batched top-k

Status: code, tests, Mac payload and CUDA window written and statically checked. NOT compiled or run anywhere: the Mac stayed at load 16-50 (limit 6, `ssh mini-builder uptime`, 20:54-21:30) for the whole 30 min wait. No CUDA, no GPU, no gpuq submit.

## Commits (base 0cfc0723e)
- `7d4b96143` cuda : batch top-k over rows in one launch set with a deterministic order (general)
- `0d4306a1b`-successor `local: add top-k batching off switch, engagement log and negative-control mutants` (hash: `git log`; env controls + mutants + test hooks)
- codex commit: payload, window, this report.

Source changes (general): `ggml/src/ggml-cuda/top-k-batched.cuh` (new), `top-k.cu` (dispatch before the CUB branch), `tests/test-topk-batched.cpp` (new), `tests/CMakeLists.txt`, `tests/test-backend-ops.cpp` (test_top_k: `special`/`canonical` modes, `supports_backend` harness hook, new cases and perf shapes). No incidental compile fixes.
Local: `GGML_CUDA_TOPK_BATCHED=0|force`, `GGML_CUDA_TOPK_MUTANT=1..6` (`__constant__ topk_mutant`), `GGML_CUDA_TOPK_LOG`, `TOPK_CANONICAL_ANY`, `TOPK_MUTANT`, `TOPK_FAILFAST`.

## Design
- Rows with ncols <= 8192: one block per row, one launch (shared-memory 4-pass 8-bit radix select, gather, bitonic sort). Wider rows with nrows >= 2: 10 launches for all rows (4 x hist+select, gather, finalize); scratch chunked to 32 MiB. A single wide row stays on CUB (shorter chain; G, tune with `force` arm). k > 1024, HIP, MUSA: unchanged paths.
- Exactness contract: CUB's output is unsorted and nondeterministic by its own request (`top-k.cu:25-26`), so byte identity to CUB is impossible by definition. The new op output is deterministic: key (IEEE bit-pattern order, as CUB's twiddle) descending, then column ascending; ties at the k-th key resolve to the lowest columns (in-kernel ordered scan only for rows with a boundary tie). Value multiset equals CUB's always; column set equals CUB's whenever there is no tie at the k-th key.
- Sampler reliance (why tokens are unchanged): consumers are `llama-sampler.cpp:1509` and `llama-graph.cpp:3909`; both use the indices only via get_rows with the same index tensor (alignment holds for any order). The serving chain (`common/sampling.cpp:620`, then host `:195-211`) sorts by logit and falls back to the full path on any tie among the k+1 best or non-finite value, so only the unique candidate set matters. Order-dependent unsorted `dist` chains had a nondeterministic order before; now a reproducible descending one. -inf/banned logits: removed host-side; ties among them hit the same fallback. NaN placement vs CUB is assumed (same twiddle), G, only reaches the fallback.
- Hardening: gather/sort clamp writes and reads so a corrupted state cannot go out of bounds (needed so mutants cannot fault the GPU).

## Tests
- `tests/test-topk-batched.cpp`: runs the actual kernel source on CPU (256 threads per block, barrier for `__syncthreads`, real atomics) against a brute-force (key desc, column asc) reference; narrow/wide, k 1..1024, ties, boundary ties, all-equal rows, -inf banned rows, +-inf, +-0, NaN, chunking. Mutants 1-6 must each fail (`TOPK_MUTANT`, `TOPK_FAILFAST`).
- test-backend-ops `TOP_K ... canonical=1` (CUDA only, exact order, ~190 cases incl. ties/special) plus the generic set/value cases; `TOPK_CANONICAL_ANY=1` on CPU must fail them.
- Mac results (M): NOT RUN (no slot). Parent command: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 ~/repos/qwen38-eval-worktrees/bin/bw-mac $PWD codex/k10/bw-k10-mac.sh k10-mac` from the tree; commit `codex/k10/evidence/summary.json` from its output (the window preflight requires it and its source hash).
- Risk: first compile may need small fixes; emulation runtime unknown (est. minutes).

## Window
`codex/k10/bw-k10.toml` / `bw-k10.sh` (priority 18, filler, non-preemptible, host-insensitive, same UUIDs/recipe, no freeze). `gpuq submit --check codex/k10/bw-k10.toml`: `bw-k10: valid` (before evidence exists; the payload preflight needs it). shlint and `bash -n` pass on both `.sh`; `py_compile` on `k10_perf.py`.
Stages on both GPUs: build; DeviceTopK count 25; TOP_K test (all OK, >=100 canonical cases, engagement log); controls (batching off, mutants 1-6 each must fail canonical cases; off must fail only canonical ones); perf arms auto/off/force x2 reps over {248320,124160,42,1024} cols x {1,2,6,12,13} rows x k {21,41} -> `perf-CUDA*.json` (measurements, not gates).

## Expectation
SAMP is M 0.41 ms/round at np2 8k (V 0.20 for 12 rows, 5 D blocks 0.042). D: 10 launches per set, floor ~2.5 us each => V ~0.06, D ~0.025 each; saving G 0.10-0.25 ms/round (ceiling D 0.26). Narrow candidate rows (if per-row ops exist in the graph) 4 launches -> 1. Falsifier: perf JSON `auto_over_off` at 12 rows > 0.6, or an nsys V top-k set > 100 us / D set > 30 us. Depth-flat (logits shape does not depend on context), so same at 8k and 90k.

## Left open
Mac run; any CUDA compile error; NaN order vs CUB; whether the serving graph has single-row narrow top-k ops (decides the small-kernel value); the nrows==1 wide threshold.
