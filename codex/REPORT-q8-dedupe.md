# Shared q8 activations and residual RMS normalization

## Conclusion

Implemented on `topic/q8-dedupe`, based on stack6 `ac7465885dc9f13de396ccd91b4be5eaffde3730`:

- `849f16315`: reuse small shared matmul activations within a CUDA graph evaluation.
- `7967e192c`: fuse residual ADD, RMS_NORM and its weight MUL, while preserving the residual output.

CPU correctness and CUDA compilation pass. GPU correctness, engagement, capture/replay and performance remain unmeasured. Queue `codex/probe-item.toml` first; only after it passes, queue `codex/depth-item.toml`. Neither item was submitted or executed.

D: the Swift J5q target graph at six rows has 144 removable quantizations before runtime alias invalidation. Applying the supplied IQ4_XS launch-floor mean (M in the brief: 1.18 ms / 443 calls) gives D: 0.384 ms for the target verify graph. For a complete five-draft round, the attention-only MTP structure adds D: 12-13 removable calls, giving D: approximately 0.416-0.419 ms. This extrapolates another quant mix's kernel cost; it is not a GPU speed result. G: residual fusion may save up to 0.2 ms/round; no positive lower bound is established.

## Census

`codex/census.cpp` records the allocated graph through `cb_eval`, returning false to each data-read request. The 4B runs compute on CPU. Swift runs use the same callback but abort CPU execution at its first abort check; `llama_decode` returns 2, which the census-only mode treats as expected. They load the real J5q GGUF and build the real target graph, after only the first CPU embedding node (`ggml-cpu.c:3150`), without a model-layer forward pass. Context is explicitly 4096; CPU threads are 8. `codex/census-*.tsv` contains tensor identity, source identity, op, weight type and shape. `codex/census.json` records each shared group and its node positions.

| Model / configured rows | Graph nodes (M) | Quantized weight matmuls (M) | Compatible shared groups (D) | Duplicate quantizations before existing fusion (D) | ADD -> RMS_NORM pairs (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Q4_K_XL / 1 | 1925 | 201 | 64 | 72 | 64 |
| Qwen3.5-4B Q4_K_XL / 6 | 1925 | 201 | 50 | 52 | 64 |
| Qwen3.5-4B Q4_K_XL / 8 | 1925 | 201 | 54 | 54 | 64 |
| Swift J5q / 1 | 3845 | 401 | 128 | 144 | 128 |
| Swift J5q / 6 | 3845 | 401 | 128 | 144 | 128 |

On Blackwell, `ggml_cuda_should_use_mmvq` in `mmvq.cu:324` uses these limits:

| Weight type | MMVQ maximum rows | Path at configured rows 1 / 6 / 8 | MMQ layout |
| --- | ---: | --- | --- |
| q4_K | 5 | MMVQ / MMQ / MMQ | DS4 |
| q5_K | 6 | MMVQ / MMVQ / MMQ | DS4 |
| q6_K | 7 | MMVQ / MMVQ / MMQ | D4 |
| iq4_xs | 8 | MMVQ / MMVQ / MMVQ | D4 |
| q8_0 | 8 | MMVQ / MMVQ / MMVQ | D4 |

The Swift J5q census has M: 178 q6_K, 141 q5_K, 47 iq4_xs, 35 q8_0 and 96 F32 weight matmuls. The small GDN alpha/beta projections are F32; they do not invoke a q8 quantizer. It has no MUL_MAT_ID nodes. At one row, existing gate/up fusion already removes D: 64 Swift calls and 32 4B calls, leaving D: 80 and 40 additional reusable calls respectively. At six rows that fusion is disabled. GDN qkv and gate consumers can be separated by D: about 40 nodes, so adjacent fusion misses them.

The target census is observed; the complete MTP-round count above is derived from `src/models/qwen35.cpp` and the existing attention-only MTP block. It is not a census of CUDA Meta split graphs. Backend selection, alias invalidation, compute flags and existing fusion can reduce the actual skips. The canary requires positive six-row skip counts on both GPUs, rather than assuming the estimate is attained.

## Design and invariants

`ggml-cuda.cu:4585` prepares an evaluation-local plan. `q8-cache.h:24` counts consumers of each `(src1 tensor identity, quantizer variant)` and reserves only groups with multiple consumers. Tensor data, dimensions and strides are immutable during a graph compute; therefore exact tensor identity also identifies its data/layout within this scope. Distinct views do not share entries, even if their bytes happen to match.

Variant 0 is ordinary MMVQ q8_1. Variants 1..3 are MMQ D4, DS4 and D2S6. MMVQ's quantizer ignores weight type; MMQ dispatches only on its scale/sum layout (`quantize.cu:558`, `:575`). The original quantizers and matmul kernels consume the same dimensions, strides and quantized values; only redundant launches are omitted (`mmvq.cu:1548`, `mmq.cu:193`). Native FP4 and expert MMQ row permutation/scatter paths retain their existing allocations and quantization. Ordinary MMVQ MUL_MAT_ID can reuse the standard layout, but the Swift census contains no such nodes.

One pool allocation holds the plan's aligned scratch slices. It precedes temporary matmul allocations and is released after evaluation/capture/launch, preserving the pool's stack order. The allocator's existing CUDA-graph scratch lifetime applies: the backing pool remains owned by the backend. Every captured evaluation starts with all entries invalid and captures a quantizer at the first consumer. Replay executes that quantizer against current inputs; no host cache hit or quantized value survives between evaluations. The cache pointer is cleared before returning. Meta split graphs create separate plans on each CUDA backend invocation, with no reuse across splits, devices or backend instances.

After each executed node, including every node in a fused group, overlapping writes invalidate cached sources (`q8-cache.h:76`). The byte-span check includes strided views and in-place CPY. Optimizer graphs and graphs with concurrent CUDA stream regions use the old path. Scratch retention is bounded at a configured 64 MiB; inputs beyond eight total rows or the dimension bound fall back. D: Swift's unsplit six-row one-layout reservation is 4,423,680 bytes, approximately 4.22 MiB; physical Meta split allocations require GPU measurement. All backends other than CUDA retain their source paths.

The residual matcher (`ggml-cuda.cu:4496`) accepts contiguous F32 matrices with at most eight rows, equal-shaped add operands and vector RMS weights. It keeps ADD and MUL as observable outputs and elides only RMS_NORM. Exact in-place residual writes are allowed; partial input overlap, overlap with weights and overlapping residual/normalized outputs are rejected. The column bound leaves room for the kernel loop increment. The kernel (`norm.cu:162`, wrapper `:760`) stores the rounded residual, uses the existing RMS reduction/block-size arithmetic, then multiplies normalized values by the weights. This is source-level equivalence; GPU bitwise identity remains a required gate.

Rejected alternative: extending gate/up fusion to several rows would require additional fused kernel variants, would not cover the separated GDN consumers, and would miss q4_K verify matmuls dispatched to MMQ. A persistent data-pointer cache would need value-version tracking across graph replays and Meta split boundaries. The evaluation-local plan avoids those lifetimes.

## Verification and CPU numbers

M: CPU build and CUDA build passed. CUDA configuration is 13.2, SM120a, NCCL OFF, CCCL v3.4.3, GGML_CCACHE ON, offline FetchContent, forced CUDA compiler to avoid a probe executable. Compiles used `nice -n 19 bin/buildgate ... -j4`. Shared ccache was read-only, with temporary files in the worktree; the initial default-cache build failed because the sandbox rejected outside writes. No CUDA binary, llama-server, llama-cli, test-backend-ops, GPU query or sanitizer was executed.

M: `build-cpu/bin/q8-cache-contract` passes layout separation, configured widths 1/6/8, evaluation reset, capacity fallback, distinct tensor identity and overlapping view writes. `q8-cache-contract stale` suppresses the real cache's invalidation and exits 1 with `stale activation reused after an alias write`; this is the negative control. These are host protocol tests, not CUDA numerical tests. Logs: `codex/cache-contract.log`, `codex/cache-mutant.log`.

M: the 4B fixture ran through the server-context library in the custom executable `q8-identity-cpu`, with context 4096, seed 1729, greedy target sampling, probabilistic MTP draft k=3, subset draft head and two requests of 16 tokens each. CPU used 8 threads; Meta(CPU,CPU) used 4 per backend. It ran through `bin/cpugate`, one model process at a time. Reuse-off/on token bytes and final logit bytes are identical within each backend configuration, and draft counters are positive. Each final logit row contains M: 993,280 bytes. Evidence: `codex/cpu-results/identity.json` and adjacent `.bin.stats.json`/`.log` files; `codex/run_cpu_q8.sh` reproduces the build and checks.

| CPU configuration | Reuse off wall seconds (M) | Reuse on wall seconds (M) |
| --- | ---: | ---: |
| CPU | 11.256593 | 11.626937 |
| Meta(CPU,CPU) | 15.016119 | 16.755032 |

These wall times include loading and requests; they are not a performance comparison of CUDA code. CPU identity is an unchanged-backend control, not proof of CUDA results. `cpugate`'s ordinary status command could not open the read-only queue lock; the daemon snapshot was inspected and showed a running filler, not a measure item. `codex/bin/gpuq` provides a read-only snapshot adapter for future gated CPU work; on read failure it reports a running measure item and waits.

M (v2): `test-backend-ops` compiled the `MUL_MAT_SHARED` class, but its registration was inside `#if 0`. No shared-input backend cases were runnable in v2. The RMS additions were registered. No backend test was executed by this worker.

M (v2): `q8-exact reference` and `q8-exact patch` serialized separate-process results. `patch` alone was crash coverage; only `gpu_probe.sh`'s external `cmp` checked equality. The planned full probe compared D: 126 cases across both GPUs, but was not executed. v3 replaces this interface with same-process reference/patch checks.

M: recorded-artifact parser replay passes on `orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json`, memory CSV and journal fixtures, including incomplete/repeated counters, insufficient size and absent token-ID negative controls (`codex/parser-fixture.log`). This reuses the unmodified harness parser. The new engagement gates reject the real recorded baseline log; synthetic positive and zero-skip format checks pass (`codex/engagement-parser-check.log`). The new CUDA engagement lines have no real positive recording yet; their actual formatting and counts must pass the live canary.

## Freeze and queue items

Final source freeze: `codex/frozen/q8-dedupe-v2/`, SOURCE `7967e192c2f52ce32b633822ecdb157e140c7b88`, RUNPATH `$ORIGIN`, SHA256SUMS, recipe and `q8-exact`. `codex/frozen/q8-dedupe/` is an earlier superseded snapshot and is not pinned by the final items. The ggml stamp was checked against HEAD at the frozen source commit (`codex/freeze-v2.log`). CUDA binaries were inspected/copied, never executed.

- `codex/probe-item.toml`: correctness suite first, reference/patch frozen-library bytes, two GPUs, binary identity gate, capture/replay and engagement. Canary runs the exact suite without the broader backend suite; it uses no server. G: planned runtime within 180 seconds for the canary and 360 seconds for the full suite. Child caps, cleanup and an explicit overall deadline fit the respective 300/600-second stage budgets. Fault journal reads start at this script's cutoff; GPU queue guards own the external idle/RAM checks.
- `codex/depth-experiment.toml` and generated `codex/depth-item.toml`: stack6 versus the final freeze, J5q, TP2, k=5, both `LLAMA_ARG_KV_PAD=4096` and `--kv-pad 4096`, ABBA at configured depths 8192/32768/90112, cross-arm token identity, unchanged sampling flags and workload. The one-boot canary uses the patched arm at depth 8192 with the full token/steady-size and engagement gates, capped at 600 seconds. Cross-arm equality itself is exercised in the full ABBA run; one boot cannot compare arms. G: inherited boot estimate 420 seconds; total item budget is generated by `harness.mkitem` with margin and cleanup.
- `codex/gpu_ab.sh` is a guarded entry to the existing harness; `--parser-test` is CPU-only. The harness records requests before evaluating gates, uses frozen libraries, checks exact-PID server stop, port availability, budgets and scoped fault journals. No `harness/*.py` file was edited.

M: both `gpuq submit --check` validations pass (`codex/probe-submit-check.log`, `codex/depth-submit-check.log`). The depth manifest was rendered with `python3 -m harness.mkitem` in the qwen38-eval repository. The nested recorded SSE fixture is explicitly pinned as well as the existing harness and frozen trees.

## Risks, open items and upstream split

The alias scan is linear in retained inputs for each executed node, and retained scratch increases device memory. CUDA graph replay avoids this host work; direct execution and recapture may regress. Meta can split shared groups or present distinct tensor identities, reducing reuse. Quantizer scratch lifetime, actual GPU engagement and CUDA numerical equivalence remain unverified until the probe. The live canary is mandatory before interpreting depth timings. No GPU gain is claimed.

Residual reduction duplicates a small part of the existing RMS kernel; review its arithmetic and compiler output alongside the existing implementation. The queued byte comparison gates adoption. HIP/MUSA source compatibility and GPU execution were not tested. MUL_MAT_ID reuse is not covered by the prepared numerical probe. If any exactness gate fails, do not adopt either optimization for serving; isolate cache reuse from residual fusion before measuring again.

Local prior art, found without network: `74976e1ae` (CUDA matmul refactor), `658987cfc` (non-contiguous MMVQ), `f77c13b91` (general GEMV fusion), `bed0a8566` (shared-expert MMVQ), `8c988fa41` (RMS fusion), `009b709d6` (add/RMS-related fusion), `687e77892` (RMS/MUL/RoPE fusion). The existing add/RMS fusion is a different graph pattern; this change preserves the pre-norm residual as a second output.

Upstream split: quantization reuse plus shared-input tests; residual ADD/RMS/MUL fusion plus retained-residual tests; local census/queue tooling stays separate. CUDA logs and environment controls follow existing stack diagnostics, but should be reviewed for removal or integration before upstream packaging. No package tree, issue, PR, reviewer comment, commit attribution or network operation was created. Network duplicate checking remains for the owner before submission.

## v3

The disabled registration and standalone `q8-exact patch` false assurance are fixed. Source commit `cebdc99ca` contains the test changes; frozen source `0397ceb41c3cf3f05e1f51116067dea200305e41` includes the initial v3 tooling. GPU correctness and speed remain unmeasured. No queue item was submitted.

`tests/test-backend-ops.cpp:10452` now registers the shared cases outside `#if 0`. M: the CPU-only `q8-shared-contract` enumerated the real evaluation registry and built all 60 cases, checking three consumers with identical `src1`, whole-graph comparison, and the overwrite placement. The existing comparator at `tests/test-backend-ops.cpp:1560` runs the complete graph on the selected backend and the CPU reference backend and compares its concatenated outputs. GPU execution is deferred, but these cases are now runnable. Coverage is q4_K/q5_K/q6_K/iq4_xs/q8_0, configured rows 1/6/8, contiguous/strided inputs and overwrite/no-overwrite. Input width 768 exercises an odd number of K blocks; output width 257 exercises an odd output tile boundary. K dimensions must retain the quantization block alignment.

`codex/exact-gpu.cpp:124` runs reference then patch for each case in the same process on the same selected device, using fresh backend instances and identical deterministic weights/inputs. It compares every output byte at `:129` and exits 1 on mismatch. Each mode performs direct execution, capture and changed-input replay. Reference disables reuse and fusion; patch enables both. The fusion switch is now read once per direct/capture evaluation and once per graph optimization, instead of being cached statically or read per node. Existing captured graphs are not toggled between modes: each mode owns a fresh backend. D: the full suite has 63 comparisons per GPU; the canary has three per GPU (shared input, alias overwrite, retained residual). D: two direct/capture evaluations per mode yield 252 diagnostic lines per GPU in the full suite and 12 in the canary; the queue gates check these counts and positive six-row reuse/fusion engagement.

M: CPU build of `test-backend-ops` and the CUDA freeze build pass. The backend executable was compiled but never run here. `GGML_TEST_MAX_THREADS=8` caps test initialization threads in these builds; the default remains unchanged for other builds. M: `q8-exact-contract --cpu-contract` passes 60 shared graph constructions and cache reservation/hit checks. M: `q8-cache-contract` passes; `q8-cache-contract stale` exits 1 with `stale activation reused after an alias write`. This mutant suppresses the real host cache's invalidation. It is a cache-protocol negative control, not GPU numerical evidence. No CPU model run was needed or repeated for this test-only change; the earlier CPU model numbers above are v2 evidence.

Reproduction and evidence:

- `nice -n 19 bin/buildgate cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DCMAKE_CXX_FLAGS=-DGGML_TEST_MAX_THREADS=8`, then `bin/buildgate cmake --build build-cpu --target test-backend-ops -j4`. Logs: `codex/v3-cpu-configure.log`, `codex/v3-cpu-build.log`.
- CPU graph proof: compile `codex/shared-graph-contract.cpp` against `build-cpu/bin` with `-DGGML_TEST_MAX_THREADS=8`, then run `build-cpu/bin/q8-shared-contract`. Logs: `codex/v3-shared-contract.log`, `codex/v3-graph-contract.log`, `codex/v3-cache-contract.log`, `codex/v3-cache-mutant.log`.
- `bash codex/freeze-v3.sh` configured with HEAD at the frozen commit, CUDA 13.2, SM120a, NCCL OFF, offline CCCL and read-only shared ccache. Every compile used `nice -n 19 bin/buildgate` and at most `-j4`. M: `ggml_commit()` matched the frozen source. Logs: `codex/v3-freeze.log`; provenance: `codex/frozen/q8-dedupe-v3/{SOURCE,RECIPE.txt}`. `q8-exact` and the libraries resolve within the freeze, with RUNPATH `$ORIGIN` and a checked checksum manifest. No CUDA executable was run.
- M: sanitizer/probe CPU preflights pass (`codex/v3-memcheck-preflight.log`, `codex/v3-probe-preflight.log`). Recorded backend/sanitizer logs copied from the original memcheck fixtures and the harness's recorded SSE/memory/journal fixtures exercise the parsers. Negative controls reject unsupported/missing cases, absent exactness output, zero skips, missing instrumentation, timeouts and fault records. Positive reuse/fusion and racecheck summary formats still have only synthetic controls; the live canary must establish those recordings.
- M: `gpuq submit --check` passes for all three manifests below (`codex/v3-{memcheck,probe,depth}-submit-check.log`). The depth manifest is generated with `python3 -m harness.mkitem codex/depth-experiment-v3.toml --out codex/depth-item-v3.toml`.

Run order, each with its canary first:

1. `codex/memcheck-v3/item.toml`: GPU0 shared-input CPU-reference cases plus self-checking `q8-exact` under memcheck; residual fusion also runs racecheck and synccheck. D: full supported case counts are 60 shared, 19 fused per fused stage and 63 exact comparisons. Canary runs one shared, one fused per fused stage and three exact comparisons, including every parser/gate. D: full child caps sum to 840 s within a 1200 s run budget; canary child caps sum to 330 s within a 500 s budget and 550 s stage timeout. Findings, failed engagement or unavailable fault journals stop later stages.
2. `codex/probe-item-v3.toml`: full shared and filtered residual suites against CPU on CUDA0 and CUDA1, plus same-process exactness on both devices. D: full child caps sum to 360 s within a 600 s run budget; canary caps sum to 180 s within a 300 s budget, with cleanup allowance. Checks frozen identity, exact comparison completion, case counts, reuse/fusion engagement and replay counts.
3. `codex/depth-item-v3.toml`: stack6 versus v3; inherited ABBA, sampling, depths, cross-arm token identity, steady-size gates, exact-PID cleanup and scoped fault checks. `codex/gpu_ab-v3.sh` is the matching guarded entry point. The workload and control arm remain as documented above.

M: the pinned v2 checksum manifest still passes (`codex/v3-v2-frozen-check.log`). `codex/frozen/q8-dedupe-v2/` and `codex/memcheck/` were only read/copied. New frozen binaries and runtime evidence stay local; manifests pin the new paths. Residual risks are unchanged CUDA arithmetic, odd-tile memory safety and real capture/replay behavior. Adopt neither optimization until the v3 sanitizer and exactness gates pass; no GPU gain is claimed.
