# Decode flash attention KV tail bound

## Conclusion

Implemented a launch-free, device-side mask scan for dense single-sequence FA with Q <= 8. MMA partitions the live extent; VEC and TILE stop their existing loops at the bound. Padded tensor shapes, strides, op parameters and graph reuse remain intact. No sampling code changed.

Final CUDA freeze: `codex/frozen/fa-kvmax-decode-v3`, SOURCE `30dbea326257ff69a210fc8caf24650b688c7fe9`. Earlier freezes are superseded. M: CPU and CUDA builds pass; the frozen `ggml_commit()` stamp matches SOURCE. No CUDA executable or GPU tool was run. GPU correctness, numerical differences and performance remain unmeasured.

M: greedy pad token identity passes on CPU and Meta(CPU,CPU), including MTP. This is not stochastic or logit identity: the extra stochastic CPU experiment diverged across pads, and Meta's final logits differed despite matching greedy tokens. See the evidence below.

## Existing behavior and history

- `ggml/src/ggml-cuda/fattn-common.cuh:1219`: dense `KV_max` is generated only with a mask, aligned K length, and `Q->ne[1] >= 1024 || Q->ne[3] > 1`. Small single-sequence decoding passes a null pointer.
- `ggml/src/ggml-cuda/fattn-vec.cuh:250` and `fattn-tile.cuh:954`: both honor a supplied bound in the KV loop. `fattn-mma-f16.cuh:2076` and `:2125` honor it by clamping each segment. None of these dense families ignores a supplied bound.
- The old MMA stream-k partition uses padded `ne11`, before that clamp. Thus enabling the old scan alone leaves blocks assigned to masked work.
- Sparse MMA uses index lists and live counts in the same argument slot. Its compaction and partition behavior are unchanged.

Observed in local history: `92b8810ec7aa6d778bc287cc918443cf67b962e2`, `CUDA: skip masked KV slices for all FA kernels (#14924)`, introduced the gate. Its comment explicitly justifies the launch overhead for large query blocks or multiple sequences. This is a deliberate cost heuristic, not an attention-semantic restriction. Command: `GIT_NO_LAZY_FETCH=1 git log -S 'Q->ne[1] >= 1024' -- ggml/src/ggml-cuda/fattn-common.cuh`; no network search was performed.

## Design and invariants

`fattn-common.cuh:22` scans backward in D 256-column units. One physical warp uses aligned packed-word loads to read every real query row; it skips a unit only when every mask value is negative infinity. A block barrier broadcasts the result. Positive infinity, NaNs, finite biases and internal holes retain the tile. Padding query rows are never read. Unaligned masks or non-256-aligned KV extents keep the original extent. CUDA lanes load eight half values in one transfer and compare the packed negative-infinity bits, removing up to D 8 scalar loads/conversions per lane per query row.

The scan runs inside each attention block, with no extra launch, device allocation, host synchronization or persistent cache. It intentionally repeats mask work across heads/layers. D: a worst-case Q=8, tail=4095 scan reads 64 KiB per block including the first live unit. G: with 64 blocks, this is D 4 MiB per call. Cost when no tail exists is one mask unit per block; net benefit needs the queued A/B.

Alternatives:

- A per-mask cache could amortize the scan. G: an extra launch costs 2-5 microseconds per mask. D: perfect sharing across the five graph masks of a k=3 round would add 10-25 microseconds; a per-FA launch across the target and draft layers could instead add G 40-100 microseconds. Its ownership must survive capture/replay, multiple retained graphs and concurrent streams; mask pointers can also be reused or written in place. A CUDA backend graph is not necessarily the complete llama graph across tensor-split/AllReduce boundaries. That machinery is larger than this concern and does not guarantee cross-layer amortization.
- A host bound is cheap to derive, but a changing host scalar captured as a kernel argument becomes stale on replay. Updating tensor dimensions defeats the padding benefit; a device scalar input requires a new backend-generic op contract and upload. Neither is necessary here.

`fattn-mma-f16.cuh:2030` computes stream-k work from the live bound. Both fixups recompute that same bound: `fattn-common.cuh:805` and `:897`. The general fixup updates its division constants and total work. The uniform fixup skips empty contributions; a single work tile uses the direct output path. Host launch geometry and fixup buffer capacity remain padded/fixed, so very short live extents can still launch empty blocks, but masked tiles do not determine the work split. An all-masked extent retains one work tile for the existing sink/output path.

GQA does not change the bound because the supported mask is shared across heads. Small batches spanning multiple query tiles use the union of all real queries, conservatively. Original K/V lengths and strides remain available for addressing and final-tile bounds checks.

CUDA replay reads the current mask every execution. There is no pointer-generation or evaluation cache to invalidate. Single-sequence Q <= 8 is the eligibility condition; larger batches and multiple sequences keep the existing path. The physical warp is flattened across the block's x/y dimensions, including VEC's logical-warp layout on HIP. HIP was source-audited but not built.

Numerics: removed columns have exactly zero mathematical attention weight. VEC/TILE preserve the live reduction order at a fixed padded shape; ordinary finite-input results are expected to remain bitwise identical, but this was not GPU-verified. MMA changes partial boundaries and softmax rescaling/reduction order, so seeded cross-arm token identity is not promised. The queued correctness gate requires NMSE <= 5e-4 versus CPU and the unpadded reference. This is a test tolerance, not a proven universal ULP or token bound.

## Tests and CPU evidence

`tests/test-backend-ops.cpp:8129`, `:11316`: D 48 correctness cases, Q 1/4/6/8, GQA 6, head size 256, f16/q8_0 K/V, tails 0/1/255/256/257/4095. The extreme tail leaves one live column. Alternating query rows have different live ends. A packed unpadded mask and strided K/V views form the reference; `ggml_cont` preserves FA's contiguous-mask contract. Both backend comparison and padded/reference comparison use the existing FA tolerance. `:11851` adds D 48 performance shapes for the requested depths and padding choices. M: compiled in both builds. `test-backend-ops` was never executed.

M: `codex/fa-tail-cpu.cpp` executes 48 padded/unpadded cases with zero maximum absolute error. Its mutant cuts off the last live tile; M 48 mutant cases return the expected failure, maximum absolute error 6.2515769. `codex/negative-control.patch` provides the corresponding CUDA scan mutation; `git apply --check` passes, but it was not applied or GPU-run.

M: `codex/streamk_contract.py` passes 9,216 arithmetic cases covering live work coverage, empty uniform blocks, general fixup gates and dynamic division constants. This checks partition arithmetic, not CUDA execution.

Model fixture: `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`; explicit `n_ctx=4096`, nice 19, CPU runs via `cpugate`, at most eight compute threads, four per Meta rank. Greedy target sampling; MTP maximum draft length D 3; two requests per process. M 64 returned token IDs per process match across pads within each backend. The fixture also records all final logits.

| Backend / pad | Model process wall seconds M | Draft tokens M | Final logit max absolute difference M vs pad 256 |
|---|---:|---:|---:|
| CPU / 256 | 30.971027 | 47 | 0 |
| CPU / 4096 | 27.017405 | 47 | 0 |
| Meta(CPU,CPU) / 256 | 43.412221 | 46 | 0 |
| Meta(CPU,CPU) / 4096 | 43.792362 | 47 | 12.179123878479004 |

These wall times are fixture evidence, not a CPU performance A/B. The Meta draft counts and logits differ. An additional temperature=1, seed=1 CPU HTTP experiment produced different tokens across pads (`codex/cpu-results/initial-stochastic.json`). Cause was not isolated. D: CPU FA already splits by `(nek1 + nth - 1) / nth` at `ggml/src/ggml-cpu/ops.cpp:9492`; changing padding can change reduction order. CPU implementation is unchanged by this branch, so this remains a baseline validation limitation.

Commands/evidence:

- `cmake -B build-cpu -DGGML_CUDA=OFF`; `buildgate cmake --build build-cpu --target llama-server test-backend-ops test-sampling -j4`. Final case rebuild: `codex/build-cpu-final-check.log`.
- `buildgate g++ -std=c++17 -O2 -Iggml/include codex/fa-tail-cpu.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml-cpu -lggml -lggml-base -o build-cpu/bin/fa-tail-cpu`; the model fixture uses the same gate with `-pthread -DGGML_SHARED -DLLAMA_SHARED`, server/common/include/vendor/mtmd include paths, `build-cpu/tools/server/libserver-context.a`, `build-cpu/common/libllama-common-base.a`, and `-lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base`. Each compile uses nice 19 and CUDA hidden.
- `cpugate python3 codex/cpu_identity.py` generates model artifacts; `--analyze` reuses them and runs the FA controls. The fixture writes uint64 token counts; the corrected parser checks tokens separately from logits.
- `codex/cpu-results/identity-check.log`, `greedy-identity.json`, `tail-positive.log`, `tail-mutant.log`, and `codex/streamk-contract-check.log` contain the final results.
- `FROZEN_NAME=fa-kvmax-decode-v3 bash codex/freeze-fa-kvmax.sh`: CUDA 13.2, architecture 120a, NCCL OFF, local CCCL343, disconnected CMake, ccache ON. Shared ccache is read-only because writes outside the worktree are forbidden. `codex/freeze-v3-check.log` records the stamp and freeze.

## Forecast and parent items

The following is conditional on holding the baseline workload/acceptance fixed. Baseline means are D from the recorded stack5 depth summary; gains and resulting times are G. They are forecasts, not GPU results.

| Prefix depth D | Stack5 steady ms/round D | Saving ms/round G | Candidate ms/round G |
|---|---:|---:|---:|
| 8192 | 41.100 | 0.05-0.20 | 40.90-41.05 |
| 32768 | 42.533 | 0.05-0.20 | 42.33-42.48 |
| 90112 | 46.961 | 0.05-0.20 | 46.76-46.91 |

The reported D 0.154724 ms/round stack5-minus-stack4 difference at the deepest prefix is consistent with this scale, but does not isolate FA. At the shallower depths acceptance already differs across those historical arms. Additional mask traffic, shared/register occupancy, L2 residency, conversion paths and stream-k rescaling can erase the predicted saving. The kernel probe is the causal check; depth A/B must retain acceptance and prefill timings.

- `codex/probe-item.toml`: frozen stack5 libraries versus the final freeze, ABBA, identical newly compiled test executable in both arms. Each arm's libraries are selected with `LD_LIBRARY_PATH` and checked with `ldd`; SQL stamps must match that arm's SOURCE. Q 1/4/6/8, both KV types, D head 256/GQA 6, near D 8k/32k/90k, padding D 256/4096. Canary: one patched-arm iteration with short correctness/performance processes, D 180-second internal cap. Full internal cap D 1100 seconds; G 450 seconds estimate.
- `codex/depth-experiment.toml` / `codex/depth-item.toml`: cloned from `stack5-depth-j5q.toml`, stack5 versus final freeze, both `--kv-pad 4096`, J5q, MTP k=5, ABBA and within-arm identity. Canary is one patch boot at the smallest depth, D 600-second cap. `codex/gpu_ab.sh` wraps the same depth harness.

M: both items pass `gpuq submit --check`. Neither was submitted. Depth item was rendered with `python3 -m harness.mkitem` from the eval repository; no `harness/*.py` was edited. Parent item paths and logs are inside this worktree.

M: parser checks pass on recorded GPU FA SQL (provenance in `codex/probe-fixtures/SOURCE`) and the existing real SSE, memory and journal fixtures, including negative controls. Kernel preflight verifies frozen bytes and selected-library resolution. Both harnesses save raw artifacts before gating, enforce budgets, scope journal checks to the current run/boot, and terminate only owned exact process groups. GPU modes require the queue guard. Engagement is the once-per-context `decode KV bound: device=...` dispatch log at `fattn-common.cuh:1244`; its eligibility conditions force the dense scan branch. It proves selection, not timing or the numerical value of the bound.

## Upstream split and open items

Upstream candidate: the CUDA helper, VEC/TILE loop bounds, MMA live partition plus both fixups, and padded-mask tests. These belong together for the target verify path. Remove the local engagement bool/log and all `codex/` harness artifacts from an upstream patch. No public op/API or non-CUDA-family backend implementation changed.

Before upstream discussion or promotion: parent runs both live canaries and A/B items, checks numerical failures, confirms within-arm identity and engagement, and compares prefill and acceptance. Numerical equivalence on CUDA and HIP support remain unverified. The CPU stochastic/Meta-logit discrepancies need separate baseline investigation. No issue, PR, review response or upstream package was submitted.
