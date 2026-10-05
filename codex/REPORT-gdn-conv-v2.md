# GDN convolution fusion V2

V2 retains the materialized concat and fuses snapshot writes into SSM_CONV plus optional SiLU. CPU gallocr checks reproduce the dead-input/output alias and accept every layer; restoring the V1 alias restriction fails. CUDA compilation passes. GPU execution, numerical identity and speed remain unmeasured by this worker.

Labels: M = measured/observed, D = derived, G = estimate. Shapes, CLI parameters, commits and source line references are identifiers, not timing measurements.

## Design and invariants

- `ggml/src/ggml-ssm-conv.h:25`: match from the first snapshot CPY, after CONCAT has executed. Keep CONCAT as an ordinary graph node. Do not add allocation dependencies or alter the graph.
- Match F32 concat along dimension zero, contiguous concat/output, compatible history/channels/sequences and weights. Accept only VIEW/CPY nodes before SSM_CONV, with optional adjacent SiLU. The bounded search rejects intervening operations and external consumers of the removed raw convolution output.
- Snapshot copies must cover descending physical slots of one cache, with equal sequence strides and disjoint slot spans. Slot `s` reads the concat window starting at `max(0, n_tokens-s)`. The newest cache view supplies the base address. Preserve every snapshot, including clamped history snapshots for short batches.
- Cache writes must not overlap concat, weights or output. Retain conservative output-vs-concat/weights checks. Output may overlap the dead concat inputs because the fused kernel never reads those inputs.
- `ggml/src/ggml-cuda/ssm-conv.cu:211`: preserve convolution accumulation order, zero-bias addition and the existing SiLU function; copy snapshot floats without arithmetic. CUDA eligibility requires aligned channels, short batches and one of the existing convolution widths. Other shapes/backends execute the original operations.
- `ggml/src/ggml-cuda/ggml-cuda.cu:3774`: report every candidate rejection and successful launch under `GGML_CUDA_DEBUG` or `GGML_CUDA_DEBUG_SSM_CONV`. Evidence includes device, CUDA graph UID and token count; successes also include channels, sequences, slots and fused node count. A debug census counts every computed SSM_CONV and every snapshot fusion per submitted/captured CUDA graph, including graphs that never reach the matcher. Diagnostics are disabled by default.
- `ggml/src/ggml-cuda/concat.cu:83`: flatten non-contiguous concat when output `ne0 <= 32`, distributing adjacent logical elements over a one-dimensional grid. Preserve source/destination strides and all concat axes. Larger rows keep the existing path.

No sampler or random-number code changes. CPU equality proves graph-order equivalence for the fixture; CUDA arithmetic identity still requires the queued bitwise benchmark and token A/B.

## Engagement and CPU evidence

- M: `cpu-ops-v2.log` passes 100/100 SSM_CONV_STATE_FUSION cases. Cases cover snapshot counts 1..6, odd channels, short batches, multiple sequences and supported convolution widths, with and without SiLU.
- D: the gallocr matrix in `tests/test-backend-ops.cpp:4481` contains 72 placements: attention-sized scratch for depths 8192/90112, sequences 1/2, token widths 1/4/6, snapshot counts 1..6. M: all matcher assertions pass, including dead-input aliases, malformed snapshot offsets and externally exposed raw outputs. These depth allocations simulate placement pressure; they are not actual long-context model executions.
- M: `bench-layout-v2.txt` reports `galloc-model eligible=48/48 silu_overlaps_tokens=48/48`. The benchmark explicitly expands token projection before the state gather, then concat, snapshots, convolution and SiLU. M: `bench-layout-negative.status` records exit 8 when `--old-contract 1` restores rejection of the dead token alias.
- M: `cpu-concat-v2.log` passes 252/252 concat cases. Added cases cover the model's transposed token input, all axes, non-unit outer dimensions, strided inputs, F32/F16/I32 and the dispatch boundary.
- M: `cpu-identity-v2.json` passes np1 and np2: 8 trials each, 24 paired fingerprints total, covering accepts 1..6, full sequence removal and sequence-state restoration. Selected-state/logit fingerprints and greedy probe tokens match the stack4 graph-order baseline exactly; raw array dumps were not saved. Wrong snapshot index, dropped snapshot and mutated token identity are detected. M elapsed CPU run seconds (base/patch): np1 13.58/12.65, np2 17.39/12.71; these include loading and are not speed comparisons.
- M: `model-engagement-v2.log` passes all 24 GDN layers in every one of 12 real allocated model graphs: np1/np2, prefill lengths 32/512 and token widths 1/4/6. Dead token/output aliases occur in all 24 layers of every token-width 4/6 graph. The probe rebuilds use counts over the scheduler's already allocated tensors, without reallocating them. M: elapsed CPU probe time 48.20 s.

The CPU baseline library restores Qwen3.5 graph order from stack4 `590912a1d`, sharing the unchanged CPU kernels and other stack code. It is not an independently rebuilt stack4 tree. `codex/build_cpu_base.py` verifies the replacement source occurs in the unity unit. The rollback test compares selected recurrent state plus subsequent logits and greedy probe tokens; it does not execute a probabilistic MTP server.

Reproduction uses `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, nice 19 and at most the specified thread limit:

```
bin/buildgate cmake --build build-cpu --target test-backend-ops test-recurrent-state-rollback -j4
bin/buildgate python3 codex/build_cpu_base.py
bash codex/cpu_checks_v2.sh
bin/cpugate codex/cpu-bin/model-engagement /mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
```

`bin` above is `/home/xanewok/repos/qwen38-eval-worktrees/bin`. Model checks use explicit context 4096, CPU layers and four threads. The queue status lock requires sandbox escalation; the initial CPU operator logs show the status-lock failure, corrected before model runs. The operator results are correctness checks; their timings are not performance comparisons. Model checks use the working gate. Ccache remains enabled with the shared cache read-only and its temporary directory inside this worktree.

## CUDA build and freeze

M: CUDA server and backend tests compile with toolkit 13.2, NCCL OFF, requested architecture 120 (normalized by this stack to 120a), and local CCCL 3.4.3. `build-cuda-head.log` and `configure-cuda-head.log` record a configure/build at HEAD `73b27e30b63181d43b564a01749beda90c7b3348`. No built CUDA server, benchmark or backend test was executed.

Freeze: `codex/frozen/gdn-conv-v2-proof`, created by `bin/wt freeze` with `QWEN38_WT_ROOT` redirected inside the worktree. The benchmark was added and SHA256SUMS regenerated before queue manifests existed. M: `freeze-stamp-v2.txt` records the matching ggml stamp; `freeze-runpath-v2.txt` records benchmark RUNPATH `$ORIGIN`. SOURCE and SHA256SUMS pin the binary bytes. Later commits contain verification and queue artifacts without changing these compiled source files.

**Rule breach:** initial CUDA configuration executed CMake's native-architecture ABI probe despite empty `CUDA_VISIBLE_DEVICES` and explicit architecture 120. M: `cuda-probe-evidence.txt` records `No CUDA devices found.-real`; `/usr/share/cmake-3.28/Modules/Internal/CMakeCUDAArchitecturesNative.cmake:6` invokes that binary. This violated the literal prohibition on executing CUDA binaries. Subsequent configurations use cached compiler information and the freeze recipe sets `CMAKE_CUDA_COMPILER_FORCED=ON` to prevent another native probe. No deliberate GPU workload, GPU query, server launch, network request or queue submission occurred.

## GPU items and validity gates

- `codex/gdn-conv-v2.item.toml`: stack4 freeze vs this freeze, J5q, TP2, probabilistic MTP depth k=5, depths 8k/32k/90k, ABBA, two boots per arm. `gpu_ab.sh` reuses the shared depth harness for frozen hashes, model hashes, exact-PID ownership, scoped kernel journal, request persistence, budget and native token/draft-counter identity.
- Added engagement gate: every verify-shaped candidate rejection, missing census or census with `conv != fused` invalidates the patch boot. The union of names must cover all GDN layers on both devices. Record individual CUDA graph UID counts in `conv-engagement.json`. Meta splits graphs at AllReduce boundaries (`ggml-backend-meta.cpp:2469`), so requiring all layers in each CUDA fragment would be incorrect.
- `codex/gdn-conv-v2-bench.item.toml`: rebuilt benchmark against the frozen libraries, both GPUs separately, model-shaped gallocr order, short token widths and all requested snapshot counts. Each invocation checks bitwise output and every cache plane against its unfused mode before recording timing. Gates require actual CUDA fusion diagnostics for every fused mode. Child processes stop by their exact Popen PID, including timeout or termination.
- Canary: depth item uses one patch boot at 8k; benchmark item uses one small invocation that executes every mode and parser. Both fit the contract's ten-minute ceiling. Full cross-arm identity necessarily requires the full depth run.
- M: both manifests pass `gpuq submit --check`; neither was submitted. Bench preflight and real parser tests pass. Native depth preflight and final launch capture also pass; capture confirms the renamed server and patch-only diagnostics.
- M: `gpu-parser-tests-v2.json` replays real SSE/memory/journal fixtures, the recorded stack4 depth request and the supplied GPU benchmark artifact. Wrong tokens, insufficient streams, corrupted/NaN benchmark rows and missing modes fail. Real V1 GPU logs fail both V2 engagement parsers. Synthetic positive/partial-coverage tests cover full and Meta-fragmented graph logs; no real V2 GPU log is available yet.

`write_gpu_items_v2.py` renders the shared depth contract and changes only its entry point to the local engagement wrapper. The kernel item shares the exit-code classes, signatures and frozen-file pins. GPU runs remain exclusively for the parent.

## Expected benefit and optional V1

Supplied bench-a2 evidence, not a new run: M, unfused/model-like gallocr approximately 19.1 us/layer; no-alias V1 fusion approximately 7.1 us/layer; disabled fusion approximately 19.5 us/layer. Those timings include different concat treatment and must not be described as V2 results.

G: if retained flat concat costs 2 us/layer and V2 convolution/snapshots retain the supplied 7.1 us/layer cost, V2 would cost about 9.1 us/layer. D, conditional on those assumptions: `(19.1 - 9.1) * 48 / 1000 = 0.48 ms` saved per verify forward per GPU. G: approximately half a millisecond per round when a round has one matching verify; do not add the two GPUs' savings. V2 reads concat with different coalescing from V1, so retaining the 7.1 us/layer kernel cost is an assumption. The trace census sums are not an additive estimate of the microbenchmark's captured-graph latency.

M/code finding: Meta's backend interface sets `graph_optimize = nullptr` at `ggml/src/ggml-backend-meta.cpp:2814`, so CUDA allocation-dependency hooks are not forwarded through TP. Dropping concat would require recognizing the pattern before data addresses exist, retaining state/token inputs until convolution/SiLU output allocation, and forwarding/mapping those dependencies through Meta before parent gallocr allocation. CUDA-only hooks cannot fix the TP allocation. G: once flat concat costs about 2 us/layer, dropping it offers only about another 0.1 ms per verify; the allocation/Meta work is separate and unimplemented.

## Upstream split and remaining work

- `aaa00c3fb`: generic CUDA concat optimization plus concat cases; independently reviewable.
- `91d7eefa6`: V2 snapshot fusion, debug reasons, gallocr matrix and rollback test extension. Fold with the inherited V1 addition `d2c8e6d0d` when preparing upstream code. Retain the earlier model graph-order prerequisite `cb2f9457a` as a separate change; without adjacent snapshot/conv nodes, the matcher safely falls back.
- `aaa6c5e36`: debug graph census, folded with the V2 fusion upstream.
- `e5df73194`, `73b27e30b` and later artifact commits: local benchmark, CPU evidence, freeze and queue drivers; not upstream implementation.

Open: run the canaries and full items; verify CUDA exactness and engagement for actual TP graphs at each depth; run the compiled CUDA backend-op cases for the generic concat axes/types. Measure V2 before deciding whether to retain it; inspect capture/log volume because diagnostics can add host cost when CUDA graphs are recaptured frequently. No upstream discussion/duplication search or submission was performed under the no-network instruction.
