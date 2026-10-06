# sm_120 small-J MMQ

## Conclusion

Candidate, not an adopted optimization. The config uses I=64, 128 threads and launch-bounds occupancy 2 for J=8/16/24 on the compiled sm_120 architecture, for IQ4_XS, Q4_K, Q5_K, Q6_K and Q8_0. GPU correctness, memcheck, exact tokens and speed require the parent-run items. No GPU application, GPU query, network call or item submission is authorized in this task.

Labels: M measured or directly observed, D derived, G guess. Commits, paths, tensor dimensions, architecture names and configured limits are identifiers rather than measurements.

## Design

- `ggml/src/ggml-cuda/mmq-config-blackwell.cuh`: preserve each type's SRAM layout and K iteration; reduce I and thread count together. Match host and device config selection against the compiled architecture. sm_121 compiled as sm_121, other architectures, other types and J>=32 retain their original configs.
- `ggml/src/ggml-cuda/mmq.cuh`: use tiling when tile-wave efficiency is at least 66% for the new config, rather than 90%. Below that threshold, retain stream-k and its fixup. The kernel stays a stream-k specialization; grid=tiles gives each block a complete tile and skips fixup. No model-specific row count enters dispatch.
- `GGML_CUDA_MMQ_SMALLJ_DISABLE` is a local compile control. Runtime-only switching cannot safely change a constexpr tile layout without adding duplicate kernel specializations. The perf item instead uses one frozen test executable and switches its frozen libraries via `LD_LIBRARY_PATH`: stack6 vs the candidate.
- `GGML_CUDA_MMQ_SMALLJ_NEGATIVE_CONTROL` negates tile accumulators on the new path. It changes no indices or allocation sizes. Its separate frozen library must fail every selected CPU-reference comparison; it is never a performance arm.
- `GGML_CUDA_DEBUG_MMQ_SMALLJ` gives the canary an explicit config/path record. Full perf removes it.

The five requested TP2 shapes match `SP/analysis/gap-next/q1_tensors.txt`: gate/up (8704,5120), down (5120,8704), output (5120,6144), GDN gate (3072,5120), attention Q (6144,5120). Additional correctness cases cover partial I tiles and low tile counts. Q4_K is included because it also enters J5q and appears in the recorded model inventory.

The depth items retain the recorded harness's k=5. With two busy slots this produces 12 verify columns. k=3 produces at most 8 columns for two slots and does not exercise the target MMQ transition.

## Evidence and predictions

Source reports read before implementation: `SP/gpu-sampling/REPORT.md`, `items/96b-attrib.md`, `SP/patchA/REPORT.md`, the final findings of `SP/codex-reviews/patchA.txt`, and `SP/analysis/gap-next/q1_tensors.txt`. SP is the scratchpad path in the task brief. The graph-cache review's ownership defects are unrelated to this config change; no scheduler or sampler code is changed.

D: row-tile counts become 48/80/96/136 for 3072/5120/6144/8704 rows. At 36 SMs their tile-wave efficiency is 66/74/88/94% after integer truncation. All select grid=tiles, with no partial K tile or fixup. The former 128-row configs have 24/40/48/68 tiles; the first three select a 36-block stream-k grid under the original threshold.

D: dynamic shared memory for the candidate is 21024/22080/23136 bytes per block at J=8/16/24. The original config needs 40992/42048/43104 bytes. Both Q8 layouts and Q6_K have a 76-word SRAM stride here. Static shared memory and register counts must come from ptxas diagnostics, separately compiled per type and mode. Occupancy 2 is a launch-bounds target; it is not a measurement of resident blocks or achieved occupancy.

M from the brief's in-model trace: IQ4_XS GEMM component at 90k is 24.67 ms/round for np2, versus 21.99 for np1; stream-k fixup is 0.27 ms/round. D conditional projection: removing those fixups alone gives 24.40 ms/round for that component, and `R_np2 - 0.27 ms` for the complete round if core kernel time is unchanged. G: candidate round improvement 0.3-1.5 ms; no absolute np2 round time or GPU speed claim is established by this work. A cold probe and the model A/B can falsify that guess.

Cold probe: each graph rotates independent weight tensors with total weight storage strictly greater than 64 MiB. Each node retains exactly the requested (m,k,n) and one batch, so rotation does not multiply the stream-k grid. The reported time divides by the actual MUL_MAT node count. Activation quantization and launch costs remain included, as in the stock perf runner. A batched rotation was rejected because it changes the scheduling problem. Cold timings still need in-model confirmation.

## Verification

Pending the required idle gate: CPU build, host-only table contracts, recorded-artifact parser tests, CUDA build and ptxas resource diagnostics. The first CPU build started after an IDLE status; when the next `spec-trace-j5q` item was observed RUN, this worktree's exact build PID and descendants were terminated. No CPU model or backend executable was run. Parent should treat that item's host timings as potentially affected by the overlapping build interval.

Commands are in `codex/build.sh`: every compile checks for any RUN item and uses `bin/buildgate`; every CPU test checks the same gate and uses `bin/cpugate`. CUDA configuration is stack6's disconnected local CCCL configuration, with architecture 120 normalized by CMake to 120a, NCCL OFF and ccache ON. All author-side commands keep `CUDA_VISIBLE_DEVICES=""` and all git commands use `GIT_NO_LAZY_FETCH=1`.

The test executable is compiled for CPU but not run. CPU-side testing consists of the actual config table compiled into a host-only executable plus parser replay. CUDA-reference comparisons and the arithmetic negative control run only in the prepared GPU item.

## Parent-run items

Prepared by `codex/gen_items.py`, then checked with `gpuq submit --check`; never submitted here:

1. `codex/items/memcheck.gpuq.toml`: five types, seven widths, five TP2 shapes plus partial/low-occupancy cases and broadcast/non-broadcast MUL_MAT_ID; compute-sanitizer `--log-file`; safe arithmetic negative control after clean memcheck.
2. `codex/items/perf.gpuq.toml`: one GPU, no sanitizer, hot and cold probes, base/patch/patch/base process order. SQL goes to files. Requires a full memcheck PASS tied to the frozen checksum manifest.
3. `codex/items/iq4-np2.gpuq.toml` and `j5q-np2.gpuq.toml`: stack6 vs candidate, ABBA, both slots busy, identity off for timing-dependent np2 batching, existing overlap and cache gates.
4. `codex/items/j5q-np1.gpuq.toml`: regression at 8k/32k/90k with cross-arm native token identity.
5. `codex/items/j5q-prefill.gpuq.toml`: np1 ub512, fresh 32k prefill per boot, ABBA, plus decode and cross-arm identity. Read prompt_ms from each boot's recorded prefill response.

Each depth canary has one candidate boot. The kernel canaries use separate processes for hot/cold modes or clean/negative libraries. All canary caps are 600 s or less. Depth/prefill items use the shared read-only harness and its exact-PID cleanup, journal cutoff, cache, overlap, native token and frozen-byte gates. Full kernel perf is estimated at 8 x 300 s (G) within a 3600 s run budget; memcheck at 1100 s (G) within 2400 s. Stage expiry reports budget, not a socket timeout.

`codex/gpu_ab.sh` is a parent-only convenience entry with `GPUQ_ACTIVE=1` and the full-memcheck prerequisite. Submit the generated items only in the order above; depth items rendered by the shared mkitem entry do not independently enforce the memcheck prerequisite.

## Risks and upstream split

- Register allocation and achieved occupancy are unverified until compilation diagnostics are collected. Lower I can reduce memory efficiency or increase scheduling overhead even while eliminating fixup.
- Stream-k vs full-tile accumulation changes floating-point grouping. CPU-reference tolerance is necessary but does not prove identical logits or sampling outcomes. np1 identity is a separate gate; np2 identity stays off as required. No sampling algorithm is changed, and no distribution-equivalence claim is made for unmeasured model outputs.
- Shared kernels also serve MUL_MAT_ID. Broadcast and non-broadcast expert cases are included in the prepared memcheck/reference suite; their GPU results remain pending.
- Keep upstream only the architecture/type/J config selection and the general tiling-efficiency rule after GPU evidence. Remove the local compile controls, negative mutation, debug logging, environment-selected test inventory and codex artifacts. Do not create a second upstream package tree; the parent integrates any package under `patches/upstream/`.
- No PR, reviewer response, push or upstream submission was created. Local topic commits have no AI trailers as required by the owner.
