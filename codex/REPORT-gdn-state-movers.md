# GDN state movers

Implemented conv snapshots and concat removal together. The CUDA fusion reads the old conv state and new tokens directly, computes the existing convolution and optional SiLU, and writes all conv snapshot planes. The unfused graph remains valid on every backend. Gather removal is deferred. GPU correctness, engagement, and speed are not measured; the frozen build is ready for the parent's A/B.

Labels: M = measured/observed; D = derived from shapes or code; G = estimate. Flags, dimensions, commit IDs, and file line references are identifiers rather than performance measurements.

## Changes and provenance

- `cb2f9457a`: keep convolution graph expansion before the independent GDN-state gather in Qwen3.5, Qwen3.5 MoE, and Qwen3Next; add `test-recurrent-state-rollback --state-movers`.
- `d2c8e6d0d`: add the CUDA fusion, its shared internal matcher, backend cases, and operation documentation.
- Base: `590912a1de2097523db6c8efe185387bb5c7d4fc`; branch: `topic/gdn-state-movers`.
- Frozen code: `codex/frozen/gdn-state-movers`, SOURCE `d2c8e6d0d738d2533b6bc627bbf8a7de7cae6d0a`. Configuration ran with HEAD at this commit. `strings libggml-base.so.0.25.3` contains `d2c8e6d0d` (M); `readelf -d round-trace` shows RUNPATH `$ORIGIN` (M).

No GPU workload, inference run, backend GPU test, profiler, or GPU-query command was executed. Commands used `CUDA_VISIBLE_DEVICES=''`, `GIT_NO_LAZY_FETCH=1`, nice 19, and at most eight threads per command. No network, queue submission, or non-git writes outside the worktree were used. The reference model moved from the brief's old home path to `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf` (M).

## Compliance exceptions

CMake automatically executed `CMakeDetermineCompilerABI_CUDA.bin` to detect native architectures during initial CUDA configuration (M: installed `CMakeCUDAArchitecturesNative.cmake:6`, configure output `No CUDA devices found`). This breached the literal ban on executing any CUDA binary. The probe inherited empty `CUDA_VISIBLE_DEVICES`; no GPU workload ran. It was discovered after the build and cannot be undone. Future builds under this rule must suppress compiler-probe execution before the first CUDA configure.

The commits include `Assisted-by: Codex` because the session's higher-priority commit instruction requires that trailer, conflicting with the brief's requested omission.

## Causes

| Mover | Code evidence and attribution | Status |
|---|---|---|
| `concat_non_cont` | `src/models/delta-net-base.cpp:473` concatenates `[history, channels, sequences]` with transposed new tokens along time. `ggml/src/ggml-cuda/concat.cu:95` dispatches the non-contiguous input path. Recorded GPU0 grid `5120 x 1 x 1`, block 256 (M) agrees with conv channels, rather than GDN state dimensions (D). Brief: 48 calls, 0.56 ms/verify round (M, supplied trace analysis). | Elided for the matched short-batch graph. |
| Conv `cpy_scalar<float,float>` | `delta-net-base.cpp:505-522` creates one window view and cache copy per snapshot. For slot `s`, its input window starts at `max(0, n_tokens-s)` (D). Brief: approximately 288 calls and 0.545 ms/round (M, supplied trace analysis). | All matched copies are written inside the conv kernel. |
| `k_get_rows_float_vec` | `src/llama-graph.cpp:3579` gathers the selected recurrent row; `qwen35.cpp:409` invokes it for GDN state. Recorded grid `1 x 384 x 1`, block 256 (M) agrees with a 393216-element per-GPU state row (D). `getrows.cu:258` requires at least 128 vector blocks; the conv row is too small, so this vector kernel is the GDN gather (D). Brief: 48 calls and 0.30 ms/round (M). | Preserved. |

`codex/recorded-movers.json` records a read-only SQLite query against `attrib-host-s3m/full-a1/decode-8k.sqlite`. Its counts and average durations cover the whole recorded window, not a trimmed verify-round average. The above per-round times come from the brief, not a new benchmark.

The gather is unconditional in the default `build_rs` path, including full accepts with rollback index zero (D). Zero rollback does not guarantee an identity row map: `src0` can change after sequence copies, slot reassignment, or batching (D). `s_copy()` adds the rollback-plane offset and consumes `rs_idx` once at `llama-memory-recurrent.cpp:1366-1382`.

## Design and invariants

`ggml/src/ggml-ssm-conv.h:25` matches a dimension-zero F32 concat, its snapshot views/copies, SSM_CONV, and optional SiLU. It uses `ggml_can_fuse_subgraph` to reject additional consumers of omitted tensors. It verifies window offsets, sequence strides, uniform descending cache slots, non-overlapping planes, and disjoint cache/output writes versus state, tokens, and weights (`:96-129`). Destination views and copy outputs remain observable.

`ggml/src/ggml-cuda/ssm-conv.cu:218` retains the original circular input buffer, weight order, accumulation order, addition of zero bias, and SiLU helper. Its only arithmetic-independent changes are the two-source input accessor and the cache stores. Slot zero is newest; slot `s` stores the history window starting at `max(0, n_tokens-s)` (`:247-255`). Older conv slots clamp to the round-start state when the ubatch is shorter than the snapshot count, exactly as the original graph does.

`ggml-cuda.cu:3767` limits the CUDA path to short batches, supported conv widths, and channel multiples already supported by the CUDA conv backend. Larger patterns, bias-add graphs, extra readers, aliasing, unsupported widths, and other backends use the original graph. The matcher has the existing subgraph helper's node-count bound; larger snapshot counts fall back. No new ggml op, public API, model gate, sampling flag, or cache layout was added.

The enabling graph order is `qwen35.cpp:407`, with the same change in `qwen35moe.cpp` and `qwen3next.cpp`. Expanding conv before the GDN gather makes the fusion contiguous without skipping unrelated work.

State-selection invariants are unchanged:

- Partial accept / bounded `seq_rm`: the selected plane remains `rollback * mem_size + src0`; no plane selection is baked into this fusion.
- Full accept: plane zero contains the final conv state.
- Full removal: the existing memory clear/zero-row path supplies a fresh gathered state.
- Checkpoint save: `state_write()` serializes the selected plane (`llama-memory-recurrent.cpp:784-862`). Restore writes the live state and resets the pending index (`:891-893`).
- Multiple sequences: each snapshot uses the cache's sequence stride and slot stride. The retained gather materializes independent input rows before any cache writes.

Gather removal needs a separate change. A graph-build identity shortcut must include the source-row map and pending rollback index in graph reuse validity. A two-source/indexed GDN consumer must also establish safe input/output aliasing when fused snapshot writes overwrite the cache. Removing the gather without either contract would change rollback semantics. The current fusion deliberately keeps that dependency.

## CPU evidence

| Check | Result |
|---|---|
| Backend cases | 100/100 passed (M), with `K=1..6`, token counts shorter/equal/longer than K, odd channels, multiple sequences, and supported conv widths. Matcher controls reject wrong window offsets, input/cache aliasing, and an exposed concat output. Final elapsed 0.60 s (M), including test setup. |
| Model partial/full accepts | Eight deterministic trials passed (M). Baseline and patch fingerprints of all selected conv/GDN rows, all probe logits, and greedy tokens agree. Includes pending-rollback checkpoint restore and full `seq_rm`. Wrong-plane and dropped-conv-snapshot controls detect changed states (M). Final patch elapsed 16.27 s (M), including model load, at four threads. |
| Existing checkpoint test | Single-sequence rollback/checkpoint test passed (M). |
| Existing np2 split-replay test | Fails on both baseline and patch with max logit difference 13.4578 and NMSE 0.018913 (M), first at sequence zero, position 16. Baseline restores the exact `590912a1d` Qwen3.5 graph order into the otherwise identical CPU library. This failure predates the optimization; np2 end-to-end correctness is not certified. |

These CPU checks execute the unfused backend. They verify its graph order, the matcher, and state-selection regressions; they do not execute the CUDA kernel. Fingerprint agreement is strong finite-workload evidence, not a universal proof of bit identity. CUDA arithmetic and live Meta fusion engagement still require the parent's GPU checks.

Reproduction: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 nice -n 19 bash codex/cpu_checks.sh`. The script renames the CPU-only backend test executable to `cpu-ops`, and uses a local thread-count shim because that test otherwise selects threads from hardware concurrency. `codex/build_cpu_base.py` restores only the loaded Qwen3.5 source into a replacement CPU library. `codex/check_cpu_identity.py`, `cpu-identity.json`, `cpu-ops*.log`, `cpu-state-movers-*.log`, and `cpu-rollback{,-base}.log` contain the evidence. The optional `--state-movers` mode emits fingerprints for cross-build comparison; it is not a standalone prefix-equivalence oracle.

## Predicted benefit and replay alternative

With the serving verify shape, removing concat plus six snapshot launches eliminates 336 kernel launches per GPU per verify round (D), retaining one fused conv launch per layer. Conv snapshots still write 16.875 MiB/GPU/round (D: 48 layers x 6 slots x 3 history values x 5120 channels x 4 bytes). The concat intermediate is 8.4375 MiB/GPU (D).

| Part | Expected verify-round saving |
|---|---|
| Snapshot fusion | 0.35-0.50 ms (G) |
| Concat removal | 0.40-0.55 ms (G) |
| Additional fused-kernel work | Cost 0.05-0.20 ms (G); the parts cannot be independently timed inside the fused kernel. |
| Combined | Approximately 0.7-1.0 ms (G). No measured GPU speed claim. |
| Gather | Zero change (D); the brief's 0.30 ms remains. |

Replay is analysis only. A GDN plane is 72 MiB/GPU (D: 48 layers x 24 local v-heads x 128 x 128 x 4 bytes); six snapshots write 432 MiB/GPU (D). A conservative round-start copy plus final-state write costs 144 MiB/GPU, saving 288 MiB/GPU in state writes (D). Preserving the old plane by switching live planes could save 360 MiB/GPU instead (D), but requires new cache/selection machinery. Stored per-layer q/k/v/g/b inputs would be approximately 5.68 MiB/GPU for six rows (D, assuming the observed TP conv-channel partition and scalar g/b per v-head). Accepted-prefix replay is estimated at 0.4-0.8 ms/round (G); net benefit is unknown. It adds state-only kernels, input ownership, checkpoint rules, and another replay boundary. It is not implemented.

## Frozen build and GPU item

CUDA configuration used `GGML_CUDA=ON`, toolkit `/usr/local/cuda-13.2`, `GGML_CUDA_NCCL=OFF`, requested architecture `120`, Release, and read-only local CCCL 3.4.3 with FetchContent disconnected. This stack's CMake normalizes `120` to `120a` (M), matching the base build. Both `llama-server` and `test-backend-ops` built successfully (M); neither CUDA binary was executed. `round-trace` is a renamed copy for the parent. Build/configuration logs and `ggml-stamp.txt` are under `codex/`.

Item: `codex/gdn-state-movers.item.toml`, generated by `python3 -m harness.mkitem` from `codex/gdn-state-movers.experiment.toml`. Base is `~/repos/qwen38-weekend/frozen/stack4`; patch is the local freeze. J5q, TP2, np1, k=5, depths 8k/32k/90k, ABBA, cross-arm native token identity, and two boots per arm (D plan). `gpu_ab.sh` delegates to the shared harness; it starts nothing in preflight/capture mode and requires gpuq for GPU modes.

- `gpuq submit --check` and renderer `--check`: passed (M). Nothing submitted.
- CPU preflight and launch capture: passed (M). Preflight needs an explicit `--port 18123` in this sandbox to avoid creating a socket.
- Real parser replay: passed (M) against the supplied SSE/memory/journal fixtures and the recorded stack4 native-token depth artifact. The latter yields 92 steady intervals and 512 native tokens (M). Mutated token identity and an empty stream fail (M). Evidence: `gpu-parser-tests.json` and `test_gpu_parsers.py`.
- Canary: one patch boot at 8k, budget 600 s (D plan). It checks native token IDs, size/steady/prefill gates, model/launch proofs, and frozen bytes. Cross-arm equality is checked by the full ABBA. All GPU execution remains pending.
- Budget: 3450 s total (D). Stage estimate 4 x 420 + 180 = 1860 s (G), below 70% of that budget (D). Recorded stack4 adoption boots took 224-226 s each (M, read-only console artifact), supporting the estimate.

For nsys, the parent can create a new item from `work/attrib-llama-stack4/attrib-llama-stack4.toml`, retaining its script and guards, with `STACK` set to this freeze's `build/bin`, `STACK_LABEL=gdn-state-movers`, fresh output paths, and all frozen files repinned. Do not modify the existing queued item's files. The script already accepts those variables. Expected engagement: `ssm_conv_state_f32` replaces verify conv/concat/snapshot kernels; the `1 x 384` GDN gather remains. The existing attribution classifier matches `ssm_conv`, so it already categorizes the new kernel. The freeze also includes the CUDA backend test cases for the parent's kernel-suite item.

## Upstreamability and open work

The snapshot/`n_rs_seq` machinery and the GDN snapshot-copy CUDA fusion exist in the available upstream checkout `ed7ac35e1` (M source inspection), not just this serving fork. The changed Qwen graph pattern exists there too. The graph-order change, generic matcher, kernel, and backend cases can apply independently of MTP sampling, Meta, or the local host optimizations. No package tree was created; future package files belong under `codex/upstream/llama.cpp/` for the parent to merge into the authoritative upstream tree.

The natural split is the two code commits: graph ordering with the rollback comparison mode, then CUDA fusion with its backend cases. Gather indexing and replay each need separate design/testing. Existing issue/duplicate checks remain for the contributor because this task forbids network access. No PR text or submission was produced.

Before adoption: run the GPU canary/full identity A/B, verify nsys engagement, and execute the CUDA backend cases. Investigate the baseline np2 split-replay failure separately. Bias-add conv graphs, long batches, graph patterns outside the matcher's bound, and custom conv builders retain the unfused path.
