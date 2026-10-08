# K4: indexed conv-state read (GGML_OP_SSM_CONV_RS)

## Conclusion

Implemented in general form (new op, CPU + CUDA + Meta, automatic exact fallback, no env var). Mac CPU gates pass (M). CUDA is uncompiled and unrun on this host: every CUDA claim is a window gate. Meta slicing is untested on a real two-device run.

## Commits (base 0cfc0723e, branch topic/s10-k4)

- b596fe59f `ggml: GGML_OP_SSM_CONV_RS, indexed conv-state read with snapshot writes (CPU, CUDA, Meta)` (general; includes test-backend-ops cases)
- d2780f4b6 `llama: read the recurrent conv state by id through ggml_ssm_conv_rs, gathered fallback` (general; builder + model-level gate scenarios)
- 583dc2d62 `local: SSM_CONV_RS off switch, mutants and engagement log` (GGML_SSM_CONV_RS_OFF / _MUTANT / _LOG; CPU+CUDA)
- a2e999305 `K4 window and Mac payload`; fb54ddee1 PURPOSE; 3457b9794 gate fixes; 1fff13aa8 Mac evidence (`codex/k4-evidence/`)

## Design

Op `ggml_ssm_conv_rs(x, w, states, ids, snap)`: x `[n_t,d_inner,n_s]`, w `[d_conv,d_inner]`, states `[(d_conv-1)*d_inner, rows]` (whole cache), ids I32 `[n_s]` (state rows read by sequence), snap = 3D view of the cache `[(d_conv-1)*d_inner, n_s, n_slots]` written in place; result `[d_inner,n_t,n_s]`. Replaces `build_rs` conv gather + concat + K snapshot cpys + ssm_conv (`build_conv` in `src/models/delta-net-base.cpp`; used by qwen35, qwen35moe, qwen3next).

Why a new op, not an SSM_CONV extension: an extended SSM_CONV would need explicit rejection in ~9 backends I cannot compile, and an unpatched backend would silently miscompute. A new op is unsupported by default everywhere; GGML_OP_COUNT 101 -> 102 and RPC patch version bumped (RPC peers must match).

Why exact:
- Read-before-write: the cache row layout is `[channel][d_conv-1]`, so hazards are per channel. One CUDA thread (CPU: one thread's channel range) owns a channel for ALL sequences, reads every sequence's state first, then writes snapshots. Overlapping, swapped or repeated ids (fresh sequences all read the cleared cell) need no barrier.
- Snapshot semantics: slot k (0 newest) = concat columns from `max(0, n_t-k)`, length d_conv-1; slots older than the batch hold the input state. Pure copies.
- Arithmetic copied per backend from the replaced path: CPU `sumf += col*c` ascending, no `+0.0f`; CUDA `sum += win*w` ascending, `sum += 0.0f`, `ggml_cuda_op_silu_single`; fused with the following SILU via `ggml_cuda_can_fuse` (output must not overlap any src).
- Extra cached states (n_rs > n_seqs): `build_rs` keeps its gather; `src/llama-graph.cpp` now forces the gather node ahead of the getter so extras are read before snapshots overwrite them (the GDN getter needed `n_rs == n_seqs`; the conv path does not).
- Fallback: builder probes `ggml_backend_dev_supports_op` on the cache device. CUDA supports d_conv in {3,4,5,9,15}, F32, n_t <= 32, n_s <= 8; otherwise (e.g. prefill) the old path is built unchanged. Requires `offload_kqv` and F32 cache/kernel.
- Meta: x/w AXIS_1, states/snap AXIS_0, ids MIRRORED, out AXIS_0 (same pattern as the GDN snapshot dst). Unverified on hardware.

## Mac results (M; bw-mac run s10-k4-a5, HEAD 3457b9794, CPU only, -j4)

stages (wall s): build 92, op tests 11, mutants 9, model gate 146, np2 driver 277, static checks 0 (`codex/k4-evidence/stages.tsv`).
- test-backend-ops CPU `-o SSM_CONV_RS`: 328/328 tests passed; same-backend byte identity vs the gathered graph (output + whole cache, 3 replays with changing ids) PASS cases=800 (channels {128,384} x tokens {1,2,4,6,32} x seqs {1,2,3,8} x K {1,2,6,9} x 5 id modes: identity, reversed/swapped, rotated, all-read-one-cell, non-written cells). Existing SSM_CONV cases unchanged.
- Off switch (fallback): `PASS cases=0 unsupported=800`.
- Model gate (Qwen3.5-4B UD-Q4_K_XL, raw logits memcmp vs gathered arm, fills 0/0x3e): scenarios two slots, swapped cells, joint start, extra state, plus new "accepted slot" and "accepted slot, swapped cells" (seq_rm to a snapshot then replay). RS arm byte-identical to gathered arm at n_rs_seq 8 and 0. Engagement per phase from graph probe (conv nodes by id vs gathered; extra-state phase gathers only extras).
- np2 driver, plain and Meta(CPU,CPU): tokens/counters JSON identical off vs on. Not shown that the RS node engaged in these arms (no probe line); engagement is shown by the model gate.

## Negative controls (each fails on a mutated path)

- GGML_SSM_CONV_RS_MUTANT=1 (read state after earlier sequences' snapshots): op identity FAIL (output) and model gate FAIL ("swapped cells: verify batch ... logits differ").
- MUTANT=2 (older snapshots shifted one token): op identity FAIL (cache) and model gate FAIL ("accepted slot: replay after rollback ... logits differ"). It escaped the original four scenarios; the two accepted-slot scenarios were added because of it.
- `--conv-expect gathered` with RS off and `--conv-expect rs` fail if the wrong form is built.

## Window (not submitted)

`codex/bw-k4.toml` + `codex/bw-k4.sh` (filler, priority 18, preemptible=false, host_sensitive=false, same GPU UUIDs/recipe: arch 120a, CUDA 13.2, NCCL OFF, CCCL 3.4.3, DeviceTopK count 25; no freeze, build in `build-cuda` under this tree). Stages: CUDA build; stamps; on CUDA0 and CUDA1 test-backend-ops SSM_CONV_RS (identity, fused silu=1 and plain silu=0 engaged, unsupported>0 for n_t=33/n_s=9), mutants 1/2 must fail, off control; 4B model raw-logit gathered vs RS vs mutants on each GPU; Meta (`-sm tensor`) run recorded, non-gating.
- `bash codex/bw-k4.sh --preflight`: PASS preflight
- `gpuq submit --check codex/bw-k4.toml`: `codex/bw-k4.toml: bw-k4: valid`; `bash -n` and shlint pass on both .sh files.
- Pinned files list includes the K4 sources: editing them after submit parks the item.

## Expected effect

D: -0.30..-0.35 ms/round at np2 8k/90k (removes the per-layer conv gather + concat + snapshot cpys + separate ssm_conv launches across 48 conv layers; from the D cause report). Falsifier: with the window build, np2 8k/90k ms/round improves by < 0.1 ms with `ssm_conv_rs engaged` in the log, or the conv ops do not drop in the nsys kernel count. G until measured.

## Every source change (including incidental)

ggml.h (enum, API), ggml.c (names, static_asserts 102, constructor), ggml-rpc.h (patch version, assert), ggml-cpu.c (dispatch, n_tasks, work size), ggml-cpu/ops.{h,cpp}, ggml-backend-meta.cpp (handler), ggml-cuda/ssm-conv.{cu,cuh}, ggml-cuda.cu (compute, supports_op, can_fuse/try_fuse), llama-graph.cpp (gather-before-getter expand), models/{delta-net-base.cpp,models.h,qwen35.cpp,qwen35moe.cpp,qwen3next.cpp} (build_conv; unused `conv_kernel_size` removed), tests/test-backend-ops.cpp, tests/test-recurrent-state-rollback.cpp. Local commit: ggml-cpu.cpp, ops.cpp, ssm-conv.cu, ggml-cuda.cu. `codex/PURPOSE` is overwritten by `bw wt new`.

## Blockers / uncertainty left

- CUDA kernel and fusion never compiled; FMA contraction identity vs the old fused `ssm_conv_state_f32` is assumed (window byte gate decides).
- Meta split-state handler untested on two devices; qwen4exp and other models still use the gathered path.
- Other backends (Metal, Vulkan, SYCL, ...) report the new op unsupported and fall back; not compiled here.
- RPC patch-version bump is a wire-compat change.
