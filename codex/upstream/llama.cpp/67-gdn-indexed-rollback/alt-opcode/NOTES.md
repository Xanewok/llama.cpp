# 67 gdn-indexed-rollback

Lever C: with recurrent-state snapshots (`n_rs_seq > 0`, MTP/speculative rollback) the GDN layers gather the selected rollback row of each sequence into a compact state (`build_rs` -> `GET_ROWS`) before every `GATED_DELTA_NET`. The new op `GGML_OP_GATED_DELTA_NET_INDEXED` takes the whole cache `[S_v, S_v, H_v, n_state_rows]` plus an I32 `[n_seqs]` row index and reads the row in place, so the gather is not built. Four commits on master `7049ff0cb` (apply.txt); numbers: EVIDENCE.md. Status `needs-GPU-check`.
Commit messages are worker drafts; the human rewrites them and writes the PR text.

## Upstream split
Upstream content (4 commits, 19 files, +204/-71):
1. `ggml : add GGML_OP_GATED_DELTA_NET_INDEXED`: `ggml.h`/`ggml.c` (op, shared `_impl`), CPU forward (one `state_row` offset in `ggml_compute_forward_gated_delta_net_one_chunk`), meta backend (index tensor must be MIRRORED, state heads split as before), RPC (`supports_op` rejects it; `RPC_PROTO_PATCH_VERSION` and the `GGML_OP_COUNT` assertions bumped).
2. `cuda : read gated delta net state by row index`: kernel reads `state_indices[sequence]`; the GDN+CPY snapshot-store fusion accepts the new op; `supports_op` returns true only for `n_seqs == 1`.
3. `tests : add GATED_DELTA_NET_INDEXED cases`: 12 plain cases (S 16..128, B 1..2, K 1..6, a permuted/KDA case) and 3 through the cache-fusion test (`indexed=1`, swapped rows at np2).
4. `llama : read the recurrent state in place in GDN layers`: `build_recurrent_attn` builds the state itself and picks the indexed op when `n_rs_seq > 0`, `offload_kqv`, `n_rs == n_seqs` and the state buffer's device supports the op; otherwise the old `build_rs` path. Seven model files lose the `build_rs` + reshape lines and the `s` argument.

Local only, not shipped:
- `f5c5bd54f` trace (`GGML_CUDA_DEBUG_GDN_STATE` env read in the per-graph CUDA path, `GGML_LOG_INFO` line, and `ggml_round_trace_count("gdn_gather_skipped")`). It depends on package 27 (round trace) and exists to prove engagement in the A/B. An upstream-acceptable replacement would be a `GGML_LOG_DEBUG` line behind an env var cached once at init; recommendation is to ship nothing: engagement is visible from the graph (the op appears in `GGML_SCHED_DEBUG` splits) and from `test-backend-ops` support output.
- `38c59b368` (CPU fixtures `gdn-cpu.cpp`, `gdn-op-cpu.cpp`, freeze/gate scripts), `06b129688` and `b6bdd923e` (bench/gpuq items): harness only.
- From `03fe64c15`: the `docs/ops.md` legend line and `docs/ops/CPU.csv` rows are dropped (generated files; whoever regenerates them adds `GATED_DELTA_NET_INDEXED`).

## Design
- Separate opcode instead of an optional `src[6]` on `GATED_DELTA_NET`: every backend that implements the plain op (Metal, Vulkan, SYCL, WebGPU, OpenVINO, HTP, ET) ignores `src[6]` and would read the wrong state silently. A new opcode is "unsupported" everywhere by default, `llama_context` keeps the gather (`ggml_backend_dev_supports_op` at graph build). Cost: 1 enum value, appended last (existing values unchanged), RPC patch version bump.
- Kernel ordering: `ggml_cuda_pdl_sync()` moved before the index read (the index is a graph input written by the previous node under PDL).
- `n_rs == n_seqs` only: with extra rows `build_rs` relocates states (`states_extra` copy) that a direct read could see overwritten, so the gather stays.
- CUDA single sequence only (D, code reading, memcheck/racecheck clean at n_seqs = 1): the fused snapshot stores of sequence i can land on the row sequence j still has to read (different blocks), and no device-wide read barrier exists. Dropping the fusion for n_seqs > 1 would add snapshot copies, a net loss, so those batches use the old gather+fusion path. For one sequence every (head) block reads and writes only its own head slice, and each warp reads its column before any write.
- `rs_zero` handling is repeated in the indexed branch (`ggml_scale_inplace` of the cleared row) so a cleared state is zero before the read, as in `build_rs`.
- Graph node order of the fallback path changes on master: `build_rs` now runs inside `build_recurrent_attn` (after the conv/q/k/v nodes) instead of before `ssm_conv`. With package 63's prerequisite the order was already like that.

## Why upstream might want it
- MTP/speculative decoding on the hybrid Qwen3.5/3.6 family pays the gather every verify round for every GDN layer (107-128 gathers per round skipped on the Swift model, M). Gain measured: -0.31 / -0.12 / -0.29 ms/round at 8k / 32k / 90k (M; 0.3-0.8 % of a 40-46 ms round, the 32k point is inside the boot spread). Small; a feature PR needs an issue first (CONTRIBUTING), and the owner should decide whether a new public op for < 1 % is worth it.
- Related prior art on master (all CUDA, same machinery): #23940 (GDN writes snapshots into the cache), #24086 (strided snapshot copy), #29856 (gather the recurrent states once so the reserve covers every split), #29958 (unexpected graph reallocation). The OpenVINO backend squash (#29852) mentions "direct gdn outputs" for single sequences, which is the same idea at the backend level.

## Open questions for the owner
1. Idiomatic alternative on master (read before filing): Mamba already reads recurrent states in place. `mamba-base.cpp:119` passes a getter to `build_rs(inp, ssm_states_all, n_embd_s, n_seqs, get_ssm_rows)`; the lambda calls `ggml_ssm_scan(..., ids, K)` with the state-row ids as an op input (`ggml.h:2529`), and `build_rs` (#29856: "a custom getter reads the states of the ubatch straight from the cache", `s_copy_tail` for the extra rows) keeps the extras and the graph shape independent of the path. The same shape for GDN is: add `ids` to `ggml_gated_delta_net` (signature change, every backend must honour or reject it) and pass a getter that builds the GDN node. That form (a) keeps `n_rs > n_seqs` batches on the indexed path, (b) keeps the reserve and runtime graphs the same shape, (c) follows the maintainers' own precedent. This package instead uses a separate opcode plus a hand-written branch with `n_rs == n_seqs`; it is the smaller, backend-safe change but not the shape the codebase already uses. Unverified (G): it would have to be re-measured, and the extras-relocation ordering that `build_rs` relies on for Mamba would have to hold for the fused GDN snapshot stores.
2. Reserve vs runtime graph shape (G, not run) for the form in this package: the worst-case reserve has `n_rs == n_seqs`, so it takes the indexed path and no `GET_ROWS` output is sized; a later ubatch with extra rows (`n_rs > n_seqs`, shared cells after `seq_cp`) builds the gather path and may need more compute buffer than reserved. #29856 and #29958 are about this class (abort under `GGML_SCHED_NO_REALLOC`, which CI builds use). Needs a server run with `-DGGML_SCHED_NO_REALLOC=ON`, `n_rs_seq > 0`, np >= 2 and a `seq_cp`; if it reallocates, reserve with the gather path as the worst case, or take the getter form of 1.
3. Is a new opcode acceptable, or do maintainers prefer an `ids` input on `GGML_OP_GATED_DELTA_NET` itself (as `SSM_SCAN` has; smaller enum, but the Metal/Vulkan/SYCL/WebGPU/OpenVINO/HTP/ET kernels must each handle or reject it)?
4. CUDA np >= 2: acceptable that it falls back, or should the kernel get the barrier-free multi-sequence form (e.g. read all rows into registers/shared before any fused write)? Measured gain does not justify it for the serving case (np1).
5. Whether to regenerate `docs/ops` in the same PR.
6. Does package 63 go first? 0004 and 63's prerequisite touch the same lines; `on-63/` has the stacked form. Which of the two is filed first decides which form is the PR.
7. Drop the 32k data point from the PR text or keep it with its spread.

## Upstream design intent (2026-10-06, offline)
- Verdict: FEATURE (perf), additive. Precedent exists for reading state rows in place: `ggml_ssm_scan` takes `ids` and Mamba uses it through the `build_rs` getter hook (#29856). No design reason found against doing it for GDN; the open point is the shape (open question 1), not the idea.
- Offline limits: PR discussions unreadable; `git log -S` on the blob:none clone is incomplete.
- Filing implication: issue first, then PR after a GPU rerun on a master build (EVIDENCE.md, "Not measured"). CPU `test-backend-ops` cases are the fail-before/pass-after for the op; the model-level check is `test-recurrent-state-rollback`.
