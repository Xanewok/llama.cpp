# Host decode cost versus KV depth

## Conclusion

The host KQ-mask scan is a real depth-dependent cost. A dense-prefix certificate from the existing KV indexes removes the historical cell scan without retaining masks or changing tensor bytes. The predicted exposed saving is **0.077 ms/round at 8k and 0.846 ms/round at 90k (D, CPU medians)** for the requested target verification plus draft steps. GPU performance remains unmeasured. Upload volume is unchanged, so this patch does not resolve the input-upload part of the depth growth.

The supplied nsys evidence reports STEP 6.13 / 9.13 ms, cpu_pre 3.00 / 5.02 ms and input uploads 1.25 / 2.35 ms at 8k / 90k (M, supplied, not repeated here). The CPU mixed-mask growth is 0.799 ms/round (D). This supports the mask hypothesis but does not close attribution of the supplied 3.0 ms STEP growth (D).

M = measured in this worktree, or explicitly attributed to the supplied GPU evidence. D = derived, including benchmark configuration and byte formulas. G = an unverified estimate. Line numbers, hashes and PR identifiers are references, not measurements.

Baseline: `c23324d5f9f0fc00ad3f2e6441e39ba061f41544`, the task's starting commit. Patch: `59c721c98` (`kv-cache : fill dense sequence masks from position indexes`). The baseline static library was preserved before editing production code. This is a topic-branch change; no network, push, PR, or external submission was used.

## Host paths and bytes

Define K = padded `n_kv`, Q = query rows, C = allocated cache capacity, U = occupied cells, F = future-position cells, R = removed cells, S = streams, B = `sizeof(llama_kv_cells::seq_set_t)`, E = `sizeof(llama_kv_cell_ext)`. Byte formulas below are D tensor payloads or explicit metadata payloads. Total bytes written inside STL trees and allocator headers are implementation dependent and cannot be specified by K and Q alone.

The decode path enters memory preparation at `src/llama-context.cpp:1930`, applies it at `:1514`, and sets graph inputs at `:1571`. Batch validation at `src/llama-batch.cpp:294` uses indexed position queries; its local sequence sets cover the input batch, not the entire KV history.

Swift's target graph creates one hybrid attention input (`src/models/qwen35.cpp:149`, `src/llama-graph.cpp:1089`). It fills one mask shared by its attention layers. Its MTP graph creates a separate plain KV input (`src/models/qwen35.cpp:538`, `src/llama-graph.cpp:470`); `src/llama-model.cpp:2723` selects a plain cache for hybrid Qwen MTP. Neither path fills a mask per attention layer. Catch-up on the draft context has its own mask if performed; it is outside the round definition in this task.

| Operation and code path | Depth dependence | Explicit bytes written / decode (D) | Frequency |
|---|---|---|---|
| `llama_kv_cache_context::set_input_kq_mask` -> `llama_kv_cache::set_input_kq_mask` -> `set_input_kq_mask_impl`, `src/llama-kv-cache.cpp:2929`, `:1769`, `:1561` | Original scans K cells for the first row of each query sequence. Later rows copy that mask and update recorded recent cells. | F16 `2*K*Q`, F32 `4*K*Q`, plus overwritten recent entries, fixed `4*LLAMA_MAX_SEQ` minimum-position scratch and sequence maps. | Once per ubatch per attention cache/group; target and draft independently. |
| Mask input staging, `ggml/src/ggml-backend.cpp:1700`; FA meta mask is mirrored, `ggml/src/ggml-backend-meta.cpp:793` | Full mask payload grows with K even after this patch. | Logical `2*K*Q` for FA; mirrored GPU payload `4*K*Q` for the GPU pair. Staging may write another host copy. | Once per input upload, across the GPU pair, shared across attention layers. CPU benchmark excludes CUDA transfers. |
| `get_n_kv`, `src/llama-kv-cache.cpp:1250` | No cell scan: uses `used_max_p1()` and a set endpoint. Rounds to `max(n_pad,256)` and caps at C. | Scalar result only; graph input allocation reserves `2*K*Q` mask bytes when shape changes. | Once per ubatch apply; shape reuse checks also read K. |
| `find_slot`, `src/llama-kv-cache.cpp:898` | Normally Q empty cells from `v_heads`; worst case scans C occupied cells. Debug logging can additionally scan C. | Slot indices `4*Q` plus stream descriptors and STL bookkeeping. No K-sized write. | Once in `prepare` for the plain cache; target hybrid preparation also prepares recurrent metadata. |
| `prepare`, `src/llama-kv-cache.cpp:749` | Snapshots only proposed slots, applies them, then restores. Tree inserts/erases have logarithmic U dependence. | Snapshot arrays `Q*(4+E+B)` plus reset/shift fields, head copies and tree nodes. No full cache copy. | Once per decode, for each ubatch in it. |
| `apply_ubatch`, `src/llama-kv-cache.cpp:1104` | Q cells, indexed sequence-position updates; overwritten SWA cells can trigger a capacity scan via `seq_rm`. No SWA for this target. | Positions `4*Q`; ext `E*Q` when needed; changed sequence words, fixed `4*LLAMA_MAX_SEQ` scratch, tree links. | Temporary apply during prepare, then real apply. Not per layer. |
| `seq_pos_min/max`, `src/llama-kv-cache.cpp:659`, `:672`; `src/llama-kv-cells.h:371` | Set endpoints, not an O(K) search. Hybrid combines these with recurrent positions (`src/llama-memory-hybrid.cpp:169`). | Scalar results only. | Batch position inference, speculative bookkeeping and rollback checks; number depends on Q/sequence count. |
| `seq_rm`, `src/llama-kv-cache.cpp:382`; callers `common/speculative.cpp:1605`, `tools/server/server-context.cpp:3194` | Scans **C**, not K; erases R indexed cells. At fixed context capacity its scan does not explain growth with occupied depth. | `R*(4+E+4)` cleared position/ext/shift bytes when cells become empty, one bitset word per sequence removal, plus tree erases and a head update; removal of all sequences resets the whole B-byte bitset. | Rollback/catch-up/speculative housekeeping; conditional, not every `llama_decode`. |
| Position/RoPE input, `src/llama-graph.cpp:127` | Q only. Text converts scalar positions to M-RoPE coordinates. RoPE arithmetic is in the compute graph. | Qwen text conversion `16*Q` bytes plus its temporary vector. | Once per ubatch; shared by layers. |
| K/V destination indices, `src/llama-kv-cache.cpp:1480`, `:1497` | Q only with FA/non-transposed V. Transposed V would scale with head width, not K. | `8*Q` each, `16*Q` combined. | Once per attention input/cache group. |
| Quantized-KV Hadamard inputs, `src/llama-kv-cache.cpp:1838`, `:1847` | Independent of K/Q, depends on head width. Plain F16 draft has no rotation inputs. | `4*nrot_k^2 + 4*nrot_v^2`. The direct target fixture uses `262144 + 16384` bytes. | Once per target ubatch/group, not per layer. |
| Recurrent preparation and `s_copy`, `src/llama-memory-recurrent.cpp:481`, `:505`; `src/llama-graph.cpp:1103` | Scans recurrent sequence cells/rollback metadata, not attention K. | `4*n_rs` for the input plus bounded sequence-cell metadata. | Target only, once per ubatch; recurrent tensor computation/snapshots execute in the graph. |
| Speculative checkpoint, `tools/server/server-context.cpp:3181`; `src/llama-memory-hybrid.cpp:190` | `PARTIAL_ONLY` skips attention KV completely. Recurrent payload depends on layer/state dimensions and active sequences, not K. | Per active sequence: sum over recurrent layers of `row_size(type_r,n_embd_r)+row_size(type_s,n_embd_s)`, plus headers. | Conditional per speculative round, not per decode. |
| Full state save/restore, `src/llama-kv-cache.cpp:2078`, `:2224` | Scans C; metadata and tensor payload scale with U. This is a prompt checkpoint/state API path, not ordinary `set_inputs`. Plain MTP cache ignores `PARTIAL_ONLY`; a full draft save would scale with depth if requested. | Per cell `12+E` metadata when ext is enabled; KV payload `U*sum_layers(row_size(K,width_k)+row_size(V,width_v))`, plus headers. | Explicit save/restore, conditional prompt checkpoints. No full draft checkpoint in the measured mask round. |
| Relative position buckets, `src/llama-kv-cache.cpp:1813`; K-shift input, `:1526` | Buckets would be O(K*Q); shifts write O(C). Neither is a normal Swift decode input: this model has no relative buckets and M-RoPE K-shift is unsupported. | Buckets `4*K*Q`; shifts `4*C*S` if supported/requested. | Inactive here; buckets once per applicable group, shifts only on memory update. |

For the requested round the logical F16 mask bytes are `2*K*(6+5)` (D): 180224 bytes at K=8192 and 1982464 bytes at K=90112. Mirroring to the GPU pair gives 360448 and 3964928 bytes (D). The layer count does not multiply these bytes. Mask shape padding changes in 256-cell increments (D configuration), but reading the padded extent is constant time.

## CPU measurements

Host: AMD Ryzen 9 3900X (M from `/proc/cpuinfo`). Release GCC build, `nice -n 19`, GPU visibility empty, CPU decoder threads capped at 8 (D configuration). `codex/host-depth.cpp` includes the existing architecture fixture helpers; no file or test was added to `tests/` or the default CTest list.

The direct objects have 16 attention layers for the target, one attention layer for the draft, unified KV and one sequence (D configuration). Both use capacity C=102400 at both depths, K=8192/90112 and Q=1/6 (D configuration). Target KV types are q8_0; draft types are F16. `hparams.no_alloc=true` skips unused KV tensor storage; real metadata, slot, index and mask functions execute. Head dimensions in this object fixture are configured independently of loaded Swift weights and do not affect the mask logic.

Each ordinary operation is the median of 101 samples with 10 calls per sample (D configuration), with amortized loops for scalar getters. Apply/remove and round timings use individual calls. No elapsed decoder time or GPU time is inferred from these microbenchmarks. The table is **M us per call**, with entries `Q=1 / Q=6`. `seq_rm_tail` removes the existing tail plus temporary inserted rows; its setup/restoration is outside the timer. `prepare` already includes its own temporary apply/restore, so these rows must not be summed as independent decode phases.

| Real operation | Target, 8k | Target, 90k | Draft, 8k | Draft, 90k |
|---|---:|---:|---:|---:|
| `mask_f16` | 12.992 / 15.419 | 145.075 / 158.423 | 13.015 / 14.611 | 144.237 / 161.237 |
| `mask_f32` | 11.155 / 17.542 | 121.965 / 155.096 | 10.982 / 14.097 | 119.583 / 157.060 |
| `find_slot` | 0.063 / 0.100 | 0.065 / 0.080 | 0.065 / 0.085 | 0.063 / 0.081 |
| `prepare` | 2.168 / 4.375 | 2.378 / 4.994 | 2.141 / 4.013 | 2.325 / 5.063 |
| `apply_ubatch` | 0.290 / 1.012 | 0.321 / 1.252 | 0.240 / 0.721 | 0.311 / 1.243 |
| `seq_rm_tail` | 52.057 / 48.831 | 49.964 / 49.422 | 51.777 / 48.852 | 48.762 / 50.715 |
| `n_kv` | 0.007 / 0.007 | 0.007 / 0.007 | 0.007 / 0.007 | 0.007 / 0.007 |
| `seq_pos` | 0.008 / 0.007 | 0.008 / 0.007 | 0.008 / 0.008 | 0.007 / 0.007 |
| `kv_indices` | 0.017 / 0.021 | 0.017 / 0.021 | 0.017 / 0.021 | 0.017 / 0.022 |
| `position_input` | 0.034 / 0.037 | 0.034 / 0.035 | 0.036 / 0.036 | 0.035 / 0.034 |
| `k_rotation` | 5.276 / 5.216 | 4.437 / 4.524 | -- | -- |
| `v_rotation` | 0.165 / 0.164 | 0.192 / 0.187 | -- | -- |


These are real **M F16 mask round times** for a six-row mask followed by five one-row masks, separately for each cache shape:

| Cache | Depth (D configuration) | Original, us/round (M) | Patched, us/round (M) |
|---|---:|---:|---:|
| target | 8192 | 79.899 | 3.848 |
| draft | 8192 | 79.859 | 3.797 |
| target | 90112 | 869.968 | 33.843 |
| draft | 90112 | 884.504 | 33.834 |


The mixed target/draft saving is derived by summing the independently measured medians, `target(Q=6) + 5*draft(Q=1)`:

| Depth (D configuration) | Target Q=6: original / patch (M us) | Draft Q=1: original / patch (M us) | Original mixed round (D us) | Patched mixed round (D us) | Predicted saving (D ms/round) |
|---|---:|---:|---:|---:|---:|
| 8192 | 15.419 / 2.126 | 13.015 / 0.337 | 80.494 | 3.811 | 0.076683 |
| 90112 | 158.423 / 18.371 | 144.237 / 3.045 | 879.608 | 33.596 | 0.846012 |


Target `PARTIAL_ONLY` checkpoints on the tiny model contain 10572 bytes at each depth (M). Size-query / CPU-copy costs are 0.111 / 0.471 us at 8k and 0.110 / 0.360 us at 90k (M). These are tiny-model results, not timings for the full Swift recurrent state; they establish which path runs and that its payload does not grow with attention depth.

## Chosen fix and exactness

`src/llama-kv-cache.cpp:1605` checks the existing `used` and `seq_pos` indexes. `used_max_p1()==get_used()` certifies that occupied cells form a dense prefix. `seq_get_pos(seq_id).size()==get_used()` certifies that every occupied cell belongs to the query sequence. `src/llama-kv-cells.h:363` exposes a const view of the existing ordered position index; no metadata index, mutable field or cache revision was added.

For each eligible row, fill the occupied prefix with positive zero and the empty tail with negative infinity. The position index locates every cell strictly after the query position, and those entries are overwritten with negative infinity. This is exactly the old predicate: nonempty, visible to the query sequence, and causally at/before its position. Non-causal rows omit the future-cell update. F16/F32 casts match the original constants. Holes, foreign-only cells, SWA, ALiBi and 2D positions use the original implementation. Shared cells may qualify when every occupied cell is visible to the query sequence.

This recomputes masks from live indexes on every call. A position or sequence change cannot leave retained mask state stale. It also avoids graph-allocation ownership/invalidation risks identified in the supplied Patch A review. Device computation and sampling/RNG code are untouched; exact mask inputs leave the output distribution unchanged.

| Candidate | Decision |
|---|---|
| Retain/incrementally update a prior mask | Rejected for this patch. Requires tracking query positions, sequence membership, removal/restore, shared cache mutations, padding and ownership of graph input storage. Unchanged K alone is insufficient. |
| Device-built mask | Open follow-up. Could remove the increasing host-to-device mask payload, but requires a metadata representation/upload and backend support. CPU evidence does not establish its GPU cost. |
| Skip masked blocks | Empty-tail bulk fill alone saves little because padding is bounded; it does not remove inspection of the occupied history. |
| Existing-index certificate and bulk fill | Implemented. Removes history inspection with no retained mask, no knob, no additional device bytes, and a small production diff. |

## Tests and reproducibility

`codex/cpu-results.json` records byte comparisons: 43646976 mask bytes and 6656 logit bytes match exactly (M). Masks cover both depths, both cache shapes, both query-row counts, F16/F32, causal/non-causal rows, changed sequence ownership, a future position mutation, an interior hole and empty padding. The current cell index updates execute during these tests; the changed-sequence and changed-position masks must differ from the stored previous mask.

The tiny random-weight Qwen35 model has a hybrid target and an MTP head (D configuration). At both depths it executes the six-row verification-shaped decode and one-row decodes; subsequent sequence/position edits also reach the old/fallback path. Historical KV tensor values are initialized to zero and tail rows are computed normally. This is a controlled equivalence fixture, not a representative language-quality or full-weight performance test. All compared logits are finite and nonconstant (M range -0.13218790 to 0.12944171).

Negative controls:

- Suppress the post-sequence-edit mask update with `HOST_DEPTH_STALE_CONTROL=1`: exit 1 and `negative control: sequence mutation retained stale mask` (M).
- Remove `positions.size()==n_used` from the **production** fast-path guard: the same fixture fails with that error and exit 1 (M). `codex/test_guard.sh` reproduces the mutation, restores the source via an EXIT trap, and rebuilds the correct library. The source was restored and `git diff --check` passed.

Build the current fixture with `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_cpu.sh build-cpu/patch-masks.bin build-cpu/patch-logits.bin`. For a fresh baseline, archive `codex/BASE` into `build-ref/source`, configure a static CUDA-off build there, and build its `llama` target with `nice -n 19` and `-j8`. `LLAMA_LIB=$PWD/build-ref/cpu/src/libllama.a CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/build_fixture.sh` links the same fixture to that library. Preserve it as `build-ref/host-depth-base` and run it with `build-ref/masks.bin build-ref/logits.bin`, recording stdout in `codex/base-cpu.csv`. Relink/run the current fixture and record `codex/patch-cpu.csv`, run `codex/test_guard.sh`, then `CUDA_VISIBLE_DEVICES="" nice -n 19 python3 codex/check_cpu.py`. CPU outputs and medians are checked by the latter script; raw generated binaries stay in ignored build directories.

The baseline library used here was captured before the production edit as `build-ref/libllama-base.a`. The CPU configuration was `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release`; the build target was `llama-common`, `-j8`. Scripts now set `TMPDIR` to `build-tmp` and enforce niceness/thread bounds.

## Parent-only GPU A/B

`codex/gpu_ab.sh` never runs GPUs in its default plan mode. It delegates to `codex/gpu_ab.py`; execution additionally requires `--execute-parent`, `GPUQ_ACTIVE=1` and nonempty `CUDA_VISIBLE_DEVICES`. I did not run that mode.

The parent must freeze **c23324d5f versus 59c721c98**, with identical compiler/backend settings. Each frozen root contains `SOURCE`, `SHA256SUMS`, `bin/SHA256SUMS`, and a server renamed `bin/host-depth-server`; copy required libraries into `bin` and hash every file. The outer manifest hashes `SOURCE` and `bin/SHA256SUMS`. No process named `llama-server` is launched. Use the task's CUDA configuration with NCCL off and architecture 120; the parent may perform GPU-capable configure/build validation.

Supply `--up ROOT --patch ROOT --model GGUF --model-sha SHA --out FRESH_DIRECTORY`. Each boot uses tensor split, q8_0 target KV, F16 draft KV, probabilistic MTP, k=5 and one sequence (D configuration). Boots interleave `up,patch,patch,up` (D plan). Each boot requests native completions from pre-tokenized prompts at both depths, forces length with `ignore_eos`, requests raw token IDs and per-token timings, and checks exact token/content/acceptance identity across arms. This is a controlled repeated-token prompt, not the original serving workload.

Steady timing uses adjacent full-k cumulative SSE counter intervals after the initial rounds and before the shrinking tail. Null/partial/empty rows are skipped. No number of rounds is guessed from output-token count. Output records are saved before gates are evaluated; HTTP error bodies are preserved. The harness checks SHA pins, selected-GPU memory, RAM availability, reusable port binding, boot identity, word-bounded kernel hazards since the start cutoff, exact child PIDs, and the total wall budget. Budget exhaustion is recorded as `budget`; a kernel hazard as `hazard`. `summary.json` includes raw timings and `patch_minus_up_ms_D` per depth. All files participating in identity are listed in `pins.json`; use a fresh output directory after changes.

Canary: add `--canary`, select the patch boot only, run both depths and one shallow repeat to exercise the identity gate. It retains the same steady-size/identity gates as the full run. Budget is capped at 600 seconds (D); canary/full stage estimates are 400/1920 seconds (G), each within the allowed budget fraction. Deep prefill duration and live output schema must be checked by the parent canary before the full run. If it cannot finish or pass the size gates, revise the workload/estimate and use a new pinned item.

`codex/test_gpu.py` passed recorded artifact tests against `orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json`, `nvidia-smi-query.csv`, and `xid-lines.txt` (M). Timing parser result: 24 interior intervals and 37.19825 ms/round (M/D from that recorded artifact, **not this patch's GPU result**). Tests also cover repeated/null rows, rejection by the larger steady-size gate, rejection of absent native token identity in the real chat fixture, and frozen-byte mutation. The available chat artifact has no native completion token IDs, so successful native token parsing remains a mandatory live-canary gate, not a claimed recorded-fixture success. `bash -n` passed for the shell scripts.

To attribute the saving specifically to STEP/cpu_pre, the parent should also capture nsys on the same frozen builds and the same depths, retain only complete steady verification-plus-draft rounds, and compare host mask construction with input uploads separately. This script's wall-round metric establishes end-to-end change; it does not by itself attribute a change to CPU gaps.

## Offline upstream check

Already-stored `origin/master` is `836d57176dc699a726c55418e4f96b8ca628e1bf`, `mtmd : fix deprecated strdup warning on Windows (#29863)`. No fetch or API call was made. Related PR identifiers below come from Git objects; current issue/PR status is unknown offline.

- `2fbde785b`, PR `#18842`: KQ-mask row reuse and recent-cell updates. Already present in the baseline; the first historical cell scan remains.
- `031ddb2e0`, PR `#23764`, and `2ed3c1abb`, PR `#25370`: F16 FA masks, already present here. Using F32 would overstate this task's upload bytes.
- `81713121e`, PR `#13808`: existing used-cell/sequence-position indexes; this patch reuses them.
- `b356fa262`, PR `#28040`: replace full historical n-gram hash construction with indexed lookup. Earlier `62acc89c2`, PR `#28011`, shortens sequence scanning in that n-gram path. Swift Qwen35 does not invoke PLE n-gram embedding history here.
- `4e2713c16`, PR `#29824`: qwen4exp/GLM5-next sparse **device graph** mask fill/repeat changes. It does not optimize this model's host `set_input_kq_mask_impl`; the stored master retains the original scan there.
- `436f6f89e`, PR `#29856`: gather recurrent states once during graph reservation, related host/graph work but not this mask scan.

Commands: `git log -1 --oneline origin/master`, `git log origin/master -- src/llama-kv-cache.cpp src/llama-graph.cpp`, and offline `git log --grep`/`git show` on the above commits. No matching dense-prefix host mask optimization was found in those stored objects; this does not establish that no unmerged PR exists.

## Risks and execution limits

The hot-path benefit requires a dense visible prefix. Fragmented/multi-sequence layouts can use the unchanged path. Large prefill batches, multimodal input, other backends and GPU kernel execution remain unmeasured. Ordinary index maintenance is relied on exactly as the existing cache already relies on it. No product sampling change or new mutable mask state was introduced.

CPU numbers do not establish the entire supplied GPU STEP increase. The mask scan and increasing upload payload are plausible contributors; transfer latency, scheduler work, synchronization and full server bookkeeping require GPU attribution. The requested mixed round excludes draft catch-up; its extra decode must be counted separately in a complete serving-round comparison.

Execution deviation: CUDA configuration unexpectedly ran CMake's CUDA native-architecture ABI probe despite explicit architecture 120 and empty `CUDA_VISIBLE_DEVICES`. It printed `No CUDA devices found`. This violated the task's prohibition on running CUDA binaries; no GPU kernel workload or server was launched. CUDA build execution was then skipped. Initial compiler temporaries also used inherited `/home/xanewok/.cache/agent-tmp`; subsequent helpers explicitly use `build-tmp` inside this worktree. The only deliberately authorized writes outside the worktree were its Git index/objects in the linked parent repository, needed for the requested topic commits. No external files were removed. GPU A/B and a complete CUDA build are outstanding.
