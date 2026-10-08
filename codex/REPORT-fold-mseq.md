# Multi-sequence MTP catch-up fold

## Verdict

Implemented and committed. Not cleared for GPU measurement or adoption: the Mac model gates skipped with rc 75 after the required RAM wait. No model was loaded. No GPU operation, Linux compile/test/model run, or queue submission/cancellation was performed.

Labels: M measured, D derived, G guess.

## Commits

- Root: `629a725dd`, `llama : fold MTP catch-up for several sequences`. Default-on, no fold environment switch or mutant in this commit.
- Local only: `538468e53`, `common : add local MTP fold controls`. `LLAMA_MTP_FOLD_C=0` forces the original catch-up path. `LLAMA_MTP_FOLD_DEBUG_SWAP_SEQ` swaps first-row hidden inputs across active sequences; `LLAMA_MTP_FOLD_DEBUG_LAYOUT` forces a counted layout refusal. Draft dumps gain a sequence id.
- Validation/templates: `47fa2d3d9`, `c86c8213a`, `e6a9d5d14`. The last two correct the mixed-slot fixture and strengthen the mutant gate. Do not include these `codex/` artifacts in an upstream code series.

Base: stack7 `48b82b0d1559e936dbf612c871f2583cd4c79f3e`. The owner's pre-existing changes to `codex/PROMPT.md` and `codex/PURPOSE` are untouched and uncommitted.

## Design and exactness argument

`common/speculative.cpp:1550` allocates one stable `[n_seq][n_embd]` hidden-state buffer. `accept()` updates a sequence's row in place. `common/speculative.cpp:1784` arms the target with that buffer and its row count; the next target decode consumes the arm once.

`src/llama-context.cpp:2047` requires a small token batch in one target ubatch, one position axis, one sequence id per token, valid hidden-row ids, and `batch_idxs[i] == i`. `mtp_fold_prepare()` at `src/llama-context.cpp:2382` requires a single contiguous block per sequence and positions starting at that draft sequence's `seq_pos_max + 1`. Per-stream KV additionally requires equal row counts and consecutive sequence ids, matching `llama_kv_cache::init_batch()` and `split_equal()`. Unsupported layouts return null. They retain the separate catch-up through `fold_n_last != n_tokens` at `common/speculative.cpp:1667`; no new abort is introduced.

`src/llama-graph.cpp:536` copies each active sequence's own hidden row into a compact input. First-row indices select that sequence's hidden row; subsequent indices select the preceding target nextn row. `src/models/qwen35.cpp:287` gathers those rows from `concat(h_prev, h_nextn)` before the unchanged MTP K/V projection and store nodes. Graph reuse checks both token count and active-sequence count and refreshes indices and source bindings.

This preserves the separate catch-up's inputs, row order, positions and K/V writes by construction. Token/draft identity remains an inference until the model gates run. The debug log reports engagement once per active `n_seq`, including `n_seq=2` only when both sequences fold.

## Mac evidence

Snapshot `47fa2d3d9`, remote `/Users/builder/qwen38-fold-mseq/a1/src`; local evidence `codex/mseq/mac-out/a1/`.

| Gate | Result |
|---|---|
| CPU configure/build | PASS (M): `mac-out/a1/cmake.log`, `build.log`; Ninja linked `bw-np2-driver`. CUDA, Metal, Accelerate and BLAS off; i8mm off; ccache requested; nice 19, `-j4`. |
| np1/np2 model matrix | SKIPPED, rc 75 (M): first cell waited 600.9 seconds. Available RAM was 8.79 GiB (M), below the 10 GiB model-run rule. `matrix.log` and `matrix/m0-np1-kv0-k3-s1729-off.log`. Zero model cells ran (M). |
| Recorded GPU parser replay | PASS, rc 0 (M): `parser-replay.log`; recorded SSE, native token identity, overlap, memory CSV and kernel-journal parsing, malformed/short record controls, and missing np1/np2 fold-marker rejection. |

Parser replay ran only on the Mac, using copies of real artifacts from `orchestration/gpuq/fixtures/` and the recorded stack6/getter/chain GPU runs referenced by `codex/gpu_stack7.py`. It ran no GPU binary. The replay contains 24 steady intervals (M); its 37.19825 ms/round (D) is a historical parser fixture value, not this patch's performance.

The core fold and local controls compiled at `47fa2d3d9`. The corrected final mixed-slot fixture and matrix at `e6a9d5d14` have not been compiled/executed. The original skipped summary is `{}` because that first runner did not record `SystemExit`; the corrected runner records rc 75 explicitly.

`codex/mseq/matrix.py` prepares same-binary on/off checks for plain and Meta(CPU,CPU), np1 and np2, per-stream and unified KV, k3/k5, and seeds 1729/2718. It compares complete per-slot token/counter JSON and complete sequence-tagged draft dumps byte-for-byte. It requires joint engagement and fewer catch-up decodes. The wrong-row mutant must change generated tokens; the layout control must retain identical output/dumps and the separate catch-up count with a positive refusal counter. The mixed fixture leaves slot 1 one token to generate without drafting while slot 0 drafts k3; it requires a five-row verify and identity.

Rerun from the committed snapshot with `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 bash codex/mseq/run-mac.sh a2`. Every model wait is bounded to ten minutes; the detached configure/build/matrix job is bounded to sixty minutes. Results stay under `codex/mseq/mac-out/a2/`. The alternative four-GiB Mac threshold used by `bin/bw-mac` was not applied without an owner answer.

## Build window and GPU items

- Window: `codex/mseq/bw-fold-mseq.toml`, payload `bw-fold-mseq.sh`. Priority 18, filler, `host_sensitive = false`, CCX0+1 wrapper, and pins include `work/build-window-1007/mmq.sh`. `gpuq submit --check`: valid (M).
- Experiments: `codex/mseq/items/fold-mseq-np2-real-j5q.toml.in` and `fold-mseq-np1-depth-j5q.toml.in`. Priority 30, J5q real session prompts, k3, ABBA vs frozen/stack7. Fold off in the base, on in the patch; chain off in both. np1 requires cross-arm token and draft-counter identity. np2 uses the existing arrival-dependent identity exception and therefore requires the CPU exactness gates first.
- Dry contract renders: `codex/mseq/dry-items/*-ccx01.toml`, both valid (M). These use stack7 as a placeholder in the patch arm solely to check the manifest before `frozen/fold-mseq` exists. They are not runnable patch A/B items. Contract logs are alongside them.
- Entry: existing `codex/gpu_ab.sh` / `gpu_stack7.py`, backed by a content-addressed main-harness snapshot. Frozen binaries and libraries are pinned, the executable is `round-trace`, stop uses the exact server PID, and fault checks are scoped to boot start. One patch canary boot at 8k exercises all gates within 600 seconds; the full run is ABBA at 8k/32k/90k with overlap and steady-data gates. Positive fold engagement must still be established on the real GPU in that canary.

The window requires a passing Mac summary and hashes the seven changed fold source files against its `source_sha256` before compilation. Default evidence is the future `mac-out/a2/matrix/summary.json`; the current skipped result cannot unlock it. CUDA compile/freeze is deferred to the parent window, with buildgate `-j4`, CUDA 13.2, NCCL off, architecture 120, local CCCL v3.4.3 and disconnected dependency settings. `make_items.py` renders snapshots/items inside this worktree and calls only `gpuq submit --check`.

Before the parent queues the window: obtain passing Mac gates, pin that immutable summary in the window manifest, and resolve the owner's two tracked prompt edits so the existing clean-tree guard can pass. Queue the real rendered A/B items only after freeze; do not queue the dry renders.

## Expected gain and risks

Expected np2 saving is about 0.2 ms per eligible round (D), scaled from the spike's measured np1 fold savings of 0.23/0.19/0.22 ms (M, historical). Both slots already share the catch-up decode, so the saving does not double. The new gather/input overhead and eligible-round fraction are unmeasured (G); no GPU speed claim is made.

Open gates: all model identities, both negative controls, mixed-round proof, final fixture compilation, CUDA build, positive GPU engagement and GPU A/B. Unequal sequence lengths, reordered rows, split batches, coupled sequences, adapters and unsupported context/model pairs retain catch-up; this can reduce engagement. The experimental `llama-ext.h` arm signature now includes `n_seq`; out-of-tree callers must update. This change does not resolve unrelated stack graph-cache or Meta lifetime concerns documented in the background review.
