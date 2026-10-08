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

## a3

### Cause and fixes

Observed regression in root `629a725dd`, not a judge/counter or A/B-switch defect: `src/llama-context.cpp:2048` added `ubatch.n_pos == 1`. Qwen3.5 uses M-RoPE, so `src/llama-hparams.cpp:285` returns `GGML_MROPE_SECTIONS` (4, D) even for text tokens. `src/llama-batch.cpp:829` broadcasts each text token's scalar position across those sections. The old stack7 fold had no one-axis guard. Consequently, the new guard rejected every eligible Qwen3.5 token batch before `mtp_fold_prepare()`.

Root fix `e61e6bb31`, `llama : allow M-RoPE token batches in MTP fold`, removes only that guard. Token-only input, one target ubatch, original row order, valid sequence ids, contiguous sequence blocks and draft-cache position checks remain. The folded graph consumes the same broadcast `inp_pos` as the target (`src/models/qwen35.cpp:324`); the separate catch-up's token batch receives the same broadcast from the allocator. No sampling operation, default-on policy, root environment switch or fallback changes.

The a2 trace is the negative control for this regression: fold off/on each have 35 catch-up decodes (M), zero fold events (M), and identical tokens/dumps; `matrix.py:61` fails with `no catch-up saved`. At `e61e6bb31`, a3's same cell has 35/1 catch-up decodes off/on (M), 34 folds (M), and identical tokens/dumps. Its first np2 cell has 50/5 catch-up decodes (M), 45 folds (M), and joint `engaged n_seq=2` plus identity.

a3 built successfully and passed all four plain-CPU np1 pairs and the first np2 pair (M). It stopped in `m0-np2-kv0-k3-s2718-off`: the driver requires 80 output tokens (D), but slot 1 stopped on EOS after 55 tokens (M), then threw `short completion`. Remaining cells and controls did not run. Evidence: `mac-out/a3/matrix.log`, `matrix/summary.json`, and `matrix/m0-np2-kv0-k3-s2718-off.log:9179`. This failure occurs with folding disabled.

Fixture commits `fe34c623a` and `186e41a52` apply the fixed-length workload's EOS constraint. a4 showed that setting only `sampling.ignore_eos` is insufficient for direct `server_task` construction: it bypasses the schema's EOG-bias handler (`tools/server/server-schema.cpp:475`). The driver now copies the model-derived `server.get_meta().logit_bias_eog` into each task's `sampling.logit_bias` (`codex/bw-np2-driver.cpp:86`, `:125`), as that handler does. No hardcoded token ids. a4's original failure evidence is preserved. Full rerun: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 nice -n 19 bash codex/mseq/run-mac.sh a5`, snapshot `186e41a52`.

### Full Mac result (a5)

PASS, rc 0 (M). All 24 identity pairs and all controls passed across 52 model processes (M), sequentially on `mini-builder`. Evidence: `codex/mseq/mac-out/a5/{COMMIT,cmake.log,build.log,matrix.log,parser-replay.log,matrix/summary.json}` and each cell's JSON, draft dump, trace and log. Tested fold source digest: `059dd4c58ba348e0e954760dd8d6878f04cadbdec5f34e1acac2368de3fc7a06`.

Each cell runs cold and repeated-prefix requests in one loaded model, with context 4096 (D), four threads (D), Qwen3.5-4B Q4, q8_0 target KV, probabilistic MTP and rejection verify, chain disabled. `m0` is plain CPU; `m1` is Meta(CPU,CPU). `kv0` is per-stream; `kv1` is unified. Counts below are measured per model process, including prefill and both request phases; they are not steady-round timings. Every PASS includes identical per-slot output tokens, draft/acceptance counters and complete draft dumps, positive fold counts, fewer separate catch-ups, and the required joint-sequence engagement marker.

| Cell | Catch-up off (M) | Catch-up on (M) | Folds on (M) | Result |
|---|---:|---:|---:|---|
| m0-np1-kv0-k3-s1729 | 35 | 1 | 34 | PASS |
| m0-np1-kv0-k3-s2718 | 38 | 1 | 37 | PASS |
| m0-np1-kv0-k5-s1729 | 28 | 1 | 27 | PASS |
| m0-np1-kv0-k5-s2718 | 34 | 1 | 33 | PASS |
| m0-np2-kv0-k3-s1729 | 50 | 5 | 45 | PASS |
| m0-np2-kv0-k3-s2718 | 55 | 5 | 50 | PASS |
| m0-np2-kv0-k5-s1729 | 43 | 8 | 35 | PASS |
| m0-np2-kv0-k5-s2718 | 50 | 6 | 44 | PASS |
| m0-np2-kv1-k3-s1729 | 54 | 9 | 45 | PASS |
| m0-np2-kv1-k3-s2718 | 60 | 9 | 51 | PASS |
| m0-np2-kv1-k5-s1729 | 54 | 14 | 40 | PASS |
| m0-np2-kv1-k5-s2718 | 55 | 12 | 43 | PASS |
| m1-np1-kv0-k3-s1729 | 37 | 1 | 36 | PASS |
| m1-np1-kv0-k3-s2718 | 45 | 1 | 44 | PASS |
| m1-np1-kv0-k5-s1729 | 29 | 1 | 28 | PASS |
| m1-np1-kv0-k5-s2718 | 34 | 1 | 33 | PASS |
| m1-np2-kv0-k3-s1729 | 50 | 8 | 42 | PASS |
| m1-np2-kv0-k3-s2718 | 58 | 6 | 52 | PASS |
| m1-np2-kv0-k5-s1729 | 45 | 8 | 37 | PASS |
| m1-np2-kv0-k5-s2718 | 47 | 9 | 38 | PASS |
| m1-np2-kv1-k3-s1729 | 54 | 12 | 42 | PASS |
| m1-np2-kv1-k3-s2718 | 60 | 9 | 51 | PASS |
| m1-np2-kv1-k5-s1729 | 56 | 14 | 42 | PASS |
| m1-np2-kv1-k5-s2718 | 52 | 14 | 38 | PASS |

Controls (M): `mutant-swap` engaged and changed 2 output token ids relative to `m0-np2-kv0-k3-s1729-off`; `forced-layout` preserved output/dumps and all 50 separate catch-ups, with 45 refusal events and zero folds; `mixed-off/on` preserved output/dumps, drafted only slot 0, contained 2 five-row verify phases, and reduced catch-up count from 36 to 4 with 32 folds. Assertions: `codex/mseq/matrix.py:80` through the mixed-round checks.

Recorded GPU parser replay passed (M) at the final tested snapshot, using the Mac copies of real `orchestration/gpuq/fixtures/` and recorded stack6/getter/chain artifacts. It checks SSE/native token identity, overlap, memory CSV and kernel-journal parsing, malformed/short records and absent np1/np2 fold markers. No GPU binary ran. Summed model-job elapsed time was 1328.22 seconds (D); individual timings in `summary.json` include loading, warmup and verbose logging and are not a performance A/B.

This clears the model identities, joint engagement, negative controls and mixed-round gates previously left open. CUDA compilation, positive real-GPU canary engagement and GPU performance remain unmeasured. No Linux compile, test, model run or GPU operation was performed. The owner's `codex/PROMPT.md` and `codex/PURPOSE` edits remain untouched; they still need resolution before the window's clean-tree guard can pass.

### Window and item checks

The window now defaults to `mac-out/a5/matrix/summary.json` and pins that passing evidence. `CUDA_VISIBLE_DEVICES='' nice -n 19 gpuq submit --check codex/mseq/bw-fold-mseq.toml` passed (M); log: `mac-out/a5/window-contract.log`.

`CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/mseq/make_items.py 186e41a52 codex/mseq/mac-out/a5/matrix/summary.json codex/mseq/dry-items/a5 --dry` rendered both np1/np2 items and their `gpuq submit --check` calls passed (M). Logs: `dry-items/a5/*-contract.log`. These remain contract-only renders with stack7 in the patch arm, because no fold-mseq CUDA freeze exists. The parent window must freeze this branch and render the real items before queuing the canary. Nothing was submitted or executed on the GPUs.
