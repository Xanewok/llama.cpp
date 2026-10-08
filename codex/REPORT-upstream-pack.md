# Upstream packages (2026-10-08)

Labels: M = measured/observed, D = derived, G = guess. Revisions, counts of configured threads/jobs, package numbers and paths are identifiers/settings. No GPU execution or Linux compilation/model run occurred. Only SSH/rsync to mini-builder used the network.

## Conclusion

Packaging complete; Mac model validation incomplete for 21 and 72. The serialized continuation remains queued behind foreign CPU model work.

Proposed packages are in `codex/upstream-out/patches/upstream/llama.cpp/`; proposed main-repository edits are `codex/upstream-out/docs.diff`. The diff applies to the clean `main` documents read from `~/repos/qwen38-eval` (M). `docs-base-sha256.txt` records the document inputs. Nothing was written to the evaluation repository, other Linux worktrees, frozen builds, canonical scripts, or a remote service.

`69-ssm-out-proj-2d` is the cleanest independent feature. `68-backend-candidates-neginf` is distinct from package 01 and extends 16/04/12; it is not package 22's grammar/backend-draw mechanism. Package 21 replaces its old prototype, includes the lifetime np2 fallback, and has a small standalone gain. Host trims are two independent packages. The NH1 planner defect is isolated from the folding feature. Package 67 receives evidence only.

M: every root series, including its declared prerequisites, applied with `git am` onto `7049ff0cbeb1f5ead231de4522af6b75d8d773c0` and built with Clang on mini-builder. M: indexed-removal and Meta-planner regression tests and their production mutants passed their gates; the chain output-capacity contract passed. Candidate sampling and out-projection np2 identity/control pass. The fold and chain real-model tests remain pending behind other Mac model runs. **No package is ready to file.** Dupcheck pending (no network); features also require owner-filed issues, source review and owner-written public text.

## Packages

| Package | Verdict / relation | Apply/build/test on Mac (M) | Evidence and readiness |
|---|---|---|---|
| 68-backend-candidates-neginf | FEATURE (performance). Distinct from 01's backend bias-index mapping; extends candidate-prefix package 16, after 04/12. Related to 22 and the broader finite-bias experiment, not the same algorithm. | Fresh am and Clang build pass; test-sampling pass. `--tp-sampling -s 42` and `-s 1729` pass; no-extension mutant fails with rc 1; restored build and clean-source check pass. | 2026-10-07 cand-neginf-depth-j5q/full-a1: tokens and counters identical (M), -0.649/-0.260/-0.077 ms/round (D), two boots per arm; 90k overlaps zero. Not fileable: dupcheck, issue and prerequisite review. |
| 21-mtp-draft-chain | FEATURE (performance). Replaces old 21/29. Current chain requires 04/12/16, reduced-head work from 18 and compact embeddings; it does not need old package 20. da787bb40 is folded into the common driver. | Fresh am and Clang build pass. Private chain-capacity test passes seven configurations. np1/np2 token, counter, proposal, q and decode-shape comparisons plus reverse-uniform, shifted-embedding and old-np2 controls are waiting. | Stack7-depth-j5q/full-a5: tokens/acceptance identical (M), combined s7 -0.58/-0.62/-0.61 ms/round (D). Chain alone earlier -0.07/-0.21/-0.21 (D); do not assign the combined gain to it. Not fileable: fresh model controls, automatic-default GPU validation, prerequisites and issue. |
| 69-ssm-out-proj-2d | FEATURE (performance). Standalone graph reshape, no backend/model-instance special case or new API. | Fresh am and Clang build pass. np2 complete logits/tokens match byte-for-byte; old-rank baseline returns rc 42; patched rank check passes. | In-model Q6_K projection 67 -> 38 us at np2 (M, recorded 2026-10-07). Integrated stack7 np2 -1.130/-1.012/-1.277 ms/round (D) is not this patch's standalone gain. Not fileable: dupcheck and issue. |
| 70-kv-seq-rm-indexed | FEATURE (performance). Existing position-index optimization; independent of 71, same removal concern as the earlier host-bundle item. | Fresh am/build pass. `test-llama-archs --seq-rm` passes. Inclusive bound and first-position-ordered-head mutants both return 255; restored oracle passes. | Historical CPU at capacity 102400: np1 100.542 -> 1.505 us, np2 204.761 -> 3.178 us (M; five medians, 2000 repetitions, includes reinsertion). No fresh CPU/GPU speed claim. Not fileable: dupcheck, issue and owner review. |
| 71-mtp-verify-bulk-read | FEATURE (performance). Independent host trim; final v2 late-pointer path from 81f954b2a. | Fresh am/build pass. Real-model np2 patch/baseline tokens and counters match byte-for-byte at seed 1729; wrong-slot-row mutant differs; restored build/clean-source check pass. | Historical host-trims-v2 combined np2 -0.63/-0.16 (matched -0.33)/-0.41 ms/round (D), not isolated from indexed removal. Not fileable: dupcheck and issue. |
| 72-mtp-fold-catchup | FEATURE (performance). Different from old 20; append draft cache writes to target verify. Needs 51/73 for CPU Meta planning. | Fresh am/build pass, including the added matching-context gate. Plain-CPU fold engagement, proposal/target identity, hidden-row mutant and differing-context refusal tests are waiting. | Stack7-depth-j5q/full-a5: identity (M), fold alone -0.23/-0.19/-0.22 ms/round (D), two boots per arm. No GPU run of the new automatic default. Not fileable: fresh controls, context compatibility, failure rollback review, GPU validation and issue. |
| 73-meta-host-view-plan | REAL BUG (CPU Meta). Extends 51's compute-path exclusion to both planner scans; consider folding it into 51 for filing. | Fresh am/build pass. Allocator suite 38/38 passes, including six VIEW/CPY/SET_ROWS cases; removing the planner exclusions aborts with rc 134; restored suite passes. | `nh1-fold-c/codex/REPORT-nh1-meta.md` supplies original rc-134 evidence (M). New Mac test uses three direct computes with one UID because master's Meta backend does not expose graph_plan_create/compute hooks. No GPU gain claimed. Not fileable: dupcheck and owner review; package 51 prerequisite. |
| 67-gdn-indexed-rollback | Existing FEATURE. Getter evidence refresh only; opcode series unchanged. | Not rebuilt, as requested. | Updated root EVIDENCE.md and getter/EVIDENCE-wip.md with stack7 np1/np2 combined results; no getter-only attribution, no clearance of the ungated np2 form. |

All listed GPU evidence is recorded on 2026-10-07, not rerun. Settings and exact run directories are in each package EVIDENCE.md. Rebase timings cannot inherit those gains. Fresh Mac unit benchmark timings were collected under shared load and are not used as performance evidence.

Local topic package commits: `7d167748f` (68), `d22dae903` (21), `8dcdd9def` (69), `db8b3066e` (70), `4b3d2d332` (71), `32ca0f7c7` (72), `767c2fd75` (73), `f3d0e98d6` (67 evidence). They package artifacts only; the live stack source is unchanged.

## Source series and changes beyond originals

Root patches use `git format-patch`. The owner's original messages are retained as provenance; public commit text remains the owner's responsibility. Local topic commits carry no AI trailers. Prerequisites are separate and must not enter a submission. `apply.txt` records each generated replay tip and optional local base.

- 68: original `062667905`, `74e70bb95`; test include conflict resolved by retaining `llama-sampler.h` and excluding the stack's unrelated `ggml-backend-impl.h`. Bound `candidate_k` and compare the ban count in `size_t` before narrowing, avoiding overflow/truncation for extreme inputs. This is included in the original source mbox, not a new authored submission message. Omit `935c302b9` engagement logging from production.
- 21: originals `151ab739b`, `d298d3dfd`, `aa037232c`, `21ba77aa8`, `d92da203f`, `da787bb40`, with prerequisite `96e025496` and the existing 04/12/16/18 changes. Resolve sampler-helper conflict by installing inputs in `build_sampling_end`; keep the chain's mixed token/embedding input path. Remove local round-trace calls. Replace the public common parameter/CLI/env switch with private implementation state; restrict automatic output capacity to single-slot probabilistic draft-mtp, p_min zero and supported lengths. Consolidate the coherent production feature into one mbox. `local/` restores the old opt-in and `4be92a4fa` diagnostics; `validation/` preserves `374a465b8` and `4be92a4fa` provenance.
- 69: production `ce555540a` unchanged. Private np2 probe ported from `llama_batch`/`llama_decode` to `batch_ext`/`llama_process`, with four threads. Clef already has `n_seqs = 1`; no new clef np2 support is claimed.
- 70: production `36cb09139` unchanged. Move its independent scan/head oracle into existing `test-llama-archs.cpp`, add `--seq-rm` test dispatch and required private model/KV headers. Use the existing GGUF mock-model initializer, since master does not declare the stack's `llama_model_create` test helper. No new upstream test file.
- 71: reconstruct the final `3b5176efd`/`f8a00adc5` production path against master: leave the catch-up getter in place, obtain a new dense pointer after catch-up and copy each verify interval once. Exclude knobs, proof logging, cross-checks and mutants from `81f954b2a`.
- 72: runtime `7f1fe261f` under report-only `186226ceb`; `376e5c8d9` is split into 73. Omit stack-only TP constants, deferred-logit resets, sampling fields and round tracing. Replace opt-in with automatic supported-condition initialization. Add equality checks for resolved RoPE/YaRN parameters, flash-attention cache layout and op/KQV offload policy, and reject training contexts: the folded graph uses target settings. `local/` keeps the original `cd4a9d49a` dump and hidden-row control, adapted to the automatic constructor.
- 73: production `376e5c8d9` unchanged. Port its split-state helper and six cases without package 47's unrelated recursion test. Use master's existing backend declarations. Replace unsupported explicit graph-plan calls with repeated `graph_compute` calls at one graph UID; initial planning and internal reuse are checked.
- Private validation: port the deterministic np2 and in-process fold drivers without stack upload/trace hooks; cap threads at four, keep explicit 4096 contexts and suppress EOS for fixed-size requests. Add capacity/context-refusal probes, a temporary decode-shape instrumentation patch, result comparisons and production mutants. None is submission code. Mac launch/transport/replay helpers and the task-local model lock are new artifacts, not runtime changes.

## Rebased source locations

Locations are in each package's root replay tip from `apply.txt` (M), after its prerequisites:

- 68: `common/sampling.cpp:584` checks unique negative-infinity bans and the bounded candidate budget; `common/sampling.cpp:620` adds the extension.
- 21: `common/speculative.cpp:1743` enforces single-sequence chain eligibility; `common/speculative.cpp:2734` enforces single-slot output capacity. The np2 contract concerns the whole attached slot set, not only active drafting sequences.
- 69: `src/models/qwen35.cpp:470` flattens the projection input; `src/models/qwen35.cpp:474` keeps the existing projection operation. The mbox applies the same shape change to the other affected builders.
- 70: `src/llama-kv-cache.cpp:423` begins indexed removal; `tests/test-llama-archs.cpp:1066` supplies the scan/head oracle.
- 71: `common/speculative.cpp:1627` reads the dense pointer after catch-up; `common/speculative.cpp:1638` copies the correct per-sequence row interval.
- 72: `src/llama-context.cpp:2157` supplies initialization and context-compatibility gates; `common/speculative.cpp:1508` enables folding after those gates.
- 73: `ggml/src/ggml-backend-meta.cpp:2158` and `:2198` exclude Meta buffers from planner shortcuts; `tests/test-alloc.cpp:1103` begins the split-state regression helpers. The earlier exclusion at `:2024` is prerequisite 51, not a change in 73.

## Verification and outstanding runs

Mac root: `/Users/builder/qwen38-upstream-pack/20261008`. The CPU recipe is in `codex/mac_upstream_pack.sh`; builds run at nice 19 with -j4, native off, armv8.2-a+dotprod, CUDA/Metal/BLAS off and ccache ON. The Mac has no ccache executable. Model gates require at least 4 GiB free, no other known model process and a task-local exclusive lock; model contexts are 4096 and threads are at most four. No foreign process was stopped.

M: generated and Mac-applied root tree IDs match for every new/replaced series (`mac-evidence/final/tree-proof.txt`).

Compact raw log excerpts and SHA256 hashes are in `mac-evidence/final/excerpts.txt`; full `.log` files remain in the Mac run directory (repository ignore rules exclude them from commits).

Evidence snapshot: `codex/upstream-out/mac-evidence/final/{68,69,70,71,73}/`. Per-package Mac logs live in `<slug>/logs/`; candidate logs are at the Mac root `logs/`. `21-mtp-draft-chain/logs/chain-capacity-final.log` records the capacity contract. Generated source paths and original commits are listed in package NOTES.md; locations refer to the rebased tree identified by apply.txt, not the stack7 checkout.

The current serialized pipeline runs the following commands. Do not launch a duplicate while its recorded PID is alive:

```
PACK_LOCAL_APPLIED=1 bash validation/model_tests.sh "$PWD" 72-mtp-fold-catchup
bash validation/model_tests.sh "$PWD" 21-mtp-draft-chain
```

Candidate runner completed; remaining pipeline PID is 89257 (M, recorded in `model-final-v2.pid`), eligibility gate PID 89843 (M). The gate has waited over 30 minutes (M) behind foreign CPU model work. The initial eligibility fixture aborted because it omitted `load_mtp`; the flag is now corrected and rebuilt, but the corrected run has not acquired the slot. Progress is in `model-locked-v2.log`. They remain queued behind the Mac's other model work; this report does not turn a queued command into a passed test. Collect final logs and update EVIDENCE.md/apply.txt after completion. Failed controls must be diagnosed, not waived.

GCC 13 and CUDA builds were not run under the Mac-only rule; `gcc-13`/`g++-13` are not available on the configured Mac PATH (M). No new GPU queue item was authored, submitted or run; the existing stack7 GPU entry was left untouched. New GPU validation must use freshly frozen rebased binaries and the harness contract before filing. Duplicate checks remain pending (no network).
