# cand-proof

Decision (a), high confidence: the Qwen3.5-4B fixture cannot reach TP backend candidates because it has a tied LM head. This is a fixture limitation, not a defect in 062667905 and not a requirement for CUDA split buffers. CPU token/counter/probability identity remains mandatory; real Swift TP engagement belongs in the GPU canary.

## Cause and evidence

Observed GGUF tensor metadata, without reading tensor data: `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf` contains `token_embd.weight`, but neither `output.weight` nor `blk.32.nextn.shared_head_head.weight`. `src/models/qwen35.cpp:44` loads the optional output, `:51` substitutes `token_embd.weight`, and `:656` uses that same output for MTP when there is no separate head. `src/llama-model.cpp:599` splits an actual `output.weight` by vocabulary; `:619` mirrors unrecognized names including `token_embd.weight`. `llama_model::logits_vocab_split()` rejects that axis at `src/llama-model.cpp:2106`.

`llama_context::set_sampler()` tests this before the leading-top-k gate (`src/llama-context.cpp:1363`, `:1365`). Its `:1375` warning in every Meta fixture arm is exactly:

    set_sampler: seq_id = 0: backend sampling with SPLIT_MODE_TENSOR needs logits split by vocab over output.weight (no replicated or tied LM head); using CPU

Recorded Mac artifact: `/home/xanewok/repos/qwen38-eval-worktrees/work/build-window-1007/cand-neginf-mac-a5/e2e/1-1-1-cand.log:1996` is the draft refusal; `:2032` is the target refusal. Draft installation is `common/speculative.cpp:1470`; the target is `tools/server/server-context.cpp:1969`. The host arm installs no target backend chain. Thus Meta counts are cand/refused = 2 and host = 1 with MTP, cand/refused = 1 and host = 0 without MTP (M, recorded Mac and x86 artifacts); all are the tied-head reason. There are no leading-top-k refusals because execution never reaches that later condition.

`common/sampling.cpp:589` still selects the candidate chain for pure, unique `-inf` biases; `:621` extends top-k by the ban count; `:798` returns it. The refused arm adds a finite zero bias, which disables candidates and returns the bias-led ordinary chain. That chain would reach the leading-top-k refusal with a vocab-split head.

CUDA is not required for that refusal or candidate execution. The synthetic test builds an untied model on two CPU devices (`tests/test-llama-archs.cpp:1257`, `:1298`, `:1323`). Meta's split TOP_K support is backend-independent (`ggml/src/ggml-backend-meta.cpp:920`), followed by exact generic AllReduce on CPU. Both synthetic seeds pass on the Mac (M).

## Changes

All source changes made in this task:

- `common/sampling.cpp:179`: one debug message only after backend candidates pass the finite/unique boundary check and are retained. No sampling, RNG, tensor, or arithmetic changes.
- `codex/bw-cand-compare.py`: remove invalid CPU path requirements; retain exact identity and diagnostic fallback counts; require the complete run matrix.
- `codex/bw-cand-driver.cpp`: comment correction only; no compile fix or behavior change.
- `codex/bw-cand-neginf.sh`: CPU verdict uses identity, then replay controls; obsolete rc 5 removed; Mac RAM gate enforces the brief's 10 GiB requirement.
- `codex/check-cand-proof.py`: test signatures against a recorded GPU stack6 excerpt; reject stack6 as cn, an empty log, zero bans, and a fallback. Replay actual Mac JSON/logs and require rc 2 for token, counter, and probability mutations, rc 8 for a missing run.
- `codex/fixtures/stack6-tp-sampling.log`: real GPU log excerpt, with original path, source line numbers, and full-file SHA256.
- `codex/cand-neginf-depth-j5q.toml.in`: positive consumed-candidate signature; explicit `ignore_eos`; debug verbosity only for canaries; canary budget capped at 600 seconds. Interleaved frozen ABBA full plan and cross-arm identity remain in place.
- `codex/bw-mac.sh`: nice 19, four compile jobs, `GGML_CCACHE=ON`; result copies default inside this worktree instead of the shared work directory. The Mac has no ccache executable; ON does not make it available. No changes to shared `bin/`, other worktrees, or the evaluation repository.

No incidental C++ compile fixes were needed. Existing 78b1eedf2 is already an ancestor; no reconciliation cherry-pick was needed.

## GPU canary proof

`cn` must match both the real bias-led host-chain launch line and:

    set_logits: using backend candidates: n_cand = [1-9]\d*, n_ban = [1-9]\d*, top_k = 20\b

It must also match `(?s)\A(?!.*set_sampler: [^\n]*using CPU)`. The positive message proves candidates were consumed, not merely installed; nonempty bans distinguish the target from the unbanned draft chain. `common/log.h:24` makes LOG_DBG level 5, so `[canary].env` enables level 5; measured boots retain level 4 and do not print the new line.

`s6` must match the bias-led host-chain launch line and:

    set_sampler: seq_id = \d+: backend sampling with SPLIT_MODE_TENSOR needs a sampler chain whose first active sampler is top-k with 1 <= k <= 256; using CPU

Recorded frozen stack6 is the negative control for cn. Its GPU artifact is `results/2026-10-07-verify-path-j5q/canary-a1/0-fallback/server.log` in the evaluation repository, lines 486/487 and 5323/5324. The unchanged harness checks arm canary_log via `harness/depth.py:404`. Each arm has a separate one-boot canary; the item retains both arm canaries to prove the actual reference refusal and patched engagement, under a combined 600-second budget. No GPU run or queue action was performed here. Positive candidate-message syntax is tested with synthetic text; real Swift engagement and GPU exactness remain unverified until the parent queues the item.

## Mac verification

Commits: `935c302b9` (source and CPU verdict), `6244f1bc2` (canary verbosity and stronger probability control).

`CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 codex/bw-mac.sh "$PWD" codex/bw-cand-neginf.sh cand-neginf` started stage 1 from 935c302b9. Output reservation: `build-bw-mac-results/cand-neginf-mac-a1`; remote: `/Users/builder/qwen38-bw/cand-neginf/935c302b94105729abe667f97d09cb5008806706/run.4Hg1Bk`.

Clang CPU build, test-sampling, both `--tp-sampling` seeds, and the compiled no-extension mutant passed their gates (M). Stage wall times including gates: CPU build 47 s, unit tests 20 s, mutant 1 s (M, `stage-wall-seconds.tsv`). The mutant returned rc 1 with `banned: read the full row although the top k is unique` (M). Fresh fixture runs are currently waiting for the 10 GiB RAM gate: the Mac reports 61% free on 16 GiB (M), approximately 9.76 GiB (D). Stage 1 is not yet complete; the detached remote runner remains waiting at the gate.

The final 6244f1bc2 Python controls passed on the Mac against the actual previous Mac-a5 E2E artifacts (M): CPU identity, token/counter/probability negative controls, missing-run control, and GPU stack6 signature negative control. Local checks were limited to `bash -n`, `python3 -m py_compile`, and `git diff --check`; no local compile or model test.

Open: complete the fresh Mac fixture matrix after RAM reaches 10 GiB or the owner explicitly authorizes the existing 4 GiB Mac exception. GCC 13 and CUDA builds were not run under this task's Mac-only compile rule. The only runtime C++ addition is a portable printf-style debug message. No GPU speed claim is made.
