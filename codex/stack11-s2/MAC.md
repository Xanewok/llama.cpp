# Mac CPU gates of stack11-s2

Host: mini-builder (M4 arm64, CPU only, Metal/BLAS off, i8mm off), through `bin/bw-mac TREE codex/stack11-s2/mac-gate.sh OUT_NAME`. Every job: alarm 600 s,
waits for >= 3 GiB available (<= 10 min), model-free (< 2 GiB), one job at a time. rc and stage verdicts are in the `stages.tsv` files copied here.
Stage filter rides in OUT_NAME after `__` (bw-mac forwards no env).

| run | tree / HEAD | command | result |
|---|---|---|---|
| nok | wt/stack11-s2 `8fa8c06e1` (K7, K10 dropped; 87 np2 test, window compile stage) | `bin/bw-mac wt/stack11-s2 codex/stack11-s2/mac-gate.sh stack11-s2-nok-mac3` | rc 0, 82 stages, 0 FAIL (`mac-evidence/nok.tsv`) |
| with-k | wt/stack11-s2-k `2ae84326c` (K7, K10 present) | `bin/bw-mac wt/stack11-s2-k codex/stack11-s2/mac-gate.sh stack11-s2-k-mac3` | rc 0, 79 stages, 0 FAIL; `k10-topk-unit-run` 133 s (`mac-evidence/with-k.tsv`) |

Earlier runs (`stack11-s2-nok-mac2`, `stack11-s2-k-mac2`, and the first K run whose `k10-topk-unit-run` hit the 600 s alarm at Mac load 150-240) are
superseded: the sources moved (`test-host-mapped` fallback fix, 87 np2 test). The window preflight checks both files against the trees: zero FAIL,
>= 40 PASS, sources of the evidence commit identical to the tree HEAD outside `codex/`.

## Fail-before / pass-after per package

Fail-before = the production hunk reverted from `revert/*.patch` (test kept) or a one-line mutant; pass-after = tree as committed.
"expect" is what the stage asserts (rc and a string in the output); a stage is PASS when the observation matches.

| pkg | test | pass-after (stage, rc) | fail-before (stage, rc) |
|---|---|---|---|
| 87 | `test-llama-archs -a qwen35 -b CPU -s N` (`check_mtp_draft_reset`, `check_mtp_draft_np2`) | `p87-fixed-s1..5` rc 0 "MTP draft state reset: OK"; `LLAMA_MTP_FOLD_C=0` `p87-fixed-nofoldc-s1..3` rc 0; both print "MTP draft state reset: OK" and "MTP draft np2 carry: OK" | all hunks reverted (`87-prod.patch`): `p87-reverted-s1..5` rc 1 "FAIL" in both checks, `p87-reverted-nofoldc-s1..3` rc 1. Per hunk: fold refusal reverted (`87-fold.patch`) `p87-nofold-s1..3` rc 1, and fold off still passes (`p87-nofold-nofoldc` rc 0); speculative.cpp reset reverted (`87-spec.patch`) `p87-nospec-nofoldc-s1..3` and `p87-nospec-s1..3` rc 1. Over-reset mutant (zero the carry on every batch, fold off) `p87-overreset-nofoldc-s1..3` rc 1 (np1 test only: the np2 reference runs the same binary, so it cannot see a mutant that also zeroes the reference). np2 mutants: zero-all (a fresh sequence zeroes every sequence's carry) `p87-zeroall-s1..3` and `p87-zeroall-nofoldc-s1..3` rc 1 with "np2 carry: FAIL" and the np1 test still OK; fold never resumes (`p87-nofoldever-s1..3`) rc 1 "np2 carry: FAIL", np1 OK |
| 88 | `test-speculative` | `p88-fixed` rc 0 ("uses_prompt") | `p88-mutant`: the ngram-simple `uses_prompt()` flipped to false, rc 1 "CHECK failed: common_speculative_uses_prompt" |
| 91 | `test-alloc` `test_meta_subgraph_reuse_{two_splits,ten_splits,same_graph}` | `p91-test-alloc` rc 0, all three PASSED on the unmodified stack11-base Meta code | no production hunk exists to revert (see REPORT.md); control = a plan that forgets its cached uid (`backend_ctx->uid = 0`): `p91-uidchurn` rc 134 (GGML_ASSERT) |
| 90 | `test-alloc` `test_meta_host_mapped_weight`, `test-host-mapped` (SKIP on CPU), `test-backend-ops -b CPU -o GET_ROWS` | `p90-test-alloc` rc 0, `p90-host-mapped-cpu` rc 0 (SKIP), `p90-get-rows-cpu` rc 0 and no FAIL | iterative split-state driver without the host-source branch reverted (`90-meta-driver.patch`): `p90-nodriver` rc 139 (SIGSEGV, no PASSED line); Meta device without the host-buft acceptance (`90-meta-all.patch`): `p90-nometa` rc 134, no PASSED line |
| window (nok tree only) | `codex/stack11-s2/window/meta_rows.cpp` compiled with `-Wall -Wextra -Werror` against the tree's ggml | `meta-rows-build` rc 0, `meta-rows-skip` rc 0 ("meta_rows: SKIP", no GPU) | n/a (the real run is CUDA window stage 54) |
| whole | `test-llama-archs` (all archs), `test-alloc`, `test-speculative`, `test-backend-ops -b CPU -o GET_ROWS` | `whole-llama-archs`, `whole-test-alloc`, `whole-speculative`, `whole-get-rows` rc 0 | n/a |
| drops (nok tree) | CPU ops of the touched ranges | `drop-top-k`, `drop-rms-norm`, `drop-scale` rc 0; `drop-files-absent` PASS | n/a |
| drops (K tree) | K7 byte identity, K10 batched top-k unit | `k7-pair-identity` rc 0, `k10-topk-unit` build rc 0, `k10-topk-unit-run` rc 0 on rerun | n/a |

np2 test (`check_mtp_draft_np2`, `tests/test-llama-archs.cpp`): two builds of the model; the reference runs seq 0 then seq 1 alone, the mixed run warms seq 1
with an unrelated request (stale carry, then its memory is emptied), then batch A = seq 0 prompt, batch B = seq 0 continuation + seq 1 fresh prompt
(equal token counts, so only the fresh sequence makes the fold refuse), batch C = both continuations. Checks: seq 0 draft logits after B and C and seq 1 after C
equal the solo reference (nmse < 1e-9; the fixed tree gives exactly 0), B folded 0 tokens, C folded all its tokens (skipped with `LLAMA_MTP_FOLD_C=0`).
Observed np2 line on the Mac (seed 1): fixed `0/0/0 folded 0/4, 4/4`; 87-prod and 87-spec reverted seq 1 nmse 5.1e-06; 87-fold reverted `folded 4/4` on B and
seq 1 nmse 5.1e-06; zero-all seq 0 nmse 1.0e-06 / 8.5e-07.

Not on the Mac: CUDA kernels, `test-host-mapped` with a device (window stage 51), Meta over real devices.

Two test-harness facts found on the way (both in the 87 commit's test file, `tests/test-llama-archs.cpp`):
- the async logger dropped the last lines (the MTP check result and the summary) at exit in about half the runs under load (rc was right, output
  truncated; 19 of 40 runs on the Mac); `common_log_flush(common_log_main())` before the return fixes it (all later runs print every line);
- `LLAMA_MTP_FOLD_C=0` selects the catch-up decode, the default the folded decode: each production hunk of 87 is reached by one of the two, so the
  stages run both.
