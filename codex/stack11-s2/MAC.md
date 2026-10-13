# Mac CPU gates of stack11-s2

Host: mini-builder (M4 arm64, CPU only, Metal/BLAS off, i8mm off), through `bin/bw-mac TREE codex/stack11-s2/mac-gate.sh OUT_NAME`. Every job: alarm 600 s,
waits for >= 3 GiB available (<= 10 min), model-free (< 2 GiB), one job at a time. rc and stage verdicts are in the `stages.tsv` files copied here.
Stage filter rides in OUT_NAME after `__` (bw-mac forwards no env).

| run | tree / HEAD | command | result |
|---|---|---|---|
| nok | wt/stack11-s2 `86888d1f2` (K7, K10 dropped) | `bin/bw-mac wt/stack11-s2 codex/stack11-s2/mac-gate.sh stack11-s2-nok-mac` | rc 0, 69 stages, 0 FAIL, 6 min 35 s (`mac-evidence/nok.tsv`) |
| with-k | wt/stack11-s2-k `44aaf3abe` (90 commit + codex; K7, K10 present) | `bin/bw-mac wt/stack11-s2-k codex/stack11-s2/mac-gate.sh stack11-s2-k-mac` | rc 1: `k10-topk-unit-run` hit the 600 s alarm (rc 142) with the Mac at load 150-240 (Halmos); stack11-base measured 106 s for it. 67 other stages PASS (`with-k-full-run.tsv`) |
| with-k rerun | same | `bin/bw-mac wt/stack11-s2-k codex/stack11-s2/mac-gate.sh stack11-s2-k-mac__drops` | rc 0: `k10-topk-unit-run` 178 s at load 120-190, `k7-pair-identity` 13 s (`with-k-rerun-drops.tsv`) |

`mac-evidence/with-k.tsv` = the full run with the `k10-topk-unit-run` row taken from the rerun (the only change). The window preflight checks both
files against the trees: zero FAIL, >= 40 PASS, sources of the evidence commit identical to the tree HEAD outside `codex/`.

## Fail-before / pass-after per package

Fail-before = the production hunk reverted from `revert/*.patch` (test kept) or a one-line mutant; pass-after = tree as committed.
"expect" is what the stage asserts (rc and a string in the output); a stage is PASS when the observation matches.

| pkg | test | pass-after (stage, rc) | fail-before (stage, rc) |
|---|---|---|---|
| 87 | `test-llama-archs -a qwen35 -b CPU -s N` (`check_mtp_draft_reset`) | `p87-fixed-s1..5` rc 0 "MTP draft state reset: OK"; `LLAMA_MTP_FOLD_C=0` `p87-fixed-nofoldc-s1..3` rc 0 | all hunks reverted (`87-prod.patch`): `p87-reverted-s1..5` rc 1 "FAIL", `p87-reverted-nofoldc-s1..3` rc 1. Per hunk: fold refusal reverted (`87-fold.patch`) `p87-nofold-s1..3` rc 1, and fold off still passes (`p87-nofold-nofoldc` rc 0); speculative.cpp reset reverted (`87-spec.patch`) `p87-nospec-nofoldc-s1..3` and `p87-nospec-s1..3` rc 1. Over-reset mutant (zero the carry on every batch, fold off) `p87-overreset-nofoldc-s1..3` rc 1 |
| 88 | `test-speculative` | `p88-fixed` rc 0 ("uses_prompt") | `p88-mutant`: the ngram-simple `uses_prompt()` flipped to false, rc 1 "CHECK failed: common_speculative_uses_prompt" |
| 91 | `test-alloc` `test_meta_subgraph_reuse_{two_splits,ten_splits,same_graph}` | `p91-test-alloc` rc 0, all three PASSED on the unmodified stack11-base Meta code | no production hunk exists to revert (see REPORT.md); control = a plan that forgets its cached uid (`backend_ctx->uid = 0`): `p91-uidchurn` rc 134 (GGML_ASSERT) |
| 90 | `test-alloc` `test_meta_host_mapped_weight`, `test-host-mapped` (SKIP on CPU), `test-backend-ops -b CPU -o GET_ROWS` | `p90-test-alloc` rc 0, `p90-host-mapped-cpu` rc 0 (SKIP), `p90-get-rows-cpu` rc 0 and no FAIL | iterative split-state driver without the host-source branch reverted (`90-meta-driver.patch`): `p90-nodriver` rc 139 (SIGSEGV, no PASSED line); Meta device without the host-buft acceptance (`90-meta-all.patch`): `p90-nometa` rc 134, no PASSED line |
| whole | `test-llama-archs` (all archs), `test-alloc`, `test-speculative`, `test-backend-ops -b CPU -o GET_ROWS` | `whole-llama-archs`, `whole-test-alloc`, `whole-speculative`, `whole-get-rows` rc 0 | n/a |
| drops (nok tree) | CPU ops of the touched ranges | `drop-top-k`, `drop-rms-norm`, `drop-scale` rc 0; `drop-files-absent` PASS | n/a |
| drops (K tree) | K7 byte identity, K10 batched top-k unit | `k7-pair-identity` rc 0, `k10-topk-unit` build rc 0, `k10-topk-unit-run` rc 0 on rerun | n/a |

Not on the Mac: CUDA kernels, `test-host-mapped` with a device (window stage 51), Meta over real devices.

Two test-harness facts found on the way (both in the 87 commit's test file, `tests/test-llama-archs.cpp`):
- the async logger dropped the last lines (the MTP check result and the summary) at exit in about half the runs under load (rc was right, output
  truncated; 19 of 40 runs on the Mac); `common_log_flush(common_log_main())` before the return fixes it (all later runs print every line);
- `LLAMA_MTP_FOLD_C=0` selects the catch-up decode, the default the folded decode: each production hunk of 87 is reached by one of the two, so the
  stages run both.
