# 71-mtp-verify-bulk-read

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `3b5176efd + f8a00adc5, final production path of 81f954b2a`.

M historical CPU identity: `wt/host-trims-v2/codex/REPORT-host-trims.md`, CPU and Meta(CPU,CPU), np1/np2, seeds 1729/2718, tokens/counters/draft candidates equal; wrong-slot hidden-row control rejected. D recorded host-trims-v2 differences in docs/stack7.md: -0.63/-0.16 (matched -0.33)/-0.41 ms/round at np2; combined with indexed seq_rm, not isolated. M/D integrated stack7 campaign: `results/2026-10-07-stack7-np2-real-j5q/full-a3`, see package 69.

## Exactness and controls

The target nextn rows are unmasked and indexed by raw token row; each sequence already requires a contiguous interval. Acquire the dense pointer after catch-up, then copy that interval in one memcpy. Keep the earlier pointer scoped to catch-up input construction. Sampling/RNG are unchanged.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

M: Qwen3.5-4B np2 plain-CPU MTP test passes: patch/baseline JSON (tokens and counters) is byte-identical at seed 1729, fixed 16+16 output tokens. Wrong-slot-row mutant differs. Source restored, final build and clean-source check pass. Command: bash validation/model_tests.sh "$PWD" 71-mtp-verify-bulk-read; logs: codex/upstream-out/mac-evidence/final/71/.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
