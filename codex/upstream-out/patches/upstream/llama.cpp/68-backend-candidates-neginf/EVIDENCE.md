# 68-backend-candidates-neginf

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `062667905 + 74e70bb95`.

M, recorded 2026-10-07: `results/2026-10-07-cand-neginf-depth-j5q/full-a1`, frozen `cand-neginf` 4c75c6d64 on stack6; Swift 1.5 J5q, np1, tensor split, probabilistic MTP k=5, two boots per arm; tokens and acceptance identical. D from M intervals: -0.649/-0.260/-0.077 ms/round at 8192/32768/90112 tokens, SE 0.062/0.036/0.046 ms/round. The 90k result overlaps zero. This is a stack result, not a gain for the rebased series.

## Exactness and controls

Only unique -inf bans qualify. Device top-(k+1+n_ban) contains the best k+1 unbanned IDs. Finite logits and strict ordering certify the host candidate set; ties or non-finite logits read the full row. Finite/duplicate bias entries or an oversized candidate prefix retain the ordinary chain. No RNG algorithm changes. The no-extension mutant must fail with the shipped synthetic/Meta(CPU,CPU) test.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

M: test-sampling and test-llama-archs --tp-sampling -s 42 / -s 1729 pass. Removing +n_ban from backend top-k extension makes the regression fail with rc 1. Restored source builds and git diff --exit-code passes. Logs: codex/upstream-out/mac-evidence/final/68/.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
