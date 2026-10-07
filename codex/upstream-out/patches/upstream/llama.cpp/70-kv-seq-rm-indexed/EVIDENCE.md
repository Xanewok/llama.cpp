# 70-kv-seq-rm-indexed

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `36cb09139 (host-trims-v2 retains this algorithm)`.

M historical CPU proxy, `wt/host-trims-v2/codex/REPORT-host-trims.md`: at capacity 102400, np1 100.542 -> 1.505 us and np2 204.761 -> 3.178 us (median of five trials, 2000 repetitions). These include reinsertion and are not GPU round savings. D, recorded `docs/stack7.md`: host-trims-v2 np2 real-prompt differences -0.63/-0.16 (matched -0.33)/-0.41 ms/round; combined indexed-removal and bulk-read change. The stack7 np2 integrated result is confounded by out-projection/getter changes.

## Exactness and controls

For a nonnegative sequence ID, walk [p0,p1) in seq_pos_get, advancing before seq_rm erases the entry. Distinct cell removals commute; choosing the minimum freed physical index preserves the old scan head. Keep the negative-sequence full scan. Tests compare position/membership/index state with an independent scan oracle, reversed and repeated ranges, shared cells, nonmonotonic positions and the next physical head.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
