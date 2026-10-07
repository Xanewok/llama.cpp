# 73-meta-host-view-plan

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `376e5c8d9`.

M historical source validation: `wt/nh1-fold-c/codex/REPORT-nh1-meta.md`; CPU Meta fixture planning aborts before the fix; mirrored/split VIEW, CPY and SET_ROWS cases after the fix. This is a CPU Meta bug with no measured GPU performance effect. 2026-10-07 stack7 folding performance is evidence for 72, not this planner fix.

## Exactness and controls

Meta buffers can report host when backed by CPU, while their data pointers remain Meta placeholders. Planning cannot skip these views as native host no-ops. Six fixture cases compare actual cache bytes after three direct computations with the same graph UID (the Meta internal planning cache is reused).

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.

Mac test-port correction: master has no Meta graph_plan_create/compute hooks. Use three graph_compute calls with one graph UID, testing initial planning and internal-plan reuse instead of the stack-only explicit graph-plan API. No production changes.
