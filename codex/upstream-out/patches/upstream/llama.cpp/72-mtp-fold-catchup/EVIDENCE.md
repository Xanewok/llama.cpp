# 72-mtp-fold-catchup

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `7f1fe261f (runtime behind report-only 186226ceb); Meta prerequisite 376e5c8d9`.

M, recorded 2026-10-07 `results/2026-10-07-stack7-depth-j5q/full-a5`: two boots per arm, J5q np1 k=5, cross-arm tokens and acceptance identical. D: fold alone -0.23/-0.19/-0.22 ms/round at 8192/32768/90112 tokens; full s7f versus stack6 -0.80/-0.81/-0.83 ms/round. Source docs/stack7.md older "never measured" note is superseded by this campaign. No GPU run of this rebased automatic-default form.

## Exactness and controls

Enable when the existing safety predicates accept: single sequence, one MTP head, shared model/device layout, separate draft cache, no adapters or pipeline parallelism, consecutive tokens in one bounded ubatch. Other paths retain catch-up. An armed pending_h pointer remains stable until the next target decode. The folded cache mutation must roll back if target decode fails.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.

Mac test-port correction: master has no Meta graph_plan_create/compute hooks. Use three graph_compute calls with one graph UID, testing initial planning and internal-plan reuse instead of the stack-only explicit graph-plan API. No production changes.

General-default safety adaptation: fold builds the draft K/V using the target graph's cparams. Require matching resolved RoPE/YaRN parameters, flash-attention cache layout and op/KQV offload policy; reject training contexts. The original opt-in form did not check these settings. This gate preserves fallback for differing custom draft contexts; it needs a fresh runtime refusal test.
