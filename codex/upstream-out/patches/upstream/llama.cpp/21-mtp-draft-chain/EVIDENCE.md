# 21-mtp-draft-chain

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `151ab739b + d298d3dfd + aa037232c + 21ba77aa8 + d92da203f + da787bb40; validation provenance 4be92a4fa + 374a465b8`.

M, recorded 2026-10-07: `results/2026-10-07-stack7-depth-j5q/full-a5`, frozen stack7 48b82b0d1, J5q np1 k=5, two boots per arm, tokens and acceptance identical. D: integrated stack7 -0.58/-0.62/-0.61 ms/round at 8192/32768/90112 tokens. The chain alone was -0.07/-0.21/-0.21 ms/round (D from the earlier same-binary A/B). Do not assign the combined gain to the chain. Source report `wt/chain-np2/codex/REPORT-chain-np2.md` records exact np2 fallback identity, and np1 q differences up to 1.2e-7 (M); those are historical stack checks, not this rebase.

## Exactness and controls

One single-sequence slot set can draft all steps in one decode. Multi-sequence slot sets always use the ordinary loop, including idle peers; output capacity also keeps fallback limits at np2. Sorted backend top-k and per-step uniforms supply q for rejection verification. Unsupported model/backend/shape conditions fall back. General default is automatic on supported conditions; local form restores the owner opt-in switch. The old control patches in validation/ are provenance, not an apply series.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
