# 21-mtp-draft-chain

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `151ab739b + d298d3dfd + aa037232c + 21ba77aa8 + d92da203f + da787bb40; validation provenance 4be92a4fa + 374a465b8`.

M, recorded 2026-10-07: `results/2026-10-07-stack7-depth-j5q/full-a5`, frozen stack7 48b82b0d1, J5q np1 k=5, two boots per arm, tokens and acceptance identical. D: integrated stack7 -0.58/-0.62/-0.61 ms/round at 8192/32768/90112 tokens. The chain alone was -0.07/-0.21/-0.21 ms/round (D from the earlier same-binary A/B). Do not assign the combined gain to the chain. Source report `wt/chain-np2/codex/REPORT-chain-np2.md` records exact np2 fallback identity, and np1 q differences up to 1.2e-7 (M); those are historical stack checks, not this rebase.

## Exactness and controls

One single-sequence slot set can draft all steps in one decode. Multi-sequence slot sets always use the ordinary loop, including idle peers; output capacity also keeps fallback limits at np2. Sorted backend top-k and per-step uniforms supply q for rejection verification. Unsupported model/backend/shape conditions fall back. General default is automatic on supported conditions; local form restores the owner opt-in switch. The old control patches in validation/ are provenance, not an apply series.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

### np2 schedule gate, 2026-10-08

M: `validation/model_tests.sh` passed on mini-builder, rc 0, wall 64.354 seconds. Root `91d09c44b4219a746727a4b079970590b68a4c87` plus local `ebd22ec42b2412e6ed28bdd8aa4710e850d10435`; restored source clean. No root mbox or runtime behavior changed. Qwen3.5-4B-UD-Q4_K_XL, CPU only, contexts 4096, four threads, seed 1729, k=3, cold and prefix-reuse phases, np1/np2. Model gate required 10 GiB free before each job; resource/lock waits bounded to 600 seconds (skip rc 75), total gate bounded to 3600 seconds (rc 124).

M: np1 tokens/counters and draft IDs equal over 24 draft rows; max_abs_dq = 6.000000007944095e-08. np2 tokens/counters and draft dumps byte-equal over 78 draft rows; all 129 decode shape records equal, with idle-peer and two-slot drafting exercised. No np2 chain announcement. Reverse-uniform and shifted-embedding controls change tokens/counters and drafts. Removing both np2 guards, then rebuilding and relinking the driver, changes tokens/counters, drafts and schedule (132 records, chain announced). Restored build passes.

The prior shape failure was validation-only: `check_model_results.py:22` used `startswith('PACK_SHAPE ')`. `common/chat.cpp:1462` writes a debug message without a newline, and the asynchronous logger can prepend it to the direct shape write. M: original full np2 logs contain 129 equal records, but line-start filtering found 120 vs 122. The corrected parser extracts all records and rejects malformed records. Real-artifact parser controls accept prefixed records and reject missing, changed or malformed records; the schedule equality contract remains unchanged.

Evidence: `codex/REPORT-p21-np2-shape.md`, `codex/p21-evidence/summary.json`, record excerpts and raw-log SHA256 hashes. Full Mac artifacts: `/Users/builder/qwen38-upstream-pack/20261008/21-mtp-draft-chain/{model-tests,p21-np2-shape/original}`. GPU correctness/performance remain untested by this task.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
