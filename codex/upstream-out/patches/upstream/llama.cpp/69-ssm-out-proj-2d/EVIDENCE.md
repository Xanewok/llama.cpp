# 69-ssm-out-proj-2d

Labels: M = observed/measured, D = derived, G = guess. Settings, revisions and identifiers are literal. Historical evidence is copied from the sources below, not rerun by this task.

## Sources and 2026-10-07 evidence

Source commits: `ce555540a`.

M, recorded 2026-10-07, `docs/stack7.md`: in-model Q6_K ssm_out at np2 67 -> 38 us. D/G: the earlier projected J5q -1.6 and IQ4 -0.5 ms/round were estimates, not a standalone end-to-end A/B. M, recorded `results/2026-10-07-stack7-np2-real-j5q/full-a3`: two boots per arm, real prompts, identity disabled. D: integrated stack7 -1.130/-1.012/-1.277 ms/round at 8192/32768/90112 tokens; not isolated to this patch.

## Exactness and controls

Flatten the contiguous token/sequence dimensions before ssm_out, yielding an ordinary 2D input. The output remains [n_embd, n_seq_tokens*n_seqs]. Reuse build_lora_mm and existing scales. The validation outproj-cpu driver records complete np2 logits/tokens and rejects a batched projection shape with rc 42; baseline is the graph-rank negative control.

## Rebased verification (2026-10-08)

Base: `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`. Apply prerequisites/*.patch in lexical order, then root patches. Fresh Mac git-am/build/test results and commands: `apply.txt`; collector report: `codex/REPORT-upstream-pack.md`. A successful apply/build does not establish GPU correctness or speed.

M: Qwen3.5-4B-UD-Q4_K_XL np2 CPU probe passes. Baseline graph rank returns rc 42 (408 batched projection observations); patched rank has 408 observations, none batched. Complete logits/tokens match byte-for-byte, SHA256 e1b0c1664518c59287c59da2a0acf1233609ea7e4239538d719057cc2b1c3170 in both arms. Command: bash validation/model_tests.sh "$PWD" 69-ssm-out-proj-2d. Logs: codex/upstream-out/mac-evidence/final/69/.

## Limits

No GPU execution by this worker. No GCC 13 build under the Mac-only rule. Fresh upstream PR/issue state: dupcheck pending (no network). Historical GPU gains belong only to the named stack revisions. Human code review, issue discussion for features and owner-written public text remain required.
