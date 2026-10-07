# Private package facts: 69-ssm-out-proj-2d

Verdict: FEATURE (performance).

## Relation to existing packages

Standalone, no prerequisites. No sampling changes, ggml API changes or CUDA-specific dispatch. Applies to the existing recurrent model builders (qwen35, qwen35moe, qwen3next, mamba-base, plamo2, clef, qwen4exp).

## Design

Flatten the contiguous token/sequence dimensions before ssm_out, yielding an ordinary 2D input. The output remains [n_embd, n_seq_tokens*n_seqs]. Reuse build_lora_mm and existing scales. The validation outproj-cpu driver records complete np2 logits/tokens and rejects a batched projection shape with rc 42; baseline is the graph-rank negative control.

## Changes beyond original commits

Production diff unchanged. Adapt the private CPU probe from legacy llama_batch/llama_decode to batch_ext/llama_process and reduce its threads to four.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.
