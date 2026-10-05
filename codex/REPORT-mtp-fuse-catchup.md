# MTP catch-up fusion: stopped at the CPU identity gate

## Conclusion

H1 is not ready. The candidate removed the steady catch-up forward, but both CPU smoke prompts changed their generated tokens, and draft counters also changed. Section A of the supplied planner explicitly says to stop and report if CPU identity fails. The fusion source changes were therefore removed from the working source and preserved in `codex/mtp-fuse-catchup.patch`, relative to step-0 commit `f26106888`.

The retained source change is step 0 only: `common/speculative.cpp:1635` retains one hidden row when the sequence has no output before its final row. A verify batch has earlier output rows and still retains its complete hidden matrix. A prompt chunk only needs its final hidden row for the cross-batch pair. This avoids copying and retaining the full prompt hidden matrix without changing the eager draft decode.

Confidence: high that the tested fusion violates the requested token/counter identity contract. The cause of that violation is unknown. Batched versus single-row floating-point computation is a hypothesis, not an established diagnosis. The sampling-law unit test does not prove equivalence of the fused model path.

## Intended invariant and source verification

Let the target verify tokens be `t[i]` at `P+i`, for `i=0..k`, and let `h[i]` be the corresponding target hidden rows. Let `h_prev` be `pending_h` before verification, at `P-1`.

The draft context must evaluate `(t[0], h_prev)` at `P`, then `(t[i], h[i-1])` at `P+i`. After accepting `n` draft tokens, it must retain rows `0..n`. The next first draft query is `(id_last, h[n])` at `P+n+1`.

| Case | Required draft inputs |
|---|---|
| Full acceptance | Verify rows `0..k`, then the new sampled token paired with `h[k]`. |
| Partial acceptance | Verify rows `0..n`, then the rejection replacement paired with `h[n]`. |
| Zero acceptance | Verify row `0` paired with `h_prev`, then the replacement paired with `h[0]`. |
| Target checkpoint restore | Acceptance has not committed. Keep `h_prev`; discard the rejected deferred batch when replay processing replaces it. |
| Draft FULL checkpoint | Eager fallback is required: the checkpoint saved before drafting must already contain the accepted prefix. |
| `seq_rm` | Remove positions at and after the next sampled-token position. The accepted prefix must remain; D1 rebuilds the new query row. |
| np2, one sequence not drafting | Include that sequence's committed deferred prefix with no output in the other sequence's D1 batch. If neither drafts, drain the committed prefix before the next process batch overwrites its hidden rows. |

Verified baseline locations, in the retained source:

- `common/speculative.cpp:1540`: `process()` builds eager C with the previous target hidden row for each token.
- `common/speculative.cpp:1656`: `draft()` builds D1 from `id_last`, `pos0`, and `pending_h`, then D2 through D5 from draft hidden rows.
- `common/speculative.cpp:1825`: `accept()` selects target hidden row `n_accepted` into `pending_h`.
- `tools/server/server-context.cpp:530`: the server adds the sampled token and speculative tokens to the verify batch.
- `tools/server/server-context.cpp:3288`: the server sets the next draft query position and token.
- `tools/server/server-context.cpp:3328`: draft cleanup restores a FULL draft checkpoint when needed, then removes the generated draft region.
- `tools/server/server-context.cpp:4065`: the server processes the target batch through the speculative driver before acceptance.
- `tools/server/server-context.cpp:60`, `:4254`: `server_accept_replay()` advances the target sampler over already accepted replay tokens, then samples the bonus token.
- `tools/server/server-context.cpp:4278`: checkpoint restore returns before `common_speculative_accept()`; the speculative driver must not treat this batch as committed.
- `tools/server/server-context.cpp:4304`, `:4338`: acceptance selects the committed hidden row, then `slot.mem.seq_rm()` removes the rejected suffix from both contexts through `common_memory::seq_rm()` at `common/common.cpp:1566`.

These are intended state invariants, not proof that the rejected candidate meets the complete exactness contract. Checkpoint replay and mixed np2 fusion were not validated after the np1 stop condition fired.

## Rejected candidate

`codex/mtp-fuse-catchup.patch` contains a default-off `LLAMA_MTP_FUSE_CATCHUP=1` experiment. It adds an optional list of verifying sequence IDs to `common_speculative_process()`, preserving eager behavior for existing callers. The server derives this list from existing `spec_i_batch` entries; it excludes FULL draft checkpoints. The MTP driver also excludes chained heads and shared target/draft memory.

`process()` saves verify tokens, positions, the pre-verify hidden row, and target hidden rows. It omits C for these sequences and keeps the pre-verify carry until acceptance. `accept()` records the accepted prefix. `draft()` adds that prefix without output, then adds the sampled token with output. Only the combined decode is traced as D1. A sequence that does not draft is flushed without output. A later `process()` drains a committed deferred prefix before overwriting its storage; an uncommitted replay batch is discarded.

The first query position is asserted to equal `P+n+1`, and deferred rows are asserted to have contiguous positions and one sequence. No sampling implementation or RNG code was changed. That is insufficient to establish exactness: the observed token/counter mismatch remains blocking.

## CPU evidence

All commands used `CUDA_VISIBLE_DEVICES=''` and `GIT_NO_LAZY_FETCH=1` for Git. Compile commands used `bin/buildgate` with `-j4`; test servers were renamed `cpu-serving`, used `nice -n 19`, and used at most eight CPU threads. Meta tests use four threads per backend. No GPU binary or GPU inspection command was executed. Local HTTP tests required sandbox escalation for loopback sockets; they made no external network requests.

Configuration: `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF`. The shared ccache was read with `CCACHE_READONLY=1`, and `CCACHE_TEMPDIR` was placed inside the worktree because the sandbox cannot write the shared cache. Configuration and CPU server builds passed.

The fixture was `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`. Requests used probabilistic k=5, `p_min=0`, temperature 1, top-k 20, top-p 0.95, min-p 0.05, q8_0 target KV, f16 draft KV, backend top-k, and fixed seeds. Commands and full responses are stored under `codex/mtp-results/`.

M: fusion smoke, two prompts, seed 1, 24 generated tokens per request:

| Prompt | Base draft / accepted | Fusion draft / accepted | First differing token index |
|---|---|---|---|
| Iterative Fibonacci | 42 / 14 | 37 / 14 | 16 |
| Blue sky | 42 / 14 | 33 / 15 | 3 |

Evidence: `codex/mtp-results/smoke/*/records.json`, `summary.json`, and `trace.jsonl`; compact counters are in `codex/mtp-results/fusion-summary.json`. M: candidate traces contain full-accept D1 batches of seven rows and zero-accept D1 batches of two rows, with no steady C. Baseline traces contain C followed by one-row D1.

M: the same two prompts and seed with fusion disabled matched all generated tokens, draft counts, and accepted counts. Evidence: `codex/mtp-results/eager-control/summary.json`. This isolates the smoke failure to fusion rather than step 0.

`test-sampling` passed. `rejection-law` passed for full and candidate rows. M: each law case used 60,000 draws; the true-q acceptance was 0.6622 versus the calculated law value 0.6568, z=2.78, with chi-square p=0.0371. Both wrong-q negative controls failed the law gate, with acceptance 0.9737 and z=163.45. Evidence: `codex/mtp-results/sampling.log` and `rejection-law.log`.

The requested row-pair mutant is recorded in `codex/mtp-row-negative-control.patch`: apply it after the candidate patch to replace the shifted hidden row with `verify_h[i]`. It was not built or run because the unmutated candidate already failed the mandatory stop gate. The requested fusion matrix of at least twenty prompts and two seeds, Meta fusion checks, checkpoint replay checks, and mixed np2 fusion checks remain unperformed.

M: the retained step-0 build matched token IDs, draft counts, and accepted counts against the original baseline for both CPU np1 and concurrent CPU np2: two prompts, two seeds, four requests per arm per configuration, sixteen requests total. Evidence: `codex/mtp-results/step0-matrix/summary.json`. Meta(CPU,CPU) did not start: the baseline CLI rejects `--device CPU,CPU` with `invalid device: CPU`; the historical `LLAMA_ARG_ALLOW_CPU_DEVICE=1` variable is not implemented in this stack. Meta server coverage needs an in-process fixture such as the existing `codex/reuse-server-cpu.cpp`, which injects CPU backend devices directly. No Meta result is claimed.

## GPU/parser evidence and blocked deliverables

No CUDA freeze or queueable GPU item was produced after the CPU stop condition. There is no frozen commit, ggml CUDA stamp, or GPU timing evidence for fusion. `codex/gpu_ab.sh --fixtures` exercises the prepared parser adapter; all other modes refuse to proceed while the queueable experiment is absent.

`codex/mtp_gpu.py` reuses the installed depth harness and supplies acceptance-count equality and complete-round AllReduce gates. M: parser tests used the real `orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json`, recorded memory and kernel-journal fixtures, and the stack4 J5q canary artifact at `/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/canary-a1/1-patch`. The latter yielded 92 steady intervals and 95 complete eager rounds with 140 AllReduce calls. A one-count acceptance mutation and an eager trace presented to the fused-count gate both failed. Evidence: `codex/mtp-results/mtp-parser-fixtures.log`.

`attrib_pack.py:254` accepts `VC?(D+)`, so C is optional in round selection. Its kernel classifier at `:243` still requires a recognized one-column draft head; the fused CUDA kernel classification was not tested. `harness/lib.py:463` sums the phases that exist and does not require C. `harness/depth.py:179` rejects np1 `identity = "off"`; the prepared adapter handles this without editing any harness file. A future item needs the same canonical render plus entry substitution pattern used by `host-bundle/codex/write_host_bundle_item.py`; simply rendering an off-identity np1 TOML through the unmodified default entry is insufficient.

D: the intended complete round has 128 target AllReduce calls plus ten draft calls, totaling 138 instead of 140. This is a proposed canary gate, not a measured result for the candidate. G: potential savings were 0.8 / 1.0 / 1.2 ms per round at 8k / 32k / 90k. The CPU failure prevents treating any of them as a speed claim.

A future A/B should use the frozen stack4 root and a corrected frozen fusion build, ABBA fresh boots, the cloned host-bundle IQ4_XS serving configuration, all three depths, a one-boot patch canary at 8k, exact-PID cleanup, frozen-library/digest checks, journal checks, and a wall budget. GPU token identity cannot be assumed after the demonstrated CPU divergence; use `identity = "off"` with explicit acceptance equality. A single new canary boot cannot prove cross-arm acceptance equality without a compatible recorded baseline request; that reference remains an open prerequisite.

## Upstream split and next action

Step 0 is separable and follows the existing output-row contract. Fusion needs a separate change after its row-by-row model logits and candidate distributions explain the CPU mismatch. No upstream package, PR text, comment, or submission was created. Offline work did not verify current upstream discussion or duplication.

Next action: compare eager C+D1 and fused D1 logits/candidate distributions on the same fixed target hidden rows and draft KV prefix. Distinguish a state/pairing defect from batch-dependent numerical changes before reopening the fusion design or preparing a GPU item.
