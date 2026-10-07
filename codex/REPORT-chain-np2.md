# MTP chain at np2

Labels: M = measured or directly observed, D = derived, G = guess. No GPU execution or local compilation/model run was performed.

## Conclusion

The fixed np2 CPU runs are exactly identical with the chain request off and on (M). The source establishes a dispatch mismatch: the caller counts drafting sequences, while the Qwen3.5 chain graph counts attached samplers. An idle server slot keeps its sampler, so the caller requests the chain with two attached samplers and the model takes the ordinary MTP graph (D, from the constructor, full-map forwarding and graph predicate). The fix keeps every multi-sequence slot set on the ordinary step loop for its entire lifetime.

## Root cause

- `common/speculative.cpp:1867` formerly allowed a multi-sequence slot set whenever only one `dparams[s].drafting` was true. At `common/speculative.cpp:1895` it fills all chain rows with `dp.id_last` and fills hidden-state rows after the first with zeros; the chain graph is responsible for replacing those placeholders.
- `common/speculative.cpp:1522` installs a backend sampler for every configured sequence. `mtp_chain_release` restores a sampler rather than removing the idle slot's sampler.
- `src/llama-context.cpp:2907` passes the full attached-sampler map to the graph. `src/models/qwen35.cpp:640` requires `samplers.size() == 1` to enter the chain graph. With two attached samplers, it evaluates the placeholder rows through the ordinary graph. `src/llama-context.cpp:1289` accepts the chain request without checking that graph predicate.
- The two paths also keep independent RNG state in `smpls` and `mtp_chain_seqs`; switching after fallback rounds would require explicit state transfer. The observed first mismatch precedes such a transition, so this is a secondary unsupported behavior rather than the cause of the first mismatch.

The reproduction uses the exact driver workload with mixed prompt/decode batches and a second phase with cached prompts. On the baseline, round 1 requests the chain with `slots=2 active=1`, rounds 4 onward fall back with `active=2`, and later rounds return to the chain after slot 0 finishes (M). The per-round records include the path, slot count, drafting count and each sequence's draft length in `codex/chain-np2-evidence/baseline-summary.json`.

At round 1, step 0, both modes draft token `20`; the largest probability difference is approximately `6e-8` on plain CPU (M). At step 1, chain-off drafts `11`, chain-on drafts `15`, and the candidate sets differ substantially (M). This precedes any path transition. The baseline first target-token mismatch is index 58 in phase 0 and index 59 in phase 1, slot 0, in both Meta modes (M).

The reproduction build disables i8mm with `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`. Small probability differences from host/backend normalization exist, but they do not explain feeding placeholder rows through the wrong graph. This task does not claim universal equality across different CPU batch shapes or ISAs.

## Fix and contract

`common/speculative.cpp:1867` requires `n_seq == 1`, independent of which slots currently draft. `common/speculative.cpp:2869` reserves chain output capacity only with `params.n_parallel == 1`, so an np2 chain request also retains the chain-off draft-context limits.

Contract: a multi-sequence slot set always uses the step loop, including partial prefill, idle peers, completion tails and prompt-cache reuse. The np1 chain path is unchanged. Correct multi-sequence chain graphs and RNG-state transfer are outside this change.

Source commit: `da787bb40`. The Mac verification build used `0c111d934`; the amendment changes only the explanatory comment and commit message, not executable source statements.

## Tests and negative control

All compilation and model execution occurred in `~/qwen38-chain-np2/src` on `mini-builder`, with `CUDA_VISIBLE_DEVICES=''`, `nice -n 19`, Ninja, CPU-only backends and the owner's `-j10` Mac build setting. The model is `~/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. Every context is explicitly 4096; model runs use at most eight CPU threads, run sequentially, and wait for the RAM gate.

Reproduction and verification commands are recorded in `codex/chain-np2-evidence/mac-reproduce.sh`, `mac-verify.sh` and `mac-graph-proof.sh`. The temporary instrumentation is stored as patches and is absent from the source commit.

- Baseline `48b82b0d1` with round-only instrumentation: both Meta modes reproduce the identity failure (M). The restored `check-np2-interactions.sh --check-only` exits nonzero on the recorded old-binary results, and the control separately requires successful outputs plus `MTP chain: k=3` and `path=chain slots=2 active=1` in both logs (M).
- Fixed plain CPU np2: tokens, counters, complete draft IDs/candidates/q dumps and all 241 decode phase/row-count pairs are identical (M).
- Fixed Meta(CPU,CPU) np2: the same tokens, counters, dump and decode-shape identity checks pass (M).
- Stage 23 identity pairs: plain and Meta(CPU,CPU), each at `k=3` and `k=5`, pass tokens, counters, draft/candidate IDs and chain engagement; maximum probability difference is `1.2e-7` in every pair (M). Repeat and mutation controls are pending.
- Direct model-graph instrumentation: pending.

Completion is blocked by the Mac RAM gate, not a test failure. The last recorded gate reported 59% available memory on a 16 GiB machine (M), approximately 9.44 GiB (D), below the required 10 GiB start threshold. No model process was running. `mac-verify.sh` remains detached and waiting; after the repeat and mutation controls it runs `check-chain.py` and writes `~/qwen38-chain-np2/verify.rc`. `mac-graph-proof.sh` waits for that result, then runs the direct graph probe and restores/rebuilds the fixed source. No gate was bypassed and no foreign process was stopped.

To finish: free RAM on `mini-builder`, wait for `verify.rc` to contain `0`, and require `PASS: graph proof completed and fixed source restored` in `~/qwen38-chain-np2/graph-proof.log`. Copy `np1/report.json` and the graph proof back into this worktree, then update this report. The full stage 23 gate is not yet claimed to pass.

Fixed cold-phase counters, identical for chain-off and chain-on (all M):

| Backend | Slot | Drafted | Accepted | Verify rounds |
|---|---:|---:|---:|---:|
| Plain CPU | 0 | 47 | 47 | 16 |
| Plain CPU | 1 | 65 | 57 | 22 |
| Meta(CPU,CPU) | 0 | 48 | 46 | 17 |
| Meta(CPU,CPU) | 1 | 65 | 57 | 22 |

Cached-phase results and complete shape sequences are in `codex/chain-np2-evidence/fixed-summary.json`; the raw logs, traces and dumps remain in the matching `baseline/`, `fixed/` and `np1/` directories on the Mac.

`check-np2-interactions.sh` now asserts full output equality, successful nonempty traces, no chain announcement or `DC` decode, both one- and two-sequence draft rounds, identical V/C/D shapes and byte-identical draft dumps. `--check-only` validates recorded runs without loading a model; `S7_NP2_BIN` permits an explicitly selected binary.

Local static checks: `git diff --check`, `bash -n` for the changed shell scripts, and `python3 -m py_compile` for the evidence summarizer (M).

## Risks and GPU follow-up

The fallback removes chain acceleration from every multi-slot configuration even when only one slot is busy. This is the intended M1 contract. Direct users of the low-level `llama_set_mtp_chain` API remain responsible for satisfying its model-graph requirements; this patch scopes the common speculative driver.

GPU correctness and performance are unmeasured. The parent needs an untraced np2 chain-off/on item using the same frozen fixed binary, Swift 1.5, tensor split, probabilistic drafting, rejection verification, mixed short/long prompts and a cached second phase. Compare per-slot token IDs, drafted/accepted/round counters and prefix reuse. Use deterministic slot/task scheduling; exclude different batch composition before attributing a mismatch to the flag. Interleave off/on/on/off boots, retain exact-PID shutdown, queue-cutoff-scoped hazard checks and the harness wall budget. Test response parsers against recorded real llama SSE fixtures, then run a one-boot canary that exercises every parser and gate within ten minutes before the full item. No GPU speed claim follows from the CPU identity tests.
