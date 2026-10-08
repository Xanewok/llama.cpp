# p21 np2 decode shape investigation

Labels: M = measured/observed, D = derived, G = guess. Revisions and test settings are literal identifiers.

## Conclusion

M: requesting the chain does not change the recorded np2 decode schedule. Both original logs contain 129 identical `PACK_SHAPE` records in identical order. The failure was in `codex/upstream-out/validation/check_model_results.py:22` at the starting revision: `line.startswith('PACK_SHAPE ')` dropped records with a log prefix. It retained 120 records in the off arm and 122 in the requested arm. Fix the parser, retain strict schedule equality, and leave the production/root series unchanged.

M: the fresh package 21 Mac model gate passes, rc 0, wall 64.354 seconds. np1 identity passes; np2 tokens/counters, byte-identical draft dumps and the complete decode schedule pass. All existing chain controls and the correctly relinked np2 guard-removal mutant are caught. This is validation wall time, not a performance comparison.

## Code path and exactness

Production locations below refer to root replay `485f25ca0e0bb583551487685401a67ea6455f7d`. Its tree equals Mac root `91d09c44b4219a746727a4b079970590b68a4c87`: `f98d88036d51ea1755d1dab7a1947964f241e686` (M; `codex/p21-evidence/tree-proof.txt`). Mac local controls are `ebd22ec42b2412e6ed28bdd8aa4710e850d10435`.

- `common/speculative.cpp:1743`: `draft_chain()` checks the whole configured slot set, `n_seq == 1`, before inspecting active slots or attaching a sampler. At np2 it fails even when a slot drafts alone. The release loop at `:1758` is a no-op because no chain sampler has been attached in this slot set; `:1763` returns false. No chain decode is attempted or announced.
- `common/speculative.cpp:1813`: `draft()` enters the ordinary step loop after that false return. `:1825` batches active drafting slots; `:1896` samples each slot from its own output row; `:1929` stops a slot at its draft limit. One-row decodes reflect idle/completed peers; two-row decodes reflect concurrent drafting. Both arms use this same code and sampler state.
- `common/speculative.cpp:2734`: `params.n_parallel == 1` also guards output-capacity expansion. np2 keeps `n_outputs_max = n_parallel` and `n_outputs_max_per_seq = 1` in both arms. The root has automatic private `mtp_chain = true` and no environment switch; the optional local patch supplies the off/requested controls.
- `src/models/qwen35.cpp:533`: the chain graph also requires `samplers.size() == 1`. Removing only the configured-slot guard would not be a valid way to enable an idle-peer chain. The two-guard mutation control deliberately exposes this defect class.

The apparent inserted/deleted records are not split/merged decodes. `common/chat.cpp:1462` logs `No parser definition detected, assuming pure content parser.` without a newline. `common/log.cpp:334` drains logs on a worker thread and `common/log.cpp:147` writes the message verbatim. The test instrumentation writes an entire shape record directly to stderr at `src/llama-context.cpp:1531`. Depending on the writer order, a shape lands after the newline-free parser text. All records survive in the raw logs; the line-start filter loses some of them. The embedded original lines and full extracted sequences are retained in `codex/p21-evidence/excerpts.txt` and `original-np2-{0,1}.shapes.txt`.

D: this observed difference cannot change tokens, draft probabilities, RNG state or batching; it is a lossy read of the same schedule. With the same work submitted, np2 requests never enter the chain and retain the ordinary sampling path. This does not establish universal bit-identical behavior under different asynchronous arrivals, different backend arithmetic, or the separate np1 chain implementation. The np1 test allows the existing small numerical q difference; it is not a proof of an identical probability law on every backend.

## Fix and regression controls

`validation/check_model_results.py:7` extracts complete shape records anywhere in the log, in order, and requires the number of matches to equal the number of record markers. Missing or changed records still fail schedule equality; malformed records fail explicitly. No schedule contract was relaxed.

`validation/test_shape_records.py` runs the actual checker on copies of the original Mac artifacts, holding token JSON and draft dumps fixed. It accepts newline-free prefixes added to the requested arm and rejects removal, shape alteration, and a malformed output count. It checks the specific assertion for each rejection. M: positive rc 0; each negative rc 1. The original logs, rather than CPU fixtures fabricated by this task, supply every control.

`validation/model_tests.sh:117` now relinks `bw-np2-driver` after rebuilding the guard mutant and again after restoring the source. The driver links `libllama-common.a`; rebuilding that library alone left the previous common implementation in the executable. The mutant judge also checks complete schedule equality, and resource-wait rc 75 is not accepted as a caught mutant.

`validation/model_gate.py` now bounds lock/RAM/model waits and supports the task's 10 GiB minimum. `codex/p21-mac-gate.py` runs the existing model gate in an isolated process group, with a total wall budget of 3600 seconds, resource-wait limit 600 seconds, skip rc 75, and timeout rc 124. M: a zero-wait smoke control returns rc 75 before running a binary. No foreign process was stopped.

There is no new root-series commit: the premise that production changed the schedule was falsified. Root mbox SHA256 remains `5f7ae1b0dafb7c2760764383a6c10eff8dc76b461bffd39933ce3d343acacf97`. Updated package `apply.txt`, `EVIDENCE.md` and `NOTES.md` record the result; prerequisites and both root/local mboxes are unchanged. Validation fix: topic commit `da7518294`; the following topic commit records private package evidence. Neither changes the runtime series.

## Mac evidence and reproduction

Mac run: `/Users/builder/qwen38-upstream-pack/20261008/21-mtp-draft-chain/`. Original failure artifacts are archived in `p21-np2-shape/original/`; fresh results are in `model-tests/`; the bounded gate writes `p21-np2-shape/{gate.log,gate.rc,elapsed-seconds}`. Local evidence: `codex/p21-evidence/{summary.json,SHA256SUMS.txt,excerpts.txt,tree-proof.txt,wait-skip.txt}`. Full logs remain on the Mac; raw SHA256 hashes identify both original and fresh artifacts.

Recipe: CPU-only existing Clang build, CUDA/Metal/BLAS off, native off, `armv8.2-a+dotprod`, `GGML_CCACHE=ON`; incremental builds under the existing Mac RAM gate, nice 19, `-j4`. Qwen3.5-4B-UD-Q4_K_XL, explicit contexts 4096, four model threads, at most one model process. Driver: seed 1729, k=3, 16 requested tokens for slot 0 and 32 for slot 1; cold and prefix-reuse phases; np1 and np2. No compile, test, model run or gpuq command ran on Linux. Every command used `CUDA_VISIBLE_DEVICES=''`, and git used `GIT_NO_LAZY_FETCH=1`.

From the Mac run root, after restoring/removing prior shape instrumentation and with the local control patch already applied:

```sh
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
nice -n 19 python3 validation/test_shape_records.py 21-mtp-draft-chain/p21-np2-shape/original
nice -n 19 python3 validation/p21-mac-gate.py "$PWD"
```

The wrapper invokes `PACK_LOCAL_APPLIED=1 PACK_GATE_MIN_FREE_GIB=10 PACK_GATE_WAIT_SECONDS=600 bash validation/model_tests.sh "$PWD" 21-mtp-draft-chain`. M: Mac `bash -n` and Python byte-compilation passed; local `git diff --check` passed.

| Check | M result |
|---|---|
| np1 tokens/counters and draft IDs | Equal; 24 draft rows, max_abs_dq = 6.000000007944095e-08; chain announced |
| np1 schedule, expected optimization | 48 off vs 32 chain-on records |
| np2 tokens/counters and draft dump | Byte-equal; 78 draft rows, max_abs_dq = 0 |
| np2 schedule | 129 vs 129, exactly equal; one-slot and two-slot draft shapes present; no chain announcement |
| Reverse uniforms | Tokens/counters and draft dump differ; chain announced |
| Shifted embeddings | Tokens/counters and draft dump differ; chain announced |
| Both np2 guards removed, driver relinked | rc 0, but tokens/counters, draft dump and schedule differ; 132 shapes, chain announced; control caught |
| Restored source/build/driver | PASS; `git diff --exit-code` = 0 |

## M2 interaction and limits

Read-only source: `wt/chain-m2/codex/SPIKE-chain-m2.md`. M2 generalizes the graph to whole sequences in step-major/unified and seq-major/per-stream layouts, samples each sequence with its own chain sampler, refuses a requested chain when the graph does not implement it, and transfers RNG state between chain and step-loop samplers. Its recorded Mac matrix includes multi-slot, unified KV, greedy-peer path switches, and wrong-sequence/no-RNG-transfer controls; those historical results were not rerun here.

D: integrating and validating M2 would replace package 21's blanket `n_seq > 1` fallback for supported multi-sequence rounds. It would not eliminate fallback for greedy peers, capacity limits, unsupported layouts/models/backends or attach/decode failures. Its two-sequence chain deliberately changes the schedule, so a future M2 gate must require chain engagement and exactness controls instead of today's np2 off/requested shape equality. The existing parser fix remains necessary for either contract. This task does not port M2 or claim M2 GPU readiness.

No GPU correctness or speed was measured. No new GPU A/B item is warranted for a validation-only parser fix; the existing runtime binaries are unchanged. np1 q differences, general chain failure behavior, M2 integration and GPU validation remain separate concerns. No network duplicate check, public text, PR, push or other-branch rewrite was performed.
