# stack8 integration

Labels: M = measured or directly observed; D = derived from source or arithmetic; G = estimate. GPU execution, local compilation/model execution, queue submission/cancellation and network posting were not performed.

## Status

Source integration and Mac validation pass (M). The build window is ready for the parent after choosing the duo fold value. The CUDA build, freeze and GPU item contract checks are deferred to that window.

## Commits and origins

| Commit | Subject | Origin |
|---|---|---|
| `ad843282f` | common : restrict MTP draft chain to single-sequence slot sets | `da787bb40`, exact source cherry-pick |
| `c579a9127` | common : sampling: keep the backend candidates under a -inf logit bias | `062667905`, `74e70bb95`, `935c302b9` source log; overflow correction and large-top-k regression |
| `2ecda894d` | codex : integrate stack8 Mac identity checks and controls | strict checker from `374a465b8`; candidate driver/comparer from `topic/cand-neginf-mac`; stack integration |
| `8774e3ea0` | codex : prepare stack8 CUDA window and GPU item templates | stack7 window/harness structure; stack8 templates and local snapshot destination |
| `9cf7e13b3` | codex : document stack8 source and prepare final unit recheck | final-source unit payload and initial report |
| `02ae1f30a` | codex : replay stack8 fallback and candidate proof controls | recorded parser inputs and Mac fallback evidence |
| `0c0dc0d72` | codex : require nonempty banned backend candidate proofs | reject zero-width/empty-ban proof logs |
| `9054a6ee2` | codex : preserve stack8 build and contract failure gates | errexit, explicit rollback model and contract-check failure propagation |
| `0e6826c8d` | codex : validate stack8 experiment and window structures on Mac | harness schema, budget, fold and manifest checks |
| `e45e46d84` | codex : run stack8 template rechecks without repeating model stages | separate static Mac payload |
| `63e659d71` | codex : prevent stack8 unit gate starvation and preserve controls | one-second gate poll; recorded identity, fallback and parser evidence |

Starting HEAD was `ddedffacb`, with the stack7 snapshot migration and pinning commits after the requested stack7 source base `48b82b0d1`. The existing `codex/PROMPT.md` and `codex/PURPOSE` changes were preserved and committed. Neither component produced a merge conflict (M). Only `topic/stack8` was rewritten to fold the overflow correction into its candidate component; source branches were not modified.

## Design and source changes

Every production source change:

1. `common/speculative.cpp:1871`, `:2873`: require `n_seq == 1` for chain dispatch and `params.n_parallel == 1` for the chain output reservation. Multi-slot sets use the step loop throughout their lifetime, including idle peers and completion tails. Attached idle-slot samplers make active-drafting count an invalid chain gate (D).
2. `common/speculative.cpp:1499`: emit `MTP chain disabled: needs one sequence (n_seq=2)` when the chain is requested for a multi-slot set. This additional stack integration log lets the canary prove fallback positively.
3. `common/sampling.cpp:153`, `:210`, `:585`: recognize an effective, unique list of `-inf` biases; request `k + 1 + n_ban` raw backend candidates; discard banned IDs before the existing finite/strict-order check; retain the full-row fallback for ties, nonfinite banned logits, finite biases, duplicates or excessive candidate width. Add the `banned()` lookup and the candidate `LOG_DBG` from `935c302b9`.
4. `common/sampling.cpp:588`: replace the potentially overflowing `candidate_k + n_ban < 256` predicate with bounded operands and `n_ban < 256 - candidate_k`. This is an incidental correctness fix to the imported source; valid candidate widths are unchanged (D).

Test and fixture source changes:

- `tests/test-llama-archs.cpp`: import the Meta banned-ID and synthetic tie/cutoff/rejection tests from `062667905`; import the `std::mt19937 rng{uint32_t(seed)}` clang/GCC compile fix from `74e70bb95`; add the `INT32_MAX` top-k case, requiring the full sampler chain rather than a wrapped candidate width.
- `codex/bw-np2-driver.cpp`: cap the plain CPU fixture at four threads under `BW_CPU_ONLY`.
- `codex/bw-cand-driver.cpp`: import the candidate identity driver, including `json::array({p.tok, p.prob})`, the null-terminated Meta device list and four-thread Mac override. No other C++ fixture changed. Imported proof comments in the candidate source/tests were shortened without splitting sentences.
- `codex/stack7-test/CMakeLists.txt`: add the candidate driver to the existing fixture targets. `codex/chain_cpu_identity.py` caps both plain and Meta Mac runs at four threads.

Exactness argument: at most `n_ban` IDs among the raw best `k + 1 + n_ban` can be banned. Their survivors contain the best `k + 1` unbanned IDs. The existing strict finite ordering check makes the selected top-k and its order unique; ambiguous cases read the full row (D). Multi-slot chain off/on execute the same step loop and use the same output limits (D).

## Validation

All compilation and model execution use the Mac mini, CPU-only backends, `nice -n 19`, four build jobs, shared ccache, explicit small contexts and a four-GiB/no-other-model start gate. The Mac flags include `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DGGML_METAL=OFF -DGGML_BLAS=OFF`. i8mm is disabled for byte identity across CPU batch shapes.

Primary command:

`CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" nice -n 19 ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack8-mac.sh stack8`

The primary Mac source snapshot is `bdae9b6aa3f29c761b70e6a133a7a0a54bbc3c9a` (before the candidate-width overflow correction). The final bound and regression pass the rebuilt unit stage (M). For ordinary bounded top-k settings the subtraction predicate is equivalent to the original sum predicate (D); model identity evidence is not claimed for unbounded top-k.

The final successful unit/parser snapshot is `63e659d71b22e2d40ecb1e6419dc0b3578572741`, using payload/name `codex/bw-stack8-units-mac.sh stack8-units-final` with the primary command's environment and output base. The interrupted unit snapshot is `9054a6ee2c5c8015490c54c714d38e6728d1eea5`. The separate parser/template replay snapshot is `e45e46d848da9bb65d23fe7ee4a9df46a871d9cf`, using `codex/bw-stack8-static-mac.sh stack8-static`. Production source is identical in all three snapshots (M); their later commits modify validation/window files only.

The first unit stage was invalid: `test-recurrent-state-rollback` lacked `-m`, and the stage function's conditional invocation disabled Bash errexit inside the function. The committed runner executes stages in a subshell with errexit active and supplies `-m MODEL -c 4096 -t 4 -ngl 0`. The invalid result is excluded from the final evidence.

The first corrected unit run passed rollback/getter and seed 42, then its 20-second gate poll repeatedly missed gaps between another worker's model boots. Only its waiting gate shell was stopped, after checking PID 64176 against the exact owned run path and seed-1729 command. The runner recorded stage 21 as failed. The final rerun polls every second with identical memory and single-model checks; no running script or foreign process was changed.

Parser replay ran on the Mac against copies of actual GPU artifacts and `orchestration/gpuq/fixtures` (M). `codex/parser-inputs.json` records their source paths and hashes. SSE partial/null/repeated timing handling, size failure, native tokens, cross-arm token mutation, concurrent overlap, memory CSVs, kernel-fault regex boundaries, getter/chain logs and the recorded candidate proof pass. The first candidate replay used a full-run log with debug disabled and failed its engagement check; the corrected replay uses `canary-a1/1-cn/server.log` and passes.

Completed model stages and controls (M):

| Stage | Result | Wall seconds (M) |
|---|---|---:|
| 20, primary CPU build | PASS | 52 |
| 20, interrupted run's rebuild | PASS | 96 |
| 20, final-source rebuild | PASS | 64 |
| 21, final-source units and seeded sampling | PASS | 113 |
| 23, np1 chain identity and controls | PASS | 4923 |
| 29, strict np2 identity, plain and Meta | PASS | 112 |
| 32, old np2 binary negative control | PASS | 242 |
| 33, candidate CPU identity | PASS | 360 |
| 34, no-extension code mutant | PASS | 2 |
| 31, latest parser/template controls | PASS | 0 (integer timer) |

The np1 plain/Meta pairs at k3/k5 preserve tokens, counters and candidate IDs, with maximum q difference `1.2e-7` (M). The ordinary-loop repeat is bit identical and both reverse-uniform/shifted-embedding controls are caught. Strict np2 preserves complete JSON outputs, draft dumps and V/C/D shapes in both modes (M). The old binary announces the chain at np2 and the strict comparison fails on tokens/counters (M). Candidate CPU identity covers the complete expected arm matrix; the no-extension mutant fails specifically on a full-row read despite a unique top-k (M).

Final unit coverage includes `test-sampling`, `test-alloc`, model-backed recurrent rollback/getter checks, and `test-llama-archs --tp-sampling` with seeds 42 and 1729 (M). Each seed covers 120 trials with 1440 tied and 480 unique draws (M). Seed 42 yields 4 full accepts and 1916 rejections, catching the candidate mutant 132 times; seed 1729 yields 6 full accepts and 1914 rejections, catching it 136 times (M). Split replay and peer-slot independence have maximum difference and NMSE both zero (M). The interrupted stage records rc 143 and 3902 wall seconds (M), proving the fixed runner propagates failure; it is not a passing unit result.

Evidence is consolidated under `codex/mac-evidence/`, including the machine-readable `summary.json`, source/run identifiers, stage TSVs, control logs and final unit logs. Trailing whitespace was removed from the copied old-binary build console log and excess blank lines from the mutant runner; complete raw model outputs remain in the worktree's ignored `build-bw-mac-results/`. Wall times include gating and shared Mac load; they are not paired performance measurements. Final local `git diff --check ddedffacb HEAD -- . ':!codex/np2-old.patch'`, shell syntax checks and Python compilation pass (M). The exclusion preserves the unified patch's required blank context lines.

## Parent window and expected GPU outcomes

Manifest: `codex/bw-stack8.toml`, priority 20, filler, `host_sensitive = false`, `preemptible = false`. Payload: `codex/bw-stack8.sh`. It refuses a dirty tracked tree or existing `frozen/stack8`. Set `S8_DUO_FOLD_C` in the manifest before queueing; the template defaults to 0, the stack7 duo serving value from the brief.

Only stages 40-43 run locally: CUDA build with the stack6 CUDA 13.2/NCCL-off/120a/CCCL v3.4.3 recipe; require 25 `DeviceTopK` strings and the current ggml stamp; freeze through `bin/wt` as `frozen/stack8`; render items and run `gpuq submit --check`. Every compile uses `bin/buildgate` and `-j4`; no CUDA binary executes. Window estimate 1200 s (G), budget 2400 s (D, estimate x 2).

Experiments are `codex/items/stack8-*.toml.in`. Stage 43 materializes their exact-SHA experiments and `.gpuq.toml` items under `codex/items/rendered/`; snapshots stay under `codex/snapshots/`. They are intentionally rendered only after the freeze exists, so all frozen binaries are pinned. No item pins `build-bw-mac-results`.

| Item | Plan and expectation | Priority |
|---|---|---:|
| `stack8-depth-j5q` | `s7f s8f s8f s7f`; np1, chain/fold on, k5, 8k/32k/90k, `ignore_eos=true`. Cross-arm token/counter identity is expected and gated, based on the supplied prior candidate result. GPU speed remains unknown. The canary proves chain, folded decode and consumption of candidates with a nonempty ban list. | 30 |
| `stack8-np2-chain-j5q` | `off on on off`; same stack8 bytes, chain 0/1, fold 0, real concurrent prompts. The step loop and output capacity are identical, so steady speed should be equal within run noise (D). The canary requires the positive np2 fallback message and rejects any chain announcement. Cross-boot identity is off because arrivals change concurrent ubatches; the deterministic CPU matrix gates exactness. | 30 |
| `stack8-duo-soak-j5q` | One duo campaign using the serving flags, chain requested, parent-selected fold. Require np2 fallback, fit and campaign validity. | 20 |

Every run/canary command starts with `bash ~/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh`; that wrapper is pinned and each item has `host_sensitive = false`. Canary caps are 600 s (G) with one boot. The existing harness owns frozen hashes, model/launch checks, request records, exact PID/group shutdown, steady size gates and boot-scoped kernel journal checks. `codex/gpu_ab.sh` routes through `gpu_stack8.py`, removing traces and enforcing the work budget with cleanup headroom. The parent submits the window and later the checked GPU items; this worker does neither.

## Risks and open items

CUDA compilation, the final freeze, contract validation against gpuq and GPU behavior remain unmeasured until the window and items run. The 4B model has a tied LM head and cannot prove target candidate engagement under vocab splitting; the Meta synthetic unit model and GPU canary provide that proof. The Mac build uses clang. GCC 13 production compilation is deferred to the CUDA window; GCC 13 test compilation has not been verified. Multi-slot chain acceleration is deliberately unavailable, even while only one slot is busy. No upstream package, PR or external message was created.
