# np2 C -> D1: target sampling, not graph-cache thrash

Production commits: `6a9174baa` imports the existing banning-bias candidate fix `062667905` and test correction `74e70bb95` from `topic/cand-neginf-bias`; `3ed4bc1d6` makes its width guard overflow-safe. Branch `topic/np2-catchup` starts at frozen/stack7 `48b82b0d1559e936dbf612c871f2583cd4c79f3e`. The production file differs from the existing topic only in that guard.

Conclusion: two sequential target rejection samplers expose more host top-k work after C at np2. `ignore_eos=true` prevents the baseline's candidate offload. Graph reuse works in steady np2 rounds. The original stack6 trace also charges substantial time to `seq_rm`, which stack7 already addresses with its indexed path. Do not add another graph cache or change rollback ordering for this gap.

M = directly measured/observed; D = arithmetic or a sample-count estimate; G = hypothesis/estimate. Seeds, commits, line references, flags, and configured limits are identifiers, not timing measurements. No GPU speed improvement is established. No build, CPU model/test run, CUDA application, GPU query, network request, or queue mutation was performed here. Lightweight recorded-artifact analysis and required parser replay ran with `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, `nice -n 19`, one Python thread.

## 1. Reproduced GPU0 gap

These are stack6 IQ4_XS, MTP k=5 measurements, not J5q measurements or k=3 measurements. The quoted four gaps reproduce the chain-m2 brief. Files `nsys-np*.json` preserve the results.

Unit: median per complete V C D1 D2 D3 D4 D5 round, after the first 750 ms of capture, without duration trimming. A decode block consists of runtime launches until the next non-peer input H2D. At np2 the pack calls the two-row D blocks C because its head matcher recognizes only one-column GEMV; phase assignment here uses the recorded k=5 seven-block layout. These exports have no usable decode NVTX ranges. GPU0 kernel boundaries define the gap; this is not a measurement of simultaneous idleness on both GPUs.

| M, us per selected round | np1 8k | np1 90k | np2 8k | np2 90k |
|---|---:|---:|---:|---:|
| C last kernel -> D1 first kernel | 328.7 | 18.9 | 917.0 | 430.8 |
| Exposed interval before first D1 input H2D | 256.2 | 0.0 | 817.0 | 325.3 |
| Exposed first input -> first D1 host launch | 57.8 | 0.0 | 73.8 | 70.0 |
| Exposed launch -> D1 first kernel | 22.2 | 18.4 | 26.3 | 27.6 |
| C first kernel -> C last kernel | 544.3 | 958.2 | 730.3 | 1529.1 |
| C last host launch end -> D1 first host launch | 739.2 | 798.9 | 1478.1 | 1792.3 |
| cudaStreamSynchronize wall clipped to GPU gap | 0.0 | 0.0 | 26.7 | 0.0 |
| Capture/instantiate wall clipped to GPU gap | 0.0 | 0.0 | 0.0 | 0.0 |
| Selected rounds, M count | 98 | 83 | 98 | 77 |

Medians of separate intervals do not necessarily add. At np1 90k the input and first launch occur before C finishes; zero exposed input time does not mean zero input work.

Sources:

- np1: `/home/xanewok/repos/attrib-pack/runs/attrib-stack6-iq4/full-a3/decode-{8k,90k}.sqlite`.
- np2: `/home/xanewok/repos/attrib-pack/runs/attrib-stack6-duo-iq4/full-a1/decode-{8k,90k}.sqlite`.
- Loader/block definition: `/home/xanewok/repos/attrib-pack/attrib_pack.py:106`, `:225`.
- Reproduction: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/attribute_catchup.py SQLITE > codex/result.json`.

### Why 8k is worse

C is submitted asynchronously before `post_decode` verifies the draft. Target sampling and bookkeeping overlap C, then D1 starts after that host work and its own inputs. At np2, the measured C span grows by D 798.9 us from 8k to 90k, while the host launch-to-launch interval grows by D 314.2 us. The GPU idle gap therefore falls by D 486.2 us despite more total work at 90k. The exposed pre-input interval falls by D 491.7 us; the exposed input interval changes only D -3.8 us.

This supports an overlap explanation, not faster host sampling at greater context depth. Differences of independently measured medians are descriptive, not a per-round accounting identity. Mask-prefix behavior and changed prompt/acceptance histories are additional depth differences; no identical-host-work assumption is needed for the measured conclusion.

## 2. Work inside the gap

The CPU-sampled exports are separate captures: `/mnt/scratch/work/attrib-stack6-iq4/full-a3/decode-8k.sqlite` for np1 and `/mnt/scratch/work/attrib-stack6-duo-iq4/full-a1/decode-8k.sqlite` for np2.

np1: M 157 selected rounds, M 102 CPU samples in C -> D1, M 581.399 us median running-sample spacing, M 376.6 us median GPU gap. np2: M 120 selected rounds, M 194 gap samples, M 579.454 us spacing, M 922.3 us median GPU gap. The latter reproduces the separate unsampled np2 boot within D 5.2 us.

Time estimates below are D = top-of-stack sample count x measured sample spacing / selected rounds. They estimate mean CPU running time in the gap, not exact scope wall time or a median. Sparse rows are diagnostic. Do not sum these estimates with the preceding median table, or assign an unresolved allocation/memcpy sample to a specific caller.

| Work / top-of-stack symbol | np1 D us/gap (M samples) | np2 D us/gap (M samples) | Location / interpretation |
|---|---:|---:|---|
| `common_sampler::top_k_select` | 85.2 (23) | 376.6 (78) | `common/sampling.cpp:219`; host scan over the full target vocab, repeated for the rows examined by each slot's verifier |
| `llama_kv_cache::seq_rm` | 85.2 (23) | 154.5 (32) | `tools/server/server-context.cpp:4338`; one accepted-prefix trim per slot, plus rollback/replay paths; stack6's full-cell scan |
| CUDA driver, unresolved symbol | 44.4 (12) | 96.6 (20) | API/input/sync work; not enough stack information for finer assignment |
| `_int_malloc` | 11.1 (3) | 38.6 (8) | Sampler clone and per-slot vectors are reachable here; their individual costs are unknown |
| `__memcpy_avx_unaligned_erms` | 22.2 (6) | 29.0 (6) | Hidden rows, sampler state, prompt vectors and other copies; individual attribution unknown |
| `server_tokens::get_text_tokens` | 3.7 (1) | 14.5 (3) | `server-context.cpp:3284`; copies each drafting slot's prompt |
| `common_speculative_accept` | no sample | 9.7 (2) | `common/speculative.cpp:2129`; selects the accepted hidden row into `pending_h` |

Outputs: `nsys-np1-host-8k.json`, `nsys-np2-host-8k.json`. The parser uses the existing `host_sampling.load_samples`, `sample_spacing_ms` and top-of-stack classification; it does not relabel these statistical estimates as exact instrumentation.

### Steady stack7 graph and sampler evidence

Recorded artifact: `/home/xanewok/repos/qwen38-eval/results/2026-10-07-stack7-np2-real-iq4/canary-a1/0-patch/round-trace.jsonl`, with its `command.json`, `env.json`, `request.json` and `server.log`. This is another IQ4_XS workload with draft-vocab enabled, ub512 and `ignore_eos=true`; it must not be numerically subtracted from the stack6 nsys fixture.

After skipping the first three complete np2 rounds: M 95 rounds, M 95 graph hits and M 0 misses in each of C and D1..D5. C and D1 `llama_build` and `meta_build`: M 0 us. The two C misses and one D1 miss before this cut are warm-up/shape transitions. Missing counters are counted as zero; taking a median only over rounds that contain a miss would give a false steady rebuild cost.

M medians per selected round:

| Host interval/scope | us |
|---|---:|
| C decode return -> D1 decode entry | 1515.9 |
| Sampling-scope union within that interval | 1335.1 |
| Remaining interval outside sampling scopes | 149.1 |
| D1 `set_inputs` | 6.9 |
| D1 CUDA properties scopes | 0.9 |

These are host intervals, not GPU idle. The sampling union includes its nested row work, getters and waits; existing scopes cannot separate `set_logits`, candidate copies and rejection decisions. It is not charged twice. Both target slots log the TP leading-top-k refusal in `server.log:3245` and `:3259`. The actually recorded J5q `stack7-np2-real-j5q/full-a3/0-up/request.json` also has `ignore_eos=true`; the older `.toml.in` saying false is stale.

Reproduction: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/attribute_rounds.py TRACE --rows 2 --skip 3 > codex/stack7-np2-rounds.json`.

### Code path and exclusions

- `server-context.cpp:4065` calls `common_speculative_process` before `post_decode`; `speculative.cpp:1713` submits C. `speculative.cpp:1733-1785` then copies verify/pending hidden rows. Stack7 already uses the bulk accessor and indexed `seq_rm` (`llama-kv-cache.cpp:406`). The D 154.5 us stack6 trim estimate is not a fresh opportunity of that size on stack7.
- `server-context.cpp:4238-4304` clones and verifies one slot at a time. `common_sampler_sample_and_accept_n_rejection_rows` runs the target row sampler until rejection or the bonus row, then accepts the tokens. Two slots mean two independent loops; accepted length determines the number of rows visited.
- `server-context.cpp:4338` trims the accepted prefix; subsequent token processing/streaming and the next server iteration prepare the next draft. `server-context.cpp:3284` copies a prompt per drafting slot. `speculative.cpp:1978` adds one D1 row and its pending hidden state per sequence.
- `llama-context.cpp:1568` requires allocation validity and full graph/input compatibility. No weakening of this predicate is justified. The steady graph-hit counters exclude a persistent np2 C/D1 reuse defect for the recorded stack7 case.
- Unconditional sampler cloning is another possible small optimization: clone only when a target checkpoint restore is possible (`FULL`, or `RS` with draft size greater than the available rollback states). Its standalone cost is not measured here; it is outside this fix.

## 3. General fix and exactness

`tools/server/server-schema.cpp:475-480` applies EOG bans for `ignore_eos`. Baseline `common/sampling.cpp` admitted backend candidates only when `prefilter_bias.empty()`. With bans present it instead exposed the full logit-bias-led sampler chain; the TP gate at `src/llama-context.cpp:1380` rejects a chain whose first active sampler is not top-k. This explains the recorded target CPU fallback and full-vocab scans.

The production commits change only `common/sampling.cpp` and the existing `test-llama-archs.cpp`. Cumulative production diff: M 28 added / M 3 removed lines. Tests: M 394 imported lines plus M 12 added lines for the oversized-width regression and its standard header.

For top-k K and B distinct banning IDs, fetch raw top K+1+B. At most B entries are banned, so the retained list contains the K+1 best unbanned entries. Drop banned IDs, then use the existing finite/strict-order check and keep K. Unique finite logits give the same ordered candidate array as the full-row host path. The original host chain applies the same bias, top-k, temperature, top-p, min-p and dist sampler; the backend candidate chain draws no random numbers. The target and rejection RNG streams are unchanged.

Automatic full-row fallback remains for tied/non-finite candidate rows, duplicate biases, finite biases, active row-wide constraints, or an unsupported candidate width. A banned non-finite logit also triggers fallback because adding -inf can produce NaN. The route is automatic whenever the existing backend-sampling capability is enabled. It does not turn a user's explicit backend-sampling opt-out on. No new flag or environment variable is introduced; the existing per-sequence/multi-output backend path serves all active sequences.

Locations after the fix: `common/sampling.cpp:157-176` removes bans and checks exactness; `:209` searches the sorted banning IDs; `:584-588` gates the route; `:620` installs the expanded top-k. `tests/test-llama-archs.cpp:1033` tests synthetic rows and fallbacks; `:1456` tests actual Meta backend rows and rejection verdicts. The missing-width negative control both forces unwanted full-row reads and fails the compiled mutant test.

The imported width predicate evaluated `candidate_k + n_ban` before bounding K, so an untrusted `top_k=INT32_MAX` with a ban could overflow signed arithmetic. `3ed4bc1d6` first requires `0 < K < 256`, then compares B with `256 - K`; the later addition is bounded by the admitted width. `tests/test-llama-archs.cpp:1098` checks that oversized K retains the original full sampler chain rather than a candidate sampler. This extra case has not been run; reverting the guard is its negative control and should also be checked under UBSan in the parent's CPU window.

## 4. Verification and limits

No new compile or model test was run. `git apply --check`, `git diff --check`, shell syntax, TOML shape, and production-file comparison passed. The repository scope, security, general and tests checklists were read/applied; no upstream issue/PR lookup was made because network use is forbidden.

Inherited Mac evidence, not validation of this stack7 integration:

- `/home/xanewok/repos/qwen38-eval-worktrees/work/build-window-1007/cand-neginf-mac-a7/payload.log`: source `bd658589d8d6e785521ebcb48d92d8421641d1d6`, unit seeds 42/1729; M 120 trials per seed, M 1440 tied / M 480 unique draws, exact draw/candidate/rejection comparisons PASS. The unextended candidate control was caught M 132 / M 136 times.
- `mutant-run.log`: the compiled `candidate_k + 1` mutant returned M rc=1 with `banned: read the full row although the top k is unique`.
- `e2e-compare.log`: Qwen3.5-4B CPU/Meta np1 token/counter identity PASS. The original tied target head falls back under TP, so this model check alone does not prove target TP candidate engagement; the synthetic untied Meta test supplies that proof.
- `cpu-inherited-evidence.json` records this distinction and the remaining integration gates. No CPU timing or GPU gain is inferred from these correctness checks.

Required next validation: build this stack7 integration in an authorized build window or on an idle Mac via `bin/bw-mac`; run `test-sampling`, `test-llama-archs --tp-sampling` with both seeds, the compiled missing-width mutant, the oversized-width case/control, and the np2 seeded token/q matrix. Then compile CUDA and freeze. No Mac work was needed to read the existing evidence.

## 5. Parent-only GPU experiment

`np2-catchup-real-j5q.toml.in` proposes ABBA against frozen/stack7, real prompt tokens, np2 serving context, depths 8192/32768/90112, k=5, target q8_0 KV, draft f16 KV, ub128, TP50/50, probabilistic draft, both backend samplers enabled, `ignore_eos=true`, chain/fold off in both arms. It copies the actual stack7-np2-real request semantics, not the stale ignore_eos=false template.

`gpu_ab.sh` and `gpu_np2_catchup.py` reuse the existing harness and stack7 guards: frozen hashes/provenance and loaded-library roots, interleaved fresh boots, exact-child-PID cleanup, boot alarms, scoped kernel-journal checks, np2 overlap and steady-size gates. Trace output is suppressed in both canary and measurement boots. Patch boots reject the baseline's target sampler refusal. That log gate proves absence of the known fallback, not full GPU token identity.

Canary: one patch boot at the smallest depth, the same two streams and size/overlap/parser/refusal gates, configured maximum 600 s. Full boot wall allowance is 900 s; the harness derives the larger ABBA stage budget with margin and cleanup. Cross-boot np2 token identity is off because request arrival changes ubatches; the pre-freeze CPU np2 identity gate is mandatory. Report per-arm acceptance and per-boot timing spread alongside aggregate overlap-window round time. Treat an unexplained acceptance change as an invalid performance conclusion, not a speed win.

Required parser replay passed against real recorded GPU artifacts and fixtures, including a known-refusal negative control. Command: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 HARNESS_CODE_ROOT=/home/xanewok/repos/qwen38-eval nice -n 19 python3 -P codex/gpu_np2_catchup.py --fixtures`. See `parser-replay.log`. New log-gate positive input: `2026-10-08-bg-exact-j5q/canary-a1/cand-np2/server.log`; negative input: `2026-10-07-stack7-np2-real-j5q/canary-a2/0-patch/server.log`. The positive artifact tests log parsing, not this patch's GPU behavior.

The experiment is deliberately not rendered into a queue item: the candidate freeze and stack7 integration CPU summary do not exist. After they exist, substitute `@PATCH_SHA@` from `frozen/np2-catchup/SOURCE` and `@CPU_SUMMARY@`, render with `harness.mkitem`, wrap the generated stages with `codex/gpu_ab.sh`, and run `gpuq submit --check` only. The maker writes snapshots into the parent's canonical orchestration directory, so it was not run by this worker. No item was submitted or canceled. Pre-existing edits to `codex/PROMPT.md` and `codex/PURPOSE` remain uncommitted; the owner must resolve them before a freeze that requires a clean tree.

G: the fix should remove much of the exposed target full-vocab scanning and readback in forced-length np2 runs. The D 376.6 us sampled top-k component is evidence of an opportunity, not a predicted speedup: target GPU top-k cost, overlap, numerical/fallback behavior and acceptance must be measured. At 90k more host work is hidden by C; a context-independent host saving need not give a context-independent GPU gap reduction.

Remaining uncertainty: sparse CPU samples cannot assign exact us to clone, streaming, each hidden/state getter or batch allocation; software and nsys captures are not clock-aligned; no J5q-specific C -> D1 component attribution was established. The next nsys/round-trace pack should compare the same frozen binaries and requests after the untraced A/B, without treating tracing overhead as production timing.
